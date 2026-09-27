"""On-policy knowledge distillation (KD): the second training objective next to GRPO (`objective: kd`).

The student (the policy being trained) generates episodes exactly as in GRPO: same harness, same servers, same
grading (success rates are still logged). A larger teacher that shares the student's tokenizer then scores every
token the student sampled (vLLM prompt_logprobs), and the per-token signal replaces the group advantage:

    a_t = kd.coef · (log π_teacher(x_t | x_<t) − log π_sampler(x_t | x_<t)) + kd.reward_coef · A_group

LoRATrainer.step minimises −Σ_t a_t·w_t·log π(x_t | x_<t) (w_t: the TIS weight). With reward_coef = 0 its expected
gradient is ∇ KL(π_student ‖ π_teacher) on the student's own samples, one term per token with discount 0 — the
sampled-token reverse-KL estimator of on-policy distillation (Agarwal et al. 2024, "GKD"; Thinking Machines 2025,
"On-policy distillation"): a token the student likes more than the teacher does gets a negative advantage, one it
under-rates a positive one. The estimator is unbiased only because the student samples at temperature 1 with no
top-p/top-k, so the served distribution is π_student itself.

Everything here is pure (no HTTP, no torch): the loop in sarl.train calls it.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

OBJECTIVES = ("grpo", "kd")
KD_DEFAULTS = {"coef": 1.0, "reward_coef": 0.0, "workers": 2, "timeout_s": 180, "max_consecutive_failures": 3}
TEACHER_DEFAULTS = {"model": "policy", "sleep": False}


def validate_objective(cfg: dict) -> str:
    """The run's objective ("grpo" when absent) after checking its settings. A KD config gets its `teacher` and `kd`
    sections normalised in place (defaults filled, url without a trailing slash); a GRPO config is not touched, so
    every existing config behaves exactly as before."""
    obj = cfg.get("objective", "grpo")
    if obj not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {obj!r}")
    if obj == "grpo":
        if cfg.get("teacher") or cfg.get("kd"):  # a forgotten `objective: kd` would otherwise train GRPO silently
            raise ValueError("teacher/kd sections are only used with objective: kd")
        return obj
    t = cfg.get("teacher")
    if not isinstance(t, dict) or not t.get("url") or not t.get("model_key"):
        raise ValueError("objective kd needs teacher: {url, model_key[, model, sleep]}")
    from sarl.paths import MODELS

    if t["model_key"] not in MODELS:
        raise ValueError(f"teacher.model_key {t['model_key']!r} is not a sarl.paths.MODELS key")
    # KD distils on the student's own samples: a scripted gold-patch rollout has no sampler log-probs and is not the
    # student's distribution
    if int(cfg.get("oracle_per_group", 0)) != 0:
        raise ValueError("objective kd trains on the student's own samples only: set oracle_per_group: 0")
    kd = {**KD_DEFAULTS, **(cfg.get("kd") or {})}
    if set(kd) - set(KD_DEFAULTS):
        raise ValueError(f"unknown kd keys {sorted(set(kd) - set(KD_DEFAULTS))} (known: {sorted(KD_DEFAULTS)})")
    if int(kd["workers"]) < 1 or float(kd["timeout_s"]) <= 0 or int(kd["max_consecutive_failures"]) < 1:
        raise ValueError("kd.workers and kd.max_consecutive_failures must be >= 1 and kd.timeout_s > 0")
    teacher = {**TEACHER_DEFAULTS, **t, "url": str(t["url"]).rstrip("/")}
    rollout = {s["url"].rstrip("/") for s in (cfg.get("servers") or [])} | {str(cfg.get("serve_url", "")).rstrip("/")}
    if teacher["url"] in rollout:  # the teacher serves another model; it must not take rollouts or LoRA syncs
        raise ValueError(f"teacher.url {teacher['url']} is also a rollout server")
    cfg["teacher"], cfg["kd"] = teacher, kd
    return obj


def trained_positions(mask: Sequence[int]) -> list[int]:
    """The positions the trainer supervises, in LoRATrainer.token_logprobs' order: t in 1..len-1 with mask[t] == 1
    (position 0 has no context, so it is never trained)."""
    return [i for i in range(1, len(mask)) if mask[i]]


def _finite(v) -> bool:
    return v is not None and math.isfinite(float(v))


def kd_token_advantages(ids: Sequence[int], mask: Sequence[int], sampler_lp: Sequence[float | None],
                        teacher_lp: Sequence[float | None], coef: float = 1.0, reward_coef: float = 0.0,
                        a_group: float = 0.0) -> tuple[list[float | None], int]:
    """Per-token KD advantages aligned with `ids` (None at every untrained position), and how many trained positions
    had no usable log-prob.

    sampler_lp is aligned with ids (the trajectory's own log-probs, None where not sampled); teacher_lp is
    served_logprobs' output, element i-1 = log π_teacher(ids[i] | ids[:i]); it may stop early (the teacher scores only
    up to the last trained token). A trained position whose sampler or teacher log-prob is missing or not finite gets
    a_t = 0 (no push either way) and is counted."""
    if len(mask) != len(ids) or len(sampler_lp) != len(ids):
        raise ValueError(f"ids/mask/sampler_lp lengths differ: {len(ids)}/{len(mask)}/{len(sampler_lp)}")
    adv: list[float | None] = [None] * len(ids)
    missing = 0
    for i in trained_positions(mask):
        s = sampler_lp[i]
        t = teacher_lp[i - 1] if i - 1 < len(teacher_lp) else None
        if _finite(s) and _finite(t):
            adv[i] = coef * (float(t) - float(s)) + reward_coef * a_group
        else:
            adv[i] = 0.0
            missing += 1
    return adv, missing


def teacher_span(mask: Sequence[int]) -> int:
    """How many leading tokens the teacher must see: up to the last trained one. The trailing observation (mask 0)
    carries no trained token, and scoring it would only cost teacher time. 0 when nothing is trained."""
    pos = trained_positions(mask)
    return pos[-1] + 1 if pos else 0


def special_token_ids(tok) -> dict:
    out = {name: getattr(tok, f"{name}_id", None) for name in ("bos_token", "eos_token", "unk_token", "pad_token")}
    out["all"] = sorted(int(i) for i in (getattr(tok, "all_special_ids", None) or []))
    return out


def tokenizer_mismatch(student, teacher) -> str | None:
    """None when the teacher reads the student's token ids exactly as the student does (identical vocabulary, id for
    id, and identical special-token ids); otherwise a short description of the difference. KD scores the student's
    token ids with the teacher, so a different tokenizer would make every a_t meaningless without any error."""
    vs, vt = student.get_vocab(), teacher.get_vocab()
    if vs != vt:
        moved = sum(1 for k in vs.keys() & vt.keys() if vs[k] != vt[k])
        return (f"vocabularies differ: {len(vs)} vs {len(vt)} tokens, {len(vs.keys() - vt.keys())} only in the student, "
                f"{len(vt.keys() - vs.keys())} only in the teacher, {moved} with different ids")
    ss, st = special_token_ids(student), special_token_ids(teacher)
    if ss != st:
        return f"special-token ids differ: student {ss} vs teacher {st}"
    return None


def _in_token_order(seq: Sequence[float | None], pos: list[int], offset: int = 0) -> np.ndarray:
    """seq[p - offset] for every trained position p (NaN when missing or past the end), as float64."""
    return np.array([float(seq[p - offset]) if 0 <= p - offset < len(seq) and seq[p - offset] is not None else np.nan
                     for p in pos], dtype=np.float64)


IS_CAP = 10.0  # clip of the importance weight in rkl_trainer_after (w ≈ 1 at lr 5e-5; the cap only guards outliers)


def kd_step_stats(batch: Sequence[tuple], teacher_lps: Sequence[Sequence[float | None]],
                  before: Sequence[np.ndarray] | None, after: Sequence[np.ndarray] | None) -> dict:
    """The step's KD metrics over every trained token of `batch` = [(ids, mask, per-token adv, sampler_lp)].

    Every rkl_*, mean_*_lp and msq_gap_* value averages over one token set, the n_ok trained tokens that carry a KD
    signal (finite sampler and teacher log-probs; the others have a_t = 0 and are counted in frac_missing).

    rkl_sampler = mean(log π_sampler − log π_teacher): the sampled-token estimate of KL(π_student ‖ π_teacher) on the
    student's own fresh samples — the KD learning curve, which should fall across steps. rkl_trainer_before: the same
    with the trainer's per-token log-probs before the update (so rkl_trainer_before − rkl_sampler is the trainer-vs-
    sampler mismatch). rkl_trainer_after = mean(w·(lp_after − log π_teacher)) with w = min(e^(lp_after − lp_before),
    IS_CAP): the tokens were sampled from the old policy, so the after-update estimate needs the importance weight to be
    an (up to the cap) unbiased estimate of KL(π_new ‖ π_teacher) at the same contexts. Without it the teacher term
    cancels in the difference and rkl_gain would be mean(lp_before − lp_after), positive for any update that lowers
    the sampled tokens' log-probs, towards the teacher or away from it. rkl_gain = before − after > 0: the update moved
    the student toward the teacher on this batch. is_w_max: the largest weight before the cap.

    objective_gain = Σ_t a_t·(lp_after − lp_before)/T over every trained token (T = n_tok, the token_mean
    denominator): > 0 when the update ascended its own surrogate. msq_gap_before/after = mean((lp − log π_teacher)²),
    the squared per-token gap that the update descends on these fixed tokens (exactly so when the trainer matches the
    sampler)."""
    s_all, t_all, a_all, b_all, f_all = [], [], [], [], []
    for k, (b, t_lp) in enumerate(zip(batch, teacher_lps)):
        pos = trained_positions(b[1])
        s_all.append(_in_token_order(b[3], pos))
        t_all.append(_in_token_order(t_lp, pos, offset=1))
        a_all.append(_in_token_order(b[2], pos))
        if before is not None and after is not None:
            b_all.append(np.asarray(before[k], dtype=np.float64))
            f_all.append(np.asarray(after[k], dtype=np.float64))
    n_tok = int(sum(x.size for x in s_all))
    out = {"n_tok": n_tok, "n_ok": 0, "frac_missing": None, "rkl_sampler": None, "rkl_trainer_before": None,
           "rkl_trainer_after": None, "rkl_gain": None, "is_w_max": None, "msq_gap_before": None, "msq_gap_after": None,
           "mean_teacher_lp": None, "mean_sampler_lp": None, "objective_gain": None}
    if not n_tok:
        return out
    s, t, a = np.concatenate(s_all), np.concatenate(t_all), np.nan_to_num(np.concatenate(a_all))
    ok = np.isfinite(s) & np.isfinite(t)  # the tokens with a KD signal: the one set every rkl_* value averages over
    out["n_ok"], out["frac_missing"] = int(ok.sum()), float(1.0 - ok.mean())
    if ok.any():
        out.update(rkl_sampler=float((s[ok] - t[ok]).mean()), mean_teacher_lp=float(t[ok].mean()),
                   mean_sampler_lp=float(s[ok].mean()))
    if b_all and sum(x.size for x in b_all) == n_tok:
        bf, af = np.concatenate(b_all), np.concatenate(f_all)
        if ok.any():
            gb, ga = bf[ok] - t[ok], af[ok] - t[ok]
            w = np.exp(af[ok] - bf[ok])
            rb, ra = float(gb.mean()), float((np.minimum(w, IS_CAP) * ga).mean())
            out.update(rkl_trainer_before=rb, rkl_trainer_after=ra, rkl_gain=rb - ra, is_w_max=float(w.max()),
                       msq_gap_before=float((gb ** 2).mean()), msq_gap_after=float((ga ** 2).mean()))
        out["objective_gain"] = float((a * (af - bf)).sum() / n_tok)
    return out
