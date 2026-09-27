"""On-policy KD (objective kd): per-token advantages, the trainer's per-token loss, the teacher checks and config
validation, on a tiny random Qwen2 (CPU, no download, no server)."""

import copy
import itertools
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sarl.grpo import LoRATrainer, TrainConfig, token_advantages
from sarl.kd import (kd_step_stats, kd_token_advantages, teacher_span, tokenizer_mismatch, trained_positions,
                     validate_objective)

NAN = float("nan")


def test_kd_token_advantages_alignment():
    """a_i = coef·(teacher[i-1] − sampler[i]) + reward_coef·A_group at trained positions only; a missing or NaN
    log-prob gives 0 and is counted; position 0 is never trained; a teacher list that stops early (scored up to the
    last trained token, or a truncated view) leaves the rest missing."""
    ids = [10, 11, 12, 13, 14, 15, 16, 17]
    mask = [1, 0, 1, 1, 0, 1, 1, 1]  # mask[0] = 1 must be ignored: no context
    sampler = [-9.0, None, -1.0, -2.0, None, None, -0.5, -0.25]
    teacher = [-7.0, -1.5, -1.0, -8.0, -0.2, NAN, -0.75]  # positions 1..7
    adv, miss = kd_token_advantages(ids, mask, sampler, teacher, coef=2.0, reward_coef=0.5, a_group=1.0)
    assert adv[0] is None and adv[1] is None and adv[4] is None
    assert adv[2] == pytest.approx(2.0 * (-1.5 + 1.0) + 0.5) and adv[3] == pytest.approx(2.0 * (-1.0 + 2.0) + 0.5)
    assert adv[5] == 0.0 and adv[6] == 0.0  # sampler None; teacher NaN
    assert adv[7] == pytest.approx(2.0 * (-0.75 + 0.25) + 0.5)
    assert miss == 2
    # the trainer reads exactly these values, in token_logprobs' order, and ignores what sits at untrained positions
    assert token_advantages(adv, ids, mask) == [adv[i] for i in trained_positions(mask)] == [adv[2], adv[3], 0.0, 0.0, adv[7]]
    junk = list(adv)
    junk[1] = 99.0
    assert token_advantages(junk, ids, mask) == token_advantages(adv, ids, mask)
    assert token_advantages(1.5, ids, mask) is None  # a scalar keeps the GRPO path
    with pytest.raises(ValueError):
        token_advantages(adv[:-1], ids, mask)
    # the teacher only scores up to the last trained token; a shorter teacher list (truncated view) leaves the rest missing
    assert teacher_span(mask) == 8 and teacher_span([0, 1, 1, 0, 0]) == 3 and teacher_span([1, 0, 0]) == 0
    adv_t, miss_t = kd_token_advantages(ids[:5], mask[:5], sampler[:5], teacher[:2])
    assert adv_t[2] == pytest.approx(-0.5) and adv_t[3] == 0.0 and miss_t == 1
    with pytest.raises(ValueError):
        kd_token_advantages(ids, mask[:-1], sampler, teacher)


def _batch(n=3, T=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    mask = [0] * 10 + [1] * 12 + [0] * 8 + [1] * 10
    return [(torch.randint(0, 64, (T,), generator=g).tolist(), mask) for _ in range(n)]


def _step_with_grads(tr, batch):
    """step() and the gradients it hands to AdamW (captured at opt.step)."""
    seen, orig = {}, tr.opt.step

    def spy(*a, **k):
        seen["g"] = [p.grad.detach().clone() for p in tr.params]
        return orig(*a, **k)

    tr.opt.step = spy
    try:
        out = tr.step(batch)
    finally:
        tr.opt.step = orig
    return out, seen["g"]


@pytest.mark.parametrize("agg", ["token_mean", "seq_mean"])
def test_constant_per_token_advantage_equals_scalar(fresh_tiny_qwen, agg):
    """A per-token sequence that is c at every trained position is the scalar c: the same gradient bit for bit (the
    backward pass multiplies by the same c either way) and the same loss up to float32 summation order (c·Σx vs Σc·x),
    with TIS weights, some capped. lr = 0 keeps the weights fixed between the two calls."""
    tr = LoRATrainer("tiny", model=fresh_tiny_qwen, device="cpu",
                     cfg=TrainConfig(lr=0.0, lora_r=4, lora_alpha=8, loss_agg=agg, tis_cap=2.0, grad_checkpointing=False))
    b = _batch(3, seed=11)
    torch.manual_seed(0)
    for p in tr.params:  # LoRA B starts at 0 (so A gets no gradient); give both a value
        p.data.normal_(0.0, 0.05)
    samp = []
    for ids, m in b:
        t = tr.logprobs_nograd(ids, m)
        s = [None] * len(ids)
        for k, pos in enumerate(trained_positions(m)):
            s[pos] = float(t[k]) - (1.0 if k % 3 == 0 else 0.0)  # every third token: w = e^1, capped at 2
        samp.append(s)
    advs = [0.7, -1.3, 2.0]
    out_s, g_s = _step_with_grads(tr, [(ids, m, a, s) for (ids, m), a, s in zip(b, advs, samp)])
    seqs = [[a if (i >= 1 and m[i]) else None for i in range(len(ids))] for (ids, m), a in zip(b, advs)]
    out_t, g_t = _step_with_grads(tr, [(ids, m, q, s) for (ids, m), q, s in zip(b, seqs, samp)])
    assert out_s["tis"]["frac_capped"] > 0
    assert out_t["loss"] == pytest.approx(out_s["loss"], rel=1e-6, abs=1e-9)
    assert out_t["grad_norm"] == out_s["grad_norm"]
    assert all(torch.equal(x, y) for x, y in zip(g_s, g_t))
    assert any(float(x.abs().sum()) > 0 for x in g_s)


def _served(model, ids):
    """served_logprobs' layout from a local model: element i-1 = log p(ids[i] | ids[:i])."""
    with torch.no_grad():
        lp = torch.log_softmax(model(input_ids=torch.tensor([ids])).logits[0, :-1].float(), -1)
    return lp.gather(-1, torch.tensor(ids[1:]).unsqueeze(-1)).squeeze(-1).tolist()


@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["toward", "away"])
def test_kd_steps_move_the_student_toward_the_teacher(fresh_tiny_qwen, sign):
    """A few KD steps, each on fresh samples of the current student (on-policy, as in a run), with a tiny teacher
    sharing the vocabulary (the same network, flatter and perturbed). sign = -1 negates the advantage, so the update
    moves the student away from the teacher. In both cases the exact reverse KL moves as intended, and each step's
    rkl_gain has the sign of the exact change at that step's contexts. A gain without the importance weight would be
    positive in both cases, because the teacher term cancels."""
    student, teacher = fresh_tiny_qwen, copy.deepcopy(fresh_tiny_qwen)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        student.model.norm.weight.mul_(8.0)  # random-init logits are nearly uniform: make the student peaky
        teacher.model.norm.weight.mul_(2.0)  # and the teacher flatter than it, with its own perturbation
        for p in teacher.parameters():
            p.add_(0.02 * torch.randn(p.shape, generator=g))
    torch.manual_seed(0)  # LoRA init
    tr = LoRATrainer("tiny", model=student, device="cpu",
                     cfg=TrainConfig(lr=5e-3, lora_r=4, lora_alpha=8, loss_agg="token_mean", tis_cap=2.0, grad_checkpointing=False,
                                     targets=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")))

    def sample(n=8, prompt=6, T=24):  # the current student's own samples (T = 1), after a random prompt
        out = []
        with torch.no_grad():
            tr.model.eval()
            for _ in range(n):
                ids = torch.randint(0, 64, (prompt,), generator=g).tolist()
                for _ in range(T):
                    probs = torch.softmax(tr.model(input_ids=torch.tensor([ids])).logits[0, -1].float(), -1)
                    ids.append(int(torch.multinomial(probs, 1, generator=g)))
                out.append((ids, [0] * prompt + [1] * T))
        return out

    def exact_rkl(samples):  # KL(π_student(·|ctx) ‖ π_teacher(·|ctx)) over the full 64-token vocabulary, trained positions
        kls = []
        with torch.no_grad():
            tr.model.eval()
            for ids, m in samples:
                x = torch.tensor([ids])
                pos = torch.tensor([p - 1 for p in trained_positions(m)])
                ls = torch.log_softmax(tr.model(input_ids=x).logits[0].float(), -1)[pos]
                lt = torch.log_softmax(teacher(input_ids=x).logits[0].float(), -1)[pos]
                kls.append((ls.exp() * (ls - lt)).sum(-1))
        return float(torch.cat(kls).mean())

    held_out = sample()  # fixed contexts for the trend over all steps
    k0 = exact_rkl(held_out)
    assert k0 > 0
    for _ in range(4):
        samples = sample()
        t_lps = [_served(teacher, ids) for ids, _ in samples]
        k_before = exact_rkl(samples)
        batch = []
        for (ids, m), t in zip(samples, t_lps):
            cur = tr.logprobs_nograd(ids, m)  # the served student = the current weights (synced every step)
            s = [None] * len(ids)
            for k, pos in enumerate(trained_positions(m)):
                s[pos] = float(cur[k])
            adv, miss = kd_token_advantages(ids, m, s, t)
            assert miss == 0
            batch.append((ids, m, [None if a is None else sign * a for a in adv], s))
        out = tr.step(batch, keep_token_logprobs=True)
        assert out["stepped"] and len(out["token_logprobs"]) == len(batch)
        after = [tr.logprobs_nograd(i, m) for i, m in samples]
        st = kd_step_stats(batch, t_lps, out["token_logprobs"], after)
        dk = k_before - exact_rkl(samples)  # > 0: the update moved the student toward the teacher at these contexts
        assert st["n_tok"] == st["n_ok"] == 8 * 24 and st["frac_missing"] == 0.0
        assert st["rkl_sampler"] == pytest.approx(st["rkl_trainer_before"], abs=1e-4)  # on-policy: sampler = trainer
        assert sign * dk > 0 and np.sign(st["rkl_gain"]) == np.sign(dk)
        assert st["objective_gain"] > 0  # the update ascends its own surrogate whichever way it points
        assert np.sign(st["msq_gap_before"] - st["msq_gap_after"]) == np.sign(sign)
        assert 1.0 <= st["is_w_max"] < 10.0  # the cap never bit
    k1 = exact_rkl(held_out)  # contexts never trained on (measured 2026-09-26: 0.70·k0 toward, 1.08·k0 away)
    assert (k1 < 0.8 * k0) if sign > 0 else (k1 > 1.03 * k0)


def test_kd_step_stats_by_hand():
    """The metrics on a hand-built trajectory whose trainer log-probs differ from the sampler's. Every rkl_*/msq_*
    value uses the tokens with both a sampler and a teacher log-prob (n_ok); the after-update estimate is importance
    weighted; objective_gain runs over every trained token with the token_mean denominator."""
    ids, mask = [1, 2, 3, 4, 5, 6], [0, 1, 1, 0, 1, 1]
    sampler = [None, -1.0, -2.0, None, -0.5, None]  # position 5 is trained but has no sampler log-prob
    teacher = [-1.5, -1.0, -3.0, NAN, -2.0]  # positions 1..5; position 4 has no teacher log-prob
    adv, miss = kd_token_advantages(ids, mask, sampler, teacher)
    assert miss == 2 and adv == [None, pytest.approx(-0.5), pytest.approx(1.0), None, 0.0, 0.0]
    bf, af = np.array([-1.1, -2.2, -0.4, -1.0]), np.array([-1.3, -1.6, -0.4, -1.0])  # trained positions 1, 2, 4, 5
    st = kd_step_stats([(ids, mask, adv, sampler)], [teacher], [bf], [af])
    assert st["n_tok"] == 4 and st["n_ok"] == 2 and st["frac_missing"] == pytest.approx(0.5)
    assert st["rkl_sampler"] == pytest.approx(-0.25) and st["mean_teacher_lp"] == pytest.approx(-1.25)
    assert st["mean_sampler_lp"] == pytest.approx(-1.5)
    assert st["rkl_trainer_before"] == pytest.approx((0.4 - 1.2) / 2)  # from the trainer, not the sampler
    w = [math.exp(-0.2), math.exp(0.6)]
    ra = (w[0] * 0.2 + w[1] * -0.6) / 2
    assert st["rkl_trainer_after"] == pytest.approx(ra) and st["rkl_gain"] == pytest.approx(-0.4 - ra)
    assert st["is_w_max"] == pytest.approx(w[1])
    assert st["msq_gap_before"] == pytest.approx((0.16 + 1.44) / 2) and st["msq_gap_after"] == pytest.approx((0.04 + 0.36) / 2)
    assert st["objective_gain"] == pytest.approx((-0.5 * -0.2 + 1.0 * 0.6) / 4)
    # a finite teacher log-prob without a sampler log-prob (a_t = 0, never trained) stays out of every rkl value: with
    # the trainer equal to the sampler wherever both exist, trainer and sampler estimates agree
    st2 = kd_step_stats([([1, 2, 3], [0, 1, 1], [None, 0.0, 0.0], [None, -1.0, None])], [[-1.0, -5.0]],
                        [np.array([-1.0, -0.1])], [np.array([-1.0, -0.1])])
    assert st2["n_ok"] == 1 and st2["frac_missing"] == 0.5
    assert st2["rkl_sampler"] == st2["rkl_trainer_before"] == st2["rkl_trainer_after"] == st2["msq_gap_before"] == 0.0
    # the IS weight is capped
    st3 = kd_step_stats([([1, 2], [0, 1], [None, 1.0], [None, -3.0])], [[-2.0]], [np.array([-3.0])], [np.array([0.0])])
    assert st3["is_w_max"] == pytest.approx(math.exp(3.0)) and st3["rkl_trainer_after"] == pytest.approx(10.0 * 2.0)
    empty = kd_step_stats([], [], None, None)
    assert empty["n_tok"] == 0 and empty["rkl_sampler"] is None and empty["objective_gain"] is None


class _Tok:
    def __init__(self, vocab, eos=2, pad=3, specials=(2, 3)):
        self._v, self.eos_token_id, self.pad_token_id = dict(vocab), eos, pad
        self.bos_token_id = self.unk_token_id = None
        self.all_special_ids = list(specials)

    def get_vocab(self):
        return dict(self._v)


def test_tokenizer_identity_check():
    v = {"a": 0, "b": 1, "<|im_end|>": 2, "<|endoftext|>": 3}
    assert tokenizer_mismatch(_Tok(v), _Tok(v)) is None
    assert "vocabularies differ" in tokenizer_mismatch(_Tok(v), _Tok({**v, "c": 4}))
    assert "2 with different ids" in tokenizer_mismatch(_Tok(v), _Tok({**v, "a": 1, "b": 0}))
    assert "special-token ids differ" in tokenizer_mismatch(_Tok(v), _Tok(v, eos=3))
    assert "special-token ids differ" in tokenizer_mismatch(_Tok(v), _Tok(v, specials=(2,)))


def test_real_teacher_and_student_share_a_tokenizer():
    """The configured pair passes the start-up check (skips where the snapshots are not in the HF cache)."""
    from transformers import AutoTokenizer

    from sarl.paths import model_snapshot

    snaps = [model_snapshot(k) for k in ("mini-coder-1.7b", "mini-coder-4b")]
    if not all((s / "tokenizer.json").exists() for s in snaps):
        pytest.skip("mini-coder snapshots not in the HF cache")
    a, b = (AutoTokenizer.from_pretrained(str(s)) for s in snaps)
    assert tokenizer_mismatch(a, b) is None


def _kd_cfg(**over):
    cfg = {"objective": "kd", "oracle_per_group": 0, "serve_url": "http://localhost:8001",
           "servers": [{"url": "http://localhost:8001", "slots": 8}, {"url": "http://node04:8003", "slots": 4}],
           "teacher": {"url": "http://localhost:8002/", "model_key": "mini-coder-4b", "sleep": True}}
    cfg.update(over)
    return cfg


def test_config_validation():
    grpo = {"oracle_per_group": 1, "servers": [{"url": "http://a"}]}
    before = copy.deepcopy(grpo)
    assert validate_objective(grpo) == "grpo" and grpo == before  # existing configs: untouched
    assert validate_objective({**before, "objective": "grpo"}) == "grpo"
    with pytest.raises(ValueError, match="only used with objective: kd"):
        validate_objective({**before, "teacher": {"url": "http://t"}})
    with pytest.raises(ValueError, match="objective must be"):
        validate_objective({"objective": "sft"})
    cfg = _kd_cfg()
    assert validate_objective(cfg) == "kd"
    assert cfg["teacher"] == {"url": "http://localhost:8002", "model": "policy", "model_key": "mini-coder-4b", "sleep": True}
    assert cfg["kd"] == {"coef": 1.0, "reward_coef": 0.0, "workers": 2, "timeout_s": 180, "max_consecutive_failures": 3}
    with pytest.raises(ValueError, match="needs teacher"):
        validate_objective(_kd_cfg(teacher=None))
    with pytest.raises(ValueError, match="needs teacher"):
        validate_objective(_kd_cfg(teacher={"url": "http://localhost:8002"}))  # no model_key
    with pytest.raises(ValueError, match="MODELS key"):
        validate_objective(_kd_cfg(teacher={"url": "http://t:1", "model_key": "nope"}))
    with pytest.raises(ValueError, match="oracle_per_group: 0"):
        validate_objective(_kd_cfg(oracle_per_group=1))
    with pytest.raises(ValueError, match="unknown kd keys"):
        validate_objective(_kd_cfg(kd={"alpha": 1.0}))
    with pytest.raises(ValueError, match="workers"):
        validate_objective(_kd_cfg(kd={"workers": 0}))
    with pytest.raises(ValueError, match="also a rollout server"):
        validate_objective(_kd_cfg(teacher={"url": "http://node04:8003", "model_key": "mini-coder-4b"}))


def test_kd_config_file_validates():
    from sarl.train import load_config

    cfg = load_config("configs/kd_minicoder_v1.yaml")
    assert validate_objective(cfg) == "kd" and cfg["teacher"]["model_key"] == "mini-coder-4b" and cfg["keep_adapters"] == 2
    v2 = load_config("configs/grpo_minicoder_v2.yaml")
    same = ("model", "revision", "pool", "split", "steps", "tasks_per_step", "group_size", "oracle_per_group", "seed",
            "sampling", "limits", "train", "protocol", "chat_template_kwargs", "served_model", "diagnostics", "grading", "env")
    assert {k: cfg[k] for k in same} == {k: v2[k] for k in same}  # like-for-like but for the objective and servers


def test_teacher_scoring_retries_drops_and_raises(monkeypatch):
    """One retry per trajectory; a trajectory failing twice is dropped; all failing raises. The teacher sees the
    trajectory only up to its last trained token and its answer is aligned with ids."""
    import httpx

    import sarl.train as T

    calls = []
    fails = {"a": 1, "b": 2}  # a: fails once (retried), b: fails twice (dropped)

    def fake(url, model, ids, timeout=120, strict=False):
        calls.append((ids[0], len(ids)))
        assert url == "http://t" and model == "policy" and strict and timeout == 5.0
        key = {1: "a", 2: "b"}.get(ids[0], "c")
        if fails.get(key, 0) > 0:
            fails[key] -= 1
            raise httpx.ConnectError("down")
        return [-0.5] * (len(ids) - 1)

    monkeypatch.setattr(T, "served_logprobs", fake)
    cfg = _kd_cfg(teacher={"url": "http://t", "model_key": "mini-coder-4b"}, kd={"workers": 2, "timeout_s": 5.0})
    validate_objective(cfg)
    mask = [0, 1, 1, 0, 0]  # the trailing observation is not sent
    batch = [([k, 5, 6, 7, 8], mask, 0.0, [None, -1.0, -0.25, None, None]) for k in (1, 2, 3)]
    out, lps, errors = T.score_with_teacher(batch, cfg)
    assert [b[0][0] for b in out] == [1, 3] and len(errors) == 1 and "ConnectError" in errors[0]
    assert all(n == 3 for _, n in calls) and len(calls) == 2 + 2 + 1
    assert out[0][2] == [None, pytest.approx(0.5), pytest.approx(-0.25), None, None] and lps[0] == [-0.5, -0.5]
    # the group advantage (item[2]) reaches a_t through kd.reward_coef: a_t = coef·(t − s) + reward_coef·A_group
    cfg_r = {**cfg, "kd": {**cfg["kd"], "coef": 2.0, "reward_coef": 0.5}}
    out_r, _, _ = T.score_with_teacher([([3, 5, 6, 7, 8], mask, 2.0, [None, -1.0, -0.25, None, None])], cfg_r)
    assert out_r[0][2] == [None, pytest.approx(2.0 * 0.5 + 1.0), pytest.approx(2.0 * -0.25 + 1.0), None, None]
    fails.update(a=2, b=2, c=2)
    cfg_all = {**cfg, "kd": {**cfg["kd"], "max_consecutive_failures": 10}}  # below the circuit breaker's threshold
    with pytest.raises(RuntimeError, match="failed on all 3"):
        T.score_with_teacher(batch, cfg_all)

    def short(url, model, ids, timeout=120, strict=False):
        return [-0.5] * (len(ids) - 2)

    monkeypatch.setattr(T, "served_logprobs", short)
    with pytest.raises(RuntimeError, match="prompt log-probs"):
        T.teacher_logprobs(cfg["teacher"], [1, 5, 6], [0, 1, 1], 5.0)


class _Resp:
    def __init__(self, data):
        self._data = data

    def json(self):
        return {"data": self._data}


def test_preflight_teacher(monkeypatch):
    """The teacher serves teacher.model from MODELS[model_key]'s snapshot and fits max_train_tokens + 1 tokens."""
    import httpx

    import sarl.train as T
    from sarl.paths import MODELS

    mid, rev = MODELS["mini-coder-4b"]
    card = {"id": "policy", "root": f"/var/tmp/swe/hf-cache/hub/models--{mid.replace('/', '--')}/snapshots/{rev}", "max_model_len": 25600}
    monkeypatch.setattr(httpx, "get", lambda url, timeout: _Resp([card]))
    cfg = _kd_cfg(limits={"max_context_tokens": 24576}, train={"max_train_tokens": 24576})
    validate_objective(cfg)
    ident = T.preflight_teacher(cfg)
    assert ident["max_model_len"] == 25600 and ident["root"].endswith(rev)
    card["max_model_len"] = 24576  # a 24576-token view + max_tokens 1 would be refused by vLLM
    with pytest.raises(RuntimeError, match="max_model_len 24576 < 24577"):
        T.preflight_teacher(cfg)
    card.update(max_model_len=32768, root=card["root"].replace("mini-coder-4b", "mini-coder-1.7b"))
    with pytest.raises(RuntimeError, match="not mini-coder-4b"):
        T.preflight_teacher(cfg)
    card["id"] = "other"
    with pytest.raises(RuntimeError, match="does not serve policy"):
        T.preflight_teacher(cfg)


@pytest.mark.parametrize("finite", [True, False])
def test_served_logprobs_strict(monkeypatch, finite):
    """strict=True never takes another token's entry (the KD target must be the sampled token's log-prob)."""
    import httpx

    import sarl.train as T

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            other = {"99": {"logprob": -0.1}}
            return {"choices": [{"prompt_logprobs": [None, {"5": {"logprob": -1.0}}, other if not finite else {"6": {"logprob": -2.0}}]}]}

    monkeypatch.setattr(httpx, "post", lambda *a, **k: R())
    lax = T.served_logprobs("http://t", "policy", [1, 5, 6])
    strict = T.served_logprobs("http://t", "policy", [1, 5, 6], strict=True)
    assert lax[0] == strict[0] == -1.0
    if finite:
        assert lax[1] == strict[1] == -2.0
    else:
        assert lax[1] == -0.1 and math.isnan(strict[1])  # the old fallback stays for the existing callers


@pytest.mark.parametrize("mismatch", ["batch", "all"])
def test_update_returns_per_token_logprobs_for_kd_only(fresh_tiny_qwen, tmp_path, mismatch):
    """_update(per_token=True) hands back the trainer's per-token log-probs before/after the update (token_logprobs
    order) and never leaves them in the logged update record; the GRPO call is unchanged."""
    from sarl.train import _update

    tr = LoRATrainer("tiny", model=fresh_tiny_qwen, device="cpu", cfg=TrainConfig(lr=1e-2, lora_r=4, lora_alpha=8, loss_agg="token_mean"))
    b = _batch(2, seed=5)
    cfg, trs = {"diagnostics": {"mismatch": mismatch}}, {"max_train_tokens": 64}
    samp = []
    for ids, m in b:
        s = [None] * len(ids)
        for k, pos in enumerate(trained_positions(m)):
            s[pos] = float(tr.logprobs_nograd(ids, m)[k])
        samp.append(s)
    kd_batch = [(ids, m, [None if not (i and m[i]) else 0.3 for i in range(len(ids))], s) for (ids, m), s in zip(b, samp)]
    first = [tr.logprobs_nograd(ids, m) for ids, m in b]
    upd, mm, before, after, h, tok = _update(tr, kd_batch, [], cfg, trs, True, tmp_path, 1, per_token=True)
    assert "token_logprobs" not in upd and upd["stepped"] and h
    tb, ta = tok
    assert [x.shape for x in tb] == [x.shape for x in ta] == [(22,), (22,)]
    assert all(np.allclose(x, y, atol=1e-5) for x, y in zip(tb, first))  # the weights the step started from
    now = [tr.logprobs_nograd(ids, m) for ids, m in b]
    assert all(np.array_equal(x, y) for x, y in zip(ta, now))  # "after" = the weights the update produced
    assert any(not np.allclose(x, y, atol=1e-4) for x, y in zip(ta, tb))
    assert np.allclose(after, [x.mean() for x in ta]) and np.allclose(before, [x.mean() for x in tb], atol=1e-5)
    t_lps = [[-1.0] * (len(ids) - 1) for ids, _ in b]
    assert kd_step_stats(kd_batch, t_lps, tb, ta)["objective_gain"] > 0  # a = 0.3 everywhere: the update raised log-probs
    grpo = _update(tr, [(ids, m, 1.0, s) for (ids, m), s in zip(b, samp)], [], cfg, trs, True, tmp_path, 2)
    assert grpo[5] is None and "token_logprobs" not in grpo[0]


def test_check_teacher_tokenizer_wiring(monkeypatch, tmp_path):
    """run()'s start-up check loads MODELS[teacher.model_key]'s snapshot tokenizer and compares it with the student's."""
    import transformers

    import sarl.train as T

    asked, loaded = [], []

    def fake_snapshot(key):
        asked.append(key)
        (tmp_path / key).mkdir(exist_ok=True)
        return tmp_path / key

    teacher_vocab = {"a": 0, "b": 1, "<|im_end|>": 2, "<|endoftext|>": 3, "c": 4}

    def fake_from_pretrained(path, *a, **k):
        loaded.append(path)
        return _Tok(teacher_vocab)

    monkeypatch.setattr(T, "model_snapshot", fake_snapshot)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake_from_pretrained)
    cfg = _kd_cfg(model="ricdomolm/mini-coder-1.7b", revision="abc")
    validate_objective(cfg)
    student = _Tok({"a": 0, "b": 1, "<|im_end|>": 2, "<|endoftext|>": 3})
    with pytest.raises(RuntimeError, match="no tokenizer"):
        T.check_teacher_tokenizer(student, cfg)
    assert loaded == []
    (tmp_path / "mini-coder-4b" / "tokenizer_config.json").write_text("{}")
    with pytest.raises(RuntimeError, match="does not share the student's tokenizer.*vocabularies differ"):
        T.check_teacher_tokenizer(student, cfg)
    assert set(asked) == {"mini-coder-4b"} and loaded == [str(tmp_path / "mini-coder-4b")]
    del teacher_vocab["c"]
    assert T.check_teacher_tokenizer(student, cfg) is None


_IDS = itertools.count(1000, 1000)


def _ro(task, idx, reward, n=10, error=None, infra=False, graded=True):
    from sarl.train import Rollout

    base = next(_IDS)  # distinct token ids per rollout, so a batch item identifies its rollout
    traj = SimpleNamespace(ids=list(range(base, base + n)), mask=[0] * 3 + [1] * (n - 3), logprobs=[None] * 3 + [-1.0] * (n - 3),
                           n_tokens=n)
    g = SimpleNamespace(reward=reward, infra_failure=infra, resolved=bool(reward)) if graded else None
    return Rollout(task, "policy", idx, traj=traj, grade=g, error=error)


@pytest.mark.parametrize("truncate", [False, True])
def test_build_batch_grpo_and_kd(truncate):
    """GRPO trains only the non-degenerate valid groups, with the group advantages; kd trains every valid policy
    rollout (degenerate groups and a lone valid rollout with A_group = 0); both record the same groups. Crashed,
    ungraded and infra-failed rollouts are never trained."""
    from sarl.grpo import group_advantages
    from sarl.train import build_batch

    zero, mixed, lone, long_ = (SimpleNamespace(instance_id=x) for x in ("zero", "mixed", "lone", "long"))
    ros = [_ro(zero, 0, 0.0), _ro(zero, 1, 0.0), _ro(zero, 2, 0.0),
           _ro(mixed, 0, 1.0), _ro(mixed, 1, 0.0), _ro(mixed, 2, 0.0), _ro(mixed, 3, 1.0, graded=False),
           _ro(lone, 0, 1.0), _ro(lone, 1, 0.0, error="RuntimeError: crash"), _ro(lone, 2, 0.0, infra=True),
           _ro(long_, 0, 1.0, n=50), _ro(long_, 1, 0.0)]
    tasks = [zero, mixed, lone, long_]
    cfg = {"advantage": {"eps": 1e-4, "skip_degenerate": True}}
    a_mixed, a_long = [float(a) for a in group_advantages([1.0, 0.0, 0.0], 1e-4)], [float(a) for a in group_advantages([1.0, 0.0], 1e-4)]
    trained = lambda batch: [next(r for r in ros if r.traj.ids[:len(b[0])] == b[0]) for b in batch]  # noqa: E731

    g_batch, g_groups, g_deg, g_drop, g_trunc, g_or = build_batch(tasks, ros, cfg, False, 20, truncate)
    got = [(r.task.instance_id, r.idx) for r in trained(g_batch)]
    assert got == [("mixed", 0), ("mixed", 1), ("mixed", 2)] + ([("long", 0)] if truncate else []) + [("long", 1)]
    assert [b[2] for b in g_batch] == a_mixed + (a_long if truncate else a_long[1:])
    assert all(isinstance(b[2], float) for b in g_batch) and g_or == []
    assert (g_deg, g_drop, g_trunc) == (2, 0 if truncate else 1, 1 if truncate else 0)
    assert [(g["task"], g["degenerate"], g.get("invalid", False)) for g in g_groups] == [
        ("zero", True, False), ("mixed", False, False), ("lone", True, True), ("long", False, False)]
    assert g_groups[1]["adv"] == [round(a, 4) for a in a_mixed] and g_groups[0]["rewards"] == [0.0, 0.0, 0.0]

    k_batch, k_groups, k_deg, k_drop, k_trunc, k_or = build_batch(tasks, ros, cfg, True, 20, truncate)
    got = [(r.task.instance_id, r.idx) for r in trained(k_batch)]
    assert got == [("zero", 0), ("zero", 1), ("zero", 2), ("mixed", 0), ("mixed", 1), ("mixed", 2), ("lone", 0)] + (
        [("long", 0)] if truncate else []) + [("long", 1)]
    assert [b[2] for b in k_batch] == [0.0] * 3 + a_mixed + [0.0] + (a_long if truncate else a_long[1:])
    assert (k_groups, k_deg, k_drop, k_trunc, k_or) == (g_groups, g_deg, g_drop, g_trunc, g_or)
    assert ros[7].adv == 0.0 and not hasattr(ros[8], "adv") and not hasattr(ros[6], "adv")


def test_sleepers_for():
    import sarl.train as T

    servers = [{"url": "http://localhost:8001", "slots": 8}, {"url": "http://node04:8003/", "slots": 4, "sleep": True}]
    cfg = _kd_cfg(servers=servers)
    validate_objective(cfg)
    assert T.sleepers_for(cfg, servers) == ["http://node04:8003", "http://localhost:8002"]
    cfg = _kd_cfg(servers=servers, teacher={"url": "http://localhost:8002", "model_key": "mini-coder-4b", "sleep": False})
    validate_objective(cfg)
    assert T.sleepers_for(cfg, servers) == ["http://node04:8003"]
    cfg = _kd_cfg(teacher={"url": "http://localhost:8002", "model_key": "mini-coder-4b"})  # sleep defaults to False
    validate_objective(cfg)
    assert T.sleepers_for(cfg, cfg["servers"]) == []
    grpo = {"servers": servers}
    assert T.sleepers_for(grpo, servers) == ["http://node04:8003"] and T.sleepers_for(grpo, [{"url": "http://a"}]) == []


class _FakeTrainer:
    def __init__(self, events):
        self.events = events

    def onload(self):
        self.events.append("onload")

    def offload(self):
        self.events.append("offload")


def test_update_in_sleep_window(monkeypatch):
    """Sleep, onload, update, offload, wake, in that order and whatever the update does; with nothing to train no
    server sleeps and the trainer never moves onto the GPU of an awake server."""
    import sarl.train as T

    ev = []
    monkeypatch.setattr(T, "vllm_sleep", lambda u: ev.append(f"sleep {u}"))
    monkeypatch.setattr(T, "vllm_wake", lambda u: ev.append(f"wake {u}"))
    tr = _FakeTrainer(ev)

    def upd():
        ev.append("update")
        return "res"

    assert T.update_in_sleep_window(tr, [], ["http://t"], upd)[0] is None and ev == []
    res, t_sleep, t_wake = T.update_in_sleep_window(tr, [1], ["http://s", "http://t"], upd)
    assert res == "res" and t_sleep >= 0 and t_wake >= 0
    assert ev == ["sleep http://s", "sleep http://t", "onload", "update", "offload", "wake http://s", "wake http://t"]
    ev.clear()

    def boom():
        ev.append("update")
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="out of memory"):
        T.update_in_sleep_window(tr, [1], ["http://t"], boom)
    assert ev == ["sleep http://t", "onload", "update", "offload", "wake http://t"]
    ev.clear()
    assert T.update_in_sleep_window(tr, [1], [], upd)[0] == "res" and ev == ["onload", "update"]  # no sleep server
    ev.clear()
    assert T.update_in_sleep_window(tr, [], [], upd)[0] is None and ev == []


def test_kd_record():
    import sarl.train as T

    b = [([1, 2, 3], [0, 1, 1], [None, 0.5, -0.5], [None, -1.0, -1.0])]
    rec, gain = T.kd_record(b, [[-0.5, -1.5]], ([np.array([-1.0, -1.0])], [np.array([-0.8, -1.1])]), ["e1", "e2"], 12.34)
    assert rec["dropped"] == 2 and rec["teacher_seconds"] == 12.3 and "objective_gain" not in rec
    assert gain == pytest.approx((0.5 * 0.2 + -0.5 * -0.1) / 2) and rec["n_ok"] == 2
    rec0, gain0 = T.kd_record([], [], None, [], 0.0)
    assert rec0["n_tok"] == 0 and gain0 is None and rec0["dropped"] == 0


def test_teacher_smoke(monkeypatch):
    """One max_train_tokens-long strict request at start-up; a failure or a non-finite log-prob stops the run."""
    import sarl.train as T

    seen = []

    def fake(url, model, ids, timeout=120, strict=False):
        seen.append((url, len(ids), max(ids), timeout, strict))
        return [-3.0] * (len(ids) - 1)

    monkeypatch.setattr(T, "served_logprobs", fake)
    cfg = _kd_cfg(train={"max_train_tokens": 3000}, kd={"timeout_s": 7.0})
    validate_objective(cfg)
    out = T.teacher_smoke(cfg, vocab_size=100)
    assert len(seen) == 1 and seen[0][:2] == ("http://localhost:8002", 3000) and seen[0][2] < 100 and seen[0][3:] == (7.0, True)
    assert out["n_tokens"] == 3000 and out["mean_lp"] == -3.0
    monkeypatch.setattr(T, "served_logprobs", lambda url, model, ids, timeout=120, strict=False: [-3.0] * (len(ids) - 2) + [NAN])
    with pytest.raises(RuntimeError, match="1 non-finite log-probs of 2999"):
        T.teacher_smoke(cfg, 100)
    monkeypatch.setattr(T, "served_logprobs", lambda url, model, ids, timeout=120, strict=False: [-3.0] * 1024)  # short
    with pytest.raises(RuntimeError, match="failed a 3000-token scoring request.*1024 prompt log-probs"):
        T.teacher_smoke(cfg, 100)


def test_teacher_circuit_breaker(monkeypatch):
    """A teacher that keeps failing stops the step after kd.max_consecutive_failures trajectories (each tried twice),
    instead of spending two timeouts on every trajectory of the batch; one success resets the streak."""
    import httpx

    import sarl.train as T

    calls = []

    def dead(url, model, ids, timeout=120, strict=False):
        calls.append(ids[0])
        raise httpx.ReadTimeout("wedged")

    monkeypatch.setattr(T, "served_logprobs", dead)
    cfg = _kd_cfg(teacher={"url": "http://t", "model_key": "mini-coder-4b"}, kd={"workers": 1, "timeout_s": 5.0})
    validate_objective(cfg)
    assert cfg["kd"]["max_consecutive_failures"] == 3
    mask = [0, 1, 1]
    batch = [([k, 5, 6], mask, 0.0, [None, -1.0, -1.0]) for k in range(1, 11)]
    with pytest.raises(RuntimeError, match="3 trajectories in a row"):
        T.score_with_teacher(batch, cfg)
    assert len(calls) == 3 * 2  # with one worker: exactly three trajectories, two attempts each, then nothing more

    calls.clear()

    def flaky(url, model, ids, timeout=120, strict=False):  # every other trajectory fails twice: the streak resets
        calls.append(ids[0])
        if ids[0] % 2:
            raise httpx.ConnectError("down")
        return [-0.5] * (len(ids) - 1)

    monkeypatch.setattr(T, "served_logprobs", flaky)
    out, _, errors = T.score_with_teacher(batch, cfg)
    assert [b[0][0] for b in out] == [2, 4, 6, 8, 10] and len(errors) == 5


def test_sleep_window_always_wakes(monkeypatch):
    """The servers are woken even when the trainer's offload fails or a sleep fails part-way; an update error is not
    masked by cleanup; a cleanup failure with no other error stops the run."""
    import sarl.train as T

    ev = []
    monkeypatch.setattr(T, "vllm_wake", lambda u: ev.append(f"wake {u}"))

    class BadOffload(_FakeTrainer):
        def offload(self):
            self.events.append("offload")
            raise RuntimeError("offload failed")

    monkeypatch.setattr(T, "vllm_sleep", lambda u: ev.append(f"sleep {u}"))
    with pytest.raises(RuntimeError, match="sleep-window cleanup failed"):
        T.update_in_sleep_window(BadOffload(ev), [1], ["http://s", "http://t"], lambda: ev.append("update"))
    assert ev[-2:] == ["wake http://s", "wake http://t"]

    ev.clear()

    def boom():
        raise ValueError("update broke")

    with pytest.raises(ValueError, match="update broke"):  # the original error wins over the cleanup error
        T.update_in_sleep_window(BadOffload(ev), [1], ["http://t"], boom)
    assert ev[-1] == "wake http://t"

    ev.clear()

    def sleep_fails_on_t(u):
        ev.append(f"sleep {u}")
        if u == "http://t":
            raise RuntimeError("sleep refused")

    monkeypatch.setattr(T, "vllm_sleep", sleep_fails_on_t)
    with pytest.raises(RuntimeError, match="sleep refused"):
        T.update_in_sleep_window(_FakeTrainer(ev), [1], ["http://s", "http://t"], lambda: ev.append("update"))
    assert "update" not in ev and ev[-2:] == ["wake http://s", "wake http://t"]
