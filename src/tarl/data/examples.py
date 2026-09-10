"""Trajectory -> (token ids, assistant mask) training examples.

Two renderings:

* ``legacy06`` — bit-exact reproduction of scripts/06_grpo_micro.py:154-183 (used by the G4
  micro-run): every ATIF step becomes a chat turn whose content is ``step["message"]``. Because
  Harbor renders agent steps as "Analysis: …\\nPlan: …" prose, this sequence contains NO commands,
  NO terminal observations and NO thinking (finding #1 in the 2026-09-09 review).
* ``terminus_chat`` — the conversation Terminus 2 actually ran: user(system prompt + task) ->
  assistant(synthesised JSON action) -> user(terminal observation) -> … ; optionally with the
  ``reasoning_content`` attached to assistant turns. Still an approximation of the sampled tokens
  (raw completions were not recorded); exact ids need Harbor's ``collect_rollout_details``.

The assistant mask is the single most important implementation detail of the whole method
(docs/03-math.md §1): only tokens the policy emitted may carry loss and importance ratio.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from tarl.common.paths import BASE_MODEL, HF_CACHE
from tarl.judge.compressor import load_trajectory

Render = Literal["legacy06", "terminus_chat", "token_ids"]


class TokenizerUnavailable(RuntimeError):
    pass


@dataclass
class Span:
    turn: int
    role: str
    start: int
    end: int
    n_tokens: int
    preview: str


@dataclass
class TrainingExample:
    ids: list[int]
    mask: list[int]
    spans: list[Span]
    render: str
    prefix_stable: bool = True
    dropped_reason: str | None = None  # None | "no_steps" | "overlong" | "empty_mask"
    meta: dict = field(default_factory=dict)

    @property
    def n_tokens(self) -> int:
        return len(self.ids)

    @property
    def n_assistant_tokens(self) -> int:
        return int(sum(self.mask))

    @property
    def kept(self) -> bool:
        return self.dropped_reason is None


# --------------------------------------------------------------------------- rendering
def _as_traj(trajectory: dict | str | Path) -> dict:
    if isinstance(trajectory, (str, Path)):
        return load_trajectory(trajectory)
    return trajectory


def synth_terminus_json(step: dict) -> str:
    """Rebuild the Terminus-2 JSON action from Harbor's rendered step (message + tool_calls)."""
    msg = str(step.get("message") or "")
    analysis, plan = msg, ""
    if "\nPlan:" in msg:
        analysis, plan = msg.split("\nPlan:", 1)
    analysis = analysis.removeprefix("Analysis:").strip()
    plan = plan.strip()
    cmds: list[dict] = []
    complete = False
    for tc in step.get("tool_calls") or []:
        if tc.get("function_name") == "mark_task_complete":
            complete = True
            continue
        args = tc.get("arguments") or {}
        if args.get("keystrokes") is not None:
            cmds.append({"keystrokes": args["keystrokes"], "duration": args.get("duration", 1.0)})
    obj: dict[str, Any] = {"analysis": analysis, "plan": plan, "commands": cmds}
    if complete:
        obj["task_complete"] = True
    return json.dumps(obj, ensure_ascii=False)


def render_conversation(
    trajectory: dict | str | Path,
    *,
    render: Render = "legacy06",
    include_reasoning: bool = False,
) -> list[dict]:
    traj = _as_traj(trajectory)
    steps = list(traj.get("steps") or [])
    if render == "legacy06":
        return [
            {"role": "assistant" if s.get("source") == "agent" else "user", "content": str(s.get("message", ""))}
            for s in steps
        ]
    if render != "terminus_chat":
        raise ValueError(f"unknown render {render!r}")
    from tarl.data.loader import step_observation

    convo: list[dict] = []
    agent_idx = [i for i, s in enumerate(steps) if s.get("source") == "agent"]
    last_agent = agent_idx[-1] if agent_idx else -1
    for i, s in enumerate(steps):
        if s.get("source") != "agent":
            convo.append({"role": "user", "content": str(s.get("message") or "")})
            continue
        turn: dict[str, Any] = {"role": "assistant", "content": synth_terminus_json(s)}
        if include_reasoning and s.get("reasoning_content"):
            turn["reasoning_content"] = str(s["reasoning_content"])
        convo.append(turn)
        if i != last_agent:  # the observation after the final turn was never shown to the model
            obs = step_observation(s)
            if obs:
                convo.append({"role": "user", "content": obs})
    return convo


# --------------------------------------------------------------------------- tokenization
def _apply_template(tok, convo: list[dict], chat_template: str | None) -> str:
    kw: dict[str, Any] = {"tokenize": False, "add_generation_prompt": False}
    if chat_template is not None:
        kw["chat_template"] = chat_template
    return tok.apply_chat_template(convo, **kw)


def _from_token_ids(trajectory: dict) -> TrainingExample | None:
    """Assemble from recorded token ids (Harbor collect_rollout_details) when they exist."""
    steps = [s for s in (trajectory.get("steps") or []) if s.get("source") == "agent"]
    if not steps:
        return None
    mets = [s.get("metrics") or {} for s in steps]
    if not all(m.get("completion_token_ids") for m in mets) or not mets[-1].get("prompt_token_ids"):
        return None
    seq = list(mets[-1]["prompt_token_ids"]) + list(mets[-1]["completion_token_ids"])
    mask = [0] * len(seq)
    spans: list[Span] = []
    stable = True
    pos = 0
    for k, m in enumerate(mets):
        comp = list(m["completion_token_ids"])
        n = len(comp)
        found = -1
        for start in range(pos, len(seq) - n + 1):
            if seq[start : start + n] == comp:
                found = start
                break
        if found < 0:
            stable = False
            continue
        for j in range(found, found + n):
            mask[j] = 1
        spans.append(Span(turn=k, role="assistant", start=found, end=found + n, n_tokens=n, preview=""))
        pos = found + n
    return TrainingExample(ids=seq, mask=mask, spans=spans, render="token_ids", prefix_stable=stable)


def build_example(
    tok,
    trajectory: dict | str | Path,
    *,
    render: Render = "legacy06",
    max_seq: int | None = 1536,
    chat_template: str | None = None,
    include_reasoning: bool = False,
    prefer_token_ids: bool = True,
) -> TrainingExample:
    """(ids, mask) with mask==1 exactly on tokens of assistant-authored turns.

    Tokenizes the conversation prefix-by-prefix (as 06 did) so each turn's span is delimited by
    the chat template itself. ``max_seq`` is an overlong FILTER (DAPO): longer episodes are
    flagged ``dropped_reason="overlong"`` rather than truncated, because truncation would
    attribute the episode's reward to a cut prefix. ``max_seq=None`` disables it.
    """
    traj = _as_traj(trajectory)
    steps = traj.get("steps") or []
    if not steps:
        return TrainingExample(ids=[], mask=[], spans=[], render=render, dropped_reason="no_steps")
    if prefer_token_ids:
        ex = _from_token_ids(traj)
        if ex is not None:
            if max_seq is not None and ex.n_tokens > max_seq:
                ex.dropped_reason = "overlong"
            elif ex.n_assistant_tokens == 0:
                ex.dropped_reason = "empty_mask"
            return ex
    convo = render_conversation(traj, render=render, include_reasoning=include_reasoning)
    ids: list[int] = []
    mask: list[int] = []
    spans: list[Span] = []
    prev_len = 0
    prev_ids: list[int] = []
    stable = True
    for i in range(len(convo)):
        text = _apply_template(tok, convo[: i + 1], chat_template)
        cur = tok(text, add_special_tokens=False)["input_ids"]
        span = len(cur) - prev_len
        if span <= 0:
            continue
        if cur[:prev_len] != prev_ids:
            stable = False
        flag = 1 if convo[i]["role"] == "assistant" else 0
        spans.append(
            Span(
                turn=i,
                role=convo[i]["role"],
                start=prev_len,
                end=len(cur),
                n_tokens=span,
                preview=str(convo[i]["content"])[:60].replace("\n", " "),
            )
        )
        ids = cur
        mask.extend([flag] * span)
        prev_len = len(cur)
        prev_ids = cur
    ex = TrainingExample(ids=ids, mask=mask[: len(ids)], spans=spans, render=render, prefix_stable=stable)
    if not stable:
        # The incremental method assumes tokenizing a longer prefix never changes earlier tokens.
        # The Gemma-4 template breaks that for consecutive same-role messages (legacy06 renders
        # every agent step as `assistant`, so the template merges them into ONE model turn and the
        # `<turn|>\n` boundary tokens disappear). Recompute the mask from character offsets.
        try:
            ids2, mask2, spans2 = _offsets_mask(tok, convo, chat_template)
            ex.meta["incremental_ids"] = ids
            ex.meta["incremental_mask"] = mask[: len(ids)]
            inc_mask = mask[: len(ids)]
            ex.meta["ids_changed"] = sum(1 for a, b in zip(ids, ids2) if a != b) + abs(len(ids) - len(ids2))
            ex.meta["mask_tokens_changed"] = sum(1 for a, b in zip(inc_mask, mask2) if a != b) + abs(len(inc_mask) - len(mask2))
            ex.ids, ex.mask, ex.spans = ids2, mask2, spans2
            ex.meta["mask_method"] = "offsets"
        except Exception as e:  # pragma: no cover - tokenizer without offsets
            ex.meta["mask_method"] = f"incremental (offsets unavailable: {e})"
    else:
        ex.meta["mask_method"] = "incremental"
    if max_seq is not None and len(ex.ids) > max_seq:
        ex.dropped_reason = "overlong"
    elif ex.n_assistant_tokens == 0:
        ex.dropped_reason = "empty_mask"
    return ex


def _offsets_mask(tok, convo: list[dict], chat_template: str | None) -> tuple[list[int], list[int], list[Span]]:
    """Mask from character offsets: tokens overlapping an assistant message's content are trained.
    Works regardless of how the template joins turns (needs a fast tokenizer)."""
    text = _apply_template(tok, convo, chat_template)
    ranges: list[tuple[int, int, int, str]] = []
    pos = 0
    for i, m in enumerate(convo):
        content = str(m["content"])
        start = text.find(content, pos) if content else -1
        if start < 0:
            continue
        ranges.append((start, start + len(content), i, m["role"]))
        pos = start + len(content)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offs = enc["offset_mapping"]
    mask = [0] * len(ids)
    spans: list[Span] = []
    for (a, b, turn, role) in ranges:
        first = last = None
        for t, (s0, e0) in enumerate(offs):
            if e0 <= a or s0 >= b or e0 == s0:
                continue
            if role == "assistant":
                mask[t] = 1
            first = t if first is None else first
            last = t
        if first is not None:
            spans.append(Span(turn=turn, role=role, start=first, end=last + 1, n_tokens=last + 1 - first, preview=str(convo[turn]["content"])[:60].replace("\n", " ")))
    return ids, mask, spans


def to_tensors(ex: TrainingExample):
    """(input_ids, agent_mask) tensors — what tarl.training.loss.grpo_step consumes."""
    import torch

    return torch.tensor(ex.ids, dtype=torch.long), torch.tensor(ex.mask, dtype=torch.long)


def mask_table(ex: TrainingExample, tok, *, max_rows: int | None = None):
    """DataFrame(pos, token_id, token, mask, turn, role) for display."""
    import pandas as pd

    role_at: dict[int, tuple[int, str]] = {}
    for sp in ex.spans:
        for p in range(sp.start, sp.end):
            role_at[p] = (sp.turn, sp.role)
    toks = tok.convert_ids_to_tokens(ex.ids)
    rows = []
    for p, (tid, tk, m) in enumerate(zip(ex.ids, toks, ex.mask)):
        turn, role = role_at.get(p, (-1, "?"))
        rows.append({"pos": p, "token_id": tid, "token": tk, "mask": m, "turn": turn, "role": role})
        if max_rows is not None and len(rows) >= max_rows:
            break
    return pd.DataFrame(rows)


def reconstruction_gap(ex: TrainingExample, trajectory: dict | str | Path):
    """Per assistant turn: tokens in the re-rendered span vs the completion tokens the model
    actually generated (``metrics.completion_tokens``). Large negative deltas = content that
    the rendering dropped (reasoning, JSON scaffolding, commands)."""
    import pandas as pd

    traj = _as_traj(trajectory)
    agent_steps = [s for s in (traj.get("steps") or []) if s.get("source") == "agent"]
    asst = [sp for sp in ex.spans if sp.role == "assistant"]
    rows = []
    for k, s in enumerate(agent_steps):
        m = s.get("metrics") or {}
        gen = m.get("completion_tokens")
        span = asst[k].n_tokens if k < len(asst) else None
        rows.append(
            {
                "agent_turn": k + 1,
                "retokenized_span": span,
                "completion_tokens": gen,
                "delta": (span - gen) if (span is not None and gen is not None) else None,
                "has_reasoning": bool(s.get("reasoning_content")),
                "n_commands": len([tc for tc in s.get("tool_calls") or [] if (tc.get("arguments") or {}).get("keystrokes")]),
            }
        )
    return pd.DataFrame(rows)


def load_gemma_tokenizer(model: str = BASE_MODEL, *, offline: bool = True):
    """The policy tokenizer from the offline HF cache (never touches the 16 GB weights)."""
    os.environ.setdefault("HF_HOME", str(HF_CACHE))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer

    errors: list[str] = []
    snap_root = HF_CACHE / "hub" / f"models--{model.replace('/', '--')}" / "snapshots"
    candidates: list[str] = []
    if snap_root.exists():
        for snap in sorted(snap_root.iterdir()):
            if (snap / "tokenizer_config.json").exists():
                candidates.append(str(snap))
    candidates.append(model)
    for cand in candidates:
        try:
            return AutoTokenizer.from_pretrained(cand)
        except Exception as e:  # pragma: no cover - environment dependent
            errors.append(f"{cand}: {e}")
    raise TokenizerUnavailable(f"cannot load tokenizer {model} (HF_HOME={HF_CACHE}): " + " | ".join(errors))


def load_serve_chat_template() -> str | None:
    """The thinking-enabled template vLLM served with (configs/serve/*_thinking.jinja)."""
    from tarl.common.paths import SERVE_THINKING_TEMPLATE

    try:
        return SERVE_THINKING_TEMPLATE.read_text()
    except OSError:
        return None
