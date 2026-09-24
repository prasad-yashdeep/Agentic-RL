"""The harness: prepare → call → parse → execute → append, until an exit. Deterministic code, never trained.

It owns the transcript as TOKEN IDS, not text: the prompt is rendered once with the chat template; every policy
reply is appended exactly as sampled (mask 1, with its log-probs); every environment observation is appended as a
templated `tool` turn (mask 0). Re-rendering the transcript later would not reproduce the sampled tokens (templates
re-serialise tool calls, merge consecutive tool messages, drop <think>), so the trainer consumes these ids directly.

    ids:   [system+tools | user issue | asst reply₁ | tool obs₁ | asst reply₂ | tool obs₂ | … ]
    mask:  [     0       |     0      |   1 … 1    |  0 … 0    |   1 … 1    |  0 … 0    | … ]
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from sarl.protocol import BASH_TOOL, INSTANCE_TEMPLATE, SUBMIT, action_text, format_observation, is_submit, parse_reply, system_prompt, tools_for


@dataclass
class Limits:
    max_turns: int = 20  # the diagram: 50
    max_context_tokens: int = 16_384  # the diagram: 32k (Qwen2.5-Coder-0.5B native); 16k keeps training on one 16 GB card
    max_new_tokens: int = 1024
    min_reply_tokens: int = 64
    wall_clock_s: float = 1800.0  # the diagram: 30 min
    command_timeout_s: float = 120.0
    max_obs_chars: int = 10_000
    max_issue_chars: int = 6000
    max_format_errors: int = 3  # consecutive


@dataclass
class Trajectory:
    task_id: str
    ids: list[int]
    mask: list[int]
    logprobs: list[float | None]
    messages: list[dict]
    turns: list[dict]
    exit_reason: str
    patch: str
    commands: list[str]
    n_format_errors: int
    wall_s: float
    policy: str = ""
    off_policy: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def n_tokens(self) -> int:
        return len(self.ids)

    @property
    def n_assistant_tokens(self) -> int:
        return int(sum(self.mask))

    def to_json(self, with_tokens: bool = True) -> dict:
        d = asdict(self)
        if not with_tokens:
            for k in ("ids", "mask", "logprobs"):
                d.pop(k)
        d["n_tokens"], d["n_assistant_tokens"] = self.n_tokens, self.n_assistant_tokens
        return d


class ChatFormat:
    """Token-level turn deltas derived from the tokenizer's own chat template (ChatML for Qwen2.5)."""

    PROBE = "\x00SARL_PROBE\x00"

    def __init__(self, tokenizer, tools: list[dict] | None = None, template_kwargs: dict | None = None):
        self.tok = tokenizer
        self.tools = [BASH_TOOL] if tools is None else list(tools)  # [] = no tools block (backticks protocol)
        # Qwen3 templates take enable_thinking; sarl runs them in non-thinking mode. Templates that do not know the
        # kwarg ignore it (Jinja), so passing it for Qwen2.5 is harmless.
        self.template_kwargs = {"enable_thinking": False, **(template_kwargs or {})}
        self._special = sorted({t for t in getattr(tokenizer, "all_special_tokens", []) if t.startswith("<|")} |
                               {t for t in getattr(tokenizer, "added_tokens_encoder", {}) if t.startswith("<|")}, key=len, reverse=True)
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.eot_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self.tool_prefix, self.tool_suffix = self._delta("tool")
        self.user_prefix, self.user_suffix = self._delta("user")

    def _render(self, messages, gen=True) -> str:
        return self.tok.apply_chat_template(messages, tools=self.tools or None, add_generation_prompt=gen, tokenize=False,
                                            **self.template_kwargs)

    def _delta(self, role: str) -> tuple[str, str]:
        """The text between the end of a sampled assistant reply (its <|im_end|>) and the start of the next one.

        Derived from a single render in which the assistant turn is followed by the new turn and the generation
        prompt. We do not require the template to be append-only: Qwen3's template renders the *last* assistant turn
        with an empty <think></think> block but earlier ones without, so the transcript is never re-rendered — the
        harness keeps the sampled ids, and only these deltas come from the template."""
        base = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
                {"role": "assistant", "content": "SARL_ASSISTANT_REPLY"}]
        after = self._render(base + [{"role": role, "content": self.PROBE}], gen=True)
        cut = after.index("SARL_ASSISTANT_REPLY") + len("SARL_ASSISTANT_REPLY")
        tail = after[cut:]
        assert tail.startswith("<|im_end|>"), f"unexpected assistant turn ending: {tail[:40]!r}"
        pre, post = tail[len("<|im_end|>"):].split(self.PROBE)  # the policy samples <|im_end|> itself
        return pre, post

    def encode(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def sanitize(self, text: str) -> str:
        """Untrusted text (issue, command output) must not contain literal special tokens: '<|im_end|>' printed by a
        command would otherwise tokenize as the real turn delimiter and forge a turn boundary (review finding). A word
        joiner after '<|' keeps it readable but plain text."""
        if "<|" not in text:
            return text
        for t in self._special:
            if t in text:
                text = text.replace(t, t[:2] + "\u2060" + t[2:])
        return text

    def prompt_ids(self, messages: list[dict]) -> list[int]:
        return self.encode(self._render(messages, gen=True))

    def tool_turn(self, observation: str) -> list[int]:
        return self.encode(self.tool_prefix + self.sanitize(observation) + self.tool_suffix)

    def user_turn(self, text: str) -> list[int]:
        return self.encode(self.user_prefix + self.sanitize(text) + self.user_suffix)


def initial_messages(task, limits: Limits, protocol: str = "tool_call", fmt: "ChatFormat | None" = None) -> list[dict]:
    issue = fmt.sanitize(task.problem_statement) if fmt is not None else task.problem_statement
    if len(issue) > limits.max_issue_chars:
        issue = issue[: limits.max_issue_chars] + "\n[... issue truncated ...]"
    return [{"role": "system", "content": system_prompt(protocol)},
            {"role": "user", "content": INSTANCE_TEMPLATE.format(problem_statement=issue)}]


def run_episode(policy, env, task, fmt: ChatFormat, limits: Limits | None = None, *, policy_name: str = "",
                protocol: str = "tool_call") -> Trajectory:
    """One rollout. The env must already be started (callers own its lifecycle). `fmt` must be built with
    tools_for(protocol) so the prompt advertises the same action format the parser accepts."""
    lim = limits or Limits()
    t0 = time.time()
    messages = initial_messages(task, lim, protocol, fmt)
    ids = fmt.prompt_ids(messages)
    mask, lps = [0] * len(ids), [None] * len(ids)
    turns: list[dict] = []
    commands: list[str] = []
    n_fmt = streak = 0
    exit_reason = "max_turns"
    scripted = False
    for k in range(lim.max_turns):
        budget = lim.max_context_tokens - len(ids)
        if budget < lim.min_reply_tokens:
            exit_reason = "context"
            break
        if time.time() - t0 > lim.wall_clock_s:
            exit_reason = "wall_clock"
            break
        gen = policy.generate(ids, min(lim.max_new_tokens, budget))
        scripted = scripted or bool(gen.meta.get("scripted"))
        g_ids, g_lp = list(gen.token_ids), list(gen.logprobs)
        if len(g_lp) != len(g_ids):
            g_lp = (g_lp + [None] * len(g_ids))[: len(g_ids)]
        ids += g_ids
        mask += [1] * len(g_ids)
        lps += g_lp
        closed = bool(g_ids) and g_ids[-1] == fmt.im_end_id
        if not closed:  # cut off by max_tokens, or <|endoftext|>: close the turn ourselves (not sampled → mask 0)
            ids.append(fmt.im_end_id); mask.append(0); lps.append(None)
        messages.append({"role": "assistant", "content": gen.text})
        rec: dict[str, Any] = {"turn": k, "gen_tokens": len(g_ids), "finish": gen.finish_reason, "closed": closed}
        parsed = parse_reply(gen.text, protocol)
        if not parsed.ok:
            n_fmt += 1
            streak += 1
            rec.update(format_error=parsed.error, n_calls=parsed.n_calls)
            turns.append(rec)
            if streak >= lim.max_format_errors:
                exit_reason = "format_errors"
                break
            messages.append({"role": "user", "content": parsed.error})
            d = fmt.user_turn(parsed.error)
        else:
            streak = 0
            cmd = parsed.command
            commands.append(cmd)
            res = env.run(cmd, timeout=lim.command_timeout_s)
            rec.update(command=cmd, returncode=res.returncode, timed_out=res.timed_out, obs_chars=len(res.output), exec_s=round(res.seconds, 3))
            turns.append(rec)
            if is_submit(cmd):
                exit_reason = "submitted"
                break
            obs = fmt.sanitize(format_observation(res.returncode, res.output, max_chars=lim.max_obs_chars, timed_out=res.timed_out))
            if protocol == "tool_call":
                messages.append({"role": "tool", "content": obs})
                d = fmt.tool_turn(obs)
            else:  # the reference appends the observation as a plain user turn
                messages.append({"role": "user", "content": obs})
                d = fmt.user_turn(obs)
        ids += d
        mask += [0] * len(d)
        lps += [None] * len(d)
    patch = env.patch()
    return Trajectory(task_id=task.instance_id, ids=ids, mask=mask, logprobs=lps, messages=messages, turns=turns,
                      exit_reason=exit_reason, patch=patch, commands=commands, n_format_errors=n_fmt,
                      wall_s=round(time.time() - t0, 2), policy=policy_name, off_policy=scripted, meta={"protocol": protocol})


def oracle_replies(task, protocol: str = "tool_call") -> list[str]:
    """A scripted expert for a real SWE-bench task: write the gold patch, apply it, submit (3 actions), in the same
    action format the policy is asked to use, so its tokens teach that format."""
    patch = task.patch if task.patch.endswith("\n") else task.patch + "\n"
    write = f"cat > /tmp/sarl_fix.patch <<'SARL_PATCH_EOF'\n{patch}SARL_PATCH_EOF"
    return [
        action_text(write, "I know the fix; I'll write it as a patch file first.", protocol),
        action_text("git apply /tmp/sarl_fix.patch && git diff --stat", "Now apply it and check what changed.", protocol),
        action_text(f"echo {SUBMIT}", "The fix is in place. Submitting.", protocol),
    ]
