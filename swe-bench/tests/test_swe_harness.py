"""The harness keeps the transcript token-exact and masks only environment/prompt tokens."""

import pytest

from sarl.env import MockEnv
from sarl.harness import ChatFormat, Limits, run_episode
from sarl.policy import Generation, ScriptedPolicy
from sarl.protocol import SUBMIT, action_text, tools_for

pytestmark = pytest.mark.qwen


class _Task:
    instance_id = "toy__toy-1"
    problem_statement = "f(1) should return 2, not 1."


SRC = "def f(x):\n    return x\n"
FIX = "def f(x):\n    return x + 1\n"


@pytest.mark.parametrize("protocol", ["tool_call", "backticks"])
def test_scripted_episode_is_token_exact(qwen_tok, protocol):
    fmt = ChatFormat(qwen_tok, tools_for(protocol))
    replies = [action_text("cat pkg/f.py", "read", protocol),
               action_text(f"cat > pkg/f.py <<'EOF'\n{FIX}EOF", "fix", protocol),
               action_text(f"echo {SUBMIT}", "submit", protocol)]
    env = MockEnv({"pkg/f.py": SRC})
    tr = run_episode(ScriptedPolicy(qwen_tok, replies), env, _Task(), fmt, Limits(max_turns=6), protocol=protocol)
    assert tr.exit_reason == "submitted" and len(tr.turns) == 3 and tr.off_policy
    assert "+    return x + 1" in tr.patch
    assert len(tr.ids) == len(tr.mask) == len(tr.logprobs)
    # incremental ids == one-shot tokenisation of the rendered conversation (assistant turns as plain text)
    full = qwen_tok.apply_chat_template(tr.messages, tools=fmt.tools or None, add_generation_prompt=False, tokenize=False)
    full_ids = qwen_tok(full, add_special_tokens=False)["input_ids"]
    assert full_ids[: len(tr.ids)] == tr.ids
    # mask 1 exactly on the assistant replies (each ends with <|im_end|>)
    ends = [i for i, (t, m) in enumerate(zip(tr.ids, tr.mask)) if m and t == fmt.im_end_id]
    assert len(ends) == 3


def test_format_error_path_and_truncation(qwen_tok):
    fmt = ChatFormat(qwen_tok, tools_for("tool_call"))

    class Chatty:
        def generate(self, ids, max_tokens):
            toks = qwen_tok("I think the bug is in f.", add_special_tokens=False)["input_ids"]  # no tool call, no <|im_end|>
            return Generation(toks, [-0.1] * len(toks), "I think the bug is in f.", "length")

    tr = run_episode(Chatty(), MockEnv({"pkg/f.py": SRC}), _Task(), fmt, Limits(max_turns=10, max_format_errors=3))
    assert tr.exit_reason == "format_errors" and tr.n_format_errors == 3 and tr.patch == ""
    # a cut-off reply is closed with <|im_end|> that was never sampled: mask 0, no log-prob
    closers = [i for i, t in enumerate(tr.ids) if t == fmt.im_end_id and tr.mask[i] == 0 and tr.mask[i - 1] == 1]
    assert closers and all(tr.logprobs[i] is None for i in closers)


def test_context_budget_exit(qwen_tok):
    fmt = ChatFormat(qwen_tok, tools_for("backticks"))
    tr = run_episode(ScriptedPolicy(qwen_tok, [action_text("ls", "look", "backticks")]), MockEnv({"a.py": "x"}), _Task(), fmt,
                     Limits(max_turns=50, max_context_tokens=700, min_reply_tokens=64), protocol="backticks")
    assert tr.exit_reason == "context" and len(tr.ids) <= 700 + 64
