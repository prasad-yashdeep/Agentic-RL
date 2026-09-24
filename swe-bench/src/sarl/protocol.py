"""The action/observation protocol from the lecture's diagram (mini-swe-agent v2 in tool-calling mode).

ACTION: the policy emits exactly ONE bash tool call per turn. Qwen2.5 renders tool calls hermes-style:

    <tool_call>
    {"name": "bash", "arguments": {"command": "grep -n cright astropy/modeling/separable.py"}}
    </tool_call>

(The diagram abbreviates this to {"name":"bash","command":…}; we accept both.)

OBSERVATION: the harness executes the command in the sandbox and appends a `tool` message:

    <returncode>0</returncode>
    <output>
    …
    </output>

A second action format, `backticks`, is the reference notebook's (phase 1): exactly one fenced ```bash block per
reply, the observation appended as a plain user turn. A 0.5B model follows it; it does not follow the JSON tool call
(0/48 valid calls in 16 real episodes on 2026-09-24), which is the lecture's point at 2:50 — a base model that does
not know the action space needs SFT (or a larger model) before RL on the diagram's protocol.

Exits: the submit sentinel, the turn budget, the context budget, the wall clock, or repeated format errors.
Nothing in the loop is graded; grading happens once, after the episode (sarl.reward).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from sarl import prompts_mswea

SUBMIT = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Run one bash command in the repository sandbox. The working directory is the repository root "
            "(/testbed). Each call runs in a fresh subshell, but file changes persist between calls."
        ),
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to execute."}},
            "required": ["command"],
        },
    },
}

SYSTEM_PROMPT = f"""You are a software engineer fixing a bug in a Python repository. The repository is checked out in /testbed.

Work in small steps. Every reply must contain a short thought followed by exactly ONE bash tool call.

Recommended workflow:
1. Find the relevant code (ls, grep -rn, sed -n 'START,ENDp' FILE).
2. Reproduce the bug with a small script if useful.
3. Edit the source code (for example with sed -i, or a python one-liner). Do not modify tests.
4. Verify your fix.
5. When you are done, submit by running exactly: echo {SUBMIT}

Your changes to the repository are collected automatically when you submit.

Example of a correct reply (one thought, then one tool call, nothing else):
Let me look at the repository layout first.
<tool_call>
{{"name": "bash", "arguments": {{"command": "ls"}}}}
</tool_call>"""

BACKTICKS_SYSTEM_PROMPT = f"""You are a software engineering agent. The repository is checked out in /testbed.
Fix the bug described in the issue.

Respond with exactly ONE bash command per message, in a fenced block:

```bash
your command here
```

Useful commands:
  ls
  grep -rn "pattern" src/
  sed -n '1,80p' path/to/file.py
  sed -i 's/old text/new text/' path/to/file.py
  python -m pytest path/to/test_file.py -x -q

When the fix is done, submit with:

```bash
echo {SUBMIT}
```

One short sentence of reasoning, then exactly one command block."""

PROTOCOLS = ("tool_call", "backticks", "mswea")  # mswea = mini-swe-agent v1 text (the mini-coder SFT format)

INSTANCE_TEMPLATE = """Fix the issue described below in the repository at /testbed.

<issue>
{problem_statement}
</issue>"""

FORMAT_ERROR_NONE = (
    "Format error: your reply contained no tool call. Reply with a short thought and exactly one "
    '<tool_call>{"name": "bash", "arguments": {"command": "..."}}</tool_call>.'
)
FORMAT_ERROR_MANY = "Format error: your reply contained {n} tool calls. Use exactly ONE bash tool call per reply."
FORMAT_ERROR_BAD = "Format error: {why}. The call must look like {{\"name\": \"bash\", \"arguments\": {{\"command\": \"...\"}}}}."

NO_COMMAND = "No command found. Reply with exactly one ```bash block."  # the reference's message

_BASH_RE = re.compile(r"```(?:bash|sh)?\s*\n(.*?)```", re.S)  # the reference's first_bash_block
_MSWEA_RE = re.compile(r"```bash\s*\n(.*?)\n```", re.S)  # mini-swe-agent v1 action_regex; exactly one match required
_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_OPEN_RE = re.compile(r"<tool_call>(?!.*</tool_call>)", re.S)
_SUBMIT_RE = re.compile(r"^\s*echo\s+['\"]?" + SUBMIT + r"['\"]?\s*(?:&&.*)?$", re.S)


@dataclass(frozen=True)
class Parsed:
    command: str | None  # the one bash command, or None
    error: str | None  # a format-error message to feed back, or None
    n_calls: int

    @property
    def ok(self) -> bool:
        return self.command is not None


def _command_from(obj) -> tuple[str | None, str | None]:
    # the Qwen/hermes form {"name":"bash","arguments":{"command":…}} and the diagram's {"name":"bash","command":…}.
    # A JSON value that decodes to a string/list must not crash the batch (the lecturer's trial_10 parser bug).
    if not isinstance(obj, dict):
        return None, f"the tool call decoded to a {type(obj).__name__}, not an object"
    if obj.get("name") not in (None, "bash"):
        return None, f"unknown tool {obj.get('name')!r} (only 'bash' exists)"
    args = obj.get("arguments", obj.get("parameters"))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"command": args}
    cmd = args.get("command") if isinstance(args, dict) else obj.get("command")
    if not isinstance(cmd, str) or not cmd.strip():
        return None, "the tool call has no non-empty 'command' string"
    return cmd, None


def system_prompt(protocol: str = "tool_call") -> str:
    if protocol == "mswea":
        return prompts_mswea.SYSTEM
    return SYSTEM_PROMPT if protocol == "tool_call" else BACKTICKS_SYSTEM_PROMPT


def instance_prompt(problem_statement: str, protocol: str = "tool_call") -> str:
    if protocol == "mswea":
        return prompts_mswea.instance(problem_statement)
    return INSTANCE_TEMPLATE.format(problem_statement=problem_statement)


def tools_for(protocol: str = "tool_call") -> list[dict]:
    return [BASH_TOOL] if protocol == "tool_call" else []


def parse_reply(text: str, protocol: str = "tool_call") -> Parsed:
    """tool_call: exactly one well-formed bash call. backticks: the first fenced bash block (reference semantics).
    mswea: exactly one ```bash block (mini-swe-agent v1). Anything else is a format error fed back to the policy
    (mask 0, not graded)."""
    if protocol == "backticks":
        m = _BASH_RE.search(text)
        cmd = m.group(1).strip() if m else ""
        return Parsed(cmd, None, 1) if cmd else Parsed(None, NO_COMMAND, 0)
    if protocol == "mswea":
        found = _MSWEA_RE.findall(text)
        if len(found) == 1:  # v1 runs even an empty block (it just returns empty output)
            return Parsed(found[0].strip(), None, 1)
        return Parsed(None, prompts_mswea.FORMAT_ERROR.format(n=len(found)), len(found))
    if protocol != "tool_call":
        raise ValueError(f"unknown protocol {protocol!r}")
    bodies = _CALL_RE.findall(text)
    if not bodies:
        why = "an unclosed <tool_call> (the reply was cut off?)" if _OPEN_RE.search(text) else None
        return Parsed(None, FORMAT_ERROR_BAD.format(why=why) if why else FORMAT_ERROR_NONE, 0)
    if len(bodies) > 1:
        return Parsed(None, FORMAT_ERROR_MANY.format(n=len(bodies)), len(bodies))
    try:
        obj = json.loads(bodies[0])
    except json.JSONDecodeError as e:
        return Parsed(None, FORMAT_ERROR_BAD.format(why=f"the tool call is not valid JSON ({e.msg})"), 1)
    cmd, why = _command_from(obj)
    return Parsed(cmd, None if cmd else FORMAT_ERROR_BAD.format(why=why), 1)


def is_submit(command: str) -> bool:
    return bool(_SUBMIT_RE.match(command.strip())) or command.strip().startswith(f"echo {SUBMIT}")


def format_observation(returncode: int | None, output: str, *, max_chars: int = 10_000, timed_out: bool = False) -> str:
    """<returncode>/<output>; long output keeps head and tail with a warning (mini-swe-agent's 10k rule)."""
    rc = "timeout" if timed_out else ("" if returncode is None else str(returncode))
    if len(output) > max_chars:
        half = max_chars // 2
        elided = len(output) - 2 * half
        body = (f"<warning>The output was {len(output)} characters; {elided} characters in the middle were elided. "
                "Use more selective commands (grep, head, sed -n).</warning>\n"
                f"<output_head>\n{output[:half]}\n</output_head>\n<output_tail>\n{output[-half:]}\n</output_tail>")
    else:
        body = f"<output>\n{output}</output>" if output.endswith("\n") or not output else f"<output>\n{output}\n</output>"
    return f"<returncode>{rc}</returncode>\n{body}"


def format_observation_mswea(returncode: int | None, output: str, *, command: str = "", timed_out: bool = False) -> str:
    """mini-swe-agent v1's observation text, byte-for-byte as in the mini-coder corpus: `<output>\\n` + raw output +
    `</output>`; 10k characters or more keep a 5k head and a 5k tail behind a <warning>."""
    if timed_out:
        return prompts_mswea.TIMEOUT.format(command=command, output=output)
    head = f"<returncode>{'' if returncode is None else returncode}</returncode>\n"
    if len(output) < 10_000:
        return head + f"<output>\n{output}</output>"
    return (head + prompts_mswea.LONG_OUTPUT_WARNING + f"<output_head>\n{output[:5000]}\n</output_head>\n"
            f"<elided_chars>\n{len(output) - 10_000} characters elided\n</elided_chars>\n"
            f"<output_tail>\n{output[-5000:]}\n</output_tail>")


def action_text(command: str, thought: str = "", protocol: str = "tool_call") -> str:
    """How a well-formed reply looks in either protocol (used by the scripted oracle and tests)."""
    if protocol == "mswea":
        return f"THOUGHT: {thought.strip()}\n\n```bash\n{command}\n```"
    if protocol == "backticks":
        return (thought.strip() + "\n" if thought else "") + f"```bash\n{command}\n```"
    return tool_call_text(command, thought)


def tool_call_text(command: str, thought: str = "") -> str:
    """How a well-formed reply looks (used by the scripted oracle and tests)."""
    call = json.dumps({"name": "bash", "arguments": {"command": command}})
    return (thought.strip() + "\n" if thought else "") + f"<tool_call>\n{call}\n</tool_call>"
