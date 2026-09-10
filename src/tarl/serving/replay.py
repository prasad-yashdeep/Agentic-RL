"""`tarl replay`: print one recorded episode turn by turn (the CLI 'interaction platform')."""

from __future__ import annotations

from pathlib import Path

from tarl.data.cast import keystrokes, parse_cast, strip_ansi
from tarl.data.loader import Episode, step_commands, step_declares_complete, step_observation


def _clip_lines(text: str, max_lines: int) -> str:
    lines = text.rstrip("\n").splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    head = lines[: max_lines // 2]
    tail = lines[-(max_lines - len(head)) :]
    return "\n".join(head + [f"… [{len(lines) - max_lines} lines elided] …"] + tail)


def render_episode(
    ep: Episode,
    *,
    console=None,
    speed: float = 0.0,
    show_reasoning: bool = False,
    max_output_lines: int = 20,
    judge=None,
    use_cast: bool = False,
    no_color: bool = False,
) -> None:
    import time

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = console or Console(no_color=no_color, highlight=False, soft_wrap=True)
    t = ep.trial
    inst = ep.instruction() or "(instruction not found in trajectory)"
    hdr = Table.grid(padding=(0, 2))
    hdr.add_row("task", f"[bold]{t.task}[/]  ({t.trial_name})")
    hdr.add_row("model", f"{ep.model_name}  temperature={ep.temperature}  parser={ep.parser}")
    hdr.add_row("outcome", f"status={t.status}  reward={t.reward}  " + (f"tests {ep.ctrf.label}" if ep.ctrf else "no ctrf") + (f"  error={t.exception_type}" if t.exception_type else ""))
    ph = ep.phases
    hdr.add_row("phases (s)", "  ".join(f"{k.replace('_', ' ')}={'' if v is None else round(v)}" for k, v in ph.items()) + (f"  wall={round(ep.wall_s)}" if ep.wall_s else ""))
    tk = ep.tokens
    hdr.add_row("tokens", f"prompt={tk.get('total_prompt_tokens', tk.get('n_input_tokens'))}  completion={tk.get('total_completion_tokens', tk.get('n_output_tokens'))}  model calls={ep.n_agent_steps}  summarizations={ep.summarization_count}")
    console.print(Panel(hdr, title="episode", expand=False))
    console.print(Panel(Text(inst.strip()), title="task instruction", expand=False))

    cast_keys = keystrokes(parse_cast(ep.cast_path)) if (use_cast and ep.cast_path.exists()) else None
    ki = 0
    for i, s in enumerate(ep.agent_steps):
        msg = str(s.get("message") or "")
        analysis, plan = msg, ""
        if "\nPlan:" in msg:
            analysis, plan = msg.split("\nPlan:", 1)
        analysis = analysis.removeprefix("Analysis:").strip()
        body = Text()
        if show_reasoning and s.get("reasoning_content"):
            body.append("thinking: ", style="dim")
            body.append(_clip_lines(str(s["reasoning_content"]), 8) + "\n", style="dim")
        body.append("analysis: ", style="bold")
        body.append(analysis + "\n")
        if plan.strip():
            body.append("plan: ", style="bold")
            body.append(plan.strip() + "\n")
        cmds = step_commands(s)
        if cast_keys is not None and cmds:
            shown = cast_keys[ki : ki + len(cmds)]
            ki += len(cmds)
        else:
            shown = cmds
        for c in shown:
            body.append("$ " + c.rstrip("\n") + "\n", style="bold cyan")
        if step_declares_complete(s):
            body.append("<mark_task_complete>\n", style="bold magenta")
        obs = strip_ansi(step_observation(s))
        if obs:
            body.append(_clip_lines(obs, max_output_lines), style="green")
        m = s.get("metrics") or {}
        title = f"turn {i + 1}/{ep.n_agent_steps}  prompt={m.get('prompt_tokens')} tok  completion={m.get('completion_tokens')} tok"
        console.print(Panel(body, title=title, expand=False))
        if speed > 0:
            time.sleep(speed)

    foot = Table.grid(padding=(0, 2))
    foot.add_row("verifier", f"reward={t.reward}" + (f"  tests={ep.ctrf.label}  failed={ep.ctrf.failed_names}" if ep.ctrf else ""))
    if ep.ctrf and ep.ctrf.messages:
        for name, m in list(ep.ctrf.messages.items())[:3]:
            foot.add_row("", f"{name}: {_clip_lines(m, 4)}")
    if judge is not None and not t.passed:
        entry = judge.lookup_trial(t.trial_dir) if hasattr(judge, "lookup_trial") else None
        if entry is not None:
            foot.add_row("judge", f"score={entry.score:.2f}  ({entry.source}, {entry.confidence})  {entry.rationale}")
        elif hasattr(judge, "judge"):
            from tarl.judge.compressor import compress
            from tarl.judge.rubric import load_rubric

            try:
                v = judge.judge(load_rubric()[0], compress(ep.trajectory or {"steps": []}))
                foot.add_row("judge", f"score={v.score:.2f}  ({getattr(judge, 'name', '?')})  {v.rationale}")
            except Exception as e:
                foot.add_row("judge", f"no verdict: {e}")
        else:
            foot.add_row("judge", "no cached verdict for this trial")
    elif t.passed:
        foot.add_row("judge", "not consulted (verifier passed)")
    console.print(Panel(foot, title="outcome", expand=False))


def episode_json(ep: Episode) -> dict:
    t = ep.trial
    return {
        "task": t.task,
        "trial": t.trial_name,
        "trial_dir": str(t.trial_dir),
        "status": t.status,
        "reward": t.reward,
        "passed": t.passed,
        "exception_type": t.exception_type,
        "n_steps": ep.n_steps,
        "n_agent_steps": ep.n_agent_steps,
        "commands": ep.commands,
        "task_complete_declared": ep.task_complete_declared,
        "tokens": ep.tokens,
        "phases_s": ep.phases,
        "wall_s": ep.wall_s,
        "ctrf": (ep.ctrf.__dict__ if ep.ctrf else None),
        "model": ep.model_name,
        "temperature": ep.temperature,
        "instruction": ep.instruction(),
    }
