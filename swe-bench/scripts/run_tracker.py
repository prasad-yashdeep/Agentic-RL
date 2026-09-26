#!/usr/bin/env python3
"""Turn a GRPO run directory into time series and inline-SVG charts for the explainer.

    python scripts/run_tracker.py runs/grpo-gemma-v1 --compare runs/grpo-minicoder-v2 \
        --out experiments/tracker-gemma-v1.json --html experiments/tracker-gemma-v1.html

Reads `metrics.jsonl` (one row per step) and computes, per step: policy success, loss, gradient norm,
objective gain, trainer-vs-server log-prob gap, importance-weight statistics, drift of the served policy
from the base, exit reasons, valid-action rate, turns, tokens and wall time.  It also computes the running
"late minus early" estimate: at step k, split steps 1..k in half and compare, task by task, how often the
second half solved each task against the first half (task-stratified, with one standard error).  The SVG
charts use the page's CSS classes (`svg .t`, `.tm`, `.grid`, ...) and CSS colour tokens so they follow the
page's light and dark themes; every point carries a <title> so hovering shows its value.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


# ----------------------------------------------------------------------------- data


def load_steps(run: Path) -> list[dict]:
    rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("kind", "step") == "step"]
    rows.sort(key=lambda r: r["step"])
    return rows


def load_rollouts(run: Path) -> list[dict]:
    p = run / "rollouts.jsonl"
    if not p.exists():
        return []
    out = []
    with p.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            out.append(
                {
                    "step": r["step"],
                    "task": r.get("task") or r.get("task_id"),
                    "kind": r.get("kind"),
                    "exit": r.get("exit_reason"),
                    "n_tokens": r.get("n_tokens"),
                    "n_assistant_tokens": r.get("n_assistant_tokens"),
                    "turns": len(r.get("turns") or []),
                    "wall_s": r.get("wall_s"),
                    "format_errors": r.get("n_format_errors"),
                    "resolved": bool((r.get("grade") or {}).get("resolved")),
                    "applied": (r.get("grade") or {}).get("applied"),
                    "empty_patch": (r.get("grade") or {}).get("empty_patch"),
                    "f2p": ((r.get("grade") or {}).get("f2p_passed"), (r.get("grade") or {}).get("f2p_total")),
                    "p2p": ((r.get("grade") or {}).get("p2p_passed"), (r.get("grade") or {}).get("p2p_total")),
                    "grade_s": (r.get("grade") or {}).get("seconds"),
                }
            )
    return out


def group_rows(steps: list[dict]) -> list[tuple[int, str, int, int]]:
    """(step, task, successes, trials) for every policy group with rewards."""
    rows = []
    for m in steps:
        for g in m.get("groups", []):
            if g.get("rewards"):
                rows.append((m["step"], g["task"], int(round(sum(g["rewards"]))), len(g["rewards"])))
    return rows


def stratified_delta(rows, early_pred, late_pred):
    """Task-stratified difference late - early over tasks present in both halves.

    Weight per task w = n_e n_l / (n_e + n_l) (inverse-variance weight for a difference of two proportions
    with a common p); returns (delta, se, n_tasks, per_task) or None when fewer than two tasks qualify.
    Same estimator as the v2 analysis, so the numbers are comparable across runs.
    """
    tasks = sorted({r[1] for r in rows})
    num = den = var = 0.0
    per_task = {}
    n_tasks = 0
    for t in tasks:
        es = sum(r[2] for r in rows if r[1] == t and early_pred(r[0]))
        en = sum(r[3] for r in rows if r[1] == t and early_pred(r[0]))
        ls = sum(r[2] for r in rows if r[1] == t and late_pred(r[0]))
        ln = sum(r[3] for r in rows if r[1] == t and late_pred(r[0]))
        if not (en and ln):
            continue
        n_tasks += 1
        w = en * ln / (en + ln)
        p = (es + ls) / (en + ln)
        d = ls / ln - es / en
        num += w * d
        den += w
        var += w * w * p * (1 - p) * (1 / en + 1 / ln)
        per_task[t] = {"early": [es, en], "late": [ls, ln], "delta": d}
    if n_tasks < 2 or den == 0:
        return None
    return {"delta": num / den, "se": math.sqrt(var) / den, "n_tasks": n_tasks, "per_task": per_task}


def running_delta(rows, last_step: int) -> list[dict]:
    """At each step k >= 4: first half (1..k//2) vs second half (k//2+1..k)."""
    out = []
    for k in range(4, last_step + 1):
        half = k // 2
        sub = [r for r in rows if r[0] <= k]
        d = stratified_delta(sub, lambda s, h=half: s <= h, lambda s, h=half: s > h)
        if d:
            out.append({"step": k, "delta": d["delta"], "se": d["se"], "n_tasks": d["n_tasks"]})
    return out


def rolling_mean(xs: list[float], w: int) -> list[float | None]:
    out = []
    for i in range(len(xs)):
        if i + 1 < w:
            out.append(None)
        else:
            out.append(sum(xs[i + 1 - w : i + 1]) / w)
    return out


def series(steps: list[dict], rollouts: list[dict]) -> dict:
    def g(m, *ks, default=None):
        cur = m
        for k in ks:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    s = {
        "step": [m["step"] for m in steps],
        "reward": [m["policy_reward_mean"] for m in steps],
        "resolved": [m["policy_resolved"] for m in steps],
        "n_rollouts": [m["n_rollouts"] for m in steps],
        "n_errors": [m.get("n_errors", 0) for m in steps],
        "degenerate": [m["n_degenerate_groups"] for m in steps],
        "n_groups": [len(m.get("groups", [])) for m in steps],
        "n_train": [m["n_train"] for m in steps],
        "loss": [g(m, "update", "loss") for m in steps],
        "grad_norm": [g(m, "update", "grad_norm") for m in steps],
        "tokens": [g(m, "update", "tokens") for m in steps],
        "objective_gain": [m.get("objective_gain") for m in steps],
        "policy_dlogp": [m.get("policy_dlogp_mean") for m in steps],
        "mismatch_mean": [g(m, "mismatch", "mean") for m in steps],
        "mismatch_p99": [g(m, "mismatch", "p99") for m in steps],
        "mismatch_max": [g(m, "mismatch", "max") for m in steps],
        "mismatch_frac_gt_0.5": [g(m, "mismatch", "frac_gt_0.5") for m in steps],
        "tis_mean": [g(m, "update", "tis", "mean_w") for m in steps],
        "tis_min": [g(m, "update", "tis", "min_w") for m in steps],
        "tis_capped": [g(m, "update", "tis", "frac_capped") for m in steps],
        "drift": [g(m, "sync", "served_dlogp_vs_base") for m in steps],
        "valid_action": [m.get("policy_valid_action_rate") for m in steps],
        "mean_turns": [m.get("policy_mean_turns") for m in steps],
        "nonempty_patches": [m.get("policy_nonempty_patches") for m in steps],
        "sec_total": [g(m, "seconds", "total") for m in steps],
        "sec_rollouts": [g(m, "seconds", "rollouts") for m in steps],
        "dropped_overlong": [m.get("dropped_overlong", 0) for m in steps],
        "truncated_overlong": [m.get("truncated_overlong", 0) for m in steps],
        "exit_reasons": [m.get("exit_reasons", {}) for m in steps],
    }
    s["sec_update"] = [
        (t - r) if (t is not None and r is not None) else None for t, r in zip(s["sec_total"], s["sec_rollouts"])
    ]
    s["reward_roll4"] = rolling_mean(s["reward"], 4)
    # per-step token / turn stats from rollouts (policy rollouts only)
    by_step: dict[int, list[dict]] = {}
    for r in rollouts:
        if r["kind"] == "policy":
            by_step.setdefault(r["step"], []).append(r)
    s["assistant_tokens_mean"] = []
    s["assistant_tokens_max"] = []
    s["episode_tokens_mean"] = []
    s["wall_mean"] = []
    s["format_errors_mean"] = []
    s["f2p_partial"] = []  # rollouts with some but not all F2P passing
    s["patch_outcome"] = []  # per step: what the submitted patches did
    for k in s["step"]:
        rs = by_step.get(k, [])
        oc = {"fixed": 0, "no fix, tests still green": 0, "broke tests or did not apply": 0, "empty patch": 0}
        for r in rs:
            pp, pt = r["p2p"]
            if r["resolved"]:
                oc["fixed"] += 1
            elif r["empty_patch"] or r["applied"] is False and not pt:
                oc["empty patch"] += 1
            elif pt and pp == pt:
                oc["no fix, tests still green"] += 1
            else:
                oc["broke tests or did not apply"] += 1
        s["patch_outcome"].append(oc)
        at = [r["n_assistant_tokens"] for r in rs if r["n_assistant_tokens"] is not None]
        et = [r["n_tokens"] for r in rs if r["n_tokens"] is not None]
        ws = [r["wall_s"] for r in rs if r["wall_s"] is not None]
        fe = [r["format_errors"] for r in rs if r["format_errors"] is not None]
        s["assistant_tokens_mean"].append(sum(at) / len(at) if at else None)
        s["assistant_tokens_max"].append(max(at) if at else None)
        s["episode_tokens_mean"].append(sum(et) / len(et) if et else None)
        s["wall_mean"].append(sum(ws) / len(ws) if ws else None)
        s["format_errors_mean"].append(sum(fe) / len(fe) if fe else None)
        s["f2p_partial"].append(
            sum(1 for r in rs if r["f2p"][1] and 0 < (r["f2p"][0] or 0) < r["f2p"][1])
        )
    return s


def exit_totals(steps: list[dict]) -> dict:
    tot: dict[str, int] = {}
    for m in steps:
        for k, v in (m.get("exit_reasons") or {}).items():
            tot[k] = tot.get(k, 0) + v
    return tot


def per_task_table(rows, last_step: int) -> list[dict]:
    half = last_step // 2
    d = stratified_delta(rows, lambda s: s <= half, lambda s: s > half)
    out = []
    tasks = sorted({r[1] for r in rows})
    for t in tasks:
        tot_s = sum(r[2] for r in rows if r[1] == t)
        tot_n = sum(r[3] for r in rows if r[1] == t)
        pt = (d or {}).get("per_task", {}).get(t)
        by_step: dict[str, list[int]] = {}
        for r in rows:
            if r[1] == t:
                cur = by_step.setdefault(str(r[0]), [0, 0])
                cur[0] += r[2]
                cur[1] += r[3]
        out.append(
            {
                "task": t,
                "total": [tot_s, tot_n],
                "early": pt["early"] if pt else None,
                "late": pt["late"] if pt else None,
                "delta": pt["delta"] if pt else None,
                "by_step": by_step,
            }
        )
    out.sort(key=lambda r: -(r["delta"] if r["delta"] is not None else -9))
    return out


# ----------------------------------------------------------------------------- svg

W, H = 560, 210
ML, MR, MT, MB = 46, 14, 14, 30


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if step >= raw:
            break
    t0 = math.floor(lo / step) * step
    ticks = []
    x = t0
    while x <= hi + 1e-9:
        if x >= lo - 1e-9:
            ticks.append(round(x, 10))
        x += step
    return ticks


def _fmt(v: float, kind: str) -> str:
    if kind == "pct":
        return f"{100 * v:.0f} %"
    if kind == "pp":
        return f"{100 * v:+.0f}"
    if kind == "int":
        return f"{v:.0f}"
    if kind == "s":
        return f"{v / 60:.0f} min" if abs(v) >= 120 else f"{v:.0f} s"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 1:
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return f"{v:.3f}".rstrip("0").rstrip(".") if v != 0 else "0"


def chart(
    title: str,
    xs: list[int],
    lines: list[dict],
    *,
    kind: str = "num",
    ylo: float | None = None,
    yhi: float | None = None,
    zero_line: bool = False,
    band: dict | None = None,
    x_max: int | None = None,
    ref: float | None = None,
    ref_label: str = "",
    aria: str = "",
) -> str:
    """One chart, one scale.  lines: [{name, ys, cls, dash?, marker?, fmt?}]; band: {ys, lo, hi, cls}."""
    vals = [v for ln in lines for v in ln["ys"] if v is not None]
    if band:
        vals += [v for v in band["lo"] if v is not None] + [v for v in band["hi"] if v is not None]
    if ref is not None:
        vals.append(ref)
    if not vals:
        vals = [0, 1]
    lo = min(vals) if ylo is None else ylo
    hi = max(vals) if yhi is None else yhi
    if zero_line:
        lo, hi = min(lo, 0), max(hi, 0)
    if hi == lo:
        hi = lo + (abs(lo) or 1)
    pad = (hi - lo) * 0.08
    if ylo is None:
        lo -= pad
    if yhi is None:
        hi += pad
    xmax = x_max or (max(xs) if xs else 1)
    xmin = 1
    pw, ph = W - ML - MR, H - MT - MB

    def X(x):
        return ML + (x - xmin) / max(1, xmax - xmin) * pw

    def Y(y):
        return MT + (hi - y) / (hi - lo) * ph

    out = [
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{_esc(aria or title)}">',
        f'<text x="{ML}" y="10" class="tb">{_esc(title)}</text>',
    ]
    for t in _ticks(lo, hi):
        y = Y(t)
        out.append(f'<line x1="{ML}" x2="{W - MR}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        out.append(f'<text x="{ML - 6}" y="{y + 3.5:.1f}" class="tm" text-anchor="end">{_esc(_fmt(t, kind))}</text>')
    if zero_line and lo < 0 < hi:
        out.append(f'<line x1="{ML}" x2="{W - MR}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" class="zero"/>')
    if ref is not None and lo <= ref <= hi:
        out.append(f'<line x1="{ML}" x2="{W - MR}" y1="{Y(ref):.1f}" y2="{Y(ref):.1f}" class="refline"/>')
        if ref_label:
            out.append(f'<text x="{W - MR}" y="{Y(ref) - 4:.1f}" class="tm" text-anchor="end">{_esc(ref_label)}</text>')
    # x axis: every step tick label when few, else every 5
    stride = 1 if xmax <= 12 else 5
    for x in range(xmin, xmax + 1):
        if x == 1 or x % stride == 0:
            out.append(f'<text x="{X(x):.1f}" y="{H - 10}" class="tm" text-anchor="middle">{x}</text>')
    out.append(f'<text x="{W - MR}" y="{H - 10}" class="tm" text-anchor="end">step</text>' if xmax > 12 else "")
    if band:
        pts_hi = [(X(x), Y(h)) for x, h in zip(band["ys"], band["hi"]) if h is not None]
        pts_lo = [(X(x), Y(l)) for x, l in zip(band["ys"], band["lo"]) if l is not None]
        if pts_hi and pts_lo:
            d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts_hi) + " L " + " L ".join(
                f"{x:.1f} {y:.1f}" for x, y in reversed(pts_lo)
            ) + " Z"
            out.append(f'<path d="{d}" class="{band.get("cls", "band")}"/>')
    for ln in lines:
        pts = [(x, y) for x, y in zip(xs, ln["ys"]) if y is not None]
        if not pts:
            continue
        dash = ' stroke-dasharray="5 4"' if ln.get("dash") else ""
        # a None in the middle (a step with no update) leaves a gap instead of a bridge
        segs: list[list[tuple[int, float]]] = [[]]
        for x, y in zip(xs, ln["ys"]):
            if y is None:
                if segs[-1]:
                    segs.append([])
            else:
                segs[-1].append((x, y))
        for seg in segs:
            if len(seg) > 1:
                d = "M " + " L ".join(f"{X(x):.1f} {Y(y):.1f}" for x, y in seg)
                out.append(f'<path d="{d}" class="{ln["cls"]}"{dash}/>')
        if ln.get("marker", True):
            f = ln.get("fmt", kind)
            for x, y in pts:
                out.append(
                    f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3.2" class="{ln["cls"]}-dot">'
                    f"<title>{_esc(ln['name'])} · step {x}: {_esc(_fmt(y, f))}</title></circle>"
                )
    # legend (only when more than one line)
    if len(lines) > 1:
        lx = ML + 4
        ly = MT + 8
        for ln in lines:
            dash = ' stroke-dasharray="5 4"' if ln.get("dash") else ""
            out.append(f'<line x1="{lx}" x2="{lx + 16}" y1="{ly}" y2="{ly}" class="{ln["cls"]}"{dash}/>')
            out.append(f'<text x="{lx + 20}" y="{ly + 3.5}" class="tm">{_esc(ln["name"])}</text>')
            lx += 20 + 6.2 * len(ln["name"]) + 22
    out.append("</svg>")
    return "\n".join(o for o in out if o)


def stacked(title: str, steps_x: list[int], dicts: list[dict], order: list[str], cls: dict, aria: str, x_max: int | None = None) -> str:
    """Stacked shares per step (one bar per step, segments in `order`)."""
    seen = {k for e in dicts for k, v in e.items() if v}
    keys = [k for k in order if k in seen] + sorted(k for k in seen if k not in order)
    xmax = x_max or max(steps_x)
    pw, ph = W - ML - MR, H - MT - MB
    bw = max(4, pw / max(1, xmax) * 0.7)
    out = [
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{_esc(aria)}">',
        f'<text x="{ML}" y="10" class="tb">{_esc(title)}</text>',
    ]
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        y = MT + (1 - t) * ph
        out.append(f'<line x1="{ML}" x2="{W - MR}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        out.append(f'<text x="{ML - 6}" y="{y + 3.5:.1f}" class="tm" text-anchor="end">{int(t * 100)} %</text>')
    for x, e in zip(steps_x, dicts):
        tot = sum(e.values()) or 1
        cx = ML + (x - 1) / max(1, xmax - 1) * pw
        acc = 0.0
        for k in keys:
            v = e.get(k, 0)
            if not v:
                continue
            h = v / tot * ph
            y = MT + ph - acc - h
            out.append(
                f'<rect x="{cx - bw / 2:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{max(0, h - 1):.1f}" class="{cls.get(k, "ex-other")}">'
                f"<title>{_esc(k)} · step {x}: {v}/{tot}</title></rect>"
            )
            acc += h
    stride = 1 if xmax <= 12 else 5
    for x in range(1, xmax + 1):
        if x == 1 or x % stride == 0:
            cx = ML + (x - 1) / max(1, xmax - 1) * pw
            out.append(f'<text x="{cx:.1f}" y="{H - 10}" class="tm" text-anchor="middle">{x}</text>')
    # legend below the title, wrapping when long
    lx, ly = ML + 4, MT + 2
    for k in keys:
        wk = 14 + 6.2 * len(k) + 16
        if lx + wk > W - MR:
            lx, ly = ML + 4, ly + 14
        out.append(f'<rect x="{lx}" y="{ly}" width="10" height="10" class="{cls.get(k, "ex-other")}"/>')
        out.append(f'<text x="{lx + 14}" y="{ly + 9}" class="tm">{_esc(k)}</text>')
        lx += wk
    out.append("</svg>")
    return "\n".join(out)


def heat(title: str, rows: list[dict], x_max: int, group_size: int) -> str:
    """Task × step grid; cell shade = successes / group size for that task at that step."""
    left = 150
    ch = 14
    top = 22
    h = top + ch * len(rows) + 26
    pw = W - left - MR
    cw = pw / max(1, x_max)
    out = [
        f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="{_esc(title)}">',
        f'<text x="{left}" y="10" class="tb">{_esc(title)}</text>',
    ]
    for i, r in enumerate(rows):
        y = top + i * ch
        name = r["task"].split("__")[-1]
        out.append(f'<text x="{left - 6}" y="{y + ch - 3.5:.1f}" class="tm" text-anchor="end">{_esc(name)}</text>')
        for step, (s_, n_) in r["by_step"].items():
            x = left + (int(step) - 1) * cw
            op = 0.12 + 0.88 * (s_ / n_ if n_ else 0)
            out.append(
                f'<rect x="{x + 0.5:.1f}" y="{y + 0.5:.1f}" width="{cw - 1:.1f}" height="{ch - 1:.1f}" class="heat" fill-opacity="{op:.2f}">'
                f"<title>{_esc(name)} · step {step}: {s_}/{n_} solved</title></rect>"
            )
            if s_:
                out.append(f'<text x="{x + cw / 2:.1f}" y="{y + ch - 3.5:.1f}" class="heat-n" text-anchor="middle">{s_}</text>')
    stride = 1 if x_max <= 12 else 5
    for x in range(1, x_max + 1):
        if x == 1 or x % stride == 0:
            out.append(f'<text x="{left + (x - 0.5) * cw:.1f}" y="{h - 8}" class="tm" text-anchor="middle">{x}</text>')
    out.append(f'<text x="{left - 6}" y="{h - 8}" class="tm" text-anchor="end">step →</text>')
    out.append("</svg>")
    return "\n".join(out)


def stacked_exits(steps_x: list[int], exits: list[dict], x_max: int | None = None) -> str:
    keys = ["submitted", "max_turns", "context", "wall_clock", "format_errors", "error"]
    seen = {k for e in exits for k in e}
    keys = [k for k in keys if k in seen] + sorted(k for k in seen if k not in keys)
    cls = {
        "submitted": "ex-submitted",
        "max_turns": "ex-turns",
        "context": "ex-context",
        "wall_clock": "ex-wall",
        "format_errors": "ex-format",
        "error": "ex-error",
    }
    xmax = x_max or max(steps_x)
    pw, ph = W - ML - MR, H - MT - MB
    bw = max(4, pw / max(1, xmax) * 0.7)
    out = [
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="How every episode ended, per step">',
        f'<text x="{ML}" y="10" class="tb">How each episode ended (share of the step&#x27;s rollouts)</text>',
    ]
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        y = MT + (1 - t) * ph
        out.append(f'<line x1="{ML}" x2="{W - MR}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        out.append(f'<text x="{ML - 6}" y="{y + 3.5:.1f}" class="tm" text-anchor="end">{int(t * 100)} %</text>')
    for x, e in zip(steps_x, exits):
        tot = sum(e.values()) or 1
        cx = ML + (x - 1) / max(1, xmax - 1) * pw
        acc = 0.0
        for k in keys:
            v = e.get(k, 0)
            if not v:
                continue
            h = v / tot * ph
            y = MT + ph - acc - h
            out.append(
                f'<rect x="{cx - bw / 2:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{max(0, h - 1):.1f}" class="{cls.get(k, "ex-other")}">'
                f"<title>{_esc(k)} · step {x}: {v}/{tot}</title></rect>"
            )
            acc += h
    stride = 1 if xmax <= 12 else 5
    for x in range(1, xmax + 1):
        if x == 1 or x % stride == 0:
            cx = ML + (x - 1) / max(1, xmax - 1) * pw
            out.append(f'<text x="{cx:.1f}" y="{H - 10}" class="tm" text-anchor="middle">{x}</text>')
    lx = ML + 4
    for k in keys:
        out.append(f'<rect x="{lx}" y="{MT + 2}" width="10" height="10" class="{cls.get(k, "ex-other")}"/>')
        out.append(f'<text x="{lx + 14}" y="{MT + 11}" class="tm">{_esc(k)}</text>')
        lx += 14 + 6.2 * len(k) + 16
    out.append("</svg>")
    return "\n".join(out)


CSS = """
svg .grid{stroke:var(--line);stroke-width:1}
svg .zero{stroke:var(--line-strong);stroke-width:1.2}
svg .refline{stroke:var(--muted);stroke-width:1;stroke-dasharray:2 3}
svg .l-main{stroke:var(--accent);stroke-width:2;fill:none} svg .l-main-dot{fill:var(--accent)}
svg .l-cmp{stroke:var(--muted);stroke-width:1.6;fill:none} svg .l-cmp-dot{fill:var(--muted)}
svg .l-teal{stroke:var(--teal);stroke-width:2;fill:none} svg .l-teal-dot{fill:var(--teal)}
svg .l-amber{stroke:var(--amber);stroke-width:2;fill:none} svg .l-amber-dot{fill:var(--amber)}
svg .l-green{stroke:var(--green);stroke-width:2;fill:none} svg .l-green-dot{fill:var(--green)}
svg .l-red{stroke:var(--red);stroke-width:2;fill:none} svg .l-red-dot{fill:var(--red)}
svg .band{fill:var(--accent);opacity:.14}
svg .ex-submitted{fill:var(--teal)} svg .ex-turns{fill:var(--amber)} svg .ex-context{fill:var(--accent)}
svg .ex-wall{fill:var(--red)} svg .ex-format{fill:var(--line-strong)} svg .ex-error{fill:var(--muted)} svg .ex-other{fill:var(--line)}
svg .po-fixed{fill:var(--green)} svg .po-green{fill:var(--teal)} svg .po-broke{fill:var(--amber)} svg .po-empty{fill:var(--line-strong)}
svg .heat{fill:var(--accent)} svg .heat-n{fill:#ffffff;font-family:"JetBrains Mono",monospace;font-size:9px;font-weight:600}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1rem 1.4rem;margin:1rem 0 1.2rem}
.charts figure.fig{margin:0}
.charts figure.fig svg{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:.35rem .4rem}
"""


# ----------------------------------------------------------------------------- assemble


def build(run: Path, compare: Path | None, planned_steps: int | None) -> dict:
    steps = load_steps(run)
    rollouts = load_rollouts(run)
    cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
    planned = planned_steps or cfg.get("steps") or (steps[-1]["step"] if steps else 0)
    rows = group_rows(steps)
    last = steps[-1]["step"] if steps else 0
    s = series(steps, rollouts)
    out = {
        "run": run.name,
        "model": cfg.get("model"),
        "steps_done": last,
        "steps_planned": planned,
        "series": s,
        "running_delta": running_delta(rows, last),
        "per_task": per_task_table(rows, last),
        "exit_totals": exit_totals(steps),
        "totals": {
            "resolved": sum(s["resolved"]),
            "rollouts": sum(s["n_rollouts"]),
            "errors": sum(s["n_errors"]),
            "degenerate": sum(s["degenerate"]),
            "groups": sum(s["n_groups"]),
            "trained": sum(s["n_train"]),
            "tokens": sum(t for t in s["tokens"] if t),
            "objective_up": sum(1 for v in s["objective_gain"] if v is not None and v > 0),
            "steps_with_update": sum(1 for v in s["objective_gain"] if v is not None),
            "seconds": sum(t for t in s["sec_total"] if t),
        },
    }
    half = last // 2
    d = stratified_delta(rows, lambda x: x <= half, lambda x: x > half) if last >= 4 else None
    out["early_late"] = (
        {"delta": d["delta"], "se": d["se"], "n_tasks": d["n_tasks"], "early_steps": [1, half], "late_steps": [half + 1, last]}
        if d
        else None
    )
    if compare and (compare / "metrics.jsonl").exists():
        cs = load_steps(compare)
        crows = group_rows(cs)
        c = series(cs, [])
        out["compare"] = {
            "run": compare.name,
            "model": (json.loads((compare / "config.json").read_text()).get("model") if (compare / "config.json").exists() else None),
            "steps_done": cs[-1]["step"],
            "series": {k: c[k] for k in ("step", "reward", "reward_roll4", "loss", "objective_gain", "drift", "mismatch_mean", "tis_mean", "sec_total", "degenerate", "n_groups")},
            "running_delta": running_delta(crows, cs[-1]["step"]),
        }
    return out


def render_charts(d: dict) -> dict[str, str]:
    s = d["series"]
    xs = s["step"]
    xmax = d["steps_planned"] or max(xs)
    c = d.get("compare")
    charts: dict[str, str] = {}

    reward_lines = [
        {"name": "per step", "ys": s["reward"], "cls": "l-main", "fmt": "pct"},
        {"name": "4-step mean", "ys": s["reward_roll4"], "cls": "l-teal", "dash": True, "marker": False, "fmt": "pct"},
    ]
    charts["reward"] = chart(
        "Success on training tasks per step (resolved / 32)", xs, reward_lines, kind="pct", ylo=0, yhi=max(0.35, max(s["reward"]) * 1.15), x_max=xmax,
        aria="Line chart of the share of rollouts that passed the tests, per training step",
    )
    charts["heat"] = heat(
        "Solved per task and step (of 8 rollouts; darker = more)",
        sorted(d["per_task"], key=lambda r: -r["total"][0] / max(1, r["total"][1])),
        xmax,
        8,
    )
    charts["patches"] = stacked(
        "What the submitted patches did (share of the step's rollouts)", xs, s["patch_outcome"],
        ["fixed", "no fix, tests still green", "broke tests or did not apply", "empty patch"],
        {"fixed": "po-fixed", "no fix, tests still green": "po-green", "broke tests or did not apply": "po-broke", "empty patch": "po-empty"},
        "Stacked bars per step: patches that fixed the bug, patches that changed nothing the tests notice, patches that broke tests or did not apply, and empty patches",
        x_max=xmax,
    )

    rd = d["running_delta"]
    rxs = [r["step"] for r in rd]
    if rd:
        # the comparison run has its own x positions, so align its values by step before zipping
        cmap = {r["step"]: r["delta"] for r in c["running_delta"]} if c else {}
        charts["delta"] = chart(
            "Running late-minus-early success, task by task (pp, ±1 SE)",
            rxs,
            [{"name": d["run"], "ys": [r["delta"] for r in rd], "cls": "l-main", "fmt": "pp"}]
            + ([{"name": c["run"], "ys": [cmap.get(x) for x in rxs], "cls": "l-cmp", "marker": False, "fmt": "pp"}] if c else []),
            kind="pp",
            zero_line=True,
            band={"ys": rxs, "lo": [r["delta"] - r["se"] for r in rd], "hi": [r["delta"] + r["se"] for r in rd], "cls": "band"},
            x_max=xmax,
            aria="Line chart of the task-stratified difference between the second and first half of the steps so far, with a one standard error band",
        )

    charts["loss"] = chart(
        "Loss per step (token mean of −A·w·log p)", xs,
        [{"name": "loss", "ys": s["loss"], "cls": "l-main"}],
        zero_line=True, x_max=xmax, aria="Line chart of the training loss per step",
    )
    charts["objective"] = chart(
        "Objective gain per step (mean A·Δlog p after the update)", xs,
        [{"name": "gain", "ys": s["objective_gain"], "cls": "l-main"}],
        zero_line=True, x_max=xmax, aria="Line chart of the objective gain per step; above zero means the update moved probability toward high-advantage tokens",
    )
    charts["grad"] = chart("Gradient norm before clipping (clip at 1.0)", xs, [{"name": "grad norm", "ys": s["grad_norm"], "cls": "l-main"}], ylo=0, x_max=xmax)
    charts["mismatch"] = chart(
        "Trainer vs server log-prob gap (nats per token)", xs,
        [{"name": "mean", "ys": s["mismatch_mean"], "cls": "l-main"}, {"name": "p99", "ys": s["mismatch_p99"], "cls": "l-amber", "dash": True}]
        + ([{"name": c["run"] + " mean", "ys": c["series"]["mismatch_mean"], "cls": "l-cmp", "marker": False}] if c else []),
        ylo=0, x_max=xmax, aria="Line chart of the mean and 99th percentile absolute log-probability gap between the trainer and the server",
    )
    charts["tis"] = chart(
        "Importance weight w = p_trainer / p_server, mean over tokens (cap 2)", xs,
        [{"name": "mean w", "ys": s["tis_mean"], "cls": "l-main"}],
        ref=1.0, ref_label="w = 1", ylo=0.995, yhi=1.005, x_max=xmax,
    )
    charts["drift"] = chart(
        "Drift of the served policy from the base (nats per token)", xs,
        [{"name": d["run"], "ys": s["drift"], "cls": "l-main"}] + ([{"name": c["run"], "ys": c["series"]["drift"], "cls": "l-cmp", "marker": False}] if c else []),
        ylo=0, x_max=xmax, aria="Line chart of the mean absolute log-probability difference between the served adapter and the base model on a fixed probe",
    )
    charts["degenerate"] = chart(
        "Groups with a gradient (mixed) out of 4 per step", xs,
        [{"name": "mixed groups", "ys": [g - dg for g, dg in zip(s["n_groups"], s["degenerate"])], "cls": "l-main", "fmt": "int"}],
        kind="int", ylo=0, yhi=max(s["n_groups"]) + 0.5, x_max=xmax,
    )
    charts["valid"] = chart(
        "Valid actions (share of replies with exactly one bash block)", xs,
        [{"name": "valid", "ys": s["valid_action"], "cls": "l-main", "fmt": "pct"}], kind="pct", ylo=0.8, yhi=1.0, x_max=xmax,
    )
    charts["turns"] = chart("Mean turns per episode", xs, [{"name": "turns", "ys": s["mean_turns"], "cls": "l-main"}], ylo=0, x_max=xmax)
    charts["tokens"] = chart(
        "Sampled tokens per episode (mask-1, mean and max)", xs,
        [{"name": "mean", "ys": s["assistant_tokens_mean"], "cls": "l-main", "fmt": "int"}, {"name": "max", "ys": s["assistant_tokens_max"], "cls": "l-amber", "dash": True, "fmt": "int"}],
        kind="int", ylo=0, x_max=xmax,
    )
    charts["seconds"] = chart(
        "Wall time per step", xs,
        [{"name": "whole step", "ys": s["sec_total"], "cls": "l-main", "fmt": "s"}, {"name": "rollouts", "ys": s["sec_rollouts"], "cls": "l-teal", "dash": True, "fmt": "s"}],
        kind="s", ylo=0, x_max=xmax,
    )
    charts["exits"] = stacked_exits(xs, s["exit_reasons"], x_max=xmax)
    return charts


def per_task_html(d: dict) -> str:
    el = d.get("early_late") or {}
    e0, e1 = el.get("early_steps", [1, d["steps_done"] // 2])
    l0, l1 = el.get("late_steps", [d["steps_done"] // 2 + 1, d["steps_done"]])
    out = [
        '<div class="tablewrap"><table class="data">',
        f"<tr><th>Task</th><th>Steps {e0}–{e1}</th><th>Steps {l0}–{l1}</th><th>Change</th><th>All steps</th></tr>",
    ]

    def cell(v):
        if not v:
            return "—"
        s_, n_ = v
        return f"{s_}/{n_} ({100 * s_ / n_:.0f} %)" if n_ else "—"

    for r in d["per_task"]:
        ch = f"{100 * r['delta']:+.0f} pp" if r["delta"] is not None else "—"
        out.append(f"<tr><td>{r['task'].split('__')[-1]}</td><td>{cell(r['early'])}</td><td>{cell(r['late'])}</td><td class=\"num\">{ch}</td><td>{cell(r['total'])}</td></tr>")
    if el:
        out.append(
            f"<tr><td><b>Task-stratified, {el['n_tasks']} tasks</b></td><td></td><td></td>"
            f"<td class=\"num\"><b>{100 * el['delta']:+.1f} ± {100 * el['se']:.1f} pp</b></td><td>{d['totals']['resolved']}/{d['totals']['rollouts']} ({100 * d['totals']['resolved'] / max(1, d['totals']['rollouts']):.1f} %)</td></tr>"
        )
    out.append("</table></div>")
    return "\n".join(out)


CAPTIONS = {
    "reward": "Share of the step's 32 rollouts that passed the tests. The dashed line is the mean of the last 4 steps.",
    "delta": "At step k the steps so far are split in half and each task's late success rate minus its early rate is averaged across tasks (weighted by rollout counts). The band is ±1 standard error. Grey: the same statistic for mini-coder v2, which read positive early and ended at −1.2 pp.",
    "heat": "Each cell is one task at one step: how many of its 8 rollouts passed. Blank cells: the task was not drawn that step.",
    "patches": "Every rollout ends with a patch. Green fixed the bug. Teal changed nothing the tests notice. Amber broke a pass-to-pass test or did not apply. Grey was empty.",
    "loss": "The number the optimizer minimises. It is not a progress meter: its sign follows the advantage mix of that step's sample, so read the objective gain instead.",
    "objective": "Mean advantage-weighted change in log-probability on the trained tokens, measured after the update. Above zero on every step means each update moved the policy the way the rewards asked.",
    "grad": "Norm of the LoRA gradient before clipping at 1.0. Flat and far below the clip: no step was clipped.",
    "mismatch": "Absolute difference between the server's and the trainer's log-probability of the same sampled tokens. The mean is the lecture's mismatch diagnostic; the dashed line is the 99th percentile.",
    "tis": "The truncated importance weight applied per token. A mean of 1.000 means the trainer scored the tokens almost exactly as the server that sampled them.",
    "drift": "How far the served adapter has moved from the base model on a fixed probe prompt, in nats per token. Grey: mini-coder v2, which drifted steadily; Gemma jumped after step 1 and then held.",
    "degenerate": "Groups are 8 rollouts of one task. Only mixed groups (some pass, some fail) carry a gradient. Four tasks per step, so the most possible is 4.",
    "valid": "Share of replies that held exactly one bash block. Below 100 % is the price of sampling at temperature 1.",
    "turns": "Commands per episode before submit.",
    "tokens": "Tokens the model itself wrote per episode (mask 1). The max shows the longest episode in the step; anything over 16,384 trained tokens is truncated.",
    "seconds": "Wall time per step. The gap between the two lines is grading plus the update (16 to 32 trajectories at about 66 s each on the trainer GPU).",
    "exits": "Why each episode stopped. Nearly all submit; a few hit the format-error limit or the context cap.",
}

ORDER = ["reward", "delta", "heat", "patches", "objective", "loss", "grad", "mismatch", "tis", "drift", "degenerate", "valid", "turns", "tokens", "seconds", "exits"]

START, END = "<!-- tracker:auto:start -->", "<!-- tracker:auto:end -->"


def auto_fragment(d: dict, charts: dict[str, str], stamp: str) -> str:
    figs = []
    for k in ORDER:
        if k in charts and charts[k]:
            figs.append(f'<figure class="fig" id="fig-tr-{k}">{charts[k]}<figcaption>{CAPTIONS.get(k, "")}</figcaption></figure>')
    return "\n".join(
        [
            START,
            f'<p class="tiny">Generated by <code>scripts/run_tracker.py</code> from <code>runs/{d["run"]}/metrics.jsonl</code> and <code>rollouts.jsonl</code>, {stamp}: step {d["steps_done"]} of {d["steps_planned"]}. Hover a point for its value.</p>',
            '<div class="charts">',
            *figs,
            "</div>",
            "<h3>Per task: first half against second half</h3>",
            per_task_html(d),
            END,
        ]
    )


def splice(page: Path, fragment: str) -> None:
    html = page.read_text()
    a, b = html.find(START), html.find(END)
    if a < 0 or b < 0:
        raise SystemExit(f"{page}: tracker markers not found")
    page.write_text(html[:a] + fragment + html[b + len(END) :])


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def text_summary(d: dict) -> str:
    s = d["series"]
    t = d["totals"]
    drift = [x for x in s["drift"] if x is not None]
    secs = sorted(x for x in s["sec_total"] if x is not None)
    lines = [f"{d['run']} ({d['model']}): step {d['steps_done']}/{d['steps_planned']}"]
    lines.append(
        f"  resolved {t['resolved']}/{t['rollouts']} = {100 * t['resolved'] / max(1, t['rollouts']):.1f} %; errors {t['errors']}; "
        f"mixed groups {t['groups'] - t['degenerate']}/{t['groups']}; trained trajectories {t['trained']}; tokens {t['tokens']:,}"
    )
    lines.append(
        f"  objective gain > 0 on {t['objective_up']}/{t['steps_with_update']} steps with an update; "
        f"drift {drift[0] if drift else float('nan'):.3f} -> {drift[-1] if drift else float('nan'):.3f}; "
        f"TIS mean {_mean(s['tis_mean']):.4f}; mismatch mean {_mean(s['mismatch_mean']):.4f}, p99 {_mean(s['mismatch_p99']):.3f}"
    )
    lines.append(
        f"  step time median {secs[len(secs) // 2] if secs else float('nan'):.0f} s; valid actions {100 * _mean(s['valid_action']):.1f} %; "
        f"mean turns {_mean(s['mean_turns']):.1f}; sampled tokens/episode {_mean(s['assistant_tokens_mean']):.0f}; exits {d['exit_totals']}"
    )
    el = d.get("early_late")
    if el:
        lines.append(
            f"  late - early (steps {el['late_steps'][0]}-{el['late_steps'][1]} vs {el['early_steps'][0]}-{el['early_steps'][1]}, "
            f"{el['n_tasks']} tasks): {100 * el['delta']:+.1f} ± {100 * el['se']:.1f} pp"
        )
    lines.append("  per-step reward: " + " ".join(f"{v:.2f}" for v in s["reward"]))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--compare")
    ap.add_argument("--steps", type=int, help="planned steps (default: config.json)")
    ap.add_argument("--out", help="write the series as JSON")
    ap.add_argument("--html", help="write the charts as an HTML fragment {name: svg}")
    ap.add_argument("--standalone", help="write a self-contained preview page")
    ap.add_argument("--splice", help="HTML page with tracker markers to refresh in place")
    ap.add_argument("--stamp", default="", help="text for the 'generated' line, e.g. a date")
    a = ap.parse_args(argv)
    d = build(Path(a.run), Path(a.compare) if a.compare else None, a.steps)
    print(text_summary(d))
    if a.out:
        Path(a.out).write_text(json.dumps(d, indent=1))
    charts = render_charts(d) if d["steps_done"] else {}
    if a.html:
        Path(a.html).write_text(json.dumps({"css": CSS, "charts": charts, "summary": text_summary(d), "fragment": auto_fragment(d, charts, a.stamp)}, indent=1))
    if a.splice:
        splice(Path(a.splice), auto_fragment(d, charts, a.stamp))
        print(f"spliced {len(charts)} charts into {a.splice}")
    if a.standalone:
        body = "\n".join(f'<figure class="fig">{svg}</figure>' for svg in charts.values())
        Path(a.standalone).write_text(
            "<!doctype html><meta charset=utf-8><title>tracker</title><style>"
            ":root{--paper:#f6f7f9;--surface:#fff;--ink:#1a1f26;--muted:#5b6472;--line:#d9dee6;--line-strong:#b9c2cf;--accent:#2a78d6;--teal:#0f8f8a;--amber:#b7791f;--green:#178a5f;--red:#c23b3b}"
            "body{background:var(--paper);color:var(--ink);font-family:system-ui;margin:1rem}"
            "svg .t,svg .tb{fill:var(--ink);font:13px system-ui} svg .tb{font-weight:600} svg .tm{fill:var(--muted);font:10.5px ui-monospace,monospace}"
            "figure.fig{margin:0} figure.fig svg{width:100%;height:auto;display:block;overflow:visible}" + CSS + "</style>"
            f'<div class="charts">{body}</div>'
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
