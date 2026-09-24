#!/usr/bin/env python3
"""Assemble docs/blog/terminal-agent-rl.html from template.html + figures/headline.json + figures/*.svg|csv.

    .venv/bin/python docs/blog/build_page.py        # then publish the HTML as an Artifact

Every number on the page comes from headline.json (the notebook's output); figures are inlined SVGs;
tables are rendered from the notebook's CSV twins. Re-run after `make notebook`.
"""

from __future__ import annotations

import csv
import html
import json
import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
OUT = HERE / "terminal-agent-rl.html"


def _fig(name: str, caption: str = "", *, cls: str = "") -> str:
    p = FIG / f"{name}.svg"
    if not p.exists():
        return f'<figure class="missing"><figcaption>missing figure {name}</figcaption></figure>'
    svg = p.read_text()
    svg = re.sub(r"<\?xml[^>]*\?>", "", svg)
    svg = re.sub(r"<!DOCTYPE[^>]*>", "", svg)
    svg = svg.replace("<svg ", f'<svg class="fig-svg" role="img" aria-label="{html.escape(caption or name)}" ', 1)
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return f'<figure class="fig {cls}" id="fig-{name}">{svg}{cap}</figure>'


def _fmt_cell(v: str) -> str:
    v = v.strip()
    if v == "" or v.lower() == "nan":
        return ""
    try:
        f = float(v)
        if v.isdigit() or re.fullmatch(r"-?\d+", v):
            return f"{int(v):,}"
        if abs(f) >= 100:
            return f"{f:,.0f}"
        if abs(f) >= 10:
            return f"{f:.1f}"
        return f"{f:.3g}" if abs(f) < 1 else f"{f:.2f}"
    except ValueError:
        return html.escape(v)


def _table(name: str, *, cols: list[str] | None = None, rename: dict | None = None, max_rows: int | None = None, cls: str = "", raw_cols: tuple = ()) -> str:
    p = FIG / f"{name}.csv"
    if not p.exists():
        return f"<p><em>missing table {name}</em></p>"
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return "<p><em>empty table</em></p>"
    cols = cols or list(rows[0].keys())
    rename = rename or {}
    if max_rows:
        rows = rows[:max_rows]
    head = "".join(f"<th>{html.escape(rename.get(c, c))}</th>" for c in cols)
    body = []
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append(f"<td>{html.escape(v) if c in raw_cols else _fmt_cell(v)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="tablewrap"><table class="data {cls}"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _pct(x, nd=1):
    return f"{float(x) * 100:.{nd}f} %"


def _num(x, nd=0):
    x = float(x)
    return f"{x:,.{nd}f}" if nd else f"{int(round(x)):,}"


def _r(x, nd=2):
    return f"{float(x):.{nd}f}"


def main() -> Path:
    H = json.loads((FIG / "headline.json").read_text())
    env = Environment(loader=FileSystemLoader(str(HERE)), autoescape=False, trim_blocks=True, lstrip_blocks=True)
    env.filters.update({"pct": _pct, "num": _num, "r": _r})
    rubric = (HERE.parent.parent / "configs" / "judge" / "rubric-v1.md").read_text()
    env.globals.update({"fig": _fig, "table": _table, "H": H, "rubric": html.escape(rubric), "H_json": json.dumps({k: H[k] for k in ("a0", "judge_pilot", "g4", "planned_runs", "pre_registered_success", "generated_at", "git_commit")})})
    text = env.get_template("template.html").render()
    OUT.write_text(text)
    kb = OUT.stat().st_size / 1024
    words = len(re.sub(r"<[^>]+>", " ", text).split())
    print(f"wrote {OUT} ({kb:.0f} KB, ~{words:,} words incl. tables)")
    return OUT


if __name__ == "__main__":
    main()
