"""Figure/table helpers for the anchor notebook. Every chart is an SVG with text and axes painted
`currentColor` so the blog page reads in light and dark themes; every chart has a table twin."""

from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

from tarl.common.paths import FIGURES

PALETTE = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "gray": "#898781",
    "good": "#0ca30c",
    "serious": "#ec835a",
    "critical": "#d03b3b",
    "wash": "#2a78d6",
}
SERIES = [PALETTE["blue"], PALETTE["orange"], PALETTE["aqua"]]
_SENTINEL = "#123457"  # replaced by currentColor in the saved SVG
INK = _SENTINEL  # use for any explicit "text/axis" colour argument in matplotlib calls


def setup_matplotlib(font_size: int = 11):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "svg.hashsalt": "tarl",
            "figure.figsize": (7.2, 3.6),
            "figure.dpi": 100,
            "font.size": font_size,
            "font.family": "sans-serif",
            "axes.edgecolor": _SENTINEL,
            "axes.labelcolor": _SENTINEL,
            "xtick.color": _SENTINEL,
            "ytick.color": _SENTINEL,
            "text.color": _SENTINEL,
            "axes.titlecolor": _SENTINEL,
            "legend.edgecolor": _SENTINEL,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": PALETTE["gray"],
            "grid.alpha": 0.25,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "lines.linewidth": 2.0,
            "lines.markersize": 7,
            "legend.frameon": False,
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "savefig.facecolor": "none",
            "savefig.transparent": True,
        }
    )
    return plt


def save_svg(fig, name: str, *, fig_dir: Path = FIGURES, max_kb: int = 300) -> Path:
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"{name}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight", transparent=True, metadata={"Date": None})
    svg = path.read_text()
    svg = svg.replace(_SENTINEL, "currentColor")
    svg = re.sub(r"<metadata>.*?</metadata>", "", svg, flags=re.S)
    path.write_text(svg)
    import matplotlib.pyplot as plt

    plt.close(fig)
    kb = path.stat().st_size / 1024
    assert kb <= max_kb, f"{path} is {kb:.0f} KB (> {max_kb} KB): simplify the figure"
    return path


def df_to_markdown(df, *, floatfmt: str = ".3g", max_rows: int | None = None) -> str:
    import pandas as pd

    d = df if max_rows is None else df.head(max_rows)

    def fmt(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        if isinstance(v, float):
            return format(v, floatfmt)
        if isinstance(v, (list, tuple)):
            return ", ".join(str(x) for x in v)
        return str(v).replace("|", "\\|").replace("\n", " ")

    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in row.tolist()) + " |")
    return "\n".join(lines) + "\n"


def save_table(df, name: str, *, fig_dir: Path = FIGURES, floatfmt: str = ".3g", max_rows: int | None = None) -> tuple[Path, Path]:
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    md = fig_dir / f"{name}.md"
    csv = fig_dir / f"{name}.csv"
    md.write_text(df_to_markdown(df, floatfmt=floatfmt, max_rows=max_rows))
    df.to_csv(csv, index=False)
    return md, csv


def token_grid_svg(mask: Sequence[int], roles: Sequence[str] | None = None, *, per_row: int = 48, cell: int = 9, gap: int = 2, title: str = "") -> str:
    """Compact token strip: gray = context (user/system), blue = trained (assistant)."""
    n = len(mask)
    rows = (n + per_row - 1) // per_row
    w = per_row * (cell + gap) + gap
    h = rows * (cell + gap) + gap + (18 if title else 0)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="token mask grid">']
    if title:
        parts.append(f'<text x="{gap}" y="12" font-size="11" fill="currentColor" font-family="system-ui,sans-serif">{title}</text>')
    y0 = 18 if title else 0
    i = 0
    n = len(mask)
    while i < n:  # one rect per run of equal mask values within a row (keeps the SVG small)
        r, c = divmod(i, per_row)
        j = i
        while j + 1 < n and (j + 1) // per_row == r and mask[j + 1] == mask[i]:
            j += 1
        x = gap + c * (cell + gap)
        y = y0 + gap + r * (cell + gap)
        wdt = (j - i + 1) * (cell + gap) - gap
        color = PALETTE["blue"] if mask[i] else PALETTE["gray"]
        op = "1" if mask[i] else "0.45"
        parts.append(f'<rect x="{x}" y="{y}" width="{wdt}" height="{cell}" rx="1.5" fill="{color}" fill-opacity="{op}"/>')
        i = j + 1
    parts.append("</svg>")
    return "".join(parts)


def spans_html(text_spans: Sequence[tuple[str, str]]) -> str:
    """[(role, text)] -> HTML with <mark class="trained"> around assistant spans."""
    import html

    out = []
    for role, text in text_spans:
        esc = html.escape(text)
        if role == "assistant":
            out.append(f'<mark class="trained">{esc}</mark>')
        else:
            out.append(f'<span class="context">{esc}</span>')
    return '<pre class="tokens">' + "".join(out) + "</pre>"


@contextmanager
def Timer(name: str, sink: dict | None = None):
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        if sink is not None:
            sink[name] = round(dt, 3)
        print(f"[{name}] {dt:.1f}s")


HEADLINE_SCHEMA: dict[str, list[str]] = {
    "": ["generated_at", "git_commit", "host", "versions", "flags", "a0", "f1", "domains", "pair", "judge_pilot", "coldstart", "g4", "loss_mask", "embeddings", "stats", "reference_points", "planned_runs", "infra", "runtime_s", "figures"],
    "a0": ["pass_at_1", "pass_at_1_ci95", "n_clean", "n_pass", "n_error", "errors_by_type", "tasks_solved", "solved_tasks", "median_steps", "max_steps", "median_prompt_tokens", "n_summarized", "wall_clock_h", "phase_median_s", "status"],
    "judge_pilot": ["n", "mean", "stdev", "min", "max", "rubric", "model", "adversarial_scores", "judge_input"],
    "g4": ["steps", "grad_norms", "adapter_hashes", "probe_changed", "step3_fix_git", "all_groups_all_fail"],
    "coldstart": ["useful_fraction", "sim"],
    "embeddings": ["model", "n_tasks", "pca_var_explained", "near_duplicate_pairs", "leakage_hits", "cluster_mean_judge"],
}


def check_headline(d: dict) -> list[str]:
    missing = []
    for section, keys in HEADLINE_SCHEMA.items():
        node = d if section == "" else d.get(section)
        if node is None:
            missing.append(section)
            continue
        for k in keys:
            if k not in node:
                missing.append(f"{section}.{k}" if section else k)
    return missing


def _round(o, nd: int = 4):
    if isinstance(o, float):
        return round(o, nd)
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round(v, nd) for v in o]
    return o


def write_headline(d: dict, path: Path = FIGURES / "headline.json") -> Path:
    missing = check_headline(d)
    if missing:
        raise KeyError(f"headline.json missing keys: {missing}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_round(d), indent=2, sort_keys=True, default=str))
    return path


def verify_figures(expected: Sequence[str], *, fig_dir: Path = FIGURES, max_kb: int = 300, total_mb: float = 3.0) -> "object":
    """DataFrame(file, exists, kb) and assertions on size; raises on a missing file."""
    import xml.etree.ElementTree as ET

    import pandas as pd

    rows = []
    total = 0
    for name in expected:
        p = Path(fig_dir) / name
        ok = p.exists()
        kb = p.stat().st_size / 1024 if ok else 0
        total += kb
        if ok and p.suffix == ".svg":
            ET.parse(p)  # well-formed
            assert kb <= max_kb, f"{p} is {kb:.0f} KB"
        rows.append({"file": name, "exists": ok, "kb": round(kb, 1)})
    df = pd.DataFrame(rows)
    missing = df.loc[~df["exists"], "file"].tolist()
    assert not missing, f"missing figures: {missing}"
    assert total / 1024 <= total_mb, f"figures total {total/1024:.1f} MB > {total_mb} MB"
    return df
