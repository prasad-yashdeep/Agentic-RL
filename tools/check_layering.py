#!/usr/bin/env python3
"""Import boundaries for the Agentic-RL monorepo: the library packages stay independent.

- terminal-bench/src/tarl must not import sarl
- swe-bench/src/sarl must not import tarl
Scripts, notebooks and tests may cross-import (e.g. swe-bench/scripts/pool_guard.py reuses tarl.embed).
Exit 1 and list offenders on violation.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ARL = Path(__file__).resolve().parents[1]
RULES = {ARL / "terminal-bench" / "src" / "tarl": "sarl", ARL / "swe-bench" / "src" / "sarl": "tarl"}


def imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(), filename=str(path))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(node.lineno, a.name) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.append((node.lineno, node.module))
    return out


def main() -> int:
    bad = []
    for pkg, forbidden in RULES.items():
        if not pkg.exists():
            continue
        for f in sorted(pkg.rglob("*.py")):
            for line, mod in imports(f):
                if mod == forbidden or mod.startswith(forbidden + "."):
                    bad.append(f"{f.relative_to(ARL)}:{line}: imports {mod}")
    for b in bad:
        print(b)
    print("LAYERING_OK" if not bad else f"LAYERING_FAIL ({len(bad)})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
