"""The fetched reference is exactly the pinned upstream, and the cluster notebook differs only as declared."""

import hashlib
import json

import pytest

from sarl.paths import REFERENCE, UPSTREAM

pytestmark = pytest.mark.upstream


def test_sha256_pins():
    for line in (REFERENCE / "UPSTREAM.sha256").read_text().splitlines():
        digest, name = line.split()
        assert hashlib.sha256((UPSTREAM / name).read_bytes()).hexdigest() == digest, name


def test_cluster_notebook_only_drops_pip_lines():
    up = json.loads((UPSTREAM / "swe_grpo_one_step.ipynb").read_text())
    cl = json.loads((UPSTREAM / "swe_grpo_one_step.cluster.ipynb").read_text())
    assert len(up["cells"]) == len(cl["cells"]) == 44
    for i, (a, b) in enumerate(zip(up["cells"], cl["cells"])):
        if i == 1:
            dropped = [ln for ln in a["source"] if ln not in b["source"]]
            assert len(dropped) == 2 and all(ln.lstrip().startswith("!pip") for ln in dropped)
            assert not any(ln.lstrip().startswith("!") for ln in b["source"])
        else:
            assert a["source"] == b["source"], i
    assert cl["metadata"]["kernelspec"]["name"] == "sarl" and "agentic_rl" in cl["metadata"]
