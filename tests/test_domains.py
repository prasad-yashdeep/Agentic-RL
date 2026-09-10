import json

import pytest

from tarl.data.domains import domain_of, domain_table, task_domains


def test_taxonomy_shape():
    td = task_domains()
    assert len(td) == 89
    assert len(set(td.values())) == 8
    assert domain_of("nope") == "unknown"
    assert domain_table()["n_tasks"].sum() == 89


@pytest.mark.real_data
def test_taxonomy_matches_lock(a0_job):
    lock = json.loads((a0_job / "lock.json").read_text())
    names = {t["task"]["name"] for t in lock["trials"]}
    assert names == set(task_domains())
