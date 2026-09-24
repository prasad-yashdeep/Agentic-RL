import pytest

from sarl.paths import CONFIGS
from sarl.tasks import VERIFIED_REPOS, contamination_report, get_task, image_for, load_split, read_pool, select_pool

pytestmark = pytest.mark.swe_data


def test_splits_and_disjointness():
    dev, ver = load_split("dev"), load_split("verified")
    assert len(dev) == 225 and len(ver) == 500
    assert {t.repo for t in ver} == set(VERIFIED_REPOS)
    rep = contamination_report(dev, ver)
    assert rep["clean"], rep["hits"]


def test_reference_task_fields():
    t = get_task("astropy__astropy-12907", splits=("verified_princeton",))
    assert t.base_commit.startswith("d16bfe05") and len(t.fail_to_pass) == 2 and len(t.pass_to_pass) == 13
    assert t.n_patch_files == 1 and len(t.patch) < 1800 and t.contaminates_eval
    assert t.image == image_for(t.instance_id)


def test_pool_is_clean_and_small():
    pool = read_pool(CONFIGS / "pool_dev.txt")
    assert len(pool) == 8 and all(not t.contaminates_eval and t.n_patch_files == 1 for t in pool)
    assert contamination_report(pool, load_split("verified"))["clean"]
    assert [t.instance_id for t in select_pool(load_split("dev"), 3)] == [t.instance_id for t in select_pool(load_split("dev"), 3)]
    g = pool[0].grader_row()
    assert set(g) >= {"instance_id", "image", "eval_script", "log_parser", "eval_type", "FAIL_TO_PASS", "PASS_TO_PASS"}
