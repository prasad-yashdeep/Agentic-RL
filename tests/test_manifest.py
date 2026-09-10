import pytest

from tarl.eval.manifest import manifest_from_job, read_manifest, write_manifest


@pytest.mark.real_data
def test_manifest_roundtrip(a0_job, tmp_path):
    m = manifest_from_job(a0_job)
    assert m.harness["harbor_version"] == "0.20.0" and m.harness["agent"] == "terminus-2"
    assert m.model["parser"] == "json" and m.model["temperature"] == 1.0
    assert m.tasks["n_tasks"] == 89 and m.tasks["n_planned_trials"] == 445
    p = write_manifest(m, tmp_path / "m.json")
    assert read_manifest(p).run_id == a0_job.name
    assert m.git_commit == "unknown" or len(m.git_commit) == 40
