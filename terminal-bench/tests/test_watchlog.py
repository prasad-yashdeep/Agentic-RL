import pytest

from tarl.data.watchlog import events, parse


def test_parse_synthetic(tmp_path):
    p = tmp_path / "watch.log"
    p.write_text(
        "07-21 05:36:42 serve=ok tmp_free=14G jobs=8\n"
        "synced 05:40:01\n"
        "07-21 05:37:42 serve=DOWN tmp_free=512M jobs=8\n"
        "07-21 05:38:42 ALARM serve dead -> relaunched (thinking=1)\n"
    )
    df = parse(p)
    assert len(df) == 2 and df["serve_ok"].tolist() == [True, False]
    assert df["tmp_free_gb"].iloc[1] == pytest.approx(0.5)
    assert str(df["ts_utc"].iloc[0]) == "2026-07-21 12:36:42+00:00"
    ev = events(p)
    assert (ev["kind"] == "alarm").sum() == 1


@pytest.mark.real_data
def test_parse_real(repo):
    df = parse(repo / "watch.log")
    assert len(df) > 9_000 and df["jobs"].max() >= 8
