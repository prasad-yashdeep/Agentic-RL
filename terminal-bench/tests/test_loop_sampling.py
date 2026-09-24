"""Per-step task sampling in the run loop (pool > tasks_per_step) and the source contract that carries it."""

from __future__ import annotations

from tarl.training.curriculum import Curriculum
from tarl.training.loop import sample_step_tasks
from tarl.training.rollouts import ReplayRolloutSource
from tarl.training.run_config import RunConfig


def test_sample_step_tasks_none_when_pool_is_the_batch():
    cfg = RunConfig(tasks=["a", "b"], tasks_per_step=None)
    assert sample_step_tasks(cfg, Curriculum(cfg.tasks, seed=0)) is None
    cfg = RunConfig(tasks=["a", "b"], tasks_per_step=2)
    assert sample_step_tasks(cfg, Curriculum(cfg.tasks, seed=0)) is None


def test_sample_step_tasks_draws_distinct_pool_tasks():
    pool = [f"task_{i:02d}" for i in range(12)]
    cfg = RunConfig(tasks=pool, tasks_per_step=4, seed=7)
    cur = Curriculum(pool, seed=cfg.seed)
    picks = [sample_step_tasks(cfg, cur) for _ in range(5)]
    for p in picks:
        assert len(p) == 4 and len(set(p)) == 4 and set(p) <= set(pool)
    assert len({tuple(p) for p in picks}) > 1  # not the same batch every step
    cfg2 = RunConfig(tasks=pool, tasks_per_step=4, seed=7)
    assert sample_step_tasks(cfg2, Curriculum(pool, seed=7)) == picks[0]  # seeded => reproducible


def test_replay_source_accepts_step_tasks(tmp_path):
    src = ReplayRolloutSource(tmp_path, {1: "missing"})
    try:
        src.collect(1, tasks=["x"])
    except KeyError:
        raise AssertionError("step 1 is mapped; a missing job dir must surface as a path error, not KeyError")
    except (FileNotFoundError, NotADirectoryError, OSError):
        pass
