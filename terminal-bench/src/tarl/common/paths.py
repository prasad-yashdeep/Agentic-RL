"""Single source of truth for storage locations (tiers: docs/01-infra.md, CLAUDE.md).

Durability (NFS home) > capacity (node05 /local, wipeable) > hot (/tmp, janitor).
REPO is derived from this file's location (src/tarl/common/paths.py -> repo root) so a
worktree or a different clone location works; override with TARL_REPO.
"""

import os
from pathlib import Path

HOME = Path.home()
_env_repo = os.environ.get("TARL_REPO")
REPO = Path(_env_repo).resolve() if _env_repo else Path(__file__).resolve().parents[3]

# durability tier (NFS): anything irreplaceable
JOBS = REPO / "jobs"                       # Harbor eval jobs (A0 baseline lives here)
JOBS_G4 = REPO / "jobs-g4"                 # GRPO micro-run rollout jobs
RUNS = REPO / "runs"                       # scripts/08_grpo_run.py outputs
EXPERIMENTS = REPO / "experiments"
DERIVED = EXPERIMENTS / "derived"          # small derived tables (committed)
CONFIGS = REPO / "configs"
DOCS = REPO / "docs"
FIGURES = DOCS / "blog" / "figures"        # notebook outputs consumed by the blog page
CHECKPOINTS = REPO / "checkpoints"
G4_LOG = REPO / "g4.log"
WATCH_LOG = REPO / "watch.log"
PILOT_JSONL = EXPERIMENTS / "judge-pilot.jsonl"
RUNS_YAML = EXPERIMENTS / "runs.yaml"
TB2_DOMAINS = CONFIGS / "tasks" / "tb2_domains.yaml"
JUDGE_YAML = CONFIGS / "judge" / "judge.yaml"
SERVE_THINKING_TEMPLATE = CONFIGS / "serve" / "tool_chat_template_gemma4_thinking.jinja"
A0_JOB = JOBS / "2026-07-21__05-38-50"     # the interim A0 baseline job
HF_CACHE = Path(os.environ.get("HF_HOME", str(HOME / "hf-cache")))
EMBED_CACHE = REPO / ".cache" / "embeddings"
SECRETS_ENV = HOME / ".config" / "tarl" / "env"

# node-local hot tier on node08 (16 GB /tmp, evicted by scripts/_janitor.sh)
SIF_CACHE = Path(os.environ.get("TARL_SIF_CACHE", "/tmp/sif-cache"))
# Endless Terminals (arXiv:2601.16443) — the RL training corpus (TB2 is eval-only):
#   ET_ROOT   raw HF snapshot (obiwan96/endless-terminals) on shared NFS
#   ET_HARBOR converted Harbor task dirs (task.toml gains docker_image=<sif>) — small, on NFS
#   ET_SIF    per-task .sif images built from container.def — node-local (node08 /var/tmp)
ET_ROOT = Path(os.environ.get("TARL_ET_ROOT", "/nfs-share/datasets/endless-terminals"))
ET_HARBOR = Path(os.environ.get("TARL_ET_HARBOR", str(HOME / "datasets" / "endless-terminals-harbor")))
ET_SIF = Path(os.environ.get("TARL_ET_SIF", "/var/tmp/tarl/et-sif"))
ET_SPLITS = EXPERIMENTS / "et-splits.json"
TB2_LOCAL = Path(os.environ.get("TARL_TB_PATH", "/tmp/tb2"))

# capacity tier: node05 /local — big, wipeable on reprovision; reached via
# ssh/rsync from node08 (scripts/07_sync_node05.sh), never mounted
NODE05_HOST = "node05"
NODE05_ROOT = "/local/yashp/tarl"

BASE_MODEL = "google/gemma-4-E4B-it"
SERVE_MODEL_FP8 = "prithivMLmods/gemma-4-E4B-it-FP8"
