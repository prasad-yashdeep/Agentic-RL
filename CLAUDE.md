# CLAUDE.md — Agentic-RL on the BCM11 cluster

Read this first. **Agentic-RL** is a monorepo of RL-for-agents projects that share one cluster and one venv:

| subproject | what | package | its own notes |
|---|---|---|---|
| `terminal-bench/` | GRPO on terminal tasks (Terminus-2 via Harbor, Endless Terminals train pool, TB 2.1 eval), LLM-judge-shaped reward | `tarl` | `terminal-bench/CLAUDE.md` |
| `swe-bench/` | RL for SWE tasks with a basic agentic harness loop (Vizuara SWE-RL reference → real SWE-bench containers, test reward, vLLM + LoRA GRPO) | `sarl` | `swe-bench/CLAUDE.md` |

This file records the cluster (facts that cannot be inferred from the code) and the monorepo rules.
Last full cluster survey: 2026-09-09; status updated 2026-09-24.

## Monorepo rules (2026-09-24 cut-over)

- Location: `/home/yashp/Agentic-RL` (git root, 38 pre-move commits kept; tag `pre-agentic-rl` = old layout).
  The old checkout `/home/yashp/terminal-agent-rl` moved here with **no compatibility symlink**; every stored
  path (venv shebangs, editable `.pth`, Jupyter kernels, 576 Harbor/run JSON records) was rewritten.
  `make doctor` (= `tools/doctor.sh`) checks all of it and gates `make exp`.
- **One shared venv** at `/home/yashp/Agentic-RL/.venv` (13 GB, py3.12, torch 2.11 cu128, vLLM 0.25.1,
  transformers 5.14.1). Installs: `uv pip install --python .venv/bin/python --no-deps -e <subproject>` only —
  never re-resolve (a stale `transformers==5.5.4` pin in terminal-bench would break the CUDA lane).
  Jupyter kernels `tarl` and `sarl` both point at it; the cluster `jupyter` module is broken.
- Scripts locate themselves (`REPO="${TARL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"`);
  never write an absolute checkout path into code (tests enforce it).
- Subprojects are independent packages: `sarl` never imports `tarl` and vice versa (`tools/check_layering.py`).
  Shared GRPO/masking/stats code gets extracted into a core package later, not copied ad hoc.
- Tests: `make test NODE=node08` (each subproject's suite runs separately, over ssh; never on the headnode).
- Secrets: `XAI_API_KEY` in `~/.config/tarl/env` (never print or commit). GitHub remote: private
  `prasad-yashdeep/Agentic-RL` (history contains cluster IPs, lab account names and TB2 task text: keep private).

## Where you are

- This shell runs on **`bcm11-headnode`** (10.141.255.254 internal,
  192.168.1.57 external): the NVIDIA Base Command Manager 11 head node —
  Slurm controller (slurmctld, slurmdbd, MySQL), NFS server for `/home`,
  `/cm/shared`, `/nfs-share`, JupyterHub. Ubuntu 24.04, 16 threads, 28 GiB
  RAM, **no GPU, no nvidia-smi**. Time zone America/Los_Angeles (PDT).
- User `yashp` (groups `yashp nfs-share`). **No sudo anywhere.** Other
  accounts: bhumikad, riteshn, yashn, prasanna, deeplm (project account),
  cmsupport.
- **Standing rule: never run compute on the headnode** — no training,
  inference, simulators, pytest suites, container builds, bulk parsing.
  `ssh nodeXX '...'` or `sbatch`; `/home` is NFS so paths are identical on
  every node. Non-interactive ssh lands in `$HOME` — use absolute paths.
- GPU compute for both subprojects runs on **node08** (tmux: terminal-bench `serve`, `watch`,
  `exp`; swe-bench `swe-serve`, `swe-train`); capacity tier is **node05:/local**.

## Node map

| host | IP 10.141.x | role | threads (Slurm) | RAM (Slurm) | GPU | IB | local disks | state 2026-09-09 |
|---|---|---|---|---|---|---|---|---|
| bcm11-headnode | .255.254 | head/login, Slurm ctld, NFS server | 16 | 28 GiB | none | no | 715 GB root = the NFS disk; /var 180 GB; /tmp 20 GB | up |
| dlmdb01 | .0.6 | services host: Prometheus :9090, Grafana :3001, slurm_exporter :9341, DeepLM scheduler harness + feeders. In `defq` but **never place jobs here** (`--exclude=dlmdb01`) | 20 (12) | 31 GiB (28 GB) | none (Intel iGPU) | no | 804 GB root | up, ssh OK |
| node01 | .0.1 | AMD ROCm box: Radeon 780M iGPU (gfx1103, ~6 GB shared RAM), ROCm 7.1; not in GRES | 16 | 23 GB | AMD 780M | no | ? | **DOWN since 2026-08-10** |
| node02 | .0.2 | AMD ROCm box, same class as node01 per dlm-core AMD runbook | 16 | 23 GB | AMD 780M (unverified) | no | ? | **DOWN since 2026-08-10** |
| node03 | .0.3 | CPU compute; AMD Radeon PRO W6600 (Navi 23, gfx1032, 8 GB) invisible to Slurm; ROCm 7.1; nvidia-smi binary present but no NVIDIA driver; unprivileged userns enabled | 20 (16) | 31 GiB (28 GB) | AMD W6600 | no HCA | 349 GB root; /tmp 16 GB; /var 100 GB | up, ssh OK, idle |
| node04 | .0.4 | small NVIDIA node (laptop-class chassis): RTX 4060 Laptop 8 GB, 110 W, Ada sm_89. **dlm-baseline .deb v0.1.17 installed here** = baseline coordinator | 22 (12) | 15 GiB (13.8 GB) | 1× RTX 4060L 8 GB | none | 803 GB root; /tmp 16 GB; /var 100 GB; NIC `enp45s0` 1 GbE | **DOWN since 2026-09-08 22:50 PDT** |
| node05 | .0.5 | NVIDIA node: RTX A2000 12 GB, **70 W cap**, Ampere sm_86. `/local` 2.43 TiB fast scratch (**ephemeral**, wiped on reprovision) = our capacity tier + `/local/mit-supercloud` dataset. IB `mlx5_0`/`ibp78s0` 100 Gb, no IPoIB IP. **PCIe host link stuck at 6.7 GB/s (21 % of Gen4 x16)** — BIOS/slot issue | 16 | 30.8 GiB (28.4 GB) | 1× A2000 12 GB | yes, lid 1 | 825 GB root; /tmp 16 GB; /var 100 GB; /local 2.43 TiB | **DOWN since 2026-09-08 22:50 PDT** |
| node07 | .0.7 | NVIDIA node: RTX A4000 16 GB, 140 W, sm_86, PCIe Gen4 (25 GB/s). CARLA/DreamerV3 (`~/cardreamer`) lives here. **Root disk only 51 GB** | 12 | 15.1 GiB (13.9 GB) | 1× A4000 16 GB | yes, lid 3 | 51 GB root; /tmp 16 GB; /var 40 GB | **DOWN since 2026-09-08 22:45 PDT** |
| node08 | .0.8 | **the workhorse** (Threadripper, one NUMA node): 2× RTX A4000 16 GB, PCIe Gen4 (26.8 GB/s per GPU), GPU↔GPU 26.3 GB/s over PCIe (A4000 has no NVLink). Driver 570.133.20 = CUDA 12.8, **no nvcc**. Unprivileged userns enabled (apptainer works) | 24 | 62 GiB (57 GB) | 2× A4000 16 GB | yes, lid 2 | ≈40 GB free on root (Jul 2026); /tmp 16 GB; /var/tmp 39 GB partition | **DOWN since 2026-09-08 ≈22:53 PDT** |

"threads (Slurm)": hardware thread count vs. what `slurm.conf` declares — Slurm
under-declares CPUs and RAM on several nodes.

**Slurm**: 24.11.7, cluster `slurm`, one partition `defq` =
`dlmdb01,node[01-05,07-08]`, `select/cons_tres` `CR_CORE`, backfill,
multifactor priority, no time limits, `DefMemPerNode=UNLIMITED`, GRES
`gpu:nvidia` (node04/05/07 ×1, node08 ×2, `AutoDetect=NVML`). Binaries in
`/cm/local/apps/slurm/current/bin` (on PATH via `module load slurm` in
`~/.bashrc`; **not** on PATH for non-interactive ssh — export it plus
`SLURM_CONF=/cm/shared/apps/slurm/etc/slurm/slurm.conf`). `sacct -S` rejects
`Z`-suffixed timestamps. `sbatch` CLI flags silently override `#SBATCH` lines.

## Network fabric

- **Ethernet** `internalnet` 10.141.0.0/16: 1 GbE on every node (`eno1`;
  node04 `enp45s0`), MTU 1500 (no jumbo frames). Measured TCP 936–943 Mbit/s,
  UDP 949 Mbit/s with ≈0 loss. Quirk seen 2026-07-27: traffic **into node04**
  from node05/node07 fell to ≈225 Mbit/s with thousands of retransmits while
  node08→node04 was fine.
- **InfiniBand** 100 Gb (`mlx5_0`) on **node05, node07, node08 only**; all
  `PORT_ACTIVE`, one subnet manager at lid 4. RDMA write BW ≈ 11.5 GB/s
  (11,507 MiB/s) per pair, latency 1.0–1.15 µs. node04, node01–03, dlmdb01
  and the headnode have no IB. IPoIB carries no IPs, so NCCL needs
  `NCCL_SOCKET_IFNAME=eno1` for bootstrap and `NCCL_NET=IB` for data.
  GPUDirect RDMA is off (`GDR 0`): ≥16 MB messages bounce through host RAM.
- **NCCL multi-node busbw** (2026-08-19, IB): node07↔node08 7.92 GB/s;
  node05↔node07 4.09; node05↔node08 4.10 (node05 is PCIe-bound); 3-node
  N-way 4.85 GB/s. Single-node `all_reduce_perf` algbw: node05 129 GB/s,
  node07/08 195 GB/s.
- **node08 intra-node**: PCIe P2P between the two A4000s is broken → every
  2-GPU collective needs `NCCL_P2P_DISABLE=1` (and vLLM
  `--disable-custom-all-reduce` for TP=2). Otherwise: hang, then illegal
  memory access.

## Storage

- **One 715 GB NFS disk** on the headnode serves `/home` (all users),
  `/cm/shared` and `/nfs-share`; chronically ≈96 % full (31 GB free on
  2026-09-09). `df -h /nfs-share` before any multi-GB write — filling it takes
  the whole cluster down. Never build private venvs or download models onto
  NFS when a shared copy exists.
- Shared stack (world-readable): `/nfs-share/deeplm/venvs/{solver,llm-serving,diffusion}`
  (≈30 GB, built on system python3.12), HF cache
  `/nfs-share/deeplm/hf-cache/hub` (gpt2, SD-1.5, Mistral-24B-AWQ). Datasets:
  `/nfs-share/testcode` (tiny-imagenet), `/nfs-share/datasets/kaggle`,
  `/nfs-share/datasets/mit-supercloud`.
- Node-local: `/tmp` = 16 GB xfs on every compute node (small); `/var`
  100 GB (node07: 40 GB); node05 `/local` 2.43 TiB (ephemeral). This
  project's image caches live on node08 `/var/tmp/tarl` (39 GB partition)
  with symlinks from `/tmp/sif-cache` and `/tmp/apptainer-cache`;
  `HF_HOME=$HOME/hf-cache` on NFS holds the ≈13 GB FP8 checkpoint.
- Cluster software: environment modules under `/cm/shared/modulefiles`
  (`cuda12.8/toolkit`, `openmpi4/gcc/4.1.8`, `iperf/3.17.1`, `ucx`, `hwloc`,
  `miniconda3` → `/cm/shared/apps/miniconda3/envs/{pytorch311,pytorch311-cpu}`,
  `pytorch/3.11-gpu`, `jupyter`). NVIDIA nvssvt microbenchmarks at
  `/cm/shared/apps/nvssvt/1.3.2/microbenchmarks/amd64/` (`nccl-tests/build/all_reduce_perf`,
  `nvbandwidth`, `iperf3`, `cublasMatmulBench`). System python3 (3.12) has
  **no ensurepip** → use `uv` (`~/.local/bin/uv`). **No docker/podman
  anywhere**; the container path is user-space apptainer 1.5.2 at
  `~/apptainer-env` (details: `docs/01-infra.md`, `docs/02-architecture.md` §2).

## Monitoring (shared, user-mode, runs as yashp from NFS)

- Prometheus `http://10.141.0.6:9090` (dlmdb01), ≈15-day retention. Scrapes
  node_exporter `:9100`, dcgm-exporter `:9400` (uses the root nv-hostengine
  at 127.0.0.1:5555 — always pass `-r`), process-exporter `:9256` on
  node04/05/07/08 at 5 s; slurm_exporter `dlmdb01:9341`. The three
  `deeplm-api` targets on `bcm11-headnode:5001` are the legacy monolith and
  show down — ignore.
- Grafana `http://10.141.0.6:3001`, dashboard `dlm-live`; community DCGM /
  node / Slurm dashboards in `~/exporters/dashboards`.
- Control: `bash ~/exporters/exporters_ctl.sh start|stop|status`, logs in
  `~/exporters/logs`. **node08's exporters were stopped 2026-08-21 and never
  restarted** — run `start` once node08 is back.
- Never write into the live Prometheus TSDB (`~/prometheus/data` on dlmdb01).

## Cluster health — DeepLM baseline report (session 20260819_0616)

Coordinator node04, nodes node05/07/08, run by user yashn, 265 min. Verdict:
all three GPU nodes "Needs Attention".

| node | GPU | cooling | compute | PCIe host link | other |
|---|---|---|---|---|---|
| node05 | A2000 12 GB | FAIL: 83 °C at idle, 84 °C stable (limit 83), 12.4 % clock drop | FAIL (FP32 3.9 of 8.0 TFLOPS rated) | **FAIL**: 6.7/6.8 GB/s, gen `DEFAULT`, 21 % of 32 GB/s (floor 8) → reseat GPU / BIOS PCIe gen | mem BW 257 GB/s OK, capacity OK, 70 W limit; IB port was `PORT_DOWN` on 2026-07-27, active by 08-19 |
| node07 | A4000 | FAIL: 93 °C idle, 90.5 °C stable, 3.1 % drop | FAIL | PASS (Gen4 25.0/23.1 GB/s, 80 %, "Degraded") | |
| node08 | 2× A4000 | FAIL both: 93 °C idle (93.0 / 92.0 stable) | GPU0 FAIL, GPU1 PASS | PASS (Gen4 26.8/27.1 GB/s per GPU, 85 %) | GPU↔GPU 26.3 GB/s PCIe; single NUMA domain |

Ethernet PASS, IB fabric PASS, multi-GPU FAIL (node05 pairs < 5 GB/s, N-way
4.85 < 5). Fleet-wide, steady state at TDP is 85–90 °C on every card (fitted
thermal model in dlm-core `intel/models/thermal_params.json`: node05 85 °C at
its 70 W cap, node07/08 ≈90 °C at 135 W) — expect throttling under sustained
load. The report's "Scaling Projection" table (ring efficiency > 100 %) is an
artifact of the tool's formula, not a measurement.

## dlm-baseline (the tool that generated the report)

- Source: dlm-core branch **`baseline-stage1`**, checked out as a git
  worktree at `~/dlm-baseline-wt` (commit 1b2bab1, package v0.1.10 + fixes).
  Package tree: `~/dlm-baseline-wt/packaging/dlm-baseline/usr/lib/dlm-baseline/`
  (`cli.py`, `shell.py`, `nodes.py`, `preflight.py`, `slurm_runner.py`,
  `stages/gpu_baseline_stage{1,2,3a,3b,3c}.py`, `stages/gpu_catalog.py`,
  `report/report.py`). Never merge `baseline-stage1` into `yash`.
- Releases: **private** GitHub repo `DeepLM/dlm-core`, tags up to v0.1.32 on
  2026-09-09. v0.1.17 is the .deb installed (with sudo, by someone else) on
  node04. Unauthenticated download returns 404 and `gh` is not installed here;
  read the release page from a browser or an authenticated laptop.
- Flow: `dlm-baseline shell --nodelist node05,node07,node08` opens a bash
  subshell with `DLM_SESSION_DIR=~/.dlm-baseline/sessions/<YYYYMMDD_HHMM>`;
  inside it `dlm-baseline run gpu|intranode|ethernet|infiniband|nccl|network|all`;
  on exit `report.pdf` lands in the session dir
  (`dlm-baseline report --session-dir DIR` regenerates). Stage 1 (GPU
  compute/memory/thermal, 300 s per GPU, six tests, per-SKU thresholds from
  `gpu_catalog.py`) = one `sbatch` per node using
  `/cm/shared/apps/miniconda3/envs/pytorch311/bin/python3`. Stage 2 (PCIe via
  `nvbandwidth`, NVLink, NUMA) and 3A/3B/3C (iperf3 mesh + MTU; `ibv_devinfo` +
  `ib_write_bw/lat` + single-node `all_reduce_perf`; `mpirun` multi-node
  `all_reduce_perf` pairwise + N-way) run over ssh from the coordinator, so the
  package is needed on the coordinator only. Thresholds: TCP ≥ 800 Mbit/s,
  UDP loss ≤ 0.1 %, RDMA ≥ 10,000 MiB/s and ≤ 5 µs, single-node NCCL
  ≥ 50 GB/s, multi-node busbw ≥ 5.0 GB/s, PCIe gen4 ≥ 20 GB/s / gen3 ≥ 10 /
  default ≥ 8, thermal limit 83 °C.
- Our own dev session (2026-07-27, nodes node04/05/07/08):
  `~/.dlm-baseline/sessions/20260727_1739/` (raw JSON per stage +
  `report.pdf`). The 2026-08-19 session lives in yashn's home (unreadable).
- No-sudo way to run it (untested from the headnode; coordinator work is
  light ssh orchestration, the GPU jobs still go through `sbatch`): create
  `~/.local/bin/dlm-baseline` = copy of
  `packaging/dlm-baseline/usr/bin/dlm-baseline` with the `sys.path.insert`
  pointing at `/home/yashp/dlm-baseline-wt/packaging/dlm-baseline/usr/lib/dlm-baseline`,
  `chmod +x`, then `dlm-baseline preflight --nodelist ...` and
  `dlm-baseline shell --nodelist ...`. Headnode has python3-yaml,
  python3-reportlab, iperf3, ib_write_bw/lat, ibv_devinfo, mpirun. node03
  lacks reportlab, iperf3 and mpirun, so it cannot coordinate as-is.

## Status snapshot (updated 2026-09-24 00:30 PDT)

- **All GPU nodes are down.** node04/05/07 exporters received SIGTERM and shut
  down cleanly 22:45–22:50 PDT on 2026-09-08 (clean OS shutdown, not a
  crash); Slurm marked node04/05/07/08 "Not responding" 22:53–22:58. No Slurm
  job has run since 2026-08-20, so it was not job-induced. node01/02 down
  since 2026-08-10. Reachable: headnode, dlmdb01, node03. Queue empty.
- Nothing in this project (serve/eval/train) can run until node08 returns.
  First checks: `sinfo -N -l`, `ssh node08 hostname`, then
  `bash ~/exporters/exporters_ctl.sh start`.
- **2026-09-10 19:00 PDT:** node08 and node04 were powered on and answer ping, but **no TCP port is open**
  (22, 9100, 9400 all dropped) and Slurm still says `NOT_RESPONDING` → most likely stuck in the BCM
  node-installer (provisioning never finished). Needs the BMC console or cmsupport; nothing user-side can
  fix it.
- **2026-09-24:** node08 is back (up since 09-22 ≈18:30 PDT) but was **reprovisioned**: every node-local
  cache is gone (`/var/tmp/tarl`, `/tmp/tb2`, `/tmp/sif-cache`, apptainer caches); NFS state is intact.
  node03 and **dlmdb01 are down** (so Prometheus/Grafana are down too); node01/02/04/05/07 down.
  node08 is the only compute node: run tests there (`make test NODE=node08`). Health:
  `ssh node08 'bash /home/yashp/Agentic-RL/terminal-bench/scripts/_node08_health.sh'`.

## Other projects on this cluster (under `/home/yashp`)

- `dlm-core` — DeepLM: LP/MILP "intelligence layer" over Slurm (hardware-aware
  placement). Branches `yash` (daily), `main` (merged), `baseline-stage1`,
  `Bhumika`. Its `AGENTS.md` and
  `.claude/skills/{slurm-dlm,dlm-demo,cluster-portforward,latent-sched-training}/SKILL.md`
  are the authoritative cluster runbooks; A/B results in
  `docs/lp-solver/intelligence_layer_2026-08-13.md`. Harness and feeders run
  on dlmdb01; demo jobs carry `--comment dlm=1`. `services/latent-sched` =
  JEPA world-model scheduler with its own py3.10 venv (node08 only).
- `hetero-minigpt` — nanoGPT across mixed NVIDIA+AMD GPUs;
  `hmg/topology/topology.yaml` is a checked-in cluster map from June 2026
  (its "node05 IB is single-node" note is outdated).
- `cardreamer` — DreamerV3/CARLA on node07 (checkpoints + data kept).
- `worldmodels/le-wm` — LeWM reproduction (reference for latent-sched; checkpoints kept).
- **Deleted 2026-09-24 to free NFS (user-approved, 93 % → 79 %)**: `CARLA_0.9.15` (re-download the 0.9.15
  tarball), `cardreamer/.venv`, `worldmodels/le-wm/.venv`, `~/.stable-wm/datasets/tworoom.h5`, dlm-core
  `venv-rocm` + `.venvs/llm-serving-rocm` (rebuild: `jobs/amd/setup_rocm_flow_env.sh`,
  `…/jobs/extended/setup_llm_rocm_venv.sh`), user pip packages in `~/.local/lib/python3.12` (uv tools
  `harbor`/`hf` untouched), and `~/.cache/huggingface` Qwen2.5-7B/1.5B-Instruct.
- Services owned by yashp: `~/prometheus`, `~/grafana`, `~/exporters`,
  `~/cassandra` (legacy monolith DB, port 9045 on dlmdb01).
- Other users' jobs: bhumikad and yashn run dlm-core workloads; the `deeplm`
  account runs on node05/07/08.

## Laptop access (from the dlm-core `cluster-portforward` skill)

`deeplm-node1` = llmdb VPN gateway (jump only, no NFS) → `deeplm-node2` =
bcm11-headnode 192.168.1.57 (ProxyJump) → `ssh nodeXX` inside, or
`deeplm-node05`. Key `~/.ssh/id_rsa_deeplm`. Tunnels:
`ssh -L 3001:dlmdb01:3001 deeplm-node2` (Grafana), `-L 9090:dlmdb01:9090`
(Prometheus), `-L 8000:node08:8000` (vLLM). Port 8080 is often blocked on the
laptop — use 8090.

## Corrections to `terminal-bench/docs/01-infra.md` (surveyed 2026-07-20, partly stale)

- node05's GPU is an RTX A2000 12 GB with a 70 W cap, not an unnamed 24 GB
  device.
- The cluster is not Ethernet-only: node05/07/08 share a working 100 Gb IB
  fabric (RDMA verified 2026-08-19). Only node04 and the CPU/AMD nodes are
  Ethernet-only.
- node04 has 22 hardware threads / 15 GiB (Slurm declares 12 / 13.8 GB).
- node08 exporters and Prometheus coverage for node08 stopped 2026-08-21.
