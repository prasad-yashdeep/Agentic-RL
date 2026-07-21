# Infrastructure Reality (surveyed 2026-07-20)

## GPU inventory — there are NO A100s on this cluster

Fleet scan from the BCM11 headnode, all defq nodes:

| Node | GPUs | VRAM | Notes |
|---|---|---|---|
| node01–03, dlmdb01 | none visible to NVIDIA driver | — | AMD cards / no driver |
| node04 | 1× RTX 4060 Laptop | 8 GB | |
| node05 | 1× "NVIDIA Graphics Device" | 24 GB | unnamed to driver; has `/local` TB-scale storage |
| node07 | 1× RTX A4000 | 16 GB | CARLA/DreamerV3 project lives here |
| **node08** | **2× RTX A4000** | **2× 16 GB** | **best node → our target**; 24 CPU cores, 62 GB RAM |

**Consequence:** the plan must be sized for **32 GB total VRAM**, not 160 GB.
Gemma E4B is ~8B raw params (~16 GB in bf16), so:

- **Serving (eval + rollouts):** vLLM with the bf16 model fits only with
  tensor-parallel across both A4000s (TP=2), leaving limited KV-cache room; an
  FP8/AWQ-quantized serve on ONE GPU is the better topology so the second GPU
  stays free for training.
- **Training:** full-parameter finetuning is impossible. **QLoRA** (4-bit base
  ≈ 5–6 GB + adapters + activations) on one 16 GB GPU is the realistic path.
  If even that is tight with long agentic contexts, fall back to **E2B**
  (~5B raw / 2B effective) for the pipeline-proving phase.
- Topology: **GPU0 = vLLM rollout/eval server (quantized base + hot-swapped
  LoRA adapter), GPU1 = QLoRA trainer.** Weight sync = push new adapter to
  vLLM between iterations (adapter reload is cheap; no full-model reload).
- **node08 NCCL gotcha (verified earlier on this cluster):** PCIe P2P between
  the two A4000s is broken — any 2-GPU collective (incl. vLLM `--tensor-parallel-size 2`)
  must run with `NCCL_P2P_DISABLE=1` and vLLM's custom all-reduce disabled
  (`--disable-custom-all-reduce`), else the first collective hangs then dies
  with an illegal-memory-access. One more reason to prefer the
  1-GPU-serve / 1-GPU-train split over TP=2.

## Container runtime — docker does not exist anywhere

Terminal-Bench executes every task inside its own docker container/compose
stack. Findings:

- `docker`/`podman`: **absent on all nodes and the headnode**.
- Rootless-docker prerequisites also absent on node08: no `newuidmap`/
  `newgidmap` binaries (uidmap package not installed), no `/etc/subuid` /
  `/etc/subgid` entries for `yashp`.
- User is not in a `docker` group (groups: `yashp nfs-share`); no sudo.

**Options, ranked:**

1. **Admin request (preferred, one-time):** ask cluster admins to
   `apt install uidmap`, add `yashp:100000:65536` to `/etc/subuid`+`/etc/subgid`
   on node08, and (ideally) install docker-ce with rootless extras. After that,
   rootless dockerd runs entirely in user space. GPU access inside task
   containers is NOT needed — tasks are CPU shell environments; GPUs serve the
   model *outside* docker.
2. **Split topology (no admin needed):** run the terminal-bench harness on any
   machine that has docker (e.g. the laptop / a workstation), and point the
   agent's model endpoint at vLLM on node08 via SSH port-forward
   (`deeplm-*` aliases → tunnel to node08:8000). The harness is CPU-only;
   only HTTP traffic crosses the VPN. Works today; rollout throughput for RL
   is then bounded by the harness machine's CPU/RAM (each task = 1+ container).
3. Apptainer/Singularity: not a substitute — terminal-bench drives the docker
   API / compose; no supported backend.

For the **baseline eval**, option 2 is sufficient. For **RL training** (needs
hundreds of concurrent rollouts colocated with the model server), option 1 on
node08 is strongly preferred — put the admin request in early.

## Other constraints

- Headnode is off-limits for compute (standing rule). Harness must never run
  its containers there.
- node08 has 40 GB free on `/`; big artifacts (checkpoints, traces, task
  images) should live on node05 `/local` (wipeable!) or NFS home with care.
- Cluster is Ethernet-only (no RDMA) — irrelevant for single-node training.
- Conda at `/cm/shared/apps/miniconda3`.
