# Upstream reference

| | |
|---|---|
| Repository | https://github.com/abgoswam/swe_in_prod_vizuara_01 ("Concepts in SWE RL Training") |
| Pinned commit | `8cddfb60c93cc6b2c7d9dc047939084858f30f76` (2026-07-31, "improve script wording") |
| Lecture | Vizuara, "Build an RL agent for Software Engineering Tasks from Scratch \| SWE-RL", https://www.youtube.com/watch?v=oNHzs7wi7TY (2026-08-02, 2:58:43), speaker Abhishek Goswami (Microsoft AI Frontiers Lab) |
| License | **none** (no LICENSE file). Files are fetched and checksum-verified into `_upstream/` (gitignored) and never committed. `sarl` is our own implementation of the lecture's diagram, not a port. Tests import the fetched module only to cross-check behaviour. |
| Files | `README.md`, `utils.py`, `swe_grpo_one_step.py`, `swe_grpo_one_step.ipynb` (44 cells, no outputs), `swe_agent_env_interface.png`; sha256 in `UPSTREAM.sha256` |

Fetch and verify:

```bash
bash reference/fetch.sh     # curl at the pinned SHA, sha256sum -c, then build _upstream/swe_grpo_one_step.cluster.ipynb
```

The cluster notebook differs from upstream only by the two `!pip` lines in cell 1 (installing into the shared venv would re-resolve the CUDA stack) and its kernelspec (`sarl`). `metadata.agentic_rl` records both edits.
