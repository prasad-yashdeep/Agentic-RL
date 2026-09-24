| arm | description | status | seeds | pass@1 mean ± 95 % CI | Δ vs A0 (pp) | McNemar p | tasks gained / lost | judge calls / cost | wall-clock |
|---|---|---|---|---|---|---|---|---|---|
| A0 | baseline, raw gemma-4-E4B-it (FP8, thinking-ON, t=1.0, 32k ctx), Terminus 2 on TB 2.1 | interim | 1 | 3.3 % [1.4, 7.6] (interim, 1 seed) | — | — | — | 0 | 10.3 h |
| B1 | GRPO, verifier-only reward (lambda_judge=0) - cold-start control | pending | 0 | pending | pending | pending | pending | 0 | pending |
| B2 | GRPO, verifier + judge-shaped gated reward (lambda_judge=0.5) - headline | pending | 0 | pending | pending | pending | pending | pending | pending |
| B3 | self-STaR warm-start on own passing rollouts, then B2 recipe | pending | 0 | pending | pending | pending | pending | pending | pending |
