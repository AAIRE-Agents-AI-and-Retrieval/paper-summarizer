# Decision table

Rule (adjustable on the command line):

- quality floor: recall − teacher ≥ -0.050
- then the fewest GPUs at 100,000 papers/day and 70% utilisation (GPUs = load × GPU-s per paper / 86,400 / utilisation); anchors are never picked

| system | recall | − teacher | unsupported | GPU-s / paper | p95 latency s | GPUs at load | recall floor | Pareto (recall, GPU-s) | eligible |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| B K2-Horizon-7B 70k | 0.5719 | -0.0463 | – | 0.89 | – | 1.47 | yes | yes | **yes** |
| C Qwen3.5-9B 20k | 0.5741 | -0.0441 | 0.0333 | 1.82 | – | 3.02 | yes |  | **yes** |
| prefcc (C + DPO) | 0.6033 | -0.0150 | 0.0378 | 1.82 | – | 3.02 | yes | yes | **yes** |
| A Qwen3-4B 70k | 0.5268 | -0.0915 | – | 0.70 | – | 1.15 | no | yes | no |

**Cheapest system that meets the rule: B K2-Horizon-7B 70k** (1.47 GPUs at the stated load, recall 0.5719).

This table applies a rule to measurements; it is an input to the team's decision, not the decision. Check the intervals: a system whose recall interval overlaps the pick's is not measurably worse, and serving costs are only comparable when measured on the same hardware (see the cost-source column of results.md).
