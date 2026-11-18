# Worked example: the previous project's students (not the new runs)

These tables and figures were made with `scripts/analysis/` from the artifacts of the **previous** project, to
show what the tools produce and to check them against numbers that are already known. They are about:

| system | student | training |
|---|---|---|
| A | Qwen3-4B-Instruct-2507 | SFT, 70,000 papers |
| B | K2-Horizon-7B (older revision) | SFT, 70,000 papers |
| C | Qwen3.5-9B | SFT, 20,000 papers |
| prefcc | Qwen3.5-9B | C + DPO on 1,233 fact-ranked pairs (two seeds) |

None of them is one of the five new 50,000-paper students. Regenerate with
`bash scripts/analysis/examples/run_old_students_example.sh <previous checkout>` (needs that checkout's
`runs/` and `state/`, which are not distributed).

## What is here

| path | content |
|---|---|
| `results.md`, `results.csv`, `results.json` | quality, criteria and cost on the papers all four share (`--common-papers`; fact test: question set A, 299 papers) |
| `verdict_check/` | C and prefcc on both question sets (899 papers): reproduces the previous project's two-seed verdict exactly (recall +0.0262 [+0.0201, +0.0322], p = 0.0001; length 0.988; ROUGE-L −0.0053; BERTScore −0.0085; unsupported share +0.0045 [−0.0003, +0.0093]; NLI −0.0243; judge −0.090) |
| `pareto/` | Pareto plots (recall and unsupported share against GPU-seconds per paper, parameters, training GPU-hours) and `decision.md` |
| `train_resources.md`, `.json` | wall time, GPU-hours, throughput and peak memory of the five training runs |
| `curves/` | learning curves (overview per measure; one page per run) |

## Read with care

* **Serving costs are not like-for-like.** A and B were served on an H200 with a 16,640-token window, C on an
  A100 with a 32,768-token window, and prefcc is assumed to cost what C costs (same base model and adapter
  rank). They come from the evaluation runs, not from `bench_inference.py`. In the new runs, every student is
  benchmarked on the same GPU type with the same settings.
* **No claim check exists for A and B**, so they are missing from the unsupported-share plots.
* **The decision table** uses `--recall-floor -0.05` only to show how the rule works.
* In `train_resources.md`, prefcc seed 2's lower throughput (1,282 against 4,410 tokens/s) reflects the GPU:
  seed 1 trained on an H200 and seed 2 on an L40S. The preference trainer did not log its GPU type, so the
  table cannot show it; compare training cost only between runs on the same GPU type.
