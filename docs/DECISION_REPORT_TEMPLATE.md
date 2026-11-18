# Production student: decision report (template)

*Copy this file to `reports/DECISION_<date>.md` and fill every `<...>`. Every number comes from
`reports/analysis/results.md`, `reports/analysis/pareto/decision.md` or a benchmark JSON. Name the source
beside each number.*

## 1. Decision

| | |
|---|---|
| Chosen student | `<tag>` `<SFT / prefcc>` (adapter `runs/<run>/ckpt/<ckpt>/adapter`, base model `<id>@<revision>`) |
| Runner-up | `<tag>` `<stage>` |
| Rule applied | quality floor `<recall − teacher ≥ ...>`, criteria `<required / not>`, latency cap `<...>`, load `<papers/day>`, utilisation `<...>` (fixed on `<date>`, before the results) |
| Serving setup | `<GPU type>`, `<LoRA adapter / merged>`, vLLM `<version>`, `<ROCm / CUDA version>` |
| Expected cost | `<GPUs at load>` GPUs for `<papers/day>`; `<GPU-s per paper>` GPU-s per paper; p95 latency `<s>` |

## 2. Evidence

### Quality (test papers; graders `<models and revisions>`; `<n>` papers for the fact test)

| system | recall [95% CI] | − teacher | unsupported share | length ratio | ROUGE-L | BERTScore | NLI | judge | criteria vs own SFT |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `<chosen>` | | | | | | | | | |
| `<runner-up>` | | | | | | | | | |
| `<others ...>` | | | | | | | | | |
| teacher | | – | | 1.00 | – | – | | | – |
| anchor C / prefcc (re-graded) | | | | | | | | | |

### Cost (one GPU type: `<...>`)

| system | params (B) | train GPU-h (SFT + 1 preference run) | GPU-s / paper | papers / h | TTFT p50 | e2e p50 / p95 | weights + KV p95 (GB) |
|---|--:|--:|--:|--:|--:|--:|--:|
| `<chosen>` | | | | | | | |

### Pareto position

- Recall vs GPU-seconds per paper: `<on the front / dominated by ...>` (figure `pareto_recall_vs_gpu_s_per_paper.png`)
- Unsupported share vs GPU-seconds per paper: `<...>`
- Systems whose recall interval overlaps the chosen one's: `<...>`

## 3. Why this one and not the runner-up

`<Two to five sentences: what the chosen student buys or saves against the runner-up, with numbers and
intervals. Say plainly when a difference is within noise.>`

## 4. Risks and checks before deployment

| risk | status |
|---|---|
| Graders differ from the previous project's (anchors re-graded?) | `<...>` |
| Faithfulness: NLI / judge below the SFT model (as for prefcc on Qwen3.5-9B)? | `<...>` |
| Long papers: share of test papers cut to fit the prompt | `<...>` |
| Serving path tested end to end (adapter applied, outputs identical to evaluation on a few papers) | `<...>` |
| Human spot check of `<n>` summaries | `<done / not done>` |

## 5. What would change the decision

`<e.g. a production latency cap below X s; a load above Y papers/day, where the cheaper student wins; a
human check finding more errors than the claim check reports.>`

## 6. Provenance

Git commit `<...>`; `results.json` created `<...>`; benchmark files `<paths>`; graders `<served names,
HF ids, revisions, precision>`; container image `<...>`.
