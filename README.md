# SciSum students: training and choosing a production summarizer for scientific papers

This repository holds everything needed to train five small open models as structured
scientific-paper summarizers, measure them, and choose one for production:

- the data;
- the code and the job scripts;
- the documentation;
- the record of the original project.

The method is the one that worked on Qwen3.5-9B in the original project:

1. Distillation (SFT) on teacher summaries of full papers.
2. Fact-rewarded preference tuning (`prefcc`).

Start with **[`docs/START_HERE.md`](docs/START_HERE.md)**.

## What to run

| Student | Hugging Face id | Parameters | Architecture |
|:--|:--|--:|:--|
| `minicpm5-2b` | `openbmb/MiniCPM5-2B` | 2.5 B | Llama-style dense transformer |
| `granite42-3b` | `ibm-granite/granite-4.2-3b` | 3.7 B | Granite dense transformer |
| `qwen35-4b` | `Qwen/Qwen3.5-4B` | 4.7 B | Hybrid: 24 linear-attention and 8 full-attention layers |
| `k2h-3.7b` | `IFM/K2-Horizon-3.7B` | 5.1 B (250k-token vocabulary) | K2-Horizon (remote code) |
| `k2h-7b` | `IFM/K2-Horizon-7B` | 9.0 B | K2-Horizon (remote code) |
| anchor: `qwen35-9b` | `Qwen/Qwen3.5-9B` | 9.4 B | Student C of the original project, with its trained adapters included |

Each student goes through the same steps:

1. **SFT on 50,000 papers.** The original project used 20,000.
2. **Test evaluation** of the SFT model.
3. **The prefcc stage:**
   - sample candidate summaries on 1,520 held-out papers;
   - score them with a fact reward;
   - build length-matched pairs;
   - run DPO with two pair orders.
4. **Test evaluation** of both DPO seeds.
5. **Inference benchmark:** latency, throughput and memory.
6. **Collection** of all results into one table and a Pareto-front plot of quality against cost.

## Where things are

| Path | What |
|:--|:--|
| `docs/` | How to run and read everything. Start with `START_HERE.md`. |
| `scisum/` | The Python package: data, prompt, training, preference training, evaluation, fact test, claim check, judge |
| `configs/students/`, `configs/recipes/` | One file per student (model facts, LoRA targets) and per recipe (`sft50k`, `prefcc`) |
| `slurm/` | Site-neutral job scripts, with a site layer for LUMI and for a CUDA Slurm cluster |
| `scripts/pipeline.py` | The orchestrator: `status`, `next`, `run` for every student and stage, resumable |
| `scripts/analysis/` | Benchmark, training resources, results collection, learning curves, Pareto plots |
| `containers/` | How to get the LUMI runtime (ROCm PyTorch and vLLM containers) |
| `data/` | **Not in git, shipped in the zip.** Corpus, evaluation sets, the improve pool with its questions, anchors. See `docs/DATA.md`. |
| `paper_docs/` | The original project's full research record: method, every number, findings, threats to validity, dated log |
| `tests/` | `pytest`. No GPU or network needed. |

## Results of the original project (Student C, Qwen3.5-9B)

Measured on held-out test papers:

| | Teacher | C after SFT (20,000 papers) | C + prefcc (mean of two seeds) |
|:--|--:|--:|--:|
| Fact recall (899 papers, two question sets) | 0.608 | 0.571 | **0.597** (+0.026 [+0.020, +0.032]) |
| Length relative to the teacher | 1.00 | 0.99 | 0.99 |
| Unsupported-statement share, change from SFT | | | +0.005 (not significant) |

The full account is in `docs/WHAT_WORKED_ON_C.md`. The evidence is in `paper_docs/`.

## Data notice

`data/` contains:

- the bodies of about 100,000 papers, extracted from publisher PDFs under various licences;
- one teacher summary per paper;
- the evaluation question sets.

They are for this project's pipeline only and must not be re-published. Details are in `docs/DATA.md`.

## Licence

Code and documentation: Apache-2.0 (`LICENSE`). Citation: `CITATION.cff`.
