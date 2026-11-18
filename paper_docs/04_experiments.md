# Experiments

*Scope (2026-11-16).* This file describes the three SFT runs. The improvement arms that continue Student C (rounds 1–4, the
blends, the reactive run) are configured in `slurm/models/q35-*.env` and described, with pair counts and
steps, in `11_reward_selected_distillation.md` and `12_reactive_training.md`.

Three runs. Everything except the student and the stated differences is held constant: the same
papers in the same order, the same prompt, the same LoRA rank, the same schedule, the same metrics,
the same judge protocol, the same seed (20261014).

## Run A — `Qwen/Qwen3-4B-Instruct-2507`

| | |
|:--|:--|
| Selected because | the deep-research survey of sub-9B open models (`../reports/Small open LLMs for paper summaries.md`): second-best faithfulness of any model under 9B on the Vectara HHEM leaderboard (5.7% hallucination rate), 99.9% answer rate, Apache-2.0, 262,144-token window |
| Architecture | dense, 36 layers, hidden 2,560, GQA 32/8, vocab 151,936, tied embeddings |
| Sequence length / prompt budget | 16,384 / 13,312 |
| Training papers | 70,000 (one pass, 8,750 steps) |
| Milestones | 1,000 / 5,000 / 10,000 / 20,000 / 50,000 / 70,000 |
| Hardware | 2 × H200, 13.75 h of training, 16.48 GiB peak per GPU (plus an 8.7 h filesystem stall, see `09_experiment_log.md`) |
| Status | complete: 7 checkpoints evaluated and judged, 15,000-paper test split, 3,000-paper base floor |

## Run B — `IFM/K2-Horizon-7B`

| | |
|:--|:--|
| Selected because | a second model family at roughly twice the parameters, Apache-2.0, to test whether Run A's plateau was a property of the student or of the data |
| Architecture | dense, 36 layers, hidden 4,096, GQA 32/8, vocab 250,624, **untied** embeddings, 524,288-token window; ~9.0 B parameters (7 B core + 1.03 B embeddings + 1.03 B head) |
| Differences forced by the model | ships its own modeling code (`trust_remote_code`, revision pinned to `4622f94a`); its chat template always opens a reasoning block, so the target closes it (`</ifm|think>` + JSON + `<|ifm|im_end|>`) |
| Sequence length / prompt budget | 16,384 / 13,312 (identical to A) |
| Training papers | 70,000 (one pass, 8,750 steps) |
| Milestones | identical to A |
| Hardware | 2 × H200, 16.63 h wall-clock, 29.63 GiB peak per GPU |
| Status | complete: 7 checkpoints evaluated, 6 judged, 15,000-paper test split; base-model floor on the test split not yet run |

## Run C — `Qwen/Qwen3.5-9B` (trained and evaluated)

Chosen from five candidates (`google/gemma-4-12B-it`, `Qwen/Qwen3.5-9B`, `Qwen/Qwen3.5-4B`,
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`, `ibm-granite/granite-4.2-8b`) by applying the
findings of A and B.

The deciding measurement: on 40 real paper bodies the candidates' tokenizers lie within 2.97–3.22
characters per token of each other, none better than K2-Horizon's 3.28, and at a 13,312-token
budget all of them fit the same 70% of papers whole. The vocabulary is therefore not the lever —
the budget is, and at 28,672 tokens 95% of bodies fit. That makes the *cost of a long window* the
criterion:

| Candidate | Attention at long context | Licence | Verdict |
|:--|:--|:--|:--|
| `Qwen/Qwen3.5-9B` | 24 linear-attention + 8 full-attention layers | Apache-2.0 | **selected** — near-linear cost, eight layers still see the whole paper |
| `google/gemma-4-12B-it` | 40 of 48 layers on a 1,024-token sliding window | Apache-2.0 | cheap but only 8 layers see the document; multimodal wrapper |
| `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B` | Mamba + attention, 128-expert MoE, 3 B active | other (not Apache) | highest ceiling, highest risk: LoRA and serving paths unproven for this architecture |
| `Qwen/Qwen3.5-4B` | same hybrid, smaller | Apache-2.0 | F3 says the larger student wins |
| `ibm-granite/granite-4.2-8b` | full attention, vocab 100,352 | Apache-2.0 | worst tokenizer of the five (2.97 chars/token), 131k window |

Configuration, applying F2 and F4:

| | value | reason |
|:--|:--|:--|
| sequence length / prompt budget | 32,768 / 28,672 | F4: at 13,312 only 60–63% of papers fit whole; at 28,672 it is **91.1%** |
| training papers | 20,000 | F2: both A and B plateaued between 5,000 and 20,000 papers |
| milestones | 500 / 1,000 / 2,500 / 5,000 / 10,000 / 20,000 | to resolve the part of the curve where the change happens |
| tokens per pass | 272 M | against 730 M (A) and 702 M (B) — 2.6× less compute |
| LoRA targets | extended with `in_proj_qkv`, `out_proj` | the 24 linear-attention layers would otherwise be adapted only in their MLPs |
| held-out loss cadence | every 2,500 papers | eight points on a 2,500-step run |

Pre-run verification completed: Apache-2.0; `.model` and `.lm_head` present so the head-slice loss
is unchanged; assistant wrapper derived from the template as `("\n</think>\n\n", "<|im_end|>")`;
vLLM 0.31 implements both `Qwen3_5ForConditionalGeneration` and `Qwen3_5ForCausalLM` with
`SupportsLoRA`; weights downloaded at pinned revision `c202236`; dataset built and verified.

**What that verification missed**: which of the two classes each side would *choose*. Training
loads with `AutoModelForCausalLM` and got the text-only `Qwen3_5ForCausalLM`; vLLM reads
`config.json`, which declares the vision-language `Qwen3_5ForConditionalGeneration`, and its text
layers sit under a different prefix. The adapter matched no vLLM module and every Run C evaluation
until 2026-10-19 was the base model (`09_experiment_log.md`). Training was unaffected.

### As run

| | value |
|:--|:--|
| hardware | 2 × A100-SXM4-80GB (an A100 node), moved from H200 after 26 h of an H200 node draining |
| self-check | no checkpointing OOMed on the longest examples (30,190–30,571 tokens); checkpointing on passed |
| training | 20,000 papers, 2,500 steps, 271,219,044 tokens, 0 skipped steps, **16.99 h** |
| throughput | median 2,298.6 tokens/s per rank (B on H200: 5,949.6) |
| peak memory | 38.07 GiB per rank |
| held-out loss | 0.7001 → 0.4497 (base → 20,000 papers, 200 papers) |
