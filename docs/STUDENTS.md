# The five students: what each one is, how it is configured, what to watch

The facts below were probed on 2026-11-17 by instantiating each architecture on the meta device, with
configs, tokenizers and remote code downloaded at the pinned revision. Raw output:
`configs/students/probe-2026-11-17.json`. To re-check one model, run `scripts/inspect_model.py` with its
profile sourced. Each profile is `configs/students/<tag>.env`. Training sources the profile, then a recipe
(`configs/recipes/sft50k.env` or `prefcc.env`).

Tokenizer efficiency was measured on 40 real paper bodies from the corpus (the first 40 indexed papers with
over 20,000 characters of body), fitted exactly as the prompt builder fits them. "Fits whole" is the
share whose whole body plus about 1,500 tokens of prompt overhead fits the 28,672-token prompt budget.

Every version check is against the earlier work's lock file: transformers 5.17.0 and vLLM 0.31.0, which
lists 387 architectures.

## Overview

| Tag | HF repo @ pinned revision | Architecture | Parameters | Layers | Context | Chars / token | Fits whole in 28,672 |
|:--|:--|:--|--:|--:|--:|--:|--:|
| `minicpm5-2b` | `openbmb/MiniCPM5-2B` @ `42433e9` | `LlamaForCausalLM`, dense, GQA (16 q / 2 kv heads) | 2.52 B | 42 | 131k | 3.11 | 95% |
| `granite42-3b` | `ibm-granite/granite-4.2-3b` @ `4e9c275` | `GraniteForCausalLM`, dense (**not** a Mamba hybrid) | 3.66 B | 40 | 131k | 2.97 | 92.5% |
| `qwen35-4b` | `Qwen/Qwen3.5-4B` @ `851bf6e` | `Qwen3_5ForConditionalGeneration`: hybrid, 24 linear-attention + 8 full-attention layers; vision-language wrapper | ~4.2 B text model (tied embeddings) | 32 | 262k | 3.14 | 95% |
| `k2h-3.7b` | `IFM/K2-Horizon-3.7B` @ `ea9a45d` | `K2HorizonForCausalLM` (remote code), dense | 5.06 B (3.77 B without embeddings) | 36 | 524k | 3.28 | 95% |
| `k2h-7b` | `IFM/K2-Horizon-7B` @ `1ad5223` | `K2HorizonForCausalLM` (remote code), dense | 9.00 B (6.95 B without embeddings) | 36 | 524k | 3.28 | 95% |
| anchor `qwen35-9b` | `Qwen/Qwen3.5-9B` @ `c202236` | as `qwen35-4b` (= Student C) | 8.95 B text model | 32 | 262k | 3.14 | 95% |

All five are Apache-2.0 and ungated. All support a 32,768-token sequence, so every student uses C's budget:
- `SCISUM_MAX_SEQ_LEN=32768`;
- `SCISUM_GEN_RESERVE_TOKENS=4096`, which gives a 28,672-token prompt.

The tokenizer matters little at this budget: 92.5–95% of bodies fit whole in every student. In the earlier
work, seeing the whole paper was worth about 0.03 ROUGE-L on the papers it reached.

**K2-Horizon-7B changed after the earlier work.**
- Run B used revision `4622f94` with 9.17 B parameters. The repo was updated on 2026-11-04, and the head
  `1ad5223` has 9.00 B parameters.
- The profile pins the new head, so Run B's numbers do not describe this revision.
- Run B was also trained on 70,000 papers at a 13,312-token budget, which this plan does not use.

## Per student

### `minicpm5-2b` — openbmb/MiniCPM5-2B

| | |
|:--|:--|
| LoRA targets | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` (all 42 layers have all seven) |
| Assistant wrapper (from the template) | prefix `<think>\n\n</think>\n\n`, suffix `<|im_end|>`: the student writes an empty reasoning block, then the JSON |
| Trunk for the head-slice loss | `.model` and `.lm_head` present; untied embeddings (130,560 tokens) |
| vLLM 0.31 | supported natively (`LlamaForCausalLM`); serve the base repo, no copy needed |
| Risks | The smallest model, so expect the largest quality gap to the teacher. In the earlier work Qwen3-4B trailed the 9 B students by 0.045 fact recall. Only 2 KV heads, so the KV cache is small and serving is cheap. |

### `granite42-3b` — ibm-granite/granite-4.2-3b

| | |
|:--|:--|
| LoRA targets | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` (40 layers) |
| Assistant wrapper | prefix `</think>`, suffix `<|im_end|>`. The template's generation prompt opens `<think>\n`, but a rendered answer starts `<think></think>`. `scisum.prompt.assistant_wrapper` therefore takes the common prefix of the two, and the student learns to close the block right after the prompt's `<think>\n`. **Needs the common-prefix fallback in `assistant_wrapper`.** The original function raises on this template. |
| Trunk | `.model`, `.lm_head`; untied embeddings (100,352 tokens) |
| vLLM 0.31 | supported natively (`GraniteForCausalLM`) |
| Risks | Least efficient tokenizer (2.97 chars/token): about 6% more tokens per paper than Qwen3.5, and 92.5% of bodies fit whole. Despite the "4" in the family name, this checkpoint is a plain dense transformer: no Mamba kernels are needed. |

### `qwen35-4b` — Qwen/Qwen3.5-4B

| | |
|:--|:--|
| LoRA targets | C's list: `q_proj k_proj v_proj o_proj` (8 full-attention layers), `in_proj_qkv out_proj` (24 linear-attention layers), `gate_proj up_proj down_proj` (32 MLPs). Frozen as in C: `in_proj_a`, `in_proj_b` (recurrence gate/decay) and `in_proj_z` (output gate). |
| Assistant wrapper | prefix `\n</think>\n\n`, suffix `<|im_end|>` (as C) |
| Trunk | Training loads the text-only `Qwen3_5ForCausalLM` (`.model`, `.lm_head`). The checkpoint also holds a vision tower and an MTP head, which are not used. **Embeddings are tied**: there is no `lm_head` tensor in the checkpoint. |
| vLLM 0.31 | supports `Qwen3_5ForConditionalGeneration`, but loads the vision-language class, whose layer names differ from the trained adapter's. Serve a text-only copy (`scripts/make_text_only.py`, path in `SCISUM_SERVE_MODEL`). This is the Student C lesson: until it was found, every evaluation of C measured the untrained model. |
| Risks | `make_text_only.py` was written for the 9 B model, which has an `lm_head` tensor. **Untested on the tied 4 B checkpoint**: run `scripts/probe_vllm_adapter.py` before trusting any evaluation. The linear-attention layers use `flash-linear-attention` Triton kernels (0.5.2 in the lock), with a slow, memory-hungry PyTorch fallback. On ROCm (LUMI), check in the smoke test that the kernels load. |

### `k2h-3.7b` — IFM/K2-Horizon-3.7B

| | |
|:--|:--|
| LoRA targets | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` (36 layers) |
| Assistant wrapper | prefix `</ifm|think>`, suffix `<|ifm|im_end|>`. The template always opens `<ifm|think>` in the generation prompt and raises if an assistant message has no thinking field (as Run B). |
| Trunk | `.model`, `.lm_head`. Untied embeddings with a 250,624-token vocabulary: 1.28 B of the 5.06 B parameters are embedding and head. |
| vLLM 0.31 | supported natively (`K2HorizonForCausalLM`); serve the base repo |
| Risks | **Remote code**: training executes the repo's `modeling_k2_horizon.py`, so keep the revision pinned. The code declares SDPA, flash-attention and flex-attention support, and the pipeline uses SDPA. The large vocabulary makes the output head relatively expensive. The head-slice loss computes logits only at the summary tokens, which keeps memory in check. |

### `k2h-7b` — IFM/K2-Horizon-7B

| | |
|:--|:--|
| LoRA targets | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` (36 layers) |
| Assistant wrapper | as `k2h-3.7b` |
| Trunk | `.model`, `.lm_head`; untied 250,624-token embeddings (2.05 B of the 9.00 B parameters) |
| vLLM 0.31 | supported natively |
| Risks | Remote code, with the revision changed since Run B (above). The largest student, about the size of C. Every layer has full attention, unlike C's hybrid, so attention cost grows quadratically with length. Expect its training to be the slowest per paper. |

## Cost estimates (not measured; check them in the first hour of each run)

These are scaled from Student C's measured SFT run:
- 20,000 papers, 272 M tokens, one pass;
- 16.99 h on 2 × A100 80 GB, about 34 A100-hours;
- gradient checkpointing on, one sequence per forward pass.

50,000 papers is about 2.5× the tokens, adjusted by each tokenizer's chars/token against Qwen3.5's.

The earlier work found that, at one sequence per forward pass, a 2.21× larger model trained only 1.24×
slower. These estimates therefore scale time by (parameters ratio)^0.27, not linearly. That rule was
measured on 7 B against 4 B dense models at a 16k context. It does not capture the quadratic attention cost
of dense models at 32k against C's mostly linear attention. Read the column as a range of ±30% at best.

| Student | Tokens for 50k papers (est.) | SFT time on 2 × A100 80 GB (est.) | A100-hours (est.) |
|:--|--:|--:|--:|
| `minicpm5-2b` | ~690 M | ~30 h | ~60 |
| `granite42-3b` | ~720 M | ~35 h | ~70 |
| `qwen35-4b` | ~680 M | ~35 h | ~70 |
| `k2h-3.7b` | ~650 M | ~35 h | ~70 |
| `k2h-7b` | ~650 M | ~40–50 h | ~80–100 |
| (C at 20k, measured) | 272 M | 16.99 h | 34 |

Rules for the real numbers:
- **Read the real ETA from the run.** `runs/<run>/metrics.jsonl` logs `eta_s`, tokens/s and `vram_gb` every
  5 steps. After the first ~200 steps, the ETA is reliable to within ~10%.
- **On LUMI** (MI250X, one GCD = 64 GB), per-GCD speed against an A100 is unknown for this workload.
  Measure it with the smoke run before booking the allocation (`docs/LUMI.md`).
- **Peak memory.** C's preference stage peaked at 37 GB on a 48 GB L40S at 32k tokens per sequence. Every
  new student is smaller or similar, so 64 GB GCDs should hold one sequence per forward pass with gradient
  checkpointing. The training job's memory self-check confirms it.

The preference stage, measured on C:
- sampling the 1,520-paper pool, 5 samples each, about 18 s per paper on a 40 GB MIG slice in its first job;
- grading the candidates (LLM graders);
- DPO on about 1,200 pairs, about 9 h on one L40S for 9 B.

Smaller students should sample and train faster. Grading depends only on the graders' throughput
(`docs/GRADERS.md`).
