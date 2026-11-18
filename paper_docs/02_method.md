# Method

*Scope (2026-11-16).* This file covers SFT (Runs A–C). The methods applied to Student C afterwards (reward-selected targets, DPO
with a likelihood term, the length-controlled pairing rules and R-DPO term, the claim penalty, weight blends,
an agent steering training, and an agent designing the arms) are in `11_reward_selected_distillation.md`
and `12_reactive_training.md`, with settings in `slurm/models/q35-*.env`.

## Adaptation

LoRA on a frozen base model. Rank 64, α = 128, dropout 0, bias none, task type causal LM.

| | A | B | C |
|:--|--:|--:|--:|
| base parameters | 4,154,588,672 | 9,173,766,144 | (recorded when the run starts) |
| trainable | 132,120,576 (3.18%) | 174,587,904 (1.90%) | 147,849,216 |
| target modules | `q,k,v,o,gate,up,down_proj` | same | `q,k,v,o,in_proj_qkv,out_proj,gate,up,down_proj` |

Run C's target list is longer because its architecture is hybrid: 8 of its 32 layers use
`q/k/v/o_proj` and the other 24 use `in_proj_qkv`/`out_proj` for linear attention. With the default
list, three quarters of the layers would have been adapted only in their MLPs. The two gate and
decay projections of the recurrence (`in_proj_a`, `in_proj_b`) are deliberately left frozen.
The module inventory is produced by `scripts/inspect_model.py`, which instantiates a candidate on
the meta device (no weights, seconds) and reports every `nn.Linear` by name.

## Optimisation

| | value |
|:--|:--|
| examples per forward | 1 (variable length, no padding) |
| examples per optimiser step | 8 (grad_accum 4 per rank × 2 ranks) |
| loss | cross-entropy over target tokens only, token-weighted across the batch |
| learning rate | 1e-4, warmup-stable-decay |
| warmup / decay | 1% of steps / last 10% of steps, cosine to 10% of peak |
| optimiser | AdamW, β = (0.9, 0.95), weight decay 0, gradient clip 1.0 |
| precision | bf16 weights, fp32 adapter, bf16 autocast, fp32 loss |
| attention | PyTorch SDPA |
| gradient checkpointing | on (required: see below) |
| epochs | 1 |

**One example per forward.** Examples are 450–28,672 tokens. Padding them into a rectangular batch
would waste 25–35% of the compute, and packing them into fixed bins needs block-diagonal attention
and makes "after *N* papers" ambiguous. A single variable-length sequence wastes nothing and makes
the milestones exact.

**Token-weighted accumulation.** Each micro-loss is the mean over that example's target tokens, so
averaging micro-losses would weight a 400-token summary like a 2,400-token one. Each is scaled by
`n_tokens_i / n_tokens_in_batch × world_size`; the batch total is known in advance from the index,
so no communication is needed to get it right.

**Warmup-stable-decay rather than cosine.** The milestones are intermediate checkpoints of a single
run. Under a cosine schedule each would sit at a different learning rate and the curve would
measure the schedule as much as the data. WSD holds the rate flat through every milestone except
the last, which also receives the decay — stated wherever the final point is reported.

**Loss computed only where there are labels.** The prompt is 85–95% of the tokens and carries no
label. The transformer runs over the whole sequence, the hidden states are sliced at the positions
that predict a label, and the LM head is applied only there. At 16k tokens with a 150k vocabulary
this saves about 20 GB of logits and ~10% of the forward FLOPs; at 32k tokens with a 248k
vocabulary it is the difference between fitting and not. `tests/test_loss.py` asserts it returns
the same number as the standard masked-label path on a small random model.

**Gradient checkpointing is required, not optional.** Measured 2026-10-14: without it, the longest
example at 16,384 tokens filled all 80 GB of an A100 (`free: 198 MB` when it requested 284 MB).
Each model records its own verdict in `state/<model>.need_checkpointing`.

## Infrastructure

Two H200 GPUs per run (DDP, NCCL), 8–16 CPUs, 96–256 GB of host memory, on a shared Slurm cluster.

| | A | B |
|:--|--:|--:|
| optimiser steps | 8,750 | 8,750 |
| wall-clock hours | 13.75 | 16.63 |
| median tokens/s per rank | 7,513.9 | 5,949.6 |
| peak VRAM per GPU | 16.48 GiB | 29.63 GiB |
| chunks (allocations used) | 2 | 1 |
| optimiser steps skipped for non-finite gradients | 0 | 0 |

Run A used two allocations because its first was lost to an infrastructure stall (see
`09_experiment_log.md`); the work itself resumed from a checkpoint at most 2,500 papers behind.

## Engineering invariants

These exist because each was violated at least once and cost measurable time:

- Training resumes by seeking to the number of papers already consumed; the order is fixed on disk.
- Rolling checkpoints are directories published with one atomic rename, `DONE` written last, two
  kept, and a resume walks them newest-first and falls back if one will not load.
- A crash does not overwrite the last good checkpoint; a non-finite gradient norm skips the update.
- Generation appends every 100 papers, so an interrupted evaluation keeps what it has.
- Every evaluation is content-addressed by `<split>-<checkpoint>-n<papers>` and never recomputed.
- A training job holds its allocation through a crash and retries, because a new process re-reads
  the Python package and a fix can land while the job waits.
