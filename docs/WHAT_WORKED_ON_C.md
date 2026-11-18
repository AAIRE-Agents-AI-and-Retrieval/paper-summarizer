# Training a factual scientific-paper summarizer: what worked on Qwen3.5-9B

*Status as of 2026-11-16. All numbers are measured on held-out test papers.*

This document describes a recipe that turns a general 9-billion-parameter language model (Qwen3.5-9B)
into a scientific-paper summarizer. The resulting summarizer:

* writes a structured summary of a full paper;
* recalls about as many facts as the much larger model it learned from;
* writes summaries of the same length as that model's;
* makes no measurably more unsupported statements than plain distillation.

The recipe has two stages:

1. **Distillation.** Supervised fine-tuning (SFT) on summaries written by a strong teacher model.
2. **Fact-rewarded preference tuning.** The distilled model writes several summaries of new papers.
   Each summary is scored by how many facts about the paper it gets right, minus penalties for
   unsupported content and excess length. The model is then trained with DPO to prefer the better
   summary of each pair.

## Result

The table compares the teacher, the model after stage 1, and the model after stage 2. Stage 2 is the mean
of two training runs that differ only in the order of the pairs.

| Measured on test papers | Teacher | After stage 1 (SFT) | After stage 2 (SFT + preference) |
|:--|--:|--:|--:|
| Fact recall (899 papers, two independent question sets) | 0.608 | 0.571 | **0.597** |
| Gain over stage 1 | | | **+0.026** [+0.020, +0.032], p = 0.0001 |
| Length relative to the teacher's summary | 1.00 | 0.99 | 0.99 |
| ROUGE-L / BERTScore against the teacher, change from stage 1 | | | −0.005 / −0.009 |
| Share of statements unsupported or contradicted by the paper, change from stage 1 | | | +0.005 [−0.000, +0.009] (not significant) |

Brackets are 95% paired bootstrap intervals over papers. Fact recall and the claim check are defined in
"How to measure" below.

Stage 1 alone already gives a usable summarizer:

* 99.9% of its outputs are valid JSON of the schema;
* 96.6% of its statements are supported by the paper;
* it recalls 94% as many facts as the teacher.

Stage 2 closes most of the remaining fact gap. The first training run alone reached 0.604, level with the
teacher.

**What stage 2 does not fix.** Two other faithfulness measures stay slightly below the stage-1 model:

* sentence-level NLI entailment: −0.024;
* an LLM judge's 1–5 faithfulness score: −0.09.

Read the gain as "more informative at the same length and about the same accuracy", not "better on every
axis".

---

## The task

| | |
|:--|:--|
| **Input** | One paper: title, up to eight authors, arXiv categories, abstract, and the full body text extracted from the PDF with GROBID. References and appendices are removed. |
| **Output** | One JSON object with eight fields: `tldr`, `problem`, `contributions` (list), `method`, `results`, `limitations`, `keywords` (list), `domain` |
| **Prompt** | One fixed production prompt, identical in training and use |

## Stage 0 — Data

1. **Papers.** Highly cited arXiv papers with full text. The teacher summaries came from 100,000 papers;
   this student used 20,000 of them for training.
2. **Teacher summaries.**
   - A strong LLM wrote one structured summary per paper from the full text. In 88% of papers it was
     DeepSeek V4 Flash.
   - About 100,000 summaries cost $132.
   - The summaries are machine-generated targets, not ground truth.
3. **Filters, each calibrated on the teacher's own output before use:**
   - drop summaries with fewer than 2 or more than 10 contributions;
   - drop duplicate papers;
   - drop bodies under 2,000 characters;
   - drop summaries written from the abstract alone;
   - drop repetitive summaries (distinct 10-grams below 80%).

   A naive "repeated n-gram" rule rejected 851 good summaries, because dense technical text repeats
   terms legitimately.
4. **Splits.**
   - Split by paper into train, validation and test, with one fixed order.
   - Keep the test split untouched until the end.
   - Set aside about 3,000 validation papers for stage 2.

## Stage 1 — Distillation (SFT)

### Fit the whole paper into the prompt

This was the single most useful design decision.

| Setting | Value |
|:--|:--|
| Sequence length | 32,768 tokens |
| Prompt budget for the paper | 28,672 tokens; a fixed 4,096 is reserved for the answer |
| Papers whose whole body fits | **91%** (median prompt 10,624 tokens, 90th percentile 25,458) |
| Target length | median 992 tokens, 90th percentile 1,255 |

* **Shrinking a paper that does not fit.**
  - Drop the least useful sections first: related work, appendices, acknowledgements.
  - Keep methods and results.
  - Only then cut the remaining text in the middle, keeping its head and tail.
* **The reserve for the answer is fixed.** It is not set from the target's length, so training and use
  see the same budget, and the target length cannot leak into the prompt.
* **What the long window is worth.** On papers that a 13k-token window would have truncated, the long
  window was worth about **+0.03 ROUGE-L**. On papers that fit either way it changed nothing.
* **Why this model makes the long window affordable.** Qwen3.5-9B has 24 linear-attention layers and 8
  full-attention layers, so its cost grows almost linearly with length.

### Adapter

* **LoRA:** rank 64, α 128, dropout 0, 148 M trainable parameters.
* **Target every projection, including the linear-attention ones:**
  - `q_proj, k_proj, v_proj, o_proj` (full-attention layers);
  - `in_proj_qkv, out_proj` (linear-attention layers);
  - `gate_proj, up_proj, down_proj` (MLP).

  With the default list, 24 of the 32 layers would have been adapted only in their MLPs.
* **Keep frozen** the recurrence's gate and decay projections (`in_proj_a`, `in_proj_b`).

### Training

| Setting | Value |
|:--|:--|
| Training papers | 20,000, one pass (272 M tokens) |
| Batch | 8 examples per optimiser step. One variable-length example per forward pass, so there is no padding. |
| Loss | Cross-entropy on the summary tokens only, weighted by token count across the batch. Logits are computed only at labelled positions. |
| Learning rate | 1e-4, warmup-stable-decay: 1% warmup, constant, then cosine down to 10% of peak over the last 10% of steps |
| Optimiser | AdamW, β = (0.9, 0.95), weight decay 0, gradient clip 1.0 |
| Precision | bf16 base weights, fp32 adapter, bf16 autocast, fp32 loss |
| Memory | Gradient checkpointing (required at 32k tokens) |
| Reasoning block | Qwen3.5 opens a thinking block. Train and decode with it empty and closed: the target is preceded by `\n</think>\n\n` and followed by `<\|im_end\|>`. |
| Cost | 17 h on 2 × A100 80 GB |

**How the model improved with data** (validation ROUGE-L):

| Papers trained on | 0 | 500 | 2,500 | 10,000 | 20,000 (after the LR decay) |
|:--|--:|--:|--:|--:|--:|
| ROUGE-L | 0.077 | 0.450 | 0.468 | 0.463 | **0.479** |

* **Schema.** The untrained model never produced the schema as clean JSON: it spent its output on a
  thinking trace. After 500 papers it produced the schema 99.7% of the time.
* **Plateau.** Quality is flat from 2,500 to 10,000 papers.
* **Learning-rate decay.** The final rise (+0.016) coincides with the decay, so take the final, decayed
  checkpoint. A few thousand papers buy most of the gain. 20,000 is enough.

**Check that serving actually loads the adapter.** Training loaded the text-only model class. The serving
engine (vLLM) loaded the vision-language class, whose layer names differ, so the adapter silently matched
nothing. Every evaluation until this was found measured the untrained model. Before trusting any number:

* compare one output with and without the adapter;
* confirm the adapter's module names exist in the served model.

**SFT does not improve faithfulness.** More data made the student more like the teacher, but NLI
faithfulness did not trend up at any data size. Faithfulness has to come from the training objective
(stage 2), not from more distillation data.

## Stage 2 — Fact-rewarded preference tuning

The idea: make the student write several summaries of papers it has never seen, find out which summary
gets the paper's facts right without making things up, and train it to prefer that one.

### 2a. Candidates

* **Pool.** About 1,500 validation papers that the student never trained on.
* **Candidates per paper:**
  - the teacher's summary;
  - 5 summaries sampled from the student at temperature 0.8, top-p 0.95.

  The first 120 papers also had the student's greedy summary and a sixth sample.
  Five samples were enough.

### 2b. A reward that measures facts

1. **Questions.** One LLM (Qwen3.8-27B) writes 14 specific questions with gold answers per paper, from
   the full text.
2. **Filter the questions with two readers** (Gemma-4-31B):
   - one reader sees the whole paper, the other sees nothing;
   - keep a question only if the whole-paper reader answers it correctly (so it is answerable) and the
     closed-book reader does not (so it does not give away its answer);
   - papers with fewer than 5 kept questions are skipped.
3. **Answer and grade.** The same reader answers each question from the candidate summary alone ("not
   stated" is allowed) and grades the answer against the gold.
4. **Claim check.** Every statement of the candidate is checked against the whole paper. It is labelled
   supported, unsupported or contradicted.
5. **Reward:**

       R = fact + 0.15 · ROUGE-L(candidate, teacher)
             − 0.30 · unsupported_numbers
             − 1.0  · max(0, length / teacher_length − 1.0)
             − 1.0  · unsupported_or_contradicted_share

   | Term | Meaning |
   |:--|:--|
   | `fact` | Mean over kept questions: **+1** correct, **+0.25** partial, **0** not stated, **−1** wrong. A wrong answer costs more than a missing one, so hedging is free and fabricating is not. |
   | `ROUGE-L` to the teacher | The imitation anchor. It is 1 for the teacher's own summary, so a sample replaces the teacher only when it is clearly more right. |
   | `unsupported_numbers` | Share of the summary's numbers (two or more digits, not years) that appear nowhere in the paper. A model-free check. |
   | Length term | Penalises any length beyond the teacher's. A longer summary answers more questions, so without this term the reward pays for length. |
   | Claim-check term | Share of the summary's statements that the paper does not support. This term turned "more facts, slightly more errors" into "more facts, about the same errors". |

   * Invalid JSON, and a summary that hit the token limit, are never chosen.
   * The weights were set once, before training, and never tuned. Tripling the claim-check weight
     (3.0 instead of 1.0) gave the same result: fact recall −0.003 and unsupported share +0.001 against
     weight 1, both within noise.

### 2c. Pairs

| Rule | Detail |
|:--|:--|
| Chosen | The highest-reward candidate. In 940 of the 1,233 pairs (76%) this was the teacher's summary. |
| Rejected | Among the student's own valid samples scoring at least 0.25 below the chosen one, the sample **closest in length** to the chosen summary |
| Count | 1,233 pairs from about 1,500 papers |

Matching the two summaries' lengths matters. DPO amplifies any length difference between chosen and
rejected summaries into longer outputs. Paired this way:

* the median ratio of rejected to chosen length is 0.994;
* the rejected summary is the shorter one in 51.5% of pairs;
* the trained model keeps the teacher's length.

### 2d. Training

| Setting | Value |
|:--|:--|
| Start and reference model | The stage-1 adapter |
| Objective | DPO with β = 0.1, plus 1.0 × the negative log-likelihood of the chosen summary. The likelihood term keeps the model close to well-formed summaries. |
| Learning rate | 2e-5 |
| Batch | 8 pairs per optimiser step |
| Passes | 1 (about 155 steps) |
| Length term in the loss | None. Length is handled in the reward and the pairing. |
| Cost | About 9 h on one 48 GB GPU (L40S) |

### 2e. Run it twice

The order of the pairs alone moved fact recall by 0.013–0.020 between otherwise identical runs. Train two
seeds and judge the method on the per-paper mean of the two:

* the two runs of this recipe gained +0.033 and +0.020;
* one run alone passed every check, and the other narrowly failed the unsupported-statements check.

---

## How to measure

Distilled summaries look good on overlap metrics while missing facts. To measure what a reader actually
learns, evaluate with instruments that do not use the teacher as the reference.

| Measure | Method | Role |
|:--|:--|:--|
| **Fact recall** (primary) | One LLM writes 10 questions per paper from the full text. Another LLM answers them from the summary alone and grades the answers. Recall = correct + ½ partial. Run two independent question sets on about 900 papers. Also report strict scoring (correct only), because partial credit can be earned by guessing. | What a reader can learn |
| **Claim check** | Every statement is checked against the whole paper. Report the unsupported-or-contradicted share. | Accuracy |
| **Length** | Relative to the teacher's summary | Guards against reward-by-length |
| **ROUGE-L, BERTScore** | Against the teacher's summary | Guards against drifting from the format and style |
| **NLI, LLM judge** | Sentence entailment against retrieved paper passages, and a 1–5 judge score | Canaries for reward hacking |

**Separate the training instruments from the evaluation instruments.**

| | Writes the questions | Answers, grades, and checks claims |
|:--|:--|:--|
| Training reward | Qwen3.8-27B | Gemma-4-31B |
| Evaluation | Gemma-4-31B | Qwen3.8-27B |

**Fix the acceptance criteria before evaluating.** A trained model was accepted only if, against the
stage-1 model:

1. fact recall is higher, with p below a Bonferroni-corrected threshold;
2. length is 0.90–1.10 × the teacher's;
3. ROUGE-L and BERTScore drop by no more than 0.010;
4. the unsupported share is not higher.

The final model meets all four on the two-seed mean.

**Calibrate the fact test once with controls:**

| Control | Fact recall |
|:--|--:|
| No summary | 0.31 (almost all partial-credit guesses) |
| Abstract only | 0.19 |
| Full paper | 0.88 (the ceiling) |

The distance from the abstract-only score to the full-paper ceiling shows how much a summary can add.

**Decode greedily.** Every number in this document is from greedy decoding.

## Where the errors are

| What the fact test showed | Teacher | Stage-1 student |
|:--|--:|--:|
| Questions answered wrongly | 4.2% | 4.7% |
| Questions left unanswered ("not stated") | 23.1% | 26.8% |

* **Omission, not fabrication.** After distillation, the student trails the teacher mainly by leaving
  facts out. Stage 2 recovers facts without adding errors on average.
* **Motivation and scope.** Questions about why the work matters or under what conditions it holds score
  0.13–0.48 for every system, the teacher included. The eight-field schema has no field for them. If
  readers need significance or scope, add a field before generating the teacher summaries. Training
  cannot recover information the format does not ask for.

## Limits of this evidence

* **One student model, one teacher.** The stage-2 settings (β, learning rate, reward weights) were never
  tuned.
* **The training reward and the evaluation are the same kind of instrument.** Both use LLM-written
  questions and an LLM claim check, with the models swapped. A bias shared by such instruments would go
  unseen.
* **The test papers were reused** across several rounds of development.
* **No human evaluation.** The NLI and judge faithfulness scores remain slightly below the stage-1
  model's.

## The recipe on one page

1. **Data.** Teacher summaries of full papers, filtered with thresholds calibrated on the teacher's
   output, split by paper.
2. **Prompt.** A 32k-token context, with a 28,672-token prompt and a fixed 4,096-token answer reserve.
   Shrink a paper by dropping its least useful sections first.
3. **SFT.**
   - LoRA rank 64 / α 128 on every attention, linear-attention and MLP projection.
   - 20,000 papers, one pass.
   - Learning rate 1e-4 with warmup-stable-decay; keep the decayed final checkpoint.
   - Loss on summary tokens only.
   - Verify that the served model really applies the adapter.
4. **Candidates.** On about 1,500 unseen papers: the teacher's summary and 5 student samples at
   temperature 0.8, top-p 0.95.
5. **Reward.** fact (+1 / +0.25 / 0 / −1) + 0.15 ROUGE-L − 0.30 unsupported numbers − 1.0 excess length
   − 1.0 unsupported-claim share.
6. **Pairs.** Chosen = best candidate. Rejected = the student sample closest in length among those at
   least 0.25 worse.
7. **DPO.** β 0.1 + 1.0 × likelihood of the chosen summary, learning rate 2e-5, 8 pairs per step, one
   pass, from the SFT adapter.
8. **Seeds.** Train two pair orders and evaluate their mean.
9. **Evaluation.** Fact recall, claim check, length, overlap, NLI and judge, with criteria fixed in
   advance.
