# 13 — A recipe for training a factual scientific-paper summarizer from any student model

*Written 2026-11-16. Step 5 gives `prefcc`'s settings. Round 4's verdict (`12` §9, F13) left `prefcc` the
recommended arm, so they stand.*

This file turns the project's results into a procedure for a new student of any size or family (for
example `ibm-granite/granite-4.2-3b`). Every step is labelled with how far our evidence carries it:

* **Measured**: shown on our students. A shows Qwen3-4B, B K2-Horizon-7B, C Qwen3.5-9B.
* **Measured on C only**: shown on Student C (Qwen3.5-9B), one starting model.
* **Untested**: our best judgement; check it on the new student before relying on it.

No step has been run on a model below 4 B parameters or outside the Qwen and K2 families.

## The pipeline at a glance

| Step | What | Evidence | Typical cost (our runs) |
|:--|:--|:--|:--|
| 0 | Check the student: context window, tokenizer, adapter targets, chat template | Measured (F3, F4, `02`) | Minutes |
| 1 | Data: teacher summaries of full papers, filtered, split by paper | Measured (`01`) | $132 for 100,826 summaries |
| 2 | SFT with LoRA on 20,000 papers, one pass | Measured, three students (F1, F2) | Hours on one or two GPUs |
| 3 | Evaluate C-style: fact test, claim check, length, overlap, NLI, judge | Measured (F7, F8, `03`) | ~2–3 h of GPU time per system |
| 4 | Build preference pairs from the student's own samples, ranked by a fact reward | Measured on C only (F8–F10) | ~1,500 papers × 6 candidates, graded by two LLMs |
| 5 | DPO + likelihood on about 1,200 length-matched pairs (`prefcc`) | Measured on C only, two seeds (F10) | ~9 h on one 48 GB GPU at 9 B |
| 6 | Two seeds, the four criteria, and a pick | Measured (F10, `12` §8) | Repeat steps 5 and 3 once |

What it bought on C: fact recall went from 0.571 to 0.597 (+0.026 [+0.020, +0.032], two-seed mean; the
teacher scores 0.608). Length stayed at the teacher's, ROUGE-L and BERTScore moved by less than 0.01, and the
unsupported share did not rise significantly. NLI (−0.024) and judge faithfulness (−0.09 of 5) fell
slightly (F10, `07` T18).

---

## Step 0 — Check the student before training it

| Check | Why | What to do |
|:--|:--|:--|
| **Usable context ≥ 32k tokens** | Seeing the whole paper is worth ~0.03 ROUGE-L on the papers it reaches (F4). At a 28,672-token prompt, 91% of papers fit whole; at 13,312, 61%. | Use `max_seq_len` 32,768 with a 28,672-token prompt budget if the model and memory allow it. Below that, the section-dropping fit (step 1) limits the loss. |
| **Tokenizer efficiency** | It sets how much paper fits. The effect is small: B's better tokenizer explained 0.0003 of its 0.026 ROUGE margin (F3). | `scripts/compare_candidates.py`. Measured characters per token: K2-Horizon 3.28, Qwen3.5 3.14, Qwen3-4B 3.09, granite-4.2-8b 2.97. |
| **Adapter targets** | Hybrid models (linear attention, Mamba, recurrence) name their projections differently. With the default list, C would have adapted only the MLPs in 24 of its 32 layers (`02`). | `scripts/inspect_model.py` lists every `nn.Linear` on the meta device. Target all attention or mixer input and output projections and the MLP. Leave gate and decay projections of a recurrence frozen. Granite 4 models are hybrids, so this check is needed. |
| **Chat template and reasoning block** | Reasoning models open a thinking block. An untrained reasoning model produced almost no parseable JSON (F1). | Derive the assistant wrapper from the template (`scisum.prompt.assistant_wrapper`). Train with an empty, closed thinking block and decode the same way. |
| **Size** | Larger is better on every overlap metric (F3). On facts, 7 B and 9 B tie, and 4 B is 0.045 lower (F7). | Expect a 3 B student to start below A (Qwen3-4B): A's fact recall is 0.527 against C's 0.574. Steps 4–5 have not been tested at that size. |

## Step 1 — Data

**Measured** (`01`). Unchanged for any student.

1. **Inputs.** Title, up to eight authors, categories, abstract, and the GROBID body. References and
   appendices are removed.
2. **Targets.** Structured summaries from a strong teacher, generated from the full text. Ours: 100,826
   summaries for $132.25, eight-field JSON: `tldr`, `problem`, `contributions`, `method`, `results`,
   `limitations`, `keywords`, `domain`.
3. **Filters.**
   - Contributions count outside 2–10.
   - Duplicate titles.
   - Bodies under 2,000 characters.
   - Abstract-only summaries.
   - Repetition: distinct-10 below 0.80, calibrated on the teacher's distribution. A naive n-gram rule
     removed 851 good summaries.
4. **Splits.** Split by paper, stratified by teacher model, with one fixed order, so that smaller runs use a
   prefix. Keep the test split untouched until the end. Set aside a pool from validation for step 4.
5. **Fitting the paper into the prompt.**
   - Use a fixed generation reserve, so training and inference see the same budget and the target's length
     cannot leak in.
   - Drop the least useful sections first (related work, appendices, acknowledgements), then cut head and
     tail.
   - Check every build with `scisum data verify`.
6. **Schema choice.** Questions on *motivation* and *scope* score 0.13–0.48 for every system, the teacher
   included, because the schema has no field for them (F7). If readers need significance or conditions,
   add a field before generating targets. No amount of training recovers it later.

## Step 2 — SFT (distillation)

**Measured on three students** (F1, F2, `02`).

| Setting | Value | Note |
|:--|:--|:--|
| Adapter | LoRA rank 64, α 128, dropout 0, all linear projections (step 0) | 1.9–3.2% of parameters trainable |
| Papers | **20,000** (5,000 is the minimum worth running) | Schema compliance is fixed by 1,000 papers (F1). ROUGE-L gains after 20,000 are ≤ 0.013 and confounded with the LR decay. The judge stops improving after 5,000 (F2). |
| Passes | 1 | |
| Batch | 8 examples per optimiser step, one variable-length example per forward, no padding | |
| Loss | Cross-entropy on target tokens only, token-weighted across the batch | The prompt is 85–95% of the tokens. Computing logits only at labelled positions saves ~20 GB at 16k tokens. |
| LR | 1e-4, warmup-stable-decay: 1% warmup, last 10% cosine to 10% of peak | Take the final, decayed checkpoint. The decay itself is worth ~+0.01–0.016 ROUGE-L (F2). |
| Optimiser | AdamW β (0.9, 0.95), weight decay 0, clip 1.0, bf16 base and fp32 adapter | |
| Memory | Gradient checkpointing on | One 16k example filled an 80 GB A100 without it |

**For small students (≤ 4 B) — untested.** A's curve kept rising slowly to 70,000 papers: +0.018 ROUGE-L
from 1,000 to 20,000 papers and +0.013 from 20,000 to 70,000 (F2). Training to 50,000–70,000 papers is cheap
at that size: 0.196 h per 1,000 papers for A on two H200s (F6). It buys teacher-likeness, not faithfulness
(F5).

**SFT does not improve faithfulness** (F5), in either family or at any data size. Faithfulness has to come
from the objective (steps 4–5) or from what the model sees (step 0), not from more SFT data.

## Step 3 — Evaluate the SFT student (the baseline "C")

**Measured** (F7, F8, `03`). Run this evaluation for every candidate, on the same papers.

| Measure | How | Papers |
|:--|:--|:--|
| **Fact recall** (primary) | Questions written from the full paper by model X. Model Y answers from the summary alone (NOT STATED allowed) and grades the answer. Recall = correct + ½ partial. Also report strict, correct-only scoring: partial credit is gameable. | ~900 papers, two question sets |
| **Claim check** | Every statement of the summary checked against the whole paper; unsupported-or-contradicted share. | 300 or more |
| **Length ratio** to the teacher | | All |
| **ROUGE-L, BERTScore** to the teacher | Teacher-likeness. Not a quality measure on its own: ROUGE ranked C above B, and the fact test did not (F7). | All, here 1,246 |
| **NLI** faithfulness, **judge** faithfulness | Canaries for reward hacking. | 300 / 500 |
| **Controls** (once per instrument) | No summary, abstract only, full paper. They bound the test: 0.31 / 0.19 / 0.88 recall in F7. | |

Use one model family to write the questions and another to answer and grade. At evaluation, swap them
against the family used in the training reward (step 4). Our reward used Qwen questions with Gemma answers;
our evaluation used Gemma questions with Qwen answers. Swapping the roles still leaves both sides as the
same kind of instrument (`07` T17).

## Step 4 — Preference pairs from the student's own summaries

**Measured on C only** (F8–F10, `11`). Settings fixed before training in rounds 1–3 and never tuned.

1. **Pool.** About 1,500 papers from the validation split that the student never trained on and that are
   not in the test split.
2. **Candidates per paper.** The teacher's summary and 5 student samples at temperature 0.8, top-p 0.95.
   The first 120 of 1,520 papers also had the greedy output and a sixth sample.
3. **Questions for the reward.**
   - Model Y writes 14 questions per paper from the full text.
   - Keep a question only if a whole-paper reader answers it and a closed-book reader does not.
   - A paper with fewer than 5 kept questions contributes no pair.
4. **Reward per candidate** (`prefcc`):

       R = fact + 0.15·ROUGE-L(candidate, teacher) − 0.30·unsupported_numbers
           − 1.0·max(0, length_ratio − 1.0) − 1.0·unsupported_claim_share

   - `fact`: +1 correct, +0.25 partial, 0 not stated, **−1 wrong**. Hedging is free and fabricating is not.
   - `unsupported_numbers`: the share of numbers with two or more digits, not years, that occur nowhere in
     the paper.
   - `unsupported_claim_share`: the claim check of step 3, run with the training-side verifier.
   - Invalid JSON, or a candidate that hit the token limit, is never chosen.
5. **Pairs.**
   - Chosen: the highest-reward candidate.
   - Rejected: among the student's valid samples at least 0.25 behind, the one **closest in length** to the
     chosen summary (`closest_length`).
   - On C this gave 1,233 pairs. The teacher's summary was the chosen one in 940 of them (76%).

**Do not pick the rejected summary by lowest reward with a loose length allowance.** With the length
allowance at 1.2× the teacher's and the rejected summary chosen by lowest reward, recall rose by +0.071, but
mostly because the summaries became 1.26× the teacher's length. They also carried more unsupported
statements (F8). DPO amplifies small length differences in the pairs.

## Step 5 — Preference training (`prefcc`)

**Measured on C only, two seeds** (F10).

| Setting | Value |
|:--|:--|
| Start and reference | The step-2 SFT adapter |
| Objective | DPO, β 0.1, + 1.0 × NLL of the chosen summary |
| LR | 2e-5 |
| Batch | 8 pairs per optimiser step |
| Passes | 1 |
| Length term in the loss (R-DPO) | **None**. With length-matched pairs it overshot to 0.84× and lost the fact gain (F9). |

**Tried and not better (do not repeat):**

| Approach | Result |
|:--|:--|
| SFT on reward-selected targets | +0.007 recall (F8) |
| R-DPO with lowest-reward pairs | +0.017 recall, but misses a criterion (F9) |
| Weight blends of C and `prefcc` | Not better than `prefcc` (F11) |
| An LLM agent steering DPO during training | Not better than fixed training (F12) |
| Claim weight 3 instead of 1 (`r4b`) | Ties `prefcc` (F13) |
| Teacher-chosen pairs only (`r4a`) | Loses two thirds of the fact gain (F13) |

## Step 6 — Seeds, criteria and the pick

**Measured** (F10, `12` §8).

* **Run two seeds** (`SCISUM_PREF_ORDER_SEED`), and judge on their mean. The pair order alone moved recall by
  0.013–0.020. `prefcc`'s seed 2 fails criterion 4 on its own, while the two-seed mean passes.
* **Four criteria against the SFT student**, fixed before evaluation:
  1. Recall higher, with p below the Bonferroni threshold.
  2. Length 0.90–1.10× the teacher's.
  3. ROUGE-L and BERTScore no more than 0.010 below.
  4. Unsupported share not higher.
* **Report NLI and the judge** even when they are not criteria. They are where `prefcc` still trails C.
* **Decode greedily** at test time: every number here is from greedy decoding.

---

## What to expect on a different student — untested

| Question | Our expectation | Why, and how to check |
|:--|:--|:--|
| Do steps 1–3 transfer? | Yes | Same results on three students from two families (F1, F2) |
| Does the +0.026 fact gain transfer? | Its sign probably does; its size is unknown | One student only. Check with two seeds and the four criteria. |
| A 3 B student's own samples are weaker. Does the pairing still work? | More pairs will have the teacher as the chosen summary | Already 76% on C (940 of `prefcc`'s 1,233 pairs). Report the share. If almost every chosen summary is the teacher's, steps 4–5 are close to DPO against the teacher. |
| β, LR, claim weight at another size | Keep ours first | They were never tuned on C either. Tune only on a validation set, never on the test split (`07` T20). |
| Context below 32k | The fact gain from the whole paper is lost on long papers | ~0.03 ROUGE-L on the papers that would not fit (F4) |
| Faithfulness (NLI and judge) | Will not improve by SFT; preference tuning costs a little | F5, F10. This is the open problem. |

## What this recipe does not settle

* **No human evaluation.** Every measure is an LLM or a metric (`07`).
* **The training reward and the evaluation are the same kind of instrument**, with the models swapped
  (`07` T17). A bias common to question-based scoring would pass unseen.
* **The test papers were reused across four rounds** (`07` T20). A new student needs a fresh test split, or
  at least a fresh question set.
* **Future work, not run:** human checks of the claim-check flags; a reward from an instrument of a
  different kind; a schema with fields for significance and scope; any tuning of β, LR or reward weights.
