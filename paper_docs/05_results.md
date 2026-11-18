# Results

*Scope (2026-11-16).* This file holds the SFT results (Runs A–C). The results of the improvement work on Student C are in
`11_reward_selected_distillation.md` (rounds 1–3) and `12_reactive_training.md` (`prefcc`'s second seed,
blends, reactive run, round 4); the headline is in `README.md`, "Where things stand".

All numbers from `data/evaluations.csv` and `data/val_loss.csv`. Validation points are 1,000 papers
each (faithfulness on the first 300); test points are 15,000 papers (3,000 for the base floor).

## Validation curve, Run A — Qwen3-4B

| papers seen | JSON ok | schema ok | ROUGE-1 | ROUGE-L | BERTScore | keyword F1 | domain exact | faithfulness | EOS | out tokens | judge Δ | tie |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 (base) | 0.997 | 0.672 | 0.5939 | 0.3271 | 0.2856 | 0.3604 | 0.045 | 0.4741 | 1.000 | 669 | −0.450 | 0.13 |
| 1,000 | 0.993 | 0.993 | 0.6842 | 0.4138 | 0.3886 | 0.5030 | 0.120 | 0.4737 | 0.993 | 884 | −0.236 | 0.37 |
| 5,000 | 0.987 | 0.987 | 0.6924 | 0.4204 | 0.3949 | 0.5275 | 0.150 | 0.4409 | 0.988 | 1,048 | −0.169 | 0.49 |
| 10,000 | 0.984 | 0.983 | 0.6882 | 0.4200 | 0.3973 | 0.5310 | 0.176 | 0.4901 | 0.985 | 1,025 | −0.203 | 0.44 |
| 20,000 | 0.993 | 0.993 | 0.6997 | 0.4315 | 0.4086 | 0.5529 | 0.152 | 0.4401 | 0.993 | 950 | −0.163 | 0.44 |
| 50,000 | 0.985 | 0.985 | 0.6990 | 0.4341 | 0.4087 | 0.5560 | 0.189 | 0.4543 | 0.985 | 1,005 | −0.166 | 0.49 |
| 70,000 | 0.986 | 0.986 | 0.7043 | 0.4448 | 0.4193 | 0.5660 | 0.207 | 0.4522 | 0.986 | 982 | −0.170 | 0.49 |

## Validation curve, Run B — K2-Horizon-7B

| papers seen | JSON ok | schema ok | ROUGE-1 | ROUGE-L | BERTScore | keyword F1 | domain exact | faithfulness | EOS | out tokens | judge Δ | tie |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 (base) | 0.000 | 0.645 | 0.4892 | 0.2774 | 0.0806 | 0.3134 | 0.026 | 0.2065 | 0.953 | 1,920 | −0.712 | 0.38 |
| 1,000 | 0.992 | 0.992 | 0.7116 | 0.4474 | 0.4209 | 0.5556 | 0.137 | 0.4239 | 0.993 | 879 | −0.125 | 0.55 |
| 5,000 | 0.992 | 0.992 | 0.7180 | 0.4569 | 0.4295 | 0.5704 | 0.154 | 0.4181 | 0.994 | 937 | −0.096 | 0.57 |
| 10,000 | 0.996 | 0.996 | 0.7227 | 0.4600 | 0.4357 | 0.5709 | 0.178 | 0.4530 | 0.996 | 918 | −0.095 | 0.56 |
| 20,000 | 0.989 | 0.989 | 0.7187 | 0.4583 | 0.4282 | 0.5805 | 0.165 | 0.4105 | 0.990 | 942 | −0.101 | 0.60 |
| 50,000 | 0.986 | 0.986 | 0.7165 | 0.4584 | 0.4299 | 0.5787 | 0.170 | 0.4426 | 0.986 | 949 | −0.113 | 0.58 |
| 70,000 | 0.984 | 0.979 | 0.7209 | 0.4683 | 0.4373 | 0.5861 | 0.203 | 0.4166 | 0.984 | 939 | −0.098 | 0.59 |

Run B's untrained row is not comparable to Run A's: K2-Horizon is a reasoning model and its
untrained output is prose with a thinking block, so few replies parse (JSON ok 0.000) and its
median output is 1,920 tokens.

## Test split (15,000 papers, final checkpoint; base floor on 3,000 of the same papers)

| metric | A base (3,000) | A ex70000 | B ex70000 |
|:--|--:|--:|--:|
| clean JSON | 0.9937 | 0.9845 | 0.9915 |
| schema complete | 0.6680 | 0.9845 | 0.9850 |
| ROUGE-1 | 0.5912 | 0.7043 | **0.7255** |
| ROUGE-L | 0.3245 | 0.4432 | **0.4691** |
| BERTScore | 0.2833 | 0.4173 | **0.4424** |
| keyword F1 | 0.3566 | 0.5637 | **0.5886** |
| domain exact | 0.0303 | 0.1877 | 0.1862 |
| NLI faithfulness | 0.4671 | **0.4520** | 0.4184 |
| stopped on EOS | 0.9997 | 0.9849 | 0.9916 |
| output tokens | 664 | 979 | 927 |
| length vs reference | 0.682 | 0.958 | **0.988** |
| distinct-4 | — | 0.9071 | 0.9166 |
| generation throughput | 1,512 tok/s | 1,407 tok/s | 1,042 tok/s |
| generation time for 15,000 papers | — | 2.9 h | 3.7 h |

### The judge on the test split (1,000 papers, 2026-10-17)

The first 1,000 papers of the test split in split order — the same papers for both runs — scored by
`qwen38-27b` against the teacher's summary in the same call. 999 of 1,000 parsed in each run.

| axis (1–5) | A ex70000 | A's teacher | B ex70000 | B's teacher |
|:--|--:|--:|--:|--:|
| faithfulness | **2.1602** | 2.1361 | **2.1862** | 2.0971 |
| coverage | 4.1421 | **4.3504** | 4.1962 | **4.2943** |
| specificity | 3.5155 | **3.9209** | 3.6356 | **3.8639** |
| fluency | 4.9259 | **4.9800** | 4.9540 | **4.9850** |
| **mean** | 3.6859 | **3.8468** | **3.7430** | 3.8101 |
| **Δ to teacher** | **−0.1609** | | **−0.0671** | |
| win / tie / loss | 0.153 / 0.498 / 0.349 | | 0.167 / 0.582 / 0.251 | |

Three things the axes say that the single delta hides:

1. **Both students are judged *more* faithful than the teacher they were distilled from** (A
   +0.024, B +0.089), and they lose on coverage (−0.208, −0.098) and specificity (−0.405, −0.228).
   The students write shorter, safer summaries; what they give up is detail, not accuracy.
2. The whole deficit is specificity and coverage. Fluency is saturated for everyone (≥ 4.93).
3. **B is closer to its teacher on every axis**, and ties more often (0.582 vs 0.498).

The test-split deltas are smaller than the validation deltas for both runs (A −0.161 vs −0.170,
B −0.067 vs −0.098), so the ordering is the same on held-out data but the gap to the teacher is
slightly narrower there. Faithfulness scores near 2.1 for *every* summary, including the teacher's,
are an artifact of the instrument: the judge sees a ~9,000-character excerpt and cannot verify a
claim drawn from the rest of the paper (`07_threats_to_validity.md`, T3).

### By how the body was fitted (test split, ROUGE-L)

Grouped by `trunc_kind`, which records whether the fitting loop had to run — **not** by whether the
whole body reached the prompt. 61.41% of Run A's test papers carry the complete body, but 72.37%
are labelled `none`; the difference is papers clipped by the initial character cap on the first
attempt, which sit in the `none` bucket (see `01_task_and_data.md`).

| `trunc_kind` | A ex70000 | B ex70000 |
|:--|--:|--:|
| `none` (no shrink round) | 0.4592 (n=10,856) | **0.4773** (n=12,356) |
| `sections` (sections dropped) | 0.4014 (n=4,143) | **0.4311** (n=2,643) |
| `hard` (head-and-tail cut) | 0.3832 (n=1) | 0.3832 (n=1) |

The `hard` bucket is a single paper in both runs and carries no statistical weight. These bucket
means are **observational**: the `none` bucket is contaminated with quietly clipped papers and the
groups differ in paper length, so the gap is not the cost of truncation. The controlled estimate,
about 0.03, is in `06_findings.md` (F4).

### By teacher model (test split, ROUGE-L, Run A)

| teacher | papers | ROUGE-L |
|:--|--:|--:|
| `deepseek/deepseek-v4-flash-0731` | 13,181 | 0.4501 |
| `deepseek/deepseek-v4.1-flash` | 1,209 | 0.4247 |
| `meta/muse-spark-1.3-contributor` | 610 | 0.3296 |

## Held-out loss, converted to nats per paper

Same 200 papers, same reference strings; per-token values are not comparable across vocabularies.

| papers seen | A per token | A per paper | B per token | B per paper |
|--:|--:|--:|--:|--:|
| 0 | 1.3224 | 1,322.9 | 0.6987 | 643.8 |
| 1,000 | 0.7028 | 703.0 | 0.6030 | 555.5 |
| 5,000 | 0.6668 | 667.0 | 0.5852 | 539.2 |
| 10,000 | 0.6532 | 653.4 | 0.5783 | 532.8 |

## Cost

| | A | B |
|:--|--:|--:|
| training time (2 × H200, stalls excluded) | 13.75 h | 16.63 h |
| training time per 1,000 papers | 0.196 h | 0.242 h |
| median tokens/s per rank | 7,513.9 | 5,949.6 |
| peak VRAM per GPU | 16.48 GiB | 29.63 GiB |
| tokens forwarded | 728.5 M | 700.8 M |
| optimiser steps skipped (non-finite gradient) | 0 | 0 |


---

## Student C — `Qwen/Qwen3.5-9B`, 20,000 papers, 28,672-token budget

Evaluated 2026-10-19 after the serving fix (`09_experiment_log.md`); all numbers are from the
text-only copy of the pinned checkpoint with the adapter verified applied. `json_ok` was re-scored
on CPU after a scorer fix (a template prefix beginning with a newline never matched); no text was
regenerated.

### Validation curve (1,000 papers)

| papers seen | JSON ok | schema ok | ROUGE-L | BERTScore | keyword F1 | NLI faith. | judge Δ | tie |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 (base) | 0.000 | 0.263 | 0.0774 | −0.347 | 0.115 | 0.123 | −2.111 | 0.031 |
| 500 | 0.997 | 0.997 | 0.4502 | 0.4231 | 0.555 | 0.454 | −0.0988 | 0.462 |
| 1,000 | 0.995 | 0.995 | 0.4597 | 0.4375 | 0.574 | 0.458 | −0.1020 | 0.576 |
| 2,500 | 0.997 | 0.997 | 0.4682 | 0.4441 | 0.582 | 0.452 | −0.0993 | 0.587 |
| 5,000 | 0.993 | 0.993 | 0.4664 | 0.4424 | 0.582 | 0.436 | −0.0648 | 0.617 |
| 10,000 | 0.996 | 0.996 | 0.4634 | 0.4460 | 0.580 | 0.482 | −0.0617 | 0.591 |
| 20,000 | 0.997 | 0.997 | **0.4790** | 0.4553 | 0.604 | 0.463 | −0.0745 | 0.591 |

The untrained model is a reasoning model: it writes a thinking trace to the 3,000-token cap
(stopped on EOS in 13% of papers, median output 2,930 tokens). After 500 papers it writes the
schema 99.7% of the time. The curve rises to 2,500 papers, is flat through 10,000, and the last
step (+0.0156) coincides with the learning-rate decay, as in A and B (F2).

### Test split (15,000 papers) — all three students

| metric | A ex70000 | B ex70000 | C ex20000 |
|:--|--:|--:|--:|
| training papers / hours / GPUs | 70,000 / 13.75 / 2×H200 | 70,000 / 16.63 / 2×H200 | 20,000 / 16.99 / 2×A100 |
| clean JSON | 0.9845 | 0.9915 | **0.9989** |
| ROUGE-1 / ROUGE-2 | 0.7043 / 0.4020 | 0.7255 / 0.4285 | **0.7351 / 0.4381** |
| ROUGE-L | 0.4432 | 0.4691 | **0.4789** |
| BERTScore | 0.4173 | 0.4424 | **0.4580** |
| keyword F1 | 0.5637 | 0.5886 | **0.6022** |
| domain exact | 0.1877 | 0.1862 | **0.1951** |
| NLI faithfulness (300 papers) | 0.4520 | 0.4184 | 0.4695 |
| stopped on EOS | 0.9849 | 0.9916 | **0.9990** |
| length vs reference | 0.958 | 0.988 | 0.988 |
| distinct-4 | 0.9071 | 0.9166 | 0.9207 |
| generation throughput (tok/s) | 1,407 (1×H200) | 1,042 (1×H200) | 528 (1×A100) |

### The judge on the test split (1,000 papers, student / teacher)

| axis | A | B | C |
|:--|--:|--:|--:|
| faithfulness | 2.160 / 2.136 | 2.186 / 2.097 | 2.172 / 2.100 |
| coverage | 4.142 / 4.350 | 4.196 / 4.294 | 4.220 / 4.328 |
| specificity | 3.515 / 3.921 | 3.636 / 3.864 | 3.654 / 3.908 |
| fluency | 4.926 / 4.980 | 4.954 / 4.985 | 4.983 / 4.983 |
| **Δ to teacher** | −0.1609 | **−0.0671** | −0.0726 |
| win / tie / loss | 0.153 / 0.497 / 0.349 | 0.167 / 0.582 / 0.251 | 0.151 / 0.582 / 0.267 |

C beats B on every reference metric but not on the judge, where the two are within 0.006 and C wins
less often. The judge reads a 9,000-character excerpt, so it cannot reward the extra context C
saw (T3).

### Run C against A and B by how much of the paper was visible (test split, ROUGE-L, paired)

See F4 for the table and its interpretation: C − B is −0.003 where both saw the whole body and
+0.028 where only C did.


---

## Fact-level evaluation without the teacher as reference (2026-10-25)

300 test papers (stratified by archive; 30 random-cohort) drawn from the first 3,000 test papers, so
the untrained base models have outputs too. Ten questions per paper were written by `gemma4-31b-it`
from the **paper's full text** (never from a summary): 1 problem, 1 motivation, 2 approach, 2 setup,
2 finding, 1 limitation, 1 scope. `qwen38-27b` answered each question from the summary alone (NOT
STATED allowed) and graded each answer against the gold. 299 papers have questions; one
(`1512.07248`) yielded no parseable reply in three tries and is excluded. Recall counts a partial as
half. Method and caveats: `03_evaluation_protocol.md`, T16.

| system | usable summary | correct | partial | wrong | not stated | **recall** | accuracy when answering |
|:--|--:|--:|--:|--:|--:|--:|--:|
| teacher | 1.000 | 0.509 | 0.218 | 0.042 | 0.231 | **0.618** | 0.662 |
| A  Qwen3-4B, 70k | 0.987 | 0.422 | 0.210 | 0.057 | 0.312 | 0.527 | 0.613 |
| B  K2-Horizon-7B, 70k | 0.997 | 0.466 | 0.212 | 0.053 | 0.269 | 0.572 | 0.637 |
| C  Qwen3.5-9B, 20k | 1.000 | 0.463 | 0.222 | 0.047 | 0.268 | 0.574 | 0.633 |
| A untrained | 1.000 | 0.310 | 0.211 | 0.047 | 0.432 | 0.416 | 0.546 |
| C untrained | 0.258 | 0.064 | 0.045 | 0.007 | 0.884 | 0.086 | 0.550 |

Paired differences in per-paper recall (bootstrap over papers, 95% interval):

| comparison | difference |
|:--|--:|
| A − teacher | −0.091 [−0.108, −0.076] |
| B − teacher | −0.046 [−0.062, −0.031] |
| C − teacher | −0.044 [−0.058, −0.030] |
| A − A untrained | **+0.111** [+0.095, +0.128] |
| B − A | +0.045 [+0.031, +0.059] |
| C − A | +0.047 [+0.032, +0.064] |
| **C − B** | **+0.002 [−0.013, +0.017]** |

Recall by question type:

| | problem | motivation | approach | setup | finding | limitation | scope |
|:--|--:|--:|--:|--:|--:|--:|--:|
| teacher | 0.522 | 0.187 | 0.696 | 0.672 | 0.806 | 0.642 | 0.483 |
| A | 0.465 | 0.130 | 0.606 | 0.583 | 0.688 | 0.518 | 0.400 |
| B | 0.492 | 0.152 | 0.661 | 0.594 | 0.753 | 0.575 | 0.483 |
| C | 0.485 | 0.155 | 0.665 | 0.612 | 0.762 | 0.602 | 0.421 |
| A untrained | 0.452 | 0.127 | 0.542 | 0.288 | 0.606 | 0.378 | 0.326 |

By cohort (top-cited n = 269, random n = 30): teacher 0.616 / 0.635, A 0.524 / 0.555, B 0.572 / 0.572,
C 0.571 / 0.605. With 30 random papers the interval is about ±0.05, so no drop is visible and none
is excluded.

**F4 on this measure** (fact recall; n = papers where one student cut text and the other did not):

| papers | n | first student | second | second − first |
|:--|--:|--:|--:|--:|
| both read B's and C's whole body | 199 | B 0.622 | C 0.611 | −0.011 [−0.026, +0.005] |
| B cut, C whole | 82 | B 0.500 | C 0.516 | +0.016 [−0.016, +0.045] |
| both read A's and C's whole body | 195 | A 0.583 | C 0.613 | +0.029 [+0.012, +0.045] |
| A cut, C whole | 86 | A 0.434 | C 0.517 | +0.083 [+0.049, +0.120] |

Differences of differences: +0.027 against B (interval includes zero), +0.054 against A.

## ACLSum: human-written aspect summaries (100 NLP papers, 2026-10-25)

`sobamchan/aclsum`, abstractive test split. Mapping: challenge → problem, approach → method, outcome →
results. Students read the dataset's `document` (median 5,756 characters: abstract, introduction and
conclusion) in their own training prompt format. Coverage is an LLM grade (covers 1, partly ½,
misses 0) of whether the field conveys the human sentence's main point.

| system | mean coverage | challenge | approach | outcome | words per field |
|:--|--:|--:|--:|--:|--:|
| A untrained | 0.852 | 0.730 | 0.925 | 0.900 | 29 / 48 / 36 |
| A | 0.883 | 0.785 | 0.955 | 0.910 | 46 / 89 / 62 |
| B | 0.903 | 0.815 | 0.965 | 0.930 | 48 / 82 / 54 |
| C | 0.915 | 0.835 | 0.990 | 0.920 | 47 / 80 / 57 |

Paired coverage differences: C − A +0.032 [+0.002, +0.063]; B − A +0.020 [−0.010, +0.052]; C − B
+0.012 [−0.013, +0.037]; C − untrained +0.063 [+0.028, +0.097]; A − untrained +0.032 [0.000, +0.060].
ROUGE-L against the one-sentence references is 0.17–0.28 and **ranks the untrained model above the
trained one on two aspects** (challenge 0.245 vs 0.234, approach 0.217 vs 0.171), because it is
shorter: the metric is dominated by the length mismatch here and is not used.


### Validity controls for the fact test (2026-11-02)

Three "systems" that say what the test measures, run on the same 299 papers and graded the same way.
`closed_book`: no summary at all and guessing allowed. `abstract`: the paper's abstract as the summary.
`full_paper`: the whole text as the summary, the ceiling set by gold and grader noise.

| system | correct | partial | wrong | not stated | **recall** | accuracy when answering |
|:--|--:|--:|--:|--:|--:|--:|
| closed book (no summary, may guess) | 0.148 | 0.318 | 0.513 | 0.020 | 0.307 | 0.151 |
| abstract only | 0.138 | 0.097 | 0.020 | 0.746 | 0.186 | 0.540 |
| untrained 4B | 0.310 | 0.211 | 0.047 | 0.432 | 0.416 | 0.546 |
| A, B, C | 0.42 – 0.47 | 0.21 – 0.22 | 0.047 – 0.057 | 0.27 – 0.31 | 0.527 – 0.574 | 0.61 – 0.64 |
| teacher | 0.509 | 0.218 | 0.042 | 0.231 | 0.618 | 0.662 |
| **full paper** (ceiling) | 0.797 | 0.174 | 0.023 | 0.005 | **0.884** | 0.801 |

Recall by question type, closed book / abstract / teacher / C / full paper: motivation 0.445 / 0.136 /
0.187 / 0.155 / 0.828; problem 0.452 / 0.254 / 0.522 / 0.485 / 0.890; setup 0.244 / 0.121 / 0.672 /
0.612 / 0.911; finding 0.202 / 0.245 / 0.806 / 0.762 / 0.911.

**Second grader.** `gemma4-31b-it` re-graded the attempted answers of three systems. Agreement on the
three-way verdict: teacher 85.7% (kappa 0.668, 2,300 items), C 85.1% (kappa 0.678, 2,189), A 85.2%
(kappa 0.697, 2,058).
