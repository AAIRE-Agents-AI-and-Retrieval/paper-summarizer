# Findings

Each claim is stated with the evidence that supports it, how strong that evidence is, and what
would overturn it. Numbers for F1–F7 are from `05_results.md`; F8–F13 take theirs from
`11_reward_selected_distillation.md` and `12_reactive_training.md`, each with its source named.

---

## F1 — Structured-output compliance is fixed by about 1,000 training papers

**Claim.** The base models' failure to emit the required schema is an adaptation problem, not a
capability problem, and it is solved by the first ~1,000 examples.

**Evidence.** Schema completeness (all eight fields, correct types) on the 1,000-paper validation
subset: Run A 0.672 → 0.993 after 1,000 papers; Run B 0.645 → 0.992. On the test split, Run A goes
0.668 untrained (3,000 papers, the base floor) → 0.9845 trained (15,000 papers). Neither run
improves on this after 1,000 papers
(A: 0.983–0.993 across all later milestones; B: 0.979–0.996).

**Strength.** Strong. Two model families, consistent on validation and on an untouched test split.

**Nuance that matters.** For Run A the base model's JSON was already *valid* 99.7% of the time; it
was dropping or mistyping individual fields in a third of replies. For Run B the untrained model is
a reasoning model and produced almost no parseable JSON at all (0.000 clean), so its base row
measures a different failure.

---

## F2 — The data-scaling curve is nearly exhausted by 5,000–20,000 papers

**Claim.** Beyond roughly 20,000 teacher summaries, additional data buys very little, and this
holds across two unrelated students.

**Evidence.** ROUGE-L on validation:

| papers seen | 1,000 | 5,000 | 10,000 | 20,000 | 50,000 | 70,000 |
|:--|--:|--:|--:|--:|--:|--:|
| Run A | 0.4138 | 0.4204 | 0.4200 | 0.4315 | 0.4341 | 0.4448 |
| Run B | 0.4474 | 0.4569 | 0.4600 | 0.4583 | 0.4584 | 0.4683 |

For Run A, 74% of the whole base→70,000 gain (+0.1177) is delivered by the first 1,000 papers
(+0.0867); 1,000→20,000 adds +0.0177 and 20,000→70,000 adds +0.0133. For Run B, 1,000→20,000 adds
+0.0109 and 20,000→70,000 adds +0.0100. The independent judge shows no improvement after 5,000
papers in either run (A: −0.169, −0.203, −0.163, −0.166, −0.170; B: −0.096, −0.095, −0.101,
−0.113, −0.098).

**Strength.** Moderate-to-strong for the *shape*; two students, two instruments, same knee.

**A third curve agrees.** Run C: 0.4502 (500 papers) → 0.4682 (2,500) → 0.4634 (10,000) → 0.4790 (20,000), flat for 7,500 papers and then a +0.016 jump at the decay. That the jump appears at the decayed point in all three runs supports reading it as schedule, not data.

**Caveats.** (i) The 70,000 point in both runs receives the warmup-stable-decay learning-rate
decay, so its gain of ~+0.010–0.013 is confounded with the schedule and is not purely data.
(ii) Differences of this size are near the run-to-run noise of greedy decoding (±0.04 per paper,
less when averaged over 1,000). (iii) Single seed per run: no variance estimate.

---

## F3 — The larger student is better on every reference metric, and reaches the smaller one's
ceiling with 1.4% of the data

**Claim.** At equal data, prompt budget and recipe, K2-Horizon-7B (9.17 B parameters) outperforms
Qwen3-4B (4.15 B) on every overlap and preference metric.

**Evidence.** Test split, 15,000 papers: ROUGE-L 0.4691 vs 0.4432, BERTScore 0.4424 vs 0.4173,
keyword F1 0.5886 vs 0.5637, length ratio 0.988 vs 0.958, clean JSON 0.9915 vs 0.9845. Judge gap to
the teacher at the final checkpoint: −0.098 vs −0.170 on validation and −0.067 vs −0.161 on the
test split, with ties at 0.58 vs 0.50 and losses at 0.25 vs 0.35 (test). Run B at **1,000** training papers (ROUGE-L 0.4474) already exceeds Run A at **70,000**
(0.4448). Held-out loss in nats per paper at matched points: 555.5 vs 703.0 (1,000 papers), 532.8
vs 653.4 (10,000).

**Strength.** Strong on overlap metrics (single seed, but the margin is large and consistent across
all milestones and the test split).

**Caveats.** (i) Cross-model judge deltas are affected by anchoring: the same teacher summaries
score 3.83 beside B and 3.87–3.92 beside A. (ii) Run B's tokenizer is better (3.28 vs 3.09
characters per token), which could have let it read more of each paper. **Measured, it barely
does** (test split): 61.4% of papers fit whole for A and 63.7% for B, and only 338 papers (2.3%)
were cut for A but whole for B. On those B − A is +0.037 against +0.024 on papers both read whole,
an extra +0.013 on 2.3% of papers, i.e. about **0.0003 of B's 0.026 margin**. The earlier
statement that "part of the margin is input, not model" overstated it and is withdrawn.
(iii) Serving cost rises: 1,042 vs 1,407 output tokens/s, i.e. 3.7 h
against 2.9 h to summarise 15,000 papers.

---

## F4 — Seeing more of the paper helps, by about 0.03 ROUGE-L on the papers it reaches

**Claim.** Truncating the paper costs quality, but the cost is roughly **0.03 ROUGE-L, less than half
the 0.07 raw gap** between papers that fit whole and papers that did not. The rest of the raw gap is
the papers being different, not the model seeing less.

**Why the raw gap is not the answer.** Within one student, papers whose whole body fitted score
0.07 ROUGE-L above papers that lost text (test split, `fits`): A 0.4717 vs 0.3978, B 0.4950 vs
0.4238, C 0.4847 vs 0.4188. But long papers are different papers, and the gap *persists for Run C*
(+0.066) although its 28,672-token window put 91% of papers in the "fits" group. A gap that does not
close when the budget grows is mostly difficulty.

**The controlled test.** Run C changes two things against B at once (student and budget), so the
test is a difference of differences, on the same 15,000 test papers, paired by paper
(`scripts/truncation_test.py`, ROUGE-L, 95% intervals on the paired mean):

| papers | n | B | C | C − B |
|:--|--:|--:|--:|--:|
| both saw the whole body | 9,550 | 0.4950 | 0.4923 | **−0.0028** ± 0.0009 |
| **B lost text, C saw the whole body** | 4,133 | 0.4388 | 0.4671 | **+0.0283** ± 0.0023 |
| both lost text (C lost less) | 1,317 | 0.3765 | 0.4188 | +0.0423 ± 0.0052 |

Where the larger window changes nothing, C is no better than B (slightly worse). Where it brought
the whole paper into view, C is better by 0.028. The same contrast against A: +0.021 where both saw
everything, +0.057 where only C did, so a difference of differences of **+0.035**. Two students, two
estimates, 0.03–0.035.

**Strength.** Moderate-to-strong for the direction and the order of magnitude: the effect appears
exactly on the papers the larger window reaches, and not on the others, which a better model alone
would not produce. The "both lost text" row is not a clean group (C cut fewer sections than B) and
is not used as evidence.

**Caveats.** (i) Single seed per run. (ii) C was trained on 20,000 papers and B on 70,000, so a
"better model" explanation would have to be hidden in the both-whole row, where C ≈ B. (iii) Paired
intervals understate run-to-run variation. (iv) The judge does not confirm it: it reads a 9,000-
character excerpt, so it cannot reward context it never sees (C's judge gap to the teacher is
−0.073 against B's −0.067, T3).

**Earlier statements withdrawn.** F4 previously said the none/sections gap was "a lower bound on the
cost of losing text" because the `none` bucket contained clipped papers. The controlled test says
the opposite: the raw gap *overstates* the cost, by about a factor of two.

**The `hard` bucket is not evidence.** It holds a single paper per split; quoted figures for it
were withdrawn on 2026-10-17.

---

## F5 — Fine-tuning does not improve faithfulness in either student

**Claim.** Distillation buys agreement with the teacher's wording, not groundedness in the paper.

**Evidence.** NLI faithfulness (300 papers per point): Run A 0.4741 untrained, 0.4401–0.4901 across
six milestones, 0.4520 on the test split, with the untrained base scoring 0.4671 on the same test
papers. Run B 0.4105–0.4530 across milestones, 0.4184 on the test split. No trend in either run,
and **B is slightly less faithful than A despite winning every overlap metric**.

**Strength.** Moderate. The metric is relative, computed on 300 papers per point, and its absolute
level is depressed by retrieval misses. The direction (no improvement) is consistent across twelve
checkpoints and two models, which is the part we rely on.

**The judge disagrees about the level, not the trend.** On the test split it rates both students
*more* faithful than their teacher (A 2.160 vs 2.136, B 2.186 vs 2.097), while the NLI metric puts
them at 0.45 and 0.42. The two instruments measure different things — the judge reads a 9,000-
character excerpt and rewards caution, the NLI metric scores every sentence against retrieved
evidence — and neither shows faithfulness improving with data.

**Why it matters.** The two objectives are separable: more data and a larger model move
teacher-similarity and leave grounding where it was. Any future work aiming at faithfulness has to
change the evidence available to the model or the objective, not the data volume.

---

## F6 — At one sequence per forward, doubling the parameters costs 24% more training time

**Claim.** Small models underuse a modern accelerator in this regime, so parameter count is a poor
proxy for training cost.

**Evidence.** Identical hardware (2 × H200), identical recipe: Run A 7,513.9 tokens/s per rank,
16.48 GiB, 13.75 h of training for 70,000 papers (0.196 h per 1,000); Run B, with 2.21× the parameters,
5,949.6 tokens/s per rank, 29.63 GiB, 16.63 h (0.242 h per 1,000). The ratio of parameters is
2.21×; the ratio of training time is 1.24×.

**Strength.** Strong for this configuration (batch of one variable-length sequence, LoRA,
gradient checkpointing, SDPA). It will not transfer to regimes with large rectangular batches.

**Consequence.** Serving, not training, is where the larger student is expensive: 1,042 against
1,407 output tokens/s under vLLM with continuous batching.

---

## F7 — On an independent fact test, B and C tie, both sit 0.045 below the teacher, and ROUGE overstated C's lead

**Claim.** Measured by what a reader can learn from the summary (questions written from the paper,
not from any summary), K2-Horizon (B) and Qwen3.5-9B (C) are indistinguishable and both better than
Qwen3-4B (A); none reaches the teacher. Training adds a large amount over the untrained 4B model.

**Method.** `gemma4-31b-it` writes ten specific questions per paper from the full text; `qwen38-27b`
answers them from one summary at a time (NOT STATED allowed) and grades the answer against the gold.
Two families write and mark the test; the teacher is a system under test, not the reference. 299
papers.

**Evidence.** Recall (correct + ½ partial): teacher 0.618, B 0.572, C 0.574, A 0.527, A untrained
0.416. C − B = +0.002 [−0.013, +0.017]; B − A = +0.045 [+0.031, +0.059]; A − untrained = +0.111
[+0.095, +0.128]. On ROUGE-L the same three are 0.4691, 0.4789, 0.4432: C ahead of B by 0.010, a gap
the fact measure does not show. The judge (−0.067 vs −0.073) agreed with the fact measure and not
with ROUGE.

**Errors are rare; omissions are not.** Wrong answers: teacher 4.2%, A 5.7%, B 5.3%, C 4.7%. Not
stated: teacher 23.1%, A 31.2%, B 26.9%, C 26.8%. The students trail the teacher mostly by leaving
facts out. This independently supports F5 ("give up detail, not accuracy") without using the judge.

**What the summaries cannot answer.** Recall on *motivation* ("why does this matter?") is 0.13–0.19
for every system including the teacher, and on *scope* 0.40–0.48. The output schema has no field for
significance or conditions, so no amount of training on this schema will fix these; a design
consequence, not a student failure.

**F4 on this measure.** The direction reproduces against A (C − A +0.083 where A cut text vs +0.029
where neither did; n = 86 and 195) and is not resolved against B (+0.016 vs −0.011; n = 82, interval
includes zero). C's larger window does not show up as extra fact recall over B (+0.002), but C was
trained on 29% as much data, so the two may offset.

**What the controls say about the test** (`05_results.md`). With *no summary* the answerer still earns
0.307 recall, but almost all of it is partial credit for plausible guesses (correct 0.148, wrong 0.513);
the abstract alone gives 0.186; the students 0.527–0.574; the teacher 0.618; the full paper 0.884. So the
summaries add a lot over the abstract and over guessing, the full-paper ceiling means about 12% of
items are noisy or ambiguous, and **partial credit is gameable** (on "why does it matter" questions
a guessing answerer scores 0.445, more than the teacher's summary at 0.187). Strict scoring, correct
only, separates better: closed book 0.148, abstract 0.138, students 0.42–0.47, teacher 0.509, paper 0.797.

**Strength.** Moderate. Intervals are paired over papers; the ordering matches the judge. **Caveats:**
(i) the gold answers are one model's reading of the paper and have not been verified by a person
(`scisum factcheck export` writes a 40-paper checklist locally; it contains paper excerpts and is not distributed); (ii) the answerer/grader is the same model
as the earlier judge; (iii) partial = ½ is a convention; (iv) ten questions per paper cannot cover a
paper, so absolute recall means little and the ranking is the result; (v) 30 random-cohort papers;
(vi) single run of each generator.

## F8 — Optimising facts directly moves fact recall past the teacher's; selecting better targets barely moves it

*Read with F9 and F10: most of round 1's +0.071 came from writing longer. With length controlled the gain is
+0.017 to +0.030 (F9), and the recommended arm `prefcc` is level with the teacher, not above it (F10).*

**Claim.** Continuing Student C with preference tuning on reward-ranked pairs of its own summaries
raises fact recall by 0.07, to above the teacher's. Continuing it on reward-selected targets, or on
the teacher's own targets, does not move recall by a useful amount. The preference gain comes with
longer, less teacher-like summaries that make about 0.4 more unsupported statements per summary
(of about 26).

**Method** (`11_reward_selected_distillation.md`). 1,520 papers from the validation split's pool, which the policy
never trained on. C wrote 5 samples
each; `qwen38-27b` wrote 14 questions per paper from the full text, and `gemma4-31b-it` answered them
from each summary and graded the answers. The reward was fact recall + 0.15·ROUGE-L, minus penalties
for unsupported numbers and for length beyond 1.2× the teacher's. Three arms, one pass each, from C's
adapter: control (teacher's targets), selection (the best summary per paper; a sample replaced the
teacher in 486), preference (DPO, β = 0.1, + the likelihood of the chosen summary, on 1,073
best-vs-worst pairs). Evaluated on the fact test of F7 with the roles of the two models swapped: 899
test papers (two disjoint question sets). Also 1,246 test papers for reference metrics and NLI, and
the judge on the first 500.

**Evidence.** Fact recall: teacher 0.608, C 0.571, control 0.568, selection 0.575, **preference
0.642**. Preference − C +0.071 [+0.064, +0.079] (both question sets; better in 65% of papers, worse
in 16%); preference − teacher +0.034 [+0.025, +0.042]. Strict scoring (fully correct only) rises more
than recall, +0.075, so the gain is not partial credit. Of answered questions, 67.9% are fully
correct (C 64.4%) and 6.6% wrong (C 6.5%). Selection − control +0.007 [+0.000, +0.014], p = 0.04,
from one question set only. The judge rates the preference arm level with the teacher (+0.017, the
only student above −0.04), with higher coverage (+0.19) and specificity (+0.41).

**The cost.** 1.26× the teacher's length (C 0.99×), ROUGE-L −0.027, BERTScore −0.034, invalid JSON 10
of 1,246 (C 1). NLI faithfulness −0.056 [−0.067, −0.044]; the judge's faithfulness −0.16 [−0.22,
−0.11]. Both instruments see the paper partially: NLI checks single sentences against 3-sentence
chunks, and the preference arm writes 17% more sentences, 11% longer; the judge sees a 9,000-character
excerpt and gives the teacher 2.1 of 5. Under a linear model of the per-paper gain on the log length
ratio, +0.051 [+0.019, +0.081] of the recall gain remains at equal length. This extrapolates: only 27
papers are within 10% of C's length. Round 2 (F9) measured it directly, with arms writing at the
teacher's length: +0.017 to +0.030, so the extrapolation overstated the length-free gain by more than half.

**The claim check settles the faithfulness question (2026-11-07): the cost is real, and small.** Every
statement of each summary was checked against the whole paper text by an LLM (`scisum/claimcheck.py`,
300 fact-test papers, Qwen3.8-27B; it flags 98.7% of statements whose numbers were changed, 99.9% of
another paper's, and 1.1% of unchanged ones). Supported statements: teacher 97.2%, C 96.6%,
preference 95.5%. Per summary, the preference arm makes 1.15 unsupported statements (C 0.73, teacher
0.59), of which 0.68 contradict the paper (C 0.40, teacher 0.18). Preference − C: +0.45 [+0.31, +0.60]
unsupported and +0.28 [+0.15, +0.40] contradicted per summary; on the 272 papers whose text was not
cut at 90,000 characters, +0.38 and +0.28. A second verifier (Gemma-4-31B, 150 papers, 97.2% agreement
with the first on 10,300 statements, kappa 0.61) agrees on the direction: +0.31 [+0.11, +0.52]
unsupported per summary. (Numbers corrected on 2026-11-07, 17:20, after a parser fix that recovered 18
replies with one malformed field; no conclusion changed. See `11_reward_selected_distillation.md`.) The
fact test missed it because its questions cover a dozen facts per paper, and the extra errors sit in
statements no question asks about.

**Strength.** Strong for "preference tuning raises fact recall": large, on both question sets, judged
by models that did not compute the reward, on held-out test papers. Strong for "and it makes more
unsupported statements": two verifiers from different families, on the whole paper text, with
controls. So the summaries are more informative and slightly less faithful, not simply better. Weak
for any difference between selection and control. **What would overturn it:** a human check of the
flagged statements (a blinded sample is in `paper_docs/samples/`, not distributed) finding the
verifiers' flags mostly wrong. **Caveats:** one round, one seed, one starting model, untuned hyperparameters. The fact
test and the reward are the same kind of instrument (questions written from the paper), with the
models' roles swapped, so a shared bias of question-based scoring would not be caught.

## F9 — With length controlled, preference tuning keeps a fact gain of +0.02 to +0.03, and no arm meets all four criteria

**Claim.** Round 1's preference gain was mostly length. Controlling length in the pairs or in the loss
brings the student back to the teacher's length and most of the way back to its wording, and keeps a
fact-recall gain over C of +0.017 to +0.030 (round 1: +0.071 at 1.26x the length). A smaller
faithfulness cost remains. No arm met all four criteria fixed before evaluation.

**Method** (`11_reward_selected_distillation.md`, "Round 2"). Three arms from C's adapter, one pass
each, beta 0.1, likelihood weight 1.0: `preflc` (1,163 pairs whose two summaries are of similar length,
reward charging length beyond 1.0x the teacher's), `prefrd` (round 1's 1,073 pairs with the R-DPO length
term, alpha 0.01), `preflcrd` (both). Evaluated like round 1, plus the claim check (300 papers,
Qwen3.8-27B). Criteria: recall above C's; length 0.90-1.10x; ROUGE-L and BERTScore no more than 0.01
below C's; unsupported-or-contradicted share not above C's.

**Evidence.**

| | C | round 1 | `preflc` | `prefrd` | `preflcrd` |
|:--|--:|--:|--:|--:|--:|
| fact recall − C | | +0.071 | **+0.030 [+0.023, +0.037]** | +0.017 [+0.010, +0.024] | −0.013 [−0.020, −0.006] |
| length ratio | 0.99 | 1.26 | 1.01 | 0.94 | 0.84 |
| ROUGE-L / BERTScore − C | | −0.027 / −0.034 | −0.006 / −0.010 | −0.007 / −0.011 | −0.008 / −0.013 |
| unsupported (contradicted) per summary | 0.73 (0.40) | 1.15 (0.68) | 0.90 (0.54) | 0.80 (0.47) | 0.82 (0.51) |
| unsupported share − C | | +0.012 [+0.006, +0.018] | +0.009 [+0.003, +0.014] | +0.006 [−0.000, +0.011] | +0.011 [+0.005, +0.017] |
| criteria met (of 4) | | 1 | 2 (misses 3 by 0.00004, and 4) | 3 (misses 3 by 0.0006) | 0 |

Both recall gains hold on each question set and pass the Bonferroni threshold (p = 0.0001). The fact
gain at equal length estimated by regression is +0.019 [+0.011, +0.028] (`preflc`) and +0.021 [+0.014,
+0.028] (`prefrd`). The judge's mean score does not separate any round-2 arm from C.

**Reading.** The arms trade facts against unsupported statements along one line: `preflc` gains the
most and adds +0.18 unsupported statements per summary of about 21; `prefrd` gains half as much and adds
+0.07, within noise. Combining the two remedies overshoots the length (0.84x) and loses the fact gain.
The held-out loss on teacher summaries rose in every arm (0.450 → 0.476-0.481), so the drift from the
teacher's wording is a property of the preference objective, not of length.

**Strength.** Strong for "most of round 1's gain was length" and for a remaining gain of +0.02-0.03:
three arms, both question sets, permutation p = 0.0001. Weak for any ranking of `preflc` against
`prefrd` on overall quality: they differ in recall (+0.013 [+0.005, +0.020]) and in the opposite
direction on faithfulness, and both miss the BERTScore limit by less than its uncertainty.
**Caveats:** one seed, untuned alpha, one verifier for round 2's claim check. The claim check's parser
was corrected after the round-2 results were first read. The correction made every arm look worse
against C, and moved `preflcrd` from passing criterion 4 to failing it.
**Next** (all three were done in round 3, F10): the claim check inside the pair-ranking reward; a smaller
alpha with length-matched pairs; a second seed.

## F10 — With the claim check in the pair-ranking reward, the student gains 0.026 fact recall at C's length, with no more unsupported statements than C on the mean of two seeds, and meets all four criteria

**Claim.** Penalising unsupported statements in the reward that ranks the training pairs removes the
faithfulness cost that every earlier preference arm paid on the claim check, at no cost in recall.
On the mean of two pair orders `prefcc` recalls 0.026 [0.020, 0.032] more facts than C, at C's length
(0.988), and its unsupported share is not above C's (+0.0045 [−0.0003, +0.0093]). It is the first arm in
three rounds to meet all four criteria fixed before evaluation. Seed 1 alone recalls 0.033 more, level with
the teacher; seed 2 alone recalls 0.020 more and fails criterion 4. The evidence table below is round 3's
(seed 1); the second seed is in the update at the end of this finding. A second seed of round 2's two arms shows that pair order alone moves recall by
0.01-0.02 and length by 0.04-0.05.

**Method** (`11_reward_selected_distillation.md`, "Round 3"). Four arms from C's adapter, as round 2:
`prefcc` (`preflc`'s length-matched pairs, the reward minus 1.0 × the candidate's unsupported-or-contradicted
share by Gemma-4-31B; 1,233 pairs), `preflcrd3` (`preflc`'s pairs with the R-DPO term at alpha 0.003),
and second seeds of `preflc` and `prefrd` (same pairs, other order). The penalty mostly changes which
sample is rejected: the chosen summaries' unsupported share falls from 0.033 to 0.024, the rejected
ones' rises from 0.050 to 0.062. Evaluation as round 2, with the claim check by a different verifier
(Qwen3.8-27B). `preflc` and `prefrd` are judged on the per-paper mean of their two seeds.

**Evidence.**

| | C | `prefcc` | `preflcrd3` | `preflc` (2 seeds) | `prefrd` (2 seeds) |
|:--|--:|--:|--:|--:|--:|
| fact recall − C | | **+0.033 [+0.026, +0.040]** | +0.019 [+0.012, +0.027] | +0.023 [+0.017, +0.029] | +0.007 [+0.001, +0.013] |
| length ratio | 0.99 | 1.02 | 0.95 | 0.99 | 0.91 |
| ROUGE-L / BERTScore − C | | −0.005 / −0.007 | −0.005 / −0.008 | −0.006 / −0.012 | −0.008 / −0.013 |
| unsupported share − C | | **+0.000 [−0.005, +0.005]** | +0.009 [+0.003, +0.016] | +0.008 [+0.003, +0.013] | +0.010 [+0.005, +0.015] |
| NLI − C | | −0.025 [−0.035, −0.015] | −0.041 | −0.042 | −0.050 |
| criteria met (of 4) | | **4** | 3 (misses 4) | 2 (misses 3, 4) | 2 (misses 3, 4; recall p = 0.025) |

`prefcc` − teacher on recall: −0.005 [−0.013, +0.003]. Against `preflc`'s two seeds, `prefcc` has 0.007
[0.002, 0.012] less unsupported share, where the seeds differ by 0.0003, and +0.009 [+0.004, +0.015] more
recall. Its recall gain holds on each question set (A +0.041, B +0.029; p = 0.0001).

**Reading.** In rounds 1-2 every gain in facts came with more unsupported statements, and the arms lay
along one trade-off line. Putting the claim check into the ranking reward moves the student off that
line: the facts stay, the unsupported statements go back to C's level. Seed noise is as large as the
round-2 differences between arms, so single-seed rankings in round 2 were weaker than they looked; the
claim effect is the exception, stable between seeds and larger than their difference. Round 2's BERTScore
misses were not noise: over two seeds `preflc` and `prefrd` are 0.012 and 0.013 below C.

**Strength.** Moderate. Strong for the claim penalty lowering the unsupported share against `preflc`
(consistent against both seeds, NLI agrees in direction: +0.022 [+0.011, +0.033]). Moderate for
"meets all four criteria": it holds on the two-seed mean, but narrowly (BERTScore 0.0015 inside its limit,
the unsupported-share interval starts at −0.0003), and not for seed 2 alone. **Caveats:** NLI and the judge's faithfulness score are still below
C's for `prefcc` (−0.025; −0.090 [−0.146, −0.036] on a 1-5 scale); they are reported, not criteria. The
training penalty and the evaluation use the same kind of claim check with different models (agreement
97.1%, kappa 0.60), so part of the effect could be learning what such a check rewards.
**Next:** the second seed was run (update below). Still open: a check of `prefcc`'s summaries by an
independent method (human reading of a sample, or a third verifier).

**Update 2026-11-09 (second seed, `paper_docs/12_reactive_training.md` §8, question 1).** Over two
pair orders the claim holds by the protocol, but it is weaker:
- recall − C: +0.0262 [+0.0201, +0.0322], not +0.033;
- length ratio 0.988;
- unsupported share − C: +0.0045 [−0.0003, +0.0093], which passes criterion 4 narrowly.

The second seed alone has +0.0087 [+0.0024, +0.0149], so "without adding unsupported statements"
holds for the two-seed mean, not for each seed.

## F11 — Blending C's and prefcc's weights keeps most of the fact gain but does not close prefcc's NLI or judge deficit

Source: `paper_docs/12_reactive_training.md` §8, "Result, question 2"; `paper_docs/data/improve_r4.json`.
Exact rank-128 blends of the two LoRA adapters at w = 0.5 and 0.75 (no training), on the 1,246 test papers.

- Both blends meet all four criteria. Fact recall − C: +0.0201 [+0.0139, +0.0263] at w = 0.5 and
  +0.0286 [+0.0218, +0.0353] at w = 0.75, against prefcc's +0.0327. So they keep 61% and 87% of prefcc's
  gain, more than their weight.
- The judge's faithfulness deficit barely moves (−0.078 and −0.074 against prefcc's −0.090). The NLI
  deficit shrinks about in proportion to w. No blend − prefcc interval on NLI or the judge excludes
  zero, so by the pre-registered rule neither blend is better than prefcc.
- Against prefcc's two-seed mean (+0.0262 recall, +0.0045 unsupported share), `ccmix75` (+0.0286, +0.0041)
  lies between prefcc's two seeds on both.
- Scope: one prefcc seed, two blend weights, evaluation-side models only. The deficits look tied to any
  move toward prefcc, not only to its full weight. That is a reading of two points, not a measured curve.

## F12 — An LLM agent steering DPO during training did not beat fixed training on the same pairs

Source: `paper_docs/12_reactive_training.md` §8, "Result, question 3"; `paper_docs/data/improve_r4.json`;
`runs/q35prefccrx/agent/decisions.md`. One reactive run (`q35prefccrx`) on prefccs2's 1,233 pairs and pair order.
The agent could change the update rule, the LR, clipping, weight
averaging, β, the NLL weight and the length α. It made 8 decisions from 48-paper validation evaluations.

- Against its same-seed fixed baseline `prefccs2`: recall −0.0061 [−0.0126, +0.0006], unsupported share
  +0.0003 [−0.0055, +0.0065], NLI −0.0040, judge +0.004, ROUGE-L/BERTScore level, length −0.012. No measurable gain.
- Against C it meets criteria 1–3 (recall +0.0136 [+0.0069, +0.0203], p = 0.0002; length 0.947; BERTScore
  −0.0098, inside the limit by 0.0002). It fails criterion 4 (+0.0082 [+0.0018, +0.0146]), like `prefccs2` alone.
  So by the pre-registered rule the agent is not credited, not even as a candidate.
- The better blend `ccmix75` (no training) recalls 0.015 more [+0.0083, +0.0218].
- The agent was well calibrated: its four falsifiable test predictions held, and five of its seven in-run
  prediction checks held fully. Its two interventions (a length term, then a restore to the step-68 weights)
  cancelled out. The restore discarded ~16% of the pairs. Calibration did not turn into improvement, because the
  48-paper validation signal is about as noisy as the effects (one SE ≈ 0.02 reward).
- Scope: one run, one pair order. A second reactive seed was not run: the protocol asks for one only after a
  positive result.

## F13 — An LLM agent designing a round found no arm better than `prefcc`; a tripled claim penalty ties it

Source: `paper_docs/12_reactive_training.md` §9, "Result, steps 6–7"; `paper_docs/data/outer_r4/verdict.json`.
Round 4 ran as follows:
- An LLM designer proposed four arms and one blend after a blind audit
  of the pairs.
- It scored the arms on a 600-paper validation set V with the training-side reward.
- It named two finalists before any test result: `r4b`, with the claim penalty in the pair-ranking reward at 3
  instead of 1, and `r4a`, with only the pairs whose chosen summary is the teacher's (940 of 1,233).
- Each finalist was judged on the per-paper mean of two seeds, under rules fixed before the build.

**Claim.** The round is a null result. No designed arm replaces `prefcc`.

**Evidence.** Two-seed means; brackets are 95% intervals unless marked.

| | `r4b` | `r4a` |
|:--|:--|:--|
| recall − C | +0.0236 [+0.0177, +0.0296], p = 0.0001 | +0.0089 [+0.0035, +0.0149] |
| unsupported share − C | +0.0051 [−0.0000, +0.0102] | +0.0074 [+0.0022, +0.0123], fails criterion 4 |
| length ratio | 1.005 | 1.013 |
| ROUGE-L / BERTScore − C | −0.0057 / −0.0063 | +0.0005 / −0.0010 |
| criteria met | 4 of 4 | 3 of 4 |
| recall − `prefcc` | −0.0025 [−0.0074, +0.0024] | −0.0173 [−0.0226, −0.0119] |
| unsupported share − `prefcc` | +0.0007 [−0.0036, +0.0052] | +0.0026 [−0.0015, +0.0067] |
| NLI − `prefcc` | +0.0080; 98.75% [−0.0016, +0.0174] | −0.0080; 98.75% [−0.0182, +0.0018] |
| judge faithfulness − `prefcc` | +0.021; 98.75% [−0.019, +0.062] | +0.004; 98.75% [−0.044, +0.050] |

**Reading.**
- **`r4b` ties `prefcc`.** It meets all four criteria and is not worse on recall or the unsupported share. Its
  NLI is better at 95% but not at the fixed 98.75%.
  - Tripling the claim penalty lowered the training grader's claim share on V (−0.0057 against `prefcc`).
  - It did not lower the evaluation grader's unsupported share on test (+0.0007).
- **`r4a` loses two thirds of the fact gain.** Without the sample-chosen pairs, the recall gain falls from
  +0.0262 to +0.0089. So the pairs where a student sample beats the teacher carry most of `prefcc`'s gain.
  - `r4a` is also 0.016 below `r4b` on NLI [−0.024, −0.008].
  - This argues against the blind auditor's view that the sample-chosen claim contrast is not real.
- **The designer was calibrated but did not find a better arm.** 26 of 32 of its V predictions held, and 9 of 12
  of its test predictions did. It rated each finalist at about 0.1 to replace `prefcc`, and neither did.
  - Its V signal could not separate the arms: every arm passed V's proxy of the four criteria.
  - The validation set is known to track the test only weakly (`07` T21).

**Strength.** Moderate for the null result: two seeds per finalist, with rules and thresholds fixed before
training. Weak for any ranking of `r4b` against `prefcc`, which differ by less than their seed noise (recall
seeds differ by 0.010–0.013).

**What would overturn it.** A further seed of `r4b` on which its NLI advantage holds at the fixed level. This was
not run: round 4 is the project's last experiment.

**Future work (not run).**
- From the designer's review (`12` §9, step 8): more on-policy pairs per paper at held length, and iterated DPO
  from `prefcc`'s own new samples.
- An NLI- or human-checked faithfulness term in the reward.
- An evaluation grader independent of the training grader.
