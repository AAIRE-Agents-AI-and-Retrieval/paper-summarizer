# Reward-selected distillation and preference tuning (Student C, rounds 1–3)

*Status 2026-11-16: rounds 1–3 are complete; `prefcc` (round 3) is the recommended arm on the mean of two
seeds (updates under "Round 3 results"). The follow-ups and round 4 are in `12_reactive_training.md`. The
status paragraph below is the original one of 2026-11-05.*

Status: round 1 complete (2026-11-05, 02:15 local). All three arms trained and evaluated,
including the judge. Results are under "Results"; the sections before them were written
before any arm was trained. Round 2 (length-controlled preference) and the claim check against the
whole paper started on 2026-11-07; their protocol, under "Round 2", was written before any round-2 arm
was evaluated.

## The question

Plain distillation maximises the likelihood of the teacher's text. The fact test showed what that
leaves: the students trail the teacher by *omission* (26.8% of questions unanswered against 23.1%) and
the fabrications that occur are invented numbers (F5, F7). The aim of this round is a training
objective that keeps imitating the teacher **and** is rewarded for being right about the paper, with
the teacher as an anchor and not as the only source of signal.

## What was built (`scisum/improve.py`, `scisum/pref_train.py`)

One round of expert iteration, with the policy being improved fixed at Student C's final adapter:

1. **Pool.** 3,000 validation papers the policy never trained on and that are not in the monitoring
   subset, with the teacher's summary. The test split is untouched.
2. **Candidates.** Per paper: the teacher's summary and 5 samples (temperature 0.8, top-p 0.95).
   *Corrected 2026-11-16:* this line said "the policy's greedy output and 6 samples". That was the setting
   of the first job only. `state/improve/samples.jsonl.gz` holds greedy + 6 samples for the first 120 of
   the 1,520 papers and 5 samples, no greedy output, for the other 1,400. The table below ("5 per paper")
   was right.
3. **Questions.** `qwen38-27b` writes 14 specific questions with gold answers from the paper's full
   text. The evaluation's questions are written by `gemma4-31b-it`, so the instruments differ.
4. **Which questions count.** Two reference readers (`gemma4-31b-it`, which also grades; the
   evaluation's grader is `qwen38-27b`): one sees the whole paper, one sees nothing. A question is
   kept only if the whole-paper reader answers it correctly (so it is valid and answerable) and the
   closed-book reader does not (so it does not leak its own answer). Fewer than 5 kept questions:
   the paper keeps its teacher target and contributes nothing to selection.
5. **Reward** for every candidate:

       R = fact + 0.15 * ROUGE-L(candidate, teacher) - 0.30 * unsupported_numbers - 1.0 * max(0, length_ratio - 1.2)

   * `fact`: mean over kept questions of +1 correct, +0.25 partial, 0 not stated, **-1 wrong**.
     A wrong answer costs more than a missing one, so hedging is free and fabricating is not.
   * `ROUGE-L` to the teacher is the imitation term. It is 1 for the teacher itself, so the teacher
     is replaced only by a sample that is more right by more than it is less like the teacher.
   * `unsupported_numbers`: the share of the summary's numbers (two or more digits, not years) that
     occur nowhere in the paper. A model-free check aimed at exactly the failure the fact test found.
   * Length beyond 1.2x the teacher's is charged, since a longer summary answers more questions.
   * A candidate that is not valid JSON of the schema, or that hit the token limit, is never chosen.
6. **Three training arms from the same pool, same steps, same learning rate (3e-5 SFT):**

   | arm | target per paper | objective |
   |:--|:--|:--|
   | `ctl` (control) | the teacher's summary | SFT |
   | `rft` | the highest-reward candidate; a sample replaces the teacher only if it beats it by 0.05 | SFT |
   | `pref` | chosen = highest-reward candidate, rejected = lowest-reward valid *student* sample, kept only if the gap is >= 0.25 | DPO (beta 0.1) against the starting adapter + 1.0 x chosen-summary NLL |

   The control matters: the arms are continued training on new papers, so any gain over the
   starting model could be extra training and not the method. `ctl` and `rft` see the same papers in
   the same order; only the targets differ.

## Why these choices

* **Offline rounds, not online RL.** The reward needs about 20 model calls per paper on a shared
  endpoint (about 60,000 calls for 3,000 papers); doing that inside an optimiser step is not
  feasible, and a round of sampling, scoring and training is the standard cheap version.
* **Different models write, answer and grade here than in the evaluation.** Otherwise the student is
  tuned to the instrument that scores it. It is still a model's reading of the paper (threat T16).
* **The teacher is a candidate, not the reference.** It competes on the same reward as the samples.
* **DPO with the starting adapter as reference, plus a likelihood term.** The reference is the model
  being improved, not the base model. The likelihood term keeps the imitation (when the chosen
  summary is the teacher's it is plain distillation of that paper) and counters DPO's known drift
  of lowering the likelihood of both members of a pair. Reference log-probabilities are computed once
  before the first update and stored in `runs/<run>/ref_logps.npy`.
* **Pairs reject only student samples.** The teacher is never the rejected side.

## Smoke tests, and what they showed

1. **Does the reward tell good from bad?** (`slurm/improve_smoke_reward.sbatch`, 16 papers, real
   endpoint.) Known-defective candidates were injected in place of samples. Mean fact reward / total
   reward / share of numbers unsupported by the paper:

   | candidate | fact | reward | unsupported numbers |
   |:--|--:|--:|--:|
   | the teacher's own summary | 0.71 | 0.837 | 0.078 |
   | an identical copy | 0.71 | 0.837 | 0.078 |
   | every number changed | 0.19 | 0.111 | 0.727 |
   | method, results, limitations cut to one sentence | 0.32 | 0.416 | 0.020 |
   | another paper's summary | 0.00 | -0.214 | 0.721 |

   The order is the right one, the copy ties the teacher (so it is correctly not preferred), and the
   teacher or its copy was chosen in 16 of 16 papers.
2. **Does the whole thing run?** (`slurm/improve_smoke.sbatch`, 6 papers on the 40 GB slice, 51 min,
   of which 20 were the first-time compile of the longer context.) Pool, sampling from the starting
   adapter (the adapter-applied check passed: 4 of 4 probes change), the endpoint reward, selection,
   both training sets, six optimiser steps from the starting adapter, and the trained adapter
   generating. The trainer's held-out loss at step 0 was 0.4572 (the untrained model is ~0.70), which
   shows the starting adapter was loaded; after six examples 0.4567.
3. **Are the targets built from samples well formed?** (`scripts/check_improve_targets.py`.) A
   sample's target is rendered by the same function as the teacher's, which reproduces the stored
   teacher targets exactly for every paper; shards compared as token ids; pairs alternate chosen and
   rejected, share the prompt and differ in target.
4. **Defects found by these tests.** A reader call that returned nothing parseable counted as "knew
   nothing", silently dropping a paper (one of six); it is now retried. The six-paper run chose the
   teacher everywhere, so the path that builds a target from a sample was not exercised by it; the
   target check forces that path.

## Protocol fixed before any arm was trained

* **Size of the round.** The pool is 3,000 papers, but sampling on the 40 GB slice ran at about 18 s
  per paper in the first job, so the round stops at the first of: 1,500 papers finished (sampled and
  scored), or 2026-11-04 22:00 local time. Papers are processed in pool order, so the finished set is a
  prefix of the pool, not a selection. Selection and all three arms use only finished papers.
* **Hyperparameters** are those above and in `slurm/models/q35-{ctl,rft,pref}.env`, chosen before
  training and not changed after any arm's evaluation: SFT learning rate 3e-5, preference learning
  rate 2e-5, beta 0.1, likelihood weight 1.0, reward weights 0.15 / 0.30 / 1.0, margin 0.05, pair gap
  0.25. They were not tuned. The reward weights were set so that the teacher is replaced only when a
  sample is clearly better; the share of papers where that happens is reported, not targeted.
* **Evaluation.** The 899 fact-test papers (two question sets written by `gemma4-31b-it`, graded by
  `qwen38-27b`) plus the first 500 test papers, for the starting model, the control and each arm,
  with paired intervals (`scripts/improve_compare.py`). The primary measure is fact recall on the
  fact-test papers; ROUGE-L against the teacher, BERTScore, keyword F1, the judge and NLI
  faithfulness are reported as well, and the latter two are canaries for reward hacking. A
  difference whose paired interval contains zero is reported as no difference.
* **A negative or null result is reported as such.**

* **Is the selection signal or luck? (decided on 2026-11-04, before any arm is trained.)** A preview on the
  first 412 scored papers showed a student sample beating the teacher's reward in 38% of papers, and
  the chosen samples averaging +0.25 fact reward over the teacher's on the same paper. The five
  samples' fact rewards spread by about 0.11 within a paper, so the best of five exceeds a typical
  draw by about 0.13 *by chance alone*; the same grader that picked cannot tell chance from quality.
  So before the selection arm is read, the papers where a sample was chosen are re-scored with the
  evaluation's independent instrument (`gemma4-31b-it` writes ten new questions, `qwen38-27b` reads and
  grades) for three summaries of the same paper: the teacher's, the chosen sample, and a sample that was
  *not* chosen (the median-reward one). The chosen sample is better than an unchosen one only if the
  paired difference in fact recall on the new questions is above zero. If it is not, the selection is
  noise and the selection arm is reported as self-distillation on arbitrary samples.

## Known limits (written before the results)

* One round, one seed, one starting model. Differences between arms smaller than the paired
  intervals in `scripts/improve_compare.py` should not be read as real.
* The reward is a model's reading of the paper and the gold answers are not human-verified (T16).
  Training on it can improve the reward without improving the summaries; the evaluation uses
  different models and held-out papers to limit this, and the canaries (NLI faithfulness, the judge,
  ROUGE against the teacher, length) are checked for the reward-hacking signature: facts up, those down.
* Selecting the best of several noisy scores overstates the best (winner's curse). The margin and
  the closed-book / whole-paper filter reduce it; they do not remove it.
* Preference tuning can lengthen outputs. The reward charges excess length and the likelihood term
  anchors it, and output length is reported.

## Results

### The round as closed (2026-11-04, 10:07 local)

The round stopped on its first rule, 1,500 finished papers, at 1,520 (the last sampling job was let
finish). It took about five and a half hours of sampling on the 40 GB slice (one job at a time, 160-200
papers an hour) and the scoring ran beside it on the endpoint. Final selection over the 1,520 papers
(`state/improve/selection-stats.json`):

| | |
|:--|--:|
| papers finished | 1,520 |
| scored (at least 5 usable questions) | 1,431 |
| too few usable questions: the teacher's target kept unscored | 89 |
| a sample replaced the teacher | **486** (34.0% of scored) |
| student samples (5 per paper) valid JSON | 99.95% of 7,387 |
| mean fact reward: teacher / samples / chosen | 0.625 / 0.573 / 0.710 |
| unsupported numbers: teacher / samples / chosen | 4.2% / 3.7% / 2.9% |
| length: samples / chosen samples, relative to the teacher | 1.002x / 1.021x |
| preference pairs (gap at least 0.25) | **1,073**, of which the chosen side is a sample in 359 |
| rejected / chosen length, median | 0.951 |
| usable questions per scored paper, mean | 8.37 of 14 |

The three training sets (`tokens/improve-{ctl,rft,pref}`): the control has the 1,520 papers with the
teacher's targets; the selection arm the same 1,520 papers with 486 targets replaced; the preference
arm the 1,073 pairs. The training-set sizes follow from the stop rule and were not chosen.

**Training setup (as run).** Each arm is one job on a single 48 GB card (the scheduler had no 80 GB
card for this account within a day), starting from Student C's 20,000-paper adapter, one pass,
gradient checkpointing, 8 papers per optimiser step. On one card the preference step backpropagates
the two summaries one after the other (`preference_coefficients` in `scisum/model.py`): this is the
same gradient as the joint loss (a unit test compares every adapter gradient), with the memory of one
sequence, at the cost of two extra forward passes. The control and the selection arm differ only in
the 486 targets; the preference arm differs in objective and in its data (1,073 pairs, not 1,520
papers), so its comparison with the control mixes the two.

The evaluation list is 1,246 test papers: the 899 with fact-test questions (299 in set A, 600 in set
B, disjoint) and the first 500 test papers (they overlap by 153).

### Before any arm: what the selection is worth (2026-11-04, preview on the first 412 scored papers)

**How often a sample wins.** A student sample beat the teacher by the margin in **38%** of scored papers
(157 of 412); 49 further papers had too few usable questions and keep the teacher's target. Mean fact
reward: teacher 0.607, samples 0.560, chosen 0.702. Samples were valid JSON 99.9% of the time, their
length averaged 1.009x the teacher's (chosen: 1.017x), and numbers unsupported by the paper were 3.8% of
the samples' numbers against 4.8% of the teacher's (chosen: 2.9%). 325 papers (79%) have a preference
pair with a reward gap of at least 0.25 (mean gap 0.43), and the chosen side is the teacher's summary in
62% of them.

**Is that selection signal or luck?** The five samples' fact rewards spread by 0.11 within a paper, so
the best of five beats a typical draw by about 0.13 by chance alone, and on the reward that chose them
the chosen samples were +0.25 fact above the teacher. Re-scored on ten *new* questions per paper, written
by a different model and read and graded by a different one (157 papers where a sample was chosen):

| summary | fact recall on the new questions |
|:--|--:|
| the teacher's | 0.569 |
| the chosen sample | 0.589 |
| a typical (median-reward) unchosen sample | 0.565 |

chosen - typical sample **+0.0245** [+0.006, +0.044]; chosen - teacher **+0.0204** [+0.001, +0.041];
typical sample - teacher -0.004 [-0.022, +0.014]. The selection carries real signal (the chosen sample is
better than an unchosen one on questions it never saw), but about nine-tenths of the apparent advantage
was the best-of-five effect, and averaged over all papers the selection arm's targets are only about
+0.008 recall better than the control's. A large difference between those two arms is therefore not
to be expected, and none should be read from a small one.

**Are the preference pairs ordered correctly?** The same independent check on 199 pairs (the chosen
summary and a student sample the reward rated at least 0.25 worse, ten new questions per paper, both read
and graded by the evaluation's models):

| | pairs | chosen minus rejected, fact recall |
|:--|--:|--:|
| all | 199 | **+0.068** [+0.049, +0.086] |
| the chosen summary is the teacher's | 123 | +0.085 [+0.061, +0.109] |
| the chosen summary is a student sample | 76 | +0.041 [+0.015, +0.066] |

The chosen summary answered more of the new questions in 59.8% of pairs and fewer in 21.1% (the rest
tied). The reward and this measure are on different scales (the reward adds an imitation term and a number
term, mean gap 0.43), so only the ordering is compared. The pairs are real, and a stronger signal than the
selection, which is the reason to expect more from the preference arm.

### Selection against the control (2026-11-04, 21:00 local)

The control is the same 1,520 papers with the teacher's own targets, so selection − control isolates
the 486 replaced targets. Selection − C mixes them with training on 1,520 more papers. All
numbers below come from `scripts/improve_compare.py` (run 5234597), written to
`paper_docs/data/improve_results.json`. Intervals are paired over papers, 2,000 bootstrap resamples,
95%.

| 1,246 test papers, against the teacher | C | control | selection |
|:--|--:|--:|--:|
| ROUGE-L | 0.4788 | 0.4803 | 0.4819 |
| BERTScore | 0.4583 | 0.4596 | 0.4614 |
| keyword F1 | 0.6041 | 0.6078 | 0.6062 |
| length ratio | 0.9913 | 0.9810 | 0.9894 |
| valid JSON / schema (failures of 1,246) | 1 | 4 | 1 |
| exact domain label | 0.1878 | 0.1974 | 0.1806 |
| NLI faithfulness (300 papers) | 0.4695 | 0.4594 | 0.4729 |

| fact test, 899 papers (sets A and B) | recall | correct | wrong | not stated |
|:--|--:|--:|--:|--:|
| teacher | 0.6082 | 0.5010 | 0.0475 | 0.2372 |
| C | 0.5708 | 0.4657 | 0.0472 | 0.2770 |
| control | 0.5678 | 0.4670 | 0.0484 | 0.2830 |
| selection | 0.5747 | 0.4713 | 0.0468 | 0.2750 |

| paired difference | control − C | selection − C | **selection − control** |
|:--|--:|--:|--:|
| fact recall (primary) | −0.0029 [−0.0092, +0.0033] | +0.0040 [−0.0023, +0.0104] | **+0.0069 [+0.0001, +0.0140]** |
| fact: correct | +0.0013 [−0.0058, +0.0090] | +0.0056 [−0.0029, +0.0136] | +0.0043 [−0.0036, +0.0123] |
| fact: wrong | +0.0012 [−0.0031, +0.0058] | −0.0003 [−0.0047, +0.0046] | −0.0016 [−0.0060, +0.0032] |
| NLI faithfulness (300 papers) | −0.0101 [−0.0201, +0.0001] | +0.0034 [−0.0062, +0.0130] | **+0.0135 [+0.0042, +0.0234]** |
| ROUGE-L | +0.0015 [−0.0006, +0.0035] | +0.0031 [+0.0012, +0.0049] | +0.0016 [−0.0004, +0.0036] |
| BERTScore | +0.0012 [−0.0021, +0.0043] | +0.0030 [+0.0011, +0.0050] | +0.0018 [−0.0012, +0.0052] |
| length ratio | −0.0104 [−0.0166, −0.0042] | −0.0019 [−0.0075, +0.0040] | +0.0085 [+0.0019, +0.0168] |

**The primary difference is at the edge, so it was checked three more ways** (added to the script
after this interval was seen; the estimand is unchanged, see `robustness()`):

| selection − control, fact recall | |
|:--|--:|
| 95% lower bound, 10,000 resamples, five seeds | +0.0003 to +0.0004 |
| paired sign-flip permutation test, two-sided | p = 0.040 |
| question set A alone (299 papers) | +0.0020 [−0.0104, +0.0137] |
| question set B alone (600 papers) | +0.0094 [+0.0019, +0.0168] |
| papers where selection answers more / fewer | 38.5% / 33.8% |

**Reading.**

* By the protocol's rule (an interval that excludes zero), the selection arm recalls more facts than
  the control: +0.0069, about a fifth of the teacher's lead over C (0.0374). The estimate is in line
  with the +0.008 predicted before training from the target-level check.
* The evidence is weak. The lower bound is +0.0003 and p = 0.04. The difference comes from set B
  alone; set A agrees in sign but not significantly. It would not survive a correction for the three
  arm comparisons on the primary measure (Bonferroni needs p < 0.017).
* Most of the gap is the control losing ground, not the selection gaining. Continuing on the
  teacher's targets moved recall by −0.003, NLI faithfulness by −0.010 (interval touching zero) and
  length by −1.0% (interval excluding zero), and produced three more invalid JSON outputs. The
  selection arm moved recall by +0.004 and NLI by +0.003 (both intervals containing zero), and
  length did not change.
* NLI faithfulness, written in as a canary against reward hacking, points the same way as the fact
  test: selection − control +0.0135 [+0.0042, +0.0234]. The fact test is graded by LLMs and NLI by a
  separate DeBERTa model, so this is a second instrument that agrees.
* No reward-hacking signature: wrong answers, length and ROUGE against the teacher did not fall for
  the selection arm.
* Neither continuation closed much of the gap to the teacher. The teacher leads the selection arm
  by +0.0334 [+0.0252, +0.0417] recall, and C by +0.0374.

**What this does and does not show.** Replacing a third of the targets with reward-selected
samples, in one pass over 1,520 papers, is at least as good as the teacher's targets on every
measure. It is probably slightly better on facts and on NLI faithfulness. The size, +0.007 recall,
is too small to matter on its own. Whether it grows with more rounds or more papers is not tested.
One seed, one round.

The final four-arm run of the script (`runs/logs/sum-imp-compare-final.log`, below) draws the
bootstrap again and moves these intervals by up to 0.0004 (selection − control: [+0.0003, +0.0136]);
the means are identical. `paper_docs/data/improve_results.json` holds that final run.

### The preference arm (2026-11-05, 01:45 local)

Preference training on 1,073 reward-ranked pairs (DPO, beta 0.1, plus the likelihood of the chosen
summary with weight 1.0), one pass, from the same starting adapter. Numbers from
`scripts/improve_compare.py` (`runs/logs/sum-imp-compare-final.log`) and `scripts/improve_length.py`
(`runs/logs/sum-imp-length.log`, `paper_docs/data/improve_length.json`).

| 1,246 test papers | C | control | selection | **preference** |
|:--|--:|--:|--:|--:|
| ROUGE-L against the teacher | 0.4788 | 0.4803 | 0.4819 | 0.4522 |
| BERTScore | 0.4583 | 0.4596 | 0.4614 | 0.4243 |
| keyword F1 | 0.6041 | 0.6078 | 0.6062 | 0.5888 |
| length relative to the teacher | 0.991 | 0.981 | 0.989 | **1.263** |
| output tokens, mean | 955 | 961 | 964 | 1,298 |
| invalid JSON, of 1,246 | 1 | 4 | 1 | 10 |
| stopped at the token limit, of 1,246 | 1 | 4 | 1 | 7 |
| NLI faithfulness (300 papers) | 0.4695 | 0.4594 | 0.4729 | **0.4137** |

| fact test, 899 papers | recall | correct | wrong | not stated |
|:--|--:|--:|--:|--:|
| teacher | 0.6082 | 0.5010 | 0.0475 | 0.2372 |
| C | 0.5708 | 0.4657 | 0.0472 | 0.2770 |
| control | 0.5678 | 0.4670 | 0.0484 | 0.2830 |
| selection | 0.5747 | 0.4713 | 0.0468 | 0.2750 |
| **preference** | **0.6419** | **0.5404** | 0.0527 | **0.2038** |

| preference minus | C | control | selection | teacher |
|:--|--:|--:|--:|--:|
| fact recall | **+0.0711 [+0.0635, +0.0790]** | +0.0741 [+0.0661, +0.0823] | +0.0672 [+0.0592, +0.0753] | +0.0338 [+0.0249, +0.0424] |
| fact: correct | +0.0747 [+0.0650, +0.0838] | +0.0733 [+0.0635, +0.0831] | +0.0691 [+0.0600, +0.0781] | |
| fact: wrong | +0.0056 [+0.0006, +0.0107] | +0.0043 [−0.0008, +0.0093] | +0.0059 [+0.0012, +0.0110] | |
| NLI faithfulness | −0.0558 [−0.0670, −0.0443] | −0.0457 [−0.0565, −0.0347] | −0.0592 [−0.0712, −0.0466] | |
| ROUGE-L | −0.0267 [−0.0296, −0.0237] | −0.0282 [−0.0309, −0.0254] | −0.0297 [−0.0327, −0.0268] | |
| BERTScore | −0.0340 [−0.0395, −0.0291] | −0.0352 [−0.0403, −0.0303] | −0.0370 [−0.0420, −0.0324] | |
| length ratio | +0.2720 [+0.2627, +0.2809] | +0.2824 [+0.2731, +0.2917] | +0.2739 [+0.2634, +0.2839] | |

The recall gain over C holds on each question set alone (A: +0.0737 [+0.0594, +0.0871]; B: +0.0699
[+0.0599, +0.0799]). The permutation test gives p = 0.0001, the smallest 10,000 flips can show. The
preference arm answers more in 65.2% of papers and fewer in 16.4%.

**How much of the gain is length?** A longer summary can answer more questions without being a
better one. `scripts/improve_length.py` (written after these results) separates the two in the ways
the fact test allows, with no new model calls:

| per answered question, pooled over 899 papers | teacher | C | control | selection | preference |
|:--|--:|--:|--:|--:|--:|
| share of questions answered (not "not stated") | 0.763 | 0.723 | 0.717 | 0.725 | **0.796** |
| correct, of answered | 0.657 | 0.644 | 0.651 | 0.650 | **0.679** |
| partial, of answered | 0.281 | 0.291 | 0.281 | 0.285 | 0.255 |
| wrong, of answered | 0.062 | 0.065 | 0.068 | 0.065 | 0.066 |

| per paper, arm − C recall regressed on log(arm tokens / C tokens) | median length ratio | slope | gain at equal length (intercept) |
|:--|--:|--:|--:|
| control | 1.000 | +0.089 [−0.003, +0.193] | −0.003 [−0.009, +0.003] |
| selection | 1.011 | +0.238 [+0.177, +0.306] | +0.001 [−0.005, +0.008] |
| preference | 1.336 | +0.067 [−0.039, +0.179] | **+0.051 [+0.019, +0.081]** |

By how much longer the preference arm wrote than C on the same paper (descriptive: the bins are
chosen on an outcome of training): at most 1.1× (27 papers) +0.039 [+0.006, +0.072]; 1.1-1.3× (335)
+0.051; 1.3-1.6× (453) +0.083; 1.6× or more (84) +0.097.

**Reading.**

* **Only the preference arm moves fact recall by a useful amount.** It gains +0.071 over C (+12%
  relative) and +0.074 over the control, and it passes the teacher by +0.034. The gain is consistent
  across both question sets and is spread over most papers.
* **It is not only length.** Of the questions it answers, more are fully correct (64.4% → 67.9%) and
  fewer partial. The wrong share of answered questions is unchanged (6.5% → 6.6%). The linear model
  puts about 70% of the gain at equal length (+0.051 [+0.019, +0.081]). That estimate extrapolates:
  only 27 of 899 papers are within 10% of C's length, and those 27 gain +0.039 [+0.006, +0.072].
* **It also shows the signature the protocol named before training**: facts up, canaries down. The
  output is 27% longer than the teacher's (the reward let length up to 1.2× the teacher's go free).
  ROUGE-L falls by 0.027, BERTScore by 0.034 and NLI faithfulness by 0.056. Invalid JSON goes from 1
  to 10 of 1,246, 7 outputs hit the token limit, and wrong answers per question rise by +0.0056
  [+0.0006, +0.0107]: more answers, at the same error rate per answer.
* **Two of three faithfulness instruments go down, and both are partial views of the paper** (see
  "The judge" below). The fact test grades each question with an LLM and finds no more errors per
  answer. NLI and the judge both rate the preference arm lower. NLI checks each summary sentence
  against the 3-sentence chunks of the paper closest to it; the preference arm writes 17% more
  sentences, each 11% longer. The judge sees only the first 9,000 characters of the paper. Whether
  the drop is invented content or claims these instruments cannot check is not settled by this
  round.
* **The pairs taught the length.** In the preference data the rejected summary is the shorter one in
  the median (0.951 of the chosen), and DPO is known to reward length when chosen outputs are
  longer. The likelihood term (weight 1.0) did not stop it. It kept the likelihood of the chosen
  summaries flat but not that of unseen teacher summaries (held-out loss 0.450 → 0.478, see the
  experiment log).

**The judge** (Qwen3.8-27B; it sees a 9,000-character excerpt of the paper and the student's and the
teacher's summaries, and scores each 1-5 on four dimensions; first 500 test papers, 499 judged for
every arm):

| | C | control | selection | preference |
|:--|--:|--:|--:|--:|
| student − teacher, mean of the four | −0.058 | −0.063 | −0.042 | **+0.017** |
| student preferred / tied | 14.0% / 62.9% | 15.0% / 60.3% | 15.8% / 61.1% | 19.8% / 65.7% |
| faithfulness | 2.158 | 2.154 | 2.162 | **1.994** |
| coverage | 4.206 | 4.208 | 4.242 | **4.399** |
| specificity | 3.631 | 3.653 | 3.685 | **4.036** |
| fluency | 4.990 | 4.986 | 4.996 | 4.966 |

Preference − C: faithfulness −0.164 [−0.224, −0.106], coverage +0.192 [+0.126, +0.263], specificity
+0.405 [+0.295, +0.509], fluency −0.024 [−0.058, +0.000], mean +0.102 [+0.052, +0.151]. No other
pair of arms differs on any dimension. The preference arm is the only student the judge rates level
with the teacher overall.

How far to trust the faithfulness dimension: the judge gives the *teacher* 2.07-2.19 of 5 on
faithfulness in the same calls, because most of a summary cannot be checked against a 9,000-character
excerpt. So the score mostly measures how much of a summary the excerpt covers, and a longer, more
specific summary makes more claims from beyond it. Next to the preference arm the teacher's own
faithfulness rises to 2.19, from 2.07 next to C: the judge scores the pair relative to each other.
Taken at face value, the judge and NLI say the preference arm makes more unsupported statements.
The fact test, the only instrument whose questions and gold answers come from the whole paper, says its
error rate per answered question is unchanged. A check that settles it would grade every claim of
both summaries against the whole paper (or a human sample). It was not run.

**Against the aim of the round** (keep imitating the teacher and be right about the paper), neither
arm does both. Selection keeps the imitation and adds at most +0.007 recall over the control.
Preference adds +0.07 recall, more than the teacher has, but moves away from the teacher's wording
and length, and two of the three faithfulness checks drop (both partial views of the paper). The next round should keep the
preference objective and control length in the pairs or in the loss. Options: pairs of similar
length only; a length term in the DPO objective (length-regularised DPO, Park et al. 2024); a
length allowance at the teacher's 1.0× instead of 1.2×; or a larger likelihood weight on the
teacher's target. One seed, one round, one starting model.

## Round 2: preference tuning with length controlled (protocol written 2026-11-07, 06:00 local, before any round-2 arm was evaluated)

**The question.** Round 1's preference arm passed the teacher on fact recall but wrote 1.26x the
teacher's length and moved away from the teacher's wording. Can the preference objective keep the fact
gain while writing at the teacher's length and staying as close to the teacher as C?

**Why length grew, as far as the data say.** In round 1's 1,073 pairs the chosen summary was the longer one
in 61.5% of pairs (target tokens; median +49 tokens, chosen 994 vs rejected 952). That is a small bias,
but DPO is known to amplify a small length difference in the pairs into much longer outputs: Park et al.
(2024, "Disentangling length from quality in direct preference optimization") report DPO answers about
twice as long as the HH-RLHF dataset, whose preferred answers are only about 5% longer. Two remedies,
tested separately and together:

* **In the data (`preflc`).** The reward charges length beyond 1.0x the teacher's (round 1: 1.2x), and
  the rejected sample is the one closest in length to the chosen summary among the samples at least 0.25
  behind it (`scisum.improve.pick_pair`, rule `closest_length`), not the lowest-reward one. The rewards
  are recomputed from the stored parts of round 1's selection (`scisum.improve.rescore`), with no new
  model call; under round 1's settings this reproduces all 1,073 round-1 pairs exactly (checked).
* **In the loss (`prefrd`).** R-DPO (Park et al. 2024, eq. 9): `alpha * (|chosen| - |rejected|)` in target
  tokens is added to the DPO logit, so a pair whose chosen summary is longer already counts as partly won
  and pushes the policy less. alpha = 0.01 per token, the value Park et al. used with beta = 0.1 on HH; not
  tuned here. On round 1's pairs the median |length difference| is 119 tokens, so the term moves the logit
  by about 1.2 for a typical pair, against DPO margins of 2-3 reached in round 1.

| | round 1 `pref` | round 2 `preflc` and `preflcrd` | round 2 `prefrd` |
|:--|--:|--:|--:|
| pairs | 1,073 | 1,163 | 1,073 (round 1's) |
| chosen summary is the teacher's | 714 | 873 | 714 |
| chosen longer than rejected (tokens) | 61.5% | 51.0% | 61.5% |
| chosen - rejected, median tokens | +49 | +3 | +49 |
| \|chosen - rejected\|, median tokens | 119 | 83 | 119 |
| mean reward gap | 0.425 | 0.372 | 0.425 |
| length allowance in the reward | 1.2x | 1.0x | 1.2x |
| R-DPO alpha | 0 | 0 / 0.01 | 0.01 |

Everything else is round 1's: C's 20,000-paper adapter as start and as reference, one pass, beta 0.1,
likelihood weight 1.0, learning rate 2e-5, 8 pairs per step, one 48 GB card per arm (two L40S and one
RTX A6000 on 2026-11-07). `prefrd` reuses round 1's reference log-probabilities (same pairs, same
starting adapter); the trainer recomputes the first pair and refuses a file that disagrees by more than
1% or 2 nats (it agreed within 0.5 and 1.4 nats of summed log-probabilities near -450 on the L40S).
Profiles: `slurm/models/q35-{preflc,prefrd,preflcrd}.env`.

**Evaluation**, as round 1: the 1,246 test papers (the 899 fact-test papers of question sets A and B, and
the first 500), the fact test on both sets, NLI on 300, the judge on 500, paired bootstrap intervals
(`scripts/improve_compare.py --round 2`, `scripts/improve_length.py --round 2`, which write separate
files and leave round 1's untouched), and the claim check below.

**Criteria, fixed before evaluation.** An arm achieves the round's aim if all four hold:

1. facts: fact recall above C's (paired interval excludes zero);
2. length: mean length ratio to the teacher within 0.90-1.10;
3. imitation: ROUGE-L and BERTScore against the teacher no more than 0.01 below C's (point estimate);
4. faithfulness: the share of statements the claim check finds unsupported or contradicted no higher
   than C's (paired interval includes zero or lies below it).

Among arms that meet all four, the one with the highest recall is recommended. If none does, the
trade-off is reported as measured. Three arms are compared with C on the primary measure, so the
Bonferroni threshold (p < 0.0167) is reported next to each interval. One seed, one round.

## The claim check: every statement against the whole paper (built 2026-11-07)

Round 1 left one question open: the preference arm's error rate per answered fact-test question did
not change, while NLI and the judge, which see only part of the paper, rated it less faithful.
`scisum/claimcheck.py` settles it without the partial view:

* **Papers**: a seeded random sample of 300 of the 899 fact-test papers. The verifier reads the stored
  paper text that the summarisers read (Student C's prompt body), cut at 90,000 characters as for the
  question writer.
* **Statements**: `claims_of`, the units NLI scores (the prose sentences of tldr, problem, method,
  results and limitations, and the contribution bullets), up to 80 per summary instead of 40.
* **Verifier**: the evaluation's grader, Qwen3.8-27B (thinking off, temperature 0), labels each
  statement `supported`, `not_in_paper` or `contradicted` and quotes the offending detail. One call per
  summary; it is not told which system wrote it.
* **Controls**, on the first 40 papers: the teacher's summary with every checkable number changed, and
  another paper's teacher summary.

Smoke test (6 papers, 2026-11-07): another paper's summary 100% `not_in_paper`; the teacher with changed
numbers 41% `contradicted` (the statements without a number stay supported); the teacher 99% supported,
C 97%, round 1's preference arm 99%. Each call takes about 25 s.

Limits: one model's reading, not a person's; a statement's verdict is that of its worst detail, so longer
statements are flagged more often; and the 90,000-character cut hides the end of the longest papers
from the verifier, equally for every system.

### Claim check results on round 1's arms (2026-11-07, 06:45 local; corrected 17:20)

**Correction (2026-11-07, 17:20 local).** 18 verifier replies (15 of 2,475 from Qwen3.8-27B, 3 of 529
from Gemma-4-31B) were complete except for one malformed field, free text after the closing quote of a
`paper` quotation. A reply that is not valid JSON as a whole was discarded, and at temperature 0 the
three retries returned the same text, so the summary was stored with no verdicts and the report counted
it as 0% supported with 0 unsupported statements. Found while reading round 2's results, where one arm
had 6 such summaries. Fixed: the parser now reads each verdict record on its own when the whole reply
does not parse, a summary with no labelled statement is left out of the report instead of counted, and
the 18 summaries were checked again (`scisum.claimcheck.parse_verdicts`, `checked_ids`; tests in
`tests/test_claimcheck.py`). The tables below are the corrected numbers. The changes are at most 0.03
per summary and 0.2 points of share, except Gemma's preference − C, which grew (old values in
`09_experiment_log.md`, 06:45). No conclusion changed. The robustness numbers and the agreement
between the verifiers are now computed by `scripts/claimcheck_robustness.py`
(`paper_docs/data/claimcheck_robustness.json`) instead of by hand.

300 fact-test papers, verifier Qwen3.8-27B (`state/claimcheck/report.qwen38-27b.json`). Paired
intervals over papers, 2,000 bootstrap resamples, 95%.

**What the verifier detects** (controls, first 40 papers): of the teacher's statements whose numbers
were changed, 98.7% flagged (96.6% as contradicted); of the unchanged statements in the same
summaries, 1.1% flagged; of another paper's statements, 99.9% flagged.

| 300 papers | statements per summary | words per statement | supported | not in paper | contradicted | unsupported per summary | contradicted per summary |
|:--|--:|--:|--:|--:|--:|--:|--:|
| teacher | 21.1 | 25.7 | 97.2% | 2.0% | 0.9% | 0.59 | 0.18 |
| C | 21.8 | 24.4 | 96.6% | 1.5% | 1.8% | 0.73 | 0.40 |
| control | 21.5 | 24.6 | 96.0% | 1.5% | 2.4% | 0.85 | 0.52 |
| selection | 21.9 | 24.3 | 96.6% | 1.4% | 2.0% | 0.74 | 0.43 |
| preference | 25.8 | 27.3 | 95.5% | 1.8% | 2.6% | **1.15** | **0.68** |

(Shares are pooled over statements; per-summary counts are means over papers. Control 299 and
preference 297 papers: the rest had no valid JSON.)

| paired difference | unsupported per summary | contradicted per summary | contradicted share |
|:--|--:|--:|--:|
| preference − C | **+0.45 [+0.31, +0.60]** | **+0.28 [+0.15, +0.40]** | +0.008 [+0.003, +0.013] |
| control − C | +0.13 [+0.02, +0.24] | +0.12 [+0.03, +0.20] | +0.006 [+0.002, +0.011] |
| selection − C | +0.01 [−0.11, +0.12] | +0.03 [−0.06, +0.13] | +0.001 [−0.003, +0.006] |
| C − teacher | +0.14 [+0.00, +0.27] | +0.22 [+0.13, +0.32] | +0.010 [+0.006, +0.015] |

**Robustness.** 27 of the 300 papers were cut at 90,000 characters, so the end of the paper was hidden
from the verifier. On the other 272, preference − C is +0.38 [+0.24, +0.51] unsupported and +0.28
[+0.15, +0.41] contradicted per summary; on the 25 cut ones, +1.28 unsupported, almost all "not in
paper", consistent with the preference arm covering more of the paper's end. A second verifier from
another family (Gemma-4-31B, the first 150 papers of the same sample, `report.gemma4-31b-it.json`)
agrees with the first on 97.2% of 10,300 statements (Cohen's kappa 0.61; both catch 98% of changed
numbers) and on the direction: preference − C +0.31 [+0.11, +0.52] unsupported per summary, +0.21
[+0.05, +0.36] contradicted; C − teacher +0.25 [+0.11, +0.39] contradicted.

**Reading.**

* **The faithfulness cost of round 1's preference arm is real, and small.** It makes about 0.4 more
  unsupported statements per summary than C, of which about 0.3 contradict the paper, in summaries of
  about 26 statements. Its supported share is 95.5% against C's 96.6%. Of the three readings left open
  above, this rules out "the partial-view instruments only missed content they could not see": most of
  the increase is contradictions, and it holds on the papers the verifier saw whole.
* **Why the fact test did not see it.** The fact test asks about a dozen facts per paper; the extra
  errors are in statements no question asks about. Its error rate per answered question is unchanged,
  and the claim check's error rate per statement rises only from 1.8% to 2.6% contradicted.
* **Students contradict the paper more often than the teacher** (C 0.40 against 0.18 per summary, both
  verifiers), and continuing on the teacher's own targets adds a little (control − C +0.12). Selection
  does not.
* NLI and the claim check put the arms in the same order (control and preference below C and selection,
  preference last) except that NLI ranks selection just above C and the claim check just below; neither
  separates those two. The judge's faithfulness drop (−0.16 of 5) is mostly its partial view.

So the round-1 preference arm is more informative and slightly less faithful than C; round 2's
criterion 4 asks whether controlling length removes that.

A blinded sample of 90 statements (15 flagged and 15 supported per system for teacher, C and
preference), each with the verifier's note and the most similar paper passage, is in
`paper_docs/samples/claimcheck_qwen38-27b.md` (not tracked: it quotes paper text), with the key in a
separate file, for a person to check the verifier.

## Round 2 results (2026-11-07, 17:30 local)

All three arms finished one pass on 2026-11-07. Numbers from `paper_docs/data/improve_results_r2.json`,
`improve_length_r2.json`, `state/claimcheck/report.qwen38-27b.json` and
`paper_docs/data/claimcheck_robustness.json`; paired bootstrap intervals over papers, 95%.

| arm | pairs | card, hours | held-out loss on teacher summaries |
|:--|--:|:--|--:|
| C (start) | | | 0.450 |
| round 1 `pref` | 1,073 | | 0.478 |
| `preflc` | 1,163 | L40S, 8.6 h | 0.476 |
| `prefrd` | 1,073 | L40S, 6.6 h | 0.479 |
| `preflcrd` | 1,163 | RTX A6000 1 h, then A100-SXM 5.3 h | 0.481 |

`prefrd` and `preflcrd` were evaluated on full 48 GB cards; `preflc` on the 40 GB A100 slice in
`short`, like round 1's arms, because every `gpu` card was taken when its training ended (same model,
same decoding, same 1,246 papers).

### What each arm does

| 1,246 test papers (fact test: 899) | C | round 1 `pref` | `preflc` | `prefrd` | `preflcrd` | teacher |
|:--|--:|--:|--:|--:|--:|--:|
| fact recall | 0.571 | 0.642 | **0.601** | 0.588 | 0.558 | 0.608 |
| recall − C | | +0.071 [+0.064, +0.079] | **+0.030 [+0.023, +0.037]** | +0.017 [+0.010, +0.024] | −0.013 [−0.020, −0.006] | +0.037 |
| permutation p (Bonferroni: 0.0167) | | 0.0001 | 0.0001 | 0.0001 | 0.0008 | |
| recall − teacher | | +0.034 [+0.025, +0.042] | −0.008 [−0.016, +0.001] | −0.020 [−0.029, −0.012] | −0.050 [−0.059, −0.043] | |
| length ratio to the teacher | 0.991 | 1.263 | **1.012** | 0.939 | 0.839 | 1 |
| output tokens | 955 | 1,298 | 1,028 | 937 | 823 | |
| ROUGE-L − C | | −0.0267 | −0.0063 | −0.0068 | −0.0084 | |
| BERTScore − C | | −0.0340 | −0.0100 | −0.0106 | −0.0125 | |
| invalid JSON (of 1,246) | 1 | 10 | 4 | 2 | 1 | |
| claim check: supported statements | 96.6% | 95.5% | 95.8% | 96.0% | 95.5% | 97.2% |
| unsupported per summary | 0.73 | 1.15 | 0.90 | 0.80 | 0.82 | 0.59 |
| contradicted per summary | 0.40 | 0.68 | 0.54 | 0.47 | 0.51 | 0.18 |
| statements per summary | 21.8 | 25.8 | 21.4 | 20.2 | 18.4 | 21.1 |
| NLI faithfulness − C (300 papers) | | −0.056 | −0.047 [−0.060, −0.036] | −0.052 [−0.062, −0.042] | −0.026 [−0.039, −0.014] | |
| judge, score − teacher's (499 papers) | −0.058 | +0.017 | −0.042 | −0.049 | −0.082 | |
| judge faithfulness − C | | −0.16 [−0.23, −0.11] | −0.12 [−0.18, −0.06] | −0.04 [−0.09, +0.02] | +0.06 [+0.01, +0.12] | |
| judge mean − C | | +0.10 [+0.05, +0.15] | +0.005 [−0.035, +0.044] | −0.010 [−0.045, +0.027] | −0.004 [−0.041, +0.034] | |

Recall on each question set alone: `preflc` − C +0.029 [+0.017, +0.041] (A, 299 papers) and +0.030
[+0.022, +0.039] (B, 600); `prefrd` − C +0.029 and +0.011 [+0.003, +0.020]. `preflc` − `prefrd` +0.013
[+0.005, +0.020], p = 0.0005. Of the answered questions, 66.1% (`preflc`) and 64.5% (`prefrd`) are fully
correct (C 64.4%) and 6.4% and 6.8% wrong (C 6.5%).

### The four criteria

| criterion (fixed before evaluation) | `preflc` | `prefrd` | `preflcrd` |
|:--|:--|:--|:--|
| 1. recall above C's, interval excludes zero | ✅ +0.030 [+0.023, +0.037] | ✅ +0.017 [+0.010, +0.024] | ❌ −0.013 |
| 2. length ratio within 0.90-1.10 | ✅ 1.012 | ✅ 0.939 | ❌ 0.839 |
| 3. ROUGE-L and BERTScore ≥ C − 0.01 (point estimates) | ❌ ROUGE-L −0.0063 ✅, BERTScore −0.010035 (by 0.00004) | ❌ ROUGE-L −0.0068 ✅, BERTScore −0.0106 (by 0.0006) | ❌ −0.0084 ✅, −0.0125 |
| 4. unsupported-or-contradicted share − C, interval includes zero or lies below | ❌ +0.0085 [+0.0031, +0.0144] | ✅ +0.0057 [−0.0001, +0.0112] | ❌ +0.0106 [+0.0046, +0.0169] |
| all four | no | no | no |

**Verdict: no arm meets all four criteria, so by the protocol none is recommended; the trade-off is
reported as measured.** Two of the three misses are by less than BERTScore's own uncertainty (its
paired interval is about 0.007 wide), but the rule was a point estimate, set before the results, and
it stands.

### Reading

* **Both length remedies work, and together they overshoot.** Closest-length pairs alone brought the
  length to 1.01x the teacher's, the R-DPO term alone to 0.94x, both to 0.84x. alpha = 0.01 was set for
  pairs whose chosen summary is 49 tokens longer in the median; on pairs already matched for length
  (median +3) it still pays the policy to write shorter.
* **Most of round 1's fact gain was length.** At the teacher's length the gain over C is +0.030
  (`preflc`) or +0.017 (`prefrd`), against +0.071 at 1.26x. Round 1's estimate of the gain at equal
  length, extrapolated from a regression on the length ratio, was +0.051; the round-2 arms, which write
  at about C's length, put it at +0.019 [+0.011, +0.028] (`preflc`) and +0.021 [+0.014, +0.028]
  (`prefrd`) by the same regression, so the extrapolation overstated it by more than half. About 0.02-0.03
  of recall is gained without writing more.
* **Imitation recovers most of the way, not all.** ROUGE-L −0.027 → −0.006, BERTScore −0.034 → −0.010.
  The held-out loss on teacher summaries rose as in round 1 (0.450 → 0.476-0.481): the preference
  objective moves the student off the teacher's wording whatever the length.
* **The faithfulness cost shrinks but stays.** Per summary, the length-controlled arms make 0.07-0.18
  more unsupported statements than C (round 1: +0.45) and 0.07-0.13 more contradicted ones (round 1:
  +0.28), in summaries no longer than C's. `prefrd`'s share interval just touches zero; `preflc`'s
  and `preflcrd`'s do not. The judge's faithfulness score orders the arms the same way.
* **The arms lie on one line: more facts, more unsupported statements.** `preflc` gains the most
  recall and adds the most unsupported statements (+0.18 per summary); `prefrd` gains about half as
  much and adds +0.07, within noise; `preflcrd` gains nothing and still adds contradictions. No arm is
  better than another on both.

**The trade-off, for a choice outside the protocol.** For the largest fact gain at the teacher's length,
`preflc`: +0.030 recall, judge level with C, about one extra unsupported statement in every five to six
summaries of 21 statements. For the smallest faithfulness cost, `prefrd`: +0.017 recall with no
detectable change in the unsupported share. Neither keeps C's closeness to the teacher's wording to
within 0.01 BERTScore.

**What would close the gap** (not run): (a) put the claim check's verdicts into the reward that ranks
the pairs (it penalises only unsupported numbers now), so the chosen summary is also the more faithful
one; (b) a smaller alpha (0.002-0.005) for closest-length pairs; (c) more than one seed, which would
tell whether the 0.00004 and 0.0006 BERTScore misses are noise.

**Limits.** One seed, one pass, untuned alpha. The claim check's parser was corrected after the
round-2 results were first read (see the correction above). The correction raised C's supported share
in the per-paper means, which made every arm look worse against C, and moved `preflcrd` from passing
criterion 4 to failing it. With the old parser, `preflc`'s per-paper supported-share difference was −0.022,
almost entirely its 6 unparsed summaries. One `preflc` reply was cut at the 3,000-token limit, leaving
7 of its 26 statements unlabelled.

## Round 3: faithfulness in the reward, a smaller length term, second seeds (protocol written 2026-11-07, 23:10 local, before any round-3 arm was trained)

Requested by the user after round 2: (a) put the claim check into the reward that ranks the pairs,
(b) a smaller length term with length-matched pairs, (c) a second seed for `preflc` and `prefrd`.

**Arms** (all from C's 20,000-paper adapter, one pass, beta 0.1, likelihood weight 1.0, lr 2e-5, as round 2):

| arm (run) | pairs | change against its round-2 parent | why |
|:--|:--|:--|:--|
| `prefcc` (`q35prefcc`) | rebuilt: `preflc`'s rule and 1.0x length allowance, reward with a claim penalty | reward − 1.0 × the candidate's share of statements the claim check finds unsupported or contradicted | round 2's arms traded facts for unsupported statements; the reward penalised only unsupported numbers |
| `preflcrd3` (`q35preflcrd3`) | `preflc`'s 1,163 | R-DPO alpha 0.003 (round 2's `preflcrd`: 0.01) | alpha 0.01 overshot to 0.84x on pairs already matched for length (median \|difference\| 83 tokens: a logit shift of 0.83 at 0.01, 0.25 at 0.003) |
| `preflcs2` (`q35preflcs2`) | `preflc`'s 1,163 | pairs in a seeded random order (seed 1) | seed variation; rounds 1-2 read the pairs in file order, so a seed meant nothing until `pref_order_seed` was added |
| `prefrds2` (`q35prefrds2`) | round 1's 1,073 | pairs in a seeded random order (seed 1), alpha 0.01 | as above |

**The claim penalty** (`scisum.improve.run_claims`, `rescore`; profile `slurm/models/q35-prefcc.env`).
Every scored candidate of the 1,520 training papers (the teacher's summary and C's five samples) is
claim-checked against the paper text with the evaluation's statements and prompt. The verifier is
Gemma-4-31B, not the evaluation's Qwen3.8-27B, for the same reason the fact reward's grader is the
other model: the evaluation should not be the instrument the student was trained on (the two agree on
97.2% of statements, kappa 0.61). Weight 1.0 on the share: a summary of 21 statements with one
unsupported loses 0.048, about two-thirds of one fully answered question of 14. A share, not a count,
so the penalty does not reward writing less. Candidates the verifier could not label take no penalty.
Not tuned; fixed here.

**Evaluation**, as round 2: 1,246 test papers, fact test on both question sets (899), NLI (300), judge
(500), claim check (300, Qwen3.8-27B), paired bootstrap intervals (`scripts/improve_compare.py --round 3`,
separate file).

**Criteria, fixed before training:** round 2's four, unchanged (recall above C's; length 0.90-1.10x;
ROUGE-L and BERTScore no more than 0.01 below C's, point estimates; unsupported-or-contradicted share not
above C's), applied to `prefcc` and `preflcrd3`. Five comparisons with C on recall in rounds 2-3 that
can lead to a recommendation (`preflc`, `prefrd`, `preflcrd`, `prefcc`, `preflcrd3`): Bonferroni
threshold p < 0.01, reported next to each interval.

**Seeds.** For `preflc` and `prefrd`, the verdict is taken on the two seeds together: each paper's score
is the mean of the two seeds' scores (a per-paper average), and the four criteria are applied to that.
Each seed is also reported alone, with the difference between seeds as the measure of seed noise.
Specifically: if the two-seed BERTScore difference is within 0.01 of C's, the round-2 misses
(0.00004 and 0.0006) were noise; if not, they were not.

Expected: `prefcc` fewer unsupported statements than `preflc` at some cost in recall; `preflcrd3`
between `preflc` and `preflcrd` in length (about 0.95x). Recorded so a different outcome is visible.

### Round 3 as run (2026-11-08, before any round-3 arm was evaluated)

**The candidates' claim check** (`paper_docs/data/improve_claims_r3.json`, `scripts/improve_claims_stats.py`;
Gemma-4-31B, done 02:54). All 1,520 training papers, 9,360 candidates (the teacher's summary and the
student's samples), 8,814 with at least one labelled statement; the other 546 had no fact score, so they
take neither reward nor penalty. Share of statements found unsupported or contradicted, mean over
candidates:

| candidate | n | mean share | median |
|:--|--:|--:|--:|
| teacher | 1,431 | 0.031 | 0 |
| C's greedy summary | 115 | 0.036 | 0 |
| C's samples | 7,268 | 0.047 | 0.038 |

**What the penalty does to the pairs.** Both pairings recomputed with the build's own functions; the
counts equal the built arms'.

| | `preflc` (no penalty) | `prefcc` (penalty 1.0 × share) |
|:--|--:|--:|
| pairs | 1,163 | 1,233 |
| chosen summary from the samples (rest: teacher) | 290 | 293 |
| rejected/chosen length, median | 0.996 | 0.994 |
| rejected summary shorter | 51.0% | 51.5% |
| reward gap, mean | 0.372 | 0.380 |
| unsupported share of the chosen summary | 0.033 | 0.024 |
| unsupported share of the rejected summary | 0.050 | 0.062 |

Of the 1,144 papers with a pair under both rules, 897 keep the same pair, 173 change only the
rejected sample, 53 only the chosen summary, 21 both; none is reversed. 19 papers lose their pair (no
sample trails the chosen one by the 0.25 gap any more) and 89 gain one. So the penalty mostly picks a
less faithful sample to reject: the gap in unsupported share between chosen and rejected doubles, from
0.017 to 0.037, while lengths stay matched.

**A slip in the launch, and why the training is unaffected.** `preflcrd3`, `preflcs2` and `prefrds2`
were submitted without `SUM_IMP_N`, which round 2 set to the number of pairs. The profile then used its
default of 1,500 for `n_train` and the milestone list. Training is unaffected: it runs
`cap = min(n_train, pairs)` pairs, and the schedule is computed from `cap` (146 steps for 1,163 pairs, as
`preflc`). But the milestone at 1,500 was never reached, so the runs wrote no `ckpt/ex<N>`. That is the
directory the evaluation reads, and the held-out loss is computed there. `scripts/promote_final.py`
publishes the last rolling checkpoint, written after the final step and checked to be at the final
pair, as `ex<N>`, computing the held-out loss on the same 200 papers as a milestone would.
`slurm/improve_promote_eval.sbatch` runs the promotion and then the unchanged evaluation on one card.
`prefcc` was submitted with `SUM_IMP_N=1233`.

### Round 3 results (2026-11-08)

All four arms were evaluated as round 2 (1,246 test papers; fact test on both question sets, 899 papers;
NLI 300; judge 499; claim check 300, Qwen3.8-27B). Numbers from `scripts/improve_compare.py --round 3`
(`paper_docs/data/improve_results_r3.json`), `scripts/improve_length.py --round 3`
(`improve_length_r3.json`), `scripts/improve_seeds.py` (`improve_seeds_r3.json`), `scisum claimcheck
report` and `scripts/claimcheck_robustness.py` (`claimcheck_robustness.json`). Intervals are 95% paired
bootstrap intervals over papers.

**Training.** Held-out loss on teacher summaries at the final pair: `prefcc` 0.4779, `preflcrd3` 0.4769,
`preflcs2` 0.4757 (seed 1: 0.4760), `prefrds2` 0.4792 (seed 1: 0.4793). `prefcc` wrote its own `ex1233`;
the other three were published with `scripts/promote_final.py` (see "Round 3 as run").

**Each arm against C**

| | C | `prefcc` | `preflcrd3` | `preflc` seed 1 / seed 2 | `prefrd` seed 1 / seed 2 |
|:--|--:|--:|--:|--:|--:|
| length ratio | 0.991 | 1.017 | 0.950 | 1.012 / 0.975 | 0.939 / 0.885 |
| ROUGE-L | 0.4788 | 0.4734 | 0.4741 | 0.4725 / 0.4729 | 0.4721 / 0.4691 |
| BERTScore | 0.4583 | 0.4513 | 0.4505 | 0.4483 / 0.4443 | 0.4478 / 0.4423 |
| fact recall − C | | **+0.0327 [+0.0259, +0.0400]** | +0.0189 [+0.0116, +0.0265] | +0.0299 / +0.0169 | +0.0171 / −0.0029 |
| fact recall − teacher | | −0.0047 [−0.0132, +0.0034] | −0.0185 | −0.0075 / −0.0205 | −0.0203 / −0.0403 |
| unsupported-or-contradicted share − C | | **+0.0003 [−0.0046, +0.0050]** | +0.0092 [+0.0033, +0.0156] | +0.0085 / +0.0076 | +0.0057 / +0.0143 |
| unsupported (contradicted) statements per summary | 0.73 (0.40) | 0.74 (0.42) | 0.83 (0.50) | 0.90 (0.54) / 0.84 (0.47) | 0.80 (0.47) / 0.88 (0.54) |
| NLI − C | | −0.025 [−0.035, −0.015] | −0.041 | −0.047 / −0.037 | −0.052 / −0.049 |
| judge faithfulness − C (1-5 scale) | | −0.090 [−0.146, −0.036] | −0.078 [−0.134, −0.020] | −0.120 / −0.096 | −0.036 / −0.054 |
| judge mean − C | | +0.006 [−0.031, +0.044] | −0.023 | +0.006 / +0.023 | −0.009 / −0.041 |

Every recall gain except `prefrds2`'s has permutation p = 0.0001 and holds on each question set
(`prefcc`: +0.041 on set A, 299 papers; +0.029 on set B, 600). `prefcc` writes 21.6 statements per summary
(C 21.4) and answers 76.1% of the fact questions (C 72.3%, teacher 76.3%); of its answers 65.5% are fully
correct (C 64.4%) and 6.8% wrong (C 6.5%). The fact gain at equal length, by the regression of round 2, is
+0.024 [+0.016, +0.031] for `prefcc`, +0.019 for `preflcrd3`, +0.015 for `preflcs2` and −0.006 [−0.019,
+0.010] for `prefrds2`.

**The criteria** (fixed before training; `prefcc` and `preflcrd3` on one seed, `preflc` and `prefrd` on
the per-paper mean of their two seeds)

| criterion | `prefcc` | `preflcrd3` | `preflc` (two seeds) | `prefrd` (two seeds) |
|:--|:--|:--|:--|:--|
| 1. recall above C's, interval excludes zero (p vs Bonferroni 0.01) | ✅ +0.0327 [+0.0259, +0.0400], p = 0.0001 | ✅ +0.0189 [+0.0116, +0.0265], p = 0.0001 | ✅ +0.0234 [+0.0171, +0.0294], p = 0.0001 | ✅ +0.0071 [+0.0008, +0.0133], p = 0.025 (above 0.01) |
| 2. length ratio within 0.90-1.10 | ✅ 1.017 | ✅ 0.950 | ✅ 0.993 | ✅ 0.912 |
| 3. ROUGE-L and BERTScore ≥ C − 0.01 | ✅ −0.0054, −0.0070 | ✅ −0.0047, −0.0079 | ❌ −0.0061, −0.0120 | ❌ −0.0082, −0.0133 |
| 4. unsupported-or-contradicted share − C, interval includes zero or lies below | ✅ +0.0003 [−0.0046, +0.0050] | ❌ +0.0092 [+0.0033, +0.0156] | ❌ +0.0078 [+0.0026, +0.0129] | ❌ +0.0100 [+0.0052, +0.0151] |
| all four | **yes** | no | no | no |

**Verdict: `prefcc` meets all four criteria; by the protocol it is the recommended arm.** It is the first
arm in three rounds to do so. It recalls as many facts as the teacher (−0.005 [−0.013, +0.003]), 0.033
more than C, at C's length, with as many unsupported statements as C by the claim check. The others miss:
`preflcrd3` on criterion 4, and `preflc` and `prefrd` on criteria 3 and 4 over two seeds.

*Update 2026-11-09: prefcc's second seed.* Over two seeds `prefcc` still meets all four criteria: recall
+0.0262 [+0.0201, +0.0322], length 0.988, ROUGE-L −0.0053, BERTScore −0.0085, unsupported share +0.0045
[−0.0003, +0.0093]. The second seed alone fails criterion 4, and its seed difference is in the same
direction as the two rows below (length −0.059, recall −0.013, unsupported +0.008). Details in
`12_reactive_training.md` §8, question 1.

*Update 2026-11-10: the reactive run.* An LLM agent steering prefcc's training on seed 2's pairs (`q35prefccrx`) does not beat the fixed seed-2 run: recall −0.0061 [−0.0126, +0.0006], unsupported share +0.0003 [−0.0055, +0.0065]. Against C it fails criterion 4 (+0.0082 [+0.0018, +0.0146]), like seed 2 alone. `prefcc` stays the recommended arm. Details in `12_reactive_training.md` §8, question 3, and F12.

**Seeds** (the per-paper difference between the two seeds; same pairs and settings, other pair order)

| seed 2 − seed 1 | length ratio | BERTScore | fact recall | unsupported share |
|:--|--:|--:|--:|--:|
| `preflc` | −0.036 [−0.044, −0.030] | −0.0040 [−0.0087, +0.0003] | −0.0130 [−0.0206, −0.0060] | −0.0003 [−0.0052, +0.0051] |
| `prefrd` | −0.054 [−0.060, −0.048] | −0.0054 [−0.0104, −0.0010] | −0.0201 [−0.0274, −0.0128] | +0.0085 [+0.0020, +0.0153] |

Pair order alone moves the length by 0.04-0.05 and recall by 0.013-0.020, as much as the differences
between round-2 arms; in both arms the second seed wrote shorter and recalled less. The pre-registered
question about round 2's BERTScore misses is answered: the two-seed BERTScore is 0.0120 (`preflc`) and
0.0133 (`prefrd`) below C's, outside 0.01, so those misses were not noise. The unsupported share was
stable between `preflc`'s seeds (+0.0079 and +0.0076 against C on the 296 papers all three have), less so
between `prefrd`'s.

**`prefcc` against `preflc`'s seeds** (descriptive, not a criterion; `prefcc` is `preflc`'s pairing rule
plus the claim penalty)

| | `prefcc` − two-seed mean | − seed 1 | − seed 2 | seed 2 − seed 1 |
|:--|--:|--:|--:|--:|
| unsupported share | −0.0069 [−0.0118, −0.0021] | −0.0071 | −0.0068 | −0.0003 |
| fact recall | +0.0093 [+0.0038, +0.0152] | +0.0028 | +0.0158 | −0.0130 |
| BERTScore | +0.0050 [+0.0020, +0.0080] | +0.0030 | +0.0070 | −0.0040 |
| length ratio | +0.024 [+0.018, +0.029] | +0.006 | +0.042 | −0.036 |

The claim penalty lowers the unsupported share by 0.007 against either seed, where the seeds differ by
0.0003: this effect is larger than the seed noise. It costs no recall: `prefcc` is level with seed 1
and above seed 2. NLI agrees in direction (`prefcc` − `preflc` +0.022 [+0.011, +0.033]); the judge's
faithfulness score moves the same way but within noise (+0.030 [−0.014, +0.074]).

**Against the expectations written before training.** `prefcc` was expected to have fewer unsupported
statements than `preflc` at some cost in recall: fewer, yes, but no recall cost. `preflcrd3` was
expected at about 0.95x the length: 0.950.

**Caveats.**

* *(Superseded 2026-11-09: the second seed was run, see the update above.)* **One seed for `prefcc`.** The seed differences above (recall 0.013-0.020, length 0.04-0.05, BERTScore
  up to 0.005) are large next to two of `prefcc`'s margins: BERTScore is 0.0030 inside criterion 3's
  limit, and its recall interval starts at +0.026. A second seed is the check that would settle it.
* **The other faithfulness measures do not reach C.** NLI (−0.025 [−0.035, −0.015]) and the judge's
  faithfulness score (−0.090 [−0.146, −0.036]) are still below C's for `prefcc`, as for every preference
  arm. Criterion 4 uses the claim check, which was fixed as the faithfulness criterion before round 2;
  the other two are reported, not criteria.
* **The training reward and the evaluation share a method.** The penalty used Gemma-4-31B's claim check
  of the candidates; the evaluation uses Qwen3.8-27B with the same prompt and labels. The two verifiers
  agree on 97.1% of C's statements (Cohen's kappa 0.60), so part of the gain could be the student learning
  what this kind of check rewards rather than what is true. NLI moving the same way against `preflc` is
  the independent sign that it is not only that.
* **The claim check sees 90,000 characters of the paper.** On the 273 papers it saw whole, `prefcc` − C
  is −0.0011 [−0.0062, +0.0045] in supported share; on the 26 cut papers +0.0086 [−0.0109, +0.0281].
