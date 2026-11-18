# 12 — An LLM agent as an active optimiser ("reactive training"): analysis and plan

Written 2026-11-08. Status (2026-11-16): §8 is complete (the second seed, the blends, and the reactive run, which was not credited). Round 4 (§9) is the last experiment and is complete: a null result, `prefcc` stays the recommended arm (verdict 2026-11-16 21:29 local).

## 1. The source idea (reactive-training skill, OPAL / *Reactive Optimizers*)

A running PyTorch job pauses at decision points and writes a report: train and validation curves,
gradient statistics, and the diagnostics the agent itself asked for in `analysis.py`. The agent reads the
report and may rewrite only the optimiser-side code (`updater.py`: update rule, LR schedule, momentum,
clipping, weight averaging, early stop, which checkpoint is final). That code is hot-reloaded, with a
snapshot and automatic rollback if it crashes or produces NaN.

The model, data, loss and evaluation stay frozen. The agent sees a validation split; the test split is
read once, after the run.

Every decision follows a fixed protocol:
1. prediction check (score the last prediction against the numbers);
2. observed;
3. changed (the smallest change that tests one hypothesis);
4. a falsifiable prediction;
5. rules (which behaviours are fixed and which react to a signal);
6. risks.

Settled verdicts and standing rules go into `NOTES.md`.

Reference result (ResNet-18, CIFAR-10 with 10% wrong labels):

| | test accuracy | test loss |
|---|---|---|
| fixed AdamW | 74.4% | 2.39 |
| agent | 79.1% | 0.70 |
| the agent's trick (weight averaging) run as a fixed recipe | 77.5% | 0.83 |

So most of the gain was the averaging, which a fixed recipe also gets. The agent run cost more than 3×
the fixed run's wall time, and the gains were small on recipes that were already well tuned. The skill's
own lessons:
- Run a fixed baseline next to the agent run.
- Then run the agent's trick as a fixed recipe (the "strong fixed baseline").
- One run is one sample.

## 2. What is different about SciSum

| Property | OPAL's CIFAR run | SciSum |
|---|---|---|
| Run length | 120 epochs, thousands of steps, ~45 min | Preference run: 146–155 steps of 8 pairs, ~2.6 h (`q35prefcc`: 9,314 s, first step logged at 2,198 s after the reference log-probabilities, then ~47 s per step). SFT C: 2,500 steps, 61,021 s (17.0 h). |
| Validation signal = target? | yes (clean-label val loss and accuracy) | **No.** The trainer sees DPO loss, margin, pair accuracy, length difference and held-out NLL on teacher summaries. The criteria are fact recall, unsupported-or-contradicted share, length ratio, ROUGE-L and BERTScore, which need generation plus an external writer, grader and verifier. |
| Cost of one validation | seconds | Generation on the training GPU (37 GB of 48 already used) plus endpoint grading. Rough estimate: ~10–15 min for ~128 papers, not measured. |
| Getting a GPU | rented H100, immediate | 1–3 days in the queue (`AGENTS.md`). An agent that stalls or crashes the job is very expensive, and a blocking agent idles a scarce card. |
| Where the real decisions are | inside the run | **between runs.** Rounds 1–3 each changed the reward, the pair rule or the length term, and each took days. |
| Noise | one run, test accuracy differences of ~5 points | Changing only the pair order moves recall by 0.013–0.020 and length by 0.04–0.05 (round 3). This is the same size as the effects we want to detect. |
| Job model | one process | chunked Slurm jobs that resume from rolling checkpoints. Harness state (updater version, scratch, averager) would have to be saved in the checkpoint. |

### A signal the agent would have caught: DPO steps are always clipped

From `runs/*/metrics.jsonl`, the pre-clip gradient norm at the logged steps (logged every 5th step),
with the clip at `max_grad_norm = 1.0`:

| run | logged steps | median grad norm | max | share above the clip |
|---|---|---|---|---|
| q35prefcc (DPO) | 31 | 10.2 | 42.9 | 0.968 |
| q35preflc (DPO) | 30 | 11.3 | 31.7 | 1.000 |
| q35 (SFT C) | 500 | 0.62 | 6.6 | 0.05 |

**Correction (2026-11-09).** A first version of this section said the clipping sets the step size. That is
wrong for AdamW: Adam's update, about lr · m/√v, is nearly invariant to the scale of the gradient, so the
learning rate still sets the step size. What clipping every step changes is how steps are weighted:
each step enters Adam's moment estimates as a unit-norm gradient, so the late steps, whose norms are
larger (prefcc: 15.3 at step 5 → 42.9 at step 155), do not count more than the early ones. Over the
same stretch the margin rose from 0.40 to 3.49, and pair accuracy over the second half averaged 0.905.

This is not a bug, and on its own it is a weak reason to sweep the clip. The rising gradient norm next
to a margin that keeps growing is the useful signal (over-optimisation); the in-run agent reads it.

## 3. Ways to integrate an LLM agent

Ordered from inside the optimiser step outward.

| # | Integration | What the agent does | Signal it reads | What it may change | Expected value for SciSum | Cost and risk |
|---|---|---|---|---|---|---|
| I1 | **Reactive optimiser in the DPO run** (OPAL proper) | every ~20–30 steps, reads a report and edits `updater.py` | DPO loss, margin, pair accuracy, chosen/rejected log-probabilities, length difference, grad norm, held-out NLL, plus a **mini-eval** (see §4) | LR schedule, clip, warmup and decay timing, LoRA-weight EMA/SWA and the jump to it, early stop, which checkpoint is final | Medium. Targets over-optimisation: the margin running away, NLI and judge faithfulness falling below C while claim-check numbers improve. The OPAL lesson (averaging) maps directly onto the trade-off we have. | ~0.5–1 day to wire into `pref_train.py` and make it resumable. +30–60 min per run for mini-evals. Agent tokens. One extra card for the baseline. |
| I1b | Same, plus **declared objective knobs** | as I1 | as I1 | also β, NLL weight, length α (loss-side, so outside OPAL's boundary) | Medium–high, since these knobs moved results in rounds 2–3 | Breaks the "loss frozen" rule. More risk of fitting the agent's validation (Goodhart). Needs a stricter hidden test (we have one). |
| I2 | **Reactive SFT** (long runs) | as I1 on the 17 h SFT | held-out NLL, mini-eval at milestones | when to start the WSD decay; several decays branched from one stable checkpoint; averaging | Low–medium. C's held-out loss went 0.4667 → 0.4612 over the stable phase from 10,000 to 17,504 papers, then 0.4497 after the decay to 20,000. An earlier decay could save hours. But F-series findings say imitation adds no facts, so this does not move the factuality goal. | 17 h runs, rarely rerun. Low priority. |
| I3 | **Outer-loop research agent** (round level) | after each round's evaluation, reads the ledgers and `NOTES.md`, scores its last prediction, proposes the next config inside a declared search space, pre-registers criteria and predictions | `paper_docs/data/improve_*_r*.json`, the agent's own validation set (never the 1,246 test papers) | reward weights (λ, number, length, claim), pair rule and gap, β, α, k, temperature, seeds | **High.** This is where every gain so far came from (round 3: +0.033 recall at C's length). I already do it by hand. Formalising it adds the prediction-check discipline and a validation and test split for the agent. | Cheap in tokens. The danger is the number of configurations tried against one test set: keep Bonferroni, two seeds per claimed winner, and a test set the agent never reads. |
| I4 | **Pair auditor / reward debugger** | each round, reads ~50–100 sampled pairs (chosen vs rejected with the paper excerpt) and labels why chosen won: real facts, gaming the questions, hedging, shorter. Proposes reward fixes. | the pairs and the reward components | the reward function, as a proposal for the next round (no in-run edits) | High for catching reward hacking early. The training reward and the evaluation share a method (claim-check agreement 97.1%, κ 0.60); an independent reader is the missing check. | Reads paper excerpts and teacher summaries: **data-policy decision** (they must not be distributed; check whether sending them to an external LLM API is acceptable). |
| I5 | **Independent faithfulness evaluator** | grades a stratified sample (~200 papers) of C, preflc and prefcc summaries for unsupported claims, blind to the arm | summaries plus papers | nothing (evaluation only) | High: directly answers prefcc's open caveat (NLI −0.025 and judge −0.090 against C while the claim check says ±0). A fourth model family. | Same data-policy question. ~2–3 h. Must not feed back into training, or it stops being independent. |
| I6 | **The LLM in the reward** (verifier or preference judge) | replaces or joins Gemma in the claim check or pair ranking | candidates plus paper | the training labels | Possibly high (a stronger verifier, less reward noise). But the evaluation would then share a model family with training only if the same LLM is also an evaluator, so keep I5 and I6 apart. | 3,000 papers × ~8 candidates of API calls; data policy; cost. |
| I7 | **The LLM as reviser** (critique → revise) | edits the student's best sample to remove unsupported statements; the revision becomes the chosen side | sample plus paper | the training targets | Potentially the largest gain on factuality | Changes the teacher (no longer the teacher pipeline's model alone), so it is a new setting and not comparable with C as "distillation". Data policy. |
| I8 | **Ops babysitter** (the literal "baby caring") | a watchdog session on a 20–30 min cadence: queue, logs, `metrics.jsonl`, stalls, NaN, missing checkpoints, redundant jobs. Acts within fixed rules and reports. | squeue, logs, run state | resubmit or cancel its own jobs; never edits running scripts | High on our record: Run A lost 8.7 h to a filesystem stall; round 3 wrote no `ex<N>` because `SUM_IMP_N` was not set (found and fixed by hand); self-requeuing eval workers had to be cancelled by hand | Cheap. Risk: automated `scancel`, so restrict it to job names it submitted. |

## 4. Design constraints if we run I1

1. **Three-way split.**
   - Training pairs from the round's pool.
   - An agent validation set: ~128 validation-split papers excluded from training, graded with the
     training-side writer and grader (qwen38-27b writes, gemma4-31b-it grades and verifies).
   - The 1,246 test papers with the evaluation-side models, read once after the run.
2. **Mini-eval at each decision.** Greedy generation on the agent's validation papers with the training
   model (HF generate, the training process pauses), then recall and claim share through the endpoint.
   Compute nodes already reach the endpoint: the round-3 claim check ran in `improve_eval_gpu.sbatch`.
   - Before trusting it, measure the mini-eval's noise: grade the same checkpoint twice, and two seeds
     at one step.
   - Round 3 says seed noise alone is 0.013–0.020 recall on 1,246 papers, so a 128-paper signal will be
     noisy. Decide on trends over checks, not on single points.
3. **Non-blocking agent.** Training continues on the current rule while the agent thinks
   (`blocking=False`). Only the mini-eval pauses the card.
4. **Resumable.** Store the updater version, `scratch` and the averager inside `_save_rolling`. The
   agent's code edits are written to a temporary name and `mv`'d (as with job scripts).
5. **Run directory** under `runs/<run>/agent` (compute nodes cannot see the login node's `/tmp`).
   The agent runs on the login node: it only reads reports and edits a Python file, which is not computation.
   It waits with a background `until` loop.
6. **Baselines.**
   - Same seed and same pairs with `mode=keep`, on a second card at the same time.
   - Afterwards, the agent's trick run as a fixed recipe.
   - Two seeds before any claim.
7. **Safety.** The skill's `check` is a string blacklist, not a sandbox; agent code runs in-process on
   the compute node. Restrict it to `updater.py` and `analysis.py`, and read every version.

## 5. Recommended order

| Step | What | Why first | Estimate |
|---|---|---|---|
| 0 | **Strong fixed baselines, no agent:** (a) interpolate C's and prefcc's merged LoRA deltas at α ∈ {0.5, 0.75} | The OPAL lesson: most of the gain was a fixed trick. (a) needs no training and targets prefcc's NLI and judge deficit. | ~1 h code + ~2 × 3 h eval |
| 1 | **I8 ops babysitter** | cheap, immediate value, no scientific risk | ~2 h |
| 2 | **I5 independent faithfulness check** of prefcc | answers the open caveat; needs your data-policy answer | ~2–3 h |
| 3 | **I1 reactive DPO run** on prefcc's config, seed 2, next to a fixed seed-2 run | The fixed run *is* the pending second seed of prefcc, so the GPU time does double duty | ~1 day implementation + smoke test; ~4 h per run on two cards; eval ~3 h each |
| 4 | **I3 outer-loop agent** for round 4, with I4 audits | biggest lever, but only worth formalising after steps 0–3 show which knobs matter | ~0.5 day setup; rounds as now |
| later | I1b, I2, I6, I7 | change the boundary, the teacher, or the cost profile | — |

## 6. What this cannot claim, whatever happens

- A single reactive run is one sample. Round 3's seed noise is as large as the effects.
- An agent tuned on our validation papers with the training-side grader may learn that grader. Report
  test numbers with the evaluation-side models, plus NLI and the judge, and ideally I5.
- If the agent's gain is reproduced by its trick as a fixed recipe, the finding is the trick, not the
  agent.
- Credit: OPAL / *Reactive Optimizers* (github.com/xingjian-bai/opal, MIT licence), the source of the harness and the
  protocol.

## 7. Decisions of 2026-11-09 and what was built

User's answers:
1. The agent runs through an agent CLI on the machine.
2. The in-run agent may also change the loss knobs (DPO beta, likelihood weight, R-DPO length alpha), so
   it is I1b.
3. Follow the recommended order.

Correction to §2: with AdamW, always-on clipping reweights steps; it does not set the step size (see §2).
The clip sweep of step 0(b) was dropped for that reason.

| Step | What | Where | Status |
|---|---|---|---|
| 0 | C ↔ prefcc weight blends at w = 0.5 and 0.75, exact rank-128 LoRA (`[(1-w) B_C, w B_p] @ [A_C; A_p]`; identity checked to 2e-8) | `scripts/blend_adapters.py`, runs `q35ccmix50/75`, profile `slurm/models/qwen3.5-9b-r128.env`, labels `C_ccmix50/75` | done (§8, question 2) |
| baseline | prefcc's second seed (pair order seed 1), the fixed baseline of the reactive run | `slurm/models/q35-prefccs2.env`, run `q35prefccs2` | done (§8, question 1) |
| I8 | job watchdog: stalls, crashes in logs, finished runs without `ex<N>`, impossible dependencies, redundant eval workers, reactive runs without a driver | `scripts/watchdog.py` (read-only) | done |
| I5 | The LLM reads the paper and four blinded summaries (teacher, C, preflc, prefcc) of 60 fact-test papers and lists unsupported or contradicted statements | `scripts/llm_faithfulness.py` (prepare / run / report) | done at 60 papers (see below) |
| I1b | reactive harness in the preference trainer; evaluation worker; mailbox CLI; agent driver; two-card job | `scisum/reactive.py`, `scisum/reactive_eval.py`, `scisum/reactive_templates/`, `scripts/reactive_mailbox.py`, `scripts/reactive_agent.py`, `slurm/reactive_train.sbatch` | done: smoke test `smoke-rx` (job 5259174); real run `q35prefccrx` (jobs 5259530 + 5259667, 155 steps, 8 decisions), test evaluation 5259531. **Not credited** (§8, question 3: fails criterion 4; recall − prefccs2 −0.0061 [−0.0126, +0.0006]) |

Design of I1b as built:
- **Agent's validation set.** 200 pool papers (validation split) with at least 5 usable training
  questions, not in prefcc's 1,233 pairs, seeded shuffle (`state/improve/reactive_val_ids.json`; 478
  usable). Each evaluation uses the first 48. The worker refuses any overlap with the test list.
- **What the agent is scored on.** The training reward exactly as `rescore` computes it, from greedy
  summaries. It also sees the reward's parts and the held-out NLL on 100 teacher summaries.
- **Anchor.** Evaluation 0 is the starting adapter (C). Every later evaluation reports per-paper
  differences against it, with a standard error.
- **Asynchrony.** Training never waits for the agent, except up to 40 min at the end for the last
  evaluation and the choice of final weights.
- **Safety.**
  - A new program is dry-run before it is used; one that fails at load is refused.
  - A program that raises or goes non-finite within 10 steps restores the weights, the optimiser state
    and the previous program.
  - Knobs are clamped to bounds.
- **Final weights.** The agent's final choice (best evaluated, live, or the running average) is written
  into `ckpt/ex<N>` and recorded in `READY`. The live weights are kept as `.adapter-live`.

### I5 result: an LLM reads 60 papers blind (2026-11-09)

`paper_docs/data/llm_faithfulness.json`, `state/llm_faith/rows.jsonl`. 60 fact-test papers (seed 0; all
among the 1,246 evaluation papers). Four summaries per paper in a seeded random order, unlabelled. 60 of
60 parsed. Usage is equivalent to $27 at API prices.

| | statements per summary | unsupported per summary | of which major | unsupported share | major share |
|---|---|---|---|---|---|
| teacher | 32.2 | 0.68 | 0.12 | 0.022 | 0.004 |
| C | 28.7 | 1.55 | 0.55 | 0.054 | 0.019 |
| preflc | 31.1 | 2.05 | 0.87 | 0.066 | 0.028 |
| prefcc | 31.0 | 1.88 | 0.82 | 0.063 | 0.027 |

Differences per paper with 95% bootstrap intervals:

| | unsupported share | major share |
|---|---|---|
| prefcc − C | +0.008 [−0.004, +0.021] | +0.008 [−0.001, +0.017] |
| preflc − C | +0.012 [−0.001, +0.024] | +0.009 [+0.001, +0.017] |
| prefcc − preflc | −0.004 [−0.016, +0.010] | — |
| teacher − C | −0.032 [−0.046, −0.019] | — |

Reading:
- The independent reader puts prefcc between C and preflc, in the same order as the claim check, but
  with a point estimate of +0.008 against C, where the claim check gives +0.0003 [−0.0046, +0.0050].
- At 60 papers its interval includes both 0 and +0.02. So it neither confirms nor refutes criterion 4
  for prefcc; it is consistent with the NLI and judge deficits being real but small.
- The reader's ordering of teacher < C is large and clear, which shows the instrument can separate
  systems.
- About 4× the papers (~$110-equivalent of usage) would halve the interval.

## 8. Protocol for the step-0 blends, prefcc's second seed and the reactive run

Written 2026-11-09, 08:10 local, before any of these results existed: the blends'
generations were half written and not scored, `q35prefccs2` was training and `q35prefccrx` had just
started.

**Arms.** All are scored on the 1,246 test papers with the evaluation-side models, like round 3.

| arm | what | label |
|---|---|---|
| `prefcc` (two seeds) | per-paper mean of `prefcc` and `prefccs2`, as round 3 did for `preflc` and `prefrd` | `C_prefcc`, `C_prefccs2` |
| `ccmix50`, `ccmix75` | C ↔ prefcc weight blends at w = 0.5 and 0.75 (no training) | `C_ccmix50`, `C_ccmix75` |
| `prefccrx` | the reactive run (I1b): prefccs2's pairs and pair order, with the agent steering | `C_prefccrx` |

**Primary: round 3's four criteria against C, thresholds unchanged.**
1. Fact recall above C's, with the bootstrap interval excluding zero and the permutation p below the
   Bonferroni level 0.05 / 4 = 0.0125 (four arms against C).
2. Length ratio within 0.90–1.10.
3. ROUGE-L and BERTScore each ≥ C − 0.01.
4. Unsupported-or-contradicted share − C: the interval includes zero or lies below it.

**Question 1: does prefcc hold over two seeds?** The verdict comes from the four criteria on the two-seed
mean. Seed 2 alone and the seed difference (seed 2 − seed 1) are reported with intervals, as for `preflc`
and `prefrd`. If the two-seed mean misses a criterion, prefcc is no longer the recommended arm.

**Question 2: are the blends strong fixed baselines?** A blend is reported as *better than prefcc* only if:
- it meets all four criteria; and
- its NLI − C and judge-faithfulness − C are both closer to zero than prefcc's (seed 1: −0.025 and
  −0.090), with the paired interval of blend − prefcc on that metric excluding zero.

Otherwise it is reported descriptively: how much of prefcc's recall gain it keeps, and how much of its
NLI and judge deficit it closes. Expected before reading anything: recall and the deficits both fall
roughly in proportion to w. A blend that keeps more than its share of the recall gain is the
interesting outcome.

**Question 3: does the agent beat fixed training?**
- The same-seed comparison is `prefccrx` − `prefccs2` (same pairs, same order), paired over papers, on
  recall, unsupported share, NLI and the judge.
- `prefccrx` is also compared with the better blend.
- The agent is credited only if `prefccrx` meets the four criteria and its recall − prefccs2 interval
  excludes zero without a worse unsupported share (interval of the difference including zero or lying
  below it).
- One reactive run is one sample. Seed noise is 0.013–0.020 recall (round 3). A positive result is
  therefore *a candidate*; it needs a second reactive seed, and the agent's trick run as a fixed recipe
  (§6), before any claim.
- The agent's own validation numbers (training reward on 48 validation papers) are reported, but never
  as evidence.

### Result, question 2: the C ↔ prefcc blends (2026-11-09 11:04 local)

`scripts/improve_r4.py` → `paper_docs/data/improve_r4.json`. 1,246 test papers. Fact recall on 899 papers
(both question sets), claim check on 300, NLI on 300, the judge on 499. Intervals: 2,000 paired bootstrap
resamples; p: 10,000 sign flips.

| against C | `ccmix50` (w = 0.5) | `ccmix75` (w = 0.75) | `prefcc` seed 1 (round 3) |
|:--|--:|--:|--:|
| fact recall | **+0.0201** [+0.0139, +0.0263], p = 0.0001 | **+0.0286** [+0.0218, +0.0353], p = 0.0001 | +0.0327 [+0.0259, +0.0400] |
| length ratio (mean) | 1.003 | 1.011 | 1.017 |
| ROUGE-L | +0.0013 | −0.0019 | −0.0054 |
| BERTScore | −0.0003 | −0.0033 | −0.0070 |
| unsupported-or-contradicted share | +0.0015 [−0.0033, +0.0063] | +0.0041 [−0.0011, +0.0093] | +0.0003 [−0.0046, +0.0050] |
| NLI | −0.0156 [−0.0245, −0.0071] | −0.0218 [−0.0319, −0.0117] | −0.025 [−0.035, −0.015] |
| judge faithfulness (1–5) | −0.078 [−0.134, −0.024] | −0.074 [−0.126, −0.020] | −0.090 [−0.146, −0.036] |
| all four criteria | **yes** | **yes** | yes |

| blend − prefcc (paired) | `ccmix50` | `ccmix75` |
|:--|--:|--:|
| fact recall | −0.0126 [−0.0189, −0.0062] | −0.0041 [−0.0101, +0.0018] |
| ROUGE-L / BERTScore | +0.0068 / +0.0068 (both intervals above 0) | +0.0036 / +0.0037 (both above 0) |
| NLI | +0.0094 [−0.0000, +0.0196] | +0.0032 [−0.0065, +0.0128] |
| judge faithfulness | +0.012 [−0.034, +0.056] | +0.016 [−0.028, +0.060] |
| unsupported share | +0.0018 [−0.0030, +0.0070] | +0.0038 [−0.0008, +0.0084] |

Verdict by §8:
- **Both blends meet all four criteria. Neither is better than prefcc:** no blend − prefcc interval on NLI
  or the judge excludes zero.
- The judge deficit hardly moves: −0.090 for prefcc, −0.078 at w = 0.5. At half of prefcc's weights the
  judge still sees almost all of the deficit.
- **The recall gain is not proportional to w; the blends keep more than their share.** w = 0.75 keeps 87%
  of prefcc's gain (0.0286 of 0.0327) and w = 0.5 keeps 61%. The NLI deficit shrinks roughly in
  proportion (62% and 87% of prefcc's).
- So `ccmix75` is a fixed baseline within noise of prefcc on recall (−0.0041 [−0.0101, +0.0018]), with
  ROUGE-L and BERTScore closer to C's.
- This is one blend of one prefcc seed. Whether a blend is *preferable* to prefcc is a trade-off the
  protocol did not pre-register, so it is not claimed.

### Result, question 1: prefcc over two seeds (2026-11-09 18:15 local)

`q35prefccs2`: the same 1,233 pairs and settings, pair order shuffled with seed 1. Seed 1 (`q35prefcc`) used the
stored order, which is the validation split's own seeded order, so it is also a random order. Same papers and
instruments as question 2.

| against C | two-seed mean (the verdict) | seed 1 | seed 2 | seed 2 − seed 1 |
|:--|--:|--:|--:|--:|
| fact recall | **+0.0262** [+0.0201, +0.0322], p = 0.0001 | +0.0327 | +0.0197 [+0.0124, +0.0266] | −0.0130 [−0.0201, −0.0060] |
| length ratio (mean) | 0.988 | 1.017 | 0.959 | −0.0585 [−0.0648, −0.0520] |
| ROUGE-L | −0.0053 | −0.0054 | −0.0052 | +0.0002 |
| BERTScore | −0.0085 | −0.0070 | −0.0099 | −0.0029 [−0.0063, +0.0003] |
| unsupported-or-contradicted share | **+0.0045** [−0.0003, +0.0093] | +0.0003 [−0.0046, +0.0050] | +0.0087 [+0.0024, +0.0149] | +0.0084 [+0.0028, +0.0144] |
| NLI | −0.0243 [−0.0341, −0.0152] | −0.025 | −0.0237 | +0.0013 |
| judge faithfulness (1–5) | −0.090 [−0.139, −0.039] | −0.090 | −0.090 | +0.000 [−0.046, +0.046] |
| all four criteria | **yes** | yes | no (criterion 4) | |

(The equal judge means are a coincidence: 71 of 500 papers get a different score, and the sums happen to
agree.)

Verdict by §8:
- **prefcc meets all four criteria on the mean of its two seeds**, so it stays the recommended arm. Its
  recall gain over two seeds is +0.026, not round 3's single-seed +0.033.
- **Criterion 4 now passes narrowly.** The interval's lower end is −0.0003. Seed 2 alone fails it
  (+0.0087, interval above zero). So "no more unsupported statements than C" holds for prefcc's two-seed
  mean, not for every seed.
- Seed noise repeats round 3. The shuffled order wrote shorter (−0.059) and recalled less (−0.013),
  with a higher unsupported share (+0.008).
  - That is the same direction as in `preflc` and `prefrd`: 3 of 3 arms. If the order were
    irrelevant, a same-direction result in all three would happen with probability 0.25, so nothing is
    claimed.
  - NLI and the judge do not move with the seed. Their deficits look tied to the recipe, not the order.
- Against the blends: `ccmix75` (+0.0286 recall, +0.0041 unsupported) sits between prefcc's two seeds on
  both. It is one blend of seed 1 only.

### Result, question 3: the reactive run (2026-11-10 00:36 local)

`q35prefccrx`: prefccs2's 1,233 pairs in prefccs2's order, with an LLM agent steering the
optimiser and loss knobs from 8 reports (D0–D7, `runs/q35prefccrx/agent/decisions.md`). Final weights by the agent's
rule fixed at D6: the EMA (decay 0.95, steps 115–155). `ckpt/ex1233` has the same sha1 as that snapshot. The final step
did not time out. Same papers and instruments as questions 1 and 2.

| | `prefccrx` − C | `prefccrx` − `prefccs2` (same pairs and order) | `prefccrx` − `ccmix75` (the better blend) |
|:--|--:|--:|--:|
| fact recall | +0.0136 [+0.0069, +0.0203], p = 0.0002 | **−0.0061 [−0.0126, +0.0006]** | −0.0150 [−0.0218, −0.0083] |
| length ratio (mean) | 0.947 (−0.0447 [−0.0510, −0.0382]) | −0.0121 [−0.0183, −0.0054] | −0.0648 [−0.0715, −0.0577] |
| ROUGE-L | −0.0041 | +0.0011 [−0.0012, +0.0035] | −0.0023 [−0.0046, −0.0000] |
| BERTScore | −0.0098 | +0.0001 [−0.0042, +0.0044] | −0.0065 [−0.0105, −0.0029] |
| unsupported-or-contradicted share | **+0.0082 [+0.0018, +0.0146]** | +0.0003 [−0.0055, +0.0065] | +0.0050 [−0.0008, +0.0106] |
| NLI | −0.0277 [−0.0374, −0.0177] | −0.0040 [−0.0137, +0.0057] | −0.0059 [−0.0158, +0.0052] |
| judge faithfulness (1–5) | −0.086 [−0.142, −0.030] | +0.004 [−0.036, +0.044] | −0.012 [−0.062, +0.038] |
| criteria | 1 ✅ 2 ✅ 3 ✅ (BERTScore by 0.0002) **4 ❌** | | |

Summaries at the token limit: 5 of 1,246 (`prefccs2`: 2).

Verdict by §8:
- **The agent is not credited, not even as a candidate.** `prefccrx` fails criterion 4: its unsupported share
  is above C's, with the interval above zero. Its recall − `prefccs2` interval does not exclude zero, and the
  point estimate is negative.
- Against its same-seed fixed baseline, the reactive run is level on unsupported share, NLI, the judge,
  ROUGE-L and BERTScore. It writes 1.2% shorter and recalls 0.006 less (interval just includes zero).
  So the agent's interventions bought nothing measurable on the test papers.
- Criterion 4 fails for the same reason as `prefccs2` alone (+0.0087): the difference between the two is +0.0003.
  The failure belongs to this pair order, not to the agent.
- Against the better blend `ccmix75`, which needs no training, `prefccrx` recalls 0.015 less, with the
  interval below zero.

What the agent did (validation reward on 48 papers; C = 0.5808):

| decision | step | validation reward | action | prediction check |
|:--|--:|:--|:--|:--|
| D0 | 0 | 0.5808 (C) | none | — |
| D1 | 21 | 0.5732 | none | 4 of 4 held (loose bounds) |
| D2 | 45 | 0.5631 | len_alpha 0 → 0.01 (claim share +2.2 SE, length up) | mostly held; NLL just below its band |
| D3 | 68 | 0.5906 | none | all held, but the pairs of steps 46–68 were not length-biased, so it could not credit len_alpha |
| D4 | 91 | 0.5575 | `restore_best` to step 68, len_alpha → 0 (length 0.945, recall down) | **3 of 6 wrong**; both pre-set triggers fired |
| D5 | 112 | 0.5839 | EMA 0.95 from step 115 | 7 of 7 held |
| D6 | 134 | live 0.5989, avg 0.6066 | none; fixed the final-choice rule | all held |
| D7 | 155 | live 0.6017, avg 0.6008 | final = avg (by the D6 rule) | all held |

D7's predictions for the test papers, written before the evaluation: recall − C +0.01 to +0.03 (**held**,
+0.0136); length 0.93–1.00 (**held**, 0.947); ROUGE-L ≥ C − 0.008 (**held**, −0.0041); BERTScore ≥ C − 0.010
(**held** by 0.0002); unsupported share "could go either way" (not a falsifiable prediction; it came out above C).
On validation the final average's claim share was +0.006 ± 0.007 against C. The test claim check confirms the
sign, with the interval above zero.

Reading and caveats:
- One reactive run is one sample. A negative result is not a proof that an in-run agent cannot help; it shows
  that this agent, with these knobs, a 48-paper validation signal and 155 steps, did not beat fixed training
  on the same pairs.
- The restore at D4 discarded the pairs of steps 69–95 (~200 of 1,233). The optimiser state and the pair counter
  continued. The run therefore trained on ~16% fewer pairs than `prefccs2`, which may account for part of the
  recall gap. That cannot be separated from the length term or the EMA in one run.
- The validation signal was too noisy to steer by. Between candidates with almost identical weights (NLL 0.4816
  and 0.4812), the agent itself measured fact-score differences of about one SE (D6). Its two interventions
  were triggered by 2.2 SE and 4.2 SE signals. Only the second (length) was clearly real, and the agent itself
  caused it with len_alpha.
- The agent's reasoning was sound and well calibrated: of its seven prediction checks (D1–D7), five held fully, D2 mostly, and D4 missed 3 of 6 items. It pre-declared
  its final rule, and it avoided selecting the maximum (`e06-avg`). Its test predictions held. Calibration is not
  the same thing as improvement.
- No second reactive seed and no fixed-recipe re-run: §8 calls for them only after a positive result.

## 9. Round 4: an outer-loop agent designs the round (I3), with a blind pair audit (I4)

Protocol written 2026-11-10, 07:04 local, before any of it was built or run. The user's instruction
of 2026-11-10 ("start the recommendation") starts step 4 of §5. Why this step: every gain so far came from choices
made between rounds (the reward, the pair rule), and steering inside a run did not help (F12).

**Roles.**
- *Designer*: an LLM agent through an agent CLI, in its own persistent session. It reads
  a ledger of the earlier rounds (the aggregate test tables of `11` and §8, as in these documents), the pair
  audit and its own validation results. It can run one tool (`scripts/outer_tool.py`) and write only its own
  notes. It never reads a test paper, a test summary or a per-paper test score.
- *Auditor*: the same LLM, no tools, reads one paper and two summaries without knowing which one the reward chose.
- *Lead* (this session): builds the harness, checks a proposal against the declared space, launches jobs and
  writes the results. The lead does not choose arms or finalists.

**Papers.**
- Training pairs come from the 1,520 sampled pool papers, as in rounds 1–3 (the teacher's summary and C's five
  samples, already graded and claim-checked). No new sampling in this round.
- *V*, the designer's validation set: 600 pool papers that were never sampled, so no arm can train on them, with at
  least 5 kept training questions, in a seeded order. Today 280 such papers have reference answers; the reference
  readers are run for the other pool papers that have questions. The never-sampled usable papers beyond 600 are a
  reserve that nothing in this round reads.
- V is scored with the training-side instruments (questions by Qwen3.8-27B, answers and grades by Gemma-4-31B,
  claim check by Gemma-4-31B), on greedy summaries. The 1,246 test papers and the evaluation-side models are used
  only for finalists.

**Step 1. Calibrate V before anyone uses it.** Nine existing adapters with known test results are scored on V:
C, `ctl`, `pref`, `preflc`, `preflcs2`, `prefrd`, `prefcc`, `prefccs2`, `prefccrx`. C's V summaries are graded
twice to measure the grading noise. Reported per arm: V differences against C (fact score, claim share, length,
ROUGE-L) next to the test differences (recall, unsupported share, length, ROUGE-L), and Spearman's rank
correlation over the eight arms. Reading fixed now: V can rank arms on a metric if the correlation is at least
0.6 and V has the test's sign for every arm whose test interval excludes zero. Where it cannot, the designer is
told so.

**Step 2. Pair audit (I4).** 80 of `prefcc`'s 1,233 pairs, seeded: 50 whose chosen side is the teacher's
summary and 30 whose chosen side is a sample (the arm has 940 and 293). The auditor reads the paper and the two
summaries as A and B in a seeded random order. It lists each summary's unsupported or contradicted statements,
names the important content one has and the other lacks, and says which it would rather train toward: on
faithfulness, on coverage and overall. Reported: how often the auditor's overall preference is the reward's
chosen side (with a bootstrap interval, weighted back to the arm's mix), by stratum and by reward gap, and what
separates the pairs where they disagree. The audit is descriptive. It is 80 pairs read by one model.

**Step 3. Proposal.** The designer proposes at most 4 trained arms, each with a hypothesis and numeric
predictions on V, and any number of weight blends of those arms with C. Every arm starts from C's 20,000-paper
adapter and makes one pass on one 48 GB card. The declared space:

| knob | range (prefcc's value) |
|:--|:--|
| reward: imitation weight, unsupported-number penalty, length penalty, free length, claim weight, margin | 0–0.5 (0.15), 0–1 (0.30), 0–3 (1.0), 0.9–1.3 (1.0), 0–5 (1.0), 0–0.3 (0.05) |
| pairs: minimum reward gap, rule, pairs per paper | 0.05–0.8 (0.25), `lowest` or `closest_length` (`closest_length`), 1–3 (1) |
| pairs: a filter, a boolean expression over the stored parts of the chosen and the rejected summary | none |
| loss: DPO beta, likelihood weight, R-DPO length alpha, learning rate | 0.02–0.5 (0.1), 0–4 (1.0), 0–0.005 (0), 5e-6–5e-5 (2e-5) |
| after training: blend weight with C | 0–1 |

The pair filter and several pairs per paper are new code, written and tested before the designer starts. The
designer may also recommend changes outside this space (new sampling, more pool papers, another sampling policy)
for a later round. They are not run in this one.

**Step 4. Train and score on V.** Each arm is trained once with the pairs in stored order, as `prefcc`'s first
seed, and scored on V.

**Step 5. Finalists.** From the V results the designer names at most 2 finalists (arms or blends), with
predictions for the test, or none. For each finalist a second seed is trained (pair order seed 1, as
`prefccs2`). Both seeds are evaluated on the 1,246 test papers with the usual instruments.

**Verdict, fixed now.**
1. *The four criteria against C* (round 3's, thresholds unchanged) on the finalist's two-seed mean. Recall's
   permutation p must be below 0.05 / 2 = 0.025 (at most two finalists).
2. *A finalist replaces prefcc as the recommended arm only if all three hold:*
   - it meets the four criteria;
   - against prefcc's two-seed mean, paired over papers, it is not worse on recall or on the unsupported share
     (the 95% interval of the difference includes zero or lies on the better side);
   - it is better on at least one of recall, unsupported share, NLI and judge faithfulness: the 98.75% interval
     (0.05 / 4) of the difference excludes zero on the better side.
3. If the designer names no finalist, or no finalist passes rule 2, prefcc stays the recommended arm and the
   round is reported as a null result.
4. The V numbers are never evidence for an arm. The calibration of step 1 and the designer's prediction checks
   are reported descriptively.

**What this cannot claim.**
- One round is one sample of the designer. A finalist that passes rule 2 is supported as an *arm* by two seeds.
  "An agent designs better rounds than a person" is not shown by one round, whatever its outcome.
- The designer reads the aggregate test tables of 14 earlier arms, as a person designing round 4 would. So the
  1,246 test papers are no longer untouched by the design process. A finalist that replaces prefcc should be
  confirmed on test papers outside the 1,246 before it is published as the result. That is not part of this round.
- V is scored by the reward's own models. An arm can look better on V by learning that reward's quirks. This is
  why finalists are judged only on the test instruments.

### Result, step 2: the pair audit (written 2026-11-15)

Run 2026-11-10, inputs written 07:22, last pair 07:38 local, `scripts/pair_audit.py` (the LLM through an agent CLI,
no tools, 4 in parallel, ~45 s a pair): 80 of 80 pairs parsed, none failed. Aggregates in
`paper_docs/data/pair_audit.json`; per-pair detail (with the auditor's reasons) only in `state/pair_audit/`.
Shares are weighted back to the arm's mix (teacher-chosen 940/1,233 = 0.762, sample-chosen 0.238); intervals are
bootstrap 95%.

| auditor's preference | chosen (weighted) [95%] | rejected | tie | teacher stratum (n = 50): chosen / rejected | sample stratum (n = 30): chosen / rejected |
|:--|--:|--:|--:|--:|--:|
| faithfulness | 0.730 [0.631, 0.822] | 0.110 | 0.160 | 0.76 / 0.04 | 0.63 / 0.33 |
| coverage | 0.653 [0.545, 0.760] | 0.109 | 0.239 | 0.70 / 0.08 | 0.50 / 0.20 |
| **overall** | **0.814 [0.728, 0.890]** | 0.156 | 0.031 | **0.88 / 0.08** | **0.60 / 0.40** |

The auditor's own count of unsupported or contradicted statements per statement: chosen 0.048, rejected 0.085
(difference −0.037 [−0.051, −0.023]). By stratum: teacher-chosen pairs 0.035 against 0.082; sample-chosen pairs
0.090 against 0.093 (−0.003). Statements the auditor marked *major*: chosen 0.005, rejected 0.033.

What separates the 16 disagreements (overall = rejected side; 4 teacher-chosen, 12 sample-chosen):
- **Stratum.** 4 of 50 teacher-chosen pairs against 12 of 30 sample-chosen pairs. When the reward prefers one of
  C's samples over another, the auditor agrees 18 times and disagrees 12 times; it finds the chosen sample no
  more faithful than the rejected one.
- **Reward gap, within the teacher stratum.** No disagreement among the 25 teacher-chosen pairs with a gap at or above
  the median (0.351 over all 80), 4 among the 25 below. All 4 had a fact gap below 0.20 and the chosen (teacher)
  summary was not the longer one; the auditor's confidence was low in 3 of them and medium in 1.
- **Within the sample stratum nothing stored separates them.** Disagreements: 5 of 15 with a gap at or above the median, 7 of 15
  below; 11 of 24 with a fact gap ≥ 0.20, 1 of 6 below; 5 of 13 when the chosen sample is the longer, 7 of 17
  otherwise. The auditor's confidence in the 12: 7 low, 5 medium, none high.
- The auditor named the main difference as errors in 47 of 80 pairs, facts in 19, specificity in 13 (in the 16
  disagreements: errors 8, facts 4, specificity 4). Its confidence over all 80: high 23, medium 32, low 25.

Reading (descriptive, as fixed above; 80 pairs, one reader): the reward's choice agrees with the auditor's overall
preference in about four of five pairs. The agreement is high when the reward picks the teacher's summary and
near chance when it picks between two samples. A pair filter therefore has more to act on in the sample stratum,
but no stored part identifies the disagreements there; a filter on the reward gap would act on the teacher
stratum's 4. The designer receives these aggregates and the per-pair detail.

### Result, step 1: the calibration of V (written 2026-11-15)

Nine adapters scored on V (600 papers, greedy summaries, training-side instruments) by jobs 5262367–9
(2026-11-10) and the gradings 5301719 (`prefccs2`, whose job a timed out mid-grading) and 5301720 (C again),
2026-11-15. `scripts/outer_calibrate.py` → `paper_docs/data/outer_calibration.json`. C on V: fact 0.5761,
claim share 0.0353, length ratio 0.9815, ROUGE-L 0.4792. Test columns are the single-seed test differences
from `11` and §8 (`ctl` and `pref`: round 1; the others: rounds 2–4).

| arm | V fact − C [95%] | test recall − C [95%] | V claim share − C | test unsupported − C [95%] | V length − C | test length − C | V ROUGE-L − C | test ROUGE-L − C |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `ctl` | +0.0111 [−0.0015, +0.0237] | −0.0029 [−0.0095, +0.0032] | −0.0018 | +0.0065 [+0.0012, +0.0119] | +0.0015 | −0.0104 | +0.0030 | +0.0015 |
| `pref` | +0.0990 [+0.0841, +0.1140] | +0.0711 [+0.0631, +0.0790] | +0.0048 | +0.0121 [+0.0064, +0.0178] | +0.2897 | +0.2720 | −0.0240 | −0.0267 |
| `preflc` | +0.0525 [+0.0392, +0.0658] | +0.0299 [+0.0228, +0.0369] | +0.0019 | +0.0085 [+0.0031, +0.0144] | +0.0390 | +0.0203 | −0.0070 | −0.0063 |
| `preflcs2` | +0.0498 [+0.0370, +0.0622] | +0.0169 [+0.0091, +0.0240] | +0.0053 | +0.0076 [+0.0022, +0.0134] | −0.0041 | −0.0161 | −0.0094 | −0.0059 |
| `prefrd` | +0.0321 [+0.0190, +0.0453] | +0.0171 [+0.0100, +0.0239] | +0.0064 | +0.0057 [−0.0001, +0.0113] | −0.0404 | −0.0524 | −0.0048 | −0.0068 |
| `prefcc` | +0.0422 [+0.0278, +0.0558] | +0.0327 [+0.0262, +0.0397] | −0.0015 | +0.0003 [−0.0046, +0.0050] | +0.0376 | +0.0258 | −0.0049 | −0.0054 |
| `prefccs2` | +0.0324 [+0.0188, +0.0455] | +0.0197 [+0.0124, +0.0266] | +0.0025 | +0.0087 [+0.0024, +0.0149] | −0.0233 | −0.0327 | −0.0058 | −0.0052 |
| `prefccrx` | +0.0179 [+0.0032, +0.0315] | +0.0136 [+0.0069, +0.0203] | +0.0038 | +0.0082 [+0.0018, +0.0146] | −0.0353 | −0.0447 | −0.0034 | −0.0041 |

The fixed reading (Spearman ρ over the eight arms ≥ 0.6, and V has the test's sign for every arm whose test
interval excludes zero):

| V metric → test metric | ρ | sign agrees where the test interval excludes zero | V can rank arms |
|:--|--:|:--|:--|
| fact → recall | 0.81 | 7 of 7 | **yes** |
| correct share → recall | 0.81 | 7 of 7 | yes |
| ROUGE-L → ROUGE-L | 0.71 | 7 of 7 | yes |
| length ratio → length ratio | 0.98 | 7 of 8 (`ctl`: V +0.0015, test −0.0104) | no, by the rule |
| claim share → unsupported share | 0.14 | 5 of 6 (`ctl` again) | **no** |

Grading noise (C's 600 summaries graded twice): second − first fact +0.0016 [−0.0006, +0.0039], 24 of 600 papers
changed (mean absolute change per paper 0.005); claim share +0.0003, 10 papers changed; reward +0.0013. The
grading's SE (0.0011 on the fact score) is small next to the paired SE of an arm's V fact difference against C
(0.006–0.007), which comes from the papers.

Reading:
- **V ranks arms on fact recall.** V's fact differences are larger than the test's (by about 1.3–3×; `preflcs2`
  +0.050 against +0.017) but in the same order, with every sign right. So V can rank recall, but its fact
  differences cannot be read as test recall gains.
- **V cannot rank arms on faithfulness.** Its claim share does not order the arms as the test's unsupported share
  does (ρ 0.14): `ctl` and `prefcc` are the two arms with a V claim share below C's, yet `ctl`'s test share is
  above C's (+0.0065), and `prefccs2`'s test deficit (+0.0087) shows on V as +0.0025. Criterion 4 and rule 2 (unsupported share not worse than prefcc's)
  therefore have no reliable signal on V. The designer is told this; a finalist's faithfulness is known only from
  the test.
- **Length.** V's length ratio orders the arms almost exactly (ρ 0.98) and sits 0.01–0.02 above the test's for
  every arm, which flips the sign only for `ctl` (a −0.010 test effect). By the rule fixed in advance, V cannot rank
  length; read as an offset, it tracks the test's length closely. Both are reported to the designer.

### Step 3: the designer's proposal (accepted 2026-11-15 21:36 local, before any arm was trained)

`scripts/outer_agent.py --phase propose`: the LLM agent through an agent CLI,
686 s, accepted at the first submission. Proposal and notes, unedited:
`paper_docs/data/outer_r4/proposal.accepted.json` and `paper_docs/data/outer_r4/NOTES.md`.

Audit of the designer's access (the lead read the session transcript): 51 tool calls (13 Read, 30 Bash, 2 Write,
6 Edit), 2 denied. It read `11`, `12`, `06`, `outer_calibration.json`, the pair audit report, its TASK and NOTES; it
wrote only `proposal.json` and NOTES.md in its agent directory; no call named `runs/`, `state/improve`,
`state/factcheck`, `state/claimcheck`, `reports/`, `.env` or the test ids. The CLI approved read-only shell
commands (`ls`, `wc`, `grep`, `jq`) by itself, outside the allow list, so the Read rules alone did not fence those
directories; phases 2 and 3 run with deny rules that refuse any shell command naming them (tested: `ls runs/` is
refused although `ls` is allowed).

| arm | change from prefcc | pairs (papers) | sample-chosen | hypothesis | V prediction, fact − prefcc | falsified on V if |
|:--|:--|--:|--:|:--|:--|:--|
| `r4a` | `imp_pair_filter = c_teacher` (drop the sample-chosen pairs) | 940 (940) | 0 | The auditor finds the chosen sample no more faithful than the rejected one; training only toward the teacher's text lowers the unsupported share at a small recall cost | −0.024 to +0.006 | fact < prefcc − 0.020, or ROUGE-L not above prefcc's |
| `r4b` | `imp_claim_weight = 3` | 1,347 (1,347) | 376 | The claim penalty's dose-response (weight 0 → 1: unsupported −0.007 on test) continues | −0.027 to +0.003 | V claim share not below prefcc's, or fact < prefcc − 0.025 |
| `r4c` | `imp_pairs_per_paper = 2` | 2,165 (1,233) | 486 | A second rejected sample per paper adds contrast and recall at an unchanged unsupported share | −0.010 to +0.028 | fact not ≥ prefcc + 0.005 with length within 0.03 of prefcc's |
| `r4d` | all three | 1,808 (971) | 0 | Faithfulness from the filter and the claim weight; the second negative wins back their recall cost | −0.027 to +0.008 | fact < prefcc − 0.015 |
| `r4cw75` | blend: 0.75 × `r4c` + 0.25 × C | – | – | Keeps ≥ 80% of r4c's V fact gain over C and recovers ≥ a third of its ROUGE-L deficit | – | – |

Full prediction intervals (against C and against prefcc on fact, claim share, length and ROUGE-L) are in the
proposal. The designer's finalist rule, written before V: `r4d` is the faithfulness finalist if its V fact is
no more than about 0.010 below prefcc's (otherwise `r4a` or `r4b`, whichever keeps more fact); `r4c` (or
`r4cw75` if r4c's ROUGE-L is more than 0.003 below prefcc's) is the recall finalist if its V fact is at least
about 0.015 above prefcc's. It names V's resolution against prefcc as about 0.017 fact (paired SE 0.007 plus
the pair-order noise). It chose not to test β, lr, the likelihood weight (its next choice), R-DPO α, the length
allowance, λ, the `lowest` rule, gap filters, `imp_beta_num`, claim weight 5 and k = 3, with reasons in NOTES.md.
Outside the space it recommends a V claim check with the evaluation-side verifier, a second verifier for the
candidates' claim check, more sampled pool papers, two pair orders per arm on V, and on-policy samples from prefcc.

### Step 4: the V scores, the designer's prediction check and its finalists (accepted 2026-11-16 09:29 local, before any test result)

**V scores.** The five arms were generated on V by 05:09 local. The first grading attempt ran five gradings at
once and saturated the training-side grader (Gemma-4-31B): 2,000–4,200 read timeouts per job, and no grading
finished within the 5 h limit. Graded again one after another from the stored summaries (1,263–1,455 s each, no
timeouts). Every arm has 597–600 of 600 papers scored. r4b has 1, r4c 2 and r4cw75 3 summaries at the token
limit. Full report: `paper_docs/data/outer_r4/val_report.json`. Training-side reward on V; these numbers guide the
designer and are not evidence.

| adapter | fact | claim share | unsupported numbers | length ratio | ROUGE-L | fact − prefcc [95% CI] | claim share − prefcc [95% CI] | ROUGE-L − prefcc |
|:--|--:|--:|--:|--:|--:|:--|:--|--:|
| C | 0.5761 | 0.0353 | 0.0264 | 0.982 | 0.4792 | −0.0422 [−0.0558, −0.0278] | +0.0015 [−0.0023, +0.0055] | +0.0049 |
| prefcc (seed 1) | 0.6182 | 0.0337 | 0.0326 | 1.019 | 0.4743 | – | – | – |
| `r4a` filter | 0.6078 | 0.0382 | 0.0281 | 1.010 | 0.4785 | −0.0110 [−0.0235, +0.0031] | +0.0045 [+0.0007, +0.0084] | +0.0042 |
| `r4b` claim 3 | 0.6129 | 0.0280 | 0.0277 | 1.026 | 0.4724 | −0.0048 [−0.0176, +0.0090] | −0.0057 [−0.0094, −0.0022] | −0.0018 |
| `r4c` k = 2 | 0.6348 | 0.0358 | 0.0279 | 1.068 | 0.4729 | +0.0148 [+0.0013, +0.0278] | +0.0022 [−0.0015, +0.0059] | −0.0018 |
| `r4cw75` | 0.6214 | 0.0342 | 0.0271 | 1.051 | 0.4754 | +0.0020 [−0.0122, +0.0153] | +0.0007 [−0.0029, +0.0042] | +0.0006 |
| `r4d` all three | 0.6035 | 0.0353 | 0.0288 | 1.029 | 0.4780 | −0.0152 [−0.0289, −0.0017] | +0.0017 [−0.0022, +0.0058] | +0.0037 |

The V proxy of the four criteria against C passes for every arm, so it separates nothing.

**Prediction check (by the designer, before choosing).** 26 of 32 prediction intervals held.
- All four length misses are on the long side. Every round-4 arm wrote longer than C on V (+0.028 to +0.087).
  The preview's rejected/chosen length median did not predict the direction of output length.
- `r4d` is falsified by its own clause: fact −0.0152 is more than 0.015 below prefcc's.
- `r4c`'s clause asked for its fact gain at a length within 0.03 of prefcc's. Its length is +0.049, so the
  clause cannot be read cleanly.
- `r4b` held 8 of 8.
- `r4a`'s ROUGE-L gain over prefcc (+0.0042, interval excluding zero) was larger than predicted.
- `r4cw75` kept 77% of r4c's fact gain (predicted ≥ 80%) and recovered 33% of its ROUGE-L deficit (predicted
  ≥ a third).

**Finalists: `r4b` and `r4a`.**
- `r4b` follows the pre-written rule: r4d is out, and of r4a and r4b, r4b keeps more fact.
- `r4a` departs from the pre-written plan, which named `r4c` as the recall finalist at a fact gain of about
  +0.015 or more. r4c reached +0.0148, but at +0.049 length. The designer attributes +0.003 to +0.012 of the
  gain to length, from earlier fact-vs-length slopes on V, and dropped r4c.
- It named `r4a` instead: the best criterion-3 margin of any preference arm, and the cheapest second seed.
  The pair also asks whether the training grader's claim flags on sample-chosen pairs are real:
  - r4b lowers Gemma's claim share and r4a raises it.
  - The blind auditor (step 2) found the sample-chosen contrast not real.
  - If r4a matches or beats r4b on the test's unsupported share and NLI, the auditor was right.
- The designer puts the chance of each replacing prefcc at about 0.1, and both test predictions say
  `replaces_prefcc: false`.

| finalist | recall − C | unsupported − C | length ratio | ROUGE-L − C | BERTScore − C | NLI − C (in NOTES) |
|:--|:--|:--|:--|:--|:--|:--|
| `r4b` | +0.014 to +0.032 | −0.006 to +0.007 | 0.96 to 1.04 | −0.009 to −0.004 | −0.014 to −0.007 | −0.025 to −0.005 |
| `r4a` | +0.012 to +0.030 | −0.005 to +0.009 | 0.95 to 1.03 | −0.004 to +0.001 | −0.007 to −0.001 | −0.025 to −0.005 |

Test predictions are for the two-seed mean, with length absolute, relative to the teacher. Finalists, prediction
check and notes, unedited: `paper_docs/data/outer_r4/finalists.accepted.json` and
`paper_docs/data/outer_r4/NOTES.md`.

**Access audit (phase 2).** Same session as phase 1, 342 s, 11 tool calls (8 Bash, 1 Write, 2 Edit). It ran
`outer_tool.py val` and `select`, read their output with `jq`, and wrote `finalists.json` and NOTES.md. One shell
call was refused (a variable in the command could not be checked against the deny rules) and was re-run with a
literal path. No call named `runs/`, `state/improve`, `state/factcheck`, `state/claimcheck`, `reports/`, `.env` or
the test ids.

### Result, steps 6–7: the test verdict (2026-11-16 21:29 local)

**Verdict: null result. `prefcc` stays the recommended arm.**
- `r4b` meets all four criteria against C and is not worse than `prefcc`, but it is better on none of the four
  measures at the fixed 98.75% level.
- `r4a` fails criterion 4 and recalls less than `prefcc`.

Computed by `scripts/outer_verdict.py` with the rules fixed above. Output: `paper_docs/data/outer_r4/verdict.json`
(aggregates only). Every finalist is judged on the per-paper mean of its two seeds, on the 1,246 test papers. The
fact test uses 899 papers and two question sets, the claim check 296–299 papers, NLI 300 and the judge 500. Seed 2
of each finalist differs from seed 1 only in the pair order (`SCISUM_PREF_ORDER_SEED=1`).

**Against C (rule 1).** Two-seed means, with 95% intervals.

| | `r4b` (claim weight 3) | `r4a` (teacher-chosen pairs only) | `prefcc` (for reference) |
|:--|:--|:--|:--|
| recall − C | **+0.0236** [+0.0177, +0.0296], p = 0.0001 | +0.0089 [+0.0035, +0.0149], p = 0.0027 | +0.0262 [+0.0201, +0.0322] |
| recall, seeds 1 / 2 | 0.5992 / 0.5896 | 0.5787 / 0.5807 | |
| length ratio (0.90–1.10) | 1.005 (seeds 1.025 / 0.985) | 1.013 | 0.988 |
| ROUGE-L / BERTScore − C (≥ −0.010) | −0.0057 / −0.0063 | +0.0005 / −0.0010 | −0.0053 / −0.0085 |
| unsupported share − C (not above C) | +0.0051 [−0.0000, +0.0102] | **+0.0074 [+0.0022, +0.0123], fails** | +0.0045 [−0.0003, +0.0093] |
| NLI − C | −0.0163 [−0.0261, −0.0073] | −0.0323 [−0.0412, −0.0238] | −0.0243 [−0.0341, −0.0152] |
| judge faithfulness − C (1–5) | −0.069 [−0.117, −0.019] | −0.086 [−0.136, −0.040] | −0.090 [−0.139, −0.039] |
| criteria met (recall p < 0.025) | **4 of 4** | 3 of 4 | 4 of 4 |

**Against `prefcc`'s two-seed mean (rule 2).** Finalist minus `prefcc`.

| | papers | `r4b` − `prefcc` 95% | 98.75% | `r4a` − `prefcc` 95% | 98.75% |
|:--|--:|:--|:--|:--|:--|
| recall | 899 | −0.0025 [−0.0074, +0.0024] | [−0.0088, +0.0037] | **−0.0173 [−0.0226, −0.0119]** | [−0.0241, −0.0105] |
| unsupported share | 297 / 296 | +0.0007 [−0.0036, +0.0052] | [−0.0048, +0.0065] | +0.0026 [−0.0015, +0.0067] | [−0.0028, +0.0077] |
| NLI | 300 | +0.0080 [+0.0006, +0.0154] | [−0.0016, +0.0174] | −0.0080 [−0.0160, −0.0002] | [−0.0182, +0.0018] |
| judge faithfulness | 500 | +0.021 [−0.011, +0.053] | [−0.019, +0.062] | +0.004 [−0.033, +0.040] | [−0.044, +0.050] |

How rule 2 applies to each finalist:
- `r4b` passes (a), meeting all four criteria, and (b), not worse on recall or on the unsupported share. It fails
  (c): no 98.75% interval excludes zero on the better side. Its NLI gain over `prefcc` (+0.008) excludes zero only
  at 95%.
- `r4a` fails (a), on criterion 4, and (b), on recall.

**What the round showed.**
- **Claim weight 3 (`r4b`).** Tripling the claim penalty in the pair-ranking reward produced an arm
  indistinguishable from `prefcc`.
  - Recall: −0.0025.
  - Unsupported share: +0.0007.
  - NLI and judge: slightly better, but below the fixed threshold.

  On V it had cut the training grader's claim share by 0.0057 against `prefcc` (step 4). That did not carry over to
  the evaluation grader's unsupported share. This fits T17: the two claim checks are of the same kind but are not
  interchangeable.
- **Teacher-chosen pairs only (`r4a`).** Restricting the pairs to those whose chosen summary is the teacher's (940
  pairs) kept C's wording but lost two thirds of the recall gain (+0.0089 against +0.0262). It also raised the
  unsupported share. The sample-chosen pairs carry most of the fact gain.
- **The pair auditor's question** (step 4). The designer's test was: if `r4a` matches or beats `r4b` on the test's
  unsupported share and NLI, the auditor was right that the sample-chosen claim contrast is not real.
  - `r4a` − `r4b`, unsupported share: +0.0021 [−0.0023, +0.0066] (a tie).
  - `r4a` − `r4b`, NLI: −0.016 [−0.024, −0.008] (worse).
  - This lead-side paired comparison was not part of the fixed verdict.

  So `r4a` does not match `r4b` on NLI, and the test does not support the auditor's reading.
- **Second seeds.** The seeds differ by 0.0096 in recall for `r4b` and 0.0020 for `r4a` (`prefcc`: 0.013). The
  pair order is again the largest single source of variation within an arm.

**The designer's test predictions** (step 4 table, two-seed means). 9 of 12 intervals held.
- `r4b` held 5 of 6. Its BERTScore (−0.0063) was better than the predicted −0.014 to −0.007.
- `r4a` held 4 of 6:
  - its recall (+0.0089) fell below the predicted +0.012 to +0.030;
  - its NLI (−0.032) fell below the predicted −0.025 to −0.005.
- Both "`replaces_prefcc`: false" predictions held. The designer's own check is in its review (step 8).

### Step 8: the designer's review (2026-11-16 21:32 local)

The designer received only the aggregate verdict (`verdict.json`). It wrote `review.md` and a line in NOTES.md.
Both are copied unedited to `paper_docs/data/outer_r4/` after a check for paper text: there is none, only
aggregate numbers.

**Its prediction check** agrees with the one above:
- 10 of 12 schema predictions held (five measures plus `replaces_prefcc`, per finalist);
- 1 of 2 NLI predictions held;
- the misses are `r4b`'s BERTScore (better than predicted), and `r4a`'s recall and NLI (both worse).

**Its main reading** is the one in F13: the 293 sample-chosen pairs (24% of `prefcc`'s) carry about two thirds of
the recall gain. It adds three points:
- By its own count, those pairs held about 13% of the reward's fact contrast. Their effect on training was far
  larger than that share.
- `r4a` also took fewer steps (118 against 155), a confound it names. `r4d`, with 226 steps and the same filter,
  did not recover fact on V, which points to the missing pairs and not the missing steps. That is one seed, so a
  hint.
- It withdrew its pre-test rule "push the claim penalty further". Weight 3 did not move the evaluation's claim
  check, so it reads that check as saturated.

**What it would do differently:**
- weigh round 1's independent re-scoring of sample-chosen pairs (+0.041 [+0.015, +0.066], `11`) over a 30-pair
  single-reader audit;
- not predict an arm's effect from its share of the reward contrast;
- ask for token-based length statistics before predicting length;
- put an evaluation-side faithfulness signal on V before proposing faithfulness arms.

It also says that `r4c`, two pairs per paper, asked the more important question and is untested on test.

**Its next-round ideas are recorded as future work, not run** (the project's experiments ended with this
round):
1. Two pairs per paper with a small R-DPO term (α ≈ 0.0015–0.002) to hold length.
2. Iterated DPO: new samples from `prefcc` or `r4b`, re-scored into new pairs.
3. A faithfulness signal from another source, for example statements flagged by two verifiers.
4. V scored with the evaluation-side claim check, and two pair orders per arm.

**Access audit (phase 3).** Same session as phases 1–2, 160 s, 5 tool calls:
- 1 Write (`review.md`);
- 2 Bash: an append to NOTES.md that failed on quoting, then a `tail` of NOTES.md;
- 1 Read and 1 Edit of NOTES.md.

No call named `runs/`, `state/improve`, `state/factcheck`, `state/claimcheck`, `reports/`, `.env` or the test ids.
