# Threats to validity

Everything here is a reason to doubt, qualify or re-check a number in `05_results.md` or a claim in
`06_findings.md`. Corrections to statements we made earlier in the project are at the end.

---

## T1 — The teacher is the reference, so every overlap metric measures imitation

ROUGE, BERTScore, keyword F1 and the domain match are all computed against the teacher summary for
the same paper. A student that reproduces a teacher error scores well; a student that writes a
better summary in different words scores badly. The absolute levels (ROUGE-L ≈ 0.44–0.47) are
therefore **not** comparable to summarisation numbers computed against human abstracts, and the
distillation ceiling is the teacher's own quality, which we never measured independently.

*Partial mitigation.* The LLM-as-judge scores both summaries on the same rubric without being told
which is which, so it can and does place a student above its teacher (Run B wins 14% of pairings).
That instrument has its own problems — see T3.

## T2 — The reference set is heterogeneous across three teachers

Three teacher models wrote the 100,826 summaries, in very different proportions, and the students
agree with them to very different degrees. Run A, test split: ROUGE-L 0.4502 against
`deepseek-v4-flash-0731` (n = 13,181), 0.4247 against `deepseek-v4.1-flash` (n = 1,209), 0.3297
against `meta/muse-spark-1.3-contributor` (n = 610). The same ordering holds for Run B. Part of
what the metric calls quality is agreement with the majority teacher's style. Any comparison
between models is only fair because both see the identical split; comparisons to outside work are
not.

## T3 — The judge is anchored, sees an excerpt, and is a single model

The judge (`qwen38-27b`, served by an OpenAI-compatible endpoint) scores a student summary and the teacher
summary **in the same call**, which makes the pair comparison stable but also means the two scores
are not independent: a weak student can depress, and a strong student can raise, the score the same
teacher text receives. This is visible in the data — identical teacher summaries score 3.87–3.92
beside Run A and 3.83 beside Run B. **Cross-run judge deltas are therefore not strictly
commensurable**, and we report them alongside ROUGE rather than instead of it.

Further limits: the judge reads a ~9,000-character excerpt of the paper, not the paper, so its
faithfulness axis is partly a guess; A/B order is fixed by a hash of the paper id, which removes
order bias on average but not per paper; and one judge model is one opinion, with no human
calibration behind it.

## T4 — Faithfulness is a relative, noisy, 300-paper measurement

The NLI score is the mean entailment probability of each summary sentence against the best-matching
retrieved evidence (BM25 top-k, `microsoft/deberta-large-mnli`). It is computed on 300 papers per
evaluation point, not 15,000. A retrieval miss is indistinguishable from a hallucination, which is
why absolute values sit near 0.42–0.49 even for text we can read and verify. Only the *direction*
across checkpoints (F5: flat) is supported, and even that rests on differences smaller than the
spread between adjacent milestones.

## T5 — The truncation effect is estimated, not measured

The raw gap between papers that fit whole and papers that did not (0.07 ROUGE-L) is a
between-group comparison of papers of different length, and it **overstates** the cost of
truncation: it persists for Run C although 91% of papers fit. The 0.03 estimate in F4 comes from a
difference of differences between two different students (B vs C, and A vs C) on shared papers; it
assumes the larger window is the only thing that helps C specifically on the papers B lost. A
larger-window run of the *same* student would be the clean experiment and was not done.
Separately, `fits` (the whole body reached the prompt) and `trunc_kind` (the fitting loop had to
run) are different groupings; 61.41% of Run A's test papers carry the whole body, 72.37% are
labelled `none`. F4 uses `fits`.

## T6 — One run per configuration, one seed

Each student was trained once, with seed 20261014, on one data order. We have no variance estimate
for any reported number, so differences of ~0.005 ROUGE-L between adjacent milestones should not be
read as real. The large inter-model gap (+0.026 ROUGE-L, consistent across all seven milestones and
the test split) is more robust to this than the within-run milestone deltas.

*Update (rounds 2–3, `11_reward_selected_distillation.md`; `12_reactive_training.md` §8).* For the
preference arms, seed noise was measured by training the same pairs in a second order. Pair order alone
moves fact recall by 0.013–0.020, the length ratio by 0.04–0.06 and BERTScore by up to 0.005, as much as
the differences between round-2 arms. The recommended arm `prefcc` is therefore judged on the mean of its
two seeds. SFT Runs A–C still have one seed each.

## T7 — Greedy decoding is not bit-reproducible

Generation is greedy (`temperature=0`), but vLLM's batched kernels reduce in an order that depends
on how requests are grouped, so the same checkpoint on the same paper can produce different text
across runs. Observed during the smoke test: re-decoding the same eight papers with the same
checkpoint moved per-paper metrics by up to 0.04. That is a small, informal observation (n = 8),
not a variance estimate; it averages down over 1,000–15,000 papers but does not vanish, and it is
the reason differences between adjacent milestones should not be over-read.

## T8 — The final point of every run also receives the learning-rate decay

The warmup-stable-decay schedule holds the learning rate at 1e-4 and decays it to 1e-5 over the last
10% of steps: from ~63,160 papers for A and B (875 of 8,750 steps) and from ~18,080 papers for C
(250 of 2,500 steps). Every earlier checkpoint was taken at the full rate. The final point therefore
differs from its predecessors in two ways at once (more data *and* annealing), and in the logs the
step is large against the trend that preceded it (held-out loss, nats per paper):

| run | change over the last stable interval | change at the final, decayed point |
|:--|--:|--:|
| A (50k → 60k → 70k) | −2.8 | **−19.5** |
| B (50k → 60k → 70k) | −3.0 | **−11.6** |
| C (15k → 17.5k → 20k) | −0.5 | **−11.5** |

Validation ROUGE-L moves the same way (A +0.011, B +0.010, C +0.016 over the preceding checkpoint).
Roughly 85% or more of each final step is therefore annealing, not the extra data. There is no
control (a 70,000-paper checkpoint at constant rate), so this is an inference from the trend.

**It affects comparisons across runs.** C's 20,000-paper checkpoint is annealed; A's and B's
20,000-paper checkpoints are not. The matched comparisons are C at 20k against A and B at 70k (all
annealed: held-out loss 449 against 603 and 508), or C's last stable point (17.5k, 461) against
A's and B's stable points (639 and 529 at 20k). C is ahead either way.

## T9 — Run B's untrained baseline measures a different thing

`IFM/K2-Horizon-7B` is a reasoning model whose template opens a mandatory thinking block. Untrained,
it produced 0.000 parseable JSON, so the base row for Run B is a statement about format, not about
summarisation ability, and **must not** be used as the floor for B's gains. Only Run A has a
meaningful untrained baseline (3,000 test papers, ROUGE-L 0.3245).

## T10 — Validation curves and the test split are different populations

The milestone curves use a fixed 1,000-paper subset of the validation split; the final numbers use
the full 15,000-paper test split, decoded once. The two are not interchangeable — test ROUGE-L for
Run A's final checkpoint is 0.4432 against 0.4448 on validation. Trends should be read within one
population.

## T11 — The base-model floor is 3,000 papers, not 15,000

To save GPU time, the untrained baselines were decoded on the first 3,000 test papers. Comparisons
of a trained checkpoint against the base model mix a 15,000-paper mean with a 3,000-paper mean.

## T12 — Run C changes two things at once

Run C uses a different student *and* a larger prompt budget (28,672 vs 13,312 tokens) *and* fewer
papers (20,000 vs 70,000). If it wins, attribution between those causes requires the ablation we
have not yet run. The 20,000-paper choice is justified by F2, but F2 is itself a finding from two
other models.

## T13 — No human evaluation and no downstream task

Nothing in this work was read and rated by a person at scale, and the summaries were never used for
the task they are meant for (retrieval, triage, literature survey). All conclusions are about
automatic metrics and one LLM judge.

## T14 — Corpus provenance

Papers are arXiv full texts parsed by GROBID. Parsing failures, reference-section leakage and
figure-caption placement vary by paper and were not audited beyond the schema checks. 28 papers in
the source corpus failed GROBID entirely and are absent.

## T15 — A served adapter can silently be the base model

The model a serving engine builds is chosen by `config.json`; the model a trainer builds is chosen
by the loading API. When the two differ, a LoRA adapter can load without matching a single module,
and vLLM 0.31 says nothing. It happened to Student C (`09_experiment_log.md`, 2026-10-19) and was
caught only because three checkpoints produced identical text. Every reported evaluation of an
adapter now passes an in-engine check that the adapter changes the output; any number produced
before that check existed was re-examined, and Runs A and B were confirmed unaffected by
checkpoint-to-checkpoint differences in their outputs. A reader replicating this with another
serving stack should apply the same check: compare *with* and *without* the adapter in one engine,
since enabling LoRA alone already perturbs the kernels.

## T16 — The fact test is a model's reading of the paper

Measured since: a second grader from another family agrees with the first on 85–86% of verdicts
(kappa 0.67–0.70), so a verdict is wrong or arguable about one time in seven; with the whole paper
as the "summary" the test reaches only 0.884 recall, so roughly one question in nine has a gold answer
or a question that even a perfect reader does not satisfy; and with no summary at all a guessing
answerer earns 0.307 recall, nearly all of it partial credit. Gold answers are still unverified by a
person; until they are, treat absolute scores as indicative and the paired differences as the result.

Gold answers come from `gemma4-31b-it` and are unverified by people; a wrong or partial gold
answer penalises a correct summary. The answerer and grader (`qwen38-27b`) are the judge used in
T3. A summary of ~1,000 tokens cannot hold ten specific facts, so recall has a ceiling well under 1
(the teacher's is 0.618) and only the ranking and the paired differences are meaningful. The
ACLSum check is weaker still: 100 papers, a one-sentence reference, an LLM grade that saturates
near 0.9, and an input that is a short extract of NLP papers unlike the training set; it separates
the best students from the untrained model and nothing finer.

## T17 — The training reward and the evaluation use the same kinds of instrument

The preference pairs are ranked by a fact test and a claim check, and the evaluation uses a fact test and a
claim check. The models are swapped: in training, Qwen3.8-27B writes the questions and Gemma-4-31B answers,
grades and checks claims; in evaluation, the roles are reversed. The two claim verifiers agree on 97.1% of C's
statements (Cohen's kappa 0.60). Part of a gain could be the student learning what this kind of check rewards
rather than what is true. NLI and the judge are independent of the reward, and on both `prefcc` stays below C
(T18). Source: `11_reward_selected_distillation.md`, round-3 caveats.

## T18 — The recommended arm is below C on NLI and on the judge's faithfulness score

`prefcc` (two-seed mean) has NLI −0.0243 [−0.0341, −0.0152] and judge faithfulness −0.090 [−0.139, −0.039]
on a 1–5 scale, against C. Every preference arm shares this deficit, and blending back toward C shrinks the
NLI deficit roughly in proportion but barely moves the judge's. Criterion 4 uses the claim check, fixed as the
faithfulness criterion before round 2, so these two are reported, not criteria. "No less faithful than C" is
supported by the claim check only. Sources: `12_reactive_training.md` §8, questions 1 and 2.

## T19 — The claim check sees at most 90,000 characters of a paper

Statements about text past the cut cannot be supported, so long papers bias the check against any summary
that uses their late sections. On the 273 papers it saw whole, `prefcc` − C in supported share is −0.0011
[−0.0062, +0.0045]; on the 26 cut papers +0.0086 [−0.0109, +0.0281]. Source: `11_reward_selected_distillation.md`.

## T20 — The 1,246 test papers were reused across rounds

All improvement arms (rounds 1–4, the blends, the reactive run) are evaluated on the same 1,246 test papers,
and the round-4 designer read the aggregate test tables of the 14 earlier arms. Pre-registered criteria and
Bonferroni-corrected tests limit, but do not remove, selection on these papers. A result that replaces
`prefcc` should be confirmed on test papers outside the 1,246; that confirmation has not been run. Source:
`12_reactive_training.md` §9.

## T21 — The validation set V ranks arms on recall, not on faithfulness or length

Round 4's designer chose finalists by the training-side reward on 600 validation papers (V). The calibration
of V against the test on nine earlier adapters found its claim share does not predict the test's unsupported
share (Spearman 0.14), so a V-chosen "faithfulness" finalist is chosen on weak evidence. Source:
`12_reactive_training.md` §9, step 1.

---

## Corrections to earlier statements

| Date | Statement made | Correction |
|:--|:--|:--|
| 2026-11-16 | `11` "What was built": each pool paper had the policy's greedy output and 6 samples as candidates | That was the first job only (120 of 1,520 papers). The other 1,400 had 5 samples and no greedy output (`state/improve/samples.jsonl.gz`). No number changes; the selection table already said 5 per paper. |
| 2026-11-16 | `09` rows of 02:13 and 02:26: q35r4b trains for 118 steps | It trained for 169 steps (1,347 pairs). |
| 2026-11-09 | F10: `prefcc` gains +0.033 recall without adding unsupported statements | That is seed 1. The two-seed mean is +0.0262 with unsupported share +0.0045 [−0.0003, +0.0093]; seed 2 alone fails criterion 4 (F10 updated). |
| 2026-11-07 | Round 1: about 70% of the preference gain survives at equal length (+0.051, regression on the length ratio) | An extrapolation. Round 2's length-controlled arms gained +0.017 to +0.030, so it overstated the length-free gain by more than half. |
| 2026-11-07 | Round 1 and 2 claim-check numbers | A parser bug dropped verdicts of summaries with one malformed field. After the fix every arm looked worse against C, and `preflcrd` moved from passing criterion 4 to failing it; round 1's conclusions were unchanged. |
| 2026-10-25 | F3 caveat: "part of B's margin is input, not model" (tokenizer) | Measured: the tokenizer changes what is read for 2.3% of test papers and is worth ~0.0003 of B's 0.026 ROUGE-L margin (F3). Withdrawn. |
| 2026-10-23 | F4/T5: the none-vs-sections gap "is a lower bound on the cost of losing text" | Wrong direction. The controlled test (F4) puts the cost near 0.03; the raw gap is ~0.07 and persists for a student that sees 91% of papers whole, so it is mostly paper difficulty. |
| 2026-10-17 | "At 38.3 of 39.25 GiB the allocator thrashes instead of failing" | Wrong. The twelve silent minutes were the held-out loss over 200 long sequences (682 s measured on the re-run), not an allocator stall. No evidence of thrashing was ever collected. |
| 2026-10-16 | "Papers truncated hardest score 0.383 ROUGE-L" | The `hard` bucket contains **one** paper per test split after the 2026-10-15 fitting fix. It is not a population and the number is withdrawn. The valid contrast is `none` vs `sections` (T5, F4). |
| 2026-10-16 | "Run B's untrained model scores 0.000 clean JSON, so training adds 0.99" | Correct as a format observation, misleading as a quality gain; see T9. |
| 2026-10-16 | "K2 produces malformed JSON (json_ok = 0.000)" | A metric bug: the scorer counted the template's own `</ifm|think>` prefix as part of the model's reply. After `score_one(target_prefix=...)` and a CPU re-score, trained K2 checkpoints read 0.992–0.996. |
