# Notes of the round-4 designer

## Phase 1 (2026-11-15): proposal

Read: TASK.md, `11` (rounds 1-3), `12` §8-§9, F8-F12, `outer_calibration.json`, the pair audit
(`state/pair_audit/report.json`: aggregates plus the stored parts and verdicts of the 80 pairs; I did not
copy any of the auditor's quotes). Everything else comes from `outer_tool.py preview`.

### Observed (numbers)

**Test record that constrains the design.**
- prefcc over two seeds against C: recall +0.0262, length 0.988, ROUGE-L -0.0053, BERTScore -0.0085,
  unsupported +0.0045 [-0.0003, +0.0093], NLI -0.024, judge faithfulness -0.090.
- Criterion 3 (BERTScore no more than 0.010 below C) is prefcc's tightest margin, 0.0015. Criterion 4
  passes by 0.0003 at the lower end of its interval.
- Rule 2 needs a 98.75% interval excluding zero. Earlier paired 95% half-widths are about 0.006 for
  recall, 0.005 for unsupported share, 0.010 for NLI and 0.045 for the judge. So a finalist needs roughly
  recall +0.008, unsupported -0.007 or NLI +0.013 against prefcc, with recall not lower than about
  -0.005.
- The claim penalty (weight 0 to 1, round 3) is the only change with a test effect on faithfulness:
  unsupported -0.0069 [-0.0118, -0.0021] and NLI +0.022 [+0.011, +0.033] against preflc, recall +0.009.
  On V the same step shows fact -0.010 (prefcc 0.0422 vs preflc 0.0525), a cost the test did not show.
- The blends move recall and NLI by the same factor (ccmix75: 87% and 87%; ccmix50: 61% and 62%). So
  blending slides along prefcc's recall-NLI trade-off without improving it. It does recover ROUGE-L and
  BERTScore (+0.0036 and +0.0037 at w = 0.75).
- NLI deficit per unit of recall gain: prefcc -0.76, round-1 pref -0.79, preflc -1.57, prefrd -3.0,
  preflcrd3 -2.2, prefccrx -2.0. The claim penalty is what halved it against preflc.

**V calibration.**
- Fact ranks recall (rho 0.81, every sign right), but V's differences are 1.3-3x the test's.
- ROUGE-L ranks (rho 0.71). Length tracks the test (rho 0.98) at an offset of +0.01 to +0.02.
- Claim share does not rank (rho 0.14).
- Pair order alone on V: fact 0.042 vs 0.032 (prefcc's two orders) and 0.0525 vs 0.0498 (preflc's);
  length +0.038 vs -0.023 and +0.039 vs -0.004; claim share -0.0015 vs +0.0025.
- The paired SE of an arm's V fact difference is about 0.007. With order noise of about 0.01, an arm-vs-
  prefcc fact difference below about 0.017 cannot be resolved on V.

**Pair audit (80 pairs, one reader). My own tallies from the 80 rows of stored parts:**
- Overall preference for the chosen side: teacher stratum 44/50, sample stratum 18/30.
- The auditor's unsupported share per statement, chosen vs rejected: teacher stratum 0.035 vs 0.082,
  sample stratum 0.090 vs 0.093.
- Gemma's stored claim share, chosen vs rejected: teacher stratum about 0.026 vs 0.058, sample stratum
  about 0.029 vs 0.056. So Gemma's contrast is confirmed by the auditor in teacher-chosen pairs and not
  in sample-chosen pairs.
- In the sample-stratum disagreements the chosen sample usually has the higher count of unsupported
  statements by the auditor (e.g. 5 vs 3, 4 vs 2, 3 vs 1) and a stored claim share of 0. No stored part
  separates the disagreements (as the report says).

**Preview (pairs; claim = mean claim share chosen / rejected; fact gap = chosen minus rejected).**

| setting | pairs | sample-chosen | fact gap | claim chosen / rejected | rejected shorter | rej/chosen length median |
|:--|--:|--:|--:|:--|--:|--:|
| prefcc | 1,233 | 293 | 0.226 | 0.024 / 0.062 | 44.8% | 1.010 |
| claim 0 (= preflc) | 1,163 | 290 | 0.252 | 0.033 / 0.050 | 43.5% | 1.013 |
| claim 2 | 1,287 | 323 | 0.199 | 0.019 / 0.067 | 47.0% | 1.005 |
| claim 3 | 1,347 | 376 | 0.178 | 0.016 / 0.068 | 47.0% | 1.005 |
| claim 5 | 1,386 | 438 | 0.149 | 0.013 / 0.068 | 46.2% | 1.005 |
| filter c_teacher | 940 | 0 | 0.201 | 0.022 / 0.057 | 46.2% | 1.007 |
| margin 0.3 | 1,128 | 48 | 0.200 | 0.024 / 0.062 | 46.5% | 1.007 |
| c_teacher + claim 3 | 971 | 0 | 0.166 | 0.012 / 0.059 | 49.0% | 1.001 |
| k = 2 | 2,165 | 486 | 0.224 | 0.025 / 0.061 | 43.5% | 1.019 |
| k = 3 | 2,834 | 596 | 0.224 | 0.025 / 0.061 | 43.4% | 1.026 |
| c_teacher + claim 3 + k = 2 | 1,808 | 0 | 0.162 | 0.012 / 0.058 | 47.6% | 1.007 |
| rule lowest | 1,233 | 293 | 0.265 | 0.024 / 0.070 | 40.9% | 1.044 |
| lambda 0.4 | 1,374 | 105 | 0.148 | 0.025 / 0.054 | 48.8% | 1.001 |
| lambda 0 | 1,146 | 462 | 0.265 | 0.025 / 0.065 | 42.8% | 1.016 |
| gap 0.15 / 0.35 | 1,401 / 951 | 353 / 203 | 0.180 / 0.271 | | | |
| beta_num 1.0 | 1,259 | 326 | 0.215 | 0.025 / 0.062 | | (1,114 pairs as prefcc) |
| len_free 1.05 / 1.1 | 1,209 / 1,187 | 330 / 353 | 0.240 / 0.250 | | 48.9% / 52.2% | 1.002 / 0.996 |

- The claim-weight return shrinks. Weight 0 to 1 added 0.020 of claim contrast for 0.025 of fact gap;
  1 to 2 adds 0.011 for 0.027; 1 to 3 adds 0.015 for 0.048. A higher weight also adds sample-chosen
  pairs (376 at weight 3).
- The sample-chosen pairs have a fact gap of about 0.31 (0.226 x 1,233 - 0.201 x 940, divided by 293).
  By the reward's grader they are the largest fact contrasts. Round 1's independent check gave them
  +0.041 recall on new questions (teacher-chosen +0.085).
- Unit warning: the preview's length measure says prefcc's rejected summary is shorter in 44.8% of pairs
  (median rejected/chosen 1.010). Doc 11's token measure says 51.5% (median 0.994). The two disagree in
  direction, so I use the preview's measure only to compare settings with each other.

### Inferred (not measured)

- The sample-chosen pairs train toward one of C's samples that the auditor finds no more faithful than
  the rejected one, and much less faithful than the teacher's text. They are a plausible source of the
  faithfulness gap between prefcc and C that remains (NLI, judge, LLM reader I5 +0.008). They are also a
  plausible source of part of the recall gain: on-policy chosen side, the largest fact contrasts. r4a
  tests which.
- Raising the claim weight is the best-evidenced faithfulness lever. Its return per unit of fact
  contrast falls, and at weight 3 it adds sample-chosen pairs whose Gemma contrast the auditor does not
  confirm. The filter and the claim weight should therefore be complementary.
- Recall at the teacher's length has been +0.02 to +0.03 in every length-controlled arm, and seed noise
  is 0.013-0.020. The one untried lever that adds information rather than length is more negatives per
  paper (k = 2).
- V will screen out large recall losses or gains (above about 0.017 fact against prefcc). It will not
  separate arms whose differences are smaller, and it says nothing reliable about faithfulness. The
  faithfulness arms pay off, or not, only on the test, so the finalist choice is a bet on mechanism,
  gated by V's recall screen.

### Decision: the arms (one change each, plus the combination)

Design: one-factor-at-a-time arms for the three mechanisms, plus the combination as the best-guess
finalist. All arms use stored order, as prefcc's seed 1, so prefcc's V numbers (seed 1) are the
like-for-like reference, with the order-noise caveat above.

| arm | change | pairs | hours | hypothesis in one line | V prediction against prefcc (fact) | falsified on V if |
|:--|:--|--:|--:|:--|:--|:--|
| r4a | filter `c_teacher` | 940 | 6.9 | sample-chosen pairs teach facts at no faithfulness gain; drop them | -0.024 to +0.006 (centre -0.008) | fact < prefcc - 0.020, or ROUGE-L not above prefcc |
| r4b | claim weight 3 | 1,347 | 9.8 | the claim penalty's dose-response continues | -0.027 to +0.003 (centre -0.010) | V claim share not below prefcc, or fact < prefcc - 0.025 |
| r4c | 2 pairs per paper | 2,165 | 15.8 | a second negative per paper adds recall at the same mix | -0.010 to +0.028 (centre +0.010) | fact not above prefcc + 0.005 at length within 0.03 |
| r4d | all three | 1,808 | 13.2 | faithfulness from the first two, recall restored by the third | -0.027 to +0.008 (centre -0.007) | fact < prefcc - 0.015 |

Full prediction intervals (against C and against prefcc, fact, claim share, length, ROUGE-L) are in
`proposal.json`. The length intervals are wide (about 0.09-0.11) because pair order alone moved V length
by 0.06.

Blend: `r4cw75` (r4c at w = 0.75). If r4c gains recall but drifts further from the teacher's wording, the
blend keeps at least 80% of its V fact gain and recovers at least a third of its ROUGE-L deficit. I
proposed no blends of the faithfulness arms. Blending toward C slides along the recall-NLI line and would
dilute any faithfulness gain.

How I expect to pick finalists (written now, before V):
- r4d is the faithfulness finalist if its V fact is no more than about 0.010 below prefcc's. Otherwise
  r4a or r4b, whichever keeps more fact.
- r4c (or r4cw75, if r4c's ROUGE-L falls more than 0.003 below prefcc's) is the recall finalist if its V
  fact is above prefcc's by at least about 0.015.
- If nothing clears these, I name fewer finalists or none.

### What I chose not to test, and why

- **Loss knobs (beta, lr).** These were never varied. The blends already show what "moving less" does,
  and it slides along the same recall-NLI line. A higher beta or a lower lr would most likely do the same.
- **Likelihood weight (`pref_sft_weight` 2).** This is the knob I would test next. It is the direct lever
  on drift from the teacher's wording (held-out NLL 0.450 to 0.478 in every arm) and on criterion 3. It
  lost to r4d because only faithfulness or recall can win rule 2, and its effect on those is speculative.
  The control arm (pure SFT on teacher targets) did not improve faithfulness.
- **R-DPO alpha.** prefcc's length is 0.988, so there is no length problem to fix. The term only pushes
  toward shorter outputs.
- **`imp_len_free`, `imp_gamma_len` (recall through length).** BERTScore is prefcc's tightest margin,
  and longer outputs have cost BERTScore (round 1: -0.034 at 1.26x).
- **lambda.** At 0.4 the fact contrast falls to 0.148 and pairs won on wording alone are added. At 0 the
  sample-chosen pairs rise to 462, the stratum the audit flags.
- **Rule `lowest`.** Under the 1.0x length allowance the lowest-reward sample is now the longer one
  (median 1.044, 40.9% shorter). That length bias pushes toward shorter outputs.
- **Gap or reward-gap filters.** Only 4 of 50 teacher-chosen disagreements sit at a small gap, about 8%
  of the stratum: too small an effect for V or the test to see.
- **A stored-parts filter for the sample stratum.** The audit found no stored part that separates its
  disagreements.
- **`imp_beta_num`.** At 1.0, 1,114 of 1,233 pairs are unchanged. Almost nothing moves.
- **Claim weight 5.** The claim contrast saturates (0.055 against 0.052 at weight 3), and the fact gap
  falls to 0.149.
- **k = 3.** It adds 669 third negatives with a larger shorter-output bias (median 1.026), at 20.7 h.

### Accepted

`outer_tool.py propose --round r4` accepted the proposal on the first submission (2026-11-15): arms
r4a-r4d and blend r4cw75. The pair counts and statistics it echoed match the previews above. Estimated
training is 6.9 + 9.8 + 15.8 + 13.2 h on one card each. The proposal is now fixed.

### What I cannot tell

- Whether the sample-chosen pairs drive the recall gain (r4a answers this on V if the effect is larger
  than about 0.017).
- Any arm's faithfulness, except for finalists on the test.
- Which length measure the DPO length bias follows (tokens or the stored ratio).
- Whether the auditor's sample-stratum reading (30 pairs, one reader, 7 of 12 disagreements at low
  confidence) holds. r4a, and r4d's choice as a finalist, rest on it.

### For phase 2

Score each prediction in `proposal.json` against `val` before choosing. Compare each arm with prefcc's
seed-1 V numbers, and remember the resolution limit: about 0.017 in fact, and 0.06 in length from order
alone. Read r4c's fact together with its length. Attribution: r4a and r4d share the filter, r4b and r4d
the claim weight, r4c and r4d the second negative. If r4d falls short on fact, r4a and r4b show which of
the first two cost recall. r4d minus the sum of the single-change effects is too noisy on V to read as
an interaction.

## Phase 2 (2026-11-16): V results, prediction check, finalists

Source: `outer_tool.py val`, 600 V papers. "vs prefcc" means against prefcc's seed-1 V scores, as `val`
reports them. Intervals are 95%.

### Observed (numbers)

| arm | fact - C | fact - prefcc | claim - C | claim - prefcc | length (abs.) | length - prefcc | ROUGE-L - C | ROUGE-L - prefcc |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| prefcc (s1) | +0.0422 | | -0.0015 | | 1.019 | | -0.0049 | |
| prefccs2 | +0.0324 | -0.0099 | +0.0025 | +0.0040 | 0.958 | -0.061 | -0.0058 | -0.0009 |
| r4a filter | +0.0317 | -0.0110 [-0.0235, +0.0031] | +0.0030 | **+0.0045** [+0.0007, +0.0084] | 1.010 | -0.0098 | **-0.0008** | **+0.0042** [+0.0014, +0.0071] |
| r4b claim 3 | +0.0370 | -0.0048 [-0.0176, +0.0090] | **-0.0072** | **-0.0057** [-0.0094, -0.0022] | 1.026 | +0.0069 | -0.0067 | -0.0018 |
| r4c k = 2 | **+0.0572** | **+0.0148** [+0.0013, +0.0278] | +0.0006 | +0.0022 | 1.068 | **+0.0486** | -0.0066 | -0.0018 |
| r4cw75 | +0.0443 | +0.0020 | -0.0010 | +0.0007 | 1.051 | +0.0320 | -0.0044 | +0.0006 |
| r4d all three | +0.0274 | **-0.0152** [-0.0289, -0.0017] | +0.0001 | +0.0017 | 1.029 | +0.0102 | -0.0013 | **+0.0037** [+0.0008, +0.0067] |

Other numbers:
- r4a's correct share against prefcc is -0.0136 [-0.0240, -0.0018].
- Every round-4 arm, and prefccs2, has a lower unsupported-number share than prefcc seed 1 (-0.004 to
  -0.006). Seed 1 is the outlier here, so this is not an arm effect.
- The tool's proxy criteria pass for every arm, so they do not separate anything.
- Mean of prefcc's two seeds on V: fact +0.0373, claim +0.0005, length 0.989.

### Prediction check (scored before choosing)

32 arm intervals: 26 held, 6 missed.

- **r4a**: 6 of 8 held. Claim share against C (+0.0030) sits exactly on the upper edge.
  - Missed against prefcc: claim share +0.0045 (interval was up to +0.004) and ROUGE-L +0.0042 (up to
    +0.004). The ROUGE-L miss is on the better side.
  - Falsification clauses did not fire. Fact is -0.011, not below -0.020. ROUGE-L is above prefcc, with
    the interval excluding zero.
  - So the "wording back toward the teacher" half was confirmed more strongly than predicted. r4d, which
    also has the filter, agrees (+0.0037).
- **r4b**: 8 of 8 held. No clause fired. The training grader's claim share fell by the predicted amount
  at a recall cost within noise.
- **r4c**: 6 of 8 held. Both length intervals missed. I predicted shorter output, because the preview
  said the rejected summary was longer (median 1.019). It wrote 0.049 longer than prefcc.
  - Fact +0.0148 held. But the clause was conditioned on length within 0.03 of prefcc's, and length is
    not, so the fact gain cannot be read as k = 2 adding recall at equal length.
- **r4d**: 6 of 8 held. Both length intervals missed on the long side.
  - Its clause fired: fact -0.0152, more than 0.015 below prefcc. The hypothesis that the extra negatives
    win back the recall cost is **falsified** on V. By the rule written in phase 1, r4d is not a
    finalist.
- **r4cw75**: kept 77% of r4c's fact gain (predicted at least 80%: missed). Recovered exactly 33% of the
  ROUGE-L deficit (predicted at least a third: held at the edge). Length came back only 19% of the way to
  C's.
- **Calibration lesson**: all 6 misses are on length or narrowly past an edge, and all 4 length misses
  are on the long side. Every round-4 arm wrote longer than C on V (+0.028 to +0.087). The preview's
  rejected/chosen length median does not predict the direction of output length (see the unit warning in
  phase 1). Next round: do not use it as a length predictor.

### Inferred

- **The filter moves wording back to the teacher.** Two arms with the filter (r4a, r4d) are both
  +0.004 ROUGE-L above prefcc, with intervals excluding zero. r4a is at C's level.
- **The filter costs some recall on V**: about -0.011 against seed 1 and -0.006 against prefcc's two-seed
  mean, with the correct-share interval below zero. The sample-chosen pairs carry part of the recall
  gain. That is consistent with round 1's independent +0.041 check, and less than the 0.020 I set as
  "not free".
- **The training grader learns the claim penalty mainly through sample-chosen pairs.**
  - Claim 3 on the full set moved V claim share by -0.0057 (r4b against prefcc).
  - Claim 3 inside the filtered set moved it by about -0.003 (r4d against r4a, from the summaries;
    unpaired, and r4d also has k = 2).
  - The filtered arms sit above prefcc on Gemma's claim share.
  - The auditor said Gemma's contrast in exactly those pairs is not real. Either Gemma's flags in samples
    are real and the auditor missed them, or the arms learn Gemma's quirks through on-policy pairs. V
    cannot separate the two; the test can (Qwen claim check, NLI, judge).
- **k = 2 on the full set adds recall but also length.** Earlier arms' fact-vs-length slopes on V are
  about 0.06-0.19 per unit of length ratio (prefcc's seeds 0.16, preflc's seeds 0.06, round 1 against
  preflc 0.19). That attributes +0.003 to +0.012 of r4c's +0.0148 to length.
- **k = 2 inside the filtered set added nothing visible**: r4d is about -0.004 against r4a in fact (from
  the summaries). Not read as an interaction (noise about 0.014).
- **r4b and r4c share a ROUGE-L of -0.0067 against C**, below both of prefcc's seeds. On the test,
  BERTScore drops have been 1.3-2.4x ROUGE-L drops (median about 1.65), so both arms risk criterion 3
  (BERTScore at or above -0.010) on the two-seed mean.

### Decision: finalists r4b and r4a

- **r4b**, by the pre-written rule: r4d is out, and of r4a and r4b, r4b keeps more fact.
  - Fact is level with prefcc's two-seed V mean (-0.0003).
  - It moved the training grader's claim share the most of any arm scored on V.
  - Risk: criterion 3 is about a coin flip.
- **r4a**, a deviation from the pre-written plan. That plan named r4c as the recall finalist if its fact
  was at least about +0.015 above prefcc; it is +0.0148.
  - Why I dropped r4c:
    1. Its own clause conditioned the fact reading on equal length, and the length-adjusted gain
       (+0.003 to +0.012) is below the threshold.
    2. Longer, less teacher-like output makes criteria 3 and 4 unlikely: about 0.3 and 0.4. prefcc
       passes criterion 4 only narrowly at 0.988 length.
    3. Its second seed is the most expensive (15.8 h).
  - Why r4a:
    1. Best criterion-3 margin of any preference arm.
    2. Its key V prediction was confirmed with an interval excluding zero.
    3. Cheapest second seed (about 7 h).
    4. Together with r4b it is the only test in this round that can tell apart "Gemma's claim penalty
       works" and "the auditor is right about the sample-chosen pairs". The two arms point in opposite
       directions on Gemma's claim share and on the auditor's reading.
  - Risk: rule 2's "not worse on recall".
- **Not named**:
  - r4c (above).
  - r4cw75: level with prefcc on fact (+0.002) at +0.032 length, so it has no route to "better on" any
    rule-2 metric.
  - r4d: falsified, and the lowest fact.
- **Honest odds**: about 0.1 each of replacing prefcc. Both test predictions say `replaces_prefcc: false`.
  A null result is the most likely outcome. I name the two finalists for what the test can tell that V
  cannot: whether either faithfulness lever moves the evaluation's unsupported share or NLI.

Test predictions (two-seed mean; length absolute; NLI is not in the schema, so it is recorded here):

| | recall - C | unsupported - C | length | ROUGE-L - C | BERTScore - C | NLI - C | replaces prefcc |
|:--|:--|:--|:--|:--|:--|:--|:--|
| r4b | +0.014 to +0.032 | -0.006 to +0.007 | 0.96 to 1.04 | -0.009 to -0.004 | -0.014 to -0.007 | -0.025 to -0.005 | no (p about 0.1) |
| r4a | +0.012 to +0.030 | -0.005 to +0.009 | 0.95 to 1.03 | -0.004 to +0.001 | -0.007 to -0.001 | -0.025 to -0.005 | no (p about 0.1) |

How phase 3 will read the pair:
- r4b beats r4a on unsupported share and NLI: Gemma's flags in samples are real, and next round should
  push the claim penalty further.
- r4a matches or beats r4b on those metrics despite its higher Gemma claim share: the auditor was right,
  and next round should build pairs on the teacher's side, fixing recall another way.
- Both are level with prefcc: neither lever is strong enough to see at 1,246 papers.

`outer_tool.py select --round r4` accepted finalists r4b and r4a on the first submission (2026-11-16).
They are now fixed.

## Phase 3 (2026-11-16): test results

Test results are in. 10 of 12 schema predictions held (misses: r4b BERTScore, better than predicted;
r4a recall, worse), plus 1 of 2 NLI predictions (r4a worse than predicted). Neither finalist replaces
prefcc (both predicted false).
- r4b meets all four criteria but is better on nothing at 98.75%. Its NLI is +0.008 at 95%.
- r4a fails criterion 4 and is worse than prefcc on recall by 0.017.

Main lessons:
- The sample-chosen (on-policy) pairs carry about two-thirds of prefcc's recall gain and help NLI, so
  the auditor's hint failed.
- Claim weight 3 did not move the evaluation's claim check.

Full review in `review.md`.