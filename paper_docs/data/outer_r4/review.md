# Round 4 review (designer, phase 3)

**Verdict:** prefcc stays the recommended arm. Neither finalist replaces it, so the round is a null
result. Both finalists were predicted as `replaces_prefcc: false`, and both predictions held.

All numbers are from the lead's aggregate test results: the 1,246 test papers, fact recall on 899, the
claim check on 297-299, NLI on 300 and the judge on 500. Values are means over two seeds; intervals are
paired 95% unless marked 98.75%.

## 1. Prediction check

### Test predictions (finalists.json; two-seed mean, length absolute)

| finalist | measure | predicted | observed | |
|:--|:--|:--|--:|:--|
| r4b | recall - C | +0.014 to +0.032 | +0.0236 [+0.0177, +0.0296] | held |
| r4b | unsupported share - C | -0.006 to +0.007 | +0.0051 [-0.0000, +0.0102] | held |
| r4b | length ratio | 0.96 to 1.04 | 1.005 | held |
| r4b | ROUGE-L - C | -0.009 to -0.004 | -0.0057 | held |
| r4b | BERTScore - C | -0.014 to -0.007 | **-0.0063** | **missed** (better than predicted) |
| r4b | replaces prefcc | false | false | held |
| r4b | NLI - C (NOTES only) | -0.025 to -0.005 | -0.0163 | held |
| r4a | recall - C | +0.012 to +0.030 | **+0.0089** [+0.0035, +0.0149] | **missed** (worse than predicted) |
| r4a | unsupported share - C | -0.005 to +0.009 | +0.0074 [+0.0022, +0.0123] | held (upper half) |
| r4a | length ratio | 0.95 to 1.03 | 1.013 | held |
| r4a | ROUGE-L - C | -0.004 to +0.001 | +0.0005 | held |
| r4a | BERTScore - C | -0.007 to -0.001 | -0.0010 | held (at the edge) |
| r4a | replaces prefcc | false | false | held |
| r4a | NLI - C (NOTES only) | -0.025 to -0.005 | **-0.0323** | **missed** (worse than predicted) |

**Totals.** 10 of 12 schema predictions held, and 1 of 2 NLI predictions.

**Direction of the misses.**
- I was too pessimistic about r4b's imitation. I expected BERTScore to fall 1.3-2.4x as much as ROUGE-L;
  it fell 1.10x as much, so criterion 3 was not the coin flip I called.
- I was too optimistic about r4a on recall and on NLI.

**Interval width.** Six of my held intervals were wide (about 0.018-0.02 on recall, 0.013-0.014 on
unsupported share). They were honest about V's resolution, not sharp.

**Earlier phases.**
- On V, 26 of 32 arm intervals held. All four length misses were on the long side.
- r4d was falsified on V by its own clause, and r4c's clean reading was voided by its length.

### The reading I wrote down before the test

From NOTES, phase 2: "r4b beats r4a on unsupported share and NLI: Gemma's flags in samples are real,
and next round should push the claim penalty further."

The outcome on those two metrics:
- NLI: r4b -0.0163, r4a -0.0323 against C. Against prefcc: +0.0080 and -0.0080.
- Unsupported share: r4b +0.0051, r4a +0.0074 against C.

On NLI r4b clearly beats r4a. On unsupported share r4b is lower, but within noise.

**So by my own rule the auditor's reading lost.** Removing the sample-chosen pairs made every test
faithfulness instrument worse or no better, not better.

The second half of that rule ("push the claim penalty further") is contradicted by r4b itself: weight 3
did not lower the evaluation's unsupported share (see 2.3). I withdraw that half.

## 2. What the round showed

### 2.1 Verdict by §9

**r4b meets all four criteria:**
- recall +0.0236, permutation p = 0.0001;
- length 1.005;
- ROUGE-L -0.0057 and BERTScore -0.0063;
- unsupported share +0.0051 [-0.00001, +0.0102]. This passes criterion 4 by 0.00001, even more narrowly
  than prefcc's lower bound of -0.0003.

**r4b against prefcc (two seeds each):**
- recall -0.0025 [-0.0074, +0.0024];
- unsupported share +0.0007 [-0.0036, +0.0052];
- NLI +0.0080, 95% [+0.0006, +0.0154], 98.75% [-0.0016, +0.0174];
- judge +0.021, 98.75% [-0.019, +0.062].

So r4b is not worse on anything and not better on anything at 98.75%. It does not replace prefcc.

**r4a** fails criterion 4 (+0.0074 [+0.0022, +0.0123]). It is also worse than prefcc on recall:
-0.0173, 98.75% [-0.0241, -0.0105].

### 2.2 The sample-chosen (on-policy) pairs carry most of prefcc's recall gain

This is the round's main result.

- r4a is a strict subset of prefcc's pairs: it drops the 293 pairs whose chosen side is one of C's samples
  (24% of the pairs). Two-seed recall went from +0.0262 to +0.0089 against C, so about **two-thirds of
  the gain disappeared**.
- Round 1's independent re-scoring had put only about 13% of the fact contrast in those pairs. Their
  influence on training was far larger than their share of the reward contrast.
- They also helped faithfulness on the test. Without them:
  - NLI was -0.0080 against prefcc [-0.0160, -0.0002];
  - the claim check was +0.0026 (not significant), and criterion 4 failed;
  - the judge was level (+0.004).
- What the teacher-chosen pairs teach on their own is the teacher's wording: ROUGE-L +0.0005 and
  BERTScore -0.0010 against C. That is the first preference arm at C's level on both, but it comes with
  a third of the recall gain and the worst NLI-per-recall ratio measured (-3.6, against prefcc's -0.93).
- **Confound.** r4a also took fewer steps (118 against 155). On V, r4d (filter, claim weight 3 and two
  pairs per paper, 226 steps) did not recover fact against r4a either (about -0.004). That points to the
  missing on-policy pairs, not the missing steps, as the cause. r4d's V result is one seed and also
  changes the claim weight, so this is a hint, not a separation.
- **Inference.** On-policy chosen texts move what the model generates far more than off-policy (teacher)
  chosen texts do, per pair. Round 1 already showed the sample-chosen pairs carry real fact signal on
  independent questions (+0.041 [+0.015, +0.066]). This round shows they carry most of the effect.

### 2.3 Claim weight 1 to 3: the claim check is saturated

- The evaluation's unsupported share moved +0.0007 against prefcc, where the step from 0 to 1 had moved
  it -0.0069 against preflc.
- Meanwhile Gemma's claim share on V moved -0.0057 against prefcc's seed 1, the largest V move recorded.
  So the extra penalty taught the training grader something the evaluation's verifier does not reward.
  The calibration's warning that V's claim share cannot rank faithfulness holds even within one mechanism.
- What weight 3 may have bought:
  - NLI +0.0080 (95% only);
  - a better BERTScore margin (-0.0063 against prefcc's -0.0085);
  - NLI deficit per unit of recall -0.69 against prefcc's -0.93.

  These are hints, not established.

### 2.4 The instruments

- **V's ranking held:**
  - r4b above r4a on recall;
  - ROUGE-L tracked the test: r4a -0.0008 on V, +0.0005 on test; r4b -0.0067 and -0.0057;
  - absolute length tracked seed 1 within 0.003 for both finalists.
- **V understated the gap between r4b and r4a:** 0.0053 on V, 0.0147 on test. Its fact score overrated
  the teacher-only arm. The ratio of test recall to V fact was 0.28 for r4a against 0.64 for r4b.
- **The blind pair audit** (one reader, 30 sample-chosen pairs) was a poor basis for a filter in this
  instance. Its high agreement in the teacher stratum may well be right, but it did not matter for the
  outcome.
- **Seed pattern.** r4b's shuffled-order seed again wrote shorter (-0.040) and recalled less (-0.0097),
  but r4a's wrote longer (+0.012) and recalled +0.002. The pattern is now 4 of 5 arms, consistent with
  noise.

## 3. What I would do differently

1. **Weigh the independent evidence I already had over a new, small audit.** Round 1's re-scoring said
   the sample-chosen pairs carry real fact signal. I discounted that against one reader's preferences on
   30 pairs and put the filter in two of four arms (r4a and r4d). One filter arm, or first a larger audit
   of the sample stratum, would have spent less of the round on a hint.
2. **Do not estimate an arm's effect from its share of the reward contrast.** I predicted r4a's recall
   cost from the 13% of fact contrast it removed; the cost was 66% of the gain. Whether a pair is
   on-policy matters more than its contrast.
3. **Ask for token-based pair statistics before predicting length.** The preview's length measure got
   the direction wrong for r4c and r4d.
4. **Do not use a fixed BERTScore-to-ROUGE-L multiplier.** It ranged from 1.10 to 2.4, and the risk I
   inferred from it for r4b was wrong.
5. **The finalist choice.** I swapped r4c for r4a, against my pre-written plan, to test the Gemma-versus-
   auditor question. The test answered that question decisively, which r4b plus r4c would not have.
   - But r4a's correct share on V was already below prefcc's, with its interval excluding zero, and I
     underweighted that against rule 2's "not worse on recall".
   - Given 2.2, r4c (the only arm with a V recall gain, from more on-policy pairs) asked the more
     important question, and its test is now missing.
   - I still would not have predicted r4c to replace prefcc: it wrote 0.049 longer, with criterion-3 and
     criterion-4 risk.
6. **Get a faithfulness signal onto V before proposing faithfulness arms**, for example the
   evaluation-side claim check. Without it, three of four arms (r4a, r4b, r4d) were bets on faithfulness
   that V could not see.

## 4. What the next round should test first

1. **More on-policy contrast at equal length (inside the space).**
   - Hypothesis: the sample-chosen pairs do the work, so more of them per paper raises recall.
   - On V, r4c (two pairs per paper) gave +0.0148 fact over prefcc but wrote +0.049 longer.
   - Arm: `imp_pairs_per_paper: 2` with a small R-DPO term to hold length at prefcc's. My guess is
     `pref_len_alpha` around 0.0015-0.002; alpha 0.003 took preflc's pairs from about 1.01 to 0.95.
   - Run it on prefcc's or r4b's reward settings, which are equivalent on the test.
   - Read on V: fact at least +0.015 over prefcc with length within 0.02 of prefcc's.
   - A useful contrast: two pairs per paper restricted to sample-chosen papers, against the same number
     of teacher-chosen pairs.
2. **New on-policy samples (outside the space).**
   - Resample from prefcc (or r4b), re-score, and build round-5 pairs from the improved policy. This is
     iterated DPO.
   - 2.2 says on-policy chosen samples move what the model generates, so a fresh round of them is the
     most direct way to extend the gain. More pool papers would add on-policy pairs as well.
3. **Faithfulness from a different source than a heavier Gemma penalty.**
   - prefcc's and r4b's criterion-4 margins are both razor-thin, and raising Gemma's weight no longer
     moves the evaluation's verifier.
   - Candidates: a penalty only on statements two verifiers both flag; or, inside the space, a filter
     that keeps sample-chosen pairs only when the chosen sample is not less faithful (`c_teacher or
     c_claim <= r_claim`). That filter must be checked for its recall cost first, given 2.2.
4. **Instruments.** Claim-check V with the evaluation-side verifier, and score two pair orders per arm on
   V. The calibration and this round both show that V's Gemma claim share does not predict the test.
5. **Not worth repeating.**
   - Teacher-only pair filters: they cost recall and faithfulness.
   - Claim weights above about 2-3: the evaluation's claim check is saturated.
   - Basing a filter on a single-reader audit of 30 pairs.
6. **If a default must be picked for the next arm's reward,** r4b's claim weight of 3 is at least as good
   as 1 on every test criterion: level recall and unsupported share, NLI +0.008 at 95%, and a wider
   BERTScore margin. This is not established at 98.75%, so prefcc remains the recommendation.
