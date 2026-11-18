# Evaluation protocol

*Scope (2026-11-16).* The improvement rounds use a further protocol on 1,246 test papers:
- a fact test on 899 papers with two question sets;
- a claim check of every summary statement against the paper (300 papers);
- four criteria fixed before evaluation, with Bonferroni-corrected permutation tests.
See `11_reward_selected_distillation.md` ("Evaluation", "Round 2") and `12_reactive_training.md` §8–§9. T16–T21 in
`07` list its limits.

Three groups of metrics plus an independent judge, because a search-engine summary can fail in
independent ways. Everything is computed per paper, stored per paper
(`runs/<run>/eval/*.scores.jsonl.gz`), and aggregated with breakdowns by teacher model and by how
the paper's body was fitted into the prompt.

Generation: vLLM, greedy (temperature 0), the LoRA adapter loaded as an adapter rather than merged,
`max_model_len` equal to the model's training window, `max_tokens = min(3000, window − prompt)`.
The 3,000-token cap is above the longest reference target (2,463 tokens) and bounds the cost of a
checkpoint that has not yet learned to stop. Greedy removes sampling noise but is not
bit-reproducible: vLLM's batched kernels moved per-paper metrics by up to 0.04 between two runs of
the same checkpoint on the same eight papers, which is why differences between adjacent milestones
of that size are not read as signal.

## 1. Is the output usable?

`json_ok` (parses as JSON with no repair, after removing the prefix the model's own template makes
it emit), `repaired`, `schema_ok` (all eight fields, correct types), `keywords_in_range` (5–10),
`contributions_in_range` (2–5), `extra_fields`, `type_errors`, `stopped_on_eos`, `hit_token_limit`,
`out_tokens`, `length_ratio` (generated ÷ reference characters), `distinct4`, `max_repeat10`.

## 2. Is it close to the teacher?

`all_rouge1/2/L` — ROUGE F1 (Lin 2004, Google `rouge_score`, Porter stemming) between the whole
rendered summary and the reference; `<field>_rougeL` per field; `bertscore_<field>` and
`bertscore_mean` — BERTScore F1 (Zhang et al. 2020) with `roberta-large` layer 17, baseline-rescaled
(0.8312), computed **per field** because a whole summary exceeds the encoder's 512-token window;
`keywords_p/r/f1` — set metrics over normalised phrases; `domain_exact`, `domain_f1`.

BERTScore is implemented in `scisum/bertscore.py` rather than through the reference package, which
calls a tokenizer method transformers 5 removed; only the package's published baseline table is
used. `tests/test_bertscore.py` checks identical text scores 1.0 and unrelated text below 0.3.

## 3. Is it faithful to the paper?

SummaC-style sentence-level NLI with retrieval (Laban et al. 2022). Each claim (sentences of the
prose fields plus each contribution, ≥25 characters, at most 40 per paper) is paired with the four
source chunks it is lexically closest to — BM25 over overlapping three-sentence chunks of the
abstract and body — and scored by `microsoft/deberta-large-mnli`. The claim takes the best
entailment probability over its chunks; the paper takes the mean over its claims
(`faithfulness`), and the share of claims whose best evidence does not entail them is
`unsupported_rate`. Computed on the first 300 papers of each evaluation.

**Read it as a relative measure only.** Retrieval can miss the right evidence and an NLI model is
not a domain expert.

## 4. Is it as good as the teacher it was distilled from?

`qwen38-27b` on an OpenAI-compatible endpoint — a third model family, neither teacher —
sees a 9,000-character excerpt of the paper and **both** summaries, labelled A and B in an order
fixed by a hash of the paper id so that its position bias cannot align with ours. It scores both on
faithfulness, coverage, specificity and fluency (1–5) and states a preference.

Reported: `judge_<axis>` and `judge_mean` for the student, `judge_ref_*` for the teacher from the
same call, `judge_delta` (student − teacher) and `judge_win/tie/loss`.

Scoring both summaries in one call is what makes the absolute numbers interpretable: on a first
sample the student scored 2.1 on faithfulness, which looked alarming until the teacher, judged
against the same excerpt, scored 2.25 — most of that number is the judge seeing a ninth of the
paper. **Only the delta and the preference are treated as meaningful.** Thinking is disabled for
every call: with reasoning enabled the model spent its whole token budget thinking and returned
nothing on seven papers in eight, and mixing modes would mix two scales in one number.

**The judge anchors.** The same teacher summaries score 3.87–3.92 when compared against a
fine-tuned Qwen3-4B checkpoint, 4.11 against the untrained one, and 3.83 against K2-Horizon. Only
in-context deltas are comparable across checkpoints, and cross-model judge deltas must be read with
this in mind.

## Protocol

| When | What | Cost |
|:--|:--|:--|
| every 2,500–10,000 training papers, and at every milestone | held-out loss (teacher-forced) on a fixed 200-paper subset | ~1 min |
| every milestone and the untrained base model | greedy generation on a fixed, stratified 1,000-paper validation subset, then every metric above | 0.3–1 h |
| once, at the end | the whole 15,000-paper test split on the final checkpoint; 3,000 of the same papers on the base model | 3–4 h |

The test split is read by one job, at the end. Everything during training uses validation, and the
1,000-paper curve subset is fixed in `state/splits.json`.

## Comparing losses across models

Per-token cross-entropy is not comparable between models with different vocabularies. Where a loss
comparison is made it is converted to **nats per paper** (loss × mean target tokens per paper),
which is the negative log-likelihood of the same reference string and is segmentation-independent.


---

## Fact-level evaluation (added 2026-10-25)

Every metric above compares a summary with the teacher's. This one does not (`scisum/factcheck.py`,
`slurm/factcheck.sbatch`). Per paper: (1) `gemma4-31b-it` reads the paper's full text, up to 90,000
characters, and writes ten questions with gold answers of at most 30 words, with a fixed mix of
types (problem 1, motivation 1, approach 2, setup 2, finding 2, limitation 1, scope 1), each specific
to the paper and answerable from it; (2) `qwen38-27b` answers all ten from *one summary only*, in the
same labelled rendering for every system, with NOT STATED allowed; (3) `qwen38-27b` grades each
attempted answer as correct, partial or incorrect against the gold. A system whose output does not
parse answers NOT STATED to everything. `recall` = (correct + ½ partial) / questions; `accuracy`
= correct / attempted. Papers: 300, stratified by archive, with the random cohort included first,
drawn from the first 3,000 test papers (which have untrained-model outputs). Differences between
systems are paired bootstrap intervals over papers (2,000 resamples). The question set is keyed by a
fingerprint of its prompt, model and size, so a changed prompt cannot silently mix with an old set.

**ACLSum** (`scripts/aclsum_eval.py`): 100 test papers with human-written challenge / approach /
outcome sentences; the mapping to our fields and the caveats are in `05_results.md`.
