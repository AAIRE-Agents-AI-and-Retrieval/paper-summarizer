# Evaluation: what is measured, how, and what it does not tell you

Three groups of metrics, because a search-engine summary can fail in three independent ways. All
of them are computed per paper, stored per paper (`runs/<run>/eval/*.scores.jsonl.gz`), and
aggregated with breakdowns by teacher model and by how the paper's body was fitted.

## 1. Is the output usable?

Measured on every generation; these are the metrics fine-tuning is expected to fix first.

| Metric | Definition |
|:--|:--|
| `json_ok` | the reply parses as JSON with no repair |
| `repaired` | it parses only after stripping a fence/preamble or closing the object (a tolerant parser is used so one stray token does not hide a good summary, but the untouched rate is what is reported) |
| `schema_ok` | all eight fields present, every type correct |
| `keywords_in_range`, `contributions_in_range` | the schema's own counts (5-10 keywords, 2-5 contributions) |
| `extra_fields`, `type_errors` | drift away from the schema |
| `stopped_on_eos`, `hit_token_limit` | generation terminated by itself rather than being cut off |
| `out_tokens`, `length_ratio` | output length, and length relative to the reference |
| `distinct4`, `max_repeat10` | repetition: the share of distinct 4-grams, and the worst repeated 10-gram |

## 2. Is it close to the teacher?

| Metric | Definition |
|:--|:--|
| `all_rouge1/2/L` | ROUGE-1/2/L F1 (Lin 2004, Google `rouge_score`, Porter stemming) between the whole rendered summary and the reference |
| `<field>_rougeL` | the same per field: `tldr`, `problem`, `method`, `results`, `limitations`, `contributions` |
| `bertscore_<field>`, `bertscore_mean` | BERTScore F1 (Zhang et al. 2020) with `roberta-large`, layer 17, baseline-rescaled (0.8312), computed **per field** because a whole summary exceeds the encoder's 512-token window. Implemented in `scisum/bertscore.py`: the `bert-score` package itself calls a tokenizer method transformers 5 removed, so only its published baseline table is used. |
| `keywords_p/r/f1` | set precision/recall/F1 over normalised keyword phrases |
| `domain_exact`, `domain_f1` | exact match and token F1 on the one-field `domain` |

ROUGE is reported because it is the standard and because the reference is a single teacher summary;
it is never read alone. Two summaries can describe the same paper correctly with little n-gram
overlap, which is what BERTScore and the faithfulness metric are for.

## 3. Is it faithful to the paper?

`scisum/faithfulness.py`: SummaC-style (Laban et al. 2022) sentence-level NLI, with retrieval.
Each claim in the generated summary (sentences of the prose fields, plus each contribution, at
least 25 characters, at most 40 claims) is paired with the four source chunks it is lexically
closest to (BM25 over overlapping three-sentence chunks of the abstract and body), scored by
`microsoft/deberta-large-mnli`, and credited with the best entailment probability over those
chunks. The paper's score is the mean over its claims (`faithfulness`), and the share of claims
whose best evidence does not entail them is reported as `unsupported_rate`.

Read it as a *relative* measure — "milestone 20 000 is more grounded than milestone 1 000" — never
as an absolute hallucination rate: retrieval can miss the right evidence, and an NLI model is not
a domain expert. It is computed on the first 300 papers of each milestone evaluation (the full
subset at the end) because it costs ~160 NLI pairs per paper.

## Protocol

| When | What | Cost |
|:--|:--|:--|
| every 5 000 training papers, and at every milestone | held-out loss (teacher-forced) on a fixed 500-paper subset of the validation split | ~1 min |
| every milestone (1k, 5k, 10k, 20k, 50k, 70k) + the base model | greedy generation (temperature 0) on a fixed, stratified 1 000-paper validation subset, then every metric above | ~0.5 h |
| once, at the end | the final checkpoint on the whole 15 000-paper test split; the base model on 3 000 of the same papers | ~4-6 h |

Generation uses vLLM with the LoRA adapter loaded as an adapter (not merged), `max_model_len`
16 640, greedy, `max_tokens = min(3 000, 16 640 − prompt)`. The 3 000-token cap is twice the
reference p99 and above the longest reference (2 463 tokens), so it never truncates a well-behaved
summary — but it bounds the cost of an early checkpoint that has not yet learned to stop. The
smoke test showed exactly that: after 16 training papers the model writes correct JSON and then
runs to the token limit, which `stopped_on_eos` and `hit_token_limit` report at every milestone. Greedy decoding removes sampling noise, and the engine is
the same one that would serve the model. It is not bit-reproducible across runs: vLLM's batched
kernels depend on how requests are grouped, which moved per-paper metrics by up to 0.04 between two
runs of the same checkpoint on the same eight papers. On the 1 000-paper curve subset that
averages out to well under a point, but it is the reason small differences between adjacent
milestones should not be over-read.

The test split is read by one job, at the end, once. Everything during training uses the validation
split, and the 1 000-paper curve subset is fixed in `state/splits.json` so the milestones are
comparable to each other.

## 4. Is it as good as the teacher it was distilled from?

The three groups above all measure the student against *one* reference, which is the teacher. None
of them can say whether a different summary is as useful — so each checkpoint is also judged by a
third model, `qwen38-27b` on an OpenAI-compatible endpoint (from neither teacher's family).

For every paper in the curve subset the judge sees an excerpt of the paper (9 000 characters) and
**both** summaries, labelled A and B in an order fixed by a hash of the paper id so that its
position bias cannot align with ours. It scores both on faithfulness, coverage, specificity and
fluency (1-5) and says which it would rather have.

| Metric | Definition |
|:--|:--|
| `judge_<axis>`, `judge_mean` | the student's scores |
| `judge_ref_<axis>`, `judge_ref_mean` | the teacher's scores, from the same call |
| `judge_delta` | student minus teacher — the number that matters |
| `judge_win`, `judge_tie`, `judge_loss` | preference rates against the teacher |

Scoring both summaries in the same call is what makes the absolute numbers readable. On a first
sample the student scored 2.1 on faithfulness — alarming until the teacher, judged against the
same excerpt, scored 2.25: most of that number is the judge seeing a ninth of the paper, not the
student hallucinating. Only the delta and the preference should be read.

Two further caveats: a judge of this size is known to prefer longer and more specific prose, and
`qwen38-27b` reasons before answering — on these prompts it spent its whole token budget thinking
and returned nothing for seven papers in eight, so thinking is disabled for every call rather than
for some (mixing the two would mix two scales in one number). Each judgement takes about 1.3
seconds, so the whole 1 000-paper subset costs about two minutes per checkpoint.

## What is deliberately not measured

* **Human judgement.** Nothing here substitutes for reading summaries; `runs/*/eval/*.pred.jsonl.gz`
  keeps every generated summary for exactly that.
* **Retrieval quality in the search system.** The summaries feed a search index, and the honest
  end-to-end metric is retrieval quality on real queries. That needs a query set and is out of
  scope here; `keywords_f1` and `domain_exact` are the proxies.
