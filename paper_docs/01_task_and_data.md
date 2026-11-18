# Task and data

*Scope (2026-11-16).* The improvement work after SFT draws on the validation split: a 3,000-paper pool (1,520 sampled for
candidates), the reactive run's 200 validation papers, and round 4's V (600 papers, 376 in reserve). It is
evaluated on 1,246 test papers. See `11_reward_selected_distillation.md` and `12_reactive_training.md`.

## Task

Input: one scientific paper — title, up to eight authors, arXiv categories, abstract, and the
GROBID-extracted body with references and appendices removed.

Output: one JSON object with exactly eight fields: `tldr`, `problem`, `contributions` (list),
`method`, `results`, `limitations`, `keywords` (list), `domain`.

The prompt is the teacher pipeline's production prompt, unchanged
(its system prompt plus its keys block). `tests/test_prompt_parity.py`
asserts, by running the teacher's own code in the teacher's own virtual environment, that the
prompt text and the body-fitting logic are byte-identical to what generated the targets.

## Corpus

| | |
|:--|:--|
| Papers | the 100,000 most-cited arXiv papers by OpenAlex citation count (123–83,445 citations) plus a 500-paper random calibration cohort |
| Full text | GROBID TEI from publisher PDFs; body and figure/table captions only |
| Metadata | arXiv API (title, authors, categories, abstract) |
| Targets | 100,826 structured summaries generated from those full texts through OpenRouter at a cost of $132.25 |
| Teacher models | `deepseek/deepseek-v4-flash-0731` (88,404 eligible), `deepseek/deepseek-v4.1-flash` (8,142), `meta/muse-spark-1.3-contributor` (4,098) |

The targets are machine-generated: this is distillation, not ground truth. Every evaluation is
therefore also reported per teacher model.

## Eligibility

100,644 of 100,826 papers are eligible. Exclusions, each recorded with its reason in
`state/index_drops.jsonl`:

| Reason | Papers |
|:--|--:|
| `bad_summary:contributions_count` (fewer than 2 or more than 10) | 85 |
| `duplicate_title` (same work under two arXiv ids; the more-cited one kept) | 72 |
| `short_body` (GROBID body under 2,000 characters) | 13 |
| `abstract_only` (summarised without a full text) | 10 |
| `bad_summary:repetitive` (distinct-10 below 0.80) | 1 |
| `bad_summary:keywords_count` | 1 |

The repetition threshold was calibrated on the distribution of all 100,827 teacher summaries
(median distinct-10 = 1.000, 1st percentile 0.965, minimum 0.583); an earlier rule that rejected
any 8-gram repeated four times removed 851 good summaries, because a repeated formula or a long
technical term looks like a loop in a dense scientific summary.

## Splits

Stratified by (teacher model, cohort), split by paper, one seed (20261014). Every eligible paper is
assigned; consumers take the first *n* of each fixed order, so a run that trains on fewer papers
uses a prefix of the same sequence.

| | train | val | test |
|:--|--:|--:|--:|
| candidates | 70,445 | 15,101 | 15,098 |
| used by runs A and B | 70,000 | 15,000 | 15,000 |
| used by run C | 20,000 (prefix of the same order) | 15,000 | 15,000 |

Fixed subsets of validation, chosen once: `curve_eval` = 1,000 papers (generation at every
milestone), `loss_eval` = its first 200 (held-out loss during training).

## How one training example is built

1. **Target**: the eight content fields in schema order as `json.dumps(..., ensure_ascii=False)`,
   wrapped in whatever the model's chat template puts around an assistant answer, derived from the
   template itself (`scisum.prompt.assistant_wrapper`): `("", "<|im_end|>")` for Qwen3-4B,
   `("</ifm|think>", "<|ifm|im_end|>")` for K2-Horizon, `("\n</think>\n\n", "<|im_end|>")` for
   Qwen3.5 — the latter two always open a reasoning block in their generation prompt.
2. **Prompt**: the production prompt through the model's chat template, with the body fitted into
   `max_seq_len − gen_reserve_tokens` tokens. The reserve is fixed rather than set from the target's
   length, so training and inference use the same budget and the reference length cannot leak into
   the prompt. Sections are dropped least-useful-first (related work, appendices, acknowledgements
   before results and methods); if one section is still too large the body is cut head-and-tail.
3. **Loss mask**: the prompt is not scored; the target and its terminator are.

## Dataset card per run

| | A (Qwen3-4B) | B (K2-Horizon-7B) | C (Qwen3.5-9B) |
|:--|--:|--:|--:|
| sequence length | 16,384 | 16,384 | 32,768 |
| prompt budget | 13,312 | 13,312 | 28,672 |
| training papers | 70,000 | 70,000 | 20,000 |
| tokens per pass | 730 M | 702 M | 272 M |
| **whole body in the prompt** | **60.58%** | **62.99%** | **91.14%** |
| no shrink round needed (`none`) | 71.48% | 81.66% | 93.21% |
| sections dropped (`sections`) | 19,960 | 12,834 | 1,357 |
| head-and-tail cut (`hard`) | 7 | 4 | 1 |
| prompt tokens, median / p90 | 10,306 / 13,136 | 9,776 / 13,062 | 10,624 / 25,458 |
| target tokens, median / p90 | 995 / 1,258 | 920 / 1,152 | 992 / 1,255 |

**Two different measures of truncation, and they must not be confused.** `fits` (the bold row) is
true only when the complete GROBID body reached the prompt. `trunc_kind` records whether the
*fitting loop* had to run: the first attempt already caps the body at `budget × 4` characters, so a
paper can lose its last sections on that first attempt and still be recorded as `none`. The gap
between the two rows — 60.58% against 71.48% for Run A — is exactly that population: 7,665 training
papers that were quietly clipped without a shrink round. Every `by_truncation` breakdown in
`05_results.md` groups by `trunc_kind`, so its `none` bucket contains them too.

Tokenizer efficiency, measured on 40 real paper bodies averaging 45,950 characters
(`scripts/compare_candidates.py`, `state/candidates.json`): K2-Horizon 3.28 characters per token,
gemma-4-12B 3.22, Qwen3.5 3.14, Qwen3-4B 3.09, Nemotron-3.5-Lightning 3.03, granite-4.2-8b 2.97.

Every build is checked by `scisum data verify`: no prompt over the budget, no example over
`max_seq_len`, every example ending in the terminator its template specifies, and a sample of
stored evaluation prompts re-tokenised to confirm they match their recorded length.
