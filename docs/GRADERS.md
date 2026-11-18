# The LLM graders: roles, self-hosting, endpoints, comparability

The reward of the preference stage and most of the evaluation use two open-weight LLMs as graders. The
training side and the evaluation side use the same two models with their roles swapped, so the model that
writes a question never also grades the answer.

| Served name (what the code asks for) | Hugging Face repo | Parameters | Licence | Architecture |
|:--|:--|--:|:--|:--|
| `gemma4-31b-it` | `google/gemma-4-31B-it` (not gated) | 31.3 B | Apache-2.0 | `Gemma4ForConditionalGeneration` (vision-language wrapper) |
| `qwen38-27b` | `Qwen/Qwen3.8-27B` (not gated) | 27.8 B | Apache-2.0 | `Qwen3_5ForConditionalGeneration` (vision-language wrapper) |

Repo ids, sizes and licences were read from the Hugging Face API on 2026-11-17. An FP8 copy exists for
Qwen (`Qwen/Qwen3.8-27B-FP8`). Use bf16 for both: MI250X GPUs (LUMI) have no FP8 support, and one precision
everywhere keeps the scores comparable.

## Who does what

| Stage | Writes the questions | Answers and grades | Checks claims against the paper | Setting |
|:--|:--|:--|:--|:--|
| **Training reward** (preference pairs; `scisum improve ...`) | `qwen38-27b` (14 per paper) | `gemma4-31b-it`, also as the two reference readers | `gemma4-31b-it` | `imp_writer`, `imp_grader`, `imp_claim_model` |
| **Evaluation: fact test** (`scisum factcheck run`) | `gemma4-31b-it` (10 per paper) | `qwen38-27b` | | `fact_gen_model`, `judge_model` |
| **Evaluation: claim check** (`scisum claimcheck run`) | | | `qwen38-27b` | `judge_model` |
| **Evaluation: judge** (1–5 scores against the teacher; `slurm` judge job) | | `qwen38-27b`, thinking off | | `judge_model` |

Two evaluation metrics need no LLM server and run locally on a GPU: NLI faithfulness
(`microsoft/deberta-large-mnli`) and BERTScore (`roberta-large`, layer 17).

**What ships, so the graders do less work.**
- The fact test's two question sets (A and B, 899 test papers) are in `data/eval/`. The evaluation therefore
  only answers and grades; it writes no questions.
- The training-side questions and reference-reader verdicts for the first 2,551 pool papers are in
  `data/improve_pool/`. Each student's improve directory is seeded with them on first use
  (`scisum.improve.seed_shared`). The preference stage then needs a grader only to answer and grade each
  student's own candidates and to check their claims.
- To have a shipped file regenerated with your graders, delete it from `state/improve-<tag>/`.

## Self-hosting with vLLM

The clients speak the OpenAI chat-completions API. Every request has temperature 0 and sends
`chat_template_kwargs: {"enable_thinking": false}`. Requests are long:
- the claim check and the question writer send up to 90,000 characters of paper (about 22,000–28,000
  tokens);
- replies are up to 3,000 tokens.

Serve with a context of at least 40,960 tokens.

| Model | bf16 weights | 64 GB GPUs (LUMI MI250X, one GCD each) | 80 GB GPUs (A100/H100) |
|:--|--:|:--|:--|
| `gemma4-31b-it` | ~62.5 GB | tensor parallel 4. TP 2 leaves no room for the KV cache. | tensor parallel 2 |
| `qwen38-27b` | ~55.6 GB | tensor parallel 4 | tensor parallel 2 (TP 1 is too tight at 40k context) |

One LUMI-G node has 8 GCDs, so both graders fit on one node at TP 4 each. This sizing is estimated from
the weight sizes, not measured: check that the server starts and that the KV-cache line in its log shows
room for at least 8 concurrent 40k-token sequences.

An example server command (the launch scripts, including the LUMI container wrapper, are in `slurm/` and
`docs/LUMI.md`):

```bash
vllm serve google/gemma-4-31B-it --revision <pinned sha> \
    --served-model-name gemma4-31b-it \
    --dtype bfloat16 --tensor-parallel-size 4 \
    --max-model-len 40960 --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    --host 0.0.0.0 --port 8000
```

- `--limit-mm-per-prompt` skips the vision encoder's memory reservation, because only text is sent. The
  flag's syntax differs between vLLM versions.
- Pin the revision you serve, and record it in the run log.
- Concurrency: the clients default to 8 parallel requests for the fact test and 16–24 for the claim check
  and the reward stage. Raise `--max-num-seqs` if the server has KV-cache room.

### Endpoint configuration

Every client asks `scisum.endpoints.resolve(settings, model)` for the URL of the model it is about to call.
The first of these that is set wins:

1. `SCISUM_GRADER_ENDPOINTS`: inline JSON or the path of a JSON file, e.g.
   `{"gemma4-31b-it": "http://nid005123:8000/v1", "qwen38-27b": "http://nid005124:8000/v1"}`. A value may
   also be `{"base_url": "...", "api_key_env": "ENV_VAR_WITH_THE_KEY"}`.
2. `state/graders/*.json`, written by a grader server job once its server answers:
   `{"model": "gemma4-31b-it", "base_url": "http://<node>:8000/v1", "job": "<slurm id>", "started": "<time>"}`.
   The newest file per model wins. Delete a file when its server is gone.
3. `SCISUM_JUDGE_BASE` and `JUDGE_API_KEY` in `.env`: one endpoint for every model, as in the earlier work.

The API key is optional. A vLLM server started without `--api-key` accepts any token, and `EMPTY` is sent
when none is set. URLs and keys are never written to `run.json` or the logs.

**Check before a grading job:**

```bash
scisum graders check          # per model: where the URL came from, reachable, serves that name; exit 1 if not
```

### Throughput to expect

Measured on the earlier shared endpoint (one run each, 2026-11-16):

| Job | Papers | Time |
|:--|--:|--:|
| Fact test, one question set, one system | 300 | ~11 min |
| Claim check, one system | ~300 | ~13 min |
| Judge, one system | 500 | ~6 min (1.4 papers/s) |

Never run several large gradings against the same server at once. Five concurrent gradings saturated one
server and every job hit its time limit with thousands of read timeouts (2026-11-16). Grade one system
after another, or give each grading its own server.

## Comparability: re-grade the anchors first

The earlier scores were produced by a shared inference service. It named the models
(`gemma4-31b-it`, `qwen38-27b`) but did not expose their exact revision, precision or chat-template
version. `Qwen/Qwen3.8-27B` was updated on Hugging Face on 2026-11-07, after most of that grading. A
self-hosted grader is therefore a new instrument. Its absolute numbers (fact recall, unsupported share,
judge scores) can differ from the published ones even for the same summaries.

Before comparing a new student with Student C or `prefcc`:
1. Grade the shipped anchor summaries with your graders, on the same question sets: the teacher, `C`,
   `C_prefcc` and `C_prefccs2` (`data/anchors/`).
   ```bash
   scisum factcheck run --systems teacher,C,C_prefcc,C_prefccs2
   ```
2. Check that the anchors keep their ordering and approximate gaps. Earlier values, two-seed mean for
   prefcc:

   | System | Fact recall |
   |:--|--:|
   | teacher | 0.608 |
   | C | 0.571 |
   | prefcc | 0.597 |

3. Compare students only with numbers from the same grader setup, and report the anchors' re-graded values
   next to them.

The shipped training-side questions and reference verdicts were made with the earlier service. Using them
with self-hosted graders changes only which questions count for each paper, not how candidates are
graded. Regenerating them costs about 2,551 × 3 grader calls; it is optional.
