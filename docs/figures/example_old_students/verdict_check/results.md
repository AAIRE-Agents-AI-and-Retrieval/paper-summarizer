## Quality on the test papers

Teacher: fact recall 0.6082 [0.5981, 0.6182] (899 papers), unsupported share 0.0296 [0.0220, 0.0385].

| system | fact recall [95% CI] | recall − teacher | strict recall | unsupported share | length ratio | ROUGE-L | BERTScore | NLI | judge faithfulness (1–5) | JSON valid |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| C Qwen3.5-9B 20k | 0.5708 [0.5604, 0.5812] | -0.0374 [-0.0450, -0.0294] | 0.4657 | 0.0339 | 0.988 | 0.4789 | 0.4580 | 0.4695 | 2.172 | 0.9989 |
| prefcc (C + DPO) | 0.5970 [0.5870, 0.6067] | -0.0112 [-0.0186, -0.0041] | 0.4944 | 0.0378 | 0.988 | 0.4735 | 0.4498 | 0.4452 | 2.072 | 0.9972 |

## Against the reference (normally the student's own SFT model)

Criterion 1 needs the recall interval above zero and p below 0.0500 (alpha 0.05 / 1 comparisons). Criteria: 1 recall, 2 length 0.90–1.10, 3 ROUGE-L and BERTScore ≥ −0.010, 4 unsupported share not higher.

| system | reference | recall − ref | p | length ratio | ROUGE-L − ref | BERTScore − ref | unsupported − ref | NLI − ref | judge − ref | 1 | 2 | 3 | 4 | all four |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| prefcc (C + DPO) | C Qwen3.5-9B 20k | +0.0262 [+0.0201, +0.0322] | 0.0001 | 0.988 | -0.0053 [-0.0072, -0.0035] | -0.0085 [-0.0113, -0.0060] | +0.0045 [-0.0003, +0.0093] | -0.0243 [-0.0341, -0.0152] | -0.090 [-0.139, -0.039] | yes | yes | yes | yes | yes |

## Cost

| system | params (B) | train GPU-h | train GPU | train peak GB | GPU-s / paper | papers / h | TTFT p50 s | e2e p50 s | e2e p95 s | weights GB | KV GB / paper p95 | serving GPU | cost source |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| C Qwen3.5-9B 20k | 9.10 | 34.0 | NVIDIA A100-SXM4-80GB | 38.1 | 1.82 | 1973 | – | – | – | – | – | A100 80GB | manual: test evaluation run, 32,768-token window |
| prefcc (C + DPO) | 9.10 | 36.1 | NVIDIA A100-SXM4-80GB | 38.1 | 1.82 | 1973 | – | – | – | – | – | A100 80GB | manual: assumed equal to C: same base model and adapter rank |
