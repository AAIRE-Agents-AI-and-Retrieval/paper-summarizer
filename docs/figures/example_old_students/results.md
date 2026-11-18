## Quality on the test papers

Teacher: fact recall 0.6182 [0.6012, 0.6355] (299 papers), unsupported share 0.0293 [0.0219, 0.0377].

| system | fact recall [95% CI] | recall − teacher | strict recall | unsupported share | length ratio | ROUGE-L | BERTScore | NLI | judge faithfulness (1–5) | JSON valid |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A Qwen3-4B 70k | 0.5268 [0.5070, 0.5462] | -0.0915 [-0.1082, -0.0758] | 0.4217 | – | 0.968 | 0.4434 | 0.4184 | 0.4520 | 2.142 | 0.9856 |
| B K2-Horizon-7B 70k | 0.5719 [0.5548, 0.5911] | -0.0463 [-0.0615, -0.0311] | 0.4659 | – | 0.992 | 0.4695 | 0.4431 | 0.4185 | 2.196 | 0.9912 |
| C Qwen3.5-9B 20k | 0.5741 [0.5572, 0.5911] | -0.0441 [-0.0574, -0.0304] | 0.4632 | 0.0333 | 0.991 | 0.4788 | 0.4583 | 0.4695 | 2.162 | 0.9992 |
| prefcc (C + DPO) | 0.6033 [0.5869, 0.6205] | -0.0150 [-0.0280, -0.0024] | 0.4945 | 0.0378 | 0.988 | 0.4735 | 0.4498 | 0.4452 | 2.072 | 0.9972 |

## Against the reference (normally the student's own SFT model)

Criterion 1 needs the recall interval above zero and p below 0.0500 (alpha 0.05 / 1 comparisons). Criteria: 1 recall, 2 length 0.90–1.10, 3 ROUGE-L and BERTScore ≥ −0.010, 4 unsupported share not higher.

| system | reference | recall − ref | p | length ratio | ROUGE-L − ref | BERTScore − ref | unsupported − ref | NLI − ref | judge − ref | 1 | 2 | 3 | 4 | all four |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| prefcc (C + DPO) | C Qwen3.5-9B 20k | +0.0292 [+0.0181, +0.0398] | 0.0001 | 0.988 | -0.0053 [-0.0072, -0.0035] | -0.0085 [-0.0113, -0.0060] | +0.0045 [-0.0003, +0.0093] | -0.0243 [-0.0341, -0.0152] | -0.090 [-0.141, -0.040] | yes | yes | yes | yes | yes |

## Cost

| system | params (B) | train GPU-h | train GPU | train peak GB | GPU-s / paper | papers / h | TTFT p50 s | e2e p50 s | e2e p95 s | weights GB | KV GB / paper p95 | serving GPU | cost source |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A Qwen3-4B 70k | 4.16 | 27.5 | NVIDIA H200 | 16.5 | 0.70 | 5177 | – | – | – | – | – | H200 | manual: test evaluation run, 15,000 papers, 16,640-token window |
| B K2-Horizon-7B 70k | 9.17 | 33.2 | NVIDIA H200 | 29.6 | 0.89 | 4046 | – | – | – | – | – | H200 | manual: test evaluation run, 15,000 papers, 16,640-token window |
| C Qwen3.5-9B 20k | 9.10 | 34.0 | NVIDIA A100-SXM4-80GB | 38.1 | 1.82 | 1973 | – | – | – | – | – | A100 80GB | manual: test evaluation run, 32,768-token window |
| prefcc (C + DPO) | 9.10 | 36.1 | NVIDIA A100-SXM4-80GB | 38.1 | 1.82 | 1973 | – | – | – | – | – | A100 80GB | manual: assumed equal to C: same base model and adapter rank |

## Missing inputs

| system | missing |
|---|---|
| A Qwen3-4B 70k | main: no claim check for label 'A' (qwen38-27b) |
| B K2-Horizon-7B 70k | k2: no claim check for label 'B' (qwen38-27b) |
