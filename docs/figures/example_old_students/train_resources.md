| run | kind | GPUs | GPU type | wall h | GPU-h | examples | tokens (M) | tok/s/GPU | peak GB | GPU-h / 1k ex | chunks | finished |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| main | sft | 2 | NVIDIA H200 | 13.76 | 27.52 | 70,000 | 728.5 | 7514 | 16.5 | 0.393 | 2 | yes |
| k2 | sft | 2 | NVIDIA H200 | 16.62 | 33.25 | 70,000 | 700.8 | 5950 | 29.6 | 0.475 | 1 | yes |
| q35 | sft | 2 | NVIDIA A100-SXM4-80GB | 16.99 | 33.97 | 20,000 | 271.2 | 2299 | 38.1 | 1.699 | 1 | yes |
| q35prefcc | preference | 1 | – | 2.12 | 2.12 | 1,233 | 33.6 | 4410 | 37.2 | 1.716 | 1 | yes |
| q35prefccs2 | preference | 1 | – | 7.28 | 7.28 | 1,233 | 33.6 | 1282 | 37.1 | 5.905 | 1 | yes |
