"""BERTScore (Zhang et al., ICLR 2020), implemented directly against transformers.

The reference package (`bert-score` 0.3.13) calls `build_inputs_with_special_tokens`, which
transformers 5 removed, so it cannot run in this environment — and the environment is pinned by
vLLM. The metric itself is twenty lines: contextual embeddings from one layer, L2-normalised,
greedy-matched between candidate and reference tokens, F1 of the matched cosine similarities.

Defaults match the published English configuration: `roberta-large`, layer 17, no IDF weighting,
special tokens excluded, and rescaling against the package's own baseline table
(`rescale_baseline/en/roberta-large.tsv`, layer 17 → 0.8312), which is read from the installed
wheel so the numbers stay comparable with published ones. Rescaling maps the baseline of unrelated
sentence pairs to 0 and identical text to 1; `tests/test_bertscore.py` checks both ends.
"""

from __future__ import annotations

import csv
import logging
from functools import lru_cache
from pathlib import Path

import torch

log = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def baseline(model_name: str = "roberta-large", layer: int = 17, lang: str = "en") -> float:
    """The published rescaling constant (F), or 0.0 if the table is not installed."""
    try:
        import bert_score

        path = Path(bert_score.__file__).parent / "rescale_baseline" / lang / f"{model_name}.tsv"
        with path.open() as f:
            for row in csv.DictReader(f):
                if int(row["LAYER"]) == layer:
                    return float(row["F"])
    except (ImportError, OSError, KeyError, ValueError) as e:  # pragma: no cover - depends on install
        log.warning("no BERTScore baseline for %s layer %d (%s); reporting unrescaled F1",
                    model_name, layer, e)
    return 0.0


class BertScorer:
    def __init__(self, model_name: str = "roberta-large", layer: int = 17, device: str = "cuda",
                 batch_size: int = 64, max_length: int = 512, rescale: bool = True):
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.layer, self.device, self.batch_size, self.max_length = layer, device, batch_size, max_length
        self.b = baseline(model_name, layer) if rescale else 0.0

    @torch.no_grad()
    def _embed(self, texts: list[str]) -> list[torch.Tensor]:
        """One L2-normalised embedding per content token (special tokens dropped)."""
        out: list[torch.Tensor] = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tok(texts[i:i + self.batch_size], truncation=True, max_length=self.max_length,
                           padding=True, return_tensors="pt").to(self.device)
            hs = self.model(**enc, output_hidden_states=True).hidden_states[self.layer]
            hs = torch.nn.functional.normalize(hs.float(), dim=-1)
            for j in range(hs.size(0)):
                n = int(enc["attention_mask"][j].sum())
                out.append(hs[j, 1:max(n - 1, 2)].cpu())  # drop <s> and </s>
        return out

    def score(self, candidates: list[str], references: list[str]) -> list[dict]:
        cand = self._embed([c if c.strip() else " " for c in candidates])
        ref = self._embed([r if r.strip() else " " for r in references])
        rows = []
        for c, r in zip(cand, ref, strict=True):
            sim = c @ r.T                                  # (cand tokens, ref tokens)
            p = float(sim.max(dim=1).values.mean())        # each candidate token's best reference
            rec = float(sim.max(dim=0).values.mean())      # each reference token's best candidate
            f = 2 * p * rec / (p + rec) if p + rec else 0.0
            rows.append({k: round((v - self.b) / (1 - self.b) if self.b else v, 5)
                         for k, v in (("p", p), ("r", rec), ("f1", f))})
        return rows
