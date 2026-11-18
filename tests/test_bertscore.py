"""BERTScore: the baseline table, and the two ends of the scale."""

from __future__ import annotations

import pytest

from scisum.bertscore import baseline

torch = pytest.importorskip("torch")


def test_baseline_is_the_published_constant():
    assert round(baseline("roberta-large", 17), 4) == 0.8312
    assert baseline("roberta-large", 0) < baseline("roberta-large", 17)
    assert baseline("no-such-model", 17) == 0.0


@pytest.mark.slow
def test_identical_text_scores_one_and_unrelated_text_scores_low():
    from scisum.bertscore import BertScorer

    s = BertScorer(device="cpu", batch_size=4)
    rows = s.score(["The model improves ROUGE-L from 71.0 to 74.0 on arXiv papers.",
                    "Bananas are yellow and grow in the tropics."],
                   ["The model improves ROUGE-L from 71.0 to 74.0 on arXiv papers.",
                    "The model improves ROUGE-L from 71.0 to 74.0 on arXiv papers."])
    assert rows[0]["f1"] == pytest.approx(1.0, abs=1e-4)
    assert rows[1]["f1"] < 0.3
