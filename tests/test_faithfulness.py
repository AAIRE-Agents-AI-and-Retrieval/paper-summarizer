"""Claim extraction and the BM25 retrieval that selects evidence for the NLI model."""

from scisum.faithfulness import Bm25, chunk_sentences, claims_of, split_sentences

SUMMARY = {
    "tldr": "A new loss for abstractive summarisation of scientific papers improves ROUGE-L by 3 points.",
    "problem": "Extractive summarisers ignore the discourse structure of a scientific paper.",
    "contributions": ["A discourse-aware training loss for long-document summarisation.",
                      "A benchmark of 70000 arXiv papers with GROBID-extracted full texts.",
                      "An ablation isolating the contribution of the discourse term."],
    "method": "A transformer trained with the new loss on 70000 arXiv papers.",
    "results": "The method improves ROUGE-L from 71.0 to 74.0 on the held-out split.",
    "limitations": "The approach is limited to English papers whose PDFs parse cleanly.",
    "keywords": ["summarisation", "discourse", "arxiv", "rouge", "transformers"],
    "domain": "natural language processing"}

PAPER = ("We introduce a new loss for abstractive summarisation of scientific papers. "
         "Prior work on extractive methods ignores the discourse structure of a paper. "
         "Our model is a transformer trained on 70000 arXiv papers with GROBID full texts. "
         "On the held-out split the method improves ROUGE-L from 71.0 to 74.0 points. "
         "The approach is limited to English and to papers whose PDFs parse cleanly.")


def test_sentence_split_and_chunking():
    sents = split_sentences(PAPER)
    assert len(sents) == 5
    chunks = chunk_sentences(sents, size=3, stride=2)
    assert chunks and all(isinstance(c, str) for c in chunks)
    assert chunk_sentences([], 3) == []


def test_claims_come_from_prose_and_contributions():
    claims = claims_of(SUMMARY)
    assert len(claims) == 8  # five prose fields, one sentence each, plus three contributions
    assert any("71.0 to 74.0" in c for c in claims)
    assert any("discourse-aware training loss" in c for c in claims)
    assert claims_of(None) == []
    assert len(claims_of(SUMMARY, max_claims=4)) == 4


def test_short_fragments_are_not_claims():
    assert claims_of({**SUMMARY, "results": "It works."})[-4:-3] != ["It works."]
    assert "ok" not in claims_of({**SUMMARY, "contributions": ["ok"]})


def test_bm25_retrieves_the_chunk_that_states_the_claim():
    chunks = chunk_sentences(split_sentences(PAPER), size=1, stride=1)
    bm = Bm25(chunks)
    top = bm.top_k("ROUGE-L improves from 71.0 to 74.0", 1)[0]
    assert "74.0" in chunks[top]
    top = bm.top_k("the model is a transformer trained on arXiv papers", 1)[0]
    assert "transformer" in chunks[top]
