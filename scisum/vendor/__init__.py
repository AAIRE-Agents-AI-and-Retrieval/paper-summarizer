"""The teacher pipeline's prompt code, copied so that this repository runs alone.

* :mod:`scisum.vendor.teacher_prompts`: the teacher's system prompt, output schema, key list and section
  selection.
* :mod:`scisum.vendor.tei_sections`: the shape of a parsed GROBID body as :func:`scisum.prompt.fit_body`
  consumes it, and the document id.

The copies are verbatim in behaviour. ``tests/test_prompt_parity.py`` checks them against hashes recorded with the
original code (``tests/fixtures/prompt_golden.jsonl``). Changing anything here changes the prompt and so
invalidates the 100,826 teacher targets: treat it as a new dataset, not an edit.
"""
