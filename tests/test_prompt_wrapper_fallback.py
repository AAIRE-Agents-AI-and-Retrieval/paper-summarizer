"""``assistant_wrapper`` on templates whose generation prompt is not an exact prefix of the rendered conversation.

Fake tokenizers stand in for real chat templates, so no download is needed: an exact-prefix template (Qwen-style),
one whose generation prompt ends in whitespace the full rendering does not repeat (the fallback), and one whose
generation prompt diverges with real text (must raise rather than guess).
"""

from __future__ import annotations

import pytest

from scisum.prompt import assistant_wrapper


class FakeTemplate:
    def __init__(self, gen_tail: str, turn: str):
        self.gen_tail, self.turn = gen_tail, turn

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False) -> str:
        head = "".join(f"<|start|>{m['role']}\n{m['content']}<|end|>\n" for m in messages if m["role"] != "assistant")
        if add_generation_prompt:
            return head + "<|start|>assistant" + self.gen_tail
        answer = next(m["content"] for m in messages if m["role"] == "assistant")
        return head + "<|start|>assistant" + self.turn.format(answer)


def test_exact_prefix_template():
    tok = FakeTemplate(gen_tail="\n", turn="\n{}<|end|>\n")
    assert assistant_wrapper(tok) == ("", "<|end|>")


def test_generation_prompt_with_trailing_whitespace_falls_back_to_the_common_prefix():
    # The generation prompt ends "assistant\n\n", the rendered turn "assistant\n<answer>": not a prefix. The
    # shared prefix ends at "assistant\n"; the generation prompt only adds whitespace beyond it.
    tok = FakeTemplate(gen_tail="\n\n", turn="\n{}<|end|>\n")
    assert assistant_wrapper(tok) == ("", "<|end|>")


def test_divergent_generation_prompt_raises():
    tok = FakeTemplate(gen_tail="<think>", turn="\n{}<|end|>\n")
    with pytest.raises(RuntimeError, match="cannot read the assistant turn"):
        assistant_wrapper(tok)
