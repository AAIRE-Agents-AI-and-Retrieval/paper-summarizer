"""Each student's assistant turn is read from its own chat template, not hardcoded.

Qwen3 wants the answer straight after the generation prompt. K2-Horizon *always* opens a reasoning
block in its generation prompt and raises if an assistant message has no thinking field, so the
target has to close that block before the JSON. Getting this wrong is silent: training would still
run, and the model would learn to answer inside a block it never closes.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scisum.prompt import assistant_wrapper

REPO = Path(__file__).resolve().parent.parent
CASES = {"qwen3-4b-instruct-2507": ("", "<|im_end|>"),
         "k2-horizon-7b": ("</ifm|think>", "<|ifm|im_end|>")}


def _tokenizer(profile: str):
    from transformers import AutoTokenizer

    env_file = REPO / "slurm" / "models" / f"{profile}.env"
    if not env_file.exists():
        pytest.skip(f"no model profile {profile}")
    out = subprocess.run(["bash", "-c", f"source {env_file}; env | grep ^SCISUM"],
                         capture_output=True, text=True, cwd=REPO).stdout.strip().splitlines()
    env = dict(line.split("=", 1) for line in out)
    try:
        return AutoTokenizer.from_pretrained(
            env["SCISUM_BASE_MODEL"], revision=env.get("SCISUM_BASE_MODEL_REVISION") or None,
            trust_remote_code=env.get("SCISUM_TRUST_REMOTE_CODE") == "true")
    except Exception as e:  # noqa: BLE001 - the weights may not be downloaded on this machine
        pytest.skip(f"tokenizer for {profile} unavailable: {type(e).__name__}")


@pytest.mark.slow
@pytest.mark.parametrize("profile,expected", CASES.items())
def test_wrapper_matches_the_template(profile, expected):
    assert assistant_wrapper(_tokenizer(profile)) == expected


@pytest.mark.slow
@pytest.mark.parametrize("profile", CASES)
def test_prompt_and_target_join_into_the_template_s_own_rendering(profile):
    """prompt + prefix + content + suffix must be exactly what the template renders for a full turn."""
    tok = _tokenizer(profile)
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    prefix, suffix = assistant_wrapper(tok)
    ours = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + prefix + "X" + suffix
    theirs = tok.apply_chat_template(msgs + [{"role": "assistant", "content": "X", "think": ""}],
                                     tokenize=False).rstrip("\n")
    assert ours == theirs
