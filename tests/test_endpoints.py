"""Grader endpoint resolution (scisum.endpoints): env table, server files, the shared endpoint, the API key."""

import json
import os
import time

import pytest

from scisum.config import Settings
from scisum.endpoints import resolve


@pytest.fixture
def s(tmp_path, monkeypatch):
    for k in ("SCISUM_GRADER_ENDPOINTS", "SCISUM_JUDGE_BASE", "JUDGE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    return Settings(repo_root=tmp_path, judge_base="", judge_api_key="")


def test_nothing_configured_says_what_to_set(s):
    with pytest.raises(RuntimeError, match="SCISUM_GRADER_ENDPOINTS"):
        resolve(s, "gemma4-31b-it")


def test_env_table_inline_and_file_win_over_everything(s, tmp_path, monkeypatch):
    s = s.model_copy(update={"judge_base": "http://grader.example/v1"})
    monkeypatch.setenv("SCISUM_GRADER_ENDPOINTS", json.dumps({"gemma4-31b-it": "http://g:8000/v1/"}))
    ep = resolve(s, "gemma4-31b-it")
    assert (ep.base_url, ep.source, ep.api_key) == ("http://g:8000/v1", "env", "EMPTY")
    assert resolve(s, "qwen38-27b").source == "judge_base"            # not in the table: falls through
    f = tmp_path / "eps.json"
    f.write_text(json.dumps({"qwen38-27b": {"base_url": "http://q:8001/v1", "api_key_env": "QKEY"}}))
    monkeypatch.setenv("SCISUM_GRADER_ENDPOINTS", str(f))
    monkeypatch.setenv("QKEY", "k-123")
    ep = resolve(s, "qwen38-27b")
    assert (ep.base_url, ep.api_key) == ("http://q:8001/v1", "k-123")


def test_server_files_newest_wins_per_model(s):
    d = s.state_dir / "graders"
    d.mkdir(parents=True)
    old, new = d / "gemma-1.json", d / "gemma-2.json"
    old.write_text(json.dumps({"model": "gemma4-31b-it", "base_url": "http://old:8000/v1"}))
    new.write_text(json.dumps({"model": "gemma4-31b-it", "base_url": "http://new:8000/v1"}))
    t = time.time()
    os.utime(old, (t - 100, t - 100))
    (d / "broken.json").write_text("{not json")
    (d / "qwen.json").write_text(json.dumps({"model": "qwen38-27b", "base_url": "http://q:8000/v1"}))
    ep = resolve(s, "gemma4-31b-it")
    assert ep.base_url == "http://new:8000/v1" and ep.source == "state/graders/gemma-2.json"
    assert resolve(s, "qwen38-27b").base_url == "http://q:8000/v1"


def test_shared_endpoint_and_key_as_before(s):
    s = s.model_copy(update={"judge_base": "https://endpoint.invalid/v1", "judge_api_key": "secret"})
    ep = resolve(s, "anything")
    assert (ep.base_url, ep.api_key, ep.source) == ("https://endpoint.invalid/v1", "secret", "judge_base")
    assert "secret" not in repr(ep.source)


def test_judge_client_uses_the_models_endpoint(s, monkeypatch):
    from scisum.factcheck import Chat

    monkeypatch.setenv("SCISUM_GRADER_ENDPOINTS", json.dumps({"m1": "http://one:1/v1", "m2": "http://two:2/v1"}))
    a, b = Chat(s, 2, "m1", "sys", 10), Chat(s, 2, "m2", "sys", 10)
    assert str(a.client.base_url).startswith("http://one:1/v1") and str(b.client.base_url).startswith("http://two:2/v1")
    assert a._body("hi", 5)["model"] == "m1"
