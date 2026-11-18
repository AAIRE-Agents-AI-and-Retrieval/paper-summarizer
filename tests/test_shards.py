"""The token-shard format: write, read back, address by a global example number."""

from __future__ import annotations

import numpy as np

from scisum.shards import ShardSet, ShardWriter


def _write(directory, part, examples):
    w = ShardWriter(directory, part)
    for prompt, target in examples:
        w.add(prompt, target, {"id": f"p{part}-{len(prompt)}"})
    return w.close()


def test_roundtrip_and_global_addressing(tmp_path):
    a = [([1, 2, 3], [4, 5]), ([6, 7], [8])]
    b = [([9] * 5, [10, 11, 12])]
    assert _write(tmp_path, 0, a) == 2
    assert _write(tmp_path, 1, b) == 1

    ss = ShardSet(tmp_path)
    assert len(ss) == 3
    assert list(ss.lengths) == [5, 3, 8]
    assert list(ss.prompt_lengths) == [3, 2, 5]
    assert list(ss.target_tokens()) == [2, 1, 3]

    ex = ss.get(0)
    assert list(ex.tokens) == [1, 2, 3, 4, 5] and ex.prompt_len == 3 and ex.target_len == 2
    ex = ss.get(2)  # second shard, first example
    assert list(ex.tokens) == [9, 9, 9, 9, 9, 10, 11, 12] and ex.prompt_len == 5
    assert [e.index for e in ss.iter_from(1)] == [1, 2]
    assert len(ss.meta()) == 3


def test_incomplete_shard_is_ignored(tmp_path):
    _write(tmp_path, 0, [([1], [2])])
    w = ShardWriter(tmp_path, 1)  # never closed: no index on disk
    w.add([3], [4], {})
    assert len(ShardSet(tmp_path)) == 1


def test_dtype_is_compact(tmp_path):
    _write(tmp_path, 0, [([151935], [1])])
    ss = ShardSet(tmp_path)
    assert ss.get(0).tokens.dtype == np.uint32
    assert int(ss.get(0).tokens[0]) == 151935
