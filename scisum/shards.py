"""Tokenised examples on disk: one flat token file per shard, plus a fixed-width index.

Format (per shard ``part-NNN``)::

    part-NNN.bin        uint32 token ids, examples concatenated in training order
    part-NNN.idx.npy    int64 [n, 3]: offset into .bin, prompt token count, total token count
    part-NNN.meta.jsonl one line per example: id, teacher, cohort, flags, lengths

Why not a single file, Arrow or ``datasets``: training reads the examples strictly in order, so a
memory-mapped flat array is the fastest possible loader (no decode, no copy, no index lookup beyond
one row), shards can be written by many processes in parallel, and resuming is a seek to a known
example number. The index is the only thing that must be read eagerly, and it is 24 bytes per
example.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TOKEN_DTYPE = np.uint32
IDX_DTYPE = np.int64


class ShardWriter:
    """Writes one shard. ``close()`` is required; a shard without its index is ignored by readers."""

    def __init__(self, directory: Path, part: int):
        self.dir = directory
        self.part = part
        self.dir.mkdir(parents=True, exist_ok=True)
        self.stem = self.dir / f"part-{part:04d}"
        self.bin_path = Path(f"{self.stem}.bin")
        self.idx_path = Path(f"{self.stem}.idx.npy")
        self.meta_path = Path(f"{self.stem}.meta.jsonl")
        self._tmp_bin = Path(f"{self.stem}.bin.tmp")
        self._tmp_meta = Path(f"{self.stem}.meta.jsonl.tmp")
        self._tmp_idx = Path(f"{self.stem}.idx.tmp.npy")
        self._bin = self._tmp_bin.open("wb")
        self._meta = self._tmp_meta.open("w")
        self._rows: list[tuple[int, int, int]] = []
        self._offset = 0

    def add(self, prompt_ids: list[int], target_ids: list[int], meta: dict) -> None:
        ids = np.asarray(prompt_ids + target_ids, dtype=TOKEN_DTYPE)
        self._bin.write(ids.tobytes())
        self._rows.append((self._offset, len(prompt_ids), len(ids)))
        self._offset += len(ids)
        self._meta.write(json.dumps(meta, ensure_ascii=False) + "\n")

    def close(self) -> int:
        self._bin.close()
        self._meta.close()
        np.save(self._tmp_idx, np.asarray(self._rows, dtype=IDX_DTYPE).reshape(-1, 3))
        # the index is renamed last: a reader only ever sees a complete shard
        self._tmp_bin.replace(self.bin_path)
        self._tmp_meta.replace(self.meta_path)
        self._tmp_idx.replace(self.idx_path)
        return len(self._rows)


@dataclass(slots=True)
class Example:
    """One training example. ``tokens[:prompt_len]`` is the prompt, the rest is the target + EOS."""

    index: int
    tokens: np.ndarray
    prompt_len: int

    @property
    def target_len(self) -> int:
        return int(len(self.tokens) - self.prompt_len)


class Shard:
    def __init__(self, stem: Path):
        self.stem = Path(stem)
        self.idx = np.load(Path(f"{stem}.idx.npy"))
        self._mm: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.idx.shape[0])

    @property
    def tokens(self) -> np.ndarray:
        if self._mm is None:  # mapped on first use, so forking workers do not inherit the mapping
            self._mm = np.memmap(Path(f"{self.stem}.bin"), dtype=TOKEN_DTYPE, mode="r")
        return self._mm

    def get(self, i: int) -> tuple[np.ndarray, int]:
        off, plen, total = (int(x) for x in self.idx[i])
        return self.tokens[off:off + total], plen

    def meta(self) -> list[dict]:
        with Path(f"{self.stem}.meta.jsonl").open() as f:
            return [json.loads(line) for line in f]


class ShardSet:
    """The shards of one split, in order, addressed by a single global example number."""

    def __init__(self, directory: Path):
        self.dir = Path(directory)
        stems = sorted({p.with_suffix("").with_suffix("") for p in self.dir.glob("part-*.idx.npy")})
        self.shards = [Shard(s) for s in stems]
        self.counts = [len(s) for s in self.shards]
        self.starts = np.cumsum([0] + self.counts)
        self.total = int(self.starts[-1])
        lens = [s.idx[:, 2] for s in self.shards]
        self.lengths = np.concatenate(lens) if lens else np.zeros(0, dtype=IDX_DTYPE)
        plens = [s.idx[:, 1] for s in self.shards]
        self.prompt_lengths = np.concatenate(plens) if plens else np.zeros(0, dtype=IDX_DTYPE)

    def __len__(self) -> int:
        return self.total

    def locate(self, i: int) -> tuple[int, int]:
        s = int(np.searchsorted(self.starts, i, side="right") - 1)
        return s, i - int(self.starts[s])

    def get(self, i: int) -> Example:
        s, j = self.locate(i)
        tokens, plen = self.shards[s].get(j)
        return Example(index=i, tokens=np.asarray(tokens), prompt_len=plen)

    def iter_from(self, start: int):
        """Examples ``start``, ``start+1``, ... in order (one sequential pass over the token files)."""
        for i in range(start, self.total):
            yield self.get(i)

    def meta(self) -> list[dict]:
        out: list[dict] = []
        for s in self.shards:
            out.extend(s.meta())
        return out

    def target_tokens(self) -> np.ndarray:
        return self.lengths - self.prompt_lengths
