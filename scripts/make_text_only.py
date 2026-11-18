"""Write a text-only copy of a vision-language checkpoint, for vLLM to serve.

``Qwen/Qwen3.5-9B`` ships as ``Qwen3_5ForConditionalGeneration``: 426 language-model tensors under
``model.language_model.``, 333 vision-tower tensors under ``model.visual.``, 15 multi-token-prediction
tensors under ``mtp.`` and ``lm_head.weight``. Training loads it with ``AutoModelForCausalLM``, which
builds the text-only ``Qwen3_5ForCausalLM`` from the language-model tensors alone. vLLM has the same
class, but asked to build it from the hub checkpoint it refuses the vision tensors
(``There is no module or parameter named 'visual'``), and served as the VL class it leaves a
text-only adapter matching nothing (paper_docs/09, 2026-10-19).

This keeps exactly what training loaded - ``model.language_model.*`` and ``lm_head.weight``, names
unchanged, since vLLM's text class strips that prefix itself - and declares the text-only class.
Tensors are copied byte for byte; PROVENANCE.json records the source revision and every count.

    srun -p general -c 8 --mem=48G -t 30 .venv/bin/python scripts/make_text_only.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

from scisum.config import get_settings

KEEP = ("model.language_model.", "lm_head.")
COPY = ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
        "chat_template.jinja", "generation_config.json", "LICENSE")


def main() -> None:
    s = get_settings()
    src = Path(snapshot_download(s.base_model, revision=s.base_model_revision, local_files_only=True))
    rev = (s.base_model_revision or src.name)[:7]
    out = Path(s.serve_model or f".cache/text-only/{s.base_model.replace('/', '--')}@{rev}")
    tmp = out.with_name(out.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)

    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted(set(index.values()))
    weight_map, kept, dropped, total = {}, 0, {}, 0
    for shard in shards:
        tensors = {}
        with safe_open(src / shard, "pt") as f:
            for k in f.keys():  # noqa: SIM118 - a safe_open handle is not a dict and not iterable
                if k.startswith(KEEP):
                    tensors[k] = f.get_tensor(k)
                else:
                    head = k.split(".")[0] if not k.startswith("model.") else ".".join(k.split(".")[:2])
                    dropped[head] = dropped.get(head, 0) + 1
        if tensors:
            save_file(tensors, tmp / shard, metadata={"format": "pt"})
            weight_map.update({k: shard for k in tensors})
            kept += len(tensors)
            total += sum(t.numel() * t.element_size() for t in tensors.values())
        print(f"{shard}: kept {len(tensors)}", flush=True)

    (tmp / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2))
    for name in COPY:
        if (src / name).exists():
            shutil.copy2(src / name, tmp / name)
    cfg = json.loads((tmp / "config.json").read_text())
    declared = cfg.get("architectures")
    cfg["architectures"] = ["Qwen3_5ForCausalLM"]
    # Qwen3.5-4B ties its output head to the embeddings, so there is no lm_head.weight to keep. The served
    # copy must then say it is tied (at the top level and in text_config), or the server would build an
    # untrained head. Untested on Qwen3.5-4B: check the first generations of the base model for sense.
    tied = cfg.get("tie_word_embeddings", (cfg.get("text_config") or {}).get("tie_word_embeddings"))
    if not any(k.startswith("lm_head.") for k in weight_map):
        if not tied:
            raise SystemExit("no lm_head.weight kept and the config does not tie the embeddings")
        cfg["tie_word_embeddings"] = True
        if isinstance(cfg.get("text_config"), dict):
            cfg["text_config"]["tie_word_embeddings"] = True
    (tmp / "config.json").write_text(json.dumps(cfg, indent=2))

    prov = {"source": s.base_model, "revision": s.base_model_revision or src.name,
            "declared_architectures": declared, "served_architectures": cfg["architectures"],
            "tied_embeddings": bool(tied), "kept_tensors": kept, "kept_bytes": total, "dropped": dropped,
            "shards": {sh: hashlib.sha256((tmp / sh).read_bytes()).hexdigest()
                       for sh in sorted(set(weight_map.values()))}}
    (tmp / "PROVENANCE.json").write_text(json.dumps(prov, indent=2))
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)                     # all or nothing: a half-written copy is never served
    print(json.dumps({k: v for k, v in prov.items() if k != "shards"}, indent=2))
    print(f"written: {out}")


if __name__ == "__main__":
    main()
