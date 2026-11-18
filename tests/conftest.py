import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".cache" / "hf"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# A Python environment that also has another checkout of `scisum` installed in editable mode (the original
# project's) resolves any `scisum.<module>` missing here from that checkout, silently. Tests must see only this
# repository's code, so drop such finders before anything is imported.
sys.meta_path[:] = [f for f in sys.meta_path if not getattr(f, "__module__", "").startswith("__editable___scisum")]
