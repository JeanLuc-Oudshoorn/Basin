"""Extract the exact setup recipe from docs; run with WSL Python 3.12."""

import ast
import hashlib
import os
import re
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source_root = Path(os.environ.get("MAMBA_SOURCE", str(root.parent / "Mamba")))
guide = (root / "docs/README_RWKV7_FAST_SETUP.md").read_text()
source = source_root / "third_party/rwkv-lm/RWKV-v7/train_temp"
destination = root / "third_party/rwkv-lm/RWKV-v7/train_temp"
expected = "63ca38c31bd9a09f1cd5107a0b7d7213c2e697e1b8b5f6cfa0e0519d53b61a1b"
if hashlib.sha256((source / "src/model.py").read_bytes()).hexdigest() != expected:
    raise AssertionError("RWKV source does not match the expected SHA-256 digest")
if destination.exists():
    raise SystemExit(f"Refusing to overwrite {destination}")
shutil.copytree(
    source,
    destination,
    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.so", "*.o", ".git"),
)
shutil.copy2(
    source_root / "third_party/rwkv-lm/LICENSE", root / "third_party/rwkv-lm/LICENSE"
)
os.chdir(root)
os.environ["MAMBA_SOURCE"] = str(source_root)
blocks = re.findall(r"```(\w+)\n(.*?)```", guide, re.S)
for language, block in blocks:
    if language == "bash" and block.startswith("python - <<'PY'\n"):
        snippet = block.split("\n", 1)[1].rsplit("\nPY", 1)[0]
        if (
            "Expected the local patched RWKV source" in snippet
            or "import ast" in snippet
        ):
            # The checked-in setup guide is trusted executable project source.
            exec(compile(snippet, "<setup-guide>", "exec"))  # nosec B102
python_blocks = [block for language, block in blocks if language == "python"]
if len(python_blocks) != 2:
    raise AssertionError("Expected two Python blocks in the setup guide")
for name, block in zip(
    ("rwkv7_backend.py", "smoke_rwkv7_timeseries.py"), python_blocks
):
    ast.parse(block)
    (root / name).write_text(block)
print(
    "Copied the verified patched source and extracted the guide's helpers, loader, and smoke test."
)
