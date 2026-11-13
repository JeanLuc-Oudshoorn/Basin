"""Keep training temporary files and compiler/driver caches on the project drive."""

import os
import tempfile
from pathlib import Path


def configure_runtime_storage(root):
    root = Path(root).resolve()
    cache = root / ".cache"
    platform = "windows" if os.name == "nt" else "wsl"
    paths = {
        "TEMP": cache / "tmp" / platform,
        "TMP": cache / "tmp" / platform,
        "TMPDIR": cache / "tmp" / platform,
        "TORCHINDUCTOR_CACHE_DIR": cache / "torchinductor",
        "TRITON_CACHE_DIR": cache / "torchinductor" / "triton",
        "TORCH_COMPILE_DEBUG_DIR": cache / "compile_debug",
        "TORCH_HOME": cache / "torch",
        "XDG_CACHE_HOME": cache / "xdg",
        "CUDA_CACHE_PATH": cache / "cuda_driver",
    }
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    tempfile.tempdir = str(paths["TMPDIR"])
    os.environ.setdefault("CUDA_CACHE_MAXSIZE", str(1024**3))
    crash = cache / "crash_dumps"
    crash.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_COREDUMP_FILE"] = str(crash / "cuda-%p-%t.nvcudmp")
    return {name: str(path) for name, path in paths.items()}
