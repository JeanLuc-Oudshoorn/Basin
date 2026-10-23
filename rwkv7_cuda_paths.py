from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def prepend_env_path(name: str, path: Path) -> None:
    path_value = str(path)
    current = os.environ.get(name)
    if not current:
        os.environ[name] = path_value
        return

    entries = current.split(os.pathsep)
    if path_value not in entries:
        os.environ[name] = os.pathsep.join([path_value, *entries])


def ninja_safe_overlay_root() -> Path:
    override = os.environ.get("MAMBA_CPP_BUILD_OVERLAY_DIR")
    candidates = [Path(override)] if override else []
    if os.name == "posix":
        candidates.append(Path.home() / ".cache" / "mamba_cpp_build_overlays")
        # Existing fallback root; link helpers below reject conflicting targets.
        candidates.append(Path("/tmp"))  # nosec B108
    else:
        candidates.append(PROJECT_ROOT / ".cache" / "cpp_build_overlays")

    for candidate in candidates:
        if any(character.isspace() for character in str(candidate)):
            continue
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        return candidate

    raise RuntimeError(
        "Could not create a no-space overlay directory for PyTorch/Ninja builds."
    )


def ninja_safe_build_path(path: Path, prefix: str, description: str) -> Path:
    path = path.resolve()
    if os.name != "posix" or not any(character.isspace() for character in str(path)):
        return path

    digest = hashlib.sha1(str(path).encode("utf-8"), usedforsecurity=False).hexdigest()[
        :12
    ]
    link_path = ninja_safe_overlay_root() / f"{prefix}-{digest}"
    if link_path.exists() or link_path.is_symlink():
        if link_path.resolve() == path:
            return link_path
        raise RuntimeError(
            f"Cannot use {description}={path} because the path contains spaces "
            f"and {link_path} already points somewhere else."
        )

    try:
        link_path.symlink_to(path, target_is_directory=True)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot use {description}={path} because the path contains spaces "
            "and creating a no-space symlink for PyTorch/Ninja failed. Move the "
            "project to a path without spaces."
        ) from exc
    return link_path


def ninja_safe_cuda_home(cuda_home: Path) -> Path:
    cuda_home = cuda_home.resolve()
    if os.name != "posix":
        return cuda_home

    has_spaces = any(character.isspace() for character in str(cuda_home))
    needs_cudart_alias = not (cuda_home / "lib" / "libcudart.so").exists() and any(
        (cuda_home / "lib").glob("libcudart.so.*")
    )
    if not has_spaces and not needs_cudart_alias:
        return cuda_home

    digest = hashlib.sha1(
        str(cuda_home).encode("utf-8"), usedforsecurity=False
    ).hexdigest()[:12]
    overlay_path = ninja_safe_overlay_root() / f"mamba-cuda-build-{digest}"
    overlay_path.mkdir(parents=True, exist_ok=True)

    for name in ("bin", "include", "nvvm"):
        source = cuda_home / name
        if source.exists():
            ensure_symlink(overlay_path / name, source)

    source_lib = cuda_home / "lib"
    overlay_lib = overlay_path / "lib"
    overlay_lib.mkdir(exist_ok=True)
    for source in source_lib.iterdir():
        ensure_symlink(overlay_lib / source.name, source)
    ensure_versioned_library_alias(overlay_lib, "libcudart.so")
    return overlay_path


def ensure_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        if link_path.resolve() == target_path.resolve():
            return
        raise RuntimeError(
            f"Cannot use {link_path} for the RWKV CUDA build because it already "
            "points somewhere else."
        )
    link_path.symlink_to(target_path, target_is_directory=target_path.is_dir())


def ensure_versioned_library_alias(lib_dir: Path, unversioned_name: str) -> None:
    unversioned_path = lib_dir / unversioned_name
    if unversioned_path.exists() or unversioned_path.is_symlink():
        return

    matches = sorted(lib_dir.glob(f"{unversioned_name}.*"))
    if not matches:
        return
    unversioned_path.symlink_to(matches[-1])


def ninja_safe_torch_lib_path(torch_lib_path: Path) -> Path:
    return ninja_safe_build_path(
        torch_lib_path,
        "mamba-torch-lib",
        "PyTorch library path",
    )


def require_cuda_cccl_headers(cuda_home: Path) -> None:
    if (cuda_home / "include" / "nv" / "target").exists():
        return

    raise RuntimeError(
        "CUDA CCCL headers are missing from the CUDA toolkit used for the RWKV "
        f"extension build: {cuda_home / 'include' / 'nv' / 'target'} was not "
        "found. Install a CUDA CCCL package matching the WSL venv's "
        "nvidia-cuda-nvcc version."
    )


def configure_torch_cpp_extension_paths(cuda_home: Path) -> None:
    import torch.utils.cpp_extension as cpp_extension

    torch_lib_path = ninja_safe_torch_lib_path(Path(cpp_extension.TORCH_LIB_PATH))
    cpp_extension.CUDA_HOME = str(cuda_home)
    cpp_extension.TORCH_LIB_PATH = str(torch_lib_path)
    prepend_env_path("LD_LIBRARY_PATH", torch_lib_path)


def configure_cuda_home() -> None:
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home and (Path(cuda_home) / "bin" / "nvcc").exists():
        cuda_home_path = ninja_safe_cuda_home(Path(cuda_home))
        require_cuda_cccl_headers(cuda_home_path)
        os.environ["CUDA_HOME"] = str(cuda_home_path)
        prepend_env_path("PATH", cuda_home_path / "bin")
        prepend_env_path("LD_LIBRARY_PATH", cuda_home_path / "lib")
        configure_torch_cpp_extension_paths(cuda_home_path)
        return

    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        Path(sys.prefix) / "lib" / python_version / "site-packages" / "nvidia" / "cu13",
        PROJECT_ROOT
        / ".venv-wsl"
        / "lib"
        / python_version
        / "site-packages"
        / "nvidia"
        / "cu13",
    ]
    for candidate in candidates:
        if (candidate / "bin" / "nvcc").exists():
            cuda_home_path = ninja_safe_cuda_home(candidate)
            require_cuda_cccl_headers(cuda_home_path)
            os.environ["CUDA_HOME"] = str(cuda_home_path)
            prepend_env_path("PATH", cuda_home_path / "bin")
            prepend_env_path("LD_LIBRARY_PATH", cuda_home_path / "lib")
            configure_torch_cpp_extension_paths(cuda_home_path)
            return

    raise RuntimeError(
        "Could not find nvcc for the RWKV CUDA extension build. Install a CUDA "
        "toolkit in the WSL venv or set CUDA_HOME to a toolkit directory that "
        "contains bin/nvcc."
    )
