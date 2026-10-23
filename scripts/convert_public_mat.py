"""Convert MATLAB datasets in data/public into Parquet tables in data/public_cleaned."""

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

PUBLIC_DIR = Path(__file__).resolve().parents[1] / "data" / "public"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "public_cleaned"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", value).strip("_") or "table"


def _vector(value):
    """Return a 1-D array for tabular numeric/string data, else None."""
    if isinstance(value, (str, bytes)) or value is None:
        return None
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if array.ndim == 0 or array.ndim > 2 or array.dtype.kind not in "biufSU":
        return None
    if array.ndim == 2 and 1 not in array.shape:
        return None
    return array.reshape(-1)


def extract_tables(mat_path: Path) -> dict[str, pd.DataFrame]:
    """Extract aligned vectors from MATLAB structs into named sample tables.

    Nested structs become separate tables when they contain independent
    measurements. Scalar fields from enclosing structs are copied as metadata;
    arrays of MATLAB structs receive a 1-based ``record_index`` column.
    """
    variables = loadmat(mat_path, simplify_cells=True)
    collected = defaultdict(list)

    def visit(node, path, metadata):
        if isinstance(node, dict):
            local_metadata = dict(metadata)
            vector_fields = {}
            children = {}
            for key, value in node.items():
                name = _safe_name(str(key))
                vector = _vector(value)
                if vector is not None and len(vector):
                    vector_fields[name] = vector
                elif isinstance(value, (int, float, np.number, str, bytes)):
                    local_metadata[name] = (
                        value.decode(errors="replace")
                        if isinstance(value, bytes)
                        else value
                    )
                elif isinstance(value, dict) or isinstance(value, list):
                    children[name] = value

            lengths = {len(values) for values in vector_fields.values()}
            if len(lengths) == 1:
                frame = pd.DataFrame(vector_fields)
                for name, value in local_metadata.items():
                    frame[name] = value
                collected[path].append(frame)
            elif vector_fields:
                # Unequal vectors are independent signals; keep them as their
                # own tables instead of truncating or padding their samples.
                for name, values in vector_fields.items():
                    frame = pd.DataFrame({name: values})
                    for metadata_name, value in local_metadata.items():
                        frame[metadata_name] = value
                    collected[f"{path}/{name}"].append(frame)

            for name, value in children.items():
                visit(value, f"{path}/{name}", local_metadata)

        elif isinstance(node, list):
            if node and all(isinstance(item, dict) for item in node):
                for index, item in enumerate(node, start=1):
                    visit(item, path, {**metadata, "record_index": index})
            elif node:
                array = _vector(node)
                if array is not None:
                    frame = pd.DataFrame({"value": array})
                    for name, value in metadata.items():
                        frame[name] = value
                    collected[path].append(frame)

    for variable, value in variables.items():
        if not variable.startswith("__"):
            visit(value, _safe_name(variable), {})

    tables = {}
    for name, frames in collected.items():
        table = pd.concat(frames, ignore_index=True, sort=False)
        table.insert(
            0,
            "sample_index",
            table.groupby(
                [column for column in ("record_index",) if column in table],
                dropna=False,
            ).cumcount()
            if "record_index" in table
            else np.arange(len(table)),
        )
        tables["/".join(_safe_name(part) for part in name.split("/"))] = table
    return tables


def convert_dataset(mat_path: Path) -> list[Path]:
    tables = extract_tables(mat_path)
    dataset_dir = OUTPUT_DIR / mat_path.stem
    dataset_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, frame in tables.items():
        # Preserve nested MATLAB group names as subdirectories.
        parts = name.split("/")
        target = dataset_dir.joinpath(*parts[:-1], f"{parts[-1]}.parquet")
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False)
        written.append(target)
    return written


def convert_all() -> list[Path]:
    files = sorted(PUBLIC_DIR.glob("*.mat"))
    if not files:
        raise FileNotFoundError(f"No .mat files found in {PUBLIC_DIR}")
    outputs = []
    for mat_path in files:
        written = convert_dataset(mat_path)
        outputs.extend(written)
        print(f"{mat_path.name}: wrote {len(written)} Parquet table(s)")
    return outputs


converted_files = convert_all()
print(f"Converted {len(converted_files)} tables to {OUTPUT_DIR}")
