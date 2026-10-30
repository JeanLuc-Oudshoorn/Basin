"""CPU-only counts of distinct observed patches under the normal training split."""

import argparse
import hashlib
import json
import sqlite3
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from training_protocol import sampled_origins, split_datasets

SPLITS = {"training": 1, "validation": 2, "test": 4}
KINDS = (
    "input_contents",
    "input_source_patches",
    "target_contents",
    "target_source_patches",
)


def parameter_budget(config):
    """Analytic count for the vendored x070 blocks plus the forecasting layers.

    No model allocation or CUDA import. Training also checks the measured count.
    """
    m = config["model"]
    d, p, layers = m["width"], m["patch_size"], m["layers"]
    output = p * len(m["quantiles"])
    ranks = [max(32, round(factor * d**0.5 / 32) * 32) for factor in (2.5, 2.5, 1.7, 5)]
    groups = {
        "embedding": 5 * (3 * p) * d + 4 * d * d + 6 * d,
        "roles": 3 * d,
        "rwkv_blocks": layers * (12 * d * d + 19 * d + 2 * d * sum(ranks)) + 2 * d,
        "variate_mixing": layers * (4 * d * d + 6 * d),
        "norm": 2 * d,
        "head": 4 * d * d + 5 * d * output + 4 * d + 2 * output,
    }
    return {"parameters": sum(groups.values()), "method": "analytic", "groups": groups}


class _Index:
    """Disk-backed digest sets: memory stays bounded as the corpus grows."""

    def __init__(self, path, patch_size):
        self.db = sqlite3.connect(path)
        # Disposable scratch data, never a source of recovery or model state.
        # Avoid a durable journal/fsync and bound the page cache to 64 MiB.
        self.db.execute("PRAGMA journal_mode=OFF")
        self.db.execute("PRAGMA synchronous=OFF")
        self.db.execute("PRAGMA cache_size=-65536")
        self.db.execute(
            "CREATE TABLE tokens (kind INTEGER, digest BLOB, splits INTEGER, "
            "PRIMARY KEY (kind, digest)) WITHOUT ROWID"
        )
        self.patch_size = patch_size
        self.pending = {}
        self.counts = {
            name: dict.fromkeys(
                (
                    "windows",
                    "input_token_occurrences",
                    "target_input_occurrences",
                    "covariate_input_occurrences",
                    "prediction_target_occurrences",
                    "observed_target_points",
                    "empty_input_patches_excluded",
                    "partial_input_occurrences",
                ),
                0,
            )
            for name in SPLITS
        }

    def flush(self):
        self.db.executemany(
            "INSERT INTO tokens VALUES (?, ?, ?) ON CONFLICT(kind, digest) "
            "DO UPDATE SET splits = tokens.splits | excluded.splits",
            [(kind, digest, mask) for (kind, digest), mask in self.pending.items()],
        )
        self.db.commit()
        self.pending.clear()

    def add(
        self,
        split,
        values,
        role,
        source,
        variable,
        start,
        stop,
        *,
        target=False,
        repeats=1,
    ):
        values = np.asarray(values, dtype="<f4")
        if values.shape != (self.patch_size,) or np.isinf(values).any():
            raise ValueError(
                "Inventory patches must have patch_size entries and no infinity"
            )
        valid = np.isfinite(values)
        if not valid.any():
            if not target:
                self.counts[split]["empty_input_patches_excluded"] += repeats
            return
        # Canonicalize missing entries and signed zero before hashing float32 data.
        clean = np.where(valid, values, 0).astype("<f4")
        clean[clean == 0] = 0
        content = hashlib.sha256(
            bytes([role]) + valid.tobytes() + clean.tobytes()
        ).digest()
        position = hashlib.sha256(
            json.dumps([source, variable, start, stop], ensure_ascii=True).encode()
        ).digest()
        offset = 2 if target else 0
        for kind, digest in ((offset, content), (offset + 1, position)):
            key = kind, digest
            self.pending[key] = self.pending.get(key, 0) | SPLITS[split]
        c = self.counts[split]
        if target:
            c["prediction_target_occurrences"] += repeats
            c["observed_target_points"] += int(valid.sum()) * repeats
        else:
            c["input_token_occurrences"] += repeats
            c[
                "target_input_occurrences"
                if role == 0
                else "covariate_input_occurrences"
            ] += repeats
            c["partial_input_occurrences"] += int(not valid.all()) * repeats
        if len(self.pending) >= 10000:
            self.flush()

    def summarize(self):
        self.flush()
        union, overlaps = {}, {}
        for kind, label in enumerate(KINDS):
            union[label] = self.db.execute(
                "SELECT count(*) FROM tokens WHERE kind = ?", (kind,)
            ).fetchone()[0]
            for split, bit in SPLITS.items():
                count = self.db.execute(
                    "SELECT count(*) FROM tokens WHERE kind = ? AND splits & ? != 0",
                    (kind, bit),
                ).fetchone()[0]
                self.counts[split][f"unique_{label}"] = count
            for first, second in combinations(SPLITS, 2):
                both = SPLITS[first] | SPLITS[second]
                pair = overlaps.setdefault(f"{first}__{second}", {})
                pair[label] = self.db.execute(
                    "SELECT count(*) FROM tokens WHERE kind = ? AND splits & ? = ?",
                    (kind, both, both),
                ).fetchone()[0]
        for split in ("validation", "test"):
            self.counts[split]["input_contents_not_in_training"] = (
                self.counts[split]["unique_input_contents"]
                - overlaps[f"training__{split}"]["input_contents"]
            )
        return self.counts, union, overlaps


def _numpy(tensor):
    return tensor.detach().cpu().float().numpy()


def _add_record(index, record, split, patch):
    batch = record["batch"]
    history = _numpy(batch["context"])
    targets = history.shape[1]
    if "past_covariates" in batch:
        history = np.concatenate([history, _numpy(batch["past_covariates"])], axis=1)
    if history.shape[0] != 1 or history.shape[1] != len(record["variable_names"]):
        raise ValueError(
            "Inventory requires one source per record and matching variable names"
        )
    length = history.shape[-1]
    left = (-length) % patch
    patches = np.pad(history[0], ((0, 0), (left, 0)), constant_values=np.nan).reshape(
        history.shape[1], -1, patch
    )
    source = record.get("source_key", record["source"])
    origin = record["context_rows"][0] - left
    train = split == "training"
    n_inputs = patches.shape[1] - int(train)
    # Cached legacy features can be checked, but lazy records must not tokenize
    # an entire window merely to count its raw patches.
    if "prepared" in record and record["prepared"]["tokens"] != n_inputs:
        raise ValueError("Patch count differs from saved features; rerun embedding")
    for variable, name in enumerate(record["variable_names"]):
        for t in range(n_inputs):
            index.add(
                split,
                patches[variable, t],
                int(variable >= targets),
                source,
                name,
                origin + t * patch,
                origin + (t + 1) * patch,
            )
        if train and variable < targets:
            for t in range(1, patches.shape[1]):
                index.add(
                    split,
                    patches[variable, t],
                    0,
                    source,
                    name,
                    origin + t * patch,
                    origin + (t + 1) * patch,
                    target=True,
                )
    if not train:
        labels = _numpy(batch["future_target"])[0]
        for variable, name in enumerate(record["targets"]):
            padded = np.pad(
                labels[variable], (0, patch - labels.shape[-1]), constant_values=np.nan
            )
            index.add(
                split,
                padded,
                0,
                source,
                name,
                record["cutoff"],
                record["cutoff"] + patch,
                target=True,
            )
    index.counts[split]["windows"] += 1


def _add_test(index, record, config):
    # Count the same sampled origins as final holdout evaluation.
    from transform import ROOT, file_digest

    path = ROOT / record["source"]
    if file_digest(path) != record["source_sha256"]:
        raise ValueError(f"Holdout source changed since transformation: {path}")
    frame = pd.read_parquet(path)
    patch = config["model"]["patch_size"]
    cap = min(config["model"]["context_length"], config["data"]["context_length"])
    origins = sampled_origins(
        len(frame),
        patch,
        config.get("holdout", {}).get("percentiles", [1, 25, 50, 75, 99]),
    )
    last = origins[-1] // patch
    targets = record["targets"]
    columns = targets + record["roles"]["past"]
    values = frame[columns].to_numpy(dtype=np.float32).T
    source = record.get("source_key", record["source"])
    for variable, name in enumerate(columns):
        for t in range(last):
            # Count repeated context use analytically, hashing each source patch once.
            repeats = sum(
                max(0, origin - cap) <= t * patch and (t + 1) * patch <= origin
                for origin in origins
            )
            if repeats:
                index.add(
                    "test",
                    values[variable, t * patch : (t + 1) * patch],
                    int(variable >= len(targets)),
                    source,
                    name,
                    t * patch,
                    (t + 1) * patch,
                    repeats=repeats,
                )
        if variable < len(targets):
            for start in origins:
                raw = values[variable, start : start + patch]
                padded = np.pad(raw, (0, patch - len(raw)), constant_values=np.nan)
                index.add(
                    "test", padded, 0, source, name, start, start + patch, target=True
                )
    index.counts["test"]["windows"] += len(origins)


def build_inventory(data, config, *, groups=None, directory=None):
    """Count content identities separately from distinct source/variable positions."""
    if config["model"].get("objective") != "next_patch":
        raise ValueError(
            "Token inventory currently supports the next_patch training protocol"
        )
    if groups is None:
        groups = split_datasets(data["records"], config)
    training, validation, test, split_audit = groups
    with tempfile.TemporaryDirectory(prefix="token_inventory_", dir=directory) as temp:
        index = _Index(Path(temp) / "counts.sqlite", config["model"]["patch_size"])
        try:
            for split, records in (("training", training), ("validation", validation)):
                for i, record in enumerate(records, 1):
                    _add_record(index, record, split, config["model"]["patch_size"])
                    if i % 1000 == 0:
                        logger.info(
                            "Token inventory: {} {}/{} windows", split, i, len(records)
                        )
            for i, record in enumerate(test, 1):
                _add_test(index, record, config)
                if i % 100 == 0:
                    logger.info("Token inventory: test {}/{} tables", i, len(test))
            counts, union, overlaps = index.summarize()
        finally:
            index.db.close()
    for name, records in (
        ("training", training),
        ("validation", validation),
        ("test", test),
    ):
        counts[name]["tables"] = len({r["source"] for r in records})
    report = {
        "schema": 2,
        "holdout_percentiles": config.get("holdout", {}).get(
            "percentiles", [1, 25, 50, 75, 99]
        ),
        "artifact_signature": data["signature"],
        "processed_sha256": data["processed_sha256"],
        "patch_size": config["model"]["patch_size"],
        "embedding_width": config["model"]["width"],
        "model": parameter_budget(config),
        "splits": counts,
        "union_unique": union,
        "overlap_unique": overlaps,
        "dataset_split": split_audit,
        "definitions": {
            "unit": "One patch of one variable; embedding dimensions are not separate tokens. Unique contents refer to raw observed patches, not equality of continuous learned embedding vectors.",
            "content_identity": "SHA-256 of role, observation mask, and raw float32 values; missing values and signed zero canonicalized. Independent of source, normalization, position and learned weights.",
            "source_identity": "Source key, variable name and patch row interval, including alignment.",
            "exclusions": "Kernel padding, entirely unobserved patches, held-out-table training records and repeated epochs do not create unique input tokens.",
            "targets": "Prediction labels counted separately from input patches; test includes the real partial tail.",
            "test": "Sampled dataset-disjoint origins, with the same capped expanding actual history as final testing.",
            "overlap": "Training/validation input overlap is expected shared history; content overlap across different sources can also occur naturally.",
            "interpretation": "Exact content counts measure duplication, not statistical independence or guaranteed data sufficiency. Similar noisy patches can all be distinct. Ratios are descriptive; no pass/fail threshold is imposed.",
        },
    }
    update_parameter_ratios(report)
    return report


def update_parameter_ratios(report, measured_parameters=None):
    if measured_parameters is not None:
        report["model"]["analytic_parameters"] = report["model"]["parameters"]
        report["model"].update(parameters=measured_parameters, method="measured")
    parameters = report["model"]["parameters"]
    for counts in report["splits"].values():
        counts["unique_input_contents_per_parameter"] = (
            counts["unique_input_contents"] / parameters
        )
        counts["unique_target_contents_per_parameter"] = (
            counts["unique_target_contents"] / parameters
        )


def save_inventory(report, directory):
    from transform import write_json

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "token_inventory.json", report)
    pd.DataFrame.from_dict(report["splits"], orient="index").rename_axis(
        "split"
    ).to_csv(directory / "token_inventory.csv")
    lines = [
        f"Patch size: {report['patch_size']}; embedding width: {report['embedding_width']}; "
        f"model parameters: {report['model']['parameters']:,} ({report['model']['method']})",
        "",
        "Split       Tables   Input occurrences   Unique contents   Unique source patches   Unique target contents   Unique inputs/parameter",
    ]
    for name, c in report["splits"].items():
        lines.append(
            f"{name:<11} {c['tables']:>6,} {c['input_token_occurrences']:>19,} "
            f"{c['unique_input_contents']:>17,} {c['unique_input_source_patches']:>23,} "
            f"{c['unique_target_contents']:>24,} {c['unique_input_contents_per_parameter']:>25.6f}"
        )
    lines += ["", *report["definitions"].values()]
    text = "\n".join(lines) + "\n"
    (directory / "token_inventory.txt").write_text(text, encoding="utf-8")
    logger.info("Pre-training token inventory:\n{}", text)


def write_token_inventory(data, config, *, groups=None):
    from transform import location

    directory = location(config, "training")
    directory.mkdir(parents=True, exist_ok=True)
    report = build_inventory(data, config, groups=groups, directory=directory)
    save_inventory(report, directory)
    return report


def prepare_token_inventory(data, config, *, groups):
    """Reuse a matching exact audit; never launch a corpus scan by default."""
    from transform import location, write_json

    options = config["training"].get("token_inventory", {"mode": "cached"})
    if not isinstance(options, dict) or set(options) - {"mode"}:
        raise ValueError("training.token_inventory must contain only mode")
    mode = options.get("mode", "cached")
    if mode not in ("cached", "exact", "off"):
        raise ValueError("token_inventory.mode must be cached, exact, or off")
    directory = location(config, "training")
    path = directory / "token_inventory.json"
    report = None
    reason = "disabled" if mode == "off" else "no matching cached inventory"
    if mode == "exact":
        logger.warning(
            "Explicit exact token inventory: scanning the entire corpus before training"
        )
        report = write_token_inventory(data, config, groups=groups)
        reason = "exact inventory computed"
    elif mode == "cached" and path.is_file():
        try:
            candidate = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(candidate, dict):
                raise ValueError("Inventory cache must be an object")
            matches = (
                candidate.get("schema") == 2
                and candidate.get("holdout_percentiles")
                == config.get("holdout", {}).get("percentiles", [1, 25, 50, 75, 99])
                and candidate.get("artifact_signature") == data["signature"]
                and candidate.get("processed_sha256") == data["processed_sha256"]
                and candidate.get("patch_size") == config["model"]["patch_size"]
                and candidate.get("embedding_width") == config["model"]["width"]
                and candidate.get("dataset_split") == groups[3]
                and set(candidate.get("splits", {})) == set(SPLITS)
            )
            if matches:
                update_parameter_ratios(candidate)
                report = candidate
                reason = "matching exact inventory reused"
        except (OSError, ValueError, KeyError, TypeError, ArithmeticError):
            reason = "invalid cached report; exact counting deferred"
    status = {
        "mode": mode,
        "status": "available" if report is not None else "not_computed",
        "reason": reason,
        "report": str(path) if report is not None else None,
        "artifact_signature": data["signature"],
        "processed_sha256": data["processed_sha256"],
        "training_windows": len(groups[0]),
        "validation_tables": len(groups[1]),
        "test_tables": len(groups[2]),
        "corpus_scan_performed": mode == "exact",
        "exact_inventory_command": "uv run python src/05_model_training/token_inventory.py --config configs/config.json",
    }
    write_json(directory / "token_inventory_status.json", status)
    logger.info(
        "Token inventory: {}; mode={}; corpus_scan={}",
        reason,
        mode,
        status["corpus_scan_performed"],
    )
    if report is None and mode != "off":
        logger.info(
            "Training proceeds without exact unique-token counts. Optional audit: {}",
            status["exact_inventory_command"],
        )
    return report


def main():
    # These local imports keep this module usable from the embedding/training APIs.
    import sys

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src/03_transforming_input_data"))
    from transform import file_digest, load_config, location, read_artifact

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(root / "configs/config.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    data = read_artifact(location(config, "embedded") / "embeddings.pt", config)
    if data["processed_sha256"] != file_digest(
        location(config, "processed") / "windows.pt"
    ):
        raise ValueError(
            "Processed data changed; rerun embedding before counting tokens"
        )
    write_token_inventory(data, config)


if __name__ == "__main__":
    main()
