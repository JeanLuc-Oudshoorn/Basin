"""Decode one raw source table into the arrays the next-patch transform stores.

Kept free of torch so worker processes can run it; the transform applies the
same checks, in the same order, to every table whether it is decoded in the
main process or in a worker.
"""

import hashlib
import io
import re
import signal

import numpy as np
import pandas as pd

_settings = None


def column_roles(names, data):
    """Split a table's columns into lag-derived exclusions and model inputs."""
    excluded = [
        c
        for c in names
        if any(re.search(pattern, c) for pattern in data["excluded_column_patterns"])
    ]
    past = data["past_covariates"]
    if past == "auto":
        past = [
            c
            for c in names
            if c not in data["targets"] + [data["time_column"]] + excluded
        ]
    return excluded, past, data["targets"] + list(past)


def decode_table(task, settings):
    """Read (table index, file bytes); stop at the first reason to reject it.

    The result always has "rows". A usable table also carries its content hash,
    column names and float32 target/covariate values for the whole table; any
    other outcome is named by "problem" for the caller to report.
    """
    index, content = task
    data = settings["data"]
    frame = pd.read_parquet(io.BytesIO(content))
    result = {"rows": len(frame)}
    if len(frame) < settings["minimum_rows"]:
        result["problem"] = "short"
        return result
    time = frame[data["time_column"]]
    if time.isna().any() or not time.is_monotonic_increasing or time.duplicated().any():
        result["problem"] = "time"
        return result
    names = list(frame)
    excluded, past, columns = column_roles(names, data)
    if set(columns) & set(excluded):
        result["problem"] = "roles"
        return result
    if np.isinf(frame[columns].to_numpy(dtype=np.float32)).any():
        result["problem"] = "infinity"
        return result
    result["sha256"] = hashlib.sha256(content).hexdigest()
    result["names"] = names
    result["targets"] = frame[data["targets"]].to_numpy(dtype=np.float32)
    if past:
        result["past"] = frame[list(past)].to_numpy(dtype=np.float32)
    if settings["export_directory"] is not None:
        frame[[data["time_column"]] + columns].to_parquet(
            f"{settings['export_directory']}/table_{index:03d}.parquet",
            index=False,
            compression="zstd",
        )
    return result


def start_worker(settings):
    """Pool initializer: keep the settings and leave Ctrl+C to the parent."""
    global _settings
    _settings = settings
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def decode_in_worker(task):
    return decode_table(task, _settings)
