"""Stage 11: forecasting and anomaly detection on public operational datasets.

Runs Basin, Naive and the holdout stage's foundation models on the cases
`prepare_cases.py` cut from CATS and from the operational series of the UCR anomaly
archive, and scores two tasks on each dataset: forecasting normal data, and
detecting anomalies the model has not seen the start of. Nothing here touches
training data or the holdout.
"""
# ruff: noqa: E402

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT / "src" / "09_fakebench", Path(__file__).resolve().parent):
    sys.path.insert(0, str(folder))
import fakebench
from anomaly_examples import write_examples
from fakebench_ts import models as fakebench_models
from fakebench_ts.dataset import QUANTILE_LEVELS
from operational_scoring import leaderboard, score_model, summarise
from performance import configure_cuda_memory
from prepare_cases import (
    RECORD,
    case_settings,
    data_directory,
    load_cases,
    settings,
)
from train import launch_wsl
from transform import load_config, run_context, timed, write_json

COMPLETED = "results.json"  # written last, so it marks a finished run


def results_root(config):
    return ROOT / config["paths"].get(
        "operational_benchmarks", "results/model/operational_benchmarks"
    )


def build(name, config, batch_size):
    """A forecaster that returns quantiles: stage 9's models, Basin with its quantiles."""
    if name == "basin":
        return fakebench.basin(
            config, batch_size=batch_size, quantiles=True, covariates=True
        )
    return fakebench.build(name, config)


def forecast_cases(forecaster, cases, horizon, batch_size, progress=None):
    """Quantiles [cases, levels, horizon] in the order of `cases`.

    A model call holds contexts of one length and one number of covariates, so
    no model has to pad. A case's covariates go to the models that take past
    covariates (`uses_covariates`); the others get the target's history alone.
    """
    quantiles = np.empty((len(cases), len(QUANTILE_LEVELS), horizon), dtype=np.float32)
    layouts = list(zip(cases["context_length"], cases["covariates"].map(len)))
    takes_covariates = getattr(forecaster, "uses_covariates", False)
    done = 0
    for length, count in sorted(set(layouts)):
        positions = np.flatnonzero([layout == (length, count) for layout in layouts])
        for start in range(0, len(positions), batch_size):
            members = positions[start : start + batch_size]
            contexts = [
                np.asarray(cases["values"].iloc[i][:length], dtype=np.float32)
                for i in members
            ]
            if count and takes_covariates:
                past = np.stack(
                    [
                        np.asarray(
                            cases["covariate_values"].iloc[i], dtype=np.float32
                        ).reshape(count, length)
                        for i in members
                    ]
                )
                result = forecaster(
                    contexts, horizon, past_covariates=past, known_covariates=None
                )
            else:
                result = forecaster(contexts, horizon)
            result = np.asarray(result, dtype=np.float32)
            if result.shape != (len(members), len(QUANTILE_LEVELS), horizon):
                raise ValueError(
                    "The anomaly task needs quantile forecasts of shape "
                    f"[contexts, {len(QUANTILE_LEVELS)}, {horizon}]; the model "
                    f"returned {result.shape}"
                )
            if not np.isfinite(result).all():
                raise ValueError("The model returned a forecast that is not finite")
            quantiles[members] = result
            done += len(members)
            if progress:
                progress(done, len(cases))
    return quantiles


def score_run(output, cases, options, record):
    """Score the forecasts saved in `output` and write the result files."""
    scored = {}
    for path in sorted((output / "forecasts").glob("*.npz")):
        saved = np.load(path)
        if not np.array_equal(saved["case"], cases["case"].to_numpy()):
            raise ValueError(f"{path} holds forecasts of other cases")
        scored[path.stem] = score_model(
            cases, saved["quantiles"], saved["levels"], options
        )
    # The order of the request, Basin first.
    order = [m for m in record.get("models", {}) if m in scored]
    scored = {m: scored[m] for m in [*order, *(m for m in scored if m not in order)]}
    results = summarise(cases, scored, options)
    details = cases.drop(columns=["values", "covariate_values"])
    for model, table in scored.items():
        details = details.merge(
            table.rename(columns=lambda c: c if c == "case" else f"{model}_{c}"),
            on="case",
        )
    details.to_parquet(output / "per_case.parquet")
    write_examples(output / "examples.html", cases, output / "forecasts", options)
    facts = {
        "Basin": record.get("basin", {}),
        "Cases": f"{len(cases):,} ({record.get('prepared', '')})",
        "Inputs": "; ".join(f"{m}: {v}" for m, v in record.get("models", {}).items()),
    }
    (output / "leaderboard.md").write_text(
        leaderboard(results, options, facts), encoding="utf-8"
    )
    write_json(output / COMPLETED, {"run": record, "settings": options, **results})
    return results


def prepared_cases(config, options):
    directory = data_directory(config)
    if not (directory / RECORD).is_file():
        raise FileNotFoundError(
            f"No prepared cases in {directory}. Make them with: "
            "python src/11_operational_benchmarks/prepare_cases.py"
        )
    cases, prepared = load_cases(directory)
    if case_settings(prepared["settings"]) != case_settings(options):
        raise ValueError(
            f"The cases in {directory} were prepared with other settings than "
            "operational_benchmarks in the configuration now has. Prepare again: "
            "python src/11_operational_benchmarks/prepare_cases.py"
        )
    return cases, prepared


def rescore(config, name):
    """Score the saved forecasts of a finished run again (no model is run)."""
    options = settings(config)
    output = results_root(config) / name
    record = json.loads((output / "run.json").read_text(encoding="utf-8"))
    cases, _ = prepared_cases(config, options)
    saved = np.load(next((output / "forecasts").glob("*.npz")))["case"]
    cases = cases[cases["case"].isin(saved)].reset_index(drop=True)
    score_run(output, cases, options, record)
    logger.success("Scored again: {}", output / "leaderboard.md")


def run(config, args):
    options = settings(config)
    torch.set_num_threads(config["training"].get("cpu_threads", 4))
    if torch.cuda.is_available():
        configure_cuda_memory(config["training"])
    cases, prepared = prepared_cases(config, options)
    if args.limit and args.limit < len(cases):
        keep = np.unique(np.linspace(0, len(cases) - 1, args.limit).round().astype(int))
        cases = cases.iloc[keep].reset_index(drop=True)
    # Basin first: its checkpoint names the run folder.
    names = sorted(args.models or options["models"], key=lambda name: name != "basin")
    batch_size = args.batch_size or options["batch_size"]
    record = {
        "prepared": f"{prepared['created_utc']}, sha256 {prepared['cases_sha256'][:12]}",
        "cases": len(cases),
        "models": {},
    }
    loaded, cuda = {}, torch.cuda.is_available()
    if "basin" in names:
        with timed("load basin", cuda=True):
            loaded["basin"] = build("basin", config, batch_size)
        record["basin"] = loaded["basin"].details
    output = fakebench.new_run(
        config,
        epoch=record.get("basin", {}).get("epoch"),
        step=record.get("basin", {}).get("step"),
        root=results_root(config),
    )
    logger.info("OPERATIONAL BENCHMARKS {} cases; output={}", len(cases), output)
    (output / "forecasts").mkdir()
    for name in names:
        if name not in loaded:
            with timed(f"load {name}", cuda=cuda):
                loaded[name] = build(name, config, batch_size)
        forecaster = loaded.pop(name)
        with timed(f"forecast {name}", cuda=cuda):
            quantiles = forecast_cases(
                forecaster, cases, options["horizon"], batch_size
            )
        np.savez_compressed(
            output / "forecasts" / f"{name}.npz",
            case=cases["case"].to_numpy(),
            levels=np.array(QUANTILE_LEVELS),
            quantiles=quantiles,
        )
        record["models"][name] = (
            # TimeH2O's adapter accepts past covariates and does not use them.
            "target history and the case's past covariates"
            if getattr(forecaster, "uses_covariates", False) and name != "timeh2o"
            else "target history only"
        )
        del forecaster
        if cuda:
            torch.cuda.empty_cache()
    write_json(output / "run.json", record)
    score_run(output, cases, options, record)
    logger.success("Operational benchmarks complete: {}", output / "leaderboard.md")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/config.json"))
    parser.add_argument(
        "--models",
        nargs="+",
        help="default: operational_benchmarks.models; FakeBench-TS baselines "
        "that give quantiles (naive, mean, median) also work",
    )
    parser.add_argument("--batch-size", type=int, help="contexts per model call")
    parser.add_argument("--limit", type=int, help="run an even subset of cases (trial)")
    parser.add_argument(
        "--rescore",
        metavar="RUN",
        help="score the saved forecasts of this run folder again; runs no model",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.rescore:
        rescore(config, args.rescore)
        return
    names = args.models or settings(config)["models"]
    extra = ["--models", *args.models] if args.models else []
    extra += ["--batch-size", str(args.batch_size)] if args.batch_size else []
    extra += ["--limit", str(args.limit)] if args.limit else []
    with run_context(config, "operational_benchmarks"):
        if set(names) <= set(fakebench_models.BASELINES):
            # Baselines need no GPU: run where the command was given.
            with timed("OPERATIONAL BENCHMARKS stage"):
                run(config, args)
            return
        if launch_wsl(__file__, args.config, extra_args=extra):
            return
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Use the configured WSL CUDA bf16 runtime")
        with timed("OPERATIONAL BENCHMARKS stage", cuda=True):
            run(config, args)


if __name__ == "__main__":
    main()
