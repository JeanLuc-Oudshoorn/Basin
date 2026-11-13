"""Compare both frozen trained models on the largest 20% MASE contributors."""

# ruff: noqa: E402
# isort: skip_file
import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for stage in ("03_transforming_input_data", "05_model_training", "06_holdout_test"):
    sys.path.insert(0, str(ROOT / "src" / stage))
sys.path.insert(0, str(ROOT))
from train import RWKVForecaster
import numpy as np
import pandas as pd
import torch
from inference import forecast_contexts
from performance import configure_cuda_memory
from variate_attention import configure_variate_execution
from transform import load_config, run_context, write_json
from investigate_forecast_regression import OUT


@torch.inference_mode()
def main():
    config = load_config()
    torch.set_num_threads(4)
    configure_cuda_memory(config["training"])
    cases = pd.read_parquet(OUT / "twenty_percent_worst_origins.parquet")
    rows = []
    with run_context(config, "regression_outliers"):
        for name in ("model", "447m_model"):
            model = RWKVForecaster.load(
                ROOT / "results" / name / "training/model.pt"
            ).eval()
            configure_variate_execution(model, config["training"], ROOT)
            model.fast_inference = None
            for case in cases.itertuples():
                raw = pd.read_parquet(ROOT / case.source)
                history = raw.iloc[max(0, case.origin - 2048) : case.origin]
                covariates = [c for c in raw if c not in ("outcome", "time_index")]
                context = dict(
                    context=torch.tensor(
                        history[["outcome"]].to_numpy(dtype=np.float32).T.copy()
                    )[None],
                    past_covariates=torch.tensor(
                        history[covariates].to_numpy(dtype=np.float32).T.copy()
                    )[None]
                    if covariates
                    else None,
                )
                q = list(forecast_contexts(model, [context], 1))[0].numpy()
                actual = raw.outcome.iloc[case.origin : case.origin + 16].to_numpy()
                predicted = q[0, : len(actual), 4]
                denominator = float(
                    np.nanmean(np.abs(np.diff(history.outcome.to_numpy())))
                )
                rows.append(
                    dict(
                        model=name,
                        source=case.source,
                        origin=case.origin,
                        points=len(actual),
                        mase=float(np.abs(actual - predicted).mean() / denominator),
                        mae=float(np.abs(actual - predicted).mean()),
                        scale=denominator,
                        saved_307_mase=case.rwkv,
                        actual=actual.tolist(),
                        prediction=predicted.tolist(),
                    )
                )
            del model
            gc.collect()
            torch.cuda.empty_cache()
        write_json(OUT / "outlier_checkpoint_comparison.json", rows)


if __name__ == "__main__":
    main()
