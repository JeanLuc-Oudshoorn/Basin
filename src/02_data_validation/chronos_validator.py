"""Covariate-aware Chronos-2 screening with bounded, independent forecast batches."""

import json
import re

import numpy as np
import pyarrow.parquet as pq
import torch
from bolt_validator import BoltValidator


def chronos_admitted(results, tolerance=0.15):
    """Enforce the relative error bound even when historical MASE is undefined."""
    baseline, score = results.baseline_mae, results.chronos2_mae
    return (
        results.status.eq("ok")
        & np.isfinite(baseline)
        & np.isfinite(score)
        & baseline.ge(0)
        & score.ge(0)
        & score.le((1 + tolerance) * baseline)
    )


class ChronosValidator(BoltValidator):
    engine = "chronos2"
    label = "Chronos-2 Small"
    admission_metric = "mae"
    admitted = staticmethod(chronos_admitted)

    def _roles(self, columns):
        data = self.config["data"]
        excluded = {
            column
            for column in columns
            if any(
                re.search(pattern, column)
                for pattern in data["excluded_column_patterns"]
            )
        }
        known = list(data["known_covariates"])
        past = data["past_covariates"]
        if past == "auto":
            past = [
                column
                for column in columns
                if column not in data["targets"] + known + [data["time_column"]]
                and column not in excluded
            ]
        past = list(past)
        inputs = data["targets"] + past + known
        if len(inputs) != len(set(inputs)) or data["time_column"] in inputs:
            raise ValueError("Target and covariate roles must be disjoint and unique")
        if set(inputs) & excluded:
            raise ValueError("Explicit input roles include excluded lag columns")
        if set(inputs) - set(columns):
            raise ValueError("Configured target/covariate columns are missing")
        return past, known

    def _read_frame(self, path):
        data = self.config["data"]
        with pq.ParquetFile(path) as parquet:
            past, known = self._roles(parquet.schema_arrow.names)
            return parquet.read(
                columns=[data["time_column"]] + data["targets"] + past + known
            ).to_pandas()

    def _model_input(self, frame, history, split, horizon, record):
        past, known = self._roles(frame.columns)
        start = split - len(history)
        covariates = {}
        future = {}
        for column in past + known:
            values = frame[column].iloc[start:split].to_numpy(dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError(
                    f"Historical covariate {column!r} contains missing/infinite values"
                )
            covariates[column] = values
        for column in known:
            values = (
                frame[column].iloc[split : split + horizon].to_numpy(dtype=np.float32)
            )
            if not np.isfinite(values).all():
                raise ValueError(
                    f"Known future covariate {column!r} contains missing/infinite values"
                )
            future[column] = values
        record.update(
            past_covariate_columns=json.dumps(past),
            known_covariate_columns=json.dumps(known),
            covariate_count=len(covariates),
        )
        inputs = {"target": np.asarray(history, dtype=np.float32)}
        if covariates:
            inputs["past_covariates"] = covariates
        if future:
            inputs["future_covariates"] = future
        return inputs

    def _group_key(self, item):
        inputs = item[2]
        return (
            len(item[3]),
            len(inputs.get("past_covariates", {})),
            len(inputs.get("future_covariates", {})),
        )

    @staticmethod
    def _canonical_input(inputs):
        # Names are lookup keys, not model features. Normalize within each table
        # so different generated names don't fragment compatible batches.
        historical = inputs.get("past_covariates", {})
        future = inputs.get("future_covariates", {})
        names = {
            column: f"past_{index}"
            for index, column in enumerate(sorted(set(historical) - set(future)))
        }
        names.update(
            {column: f"known_{index}" for index, column in enumerate(sorted(future))}
        )
        result = {"target": inputs["target"]}
        if historical:
            result["past_covariates"] = {
                names[column]: values for column, values in historical.items()
            }
        if future:
            result["future_covariates"] = {
                names[column]: values for column, values in future.items()
            }
        return result

    def _predict(self, inputs, horizon):
        quantiles, _ = self.pipeline.predict_quantiles(
            [self._canonical_input(item) for item in inputs],
            prediction_length=horizon,
            quantile_levels=[0.5],
            batch_size=self.batch_size,
            context_length=self.context_length,
            cross_learning=False,
        )
        # Chronos-2 returns one (target, horizon, quantile) tensor per table.
        return np.stack(
            [torch.as_tensor(value)[0].float().cpu().numpy() for value in quantiles]
        )
