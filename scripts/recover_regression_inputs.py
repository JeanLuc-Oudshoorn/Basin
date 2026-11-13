"""Recover missing old holdout inputs from archived processed tensors only."""

import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/audits/2026-11-02/forecast_regression"


def main():
    torch.set_num_threads(1)
    sources = set(
        json.loads(
            (ROOT / "results/447m_model/holdout_test/sample_selection.json").read_text()
        )["selected_sources"]
    )
    data = torch.load(
        ROOT / "data/02_processed/447m_model/windows.pt",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    recovered = {}
    for record in data["records"]:
        if record["split"] != "evaluation" or record["source"] not in sources:
            continue
        batch = record["batch"]
        if record["context_rows"][0] != 0:
            raise ValueError("Archive does not contain full historical prefix")
        target = torch.cat((batch["context"], batch["future_target"]), dim=-1)[
            0
        ].clone()
        columns = list(record["targets"])
        values = [target]
        if "past_covariates" in batch:
            values.append(
                torch.nn.functional.pad(
                    batch["past_covariates"][0], (0, 16), value=float("nan")
                )
            )
            columns += record["roles"]["past"]
        recovered[record["source"]] = dict(
            columns=columns,
            values=torch.cat(values).clone(),
            cutoff=record["cutoff"],
            source_sha256=record["source_sha256"],
        )
    if set(recovered) != sources:
        raise ValueError("Archive lacks selected sources")
    torch.save(recovered, OUT / "recovered_old_inputs.pt")
    print(
        "Recovered",
        len(recovered),
        "old tables; unavailable covariate tail is never used as history",
        flush=True,
    )


if __name__ == "__main__":
    main()
