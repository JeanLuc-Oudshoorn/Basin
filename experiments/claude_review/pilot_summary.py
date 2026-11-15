"""Summarise the short pilot trainings under results/pilot_v2 (CPU only, read-only).

Each pilot trains one model variant for one pass over the same small corpus,
with the same seed, order and schedule. The table compares their validation and
holdout scores. A few thousand updates say how quickly a variant starts to
learn, not where it ends up after a full run.

    python experiments/claude_review/pilot_summary.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PILOTS = ROOT / "results/pilot_v2"


def load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main():
    rows, competitors = [], {}
    for config_path in sorted(PILOTS.glob("*.json")):
        name = config_path.stem
        history = load(PILOTS / name / "training/validation_history.json")
        summary = load(PILOTS / name / "training/early_stopping.json")
        if not history or not summary:
            continue
        final = history[-1]
        row = {
            "pilot": name,
            "parameters_m": round(
                load(PILOTS / name / "training/architecture.json")["total_parameters"]
                / 1e6,
                1,
            ),
            "updates": summary["completed_updates"],
            "training_minutes": round(summary["active_training_seconds"] / 60, 1),
            "checks": [(r["step"], round(r["trimmed_table_loss"], 4)) for r in history],
            "final_step": final["step"],
            "trimmed_table_loss": round(final["trimmed_table_loss"], 4),
            "median_table_loss": round(final["median_table_loss"], 4),
            "median_table_mase": round(final["median_table_mase"], 3),
            "table_loss_p99": round(final["table_loss_p99"], 3),
        }
        metrics = load(PILOTS / name / "holdout_test/test_metrics.json")
        if metrics:
            board = {r["model"]: r for r in metrics["leaderboard"]}
            row["holdout_table_mase_median"] = round(
                board["rwkv"]["table_mase_median"], 3
            )
            row["holdout_median_ase"] = round(
                board["rwkv"]["median_absolute_scaled_error"], 3
            )
            row["holdout_pooled_mase"] = round(board["rwkv"]["mase"], 2)
            row["holdout_median_ase_by_token"] = [
                round(v["rwkv"]["median_absolute_scaled_error"], 2)
                for v in metrics["by_forecast_token"].values()
            ]
            for other in ("naive", "chronos", "timestral"):
                if other in board:
                    competitors[other] = {
                        "holdout_table_mase_median": round(
                            board[other]["table_mase_median"], 3
                        ),
                        "holdout_median_ase": round(
                            board[other]["median_absolute_scaled_error"], 3
                        ),
                        "holdout_pooled_mase": round(board[other]["mase"], 2),
                        "holdout_median_ase_by_token": [
                            round(v[other]["median_absolute_scaled_error"], 2)
                            for v in metrics["by_forecast_token"].values()
                        ],
                    }
        rows.append(row)
    if not rows:
        sys.exit("No completed pilots under results/pilot_v2")
    for row in rows:
        print(json.dumps(row))
    for name, row in competitors.items():
        print(json.dumps({"reference": name, **row}))
    (PILOTS / "summary.json").write_text(
        json.dumps({"pilots": rows, "references": competitors}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
