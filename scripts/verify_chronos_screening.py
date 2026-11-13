"""Run the pinned screening model on a small real-data sample, without admission writes."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for stage in ("02_data_validation", "03_transforming_input_data"):
    sys.path.insert(0, str(ROOT / "src" / stage))


def main():
    from chronos_validator import ChronosValidator
    from transform import load_config, location, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tables", type=int, default=4)
    parser.add_argument("--output", default="results/chronos2-screening-smoke.json")
    args = parser.parse_args()
    config = load_config()
    manifest = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    paths = [location(config, "raw") / key for key in manifest["paths"][: args.tables]]
    validator = ChronosValidator(config)
    started = perf_counter()
    results = [record for _, record in validator.evaluate_tables(paths)]
    elapsed = perf_counter() - started
    if len(results) != args.tables or any(
        record["status"] != "ok" for record in results
    ):
        raise RuntimeError(f"Screening smoke failed: {results}")
    write_json(
        ROOT / args.output,
        {
            "elapsed_seconds": elapsed,
            "model_metadata": validator.metadata,
            "results": results,
        },
    )
    print(
        json.dumps(
            {
                "tables": len(results),
                "elapsed_seconds": elapsed,
                "covariate_counts": [row["covariate_count"] for row in results],
                "model": validator.metadata,
            }
        )
    )


if __name__ == "__main__":
    main()
