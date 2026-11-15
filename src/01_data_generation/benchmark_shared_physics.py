"""Compare Parquet-only generation with a pre-change source snapshot.

Set REFERENCE_DIRECTORY to a directory containing reference/ and options.json.
The snapshot must be taken before editing. No production generate() call runs.
"""

import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

from benchmark_generation_exact import assert_exact, reference_class, runner_options
from scm_generator import OperationalSCMGenerator

REFERENCE_DIRECTORY = None
TABLES = 128
REPEATS = 5
ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "results/data_generation/shared_physics_benchmark.json"


def main():
    reference = REFERENCE_DIRECTORY
    if reference is None:
        reference = Path(
            (ROOT / "results/data_generation/shared_reference.txt").read_text(
                encoding="utf-8"
            )
        )
    reference = Path(reference)
    before = reference_class(reference / "reference")
    old = json.loads((reference / "options.json").read_text(encoding="utf-8"))
    new = runner_options()
    for options in (old, new):
        options.update(
            save_scm=False, save_plot=False, save_dag=False, generation_workers=1
        )
    # Replay invariance is separate from the intentionally changed sampling prior.
    original = before("unused", **old)
    candidate = OperationalSCMGenerator("unused", **new)
    for index in range(32):
        model = original.sample_model(trend_free=bool(index % 2))
        assert_exact(original.simulate(model), candidate.simulate(model))
    seconds = {"before": [], "after": []}
    counts = {"before": [], "after": []}
    attempts = {"before": [], "after": []}
    with tempfile.TemporaryDirectory() as folder:
        for repeat in range(REPEATS):
            batches = [("before", before, old), ("after", OperationalSCMGenerator, new)]
            for name, cls, options in batches[:: 1 if repeat % 2 == 0 else -1]:
                output = Path(folder) / f"{name}_{repeat}"
                output.mkdir()
                options = dict(options, seed=81 + repeat)
                generator = cls(output, **options)
                count, tried = 0, 0
                start = time.perf_counter()
                for i in range(TABLES):
                    model, frame = generator._sample_accepted_model(bool(i % 2))
                    directory = output / f"table_{i}"
                    generator._write_table(model, frame, directory)
                    count += 1 + len(
                        generator._write_extra_copies(model, frame, directory)
                    )
                    tried += model["quality"]["sampling_attempts"]
                elapsed = time.perf_counter() - start
                seconds[name].append(elapsed)
                counts[name].append(count)
                attempts[name].append(tried)
                print(name, repeat, round(elapsed, 3), count, flush=True)
    medians = {key: statistics.median(value) for key, value in seconds.items()}
    per_table = {
        key: statistics.median([s / n for s, n in zip(seconds[key], counts[key])])
        for key in seconds
    }
    report = dict(
        originals_per_batch=TABLES,
        repeats=REPEATS,
        seconds=seconds,
        saved_tables=counts,
        attempts=attempts,
        median_seconds=medians,
        median_seconds_per_output=per_table,
        time_change_percent=100 * (medians["after"] / medians["before"] - 1),
        per_output_change_percent=100 * (per_table["after"] / per_table["before"] - 1),
        old_models_replayed_exactly=32,
        python=sys.version,
        note="Serial end-to-end sampling, existing admission checks and Parquet writes including copies. No plots/SCM. Sampling distribution intentionally changed; no new rejection rules.",
    )
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
