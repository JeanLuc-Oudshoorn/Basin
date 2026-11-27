"""How many proposed tables does REJECT_CONSTANT_OUTCOME throw away? CPU only, writes nothing."""

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src/01_data_generation"))
import generate_operational_data as g  # noqa: E402
import scm_generator as s  # noqa: E402

captured = {}


class Capture:
    def __init__(self, *args, **kwargs):
        captured.update(kwargs)

    def generate(self, count):
        return [Path("a/b")]


g.OperationalSCMGenerator = Capture
g.main()
captured.update(
    reject_constant_outcome=False,
    seed=20261121,
    generation_workers=1,
    save_plot=False,
    save_dag=False,
    save_scm=False,
)
gen = s.OperationalSCMGenerator(tempfile.mkdtemp(), **captured)
count = int(sys.argv[1])
rng = np.random.default_rng(1)
constant = total = 0
ranges = []
start = time.time()
for index in range(count):
    trend_free = bool(rng.random() < gen.trend_free_fraction)
    model = gen.sample_model(trend_free=trend_free)
    clean = gen._simulate_process(model)
    y = s.recorded_window(clean, model.get("recorded_rows"))["outcome"].to_numpy()
    relative = np.ptp(y) / max(1.0, abs(float(np.median(y))))
    ranges.append(relative)
    constant += relative <= 1e-5
    total += 1
ranges = np.array(ranges)
print(
    f"proposals {total}, constant {constant} ({constant / total:.3%}), {time.time() - start:.0f}s"
)
print("relative range quantiles", np.quantile(ranges, [0.001, 0.01, 0.05, 0.5]))
