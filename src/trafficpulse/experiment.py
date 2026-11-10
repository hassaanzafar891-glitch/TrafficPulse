"""Synthetic-only smoke experiment; no real-data loader or dataset-path option."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from .baselines import Persistence, SeasonalHistoricalAverage
from .data import chronological_partitions, windows
from .metrics import HorizonMetrics
from .synthetic import make_synthetic


def run_synthetic():
    started = time.perf_counter()
    with threadpool_limits(limits=4):
        fixture = make_synthetic()
        parts = chronological_partitions(fixture.timestamps, fixture.values, fixture.sensors)
        models = {"persistence": Persistence().fit(parts["train"]),
                  "seasonal_historical_average": SeasonalHistoricalAverage().fit(parts["train"])}
        meters = {name: HorizonMetrics() for name in models}
        count = 0
        for window in windows(parts["tuning"]):
            for name, model in models.items():
                meters[name].update(window.targets, model.predict(window.inputs))
            count += 1
        pools = [{"internal_api": p["internal_api"], "num_threads": p["num_threads"]}
                 for p in threadpool_info()]
        if any(p["num_threads"] > 4 for p in pools):
            raise RuntimeError("Numerical thread limit exceeded")
    return {"data_origin": fixture.origin, "units": fixture.units,
            "real_dataset_access": "none; in-memory synthetic fixture only",
            "synthetic_values_sha256": hashlib.sha256(fixture.values.tobytes()).hexdigest(),
            "parameters": {"seed": fixture.seed, "days": 28, "sensors": 4,
                           "input_steps": 12, "target_steps": 12,
                           "scored_horizons": "3,6,12", "scored_partition": "tuning",
                           "cpu_threads_max": 4, "wall_time_limit_seconds": 1800},
            "partitions": {name: {"start": p.offset, "stop": p.offset+len(p.timestamps),
                                   "potential_windows": max(0,len(p.timestamps)-23)}
                           for name, p in parts.items()},
            "scored_windows": count,
            "synthetic_zero_cells": int((fixture.values == 0).sum()),
            "synthetic_nan_cells": int(np.isnan(fixture.values).sum()),
            "metrics": {name: meter.result() for name, meter in meters.items()},
            "numerical_threadpools": pools,
            "compute_elapsed_seconds": time.perf_counter()-started,
            "tracking": {"status": "disabled"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlflow", action="store_true", help="Log metrics to workspace SQLite only")
    args = parser.parse_args()
    started = time.perf_counter()
    workspace = Path(__file__).resolve().parents[2]
    result = run_synthetic()
    if args.mlflow:
        from .tracking import log_local_run
        result["tracking"] = log_local_run(result, workspace)
    result["total_elapsed_seconds"] = time.perf_counter()-started
    output = workspace / "reports/generated/phase2a_synthetic.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
