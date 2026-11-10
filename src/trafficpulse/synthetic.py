"""Small fabricated traffic-like fixture. No external dataset or sensor metadata."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticSeries:
    timestamps: np.ndarray
    values: np.ndarray
    sensors: tuple[str, ...]
    seed: int
    origin: str = "synthetic_generated_not_real_traffic"
    units: str = "synthetic_speed_units"


def make_synthetic(days=28, sensors=4, seed=1729, *, stop_rows=None):
    """Optionally reproduce only a prefix without constructing suffix observations.

    Consume/discard remaining random noise draws to preserve the original global
    missingness permutation. RNG draws/placement indices are not traffic values.
    """
    if not 1 <= days <= 60 or not 1 <= sensors <= 16:
        raise ValueError("Synthetic smoke fixture limited to 1..60 days and 1..16 sensors")
    rng = np.random.Generator(np.random.PCG64(seed))
    total_rows = days * 288
    rows = total_rows if stop_rows is None else stop_rows
    if not isinstance(rows,int) or not 1 <= rows <= total_rows:
        raise ValueError("stop_rows must be an integer within the original fixture")
    times = np.datetime64("2024-01-01T00:00", "ns") + np.arange(rows) * np.timedelta64(5, "m")
    hours = (np.arange(rows) % 288) / 12
    rush = 18*np.exp(-0.5*((hours-8)/1.3)**2) + 22*np.exp(-0.5*((hours-17)/1.6)**2)
    speeds = 62 - rush[:,None] - np.arange(sensors)[None,:]*1.5
    speeds = np.maximum(5, speeds + rng.normal(0, 1, (rows, sensors)))
    remaining=(total_rows-rows)*sensors
    while remaining:
        count=min(remaining,4096)
        rng.normal(0,1,count)  # discarded RNG state advancement, no suffix observations
        remaining-=count
    order = rng.permutation(total_rows*sensors)
    zero_end = total_rows*sensors * 2 // 100
    missing_end = zero_end + total_rows*sensors // 100
    zero_indices=order[:zero_end]; missing_indices=order[zero_end:missing_end]
    speeds.flat[zero_indices[zero_indices<speeds.size]] = 0
    speeds.flat[missing_indices[missing_indices<speeds.size]] = np.nan
    times.setflags(write=False)
    speeds.setflags(write=False)
    return SyntheticSeries(times, speeds, tuple(f"synthetic_sensor_{i}" for i in range(sensors)), seed)
