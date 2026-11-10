"""In-memory chronological partitioning and causal 12-to-12 windows."""
from dataclasses import dataclass
from typing import Iterator

import numpy as np

STEP = np.timedelta64(5, "m")
HORIZONS = (3, 6, 12)
ROLES = ("train", "tuning", "calibration", "controller_selection", "test")


@dataclass(frozen=True)
class Partition:
    role: str
    timestamps: np.ndarray
    values: np.ndarray
    sensors: tuple[str, ...]
    offset: int


def chronological_partitions(timestamps, values, sensors) -> dict[str, Partition]:
    """Copy caller arrays; split raw rows before constructing any windows.

    Uses the frozen 70/10/20 rule with 40/30/remainder validation subdivision.
    Gaps are retained here and rejected at window level; duplicate/reversed times
    and invalid dimensions are contract errors. No file I/O is supported.
    """
    times = np.array(timestamps, dtype="datetime64[ns]", copy=True)
    array = np.array(values, dtype=float, copy=True)
    names = tuple(sensors)
    if times.ndim != 1 or array.ndim != 2 or array.shape != (len(times), len(names)):
        raise ValueError("Expected timestamps [T], readings [T,S], and S sensor names")
    if not names or len(set(names)) != len(names) or len(times) == 0:
        raise ValueError("Need nonempty data and unique sensor names")
    if np.isnat(times).any() or np.any(np.diff(times) <= np.timedelta64(0, "ns")):
        raise ValueError("Timestamps must be unique, ordered and non-NaT")
    times.setflags(write=False)
    array.setflags(write=False)
    n = len(times)
    train_end, val_end = 7 * n // 10, 8 * n // 10
    val_length = val_end - train_end
    tune_end = train_end + 4 * val_length // 10
    cal_end = tune_end + 3 * val_length // 10
    boundaries = (0, train_end, tune_end, cal_end, val_end, n)
    return {role: Partition(role, times[a:b], array[a:b], names, a)
            for role, a, b in zip(ROLES, boundaries, boundaries[1:])}


@dataclass(frozen=True)
class ForecastInput:
    values: np.ndarray  # [12,S]; raw historical observations only
    origin_time: np.datetime64
    sensors: tuple[str, ...]


@dataclass(frozen=True)
class Window:
    inputs: ForecastInput
    targets: np.ndarray  # [12,S]; original unfilled targets
    origin_index: int


def windows(partition: Partition) -> Iterator[Window]:
    for t in range(11, len(partition.timestamps) - 12):
        if not np.all(np.diff(partition.timestamps[t-11:t+13]) == STEP):
            continue
        yield Window(ForecastInput(partition.values[t-11:t+1],
                                   partition.timestamps[t], partition.sensors),
                     partition.values[t+1:t+13], partition.offset + t)


def usable(values):
    return np.isfinite(values) & (values > 0)


@dataclass(frozen=True)
class TrainingStatistics:
    medians: np.ndarray
    means: np.ndarray
    stds: np.ndarray
    sensors: tuple[str, ...]
    training_end: np.datetime64

    @classmethod
    def fit(cls, train: Partition):
        if train.role != "train" or len(train.timestamps) == 0:
            raise ValueError("Statistics may only be fitted on nonempty train partition")
        global_values = train.values[usable(train.values)]
        if not len(global_values):
            raise ValueError("No usable training observations")
        medians, means, stds = [], [], []
        for column in train.values.T:
            valid = column[usable(column)]
            if not len(valid):
                valid = global_values
            medians.append(np.median(valid))
            means.append(np.mean(valid))
            stds.append(np.std(valid) or 1.0)
        arrays = [np.array(x) for x in (medians, means, stds)]
        for array in arrays:
            array.setflags(write=False)
        return cls(*arrays, train.sensors, train.timestamps[-1])

    def fill(self, inputs: ForecastInput):
        """Return raw-unit filled values, original availability and scaled lags."""
        if inputs.sensors != self.sensors or inputs.values.shape != (12, len(self.sensors)):
            raise ValueError("Forecast input schema differs from fitted training schema")
        if inputs.origin_time <= self.training_end:
            raise ValueError("Forecast origin must be later than fitted training history")
        return self._fill_values(inputs.values)

    def _fill_values(self, values):
        """Shared causal fill; train feature construction calls this internally.

        Forecast callers must use fill(), which also enforces the temporal/schema
        guard. Fitting may transform train windows using frozen train-only stats.
        """
        observed = usable(values)
        filled = np.empty_like(values, dtype=float)
        last = self.medians.copy()
        for step in range(12):
            last = np.where(observed[step], values[step], last)
            filled[step] = last
        return filled, observed, (filled - self.means) / self.stds
