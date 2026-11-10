"""Transparent frozen training-only point forecasts, in original speed units."""
import numpy as np

from .data import ForecastInput, Partition, STEP, TrainingStatistics, usable


def calendar_slots(times):
    times = np.asarray(times, dtype="datetime64[ns]")
    days = times.astype("datetime64[D]")
    weekdays = (days.astype(np.int64) + 3) % 7  # 1970-01-01 was Thursday
    slots = ((times - days) / STEP).astype(np.int64)
    return weekdays * 288 + slots, slots


class Persistence:
    def fit(self, train: Partition):
        self.statistics = TrainingStatistics.fit(train)
        return self

    def predict(self, inputs: ForecastInput):
        filled, _, _ = self.statistics.fill(inputs)
        return np.repeat(filled[-1:,:], 12, axis=0)


class SeasonalHistoricalAverage:
    def fit(self, train: Partition):
        self.statistics = TrainingStatistics.fit(train)
        weekly_slots, daily_slots = calendar_slots(train.timestamps)
        observed = usable(train.values)
        self.weekly = self._means(weekly_slots, train.values, observed, 7 * 288)
        self.daily = self._means(daily_slots, train.values, observed, 288)
        return self

    @staticmethod
    def _means(slots, values, observed, size):
        sums = np.zeros((size, values.shape[1]))
        counts = np.zeros_like(sums)
        np.add.at(sums, slots, np.where(observed, values, 0))
        np.add.at(counts, slots, observed)
        means = np.full_like(sums, np.nan)
        np.divide(sums, counts, out=means, where=counts > 0)
        means.setflags(write=False)
        return means

    def predict(self, inputs: ForecastInput):
        # Check schema and temporal eligibility, without reading future labels.
        self.statistics.fill(inputs)
        target_times = inputs.origin_time + np.arange(1, 13) * STEP
        weekly, daily = calendar_slots(target_times)
        result = self.weekly[weekly].copy()
        result = np.where(np.isfinite(result), result, self.daily[daily])
        return np.where(np.isfinite(result), result, self.statistics.medians)
