"""Masked point metrics; invalid predictions never disappear from denominators."""
import math

import numpy as np

from .data import HORIZONS, usable


def error_totals(targets, predictions):
    y, p = np.asarray(targets), np.asarray(predictions)
    if y.shape != p.shape:
        raise ValueError("Prediction and target shapes must match")
    mask = usable(y)
    if not np.isfinite(p[mask]).all():
        raise ValueError("Nonfinite prediction on a valid target")
    error = p[mask] - y[mask]
    return int(mask.sum()), float(np.abs(error).sum()), float(np.square(error).sum())


def summarize(count, absolute_error, squared_error):
    return {"valid_target_count": count,
            "mae": absolute_error / count if count else None,
            "rmse": math.sqrt(squared_error / count) if count else None,
            "reason": None if count else "empty_target_mask"}


def masked_metrics(targets, predictions):
    return summarize(*error_totals(targets, predictions))


class HorizonMetrics:
    def __init__(self):
        self.totals = {h: [0, 0.0, 0.0] for h in HORIZONS}

    def update(self, targets, predictions):
        if np.shape(targets) != np.shape(predictions) or np.ndim(targets) != 2 or len(targets) != 12:
            raise ValueError("Expected equal [12,S] forecast and target arrays")
        for h in HORIZONS:
            values = error_totals(targets[h-1], predictions[h-1])
            self.totals[h] = [a+b for a, b in zip(self.totals[h], values)]

    def result(self):
        return {str(h): summarize(*totals) for h, totals in self.totals.items()}
