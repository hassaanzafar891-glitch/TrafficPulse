"""Small CPU direct-horizon regression; no file loader or real-data dependency."""
from pathlib import Path
import json
import time

import numpy as np

from .data import HORIZONS, ForecastInput, Partition, TrainingStatistics, usable, windows


class LagCalendarFeatures:
    def __init__(self, train: Partition):
        self.statistics = TrainingStatistics.fit(train)

    @property
    def names(self):
        return ([f"speed_lag_{lag}" for lag in range(11,-1,-1)] +
                [f"observed_lag_{lag}" for lag in range(11,-1,-1)] +
                ["origin_daily_sin", "origin_daily_cos", "origin_weekly_sin", "origin_weekly_cos"] +
                [f"synthetic_sensor_position_{i}" for i in range(len(self.statistics.sensors))])

    def _rows(self, inputs: ForecastInput, *, fitting=False):
        if inputs.sensors != self.statistics.sensors or inputs.values.shape != (12,len(inputs.sensors)):
            raise ValueError("Feature input schema differs from training")
        if fitting:
            _, observed, scaled = self.statistics._fill_values(inputs.values)
        else:
            _, observed, scaled = self.statistics.fill(inputs)
        day = inputs.origin_time.astype("datetime64[D]")
        phase = float((inputs.origin_time-day)/np.timedelta64(1,"D"))
        weekday = (int(day.astype(np.int64))+3)%7
        angles = 2*np.pi*np.array([phase,(weekday+phase)/7])
        calendar = [np.sin(angles[0]),np.cos(angles[0]),np.sin(angles[1]),np.cos(angles[1])]
        sensor_count = len(inputs.sensors)
        return np.concatenate([scaled.T, observed.T.astype(float),
                               np.tile(calendar,(sensor_count,1)),np.eye(sensor_count)],axis=1).astype(np.float32)

    def transform(self, inputs: ForecastInput):
        """Features depend only on history, origin clock and frozen train stats."""
        return self._rows(inputs)

    def training_matrix(self, train: Partition):
        if (train.role != "train" or train.sensors != self.statistics.sensors or
                not len(train.timestamps) or train.timestamps[-1] != self.statistics.training_end):
            raise ValueError("Training feature matrix requires the fitted train partition")
        features, targets = [], []
        for window in windows(train):
            features.append(self._rows(window.inputs,fitting=True))
            targets.append(window.targets[np.array(HORIZONS)-1].T)
        if not features:
            raise ValueError("Training partition has no complete 12+12 windows")
        return np.concatenate(features), np.concatenate(targets)


class XGBoostDirect:
    """Three independent pooled-sensor boosters, outputs [3,S] in HORIZONS order."""
    def __init__(self, parameters: dict):
        self.parameters = dict(parameters)
        if (self.parameters.get("device") != "cpu" or self.parameters.get("tree_method") != "hist"
                or not 1 <= self.parameters.get("n_jobs",0) <= 4):
            raise ValueError("Require CPU hist with 1..4 threads")

    def fit(self, train: Partition):
        # Lazy import keeps Phase 2A usable without optional XGBoost installed.
        from xgboost import XGBRegressor
        started = time.perf_counter()
        self.features = LagCalendarFeatures(train)
        x,y = self.features.training_matrix(train)
        self.training_matrix_bytes = x.nbytes + y.nbytes
        self.models = []
        self.training_target_counts = []
        for column,horizon in enumerate(HORIZONS):
            mask = usable(y[:,column])
            if not mask.any():
                raise ValueError(f"No valid training targets at horizon {horizon}")
            model = XGBRegressor(**self.parameters)
            model.fit(x[mask],y[mask,column])  # no eval_set, early stopping or later rows
            self.models.append(model)
            self.training_target_counts.append(int(mask.sum()))
        self.training_seconds = time.perf_counter()-started
        return self

    def predict_many(self, inputs):
        if not inputs:
            return np.empty((0,3,len(self.features.statistics.sensors)))
        matrix = np.concatenate([self.features.transform(item) for item in inputs])
        sensor_count = len(self.features.statistics.sensors)
        return np.stack([m.predict(matrix).reshape(len(inputs),sensor_count)
                         for m in self.models],axis=1)

    def predict(self, inputs: ForecastInput):
        return self.predict_many([inputs])[0]

    @classmethod
    def restore_synthetic(cls, train: Partition, destination: Path):
        """Restore reviewed synthetic trees and verify train-derived preprocessing."""
        from xgboost import XGBRegressor
        metadata=json.loads((destination/"preprocessing.json").read_text(encoding="utf-8"))
        if (metadata.get("data_origin")!="synthetic_generated_not_real_traffic" or
                not all(name.startswith("synthetic") for name in train.sensors) or
                metadata["horizons"]!=list(HORIZONS)):
            raise ValueError("Require explicitly synthetic matching model artifacts")
        model=cls(metadata["parameters"])
        model.features=LagCalendarFeatures(train)
        stats=model.features.statistics
        if (metadata["sensors"]!=list(train.sensors) or metadata["training_end"]!=str(stats.training_end)
                or metadata["feature_names"]!=model.features.names):
            raise ValueError("Saved preprocessing schema does not match synthetic train")
        for key in ("medians","means","stds"):
            if not np.array_equal(metadata[key],getattr(stats,key)):
                raise ValueError("Saved preprocessing constants differ from synthetic train")
        model.models=[]
        for h in HORIZONS:
            estimator=XGBRegressor(device="cpu",n_jobs=4)
            estimator.load_model(destination/f"horizon_{h}.json")
            model.models.append(estimator)
        return model

    def save_synthetic_artifacts(self, destination: Path):
        """Persist synthetic fitted trees with JSON, never pickle or training rows."""
        destination.mkdir(parents=True,exist_ok=True)
        paths = []
        for horizon,model in zip(HORIZONS,self.models):
            path=destination/f"horizon_{horizon}.json"
            model.save_model(path)
            paths.append(path)
        stats=self.features.statistics
        metadata={"data_origin":"synthetic_generated_not_real_traffic", "horizons":list(HORIZONS),
                  "feature_names":self.features.names, "parameters":self.parameters,
                  "training_end":str(stats.training_end), "sensors":list(stats.sensors),
                  "medians":stats.medians.tolist(), "means":stats.means.tolist(), "stds":stats.stds.tolist()}
        path=destination/"preprocessing.json"
        path.write_text(json.dumps(metadata,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        return paths+[path]
