"""Synthetic-only direct horizon / feature and determinism regressions."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from trafficpulse.data import ForecastInput, HORIZONS, chronological_partitions, windows
from trafficpulse.synthetic import make_synthetic
from trafficpulse.xgboost_baseline import LagCalendarFeatures, XGBoostDirect

PARAMETERS={"device":"cpu","tree_method":"hist","n_jobs":4,"random_state":1729,
            "objective":"reg:squarederror","n_estimators":4,"max_depth":2,
            "subsample":1.,"colsample_bytree":1.,"max_bin":32}


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.fixture=make_synthetic(days=4,sensors=2)
        self.parts=chronological_partitions(self.fixture.timestamps,self.fixture.values,self.fixture.sensors)
        self.features=LagCalendarFeatures(self.parts['train'])

    def test_feature_alignment_and_masks(self):
        inputs=next(windows(self.parts['tuning'])).inputs
        rows=self.features.transform(inputs)
        _,observed,scaled=self.features.statistics.fill(inputs)
        np.testing.assert_allclose(rows[:,:12],scaled.T,rtol=1e-6)
        np.testing.assert_array_equal(rows[:,12:24],observed.T)
        np.testing.assert_array_equal(rows[:,-2:],np.eye(2))
        self.assertEqual(rows.shape,(2,30))
        self.assertEqual(len(self.features.names),30)

    def test_training_labels_match_direct_horizons_without_crossing_boundary(self):
        train=self.parts['train']
        matrix,targets=self.features.training_matrix(train)
        first=next(windows(train))
        np.testing.assert_array_equal(targets[:2],first.targets[np.array(HORIZONS)-1].T)
        self.assertEqual(matrix.shape[0],(len(train.values)-23)*2)
        self.assertEqual(targets.shape,(matrix.shape[0],3))
        with self.assertRaises(ValueError):
            self.features.training_matrix(self.parts['tuning'])
        with self.assertRaises(ValueError):
            self.features.transform(first.inputs)

    def test_future_values_cannot_change_features(self):
        changed=self.fixture.values.copy(); changed[len(self.parts['train'].values):]=99999
        altered=chronological_partitions(self.fixture.timestamps,changed,self.fixture.sensors)
        other=LagCalendarFeatures(altered['train'])
        inputs=next(windows(self.parts['tuning'])).inputs
        np.testing.assert_array_equal(self.features.transform(inputs),other.transform(inputs))

    def test_all_missing_features_are_finite_and_no_future_backfill(self):
        inputs=next(windows(self.parts['tuning'])).inputs
        missing=ForecastInput(np.full((12,2),np.nan),inputs.origin_time,inputs.sensors)
        rows=self.features.transform(missing)
        self.assertTrue(np.isfinite(rows).all())
        self.assertEqual(float(rows[:,12:24].sum()),0)
        history=np.zeros((12,2)); history[6]=9999
        mixed=self.features.transform(ForecastInput(history,inputs.origin_time,inputs.sensors))
        stats=self.features.statistics
        expected=(stats.medians-stats.means)/stats.stds
        np.testing.assert_allclose(mixed[:,:6],np.repeat(expected[:,None],6,axis=1),rtol=1e-6)


@unittest.skipUnless(importlib.util.find_spec('xgboost'),'Optional XGBoost is not installed')
class XGBoostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture=make_synthetic(days=4,sensors=2)
        cls.parts=chronological_partitions(fixture.timestamps,fixture.values,fixture.sensors)
        cls.model=XGBoostDirect(PARAMETERS).fit(cls.parts['train'])
        cls.inputs=[w.inputs for w in windows(cls.parts['tuning'])]

    def test_determinism_and_prediction_shapes(self):
        other=XGBoostDirect(PARAMETERS).fit(self.parts['train'])
        a=self.model.predict_many(self.inputs)
        np.testing.assert_array_equal(a,other.predict_many(self.inputs))
        self.assertEqual(a.shape,(len(self.inputs),3,2))
        np.testing.assert_array_equal(a[0],self.model.predict(self.inputs[0]))

    def test_missing_input_predictions_and_training_counts(self):
        original=self.inputs[0]
        missing=ForecastInput(np.zeros((12,2)),original.origin_time,original.sensors)
        self.assertTrue(np.isfinite(self.model.predict(missing)).all())
        _,targets=self.model.features.training_matrix(self.parts['train'])
        counts=[int((np.isfinite(targets[:,i])&(targets[:,i]>0)).sum()) for i in range(3)]
        self.assertEqual(self.model.training_target_counts,counts)

    def test_rejects_validation_fitting_gpu_and_excess_threads(self):
        with self.assertRaises(ValueError):
            XGBoostDirect(PARAMETERS).fit(self.parts['tuning'])
        for bad in ({'device':'cuda'},{'n_jobs':8}):
            with self.assertRaises(ValueError):
                XGBoostDirect({**PARAMETERS,**bad})

    def test_json_model_artifacts_round_trip(self):
        from xgboost import XGBRegressor
        with tempfile.TemporaryDirectory() as location:
            paths=self.model.save_synthetic_artifacts(Path(location))
            self.assertEqual(len(paths),4)
            restored=XGBoostDirect.restore_synthetic(self.parts['train'],Path(location))
            np.testing.assert_array_equal(restored.predict_many(self.inputs),self.model.predict_many(self.inputs))
            matrix=self.model.features.transform(self.inputs[0])
            for column,path in enumerate(paths[:3]):
                loaded=XGBRegressor(n_jobs=4,device='cpu')
                loaded.load_model(path)
                np.testing.assert_array_equal(loaded.predict(matrix),self.model.predict(self.inputs[0])[column])
