"""Synthetic-only behavioral tests; no dependency on local METR-LA files."""
import io
import math
from datetime import datetime, timedelta
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from trafficpulse.baselines import Persistence, SeasonalHistoricalAverage, calendar_slots
from trafficpulse.data import (ForecastInput, Partition, STEP, TrainingStatistics,
                               chronological_partitions, windows)
from trafficpulse.experiment import run_synthetic
from trafficpulse.metrics import HorizonMetrics, masked_metrics
from trafficpulse.synthetic import make_synthetic


def series(rows=1000, sensors=2):
    times = np.datetime64("2024-01-01T00:00", "ns") + np.arange(rows)*STEP
    values = 1 + np.arange(rows*sensors, dtype=float).reshape(rows, sensors)
    return times, values, tuple(f"synthetic_{i}" for i in range(sensors))


class WindowTests(unittest.TestCase):
    def test_horizon_alignment_and_full_target_span(self):
        times, values, names = series()
        tuning = chronological_partitions(times, values, names)["tuning"]
        window = next(windows(tuning))
        t = window.origin_index
        np.testing.assert_array_equal(window.inputs.values, values[t-11:t+1])
        np.testing.assert_array_equal(window.targets, values[t+1:t+13])
        for h in (3, 6, 12):
            np.testing.assert_array_equal(window.targets[h-1], values[t+h])
            self.assertEqual(times[t+h]-window.inputs.origin_time, h*STEP)

    def test_partition_boundaries_and_counts_have_no_shared_raw_rows(self):
        times, values, names = series()
        parts = chronological_partitions(times, values, names)
        previous_used = set()
        for part in parts.values():
            items = list(windows(part))
            self.assertEqual(len(items), max(0,len(part.values)-23))
            used = set()
            for window in items:
                self.assertGreaterEqual(window.origin_index-11, part.offset)
                self.assertLess(window.origin_index+12, part.offset+len(part.values))
                used.update(range(window.origin_index-11, window.origin_index+13))
            self.assertTrue(previous_used.isdisjoint(used))
            previous_used.update(used)

    def test_gap_crossing_windows_are_rejected(self):
        times, values, names = series(60)
        times[30:] += STEP
        part = Partition("tuning", times, values, names, 0)
        origins = [w.origin_index for w in windows(part)]
        self.assertTrue(origins)
        self.assertTrue(all(not (t-11 < 30 <= t+12) for t in origins))

    def test_duplicate_and_reversed_timestamps_rejected(self):
        times, values, names = series()
        times[10] = times[9]
        with self.assertRaises(ValueError):
            chronological_partitions(times, values, names)
        with self.assertRaises(ValueError):
            chronological_partitions(times[::-1], values, names)

    def test_all_zero_targets_do_not_change_origin_selection(self):
        times, values, names = series()
        zero_parts = chronological_partitions(times, np.zeros_like(values), names)
        normal_parts = chronological_partitions(times, values, names)
        self.assertEqual([w.origin_index for w in windows(zero_parts["tuning"])],
                         [w.origin_index for w in windows(normal_parts["tuning"])])

    def test_partition_data_is_independent_and_read_only(self):
        times, values, names = series()
        parts = chronological_partitions(times, values, names)
        values[:] = 99999
        self.assertNotEqual(parts["train"].values[0,0], 99999)
        with self.assertRaises(ValueError):
            parts["tuning"].values[0,0] = 1


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.times, self.values, self.names = series()
        self.parts = chronological_partitions(self.times, self.values, self.names)

    def test_persistence_uses_last_valid_per_sensor(self):
        model = Persistence().fit(self.parts["train"])
        history = np.full((12,2), np.nan)
        history[3,0] = 15
        history[7,1] = 29
        history[-1] = [0, -4]
        inputs = ForecastInput(history, self.times[711], self.names)
        np.testing.assert_array_equal(model.predict(inputs), np.tile([15,29],(12,1)))

    def test_causal_fill_uses_training_median_not_later_input(self):
        stats = TrainingStatistics.fit(self.parts["train"])
        history = np.zeros((12,2))
        history[5] = [10000, 20000]
        filled, observed, scaled = stats.fill(ForecastInput(history,self.times[711],self.names))
        np.testing.assert_array_equal(filled[:5],np.tile(stats.medians,(5,1)))
        np.testing.assert_array_equal(filled[5:],np.tile([10000,20000],(7,1)))
        self.assertEqual(int(observed.sum()),2)
        self.assertTrue(np.isfinite(scaled).all())

    def test_all_missing_history_uses_train_median_and_no_carry(self):
        model = Persistence().fit(self.parts["train"])
        origin = self.times[711]
        model.predict(ForecastInput(np.full((12,2),99999.),origin,self.names))
        predicted = model.predict(ForecastInput(np.full((12,2),np.nan),origin+STEP,self.names))
        np.testing.assert_array_equal(predicted,np.tile(model.statistics.medians,(12,1)))

    def test_fit_rejects_non_train_and_invalid_training(self):
        for model in (Persistence(),SeasonalHistoricalAverage()):
            with self.assertRaises(ValueError):
                model.fit(self.parts["tuning"])
        train = self.parts["train"]
        with self.assertRaises(ValueError):
            TrainingStatistics.fit(Partition("train",train.timestamps,np.zeros_like(train.values),self.names,0))

    def test_cold_sensor_falls_back_to_global_train_statistics(self):
        values = np.full((30,2),np.nan); values[:,0]=10
        train=Partition("train",self.times[:30],values,self.names,0)
        stats=TrainingStatistics.fit(train)
        np.testing.assert_array_equal(stats.medians,[10,10])
        np.testing.assert_array_equal(stats.stds,[1,1])

    def test_full_train_statistics_cannot_predict_inside_training_period(self):
        model=Persistence().fit(self.parts["train"])
        with self.assertRaises(ValueError):
            model.predict(next(windows(self.parts["train"])).inputs)

    def test_later_values_do_not_change_models_or_predictions(self):
        changed=self.values.copy(); changed[700:]=1e9
        altered=chronological_partitions(self.times,changed,self.names)
        inputs=next(windows(self.parts["tuning"])).inputs
        for cls in (Persistence,SeasonalHistoricalAverage):
            before=cls().fit(self.parts["train"])
            after=cls().fit(altered["train"])
            np.testing.assert_array_equal(before.predict(inputs),after.predict(inputs))

    def test_seasonal_exact_weekday_and_time_slot(self):
        fixture=make_synthetic(days=28)
        parts=chronological_partitions(fixture.timestamps,fixture.values,fixture.sensors)
        model=SeasonalHistoricalAverage().fit(parts["train"])
        inputs=next(windows(parts["tuning"])).inputs
        prediction=model.predict(inputs)
        train=parts["train"]
        week,_=calendar_slots(train.timestamps)
        future,_=calendar_slots(inputs.origin_time+np.arange(1,13)*STEP)
        for h,slot in enumerate(future):
            for sensor in range(4):
                candidates=train.values[week==slot,sensor]
                candidates=candidates[np.isfinite(candidates)&(candidates>0)]
                self.assertAlmostEqual(prediction[h,sensor],float(candidates.mean()))

    def test_seasonal_empty_weekly_slot_uses_time_of_day_then_median(self):
        # Monday history, Tuesday origin: weekday absent but time-of-day exists.
        times,_,names=series(288)
        values=np.full((288,2),20.)
        train=Partition("train",times,values,names,0)
        model=SeasonalHistoricalAverage().fit(train)
        origin=times[0]+np.timedelta64(1,'D')
        prediction=model.predict(ForecastInput(np.full((12,2),99.),origin,names))
        np.testing.assert_array_equal(prediction,np.full((12,2),20.))
        sparse=Partition("train",times[:1],values[:1],names,0)
        model=SeasonalHistoricalAverage().fit(sparse)
        np.testing.assert_array_equal(model.predict(ForecastInput(np.ones((12,2)),origin,names)),np.full((12,2),20.))

    def test_seasonal_target_slots_cross_sunday_midnight(self):
        # Independent Python calendar oracle guards weekday/target off-by-one errors.
        clock=[datetime(2024,1,1)+timedelta(minutes=5*i) for i in range(14*288)]
        values=np.array([[1+stamp.weekday()*1000+stamp.hour*12+stamp.minute//5]
                         for stamp in clock],dtype=float)
        train=Partition('train',np.array(clock,dtype='datetime64[ns]'),values,('synthetic',),0)
        model=SeasonalHistoricalAverage().fit(train)
        origin=np.datetime64('2024-01-21T23:50','ns')
        forecast=model.predict(ForecastInput(np.full((12,1),50.),origin,('synthetic',)))
        np.testing.assert_array_equal(forecast[[2,5,11],0],[2,5,11])


class MetricTests(unittest.TestCase):
    def test_zero_nan_negative_targets_excluded(self):
        result=masked_metrics([10,0,np.nan,20,-1],[12,999,np.nan,16,99])
        self.assertEqual(result['valid_target_count'],2)
        self.assertEqual(result['mae'],3)
        self.assertAlmostEqual(result['rmse'],math.sqrt(10))

    def test_empty_mask_is_explicit_null(self):
        result=masked_metrics([0,np.nan],[np.nan,np.nan])
        self.assertIsNone(result['mae']); self.assertIsNone(result['rmse'])
        self.assertEqual(result['reason'],'empty_target_mask')
        self.assertEqual(result['valid_target_count'],0)

    def test_nonfinite_prediction_on_valid_target_is_error(self):
        with self.assertRaises(ValueError):
            masked_metrics([1],[np.nan])

    def test_per_horizon_index_and_counts(self):
        y=np.ones((12,2)); prediction=y.copy()
        prediction[2]+=1; prediction[5]+=2; prediction[11]+=3
        y[5,0]=0
        meters=HorizonMetrics(); meters.update(y,prediction)
        result=meters.result()
        self.assertEqual([result[str(h)]['mae'] for h in (3,6,12)],[1,2,3])
        self.assertEqual([result[str(h)]['valid_target_count'] for h in (3,6,12)],[2,1,2])

    def test_streaming_metrics_weight_valid_cells_not_window_means(self):
        meter=HorizonMetrics()
        first=np.ones((12,2)); second=first.copy(); second[:,1]=0
        meter.update(first,np.full((12,2),3.))  # two errors of 2
        meter.update(second,np.array([[5.,999.]]*12))  # one error of 4
        for result in meter.result().values():
            self.assertEqual(result['valid_target_count'],3)
            self.assertAlmostEqual(result['mae'],8/3)
            self.assertAlmostEqual(result['rmse'],math.sqrt(8))


class SyntheticExperimentTests(unittest.TestCase):
    def test_fixture_seed_and_invalid_cases_are_deterministic(self):
        a,b=make_synthetic(),make_synthetic()
        np.testing.assert_array_equal(a.values,b.values)
        self.assertFalse(np.array_equal(a.values,make_synthetic(seed=99).values,equal_nan=True))
        self.assertGreater(int((a.values==0).sum()),0)
        self.assertGreater(int(np.isnan(a.values).sum()),0)
        self.assertIn('synthetic',a.origin)

    def test_experiment_repeatability_without_real_file_access(self):
        original_open=io.open
        def guarded_open(file,*args,**kwargs):
            if isinstance(file,(str,Path)):
                normalized=str(file).replace('\\','/').lower()
                if '/data/' in normalized or 'reports/local' in normalized or normalized.endswith('.h5'):
                    raise AssertionError('Forbidden real-data access')
            return original_open(file,*args,**kwargs)
        with patch('io.open',side_effect=guarded_open):
            first,second=run_synthetic(),run_synthetic()
        self.assertEqual(first['metrics'],second['metrics'])
        self.assertEqual(first['synthetic_values_sha256'],second['synthetic_values_sha256'])
        self.assertEqual(first['scored_windows'],first['partitions']['tuning']['potential_windows'])
        self.assertTrue(all(p['num_threads']<=4 for p in first['numerical_threadpools']))
        self.assertEqual(first['tracking']['status'],'disabled')


if __name__ == '__main__':
    unittest.main()
