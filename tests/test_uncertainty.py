"""Synthetic calibration isolation, quantiles, masks and interval semantics."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from trafficpulse.data import Partition
from trafficpulse.synthetic import make_synthetic
from trafficpulse.uncertainty import SplitConformal,calibration_targets,conformal_quantile
from trafficpulse.uncertainty_experiment import calibration_prefix_parts


class UncertaintyTests(unittest.TestCase):
    def setUp(self):
        times=np.datetime64('2024-01-01T00:00')+np.arange(25)*np.timedelta64(5,'m')
        values=np.column_stack([np.arange(25)+10.,np.arange(25)+20.])
        self.part=Partition('calibration',times,values,('synthetic_a','synthetic_b'),100)
        _,self.targets=calibration_targets(self.part)

    def test_finite_sample_order_statistic_and_small_n_cap(self):
        self.assertEqual(conformal_quantile(np.arange(1,11),.8),(9.,10,9))
        self.assertEqual(conformal_quantile([4.,1.,2.],.9),(4.,3,3))
        self.assertEqual(conformal_quantile([0.]*10,.9),(0.,10,10))

    def test_invalid_quantiles_rejected(self):
        for coverage in (0,1,np.nan):
            with self.assertRaises(ValueError): conformal_quantile([1],coverage)
        for values in ([-1],[np.nan],[np.inf],[[1]]):
            with self.assertRaises(ValueError): conformal_quantile(values)

    def test_repeated_residuals_use_inclusive_endpoints_and_exact_width(self):
        # Residuals [1,2,2,4], rank ceil(5*.5)=3, q=2 without interpolation.
        errors=np.array([1.,2.,2.,4.]).reshape(2,2)
        predictions=self.targets-errors[:,None,:]
        q=SplitConformal.fit(self.part,predictions,coverage=.5)
        self.assertEqual(q.half_widths,(2.,2.,2.))
        self.assertEqual(q.ranks,(3,3,3))
        for item in q.calibration_diagnostics(self.part,predictions).values():
            self.assertEqual(item['coverage'],.75)
            self.assertEqual(item['average_interval_width'],4.)

    def test_nonfinite_diagnostics_preserve_denominator_and_masked_values(self):
        q=SplitConformal.fit(self.part,self.targets)
        for bad in (np.nan,np.inf,-np.inf):
            predictions=self.targets.copy();predictions[0,0,0]=bad
            item=q.calibration_diagnostics(self.part,predictions)['3']
            self.assertEqual(item['valid_target_count'],4)
            self.assertEqual(item['unavailable_interval_count'],1)
            self.assertIsNone(item['coverage'])
            self.assertIsNone(item['average_interval_width'])
        values=self.part.values.copy();values[14,0]=0
        part=replace(self.part,values=values);_,targets=calibration_targets(part)
        predictions=targets.copy();predictions[0,0,0]=np.nan
        q=SplitConformal.fit(part,predictions)
        item=q.calibration_diagnostics(part,predictions)['3']
        self.assertEqual(item['valid_target_count'],3)
        self.assertEqual(item['unavailable_interval_count'],0)
        self.assertEqual(item['coverage'],1.)

    def test_role_guard_precedes_any_data_access(self):
        for role in ('train','tuning','controller_selection','test'):
            with self.assertRaises(ValueError):
                SplitConformal.fit(SimpleNamespace(role=role),np.empty((0,3,1)))

    def test_horizons_and_pooled_sensor_residuals(self):
        np.testing.assert_array_equal(self.targets[0],self.part.values[[14,17,23]])
        predictions=self.targets-np.array([1.,2.,4.])[None,:,None]
        q=SplitConformal.fit(self.part,predictions)
        self.assertEqual(q.half_widths,(1.,2.,4.));self.assertEqual(q.counts,(4,4,4))
        lower,upper=q.intervals(predictions)
        np.testing.assert_array_equal(upper,self.targets)
        np.testing.assert_array_equal(lower,predictions-np.array([1.,2.,4.])[None,:,None])

    def test_zero_nonfinite_negative_target_mask(self):
        values=self.part.values.copy();values[14]=[0,np.nan];values[17]=[-1,0]
        part=replace(self.part,values=values);_,targets=calibration_targets(part)
        predictions=np.zeros(targets.shape)
        q=SplitConformal.fit(part,predictions)
        self.assertEqual(q.counts,(2,2,4))
        diagnostics=q.calibration_diagnostics(part,predictions)
        self.assertEqual([diagnostics[str(h)]['valid_target_count'] for h in (3,6,12)],[2,2,4])

    def test_empty_calibration_and_empty_horizon_are_explicit(self):
        part=replace(self.part,values=np.zeros_like(self.part.values))
        q=SplitConformal.fit(part,np.zeros(self.targets.shape))
        self.assertEqual(q.half_widths,(None,None,None))
        for item in q.calibration_diagnostics(part,np.zeros(self.targets.shape)).values():
            self.assertIsNone(item['coverage']);self.assertIsNone(item['average_interval_width'])
            self.assertEqual(item['reason'],'empty target mask')
        short=replace(self.part,timestamps=self.part.timestamps[:12],values=self.part.values[:12])
        q=SplitConformal.fit(short,np.empty((0,3,2)))
        self.assertEqual(q.counts,(0,0,0))

    def test_unavailable_width_never_silently_drops_valid_targets(self):
        q=SplitConformal(.9,(None,1.,2.),(0,4,4),(None,4,4),self.part.sensors)
        result=q.calibration_diagnostics(self.part,self.targets)
        self.assertEqual(result['3']['valid_target_count'],4)
        self.assertEqual(result['3']['unavailable_interval_count'],4)
        self.assertIsNone(result['3']['coverage'])

    def test_nonfinite_predictions_on_valid_targets_rejected(self):
        predictions=self.targets.copy();predictions[0,0,0]=np.nan
        with self.assertRaises(ValueError): SplitConformal.fit(self.part,predictions)
        with self.assertRaises(ValueError): SplitConformal.fit(self.part,np.zeros((2,12,2)))

    def test_determinism_immutability_and_unclipped_endpoints(self):
        original=self.part.values.copy(); predictions=np.full(self.targets.shape,1.)
        first=SplitConformal.fit(self.part,predictions);second=SplitConformal.fit(self.part,predictions)
        self.assertEqual(first,second)
        lower,_=first.intervals(predictions);self.assertTrue((lower<0).all())
        np.testing.assert_array_equal(self.part.values,original)
        self.assertTrue(all(v['coverage']==1 for v in first.calibration_diagnostics(self.part,predictions).values()))

    def test_prefix_reproduces_original_without_changing_default(self):
        # Small fixture only; compare the prefix, never inspect any test partition.
        original=make_synthetic(days=2)
        prefix=make_synthetic(days=2,stop_rows=100)
        np.testing.assert_array_equal(prefix.values,original.values[:100])
        np.testing.assert_array_equal(prefix.timestamps,original.timestamps[:100])
        self.assertEqual(prefix.origin,'synthetic_generated_not_real_traffic')
        for bad in (0,577,1.5):
            with self.assertRaises(ValueError): make_synthetic(days=2,stop_rows=bad)

    def test_prefix_keeps_frozen_boundaries_and_only_allowed_roles(self):
        prefix=make_synthetic(stop_rows=6208)
        train,cal=calibration_prefix_parts(prefix,8064)
        self.assertEqual((train.role,cal.role),('train','calibration'))
        self.assertEqual((len(train.values),cal.offset,len(cal.values)),(5644,5966,242))
        samples,_=calibration_targets(cal)
        self.assertEqual((len(samples),samples[0].origin_index,samples[-1].origin_index),(219,5977,6195))
        self.assertTrue(all(w.origin_index-11>=5966 and w.origin_index+12<6208 for w in samples))
        with self.assertRaises(ValueError): calibration_prefix_parts(make_synthetic(stop_rows=6207),8064)

    def test_calibration_does_not_fit_models_or_training_statistics(self):
        with patch('trafficpulse.data.TrainingStatistics.fit',side_effect=AssertionError('training forbidden')):
            q=SplitConformal.fit(self.part,self.targets)
            self.assertEqual(q.half_widths,(0.,0.,0.))
