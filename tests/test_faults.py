"""Synthetic-only golden RNG, input overlay and clean-target safety tests."""
from dataclasses import replace
import math
import unittest

import numpy as np

from trafficpulse.data import ForecastInput,Partition,TrainingStatistics,chronological_partitions,usable,windows
from trafficpulse.fault_experiment import compare_models
from trafficpulse.faults import FaultSpec,array_hash,historical_inputs,inject,noise_scales
from trafficpulse.synthetic import make_synthetic


class FaultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture=make_synthetic()
        cls.parts=chronological_partitions(fixture.timestamps,fixture.values,fixture.sensors)
        cls.train=cls.parts['train']; cls.tuning=cls.parts['tuning']
        cls.stats=TrainingStatistics.fit(cls.train); cls.scales=noise_scales(cls.train)

    def overlay(self,kind='missing',mode='point',rate=.4,seed=1729,**kwargs):
        return inject(self.tuning,FaultSpec(kind,mode,rate,seed,**kwargs),self.stats,self.scales)

    def test_point_rate_and_exact_reference_selection(self):
        overlay=self.overlay()
        eligible=np.flatnonzero(usable(self.tuning.values).ravel())
        rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([1729,1,1,1])))
        expected=np.sort(rng.permutation(eligible)[:math.floor(.4*len(eligible))])
        np.testing.assert_array_equal(np.flatnonzero(overlay.selected),expected)
        self.assertEqual(overlay.details['selected_cell_count'],math.floor(.4*len(eligible)))
        self.assertEqual(overlay.details['natural_unavailable_overlap_count'],0)

    def test_determinism_seed_separation_and_nested_rates(self):
        first=self.overlay(); second=self.overlay()
        np.testing.assert_array_equal(first.values,second.values)
        self.assertEqual(first.details,second.details)
        self.assertNotEqual(first.details['selected_mask_sha256'],self.overlay(seed=2718).details['selected_mask_sha256'])
        small=self.overlay(rate=.1)
        self.assertTrue(np.all(~small.selected|first.selected))

    def test_contiguous_sensor_count_and_golden_order(self):
        for kind,expected in [('missing',[3]),('stuck',[3]),('noisy',[1])]:
            overlay=self.overlay(kind,'contiguous')
            self.assertEqual(overlay.details['selected_sensor_indices'],expected)
            self.assertEqual(overlay.details['selected_cell_count'],24)  # two 12-row episodes
            for block in overlay.details['blocks']:
                self.assertEqual(block['stop']-block['start'],12)
                self.assertGreaterEqual(block['start'],0)
                self.assertLessEqual(block['stop'],len(self.tuning.values))
        for rate in (.1,.2):
            overlay=self.overlay(mode='contiguous',rate=rate)
            self.assertEqual(overlay.details['selected_sensor_indices'],[])
            self.assertEqual(overlay.details['changed_cell_count'],0)

    def test_contiguous_starts_and_noise_are_rate_nested(self):
        small=self.overlay('noisy','contiguous',.4)
        large=self.overlay('noisy','contiguous',.75)
        self.assertTrue(np.all(~small.selected|large.selected))
        np.testing.assert_array_equal(small.values[small.selected],large.values[small.selected])

    def test_clean_targets_and_source_unchanged_under_every_family(self):
        original=array_hash(self.tuning.values)
        target_hashes=[array_hash(w.targets) for w in windows(self.tuning)]
        for kind,mode in [('missing','point'),('missing','contiguous'),('stuck','contiguous'),('noisy','point'),('noisy','contiguous')]:
            overlay=self.overlay(kind,mode)
            inputs=historical_inputs(self.tuning,overlay)
            self.assertEqual(len(inputs),len(target_hashes))
            self.assertEqual(array_hash(self.tuning.values),original)
            self.assertEqual([array_hash(w.targets) for w in windows(self.tuning)],target_hashes)
            self.assertFalse(overlay.values.flags.writeable)
            self.assertFalse(overlay.selected.flags.writeable)

    def test_same_historical_cell_reused_across_overlapping_windows(self):
        overlay=self.overlay()
        inputs=historical_inputs(self.tuning,overlay)
        for first,second in zip(inputs,inputs[1:]):
            np.testing.assert_array_equal(first.values[1:],second.values[:-1])

    def test_partition_guard_blocks_reserved_splits_and_wrong_offsets(self):
        spec=FaultSpec('missing','point',.4,1729)
        for role in ('train','calibration','controller_selection','test'):
            # Only guard metadata checked; these partition values are not inspected.
            with self.assertRaises(ValueError):
                inject(self.parts[role],spec,self.stats,self.scales)
        with self.assertRaises(ValueError):
            historical_inputs(replace(self.tuning,offset=self.tuning.offset+1),self.overlay())

    def test_missing_nans_keep_natural_zeros_and_fill_causally(self):
        overlay=self.overlay(rate=1.)
        self.assertTrue(np.isnan(overlay.values[usable(self.tuning.values)]).all())
        self.assertTrue((overlay.values[self.tuning.values==0]==0).all())
        inputs=historical_inputs(self.tuning,overlay)
        filled,observed,_=self.stats.fill(inputs[0])
        self.assertTrue(np.isfinite(filled).all())
        self.assertFalse(observed.any())
        np.testing.assert_array_equal(filled,np.tile(self.stats.medians,(12,1)))

    def test_stuck_uses_strictly_earlier_clean_prefix(self):
        overlay=self.overlay('stuck','contiguous')
        for block in overlay.details['blocks']:
            a,b,s=block['start'],block['stop'],block['sensor_index']
            earlier=np.flatnonzero(usable(self.tuning.values[:a,s]))
            expected=self.tuning.values[earlier[-1],s] if len(earlier) else self.stats.medians[s]
            np.testing.assert_array_equal(overlay.values[a:b,s],np.repeat(expected,b-a))

    def test_stuck_block_at_partition_start_uses_train_median(self):
        part=Partition('tuning',self.tuning.timestamps[:12],self.tuning.values[:12],self.tuning.sensors,self.tuning.offset)
        overlay=inject(part,FaultSpec('stuck','contiguous',1.,1729,block_rows=12,episode_rows=12),self.stats,self.scales)
        np.testing.assert_array_equal(overlay.values,np.tile(self.stats.medians,(12,1)))

    def test_noisy_input_uses_training_scale_and_clips_only_below_zero(self):
        overlay=self.overlay('noisy',rate=.4)
        rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([1729,1,3,1])))
        rng.permutation(np.flatnonzero(usable(self.tuning.values).ravel()))
        normal=rng.standard_normal(self.tuning.values.shape)
        expected=np.maximum(0,self.tuning.values+normal*self.scales[None,:]*.2)
        np.testing.assert_array_equal(overlay.values[overlay.selected],expected[overlay.selected])
        self.assertEqual(overlay.details['natural_unavailable_overlap_count'],0)

    def test_noise_scale_zero_variance_falls_back_to_global(self):
        values=np.column_stack([np.full(24,10.),np.arange(24)+1.])
        train=Partition('train',self.train.timestamps[:24],values,self.train.sensors[:2],0)
        scales=noise_scales(train)
        self.assertAlmostEqual(scales[0],float(np.std(values)))
        self.assertAlmostEqual(scales[1],float(np.std(values[:,1])))

    def test_zero_rate_and_zero_noise_severity_are_no_ops(self):
        for kind,mode in [('missing','point'),('missing','contiguous'),('stuck','contiguous'),('noisy','point')]:
            np.testing.assert_array_equal(self.overlay(kind,mode,0.).values,self.tuning.values)
        np.testing.assert_array_equal(self.overlay('noisy',severity=0.).values,self.tuning.values)

    def test_all_models_get_identical_fault_realization_and_target_counts(self):
        overlay=self.overlay()
        inputs=historical_inputs(self.tuning,overlay)
        targets=np.stack([w.targets[[2,5,11]] for w in windows(self.tuning)])
        class Spy:
            def __init__(self): self.inputs=[]
            def predict(self,item):
                self.inputs.append(item)
                return np.full((12,len(item.sensors)),50.)
        first,second=Spy(),Spy()
        result=compare_models({'a':first,'b':second},inputs,targets)
        self.assertTrue(all(a is b for a,b in zip(first.inputs,second.inputs)))
        self.assertEqual(result['a'],result['b'])

    def test_invalid_fault_settings_fail(self):
        for args in [('stuck','point',.4,1),('missing','point',1.1,1),('noisy','point',-.1,1)]:
            with self.assertRaises(ValueError): FaultSpec(*args)

    def test_synthetic_ceiling_small_sensor_counts_and_low_rates(self):
        for sensors in range(1,5):
            fixture=make_synthetic(sensors=sensors)
            parts=chronological_partitions(fixture.timestamps,fixture.values,fixture.sensors)
            stats=TrainingStatistics.fit(parts['train']); scales=noise_scales(parts['train'])
            for kind,mode in [('missing','point'),('missing','contiguous'),('stuck','contiguous'),('noisy','point'),('noisy','contiguous')]:
                for rate in (.000001,.1,.2):
                    with self.subTest(sensors=sensors,kind=kind,mode=mode,rate=rate):
                        spec=FaultSpec(kind,mode,rate,1729,sampling_policy='synthetic_ceil')
                        first=inject(parts['tuning'],spec,stats,scales)
                        second=inject(parts['tuning'],spec,stats,scales)
                        self.assertGreater(first.details['changed_cell_count'],0)
                        self.assertEqual(first.details,second.details)
                        self.assertEqual(first.details['selection_count'],math.ceil(rate*first.details['selection_denominator']))

    def test_synthetic_ceiling_reports_actual_rates_and_nested_masks(self):
        previous=None
        for rate,achieved in [(.1,.25),(.2,.25),(.4,.5)]:
            overlay=self.overlay(mode='contiguous',rate=rate,sampling_policy='synthetic_ceil')
            self.assertEqual(overlay.details['achieved_selection_rate'],achieved)
            self.assertAlmostEqual(overlay.details['rounding_delta'],achieved-rate)
            if previous is not None: self.assertTrue(np.all(~previous.selected|overlay.selected))
            previous=overlay

    def test_synthetic_ceiling_zero_rate_remains_clean(self):
        for kind,mode in [('missing','point'),('missing','contiguous'),('stuck','contiguous'),('noisy','point'),('noisy','contiguous')]:
            overlay=self.overlay(kind,mode,0.,sampling_policy='synthetic_ceil')
            self.assertEqual(overlay.details['selection_count'],0)
            np.testing.assert_array_equal(overlay.values,self.tuning.values)

    def test_unknown_sampling_policy_rejected(self):
        with self.assertRaises(ValueError):
            FaultSpec('missing','point',.1,1729,sampling_policy='implicit_rounding')
