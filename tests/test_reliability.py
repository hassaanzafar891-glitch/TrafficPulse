"""Synthetic-only causal quality, joint switching and selection isolation."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from trafficpulse.data import ForecastInput,Partition,TrainingStatistics,windows
from trafficpulse.faults import FaultSpec,array_hash,historical_inputs,inject,inject_controller,noise_scales
from trafficpulse.reliability import (Thresholds,assess_quality,choose_thresholds,
    controller_predictions,decision_summary,select_forecast,threshold_grid)
from trafficpulse.controller_experiment import controller_prefix_parts
from trafficpulse.synthetic import make_synthetic
from trafficpulse.xgboost_experiment import score


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        times=np.datetime64('2024-01-01T00:00')+np.arange(50)*np.timedelta64(5,'m')
        values=np.column_stack([np.arange(50)+10.,np.arange(50)+20.])
        train=Partition('train',times[:24],values[:24],('synthetic_a','synthetic_b'),0)
        self.stats=TrainingStatistics.fit(train)
        self.part=Partition('controller_selection',times[25:],values[25:].copy(),train.sensors,25)
        self.samples=list(windows(self.part));self.inputs=[w.inputs for w in self.samples]
        self.targets=np.stack([w.targets[[2,5,11]] for w in self.samples])

    def conditions(self):
        faulty=[replace(x,values=np.full(x.values.shape,np.nan)) for x in self.inputs]
        return [{'name':'clean','partition_offset':25,'inputs':self.inputs,'targets':self.targets,
                 'primary':self.targets.copy(),'fallback':self.targets.copy()},
                {'name':'missing','partition_offset':25,'inputs':faulty,'targets':self.targets,
                 'primary':self.targets+5,'fallback':self.targets.copy()}]

    def test_quality_is_causal_and_uses_only_given_history(self):
        first=assess_quality(self.inputs[0],self.stats)
        self.part.values[-1]=100000  # outside the first historical window
        self.assertEqual(first,assess_quality(self.inputs[0],self.stats))
        self.assertEqual(first['unavailable_fraction'],0)
        self.assertEqual(first['repeated_pair_fraction'],0)

    def test_quality_detects_observable_patterns_without_fault_labels(self):
        values=np.full((12,2),10.);values[0,0]=0;values[1,0]=np.nan;values[-1,1]=1000
        quality=assess_quality(replace(self.inputs[0],values=values),self.stats)
        self.assertAlmostEqual(quality['unavailable_fraction'],2/24)
        self.assertGreater(quality['repeated_pair_fraction'],.8)
        self.assertGreater(quality['abrupt_pair_fraction'],0)
        self.assertTrue(all(np.isfinite(v) for v in quality.values()))

    def test_all_missing_quality_and_empty_pair_denominator(self):
        quality=assess_quality(replace(self.inputs[0],values=np.zeros((12,2))),self.stats)
        self.assertEqual(quality,{'unavailable_fraction':1.,'repeated_pair_fraction':0.,
                                 'abrupt_pair_fraction':0.,'usable_adjacent_pair_count':0})

    def test_joint_disagreement_and_horizon_alignment(self):
        primary=np.zeros((3,2));primary[2]=100;fallback=np.array([[1,2],[3,4],[5,6]])
        result,d=select_forecast(self.inputs[0],primary,fallback,self.stats,Thresholds(.6,1))
        self.assertEqual(d['choice'],'fallback');np.testing.assert_array_equal(result,fallback)
        self.assertEqual(d['interval_validity'].split(';')[0],'unresolved')
        with self.assertRaises(ValueError):select_forecast(self.inputs[0],primary.T,fallback,self.stats,Thresholds(.6,1))

    def test_strict_threshold_and_always_primary(self):
        item=replace(self.inputs[0],values=np.full((12,2),10.))
        p=np.zeros((3,2));f=np.zeros((3,2))
        _,d=select_forecast(item,p,f,self.stats,Thresholds(0,0))
        self.assertEqual(d['choice'],'primary')
        item=replace(item,values=np.full((12,2),np.nan))
        _,d=select_forecast(item,p,f,self.stats,Thresholds(0,0,True))
        self.assertEqual(d['choice'],'primary')

    def test_model_readiness_and_abstention(self):
        finite=np.ones((3,2));bad=np.full((3,2),np.nan)
        for p,f,expected in [(bad,finite,'fallback'),(finite,bad,'primary'),(bad,bad,'abstain')]:
            _,d=select_forecast(self.inputs[0],p,f,self.stats,Thresholds(.1,1))
            self.assertEqual(d['choice'],expected)

    def test_selection_role_guard_precedes_value_access(self):
        for role in ('train','tuning','calibration','test'):
            with self.assertRaises(ValueError):choose_thresholds(SimpleNamespace(role=role),[],self.stats)

    def test_threshold_selection_reproducible_and_original_targets_required(self):
        conditions=self.conditions();original=array_hash(self.targets)
        first,candidates=choose_thresholds(self.part,conditions,self.stats)
        second,again=choose_thresholds(self.part,conditions,self.stats)
        self.assertEqual(first,Thresholds(.1,1));self.assertEqual(first,second);self.assertEqual(candidates,again)
        self.assertEqual(array_hash(self.targets),original)
        conditions[1]['targets']=self.targets+1
        with self.assertRaises(ValueError):choose_thresholds(self.part,conditions,self.stats)

    def test_selection_rejects_wrong_partition_and_origins(self):
        conditions=self.conditions();conditions[1]['partition_offset']=26
        with self.assertRaises(ValueError):choose_thresholds(self.part,conditions,self.stats)
        conditions=self.conditions();conditions[1]['inputs']=[replace(x,origin_time=x.origin_time+np.timedelta64(5,'m')) for x in self.inputs]
        with self.assertRaises(ValueError):choose_thresholds(self.part,conditions,self.stats)

    def test_clean_safety_constraint_overrides_fault_gain(self):
        conditions=self.conditions()
        # Primary is exact clean; any clean switch to a worse fallback violates
        # the frozen constraint, even if it would improve the faulty condition.
        conditions[0]['fallback']=self.targets+100
        selected,candidates=choose_thresholds(self.part,conditions,self.stats)
        self.assertTrue(selected.always_primary or np.isinf(selected.disagreement))
        self.assertTrue(any(not c['eligible'] and c['clean_mean_mae']>0 for c in candidates))
        self.assertTrue(all(c['clean_mean_mae']==0 for c in candidates if c['eligible']))

    def test_disagreement_normalizes_each_sensor_before_joint_horizon_max(self):
        stats=replace(self.stats,stds=np.array([1.,10.]))
        item=replace(self.inputs[0],values=np.ones((12,2)))
        primary=np.array([[0.,0.],[1.,10.],[2.,20.]])
        fallback=np.zeros((3,2))
        result,decision=select_forecast(item,primary,fallback,stats,Thresholds(.1,2.))
        self.assertEqual(decision['disagreement_sigma'],2.)
        self.assertEqual(decision['choice'],'primary')  # strict >, not >=
        np.testing.assert_array_equal(result,primary)
        result,decision=select_forecast(item,primary,fallback,stats,Thresholds(.1,1.9))
        self.assertEqual(decision['choice'],'fallback')
        np.testing.assert_array_equal(result,fallback)  # all horizons/sensors

    def test_missingness_threshold_counts_raw_cells_not_sensor_majority(self):
        values=np.ones((12,2));values.flat[:2]=np.nan
        p=np.ones((3,2));f=np.full((3,2),2.)
        _,d=select_forecast(replace(self.inputs[0],values=values),p,f,self.stats,Thresholds(.1,np.inf))
        self.assertEqual(d['choice'],'primary')  # 2/24 <= .1
        values.flat[2]=0  # ambiguous zero follows the frozen availability policy
        _,d=select_forecast(replace(self.inputs[0],values=values),p,f,self.stats,Thresholds(.1,np.inf))
        self.assertEqual(d['choice'],'fallback')  # 3/24 > .1
        self.assertEqual(d['reason'],'input_unavailable')

    def test_empty_masks_report_null_and_cannot_select_controller(self):
        targets=np.zeros((2,3,2));metrics=score(targets,np.ones_like(targets))
        self.assertTrue(all(m['mae'] is None and m['valid_target_count']==0 for m in metrics.values()))
        self.part.values[:]=0;self.targets[:]=0
        with self.assertRaises(ValueError):choose_thresholds(self.part,self.conditions(),self.stats)

    def test_runtime_decisions_are_independent_of_future_targets(self):
        conditions=self.conditions();c=conditions[0]
        a,decisions=controller_predictions(c['inputs'],c['primary'],c['fallback'],self.stats,Thresholds(.1,1))
        self.targets[:]=100000
        b,again=controller_predictions(c['inputs'],c['primary'],c['fallback'],self.stats,Thresholds(.1,1))
        np.testing.assert_array_equal(a,b);self.assertEqual(decisions,again)

    def test_switch_frequency_has_explicit_origin_denominator(self):
        decisions=[{'choice':c,'reason':'test'} for c in ('primary','fallback','fallback','primary')]
        summary=decision_summary(decisions)
        self.assertEqual(summary['fallback_frequency'],.5);self.assertEqual(summary['switch_count'],2)
        self.assertEqual(summary['switches_per_1000_origins'],500)
        self.assertIsNone(decision_summary([])['fallback_frequency'])
        self.assertEqual(len(threshold_grid()),17)

    def test_controller_faults_share_input_overlay_preserve_targets_and_namespace(self):
        fixture=make_synthetic(stop_rows=6451);train,part=controller_prefix_parts(fixture)
        stats=TrainingStatistics.fit(train);scales=noise_scales(train)
        hashes=[array_hash(w.targets) for w in windows(part)]
        for kind,mode in [('missing','point'),('missing','contiguous'),('stuck','contiguous'),('noisy','point'),('noisy','contiguous')]:
            spec=FaultSpec(kind,mode,.1,1729,sampling_policy='synthetic_ceil')
            first=inject_controller(part,spec,stats,scales);again=inject_controller(part,spec,stats,scales)
            self.assertEqual(first.details,again.details);self.assertEqual(first.details['rng_namespace'][1],3)
            kind_code={'missing':1,'stuck':2,'noisy':3}[kind]
            mode_code={'point':1,'contiguous':2}[mode]
            rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([1729,3,kind_code,mode_code])))
            if mode=='contiguous':
                self.assertEqual(first.details['selected_sensor_indices'],rng.permutation(4)[:1].tolist())
            else:
                eligible=np.flatnonzero((np.isfinite(part.values)&(part.values>0)).ravel())
                expected=np.sort(rng.permutation(eligible)[:int(np.ceil(.1*len(eligible)))])
                np.testing.assert_array_equal(np.flatnonzero(first.selected),expected)
            self.assertGreater(first.details['changed_cell_count'],0)
            inputs=historical_inputs(part,first,role='controller_selection')
            for a,b in zip(inputs,inputs[1:]):np.testing.assert_array_equal(a.values[1:],b.values[:-1])
            self.assertEqual(hashes,[array_hash(w.targets) for w in windows(part)])
            with self.assertRaises(ValueError):inject(part,spec,stats,scales)
        for role in ('train','tuning','calibration','test'):
            with self.assertRaises(ValueError):inject_controller(SimpleNamespace(role=role),spec,stats,scales)
        self.assertEqual((part.offset,len(part.values),len(list(windows(part)))),(6208,243,220))
