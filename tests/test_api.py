"""In-process API checks using real synthetic models; no real dataset access."""
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from threadpoolctl import threadpool_limits

from trafficpulse.api.app import create_app
from trafficpulse.api.service import ReplayService
from trafficpulse.data import ForecastInput


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limits=threadpool_limits(limits=4);cls.limits.__enter__()
        root=Path(__file__).resolve().parents[1]
        # Fail if new API code tries to fit XGBoost or retune/recalibrate.
        with patch('xgboost.XGBRegressor.fit',side_effect=AssertionError('Refit forbidden')),patch('trafficpulse.reliability.choose_thresholds',side_effect=AssertionError('Retuning forbidden')),patch('trafficpulse.uncertainty.SplitConformal.fit',side_effect=AssertionError('Recalibration forbidden')):
            cls.engine=ReplayService(root)
        cls.client=TestClient(create_app(lambda:cls.engine));cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None,None,None);cls.limits.__exit__(None,None,None)

    def request(self,**kwargs):
        return self.client.post('/api/replay',json={'sensor_id':'synthetic_sensor_0','index':6300,**kwargs})

    def test_health_metadata_and_no_restricted_identifiers(self):
        self.assertEqual(self.client.get('/health').json(),{'status':'ok','synthetic_only':True,'ready':True})
        response=self.client.get('/api/metadata');self.assertEqual(response.status_code,200)
        data=response.json();self.assertEqual(data['horizons_steps'],[3,6,12])
        self.assertEqual(data['sensors'],[f'synthetic_sensor_{i}' for i in range(4)])
        self.assertFalse(data['replay']['final_test_available'])
        self.assertNotIn('E:',response.text);self.assertNotIn('latitude',response.text)

    def test_clean_replay_uses_real_models_and_causal_horizon_alignment(self):
        response=self.request();self.assertEqual(response.status_code,200)
        result=response.json();self.assertEqual(len(result['history']),12)
        local=6300-6208
        inputs=ForecastInput(self.engine.readings[local-11:local+1],self.engine.times[local],self.engine.sensors)
        expected=self.engine.primary.predict(inputs)[:,0]
        np.testing.assert_array_equal([x['prediction'] for x in result['forecasts']],expected)
        self.assertEqual([x['horizon_minutes'] for x in result['forecasts']],[15,30,60])
        for reading in result['history']:self.assertLessEqual(reading['timestamp'],result['timestamp'])
        for forecast in result['forecasts']:
            self.assertEqual(np.datetime64(forecast['timestamp'])-np.datetime64(result['timestamp']),np.timedelta64(forecast['horizon_minutes'],'m'))
            self.assertFalse(forecast['interval']['available'])
        self.assertNotIn('targets',result);self.assertNotIn('actual',result)

    def test_missing_inputs_trigger_fallback_without_intervals(self):
        result=self.request(scenario='missing').json()
        self.assertEqual(result['decision']['choice'],'fallback')
        self.assertEqual(result['decision']['reason'],'input_unavailable')
        self.assertGreater(result['decision']['quality']['unavailable_fraction'],.1)
        for f in result['forecasts']:
            self.assertEqual(f['selected_model'],'seasonal_historical_average')
            self.assertFalse(f['interval']['available']);self.assertIsNone(f['interval']['lower'])

    def test_noisy_and_stuck_inputs_are_deterministic_and_descriptive(self):
        for scenario in ('noisy','stuck'):
            first=self.request(scenario=scenario);second=self.request(scenario=scenario)
            self.assertEqual(first.status_code,200);self.assertEqual(first.json(),second.json())
            result=first.json();self.assertGreater(result['fault_summary']['changed_cell_count'],0)
            self.assertTrue(all(not x['interval']['available'] for x in result['forecasts']))
            self.assertTrue(all(np.isfinite(x['prediction']) for x in result['forecasts']))
            if scenario=='stuck':self.assertGreater(result['decision']['quality']['repeated_pair_fraction'],.3)

    def test_fixed_primary_calibration_is_separate_from_controller_and_faults(self):
        result=self.request(policy='xgboost').json()
        self.assertEqual(result['decision']['reason'],'fixed_xgboost_policy')
        for i,forecast in enumerate(result['forecasts']):
            interval=forecast['interval'];self.assertTrue(interval['available'])
            self.assertEqual(interval['nominal_coverage'],.9)
            self.assertAlmostEqual(interval['upper']-interval['lower'],2*self.engine.widths[i])
            self.assertIn('not_independent_validation',interval['reason'])
        faulty=self.request(policy='xgboost',scenario='missing').json()
        self.assertTrue(all(not f['interval']['available'] for f in faulty['forecasts']))

    def test_invalid_inputs_have_meaningful_safe_errors(self):
        for body in ({'sensor_id':'bad','index':6300},{'sensor_id':'synthetic_sensor_99','index':6300},
                     {'sensor_id':'synthetic_sensor_0','index':6300,'timestamp':'2024-01-22T21:00'},
                     {'sensor_id':'synthetic_sensor_0'}, {'sensor_id':'synthetic_sensor_0','index':6300.5},
                     {'sensor_id':'synthetic_sensor_0','index':6300,'dataset_path':'E:/private/data.h5'},
                     {'sensor_id':'synthetic_sensor_0','index':6300,'scenario':'invalid'}):
            response=self.client.post('/api/replay',json=body)
            self.assertIn(response.status_code,(404,422));self.assertNotIn('E:',response.text)
            self.assertNotIn('Traceback',response.text)

    def test_timestamp_and_index_equivalence_and_exact_boundaries(self):
        metadata=self.client.get('/api/metadata').json()['replay']
        for index,key in [(6219,'first_timestamp'),(6438,'last_timestamp')]:
            a=self.request(index=index)
            b=self.client.post('/api/replay',json={'sensor_id':'synthetic_sensor_0','timestamp':metadata[key]})
            self.assertEqual(a.status_code,200);self.assertEqual(a.json(),b.json())
        for index in (6208,6218,6439,6451,8063):self.assertEqual(self.request(index=index).status_code,422)
        for timestamp in ('2024-01-22T21:01','2024-02-30T00:00','2024-01-22T21:00Z'):
            response=self.client.post('/api/replay',json={'sensor_id':'synthetic_sensor_0','timestamp':timestamp})
            self.assertEqual(response.status_code,422)

    def test_future_values_cannot_change_any_replay_response(self):
        original=self.engine.readings
        try:
            for scenario in ('clean','missing','noisy','stuck'):
                before=self.request(scenario=scenario).json()
                changed=original.copy();changed[6300-6208+1:]=123456.
                self.engine.readings=changed
                after=self.request(scenario=scenario).json()
                self.assertEqual(before,after)
                self.engine.readings=original
        finally:self.engine.readings=original

    def test_service_unavailable_and_inference_errors_hide_internal_paths(self):
        def failed():raise FileNotFoundError('E:/private/secret.json')
        with TestClient(create_app(failed)) as client:
            self.assertEqual(client.get('/health').status_code,503)
            response=client.get('/api/metadata');self.assertEqual(response.status_code,503)
            self.assertNotIn('E:',response.text)
        with patch.object(self.engine,'replay',side_effect=RuntimeError('E:/private/secret.json')):
            response=self.request();self.assertEqual(response.status_code,503);self.assertNotIn('E:',response.text)

    def test_joint_decision_is_shared_by_sensor_requests_and_no_mutation(self):
        original=self.engine.readings.copy()
        for scenario in ('clean','missing','noisy','stuck'):
            first=self.request(scenario=scenario).json()
            second=self.request(scenario=scenario,sensor_id='synthetic_sensor_1').json()
            self.assertEqual(first['decision'],second['decision'])
            self.assertEqual(first['fault_summary'],second['fault_summary'])
            self.assertEqual([f['selected_model'] for f in first['forecasts']],
                             [f['selected_model'] for f in second['forecasts']])
        np.testing.assert_array_equal(self.engine.readings,original)

    def test_interval_provenance_and_manifest_tampering_are_rejected(self):
        from tempfile import TemporaryDirectory
        from trafficpulse.api.service import verify_manifest
        with TemporaryDirectory() as location:
            directory=Path(location);(directory/'safe.json').write_text('{}')
            with self.assertRaises(ValueError):
                verify_manifest(directory,[{'name':'../safe.json','sha256':'wrong'}],{'safe.json'})
            with self.assertRaises(ValueError):
                verify_manifest(directory,[{'name':'safe.json','sha256':'wrong'}],{'safe.json'})

    def test_public_response_contract_and_interval_policy_matrix(self):
        allowed={'contract_version','data_origin','mode','units','sensor_id','sensor_label','index','timestamp','scenario','policy',
                 'joint_decision_scope','decision','history','forecasts','fault_summary','playback'}
        for policy in ('controller','xgboost'):
            for scenario in ('clean','missing','noisy','stuck'):
                response=self.request(policy=policy,scenario=scenario)
                self.assertEqual(response.status_code,200)
                data=response.json();self.assertEqual(set(data),allowed)
                self.assertEqual(set(data['history'][0]),{'index','relative_minutes','timestamp','value','usable','filled_value'})
                for f in data['forecasts']:
                    supported=policy=='xgboost' and scenario=='clean'
                    self.assertEqual(f['interval']['available'],supported)
                    if not supported:
                        self.assertIsNone(f['interval']['lower']);self.assertIsNone(f['interval']['upper'])
                        self.assertIsNone(f['interval']['nominal_coverage'])
                for forbidden in ('calibration_residuals','target_values','latitude','longitude','tracking.db'):
                    self.assertNotIn(forbidden,response.text)
        self.assertEqual(self.request(index=True).status_code,422)
        local=6300-6208
        item=ForecastInput(self.engine.readings[local-11:local+1],self.engine.times[local],self.engine.sensors)
        expected=self.engine.fallback.predict(item)[[2,5,11],0]
        actual=self.request(scenario='missing').json()
        np.testing.assert_array_equal([f['prediction'] for f in actual['forecasts']],expected)

    def test_startup_launcher_enforces_local_single_worker_and_thread_caps(self):
        import importlib.util
        import os
        from unittest.mock import MagicMock
        path=Path(__file__).resolve().parents[1]/'scripts/run_backend.py'
        spec=importlib.util.spec_from_file_location('backend_launcher',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        server=MagicMock()
        def inspect_start(*args,**kwargs):
            self.assertEqual(kwargs,{'factory':True,'host':'127.0.0.1','port':8000,'workers':1})
            self.assertEqual(args,('trafficpulse.api.app:create_app',))
            for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMEXPR_MAX_THREADS'):
                self.assertEqual(os.environ[name],'4')
        server.run.side_effect=inspect_start
        with patch.dict('sys.modules',{'uvicorn':server}),patch.dict(os.environ,{}):
            module.main()
        server.run.assert_called_once()

    def test_dashboard_metadata_labels_options_and_geography_contract(self):
        metadata=self.client.get('/api/metadata').json()
        self.assertEqual(metadata['contract_version'],'1.1')
        self.assertEqual([s['sensor_id'] for s in metadata['sensor_details']],metadata['sensors'])
        self.assertEqual([s['label'] for s in metadata['sensor_details']],[f'Synthetic sensor {i+1}' for i in range(4)])
        self.assertTrue(all(s['synthetic'] for s in metadata['sensor_details']))
        self.assertEqual(metadata['supported_horizons'],[
            {'steps':3,'minutes':15,'label':'+15 min'}, {'steps':6,'minutes':30,'label':'+30 min'},
            {'steps':12,'minutes':60,'label':'+60 min'}])
        self.assertEqual(metadata['geography'],{'available':False,'reason':'no_geographic_coordinates_provided'})
        self.assertEqual(metadata['replay']['origin_count'],220)
        self.assertTrue(metadata['replay']['available']);self.assertFalse(metadata['replay']['final_test_available'])
        self.assertEqual(metadata['policies'],['controller','xgboost'])
        self.assertEqual(metadata['fault_rates'],[.1,.2,.4]);self.assertEqual(metadata['fault_seeds'],[1729,2718,31415])

    def test_chart_offsets_playback_bounds_and_all_horizon_timestamps(self):
        for index in (6219,6300,6438):
            for scenario in ('clean','missing','noisy','stuck'):
                data=self.request(index=index,scenario=scenario).json()
                self.assertEqual(data['contract_version'],'1.1');self.assertEqual(data['sensor_label'],'Synthetic sensor 1')
                self.assertEqual([p['index'] for p in data['history']],list(range(index-11,index+1)))
                self.assertEqual([p['relative_minutes'] for p in data['history']],list(range(-55,1,5)))
                origin=np.datetime64(data['timestamp'])
                for point in data['history']:
                    self.assertEqual(np.datetime64(point['timestamp'])-origin,np.timedelta64(point['relative_minutes'],'m'))
                self.assertEqual(data['playback'],{'previous_index':index-1 if index>6219 else None,
                                                 'next_index':index+1 if index<6438 else None,'first_index':6219,'last_index':6438})
                self.assertEqual([f['horizon_steps'] for f in data['forecasts']],[3,6,12])
                for point in data['forecasts']:
                    self.assertEqual(np.datetime64(point['timestamp'])-origin,np.timedelta64(point['horizon_minutes'],'m'))

    def test_additive_contract_preserves_phase4a_predictions_and_decisions(self):
        import json
        root=Path(__file__).resolve().parents[1]
        prior=json.loads((root/'reports/generated/phase4a_api_smoke.json').read_text())
        for policy,key in [('controller','example'),('xgboost','fixed_xgboost_example')]:
            data=self.request(policy=policy).json()
            for extra in ('contract_version','sensor_label','playback'):data.pop(extra)
            for point in data['history']:
                point.pop('index');point.pop('relative_minutes')
            for point in data['forecasts']:point['interval'].pop('status')
            self.assertEqual(data,prior[key])

    def test_typed_openapi_nested_contract_and_interval_invariants(self):
        from pydantic import ValidationError
        from trafficpulse.api.schemas import Interval
        schema=self.client.get('/openapi.json').json()
        self.assertEqual(schema['info']['version'],'1.1')
        components=schema['components']['schemas']
        self.assertTrue(components['MetadataResponse']['properties']['replay']['$ref'].endswith('/ReplayAvailability'))
        self.assertTrue(components['MetadataResponse']['properties']['sensor_details']['items']['$ref'].endswith('/SensorDetail'))
        self.assertFalse(components['ReplayResponse']['additionalProperties'])
        self.assertEqual(components['ReplayResponse']['properties']['history']['minItems'],12)
        for payload in ({'status':'unavailable','available':False,'lower':1.,'reason':'unvalidated'},
                        {'status':'calibration_derived','available':True,'reason':'missing_bounds'},
                        {'status':'calibration_derived','available':True,'lower':2.,'upper':1.,'nominal_coverage':.9,'reason':'invalid'}):
            with self.assertRaises(ValidationError):Interval(**payload)

    def test_interval_status_and_invalid_dashboard_options(self):
        for policy in ('controller','xgboost'):
            for scenario in ('clean','missing','noisy','stuck'):
                data=self.request(policy=policy,scenario=scenario).json()
                expected='calibration_derived' if policy=='xgboost' and scenario=='clean' else 'unavailable'
                self.assertTrue(all(f['interval']['status']==expected for f in data['forecasts']))
        for options in ({'fault_rate':.3},{'seed':999},{'policy':'seasonal'}, {'horizon':15},{'index':'6300'}):
            self.assertEqual(self.request(**options).status_code,422)
