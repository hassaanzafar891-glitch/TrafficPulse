"""TEST-ONLY API stub: no trained model, real forecasts or artifact dependency."""
from datetime import datetime, timedelta
from types import SimpleNamespace
import unittest

from fastapi.testclient import TestClient

from trafficpulse.api.app import create_app
from trafficpulse.api.service import ReplayInputError


class OfflineApiTests(unittest.TestCase):
    def test_unavailable_startup_hides_paths_and_has_no_substitute_forecasts(self):
        def unavailable():
            raise FileNotFoundError('private-test-only-path')
        with TestClient(create_app(unavailable)) as client:
            self.assertEqual(client.get('/health').status_code, 503)
            response = client.get('/api/metadata')
            self.assertEqual(response.status_code, 503)
            self.assertNotIn('private-test-only-path', response.text)

    def test_schema_horizons_and_fault_requests_without_a_model(self):
        calls = []
        def replay(body):
            calls.append(body.model_dump())
            origin = datetime(2024, 1, 1, 1)
            stamp = lambda t: t.isoformat()
            # Every prediction is unavailable: no fabricated model output.
            return dict(contract_version='1.1', data_origin='synthetic_generated_not_real_traffic',
                mode='synthetic_historical_replay', units='synthetic_speed_units',
                sensor_id=body.sensor_id, sensor_label='TEST ONLY stub sensor', index=body.index,
                timestamp=stamp(origin), scenario=body.scenario, policy=body.policy,
                joint_decision_scope='TEST ONLY schema fixture',
                decision=dict(choice='abstain', reason='both_unavailable', interval_validity='unvalidated',
                    quality=dict(unavailable_fraction=1, repeated_pair_fraction=0, abrupt_pair_fraction=0,
                                 usable_adjacent_pair_count=0), disagreement_sigma=None),
                history=[dict(index=body.index-11+i, relative_minutes=(i-11)*5,
                    timestamp=stamp(origin+timedelta(minutes=(i-11)*5)), value=None, usable=False,
                    filled_value=0) for i in range(12)],
                forecasts=[dict(horizon_steps=h, horizon_minutes=h*5,
                    timestamp=stamp(origin+timedelta(minutes=h*5)), prediction=None,
                    selected_model='unavailable_test_stub', interval=dict(status='unavailable',
                    available=False, reason='test_stub_has_no_calibration')) for h in (3,6,12)],
                fault_summary=None, playback=dict(first_index=11, last_index=11,
                                                   previous_index=None, next_index=None))
        with TestClient(create_app(lambda: SimpleNamespace(replay=replay))) as client:
            self.assertTrue(client.get('/health').json()['synthetic_only'])
            for scenario in ('clean', 'missing', 'noisy', 'stuck'):
                response = client.post('/api/replay', json=dict(sensor_id='synthetic_sensor_0', index=11, scenario=scenario))
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual([f['horizon_minutes'] for f in data['forecasts']], [15,30,60])
                self.assertTrue(all(f['prediction'] is None and not f['interval']['available'] for f in data['forecasts']))
                self.assertNotIn('targets', data)
            self.assertEqual([c['scenario'] for c in calls], ['clean','missing','noisy','stuck'])

    def test_invalid_inputs_are_rejected_before_stub_inference(self):
        def forbidden(body):
            raise AssertionError('Invalid request reached inference')
        with TestClient(create_app(lambda: SimpleNamespace(replay=forbidden))) as client:
            for body in ({'sensor_id':'real_sensor','index':11}, {'sensor_id':'synthetic_sensor_0','index':True},
                         {'sensor_id':'synthetic_sensor_0','index':11,'scenario':'unknown'},
                         {'sensor_id':'synthetic_sensor_0','index':11,'targets':[1]}):
                self.assertEqual(client.post('/api/replay',json=body).status_code,422)

    def test_input_and_internal_errors_remain_sanitized(self):
        def bad_input(body): raise ReplayInputError('Synthetic test boundary',422)
        with TestClient(create_app(lambda: SimpleNamespace(replay=bad_input))) as client:
            self.assertEqual(client.post('/api/replay',json={'sensor_id':'synthetic_sensor_0','index':11}).status_code,422)
        def failed(body): raise RuntimeError('private-test-only-path')
        with TestClient(create_app(lambda: SimpleNamespace(replay=failed))) as client:
            response=client.post('/api/replay',json={'sensor_id':'synthetic_sensor_0','index':11})
            self.assertEqual(response.status_code,503); self.assertNotIn('private-test-only-path',response.text)


if __name__ == '__main__': unittest.main()
