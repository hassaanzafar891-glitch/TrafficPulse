// TEST-ONLY HTTP fixtures. Never imported by the dashboard or used as forecasts.
export function metadata() {
  return { contract_version: '1.1', data_origin: 'synthetic_generated_not_real_traffic', mode: 'synthetic_historical_replay', units: 'synthetic_speed_units',
    sensor_details: [{ sensor_id: 'synthetic_sensor_0', label: 'Backend sensor Alpha', synthetic: true }, { sensor_id: 'synthetic_sensor_1', label: 'Backend sensor Beta', synthetic: true }],
    sensors: ['synthetic_sensor_0', 'synthetic_sensor_1'], models: ['xgboost_selected', 'seasonal_historical_average'],
    supported_horizons: [{ steps: 3, minutes: 15, label: '+15 min' }, { steps: 6, minutes: 30, label: '+30 min' }, { steps: 12, minutes: 60, label: '+60 min' }],
    replay: { available: true, first_index: 321, last_index: 323, first_timestamp: '2024-01-01T12:00:00', last_timestamp: '2024-01-01T12:10:00', final_test_available: false },
    scenarios: ['clean', 'missing', 'noisy', 'stuck'], policies: ['controller', 'xgboost'], geography: { available: false } };
}
export const request = { sensor_id: 'synthetic_sensor_0', index: 321, scenario: 'clean', policy: 'controller' };
export function replay(input = request, prediction = 42.125) {
  const fallback = input.scenario === 'missing' && input.policy === 'controller';
  const supported = input.policy === 'xgboost' && input.scenario === 'clean';
  const index = input.index;
  return { contract_version: '1.1', data_origin: 'synthetic_generated_not_real_traffic', mode: 'synthetic_historical_replay', units: 'synthetic_speed_units',
    ...input, sensor_label: input.sensor_id === 'synthetic_sensor_0' ? 'Backend sensor Alpha' : 'Backend sensor Beta',
    timestamp: '2024-01-01T12:00:00', joint_decision_scope: 'all sensors and horizons',
    decision: { choice: fallback ? 'fallback' : 'primary', reason: fallback ? 'input_unavailable' : 'within_thresholds',
      quality: { unavailable_fraction: fallback ? .4 : .02, repeated_pair_fraction: 0, abrupt_pair_fraction: 0, usable_adjacent_pair_count: 40 }, disagreement_sigma: .1 },
    playback: { previous_index: index > 321 ? index - 1 : null, next_index: index < 323 ? index + 1 : null, first_index: 321, last_index: 323 },
    history: Array.from({ length: 12 }, (_, i) => ({ index: index - 11 + i, relative_minutes: (i - 11) * 5, timestamp: i === 11 ? '2024-01-01T12:00:00' : `2024-01-01T11:${String((i + 1) * 5).padStart(2, '0')}:00`, value: i === 0 ? null : 40, usable: i !== 0, filled_value: 40 })),
    forecasts: [3, 6, 12].map((h, i) => ({ horizon_steps: h, horizon_minutes: h * 5, timestamp: `2024-01-01T${h === 12 ? '13:00' : h === 6 ? '12:30' : '12:15'}:00`, prediction: prediction + i,
      selected_model: fallback ? 'seasonal_historical_average' : 'xgboost_selected',
      interval: { status: supported ? 'calibration_derived' : 'unavailable', available: supported, lower: supported ? prediction + i - 2 : null, upper: supported ? prediction + i + 2 : null, nominal_coverage: supported ? .9 : null, reason: supported ? 'clean_primary_calibration_not_independent_validation' : 'unvalidated' } })), fault_summary: null };
}
