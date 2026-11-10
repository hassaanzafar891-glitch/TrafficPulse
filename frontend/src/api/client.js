import { clockOffsetMinutes } from '../utils/presentation.js';
// One read-only integration point. No forecast generation or fallback fixtures.
export class ApiError extends Error {
  constructor(message, kind = 'api', status = null) {
    super(message); this.name = 'ApiError'; this.kind = kind; this.status = status;
  }
}

const contractError = () => new ApiError('The backend returned an unsupported or invalid synthetic API contract.', 'contract');
function validateOrigin(data) {
  if (data?.contract_version !== '1.1' || data.data_origin !== 'synthetic_generated_not_real_traffic' || data.mode !== 'synthetic_historical_replay') throw contractError();
}
export function validateMetadata(data) {
  validateOrigin(data);
  if (!Array.isArray(data.sensor_details) || !Array.isArray(data.supported_horizons) || !data.replay ||
      !data.sensor_details.every(s => typeof s?.sensor_id === 'string' && typeof s.label === 'string' && s.synthetic === true) ||
      !Number.isInteger(data.replay.first_index) || !Number.isInteger(data.replay.last_index) ||
      data.replay.first_index > data.replay.last_index || data.replay.final_test_available !== false ||
      !Array.isArray(data.scenarios) || !Array.isArray(data.policies)) throw contractError();
  if (!data.scenarios.includes('clean') || !data.policies.includes('controller') ||
      (data.supported_horizons.length && ![3, 6, 12].every((h, i) => data.supported_horizons[i]?.steps === h && data.supported_horizons[i]?.minutes === h * 5))) throw contractError();
  return data;
}
export function validateReplay(data, request) {
  validateOrigin(data);
  if (data.sensor_id !== request.sensor_id || data.index !== request.index || data.scenario !== request.scenario || data.policy !== request.policy ||
      !Array.isArray(data.history) || data.history.length !== 12 || !Array.isArray(data.forecasts) || data.forecasts.length !== 3 ||
      !data.playback || !data.decision?.quality || ![3, 6, 12].every((h, i) => data.forecasts[i]?.horizon_steps === h)) throw contractError();
  if (typeof data.timestamp !== 'string' || typeof data.sensor_label !== 'string' ||
      !data.history.every((p, i) => p && typeof p.timestamp === 'string' && p.timestamp <= data.timestamp &&
        p.index === request.index - 11 + i && p.relative_minutes === (i - 11) * 5 && clockOffsetMinutes(data.timestamp, p.timestamp) === p.relative_minutes &&
        (p.value === null || Number.isFinite(p.value)) && Number.isFinite(p.filled_value) && typeof p.usable === 'boolean') ||
      !['primary', 'fallback', 'abstain'].includes(data.decision.choice) || typeof data.decision.reason !== 'string') throw contractError();
  const quality = data.decision.quality;
  if (!['unavailable_fraction', 'repeated_pair_fraction', 'abrupt_pair_fraction'].every(k => Number.isFinite(quality[k]) && quality[k] >= 0 && quality[k] <= 1) ||
      !Number.isInteger(quality.usable_adjacent_pair_count) || quality.usable_adjacent_pair_count < 0 ||
      !(data.decision.disagreement_sigma === null || Number.isFinite(data.decision.disagreement_sigma))) throw contractError();
  for (const f of data.forecasts) {
    if (!(f.prediction === null || Number.isFinite(f.prediction)) || f.horizon_minutes !== f.horizon_steps * 5 ||
        typeof f.timestamp !== 'string' || clockOffsetMinutes(data.timestamp, f.timestamp) !== f.horizon_minutes ||
        typeof f.selected_model !== 'string' || !f.interval || typeof f.interval.available !== 'boolean') throw contractError();
    const q = f.interval;
    if (q.available && !(q.status === 'calibration_derived' && request.policy === 'xgboost' && request.scenario === 'clean' &&
        f.selected_model === 'xgboost_selected' && Number.isFinite(q.lower) && Number.isFinite(q.upper) && q.lower <= q.upper &&
        Number.isFinite(q.nominal_coverage) && q.nominal_coverage > 0 && q.nominal_coverage < 1)) throw contractError();
    if (!q.available && (q.status !== 'unavailable' || q.lower !== null || q.upper !== null)) throw contractError();
  }
  for (const next of [data.playback.previous_index, data.playback.next_index]) {
    if (next !== null && (!Number.isInteger(next) || next < data.playback.first_index || next > data.playback.last_index)) throw contractError();
  }
  return data;
}

export function createApiClient({ fetchImpl = globalThis.fetch, baseUrl = import.meta.env.VITE_API_BASE_URL || '', timeoutMs = 12000 } = {}) {
  const base = baseUrl.replace(/\/$/, '');
  if (base) {
    const url = new URL(base);
    if (url.protocol !== 'http:' || !['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname) || url.username || url.password) throw new ApiError('API configuration must use localhost.', 'configuration');
  }
  async function call(path, { signal, body } = {}) {
    const controller = new AbortController();
    const abort = () => controller.abort();
    if (signal?.aborted) abort();
    signal?.addEventListener('abort', abort, { once: true });
    const timer = setTimeout(abort, timeoutMs);
    try {
      const response = await fetchImpl(`${base}${path}`, {
        method: body ? 'POST' : 'GET', signal: controller.signal,
        ...(body ? { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {}),
      });
      if (!response.ok) throw new ApiError(response.status === 503 ? 'The synthetic backend is not ready. Check its local artifact setup.' : `The API rejected this request (HTTP ${response.status}).`, 'api', response.status);
      try { return await response.json(); } catch { throw contractError(); }
    } catch (error) {
      if (signal?.aborted) throw new DOMException('Request cancelled', 'AbortError');
      if (error instanceof ApiError) throw error;
      throw new ApiError(controller.signal.aborted ? 'The backend request timed out.' : 'Cannot reach the local backend. Start FastAPI and try again.', 'offline');
    } finally {
      clearTimeout(timer); signal?.removeEventListener('abort', abort);
    }
  }
  return {
    async health(signal) { const data = await call('/health', { signal }); if (data?.synthetic_only !== true || data.ready !== true) throw new ApiError('The synthetic backend is not ready.', 'api'); return data; },
    async metadata(signal) { return validateMetadata(await call('/api/metadata', { signal })); },
    async replay(body, signal) { return validateReplay(await call('/api/replay', { body, signal }), body); },
  };
}
