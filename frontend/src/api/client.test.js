import { describe, expect, it, vi } from 'vitest';
import { ApiError, createApiClient, validateMetadata, validateReplay } from './client.js';
import { metadata, replay, request } from '../test/fixtures.js';

describe('centralized synthetic API client', () => {
  it('requests the real endpoint shapes and sends no target/forecast data', async () => {
    const fetchImpl = vi.fn(async (url, options) => ({ ok: true, json: async () => url === '/health' ? { ready: true, synthetic_only: true } : url === '/api/metadata' ? metadata() : replay(JSON.parse(options.body)) }));
    const api = createApiClient({ fetchImpl });
    await api.health(); await api.metadata(); await api.replay(request);
    expect(fetchImpl.mock.calls.map(c => c[0])).toEqual(['/health', '/api/metadata', '/api/replay']);
    expect(JSON.parse(fetchImpl.mock.calls[2][1].body)).toEqual(request);
  });
  it('rejects remote API configuration and unsupported/non-synthetic contracts', () => {
    expect(() => createApiClient({ baseUrl: 'https://example.com' })).toThrow('localhost');
    expect(() => validateMetadata({ ...metadata(), contract_version: '2' })).toThrow(ApiError);
    expect(() => validateMetadata({ ...metadata(), data_origin: 'metr_la' })).toThrow(ApiError);
  });
  it('accepts an empty synthetic sensor list for the explicit empty state', () => {
    expect(validateMetadata({ ...metadata(), sensor_details: [] }).sensor_details).toEqual([]);
  });
  it('rejects mismatched origins and unvalidated switched interval bounds', () => {
    expect(() => validateReplay({ ...replay(), index: 999 }, request)).toThrow(ApiError);
    const unsafe = replay(); unsafe.forecasts[0].interval = { available: true, status: 'calibration_derived', lower: 1, upper: 2, nominal_coverage: .9 };
    expect(() => validateReplay(unsafe, request)).toThrow(ApiError);
    const fixed = { ...request, policy: 'xgboost' };
    expect(validateReplay(replay(fixed), fixed)).toBeTruthy();
  });
  it('does not show predictions if forecast alignment or navigation is malformed', () => {
    const value = replay(); value.forecasts[0].horizon_minutes = 60;
    expect(() => validateReplay(value, request)).toThrow(ApiError);
    const badLink = replay(); badLink.playback.next_index = 9999;
    expect(() => validateReplay(badLink, request)).toThrow(ApiError);
  });
  it('turns malformed history/forecast/quality objects into a contract error before rendering', () => {
    for (const mutate of [value => { value.history[0] = null; },
      value => { value.forecasts[0] = null; }, value => { value.history[0].timestamp = undefined; },
      value => { value.decision.quality.unavailable_fraction = undefined; },
      value => { value.history[0].index = 999; }]) {
      const value = replay(); mutate(value);
      expect(() => validateReplay(value, request)).toThrow(ApiError);
    }
    expect(() => validateMetadata({ ...metadata(), sensor_details: [null] })).toThrow(ApiError);
  });
  it('rejects clock/offset mismatches before they can reach a chart', () => {
    const future=replay();future.forecasts[0].timestamp='2024-01-01T12:30:00';
    expect(() => validateReplay(future,request)).toThrow(ApiError);
    const history=replay();history.history[0].timestamp='2024-01-01T11:10:00';
    expect(() => validateReplay(history,request)).toThrow(ApiError);
  });
  it('distinguishes HTTP errors from network/offline failures', async () => {
    const rejected = createApiClient({ fetchImpl: async () => ({ ok: false, status: 503 }) });
    await expect(rejected.health()).rejects.toMatchObject({ kind: 'api', status: 503 });
    const offline = createApiClient({ fetchImpl: async () => { throw new TypeError('fetch failed'); } });
    await expect(offline.health()).rejects.toMatchObject({ kind: 'offline' });
  });
  it('classifies invalid JSON as a contract error, without exposing response internals', async () => {
    const api = createApiClient({ fetchImpl: async () => ({ ok: true, json: async () => { throw new Error('private path'); } }) });
    await expect(api.metadata()).rejects.toMatchObject({ kind: 'contract' });
  });
  it('cancels a request when its owner aborts', async () => {
    const controller = new AbortController();
    const api = createApiClient({ fetchImpl: (_, options) => new Promise((_, reject) => options.signal.addEventListener('abort', () => reject(new DOMException('cancelled', 'AbortError')))) });
    const pending = api.health(controller.signal); controller.abort();
    await expect(pending).rejects.toMatchObject({ name: 'AbortError' });
  });
  it('times out without allowing a request to hang indefinitely', async () => {
    vi.useFakeTimers();
    try {
      const api = createApiClient({ timeoutMs: 10, fetchImpl: (_, options) => new Promise((_, reject) => options.signal.addEventListener('abort', () => reject(new DOMException('timeout', 'AbortError')))) });
      const pending = expect(api.health()).rejects.toMatchObject({ kind: 'offline', message: 'The backend request timed out.' });
      await vi.advanceTimersByTimeAsync(11); await pending;
    } finally { vi.useRealTimers(); }
  });
});
