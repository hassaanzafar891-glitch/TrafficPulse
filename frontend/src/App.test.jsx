import React from 'react';
import { act, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import App from './App.jsx';
import { ApiError } from './api/client.js';
import { metadata, replay } from './test/fixtures.js';

function api() { return { health: vi.fn(async () => ({ ready: true, synthetic_only: true })), metadata: vi.fn(async () => metadata()), replay: vi.fn(async body => replay(body)) }; }
const ready = () => screen.findByRole('article', { name: '+15 min forecast' });

describe('synthetic dashboard foundation', () => {
  it('runs guide cases using backend origins and current sensor, then recovers the exact failed selection', async () => {
    const service = api(); const user = userEvent.setup(); render(<App api={service} />); await ready();
    await user.selectOptions(screen.getByLabelText('Sensor'), 'synthetic_sensor_1'); await ready();
    await user.click(screen.getByText('Demo walkthrough'));
    service.replay.mockRejectedValueOnce(new ApiError('Temporary failure', 'api', 503));
    await user.click(screen.getByRole('button', { name: 'Missing inputs' }));
    expect(await screen.findByText('The forecast request could not be completed')).toBeInTheDocument();
    expect(screen.queryByTestId('forecast-point')).not.toBeInTheDocument();
    const failed = service.replay.mock.calls.at(-1)[0];
    expect(failed).toMatchObject({sensor_id:'synthetic_sensor_1', index:321, scenario:'missing', policy:'controller'});
    await user.click(screen.getByRole('button', { name: 'Try again' })); await ready();
    expect(service.replay.mock.calls.at(-1)[0]).toEqual(failed);
    expect(screen.getByText('Fallback selected')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Clean primary intervals' })); await ready();
    expect(screen.getAllByTestId('interval-whisker')).toHaveLength(3);
    await user.click(screen.getByRole('button', { name: 'Stuck inputs' })); await ready();
    expect(screen.queryByTestId('interval-whisker')).not.toBeInTheDocument();
  });
  it('renders backend labels and all backend-provided forecasts without chart/map claims', async () => {
    const service = api(); render(<App api={service} />);
    expect(screen.getByRole('heading', { name: 'Connecting to the local workspace' })).toBeInTheDocument();
    const card = await ready(); expect(within(card).getByText('42.13')).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Backend sensor Alpha' })).toBeInTheDocument();
    expect(screen.getByRole('article', { name: '+30 min forecast' })).toBeInTheDocument();
    expect(screen.getByRole('article', { name: '+60 min forecast' })).toBeInTheDocument();
    expect(screen.getByText('2024-01-01 · 12:00:00')).toBeInTheDocument();
    expect(service.replay.mock.calls[0][0].index).toBe(321);
    expect(screen.getAllByText('Interval unavailable')).toHaveLength(3);
  });
  it('uses backend navigation links and boundaries', async () => {
    const service = api(); const user = userEvent.setup(); render(<App api={service} />); await ready();
    expect(screen.getByRole('button', { name: 'Previous replay step' })).toBeDisabled();
    await user.click(screen.getByRole('button', { name: 'Next replay step' }));
    await waitFor(() => expect(service.replay.mock.calls.at(-1)[0].index).toBe(322)); await ready();
    await user.clear(screen.getByLabelText('Replay index')); await user.type(screen.getByLabelText('Replay index'), '323');
    await user.click(screen.getByRole('button', { name: 'Go' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Next replay step' })).toBeDisabled());
    expect(service.replay.mock.calls.at(-1)[0].index).toBe(323);
  });
  it('selects only backend sensors and shows the server controller reason under faults', async () => {
    const service = api(); const user = userEvent.setup(); render(<App api={service} />); await ready();
    await user.selectOptions(screen.getByLabelText('Sensor'), 'synthetic_sensor_1'); await ready();
    expect(service.replay.mock.calls.at(-1)[0].sensor_id).toBe('synthetic_sensor_1');
    await user.selectOptions(screen.getByLabelText('Input scenario'), 'missing');
    expect(await screen.findByText('Fallback selected')).toBeInTheDocument();
    expect(screen.getByText('input_unavailable')).toBeInTheDocument();
    expect(screen.getAllByText('Interval unavailable')).toHaveLength(3);
  });
  it('shows calibration bounds only for clean fixed primary, never for faults', async () => {
    const service = api(); const user = userEvent.setup(); render(<App api={service} />); await ready();
    await user.selectOptions(screen.getByLabelText('Forecast policy'), 'xgboost');
    await waitFor(() => expect(screen.getAllByText('Calibration-derived interval')).toHaveLength(3));
    expect(screen.getAllByText('90% nominal · Not independently validated')).toHaveLength(3);
    await user.selectOptions(screen.getByLabelText('Input scenario'), 'noisy');
    await waitFor(() => expect(screen.getAllByText('Interval unavailable')).toHaveLength(3));
    expect(screen.queryByText('Calibration-derived interval')).not.toBeInTheDocument();
  });
  it('hides stale forecasts and ignores an older response after a newer selection', async () => {
    const service = api(); let older;
    service.replay.mockImplementation(body => body.scenario === 'noisy' ? new Promise(resolve => { older = resolve; }) : Promise.resolve(replay(body, body.scenario === 'stuck' ? 100 : 42.125)));
    const user = userEvent.setup(); render(<App api={service} />); await ready();
    await user.selectOptions(screen.getByLabelText('Input scenario'), 'noisy');
    expect(screen.queryByText('42.13')).not.toBeInTheDocument();expect(screen.getByText('Loading forecasts')).toBeInTheDocument();
    expect(screen.queryByRole('img', { name: /Historical readings and three forecast horizons/ })).not.toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText('Input scenario'), 'stuck'); await ready();
    expect(screen.getByText('100.00')).toBeInTheDocument();
    await act(async () => older(replay({ sensor_id: 'synthetic_sensor_0', index: 321, scenario: 'noisy', policy: 'controller' })));
    expect(screen.getByText('100.00')).toBeInTheDocument(); expect(screen.queryByText('42.13')).not.toBeInTheDocument();
  });
  it('shows empty backend availability without generating fallback forecasts', async () => {
    const service = api();service.metadata.mockResolvedValue({ ...metadata(), sensor_details: [] });
    render(<App api={service} />);expect(await screen.findByText('No replay data is available')).toBeInTheDocument();expect(service.replay).not.toHaveBeenCalled();
  });
  it('has an offline recovery state and does not substitute cached/fake predictions', async () => {
    const service = api();service.health.mockRejectedValueOnce(new ApiError('Cannot reach local backend.', 'offline'));
    const user = userEvent.setup();render(<App api={service} />);
    expect(await screen.findByText('The backend is offline')).toBeInTheDocument();expect(screen.queryByText('42.13')).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Try again' }));await ready();
  });
  it('shows API errors and clears a previous successful replay', async () => {
    const service = api();const user = userEvent.setup();render(<App api={service} />);await ready();
    service.replay.mockRejectedValueOnce(new ApiError('Request rejected.', 'api', 422));
    await user.selectOptions(screen.getByLabelText('Input scenario'), 'stuck');
    expect(await screen.findByText('The forecast request could not be completed')).toBeInTheDocument();
    expect(screen.queryByText('42.13')).not.toBeInTheDocument();
    expect(screen.queryByRole('img', { name: /Historical readings and three forecast horizons/ })).not.toBeInTheDocument();
  });
  it('keeps labeled controls keyboard navigable and the history scroll region focusable', async () => {
    const service = api(); const user = userEvent.setup(); render(<App api={service} />); await ready();
    await user.tab(); expect(screen.getByRole('link', { name: 'Skip to dashboard' })).toHaveFocus();
    act(() => screen.getByLabelText('Sensor').focus());
    await user.tab(); expect(screen.getByLabelText('Input scenario')).toHaveFocus();
    await user.tab(); expect(screen.getByLabelText('Forecast policy')).toHaveFocus();
    const region = screen.getByRole('region', { name: 'Historical readings; scroll to inspect all 12 observations' });
    expect(region).toHaveAttribute('tabindex', '0');
  });
  it('clears old forecasts when a changed sensor request fails', async () => {
    const service = api(); const user = userEvent.setup(); render(<App api={service} />); await ready();
    service.replay.mockRejectedValueOnce(new ApiError('Sensor request rejected.', 'api', 422));
    await user.selectOptions(screen.getByLabelText('Sensor'), 'synthetic_sensor_1');
    expect(await screen.findByText('The forecast request could not be completed')).toBeInTheDocument();
    expect(screen.queryByText('42.13')).not.toBeInTheDocument();
  });
  it('keeps chart identity current through overlapping sensor, scenario and replay changes', async () => {
    const service = api(); const pending = [];
    const user = userEvent.setup(); render(<App api={service} />); await ready();
    service.replay.mockImplementation(body => new Promise(resolve => pending.push({ body, resolve })));
    await user.clear(screen.getByLabelText('Replay index'));
    await user.type(screen.getByLabelText('Replay index'), '323');
    await user.click(screen.getByRole('button', { name: 'Go' }));
    expect(screen.getByRole('button', { name: 'Go' })).toBeDisabled();
    await user.selectOptions(screen.getByLabelText('Sensor'), 'synthetic_sensor_1');
    await user.selectOptions(screen.getByLabelText('Input scenario'), 'noisy');
    expect(screen.queryByRole('img', { name: /Historical readings and three forecast horizons/ })).not.toBeInTheDocument();
    expect(pending.at(-1).body).toMatchObject({ sensor_id: 'synthetic_sensor_1', scenario: 'noisy', index: 323 });
    const latest = pending.pop();
    await act(async () => latest.resolve(replay(latest.body, 123)));
    await ready();
    const chart = screen.getByRole('img', { name: /Historical readings and three forecast horizons/ });
    expect(chart).toHaveAccessibleName(/Backend sensor Beta/);
    expect(screen.getByText('123.00')).toBeInTheDocument();
    for (const old of pending.reverse()) await act(async () => old.resolve(replay(old.body, 999)));
    expect(screen.getByText('123.00')).toBeInTheDocument();
    expect(screen.queryByText('999.00')).not.toBeInTheDocument();
    expect(screen.getByRole('img', { name: /Historical readings and three forecast horizons/ })).toBe(chart);
  });
});
