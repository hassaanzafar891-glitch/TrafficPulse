import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe,expect,it } from 'vitest';
import { TrafficChart } from './TrafficChart.jsx';
import { replay,request } from '../test/fixtures.js';

describe('accessible historical and forecast visualization', () => {
  it('renders separate observed and predicted markers and exposes gap details by keyboard selector', async () => {
    const data=replay();render(<TrafficChart data={data} />);
    expect(screen.getAllByTestId('history-point')).toHaveLength(11);
    expect(screen.getAllByTestId('forecast-point')).toHaveLength(3);
    expect(screen.getAllByTestId('gap-marker')).toHaveLength(1);
    const user=userEvent.setup();await user.selectOptions(screen.getByLabelText('Inspect chart point'),'history-310');
    expect(screen.getByText('Unavailable by policy — no speed plotted')).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText('Inspect chart point'),'forecast-3');
    expect(screen.getByText('42.13 synthetic speed units')).toBeInTheDocument();
    expect(screen.getByText(/Joint primary/)).toBeInTheDocument();
  });
  it('updates hover details only from the selected backend point', () => {
    const { container }=render(<TrafficChart data={replay()} />);
    fireEvent.pointerEnter(container.querySelector('[data-point-id="forecast-6"]'));
    expect(screen.getByText('Forecast +30 min')).toBeInTheDocument();
    expect(screen.getByText('43.13 synthetic speed units')).toBeInTheDocument();
  });
  it('draws three permitted primary whiskers and none for faults or switching', () => {
    const data=replay({ ...request,policy:'xgboost' });const { rerender }=render(<TrafficChart data={data} />);
    expect(screen.getAllByTestId('interval-whisker')).toHaveLength(3);
    data.scenario='missing';rerender(<TrafficChart data={{...data}} />);
    expect(screen.queryByTestId('interval-whisker')).not.toBeInTheDocument();
  });
  it('supports touch inspection and keyboard focus without moving focus on pointer events', async () => {
    const { container } = render(<TrafficChart data={replay()} />);
    const user = userEvent.setup(); await user.tab();
    const selector = screen.getByLabelText('Inspect chart point');
    expect(selector).toHaveFocus();
    fireEvent.pointerDown(container.querySelector('[data-point-id="forecast-12"]'), { pointerType: 'touch' });
    expect(screen.getByText('Forecast +60 min')).toBeInTheDocument();
    expect(selector).toHaveValue('forecast-12'); expect(selector).toHaveFocus();
    await user.selectOptions(selector, 'forecast-3');
    expect(screen.getByText('Forecast +15 min')).toBeInTheDocument();
    expect(selector).toHaveFocus();
  });
});
