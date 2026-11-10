import { supportedInterval } from '../utils/presentation.js';

// Display mapping only. No imputation, resampling, smoothing or forecasting.
export function mapSeries(data) {
  const history = data.history.map(p => ({ id: `history-${p.index}`, kind: 'history', x: p.relative_minutes,
    timestamp: p.timestamp, raw: p.value, y: p.usable === true && Number.isFinite(p.value) && p.value > 0 ? p.value : null }));
  const forecasts = data.forecasts.map(p => ({ id: `forecast-${p.horizon_steps}`, kind: 'forecast', x: p.horizon_minutes,
    timestamp: p.timestamp, y: Number.isFinite(p.prediction) ? p.prediction : null, model: p.selected_model,
    interval: supportedInterval(data, p) ? { lower: p.interval.lower, upper: p.interval.upper, nominal: p.interval.nominal_coverage } : null }));
  const segments = []; let run = [];
  for (const point of history) {
    if (point.y === null) { if (run.length) segments.push(run); run = []; }
    else run.push(point);
  }
  if (run.length) segments.push(run);
  const values = [...history, ...forecasts].flatMap(p => [p.y, ...(p.interval ? [p.interval.lower, p.interval.upper] : [])]).filter(Number.isFinite);
  let domain = null;
  if (values.length) {
    const low = Math.min(...values); const high = Math.max(...values);
    const padding = Math.max((high - low) * .12, 1);
    domain = [low - padding, high + padding];
  }
  return { history, forecasts, segments, points: [...history, ...forecasts], domain };
}
