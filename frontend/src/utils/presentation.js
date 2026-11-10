const names = { xgboost_selected: 'XGBoost', seasonal_historical_average: 'Seasonal average' };
export const modelName = value => names[value] || value;

export function supportedInterval(data, forecast) {
  const q = forecast.interval;
  return Number.isFinite(forecast.prediction) && data.policy === 'xgboost' && data.scenario === 'clean' && forecast.selected_model === 'xgboost_selected' &&
    q?.available === true && q.status === 'calibration_derived' && Number.isFinite(q.lower) && Number.isFinite(q.upper) &&
    q.lower <= q.upper && Number.isFinite(q.nominal_coverage) && q.nominal_coverage > 0 && q.nominal_coverage < 1;
}

// Arithmetic on naive clock components only. Never localize/display these as UTC.
function clockValue(text) {
  if (typeof text !== 'string' || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/.test(text)) return null;
  const [y, m, d, h, minute, second] = text.split(/[-T:]/).map(Number);
  const value = Date.UTC(y, m - 1, d, h, minute, second); const check = new Date(value);
  return check.getUTCFullYear() === y && check.getUTCMonth() === m - 1 && check.getUTCDate() === d &&
    check.getUTCHours() === h && check.getUTCMinutes() === minute && check.getUTCSeconds() === second ? value : null;
}
export function clockOffsetMinutes(origin, timestamp) {
  const a = clockValue(origin); const b = clockValue(timestamp);
  return a === null || b === null ? null : (b - a) / 60000;
}
