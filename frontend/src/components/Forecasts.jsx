import { modelName, supportedInterval } from '../utils/presentation.js';

export function Forecasts({ data, metadata }) {
  return <section aria-labelledby="forecast-heading" className="forecast-section">
    <div className="section-heading"><div><p className="eyebrow">Forecast outlook</p><h2 id="forecast-heading">The next hour, at a glance</h2></div><span className="muted">Backend predictions · synthetic units</span></div>
    <div className="forecast-grid">{data.forecasts.map(forecast => {
      const label = metadata.supported_horizons.find(h => h.steps === forecast.horizon_steps)?.label;
      const q = forecast.interval;
      const supported = supportedInterval(data, forecast);
      return <article className="forecast-card" key={forecast.horizon_steps} aria-label={`${label} forecast`}>
        <div className="card-top"><span className="horizon">{label}</span><span className="model-chip">{modelName(forecast.selected_model)}</span></div>
        <p className="prediction">{Number.isFinite(forecast.prediction) ? forecast.prediction.toFixed(2) : 'Unavailable'}<span>synthetic speed units</span></p>
        <p className="forecast-time">For {forecast.timestamp.replace('T', ' · ')}</p>
        <div className="interval-detail">{supported ? <><span className="interval-label">Calibration-derived interval</span><strong>{q.lower.toFixed(2)} – {q.upper.toFixed(2)}</strong><small>{(q.nominal_coverage * 100).toFixed(0)}% nominal · Not independently validated</small></> : <><span className="interval-label">Interval unavailable</span><small>{data.scenario !== 'clean' ? 'Coverage under simulated faults is unvalidated.' : 'Coverage for controller or fallback predictions is unvalidated.'}</small></>}</div>
      </article>;
    })}</div>
  </section>;
}
