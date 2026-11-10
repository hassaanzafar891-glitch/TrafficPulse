import { modelName } from '../utils/presentation.js';

const explanations = {
  within_thresholds: 'The historical inputs and model disagreement remain within the controller thresholds.',
  input_unavailable: 'The historical unavailable-reading fraction exceeded the selected threshold.',
  model_disagreement: 'The primary and fallback forecasts exceeded the selected disagreement threshold.',
  fixed_xgboost_policy: 'Fixed XGBoost is selected. The controller does not switch the served forecast in this mode.',
  primary_unavailable: 'The primary forecast is unavailable; the ready fallback was selected.',
  fallback_unavailable: 'The fallback is unavailable; the ready primary was selected.',
  both_unavailable: 'Both forecasting models are unavailable.',
  always_primary: 'The fixed primary policy selected XGBoost.',
};
const percent = value => `${(value * 100).toFixed(1)}%`;

export function Details({ data }) {
  const decision = data.decision; const quality = decision.quality;
  return <div className="detail-grid" id="reliability">
    <section className="panel" aria-labelledby="decision-heading"><div className="section-heading"><div><p className="eyebrow">Reliability insight</p><h2 id="decision-heading">A decision you can inspect</h2></div><span className={`tag ${decision.choice === 'fallback' ? 'amber' : 'green'}`}>{decision.choice === 'fallback' ? 'Fallback selected' : decision.choice === 'primary' ? 'Primary selected' : 'Unavailable'}</span></div>
      <p className="decision-model">{modelName(data.forecasts[0]?.selected_model)}</p><p className="decision-copy">{explanations[decision.reason] || decision.reason}</p><code className="reason-code">{decision.reason}</code>
      <div className="quality-grid"><div><span>Unavailable readings</span><strong>{percent(quality.unavailable_fraction)}</strong></div><div><span>Repeated pairs</span><strong>{percent(quality.repeated_pair_fraction)}</strong></div><div><span>Abrupt changes</span><strong>{percent(quality.abrupt_pair_fraction)}</strong></div><div><span>Model disagreement</span><strong>{Number.isFinite(decision.disagreement_sigma) ? `${decision.disagreement_sigma.toFixed(3)} σ` : 'Unavailable'}</strong></div></div>
      <p className="panel-note">Quality describes the 12-step input across all four sensors. Repeated or abrupt readings are indicators, not confirmed faults. One decision serves every sensor and horizon.</p>
    </section>
    <section className="panel history-panel" aria-labelledby="history-heading"><div className="section-heading"><div><p className="eyebrow">Observed context</p><h2 id="history-heading">The preceding 12 readings</h2></div><span className="tag">Synthetic clock</span></div>
      <div className="table-scroll" tabIndex={0} role="region" aria-label="Historical readings; scroll to inspect all 12 observations"><table><caption className="sr-only">Historical readings for {data.sensor_label}; all observations are at or before the replay origin.</caption><thead><tr><th scope="col">Time</th><th scope="col">Reading</th><th scope="col">Availability</th></tr></thead><tbody>{data.history.map(point => <tr key={point.index}><td>{point.timestamp.slice(11, 16)}<span className="row-offset">{point.relative_minutes === 0 ? 'Origin' : `${point.relative_minutes} min`}</span></td><td>{Number.isFinite(point.value) ? point.value.toFixed(2) : 'Missing'}</td><td><span className={`table-status ${point.usable ? '' : 'unavailable'}`}>{point.usable ? 'Observed' : 'Unavailable by policy'}</span></td></tr>)}</tbody></table></div>
      <p className="panel-note">Native synthetic units. Zeros remain ambiguous; unavailable readings are not confirmed sensor failures.</p>
    </section>
  </div>;
}
