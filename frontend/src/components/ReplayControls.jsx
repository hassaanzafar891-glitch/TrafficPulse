import { useEffect, useState } from 'react';

const scenarioLabels = { clean: 'Clean inputs', missing: 'Simulated missing readings', noisy: 'Simulated noise', stuck: 'Simulated stuck readings' };
const policyLabels = { controller: 'Reliability controller', xgboost: 'Fixed XGBoost' };

export function ReplayControls({ metadata, selection, replay, change }) {
  const [draft, setDraft] = useState(String(selection.index));
  useEffect(() => setDraft(String(selection.index)), [selection.index]);
  const index = Number(draft);
  const valid = draft.trim() !== '' && Number.isInteger(index) && index >= metadata.replay.first_index && index <= metadata.replay.last_index;
  const busy = replay.status === 'loading'; const links = replay.data?.playback;
  return <section className="panel replay-panel" id="replay" aria-labelledby="replay-heading">
    <div className="section-heading"><div><p className="eyebrow">Your workspace</p><h2 id="replay-heading">Historical replay</h2></div><span className="tag">5-minute observations</span></div>
    <div className="controls-grid">
      <label>Sensor<select value={selection.sensor_id} onChange={event => change({ sensor_id: event.target.value })}>{metadata.sensor_details.map(s => <option value={s.sensor_id} key={s.sensor_id}>{s.label}</option>)}</select></label>
      <label>Input scenario<select value={selection.scenario} onChange={event => change({ scenario: event.target.value })}>{metadata.scenarios.map(s => <option value={s} key={s}>{scenarioLabels[s] || s}</option>)}</select></label>
      <label>Forecast policy<select value={selection.policy} onChange={event => change({ policy: event.target.value })}>{metadata.policies.map(p => <option value={p} key={p}>{policyLabels[p] || p}</option>)}</select></label>
    </div>
    <div className="playback-row">
      <div className="step-buttons"><button type="button" className="secondary" aria-label="Previous replay step" disabled={busy || links?.previous_index == null} onClick={() => change({ index: links.previous_index })}><span aria-hidden="true">←</span> Previous</button><button type="button" className="primary" aria-label="Next replay step" disabled={busy || links?.next_index == null} onClick={() => change({ index: links.next_index })}>Next step <span aria-hidden="true">→</span></button></div>
      <form className="index-form" onSubmit={event => { event.preventDefault(); if (valid) change({ index }); }}>
        <label htmlFor="replay-index">Replay index</label><input id="replay-index" type="number" step="1" min={metadata.replay.first_index} max={metadata.replay.last_index} value={draft} onChange={event => setDraft(event.target.value)} aria-describedby="index-help" />
        <button type="submit" className="secondary" disabled={!valid || busy}>Go</button>
      </form>
    </div>
    <div className="replay-caption"><p id="index-help">Available indices {metadata.replay.first_index}–{metadata.replay.last_index}. Navigation stops at each boundary.</p><p>{selection.scenario === 'clean' ? 'Original synthetic observations, including natural zeros and missing readings.' : 'Window-local simulation only. Fault placement can change between replay origins.'}</p></div>
  </section>;
}
