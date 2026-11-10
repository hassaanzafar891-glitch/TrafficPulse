import { useMemo } from 'react';
import { createApiClient } from './api/client.js';
import { useDashboard } from './hooks/useDashboard.js';
import { ReplayControls } from './components/ReplayControls.jsx';
import { Forecasts } from './components/Forecasts.jsx';
import { Details } from './components/Details.jsx';
import { TrafficChart } from './components/TrafficChart.jsx';
import { DemoGuide } from './components/DemoGuide.jsx';

function Message({ title, children, error = false, retry }) {
  return <section className={`panel message ${error ? 'error-message' : ''}`} role={error ? 'alert' : 'status'}><span className="message-icon" aria-hidden="true">{error ? '!' : '○'}</span><h2>{title}</h2><p>{children}</p>{retry && <button className="primary" onClick={retry}>Try again</button>}</section>;
}

export default function App({ api: suppliedApi }) {
  const api = useMemo(() => {
    try { return suppliedApi || createApiClient(); }
    catch (error) { const fail = async () => { throw error; }; return { health: fail, metadata: fail, replay: fail }; }
  }, [suppliedApi]);
  const { boot, selection, replay, change, retry } = useDashboard(api);
  const metadata = boot.metadata;
  const error = boot.error || replay.error;
  const status = error ? (error.kind === 'offline' ? 'Backend offline' : 'API unavailable') : boot.status === 'ready' ? 'Backend ready' : 'Connecting';
  return <div className="app-shell">
    <a className="skip-link" href="#main">Skip to dashboard</a>
    <aside className="sidebar" aria-label="Workspace navigation"><a href="#overview" className="brand"><span className="brand-mark" aria-hidden="true"><svg viewBox="0 0 32 32"><path d="M3 17h6l4-9 6 17 4-8h6" /></svg></span><span>TrafficPulse<small>REPLAY WORKSPACE</small></span></a>
      <div className="nav-caption">Workspace</div><nav><a className="nav-item current" href="#overview"><span aria-hidden="true">▦</span> Overview</a><a className="nav-item" href="#replay"><span aria-hidden="true">↺</span> Historical replay</a><a className="nav-item" href="#reliability"><span aria-hidden="true">◇</span> Reliability</a></nav>
      <div className="sidebar-note"><span className="sidebar-note-dot" />Synthetic data only<p>Explore model behavior with fabricated traffic observations. No live feeds or real sensor locations.</p><span className="contract-note">{metadata ? `API contract ${metadata.contract_version}` : 'Local demonstration'}</span></div>
    </aside>
    <div className="workspace"><header className="topbar"><span>Workspace <span aria-hidden="true">/</span> <strong>Overview</strong></span><span className={`connection ${error ? 'not-ready' : ''}`}><i aria-hidden="true" />{status}</span></header>
      <main id="main" tabIndex={-1}><div className="demo-notice"><span className="demo-icon" aria-hidden="true">◇</span><strong>Synthetic demonstration</strong><span>Historical replay, not live traffic. All speeds use synthetic units.</span></div>
        <section id="overview" className="hero"><div><p className="eyebrow">TRAFFIC INTELLIGENCE / LOCAL DEMO</p><h1>Every forecast.<br />A clearer perspective.</h1><p>Replay historical inputs. Inspect the forecast.<br className="desktop-break" /> Understand when the controller falls back.</p></div><div className="origin-card"><span className="eyebrow">Replay origin</span><strong>{replay.data ? replay.data.timestamp.replace('T', ' · ') : 'Awaiting replay'}</strong><span>Timezone-naive synthetic clock</span><div className="origin-rule" /><span>{replay.data?.sensor_label || 'Select from the backend sensor list'}</span></div></section>
        {boot.status === 'loading' && <Message title="Connecting to the local workspace">Loading backend health and synthetic demonstration metadata…</Message>}
        {boot.status === 'error' && <Message title={boot.error.kind === 'offline' ? 'The backend is offline' : 'The workspace is unavailable'} error retry={retry}>{boot.error.message} No demonstration forecasts have been substituted. Start or check the local backend, then try again.</Message>}
        {boot.status === 'empty' && <Message title="No replay data is available" retry={retry}>The backend has no available synthetic sensors, horizons or replay origins. No forecasts are shown.</Message>}
        {boot.status === 'ready' && selection && <>
          <ReplayControls metadata={metadata} selection={selection} replay={replay} change={change} />
          <DemoGuide metadata={metadata} change={change} />
          <div aria-live="polite" aria-busy={replay.status === 'loading'}>
            {replay.status === 'loading' && <Message title="Loading forecasts">Requesting {metadata.sensor_details.find(sensor => sensor.sensor_id === selection.sensor_id)?.label} at replay index {selection.index} ({selection.scenario}). Previous charts stay hidden until this selection is ready.</Message>}
            {replay.status === 'error' && <Message title={replay.error.kind === 'offline' ? 'Connection interrupted' : 'The forecast request could not be completed'} error retry={() => change({})}>{replay.error.message} Previous forecasts are hidden to avoid displaying stale results. Try again repeats the selected sensor, replay origin, scenario and policy; you can also choose another case.</Message>}
            {replay.status === 'ready' && replay.data && <><TrafficChart key={`${replay.data.sensor_id}:${replay.data.index}:${replay.data.scenario}:${replay.data.policy}`} data={replay.data} /><Forecasts data={replay.data} metadata={metadata} /><Details data={replay.data} /></>}
          </div>
        </>}
        <footer><span>TrafficPulse <span aria-hidden="true">/</span> Synthetic replay foundation</span><span>No live traffic · No geographic coordinates</span></footer>
      </main>
    </div>
  </div>;
}
