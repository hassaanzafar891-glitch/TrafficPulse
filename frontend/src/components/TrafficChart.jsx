import { useEffect, useId, useMemo, useRef, useState } from 'react';
import { mapSeries } from '../visualizations/series.js';
import { modelName } from '../utils/presentation.js';

function describe(point) {
  const kind = point.kind === 'history' ? 'Historical input' : `+${point.x} minute forecast`;
  return `${kind} · ${point.timestamp.replace('T', ' · ')} · ${point.y === null ? 'Unavailable' : point.y.toFixed(2)}`;
}

export function TrafficChart({ data }) {
  const id = useId(); const container = useRef(null);
  const [width, setWidth] = useState(900); const [active, setActive] = useState('');
  const series = useMemo(() => mapSeries(data), [data]);
  useEffect(() => {
    if (typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(entries => setWidth(Math.max(240, entries[0].contentRect.width)));
    observer.observe(container.current); return () => observer.disconnect();
  }, []);
  const selected = series.points.find(p => p.id === active);
  const height = 300; const margin = { left: 46, right: 20, top: 42, bottom: 49 };
  const plotWidth = width - margin.left - margin.right; const plotHeight = height - margin.top - margin.bottom;
  const firstX = Math.min(...series.history.map(p => p.x)); const lastX = Math.max(...series.forecasts.map(p => p.x));
  const x = value => margin.left + ((value - firstX) / (lastX - firstX)) * plotWidth;
  const y = value => margin.top + (1 - (value - series.domain[0]) / (series.domain[1] - series.domain[0])) * plotHeight;
  const ticks = series.domain ? Array.from({ length: 5 }, (_, i) => series.domain[0] + (series.domain[1] - series.domain[0]) * i / 4) : [];
  const timeTicks = [firstX, series.history[Math.floor(series.history.length / 2)].x, 0, ...series.forecasts.map(p => p.x)];
  return <section className="panel traffic-chart" aria-labelledby={`${id}-heading`}>
    <div className="section-heading"><div><p className="eyebrow">Historical inputs & forecast</p><h2 id={`${id}-heading`}>From observed context to the next hour</h2></div><span className="tag">{data.sensor_label}</span></div>
    <div className="chart-context"><span>{`Serving: ${modelName(data.forecasts[0].selected_model)}`}</span><span>{`Joint ${data.decision.choice} · ${data.decision.reason}`}</span><span>{`Unavailable inputs: ${(data.decision.quality.unavailable_fraction * 100).toFixed(1)}%`}</span><span>{`Repeated: ${(data.decision.quality.repeated_pair_fraction * 100).toFixed(1)}% · Abrupt: ${(data.decision.quality.abrupt_pair_fraction * 100).toFixed(1)}%`}</span></div>
    <div className="chart-legend" aria-label="Chart legend"><span><i className="legend-history" />Observed input</span><span><i className="legend-forecast" />Forecast points</span><span>× Unusable input gap</span>{series.forecasts.some(p => p.interval) && <span>↕ Calibration-derived interval</span>}</div>
    <p className="chart-axis-label">Speed · synthetic units</p>
    <div ref={container} className="chart-canvas">
      {series.domain ? <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby={`${id}-title ${id}-desc`} className="speed-chart">
        <title id={`${id}-title`}>Historical readings and three forecast horizons for {data.sensor_label}</title>
        <desc id={`${id}-desc`}>Solid historical segments break at unusable readings. Separate diamond predictions are positioned at positive minute offsets. The origin separates history and forecasts. Use the inspect-point selector for keyboard details.</desc>
        <rect x={x(0)} y={margin.top} width={x(lastX) - x(0)} height={plotHeight} className="forecast-zone" />
        {ticks.map((tick, i) => <g key={i}><line x1={margin.left} x2={width - margin.right} y1={y(tick)} y2={y(tick)} className="chart-gridline" /><text x={margin.left - 9} y={y(tick) + 4} textAnchor="end" className="chart-tick">{tick.toFixed(1)}</text></g>)}
        <line x1={x(0)} x2={x(0)} y1={margin.top - 8} y2={height - margin.bottom} className="origin-divider" />
        <text x={x(0) + 6} y={margin.top + 14} textAnchor="start" className="origin-label">Origin</text>
        {timeTicks.map((tick, i) => <text key={tick} x={x(tick)} y={height - margin.bottom + 21} textAnchor={i === 0 ? 'start' : i === timeTicks.length - 1 ? 'end' : 'middle'} className="chart-tick">{tick > 0 ? `+${tick}` : tick}</text>)}
        <text x={margin.left + plotWidth / 2} y={height - 5} textAnchor="middle" className="chart-tick">Minutes relative to replay origin</text>
        {series.segments.filter(run => run.length > 1).map(run => <polyline key={run[0].id} data-testid="history-segment" points={run.map(p => `${x(p.x)},${y(p.y)}`).join(' ')} className="history-line" />)}
        {series.points.map(point => <g key={point.id} data-point-id={point.id} onPointerEnter={() => setActive(point.id)} onPointerDown={() => setActive(point.id)}>
          <title>{describe(point)}</title>
          {point.interval && <g className="interval-whisker" data-testid="interval-whisker"><line x1={x(point.x)} x2={x(point.x)} y1={y(point.interval.lower)} y2={y(point.interval.upper)} /><line x1={x(point.x) - 6} x2={x(point.x) + 6} y1={y(point.interval.lower)} y2={y(point.interval.lower)} /><line x1={x(point.x) - 6} x2={x(point.x) + 6} y1={y(point.interval.upper)} y2={y(point.interval.upper)} /></g>}
          {point.y === null ? <text x={x(point.x)} y={point.kind === 'history' ? margin.top - 12 : margin.top - 25} textAnchor="middle" className="gap-mark" data-testid="gap-marker">×</text> : point.kind === 'history' ?
            <circle cx={x(point.x)} cy={y(point.y)} r={active === point.id ? 5 : 3.5} className="history-point" data-testid="history-point" /> :
            <path d={`M ${x(point.x)} ${y(point.y) - 6} l 6 6 l -6 6 l -6 -6 Z`} className="forecast-point" data-testid="forecast-point" />}
          <circle cx={x(point.x)} cy={point.y === null ? margin.top - 16 : y(point.y)} r={12} fill="transparent" />
        </g>)}
      </svg> : <p className="chart-empty" role="status">No usable historical observations or available forecasts to plot.</p>}
    </div>
    {!series.history.some(p => p.y !== null) && <p className="panel-note">No usable historical readings in this window. Only available forecast points are plotted.</p>}
    <div className="chart-inspector"><label htmlFor={`${id}-point`}>Inspect chart point<select id={`${id}-point`} value={active} onChange={event => setActive(event.target.value)}><option value="">Choose a point or hover over the chart</option>{series.points.map(p => <option key={p.id} value={p.id}>{describe(p)}</option>)}</select></label>
      <div className="chart-detail" aria-live="polite">{selected ? <><strong>{selected.kind === 'history' ? 'Historical input' : `Forecast +${selected.x} min`}</strong><span>{selected.timestamp.replace('T', ' · ')}</span><span>{selected.y === null ? 'Unavailable by policy — no speed plotted' : `${selected.y.toFixed(2)} synthetic speed units`}</span>{selected.kind === 'history' && selected.y === null && <small>{selected.raw === 0 ? 'Raw zero remains ambiguous; it is not plotted as a measured speed.' : 'Missing, nonfinite or otherwise unusable reading. No fill is substituted.'}</small>}{selected.kind === 'forecast' && <small>{modelName(selected.model)} · {selected.interval ? `${selected.interval.lower.toFixed(2)}–${selected.interval.upper.toFixed(2)}; ${(selected.interval.nominal * 100).toFixed(0)}% nominal, not independently validated` : 'Interval unavailable; coverage is not validated for this prediction.'}</small>}</> : <><strong>Inspect an observed or predicted point</strong><span>Hover, tap, or use the selector with a keyboard. No intermediate forecasts are generated.</span></>}</div>
    </div>
    <p className="panel-note">Historical segments connect adjacent usable observations only. Forecast diamonds show just the three returned horizons; no line bridges history to predictions. Crosses above the plot mark unavailable slots, not zero speed. {series.forecasts.some(p => p.interval) ? 'Whiskers use clean primary calibration only; coverage is not independently validated.' : 'No intervals are shown for controller, fallback or faulty inputs.'}</p>
  </section>;
}
