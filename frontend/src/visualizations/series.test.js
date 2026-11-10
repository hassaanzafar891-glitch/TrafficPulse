import { describe, expect, it } from 'vitest';
import { mapSeries } from './series.js';
import { clockOffsetMinutes, supportedInterval } from '../utils/presentation.js';
import { replay, request } from '../test/fixtures.js';

describe('backend-only chart mapping', () => {
  it('keeps exactly twelve historical slots and three real horizon points', () => {
    const data = replay(); const series = mapSeries(data);
    expect(series.history).toHaveLength(12); expect(series.forecasts).toHaveLength(3);
    expect(series.history.map(p => p.x)).toEqual(data.history.map(p => p.relative_minutes));
    expect(series.forecasts.map(p => p.x)).toEqual([15,30,60]);
    expect(series.forecasts.map(p => p.y)).toEqual(data.forecasts.map(p => p.prediction));
    expect(series.points.map(p => p.timestamp)).toEqual([...data.history,...data.forecasts].map(p => p.timestamp));
  });
  it('makes zero, missing, negative, nonfinite and unusable readings gaps without using fills', () => {
    const data = replay();
    [null,0,-1,NaN,Infinity].forEach((value,i) => { data.history[i].value=value; data.history[i].usable=true; data.history[i].filled_value=999; });
    data.history[5].usable=false;
    const series = mapSeries(data);
    expect(series.history.slice(0,6).every(p => p.y === null)).toBe(true);
    expect(series.history.slice(6).every(p => p.y === 40)).toBe(true);
    expect(series.segments).toHaveLength(1);expect(series.segments[0]).toHaveLength(6);
  });
  it('never connects across a missing interior slot or adds a forecast curve', () => {
    const data = replay();data.history[5].value=null;data.history[5].usable=false;
    const series = mapSeries(data);
    expect(series.segments.map(s => s.map(p => p.x))).toEqual([[-50,-45,-40,-35],[-25,-20,-15,-10,-5,0]]);
    expect(series.segments.flat().every(p => p.kind==='history')).toBe(true);
  });
  it('suppresses all unvalidated intervals even if a payload contains numbers', () => {
    for (const scenario of ['clean','missing','noisy','stuck']) {
      const data = replay({ ...request, scenario });
      data.forecasts.forEach(f => { f.interval={available:true,status:'calibration_derived',lower:1,upper:100,nominal_coverage:.9}; });
      expect(mapSeries(data).forecasts.every(p => p.interval===null)).toBe(true);
    }
    const fixed=replay({ ...request,policy:'xgboost' });
    expect(mapSeries(fixed).forecasts.map(p=>p.interval.lower)).toEqual(fixed.forecasts.map(f=>f.interval.lower));
    fixed.forecasts[0].prediction=null;expect(mapSeries(fixed).forecasts[0].interval).toBeNull();
    fixed.scenario='noisy';expect(mapSeries(fixed).forecasts.every(p=>p.interval===null)).toBe(true);
  });
  it('includes permitted un-clipped interval endpoints in display scaling', () => {
    const data=replay({ ...request,policy:'xgboost' });data.forecasts[0].interval.lower=-4;
    const series=mapSeries(data);expect(series.domain[0]).toBeLessThan(-4);
    expect(series.forecasts[0].interval.lower).toBe(-4);
    data.forecasts[0].interval.upper=-5;expect(supportedInterval(data,data.forecasts[0])).toBe(false);
  });
  it('has an honest empty-domain state and keeps isolated observations', () => {
    const data=replay();data.history.forEach(p=>{p.value=null;p.usable=false;});data.forecasts.forEach(p=>p.prediction=null);
    expect(mapSeries(data).domain).toBeNull();
    data.history[4].value=10;data.history[4].usable=true;
    expect(mapSeries(data).segments[0]).toHaveLength(1);
    expect(mapSeries(data).domain[0]).toBeLessThan(10);expect(mapSeries(data).domain[1]).toBeGreaterThan(10);
  });
  it('uses naive clock arithmetic across midnight without locale conversion', () => {
    expect(clockOffsetMinutes('2024-01-01T23:45:00','2024-01-02T00:00:00')).toBe(15);
    expect(clockOffsetMinutes('2024-01-02T00:00:00','2024-01-01T23:05:00')).toBe(-55);
    expect(clockOffsetMinutes('2024-02-30T00:00:00','2024-03-01T00:00:00')).toBeNull();
    expect(clockOffsetMinutes('2024-01-01T00:00:00Z','2024-01-01T00:15:00')).toBeNull();
  });
});
