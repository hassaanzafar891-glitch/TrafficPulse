"""Causal observable input quality and a frozen-grid joint forecast gate."""
from dataclasses import dataclass
import math

import numpy as np

from .data import HORIZONS, usable, windows
from .xgboost_experiment import score


def assess_quality(inputs,statistics):
    """Inspect exactly the historical 12 rows, with training-only scales.

    Repeated adjacent positive readings and >3-training-std jumps are descriptive
    signals, not confirmed faults or additional tuned controller thresholds.
    """
    statistics.fill(inputs)  # validate schema/time eligibility; no future API
    values=np.asarray(inputs.values); valid=usable(values)
    adjacent=valid[1:]&valid[:-1]
    denominator=int(adjacent.sum())
    changes=np.abs(np.diff(values,axis=0))
    return {"unavailable_fraction":float((~valid).mean()),
            "repeated_pair_fraction":float(((changes==0)&adjacent).sum()/denominator) if denominator else 0.0,
            "abrupt_pair_fraction":float(((changes>3*statistics.stds)&adjacent).sum()/denominator) if denominator else 0.0,
            "usable_adjacent_pair_count":denominator}


@dataclass(frozen=True)
class Thresholds:
    unavailable: float
    disagreement: float
    always_primary: bool = False

    def __post_init__(self):
        if not 0<=self.unavailable<=1 or math.isnan(self.disagreement) or self.disagreement<0:
            raise ValueError("Invalid controller thresholds")


def threshold_grid():
    return [Thresholds(a,d) for a in (.1,.2,.4,.6) for d in (1.,2.,3.,math.inf)]+[Thresholds(1.,math.inf,True)]


def select_forecast(inputs,primary,fallback,statistics,thresholds):
    """Return [3,S] forecasts plus one joint decision; no labels or oracle masks."""
    quality=assess_quality(inputs,statistics)
    primary=np.asarray(primary);fallback=np.asarray(fallback)
    shape=(len(HORIZONS),len(inputs.sensors))
    if primary.shape!=shape or fallback.shape!=shape:
        raise ValueError("Forecasts must follow horizons 3/6/12 and sensor schema")
    primary_ready=bool(np.isfinite(primary).all());fallback_ready=bool(np.isfinite(fallback).all())
    disagreement=float(np.max(np.mean(np.abs(primary-fallback)/statistics.stds[None,:],axis=1))) if primary_ready and fallback_ready else None
    if not primary_ready:
        choice="fallback" if fallback_ready else "abstain"
        reason="primary_unavailable" if fallback_ready else "both_unavailable"
    elif not fallback_ready:
        choice="primary";reason="fallback_unavailable"
    elif thresholds.always_primary:
        choice="primary";reason="always_primary"
    elif quality['unavailable_fraction']>thresholds.unavailable:
        choice="fallback";reason="input_unavailable"
    elif disagreement>thresholds.disagreement:
        choice="fallback";reason="model_disagreement"
    else:
        choice="primary";reason="within_thresholds"
    forecast=primary if choice=="primary" else fallback if choice=="fallback" else np.full(shape,np.nan)
    return forecast,{"choice":choice,"reason":reason,"quality":quality,"disagreement_sigma":disagreement,
                     "interval_validity":"unresolved; primary conformal calibration does not transfer to switching"}


def controller_predictions(inputs,primary,fallback,statistics,thresholds):
    if np.shape(primary)!=np.shape(fallback) or np.shape(primary)!=(len(inputs),3,len(statistics.sensors)):
        raise ValueError("Prediction arrays must align to shared inputs")
    pairs=[select_forecast(item,p,f,statistics,thresholds) for item,p,f in zip(inputs,primary,fallback)]
    predictions=np.stack([p[0] for p in pairs]) if pairs else np.empty(np.shape(primary))
    return predictions,[p[1] for p in pairs]


def decision_summary(decisions):
    n=len(decisions);choices=[d['choice'] for d in decisions]
    fallback=choices.count('fallback');abstain=choices.count('abstain')
    switches=sum(a!=b for a,b in zip(choices,choices[1:]))
    reasons={reason:sum(d['reason']==reason for d in decisions) for reason in sorted({d['reason'] for d in decisions})}
    return {'origins':n,'fallback_count':fallback,'fallback_frequency':fallback/n if n else None,
            'primary_frequency':choices.count('primary')/n if n else None,'abstention_frequency':abstain/n if n else None,
            'switch_count':switches,'switches_per_1000_origins':1000*switches/n if n else None,'reason_counts':reasons}


def choose_thresholds(partition,conditions,statistics):
    """Offline selection on controller-selection labels only; never called at runtime.

    Each condition provides the same histories/clean targets and frozen-model
    predictions for every gate. Positive fault cases are equally weighted.
    """
    if partition.role!='controller_selection':
        raise ValueError('Threshold selection is controller-selection only')
    if not conditions or conditions[0]['name']!='clean' or len(conditions)<2:
        raise ValueError('Require clean plus predeclared faulty conditions')
    if conditions[0]['partition_offset']!=partition.offset or any(c['partition_offset']!=partition.offset for c in conditions):
        raise ValueError('Conditions belong to a different partition')
    original=list(windows(partition))
    targets=np.stack([w.targets[np.array(HORIZONS)-1] for w in original]) if original else np.empty((0,3,len(partition.sensors)))
    for c in conditions:
        if (not np.array_equal(c['targets'],targets,equal_nan=True) or len(c['inputs'])!=len(original) or
                any(item.origin_time!=w.inputs.origin_time or item.sensors!=partition.sensors for item,w in zip(c['inputs'],original))):
            raise ValueError('All candidates must share original targets and partition origins')
    clean=conditions[0]
    baseline=score(clean['targets'],clean['primary'])
    if any(v['mae'] is None for v in baseline.values()):
        raise ValueError('Cannot select thresholds with empty clean target masks')
    primary_mae=float(np.mean([v['mae'] for v in baseline.values()]))
    candidates=[]
    for index,thresholds in enumerate(threshold_grid()):
        scores=[];fallback_count=0;eligible=True
        for condition in conditions:
            prediction,decisions=controller_predictions(condition['inputs'],condition['primary'],condition['fallback'],statistics,thresholds)
            if not np.isfinite(prediction).all():
                eligible=False;break
            metrics=score(condition['targets'],prediction)
            if any(v['mae'] is None for v in metrics.values()):
                eligible=False;break
            scores.append(float(np.mean([v['mae'] for v in metrics.values()])))
            fallback_count+=sum(d['choice']=='fallback' for d in decisions)
        clean_score=scores[0] if scores else None
        eligible=eligible and clean_score<=1.01*primary_mae
        candidates.append({'index':index,'thresholds':thresholds,'eligible':eligible,'clean_mean_mae':clean_score,
                           'fault_mean_mae':float(np.mean(scores[1:])) if eligible else None,'fallback_count':fallback_count})
    eligible=[c for c in candidates if c['eligible']]
    if not eligible:
        raise ValueError('No clean-safe eligible controller; stop before test')
    selected=min(eligible,key=lambda c:(c['fault_mean_mae'],c['fallback_count'],c['index']))
    return selected['thresholds'],candidates
