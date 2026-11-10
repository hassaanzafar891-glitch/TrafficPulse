"""Clean split calibration with the frozen finite-sample order statistic rule."""
from dataclasses import dataclass
import math

import numpy as np

from .data import HORIZONS, usable, windows


def conformal_quantile(residuals, coverage=0.9):
    """Return (width, count, one-based rank); empty samples are unavailable.

    The capped rank follows the project protocol. At tiny n it does not imply
    the requested finite-sample coverage (an infinite bound would be needed).
    """
    if not math.isfinite(coverage) or not 0<coverage<1:
        raise ValueError("Coverage must be strictly between zero and one")
    values=np.asarray(residuals,dtype=float)
    if values.ndim!=1 or not np.isfinite(values).all() or (values<0).any():
        raise ValueError("Residuals must be a finite nonnegative vector")
    n=len(values)
    if not n:
        return None,0,None
    rank=min(n,math.ceil((n+1)*coverage))
    return float(np.partition(values,rank-1)[rank-1]),n,rank


def calibration_targets(partition):
    # Reject other roles before reading their observations or timestamps.
    if partition.role!="calibration":
        raise ValueError("Only the reserved calibration partition is permitted")
    samples=list(windows(partition))
    if not samples:
        return samples,np.empty((0,3,len(partition.sensors)))
    return samples,np.stack([w.targets[np.array(HORIZONS)-1] for w in samples])


@dataclass(frozen=True)
class SplitConformal:
    coverage: float
    half_widths: tuple
    counts: tuple
    ranks: tuple
    sensors: tuple

    @classmethod
    def fit(cls,partition,predictions,coverage=0.9):
        """Predictions must come from an already-frozen model on clean histories.

        This API has no estimator fitting, feature selection or controller path.
        Never pass faulted calibration observations; the runner owns clean input.
        """
        _,targets=calibration_targets(partition)
        predictions=np.asarray(predictions)
        if predictions.shape!=targets.shape:
            raise ValueError("Expected predictions [calibration windows,3,sensors]")
        mask=usable(targets)
        if not np.isfinite(predictions[mask]).all():
            raise ValueError("Nonfinite prediction on a valid calibration target")
        quantiles=[conformal_quantile(np.abs(targets[:,i,:]-predictions[:,i,:])[mask[:,i,:]],coverage)
                   for i in range(3)]
        return cls(coverage,tuple(q[0] for q in quantiles),tuple(q[1] for q in quantiles),
                   tuple(q[2] for q in quantiles),partition.sensors)

    def intervals(self,predictions):
        predictions=np.asarray(predictions,dtype=float)
        if predictions.ndim!=3 or predictions.shape[1:]!=(3,len(self.sensors)):
            raise ValueError("Expected predictions [windows,3,sensors] in horizons 3/6/12 order")
        q=np.array([np.nan if v is None else v for v in self.half_widths])[None,:,None]
        return predictions-q,predictions+q  # no clipping

    def calibration_diagnostics(self,partition,predictions):
        _,targets=calibration_targets(partition)
        if partition.sensors!=self.sensors or np.shape(predictions)!=targets.shape:
            raise ValueError("Diagnostic schema mismatch")
        lower,upper=self.intervals(predictions)
        result={}
        for i,h in enumerate(HORIZONS):
            mask=usable(targets[:,i,:]); n=int(mask.sum())
            available=np.isfinite(lower[:,i,:])&np.isfinite(upper[:,i,:])
            unavailable=int((mask&~available).sum())
            defined=n>0 and not unavailable
            result[str(h)]={"coverage":float(((lower[:,i,:]<=targets[:,i,:])&(targets[:,i,:]<=upper[:,i,:]))[mask].mean()) if defined else None,
                "average_interval_width":float((upper[:,i,:]-lower[:,i,:])[mask].mean()) if defined else None,
                "valid_target_count":n,"unavailable_interval_count":unavailable,
                "calibration_count":self.counts[i],"calibrated_half_width":self.half_widths[i],
                "reason":None if defined else ("empty target mask" if not n else "interval unavailable"),
                "scope":"in-sample calibration diagnostic, not independent validation"}
        return result
