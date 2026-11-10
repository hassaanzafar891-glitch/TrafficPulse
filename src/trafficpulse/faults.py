"""Deterministic synthetic development input overlays; clean targets never edited."""
from dataclasses import dataclass
import hashlib
import math

import numpy as np

from .data import ForecastInput, Partition, STEP, TrainingStatistics, usable, windows

KINDS={"missing":1,"stuck":2,"noisy":3}
MODES={"point":1,"contiguous":2}
FAMILIES={("missing","point"),("missing","contiguous"),("stuck","contiguous"),
          ("noisy","point"),("noisy","contiguous")}


def array_hash(array):
    array=np.ascontiguousarray(array)
    return hashlib.sha256(str(array.dtype).encode()+str(array.shape).encode()+array.tobytes()).hexdigest()


@dataclass(frozen=True)
class FaultSpec:
    kind: str
    mode: str
    rate: float
    seed: int
    severity: float = 0.2
    block_rows: int = 12
    episode_rows: int = 288
    sampling_policy: str = "protocol_floor"

    def __post_init__(self):
        if ((self.kind,self.mode) not in FAMILIES or not math.isfinite(self.rate) or
                not 0<=self.rate<=1 or not math.isfinite(self.severity) or self.severity<0):
            raise ValueError("Invalid fault family, rate or severity")
        if self.seed<0 or not 1<=self.block_rows<=self.episode_rows:
            raise ValueError("Invalid seed or block/episode duration")
        if self.sampling_policy not in ("protocol_floor", "synthetic_ceil"):
            raise ValueError("Unknown sampling policy")


@dataclass(frozen=True)
class FaultOverlay:
    values: np.ndarray
    selected: np.ndarray
    source_hash: str
    time_hash: str
    offset: int
    sensors: tuple[str,...]
    details: dict


def noise_scales(train: Partition):
    if train.role!="train":
        raise ValueError("Noise scale fitting is train-only")
    global_values=train.values[usable(train.values)]
    if not len(global_values):
        raise ValueError("No usable train data")
    fallback=float(np.std(global_values)) or 1.0
    result=[]
    for column in train.values.T:
        valid=column[usable(column)]
        sigma=float(np.std(valid)) if len(valid) else 0.0
        result.append(sigma if sigma>0 else fallback)
    return np.array(result)


def inject(partition: Partition, spec: FaultSpec, stats: TrainingStatistics, scales):
    """Place one partition-wide overlay, independent of windows/model iteration.

    This Phase 3A API permits synthetic tuning only. The synthetic partition RNG
    namespace 1 is separate from the frozen controller/test namespaces 3/4.
    """
    if partition.role!="tuning" or not all(s.startswith("synthetic") for s in partition.sensors):
        raise ValueError("Phase 3A permits synthetic tuning overlays only")
    return _inject(partition,spec,stats,scales,1)


def inject_controller(partition: Partition, spec: FaultSpec, stats: TrainingStatistics, scales):
    """Phase 3C synthetic controller-selection only; frozen RNG namespace 3."""
    if partition.role!="controller_selection" or not all(s.startswith("synthetic") for s in partition.sensors):
        raise ValueError("Phase 3C permits synthetic controller-selection overlays only")
    return _inject(partition,spec,stats,scales,3)


def _inject(partition, spec, stats, scales, partition_code):
    if (partition.sensors!=stats.sensors or not len(partition.timestamps) or
            partition.timestamps[0]<=stats.training_end or
            not np.all(np.diff(partition.timestamps)==STEP)):
        raise ValueError("Partition schema, chronology or training cutoff invalid")
    scales=np.asarray(scales)
    if scales.shape!=(len(partition.sensors),) or not np.isfinite(scales).all() or (scales<=0).any():
        raise ValueError("Need finite positive train-derived noise scales")
    clean=partition.values
    values=clean.copy(); selected=np.zeros(clean.shape,dtype=bool)
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [spec.seed,partition_code,KINDS[spec.kind],MODES[spec.mode]])))
    blocks=[]; selected_sensors=[]
    rounding=math.ceil if spec.sampling_policy=="synthetic_ceil" else math.floor
    if spec.mode=="point":
        eligible=np.flatnonzero(usable(clean).ravel())
        permutation=rng.permutation(eligible)
        denominator=len(eligible)
        count=rounding(spec.rate*denominator)
        selected.flat[permutation[:count]]=True
        normal=rng.standard_normal(clean.shape)
        selected_sensors=np.flatnonzero(selected.any(axis=0)).tolist()
        rate_basis="fraction of originally usable partition cells"
    else:
        sensor_order=rng.permutation(clean.shape[1])
        denominator=clean.shape[1]
        count=rounding(spec.rate*denominator)
        selected_sensors=sensor_order[:count].tolist()
        # Draw ALL sensor/episode starts before selecting rate-specific sensors.
        all_blocks=[]
        for sensor in range(clean.shape[1]):
            for episode in range(0,len(clean),spec.episode_rows):
                stop=min(len(clean),episode+spec.episode_rows)
                if stop-episode>=spec.block_rows:
                    start=int(rng.integers(episode,stop-spec.block_rows+1))
                    all_blocks.append((sensor,start,start+spec.block_rows))
        normal=rng.standard_normal(clean.shape)
        for sensor,start,stop in all_blocks:
            if sensor in selected_sensors:
                selected[start:stop,sensor]=True
                blocks.append({"sensor_index":sensor,"start":start,"stop":stop})
        rate_basis="fraction of sensors, one block per sensor/episode"
    if spec.kind=="missing":
        values[selected]=np.nan
    elif spec.kind=="noisy":
        mask=selected&usable(clean)
        perturbed=np.maximum(0,clean+normal*scales[None,:]*spec.severity)
        values[mask]=perturbed[mask]
    else:
        for block in blocks:
            s,a,b=block["sensor_index"],block["start"],block["stop"]
            earlier=np.flatnonzero(usable(clean[:a,s]))
            constant=clean[earlier[-1],s] if len(earlier) else stats.medians[s]
            values[a:b,s]=constant  # clean prefix only; no recursive fault history
    equal=(clean==values)|(np.isnan(clean)&np.isnan(values))
    details={"kind":spec.kind,"mode":spec.mode,"rate":spec.rate,"seed":spec.seed,
             "sampling_policy":spec.sampling_policy,"selection_denominator":denominator,
             "selection_count":count,"achieved_selection_rate":count/denominator if denominator else 0.0,
             "rounding_delta":count/denominator-spec.rate if denominator else None,
             "selected_cell_fraction":float(selected.mean()),
             "changed_cell_fraction":float((~equal).mean()),
             "severity":spec.severity,"rng_namespace":[spec.seed,partition_code,KINDS[spec.kind],MODES[spec.mode]],
             "rate_basis":rate_basis,"selected_cell_count":int(selected.sum()),
             "selected_sensor_indices":selected_sensors,"blocks":blocks,
             "original_usable_cell_count":int(usable(clean).sum()),
             "natural_unavailable_overlap_count":int((selected&~usable(clean)).sum()),
             "changed_cell_count":int((~equal).sum()),
             "changed_originally_usable_count":int((~equal&usable(clean)).sum()),
             "selected_mask_sha256":array_hash(selected),"overlay_sha256":array_hash(values)}
    values.setflags(write=False); selected.setflags(write=False)
    return FaultOverlay(values,selected,array_hash(clean),array_hash(partition.timestamps),
                        partition.offset,partition.sensors,details)


def historical_inputs(partition: Partition, overlay: FaultOverlay, *, role="tuning"):
    """Return forecast histories only, always using clean partition window origins."""
    if (role not in ("tuning","controller_selection") or partition.role!=role or partition.offset!=overlay.offset or
            partition.sensors!=overlay.sensors or partition.values.shape!=overlay.values.shape or
            array_hash(partition.values)!=overlay.source_hash or
            array_hash(partition.timestamps)!=overlay.time_hash):
        raise ValueError("Overlay cannot cross source partition boundaries")
    result=[]
    for window in windows(partition):
        t=window.origin_index-partition.offset
        result.append(ForecastInput(overlay.values[t-11:t+1],window.inputs.origin_time,partition.sensors))
    return result
