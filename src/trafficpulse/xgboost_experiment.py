"""Bounded synthetic-only XGBoost comparison; never evaluates final test."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import psutil
from threadpoolctl import threadpool_info, threadpool_limits

from .baselines import Persistence, SeasonalHistoricalAverage
from .data import HORIZONS, chronological_partitions, windows
from .metrics import masked_metrics
from .synthetic import make_synthetic
from .tracking import log_local_run
from .xgboost_baseline import XGBoostDirect


def score(targets,predictions):
    if targets.shape != predictions.shape or targets.ndim != 3 or targets.shape[1] != 3:
        raise ValueError("Expected [windows,3,sensors] arrays")
    return {str(h):masked_metrics(targets[:,i,:],predictions[:,i,:]) for i,h in enumerate(HORIZONS)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()  # no dataset or remote endpoint arguments
    workspace=Path(__file__).resolve().parents[2]
    config_path=workspace/"configs/phase2b_synthetic_xgboost.json"
    config=json.loads(config_path.read_text(encoding="utf-8"))
    if (config["horizons"]!=list(HORIZONS) or config["max_threads"]!=4 or
            len(config["candidates"])!=2 or config["days"]!=28 or config["sensors"]!=4):
        raise ValueError("Experiment must retain the small predeclared synthetic scope")
    process=psutil.Process()
    start=time.perf_counter(); cpu_start=time.process_time(); baseline_rss=process.memory_info().rss
    metrics={}; candidates=[]; fitted=[]
    with threadpool_limits(limits=4):
        fixture=make_synthetic(config["days"],config["sensors"],config["seed"])
        parts=chronological_partitions(fixture.timestamps,fixture.values,fixture.sensors)
        tuning=list(windows(parts["tuning"]))
        inputs=[w.inputs for w in tuning]
        targets=np.stack([w.targets[np.array(HORIZONS)-1] for w in tuning])
        for name,cls in (("persistence",Persistence),("seasonal_historical_average",SeasonalHistoricalAverage)):
            model=cls().fit(parts["train"])
            predictions=np.stack([model.predict(item)[np.array(HORIZONS)-1] for item in inputs])
            metrics[name]=score(targets,predictions)
        for index,choice in enumerate(config["candidates"]):
            parameters={**config["common_parameters"],**choice}
            if not 1<=parameters["max_depth"]<=3 or not 1<=parameters["n_estimators"]<=60:
                raise ValueError("Tree budget exceeded")
            model=XGBoostDirect(parameters).fit(parts["train"])
            candidate_metrics=score(targets,model.predict_many(inputs))
            mean_mae=float(np.mean([item["mae"] for item in candidate_metrics.values()]))
            candidates.append({"index":index,"parameters":parameters,"mean_tuning_mae":mean_mae,
                               "training_duration_seconds":model.training_seconds,
                               "training_target_counts":model.training_target_counts})
            metrics[f"xgboost_candidate_{index}"]=candidate_metrics
            fitted.append(model)
        selected=min(range(len(candidates)),key=lambda i:(candidates[i]["mean_tuning_mae"],i))
        metrics["xgboost_selected"]=metrics[f"xgboost_candidate_{selected}"]
        for horizons in metrics.values():
            assert [item["valid_target_count"] for item in horizons.values()]==[
                item["valid_target_count"] for item in metrics["persistence"].values()]
        pools=[{"internal_api":p["internal_api"],"num_threads":p["num_threads"]} for p in threadpool_info()]
        if any(p["num_threads"]>4 for p in pools):
            raise RuntimeError("Thread budget exceeded")
        memory=process.memory_info()
        resources={"experiment_compute_seconds":time.perf_counter()-start,
                   "process_cpu_seconds":time.process_time()-cpu_start,
                   "training_duration_seconds":sum(c["training_duration_seconds"] for c in candidates),
                   "selected_training_duration_seconds":candidates[selected]["training_duration_seconds"],
                   "baseline_rss_bytes":baseline_rss,"after_compute_rss_bytes":memory.rss,
                   "peak_working_set_bytes":getattr(memory,"peak_wset",memory.rss),
                   "largest_training_matrix_bytes":max(m.training_matrix_bytes for m in fitted)}
    destination=workspace/"artifacts/synthetic_xgboost"
    artifact_files=fitted[selected].save_synthetic_artifacts(destination)
    result={"data_origin":fixture.origin,"units":fixture.units,"real_dataset_access":"none",
            "config_sha256":hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "synthetic_values_sha256":hashlib.sha256(fixture.values.tobytes()).hexdigest(),
            "parameters":{"seed":config["seed"],"days":config["days"],"sensors":config["sensors"],
                          "strategy":config["strategy"],"scored_partition":"tuning","horizons":"3,6,12",
                          "cpu_threads_max":4,"timeout_seconds":1800,"selected_candidate":selected,
                          "selected_parameters":json.dumps(candidates[selected]["parameters"],sort_keys=True)},
            "scored_windows":len(tuning),"metrics":metrics,"candidates":candidates,
            "selected_candidate":selected,"resources":resources,"numerical_threadpools":pools,
            "memory_scope":"Windows lifetime peak includes imports; matrix bytes exclude temporary copies and native trees",
            "selection_limit":"Tuning scores used for candidate choice; not held-out performance; no calibration/controller/test evaluated"}
    manifest=destination/"comparison.json"
    manifest.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    result["tracking"]=log_local_run(result,workspace,experiment_name="trafficpulse-synthetic-phase2b",
                                    run_name="synthetic-xgboost-comparison",artifacts=[*artifact_files,manifest,config_path_copy(config_path,destination)])
    result["total_elapsed_seconds"]=time.perf_counter()-start
    result["total_process_cpu_seconds"]=time.process_time()-cpu_start
    result["artifact_manifest"]=[{"name":p.name,"size_bytes":p.stat().st_size,
                                  "sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
                                 for p in [*artifact_files,manifest,destination/"experiment_config.json"]]
    output=workspace/"reports/generated/phase2b_synthetic_xgboost.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,allow_nan=False))


def config_path_copy(source,destination):
    copy=destination/"experiment_config.json"
    copy.write_bytes(source.read_bytes())
    return copy


if __name__=="__main__":
    main()
