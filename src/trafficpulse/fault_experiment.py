"""Exploratory synthetic tuning faults only; no model selection or final test."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import psutil
from threadpoolctl import threadpool_info,threadpool_limits

from .baselines import Persistence,SeasonalHistoricalAverage
from .data import HORIZONS,TrainingStatistics,chronological_partitions,windows
from .faults import FaultSpec,array_hash,historical_inputs,inject,noise_scales
from .synthetic import make_synthetic
from .tracking import log_local_run
from .xgboost_baseline import XGBoostDirect
from .xgboost_experiment import score


def compare_models(models,inputs,targets):
    """Every model sees the same shared, read-only histories and clean targets."""
    if not inputs:
        raise ValueError("No evaluation windows")
    metrics={}
    for name,model in models.items():
        if isinstance(model,XGBoostDirect):
            predictions=model.predict_many(inputs)
        else:
            predictions=np.stack([model.predict(item)[np.array(HORIZONS)-1] for item in inputs])
        metrics[name]=score(targets,predictions)
    return metrics


def aggregate(cases,clean):
    result=[]
    groups={}
    for case in cases:
        key=(case["fault"]["kind"],case["fault"]["mode"],case["fault"]["rate"])
        groups.setdefault(key,[]).append(case)
    for (kind,mode,rate),group in groups.items():
        metrics={}
        for model in clean:
            metrics[model]={}
            for h in clean[model]:
                items=[c["metrics"][model][h] for c in group]
                counts={m["valid_target_count"] for m in items}
                if len(counts)!=1:
                    raise ValueError("Seed comparisons must retain original target denominators")
                mae=float(np.mean([m["mae"] for m in items]))
                rmse=float(np.mean([m["rmse"] for m in items]))
                metrics[model][h]={"mae":mae,"rmse":rmse,"valid_target_count":counts.pop(),
                    "delta_mae":mae-clean[model][h]["mae"],"delta_rmse":rmse-clean[model][h]["rmse"],
                    "min_seed_mae":min(m["mae"] for m in items),"max_seed_mae":max(m["mae"] for m in items)}
        result.append({"kind":kind,"mode":mode,"rate":rate,"seeds":[c["fault"]["seed"] for c in group],
                       "aggregation":"arithmetic mean of per-seed metrics; counts per realization, not summed",
                       "metrics":metrics,
                       "mean_changed_cells":float(np.mean([c["fault"]["changed_cell_count"] for c in group]))})
    return result


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    root=Path(__file__).resolve().parents[2]
    config_path=root/"configs/phase3a_synthetic_faults.json"
    config=json.loads(config_path.read_text(encoding="utf-8"))
    protocol=json.loads((root/"configs/evaluation_protocol.json").read_text(encoding="utf-8"))
    if (config["partition"]!="tuning" or config["rates"]!=protocol["faults"]["rates"] or
            config["seeds"]!=protocol["faults"]["selection_seeds"] or config["synthetic_tuning_partition_code"]!=1):
        raise ValueError("Require explicit synthetic tuning-only development settings")
    previous=json.loads((root/"reports/generated/phase2b_synthetic_xgboost.json").read_text(encoding="utf-8"))
    process=psutil.Process(); started=time.perf_counter(); cpu=time.process_time()
    flat_metrics={}; cases=[]
    with threadpool_limits(limits=4):
        fixture=make_synthetic()
        if hashlib.sha256(fixture.values.tobytes()).hexdigest()!=previous["synthetic_values_sha256"]:
            raise ValueError("Synthetic fixture differs from selected model source")
        destination=root/"artifacts/synthetic_xgboost"
        for item in previous["artifact_manifest"]:
            if hashlib.sha256((destination/item["name"]).read_bytes()).hexdigest()!=item["sha256"]:
                raise ValueError("Phase 2B synthetic artifact has changed")
        parts=chronological_partitions(fixture.timestamps,fixture.values,fixture.sensors)
        # Deliberately never inspect/materialize the reserved partition windows.
        train,tuning=parts["train"],parts["tuning"]
        models={"persistence":Persistence().fit(train),
                "seasonal_historical_average":SeasonalHistoricalAverage().fit(train),
                "xgboost_selected":XGBoostDirect.restore_synthetic(train,destination)}
        stats=TrainingStatistics.fit(train); scales=noise_scales(train)
        clean_windows=list(windows(tuning))
        clean_inputs=[w.inputs for w in clean_windows]
        targets=np.stack([w.targets[np.array(HORIZONS)-1] for w in clean_windows])
        target_hash=array_hash(targets); source_hash=array_hash(tuning.values)
        clean=compare_models(models,clean_inputs,targets)
        for model,horizons in clean.items():
            for h,metrics in horizons.items():
                expected=previous["metrics"][model][h]
                if metrics!=expected:
                    raise ValueError("Clean result differs from reviewed synthetic baseline")
            flat_metrics[f"{model}__clean"]=horizons
        for kind,mode in config["families"]:
            for rate in config["rates"]:
                if rate==0:
                    continue  # single shared clean run
                for seed in config["seeds"]:
                    spec=FaultSpec(kind,mode,rate,seed,config["noise_std_multiplier"],config["block_rows"],config["episode_rows"],config["sampling_policy"])
                    overlay=inject(tuning,spec,stats,scales)
                    inputs=historical_inputs(tuning,overlay)
                    if len(inputs)!=len(clean_inputs):
                        raise ValueError("Fault injection changed the inference origin set")
                    metrics=compare_models(models,inputs,targets)
                    delta={}
                    for model,horizons in metrics.items():
                        delta[model]={}
                        for h,item in horizons.items():
                            if item["valid_target_count"]!=clean[model][h]["valid_target_count"]:
                                raise ValueError("Fault injection changed target masking")
                            delta[model][h]={key:item[key]-clean[model][h][key] for key in ("mae","rmse")}
                        key=f"{model}__{kind}_{mode}_r{str(rate).replace('.','p')}_s{seed}"
                        flat_metrics[key]=horizons
                    exposure=0
                    history_cells=np.zeros(tuning.values.shape,dtype=bool)
                    for window in clean_windows:
                        t=window.origin_index-tuning.offset
                        exposure+=int(overlay.selected[t-11:t+1].any())
                        history_cells[t-11:t+1]=True
                    equal=(overlay.values==tuning.values)|(np.isnan(overlay.values)&np.isnan(tuning.values))
                    changed_history=int((history_cells&~equal).sum())
                    if not changed_history:
                        raise ValueError("Positive synthetic experiment rate produced no changed historical inputs")
                    details={**overlay.details,"selected_window_count":exposure,
                             "changed_historical_cell_count":changed_history,
                             "total_window_count":len(inputs),"clean_target_sha256":target_hash}
                    cases.append({"fault":details,"metrics":metrics,"delta_from_clean":delta})
                    if array_hash(targets)!=target_hash or array_hash(tuning.values)!=source_hash:
                        raise ValueError("Clean ground truth/source was modified")
        summaries=aggregate(cases,clean)
        ranking={}
        for model in clean:
            ranked=sorted(summaries,key=lambda g:np.mean([m["delta_mae"] for m in g["metrics"][model].values()]),reverse=True)
            worst=ranked[0]
            ranking[model]={"kind":worst["kind"],"mode":worst["mode"],"rate":worst["rate"],
                            "mean_horizon_delta_mae":float(np.mean([m["delta_mae"] for m in worst["metrics"][model].values()]))}
        pools=[{"internal_api":p["internal_api"],"num_threads":p["num_threads"]} for p in threadpool_info()]
        if any(p["num_threads"]>4 for p in pools):
            raise RuntimeError("Thread limit exceeded")
        resources={"compute_seconds":time.perf_counter()-started,"compute_cpu_seconds":time.process_time()-cpu,
                   "peak_working_set_bytes_through_compute":getattr(process.memory_info(),"peak_wset",process.memory_info().rss),
                   "fault_realizations":len(cases)}
    artifact_dir=root/"artifacts/synthetic_faults"/config["experiment_id"]; artifact_dir.mkdir(parents=True,exist_ok=True)
    result={"data_origin":fixture.origin,"units":fixture.units,"real_dataset_access":"none",
            "parameters":{"partition":"synthetic_tuning_only","fault_config":json.dumps(config,sort_keys=True),
                          "model_source_run_id":previous["tracking"]["run_id"],"seed_policy":"all three; no seed/model selection",
                          "cpu_threads_max":4,"timeout_seconds":1800},
            "metrics":flat_metrics,"clean":clean,"cases":cases,"seed_summaries":summaries,
            "worst_mean_horizon_degradation":ranking,"resources":resources,"numerical_threadpools":pools,
            "source_tuning_sha256":source_hash,"clean_target_sha256":target_hash,
            "noise_std_per_synthetic_sensor":scales.tolist(),"window_count":len(clean_inputs),
            "limitations":["Synthetic tuning exploration, not held-out robustness evidence",
                           "Synthetic-only ceiling: requested contiguous 10/20/40% selects 25/25/50% of four sensors; frozen floor unchanged",
                           "Point and contiguous rates have different denominators",
                           "Frozen real-data protocol unchanged; synthetic tuning is separately authorized",
                           "Full synthetic stream is generated, but reserved partition windows are never inspected or scored"]}
    artifact=artifact_dir/"fault_comparison.json"
    artifact.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    config_copy=artifact_dir/"experiment_config.json"; config_copy.write_bytes(config_path.read_bytes())
    result["tracking"]=log_local_run(result,root,experiment_name="trafficpulse-synthetic-phase3a",
                                    run_name="deterministic-input-faults",artifacts=[artifact,config_copy])
    result["total_elapsed_seconds"]=time.perf_counter()-started
    result["artifacts"]=[{"name":p.name,"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"size_bytes":p.stat().st_size}
                         for p in (artifact,config_copy)]
    output=root/"reports/generated/phase3a_synthetic_faults.json"
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({key:result[key] for key in ("window_count","worst_mean_horizon_degradation","resources","total_elapsed_seconds","tracking")},indent=2))


if __name__=="__main__":
    main()
