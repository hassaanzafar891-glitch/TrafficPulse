"""Synthetic controller-selection diagnostics; no test or interval evaluation."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import psutil
from threadpoolctl import threadpool_info,threadpool_limits

from .baselines import SeasonalHistoricalAverage
from .data import HORIZONS,Partition,TrainingStatistics,windows
from .faults import FaultSpec,array_hash,historical_inputs,inject_controller,noise_scales
from .reliability import Thresholds,choose_thresholds,controller_predictions,decision_summary
from .synthetic import make_synthetic
from .tracking import log_local_run
from .xgboost_baseline import XGBoostDirect
from .xgboost_experiment import score


def safe_thresholds(thresholds):
    result=asdict(thresholds)
    if np.isinf(result['disagreement']):result['disagreement']='infinity'
    return result


def controller_prefix_parts(fixture):
    """Original 8064-row boundaries, but no final-test or calibration objects."""
    if len(fixture.values)!=6451 or len(fixture.timestamps)!=6451:
        raise ValueError('Require original prefix ending before final test')
    return (Partition('train',fixture.timestamps[:5644],fixture.values[:5644],fixture.sensors,0),
            Partition('controller_selection',fixture.timestamps[6208:6451],fixture.values[6208:6451],fixture.sensors,6208))


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    root=Path(__file__).resolve().parents[2]
    config_path=root/'configs/phase3c_synthetic_controller.json'
    config=json.loads(config_path.read_text(encoding='utf-8'))
    protocol=json.loads((root/'configs/evaluation_protocol.json').read_text(encoding='utf-8'))
    gate=protocol['controller_selection'];faults=protocol['faults']
    if (config['partition']!='controller_selection' or config['partition_code']!=3 or not config['joint_decision'] or
            config['unavailable_thresholds']!=gate['unavailable_fraction_thresholds'] or
            config['disagreement_thresholds']!=gate['model_disagreement_sigma_thresholds'] or
            config['clean_relative_mae_limit']!=1+gate['clean_mae_relative_degradation_limit'] or
            config['rates']!=faults['rates'] or config['seeds']!=faults['selection_seeds'] or
            config['sampling_policy']!='synthetic_ceil' or config['max_threads']!=4 or config['timeout_seconds']!=1800 or
            config['block_rows']!=faults['contiguous_block_rows'] or config['episode_rows']!=faults['episode_rows'] or
            config['noise_std_multiplier']!=faults['noise_std_multiplier'] or
            config['families']!=[[k,m] for k,ms in faults['modes_by_kind'].items() for m in ms]):
        raise ValueError('Unexpected controller scope or frozen grid/fault settings')
    previous=json.loads((root/'reports/generated/phase2b_synthetic_xgboost.json').read_text(encoding='utf-8'))
    destination=root/'artifacts/synthetic_xgboost'
    for item in previous['artifact_manifest']:
        if Path(item['name']).name!=item['name'] or hashlib.sha256((destination/item['name']).read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Selected synthetic artifact differs')
    started=time.perf_counter();cpu=time.process_time();process=psutil.Process()
    with threadpool_limits(limits=4):
        fixture=make_synthetic(stop_rows=6451)
        train,part=controller_prefix_parts(fixture)
        primary=XGBoostDirect.restore_synthetic(train,destination)
        fallback=SeasonalHistoricalAverage().fit(train)
        stats=TrainingStatistics.fit(train);scales=noise_scales(train)
        samples=list(windows(part));clean_inputs=[w.inputs for w in samples]
        targets=np.stack([w.targets[np.array(HORIZONS)-1] for w in samples]);targets.setflags(write=False)
        source_hash=array_hash(part.values);target_hash=array_hash(targets)
        conditions=[]
        def add_condition(name,inputs,fault=None):
            p=primary.predict_many(inputs)
            f=np.stack([fallback.predict(item)[np.array(HORIZONS)-1] for item in inputs])
            p.setflags(write=False);f.setflags(write=False)
            conditions.append({'name':name,'inputs':inputs,'targets':targets,'primary':p,'fallback':f,
                               'fault':fault,'partition_offset':part.offset})
        add_condition('clean',clean_inputs)
        for kind,mode in config['families']:
            for rate in config['rates'][1:]:
                for seed in config['seeds']:
                    spec=FaultSpec(kind,mode,rate,seed,config['noise_std_multiplier'],config['block_rows'],config['episode_rows'],config['sampling_policy'])
                    overlay=inject_controller(part,spec,stats,scales)
                    inputs=historical_inputs(part,overlay,role='controller_selection')
                    changed=(overlay.values!=part.values)&~(np.isnan(overlay.values)&np.isnan(part.values))
                    histories=np.zeros(part.values.shape,dtype=bool)
                    exposed=0
                    for w in samples:
                        t=w.origin_index-part.offset;histories[t-11:t+1]=True
                        exposed+=int(overlay.selected[t-11:t+1].any())
                    changed_history=int((changed&histories).sum())
                    if not changed_history:raise ValueError('Positive synthetic fault did not change history')
                    fault={**overlay.details,'changed_historical_cell_count':changed_history,'exposed_window_count':exposed,
                           'clean_target_sha256':target_hash}
                    add_condition(f'{kind}_{mode}_r{rate}_s{seed}',inputs,fault)
        selection_started=time.perf_counter()
        selected,candidates=choose_thresholds(part,conditions,stats)
        selection_seconds=time.perf_counter()-selection_started
        metrics={};cases=[];decision_records=[];gate_seconds=0.0
        for condition in conditions:
            s=time.perf_counter()
            prediction,decisions=controller_predictions(condition['inputs'],condition['primary'],condition['fallback'],stats,selected)
            gate_seconds+=time.perf_counter()-s
            # Predeclared simple availability heuristic, not an expanded search.
            heuristic,_=controller_predictions(condition['inputs'],condition['primary'],condition['fallback'],stats,Thresholds(.2,np.inf))
            choices={'always_xgboost':condition['primary'],'always_seasonal':condition['fallback'],
                     'availability_heuristic':heuristic,'controller':prediction}
            outcome={name:score(targets,value) for name,value in choices.items()}
            summary=decision_summary(decisions)
            for name,result in outcome.items():
                for h in result:
                    if name=='controller':
                        result[h].update({k:summary[k] for k in ('fallback_frequency','primary_frequency','abstention_frequency','switches_per_1000_origins')})
                    metrics[f'{name}__{condition["name"]}']=result
            cases.append({'name':condition['name'],'fault':condition['fault'],'metrics':outcome,'decisions':summary})
            decision_records.append({'condition':condition['name'],'origins':[
                {'origin_index':w.origin_index,**d} for w,d in zip(samples,decisions)]})
            if array_hash(targets)!=target_hash or array_hash(part.values)!=source_hash:
                raise ValueError('Ground truth/source changed')
        means={name:{str(h):{key:float(np.mean([c['metrics'][name][str(h)][key] for c in cases[1:]]))
                   for key in ('mae','rmse')} for h in HORIZONS} for name in choices}
        pools=[{'internal_api':p['internal_api'],'num_threads':p['num_threads']} for p in threadpool_info()]
        if any(p['num_threads']>4 for p in pools):raise RuntimeError('Thread cap exceeded')
        resources={'compute_seconds':time.perf_counter()-started,'compute_cpu_seconds':time.process_time()-cpu,
                   'peak_working_set_bytes_through_compute':getattr(process.memory_info(),'peak_wset',process.memory_info().rss),
                   'threshold_selection_seconds':selection_seconds,'selected_gate_seconds':gate_seconds,
                   'selected_gate_ms_per_origin':1000*gate_seconds/(len(samples)*len(cases))}
    result={'data_origin':fixture.origin,'units':fixture.units,'real_dataset_access':'none',
            'parameters':{'config':json.dumps(config,sort_keys=True),'selected_thresholds':json.dumps(safe_thresholds(selected),sort_keys=True),
                          'model_source_run_id':previous['tracking']['run_id'],'cpu_threads_max':4,'timeout_seconds':1800,
                          'partition':'synthetic_controller_selection_only','interval_validity':'unresolved; no intervals emitted'},
            'metrics':metrics,'cases':cases,'fault_mean_per_horizon':means,'selected_thresholds':safe_thresholds(selected),
            'candidates':[{**c,'thresholds':safe_thresholds(c['thresholds'])} for c in candidates],
            'resources':resources,'numerical_threadpools':pools,'window_count':len(samples),
            'partition_rows':[6208,6451],'origin_bounds':[samples[0].origin_index,samples[-1].origin_index],
            'source_sha256':source_hash,'target_sha256':target_hash,
            'limitations':['Selection-set diagnostics, not held-out controller evaluation',
                           'Switched/fallback interval validity unresolved; calibration untouched',
                           'Synthetic ceiling rates differ from frozen real-data floor; actual rates reported',
                           'Repeated/stuck and abrupt quality signals are descriptive, not confirmed fault labels',
                           'Prefix contains prior roles but only train/controller objects are inspected; final observations not constructed']}
    output_dir=root/'artifacts/synthetic_controller'/config['experiment_id'];output_dir.mkdir(parents=True,exist_ok=True)
    artifact=output_dir/'controller_comparison.json';artifact.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    decisions_artifact=output_dir/'synthetic_decisions.json';decisions_artifact.write_text(json.dumps(decision_records,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    config_artifact=output_dir/'experiment_config.json';config_artifact.write_bytes(config_path.read_bytes())
    artifacts=[artifact,decisions_artifact,config_artifact]
    result['tracking']=log_local_run(result,root,experiment_name='trafficpulse-synthetic-phase3c',run_name='joint-controller-selection',artifacts=artifacts)
    result['total_elapsed_seconds']=time.perf_counter()-started
    result['artifacts']=[{'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size} for p in artifacts]
    (root/'reports/generated/phase3c_synthetic_controller.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('selected_thresholds','fault_mean_per_horizon','resources','window_count','total_elapsed_seconds','tracking')},indent=2))


if __name__=='__main__':main()
