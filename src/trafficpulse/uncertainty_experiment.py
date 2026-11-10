"""Bounded synthetic calibration diagnostics only; no held-out/fault evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import psutil
from threadpoolctl import threadpool_info,threadpool_limits

from .data import Partition
from .faults import array_hash
from .synthetic import make_synthetic
from .tracking import log_local_run
from .uncertainty import SplitConformal,calibration_targets
from .xgboost_baseline import XGBoostDirect


def calibration_prefix_parts(fixture,total_rows):
    """Keep original boundaries; never construct tuning/controller/test partitions."""
    train_end=7*total_rows//10; val_end=8*total_rows//10
    tune_end=train_end+4*(val_end-train_end)//10
    cal_end=tune_end+3*(val_end-train_end)//10
    if len(fixture.values)!=cal_end or len(fixture.timestamps)!=cal_end:
        raise ValueError("Require exactly the original fixture prefix through calibration")
    return (Partition('train',fixture.timestamps[:train_end],fixture.values[:train_end],fixture.sensors,0),
            Partition('calibration',fixture.timestamps[tune_end:cal_end],fixture.values[tune_end:cal_end],fixture.sensors,tune_end))


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    root=Path(__file__).resolve().parents[2]
    config_path=root/'configs/phase3b_synthetic_uncertainty.json'
    config=json.loads(config_path.read_text(encoding='utf-8'))
    protocol=json.loads((root/'configs/evaluation_protocol.json').read_text(encoding='utf-8'))
    if (config['nominal_coverage']!=protocol['intervals']['nominal_coverage'] or
            config['horizons']!=protocol['window']['evaluation_horizons'] or
            config['calibration_role']!='calibration' or config['diagnostic_role']!='calibration' or
            config['fault_evaluation'] or config['max_threads']!=4 or config['timeout_seconds']!=1800 or
            config['fixture']!={'days':28,'sensors':4,'seed':1729}):
        raise ValueError('Require frozen synthetic-only calibration settings')
    previous=json.loads((root/'reports/generated/phase2b_synthetic_xgboost.json').read_text(encoding='utf-8'))
    destination=root/'artifacts/synthetic_xgboost'
    manifest=previous['artifact_manifest']
    expected={'horizon_3.json','horizon_6.json','horizon_12.json','preprocessing.json','comparison.json','experiment_config.json'}
    if {a['name'] for a in manifest}!=expected:
        raise ValueError('Unexpected selected synthetic model manifest')
    for a in manifest:
        if hashlib.sha256((destination/a['name']).read_bytes()).hexdigest()!=a['sha256']:
            raise ValueError('Selected synthetic model artifact changed')
    process=psutil.Process(); started=time.perf_counter(); cpu=time.process_time()
    with threadpool_limits(limits=4):
        total_rows=28*288
        train_end=7*total_rows//10; val_end=8*total_rows//10
        cal_end=train_end+4*(val_end-train_end)//10+3*(val_end-train_end)//10
        fixture=make_synthetic(**config['fixture'],stop_rows=cal_end)
        train,calibration=calibration_prefix_parts(fixture,total_rows)
        source_hash=array_hash(calibration.values)
        model=XGBoostDirect.restore_synthetic(train,destination)
        samples,targets=calibration_targets(calibration)
        predictions=model.predict_many([w.inputs for w in samples])
        calibrator=SplitConformal.fit(calibration,predictions,config['nominal_coverage'])
        diagnostics=calibrator.calibration_diagnostics(calibration,predictions)
        # A second inference checks deterministic predictions/widths, without refitting.
        repeated=model.predict_many([w.inputs for w in samples])
        if not np.array_equal(predictions,repeated):
            raise ValueError('Synthetic model inference was not deterministic')
        if SplitConformal.fit(calibration,repeated,config['nominal_coverage'])!=calibrator:
            raise ValueError('Calibration was not deterministic')
        if array_hash(calibration.values)!=source_hash:
            raise ValueError('Calibration source changed')
        pools=[{'internal_api':p['internal_api'],'num_threads':p['num_threads']} for p in threadpool_info()]
        if any(p['num_threads']>4 for p in pools):
            raise RuntimeError('Thread cap exceeded')
        resources={'compute_seconds':time.perf_counter()-started,'compute_cpu_seconds':time.process_time()-cpu,
                   'peak_working_set_bytes_through_compute':getattr(process.memory_info(),'peak_wset',process.memory_info().rss)}
    calibration_summary={'nominal_coverage':calibrator.coverage,'horizons':[3,6,12],
                         'half_widths':list(calibrator.half_widths),'valid_counts':list(calibrator.counts),
                         'one_based_ranks':list(calibrator.ranks),'pooling':'sensors and origins within each horizon',
                         'empty_policy':'unavailable; null width/coverage, never a fabricated quantile'}
    result={'data_origin':fixture.origin,'units':fixture.units,'real_dataset_access':'none',
            'parameters':{'config':json.dumps(config,sort_keys=True),'nominal_coverage':calibrator.coverage,
                          'model_source_run_id':previous['tracking']['run_id'],'model_refitted':False,
                          'diagnostic_scope':'calibration_only_in_sample','independent_coverage_measured':False,
                          'cpu_threads_max':4,'timeout_seconds':1800},
            'metrics':{'xgboost_calibration_diagnostic':diagnostics},'calibration':calibration_summary,
            'resources':resources,'numerical_threadpools':pools,'calibration_source_sha256':source_hash,
            'calibration_target_sha256':array_hash(targets),'predictions_sha256':array_hash(predictions),
            'partitions_used':{'train':[0,len(train.values)],'calibration':[calibration.offset,cal_end]},
            'calibration_window_count':len(samples),
            'calibration_origin_bounds':[samples[0].origin_index,samples[-1].origin_index],
            'final_test_access':'none; suffix observations not constructed',
            'independent_coverage_measured':False,'fault_coverage_measured':False,
            'limitations':['Calibration diagnostics reuse width-fitting samples, not independent coverage evidence',
                           'Temporal overlap and correlated sensors do not establish exchangeability',
                           'Coverage is not guaranteed under sensor faults or distribution shift',
                           'Only horizons 3/6/12 implemented for the selected synthetic XGBoost; frozen broader protocol unchanged',
                           'Original RNG noise draws are advanced/discarded and full placement indices generated, but suffix observations are not constructed']}
    artifact_dir=root/'artifacts/synthetic_uncertainty'/config['experiment_id']
    artifact_dir.mkdir(parents=True,exist_ok=True)
    artifact=artifact_dir/'calibration_summary.json'
    artifact.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    config_copy=artifact_dir/'experiment_config.json';config_copy.write_bytes(config_path.read_bytes())
    result['tracking']=log_local_run(result,root,experiment_name='trafficpulse-synthetic-phase3b',
                                    run_name='clean-split-calibration-diagnostics',artifacts=[artifact,config_copy])
    result['total_elapsed_seconds']=time.perf_counter()-started
    result['artifacts']=[{'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size}
                         for p in (artifact,config_copy)]
    output=root/'reports/generated/phase3b_synthetic_uncertainty.json'
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('calibration','metrics','calibration_window_count','resources','total_elapsed_seconds','tracking')},indent=2))


if __name__=='__main__':
    main()
