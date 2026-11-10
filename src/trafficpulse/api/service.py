"""Read-only synthetic model bundle and history-only replay inference."""
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from threading import Lock

import numpy as np
from threadpoolctl import threadpool_limits

from ..baselines import SeasonalHistoricalAverage
from ..data import ForecastInput,HORIZONS,Partition,STEP,usable
from ..faults import FaultSpec,inject_controller,noise_scales
from ..reliability import Thresholds,select_forecast
from ..synthetic import make_synthetic
from ..xgboost_baseline import XGBoostDirect


class ReplayInputError(ValueError):
    def __init__(self,message,status=422):
        super().__init__(message);self.status=status


def stamp(time):
    return np.datetime_as_string(time,unit='s')


def verify_manifest(directory,manifest,expected):
    if {a['name'] for a in manifest}!=expected or len(manifest)!=len(expected):
        raise ValueError('Unexpected synthetic artifact manifest')
    for a in manifest:
        p=directory/a['name']
        if hashlib.sha256(p.read_bytes()).hexdigest()!=a['sha256']:
            raise ValueError('Synthetic artifact integrity check failed')


class ReplayService:
    FIRST=6219
    LAST=6438

    def __init__(self,workspace: Path):
        self._lock=Lock()
        root=Path(workspace)
        # Fixed local sources only: no client supplied paths or real-data loaders.
        previous=json.loads((root/'reports/generated/phase2b_synthetic_xgboost.json').read_text(encoding='utf-8'))
        models=root/'artifacts/synthetic_xgboost'
        verify_manifest(models,previous['artifact_manifest'],
                        {'horizon_3.json','horizon_6.json','horizon_12.json','preprocessing.json','comparison.json','experiment_config.json'})
        controller=json.loads((root/'reports/generated/phase3c_synthetic_controller.json').read_text(encoding='utf-8'))
        controller_dir=root/'artifacts/synthetic_controller/phase3c-synthetic-joint-controller-v1'
        verify_manifest(controller_dir,controller['artifacts'],{'controller_comparison.json','synthetic_decisions.json','experiment_config.json'})
        reviewed=json.loads((controller_dir/'controller_comparison.json').read_text(encoding='utf-8'))
        calibration=json.loads((root/'reports/generated/phase3b_synthetic_uncertainty.json').read_text(encoding='utf-8'))
        calibration_dir=root/'artifacts/synthetic_uncertainty/phase3b-synthetic-split-conformal-v1'
        verify_manifest(calibration_dir,calibration['artifacts'],{'calibration_summary.json','experiment_config.json'})
        calibrated=json.loads((calibration_dir/'calibration_summary.json').read_text(encoding='utf-8'))
        for item in (previous,reviewed,calibrated):
            if item['data_origin']!='synthetic_generated_not_real_traffic':
                raise ValueError('Require synthetic artifact provenance')
        source_run=previous['tracking']['run_id']
        if reviewed['parameters']['model_source_run_id']!=source_run or calibrated['parameters']['model_source_run_id']!=source_run:
            raise ValueError('Synthetic model provenance mismatch')
        self.thresholds=Thresholds(**reviewed['selected_thresholds'])
        if self.thresholds!=Thresholds(.1,1.,False):
            raise ValueError('Require the reviewed controller settings')
        cal=calibrated['calibration']
        if cal['nominal_coverage']!=.9 or cal['horizons']!=list(HORIZONS):
            raise ValueError('Unexpected interval calibration schema')
        if len(cal['half_widths'])!=3 or len(cal['valid_counts'])!=3:
            raise ValueError('Incomplete interval calibration')
        self.widths=tuple(cal['half_widths'])
        if any(q is None or not math.isfinite(q) or q<0 for q in self.widths) or any(n<=0 for n in cal['valid_counts']):
            raise ValueError('Unavailable interval calibration')
        with threadpool_limits(limits=4):
            fixture=make_synthetic(stop_rows=6451)
            self.sensors=fixture.sensors;self.units=fixture.units;self.origin=fixture.origin
            train=Partition('train',fixture.timestamps[:5644],fixture.values[:5644],fixture.sensors,0)
            self.primary=XGBoostDirect.restore_synthetic(train,models)
            self.fallback=SeasonalHistoricalAverage().fit(train)
            self.statistics=self.primary.features.statistics
            self.scales=noise_scales(train)
            # Store only permitted replay history; no target/window objects.
            self.times=fixture.timestamps[6208:6451]
            self.readings=fixture.values[6208:6451]

    def metadata(self):
        return {'contract_version':'1.1','data_origin':self.origin,'mode':'synthetic_historical_replay','units':self.units,
                'sensors':list(self.sensors),'models':['xgboost_selected','seasonal_historical_average'],
                'sensor_details':[{'sensor_id':name,'label':f'Synthetic sensor {i+1}','synthetic':True} for i,name in enumerate(self.sensors)],
                'horizons_steps':list(HORIZONS),'horizons_minutes':[15,30,60],
                'supported_horizons':[{'steps':h,'minutes':h*5,'label':f'+{h*5} min'} for h in HORIZONS],
                'replay':{'available':True,'partition':'controller_selection_demonstration_only','first_index':self.FIRST,'last_index':self.LAST,
                          'origin_count':self.LAST-self.FIRST+1,'index_stride':1,
                          'first_timestamp':stamp(self.times[self.FIRST-6208]),'last_timestamp':stamp(self.times[self.LAST-6208]),
                          'history_steps':12,'interval_minutes':5,'timestamp_convention':'timezone-naive synthetic clock',
                          'final_test_available':False},
                'scenarios':['clean','missing','noisy','stuck'],
                'policies':['controller','xgboost'],'fault_rates':[.1,.2,.4],'fault_seeds':[1729,2718,31415],
                'geography':{'available':False,'reason':'no_geographic_coordinates_provided'},
                'interval_policy':'Clean fixed-XGBoost only: calibration-derived, not independently validated. Controller/fallback/fault intervals unavailable.',
                'cpu_threads_max':4}

    def resolve_origin(self,request):
        if request.sensor_id not in self.sensors:
            raise ReplayInputError('Unknown synthetic sensor identifier',404)
        index=request.index
        if index is None:
            try:
                parsed=datetime.fromisoformat(request.timestamp)
                if parsed.tzinfo is not None:raise ValueError('Timezone must be absent')
                timestamp=np.datetime64(parsed,'ns')
                # Bound before arithmetic; this also prevents distant-date overflow.
                if parsed.year!=2024:raise ValueError('Outside demonstration range')
                delta=timestamp-self.times[0]
                if delta%STEP!=np.timedelta64(0,'ns'):raise ValueError('Off-grid timestamp')
                index=6208+int(delta/STEP)
            except (ValueError,OverflowError,TypeError):
                raise ReplayInputError('Timestamp must be a valid five-minute synthetic replay time') from None
        if not self.FIRST<=index<=self.LAST:
            raise ReplayInputError('Replay origin is outside the permitted demonstration range')
        return index

    def replay(self,request):
        index=self.resolve_origin(request)
        # Serialize inference in this single-worker process; process-wide native
        # pools remain capped even when multiple HTTP requests arrive together.
        with self._lock,threadpool_limits(limits=4):
            local=index-6208
            values=self.readings[local-11:local+1].copy()
            times=self.times[local-11:local+1]
            fault_summary=None
            if request.scenario!='clean':
                mode='contiguous' if request.scenario=='stuck' else 'point'
                part=Partition('controller_selection',times,values,self.sensors,index-11)
                spec=FaultSpec(request.scenario,mode,request.fault_rate,request.seed,sampling_policy='synthetic_ceil')
                overlay=inject_controller(part,spec,self.statistics,self.scales)
                values=overlay.values
                fault_summary={k:overlay.details[k] for k in ('kind','mode','rate','seed','sampling_policy','achieved_selection_rate','selected_cell_count','changed_cell_count')}
                fault_summary['scope']='historical window only; not the Phase 3C experiment placement'
            inputs=ForecastInput(values,times[-1],self.sensors)
            p=self.primary.predict(inputs)
            f=self.fallback.predict(inputs)[np.array(HORIZONS)-1]
            prediction,decision=select_forecast(inputs,p,f,self.statistics,self.thresholds)
            if request.policy=='xgboost':
                prediction=p
                decision={**decision,'choice':'primary','reason':'fixed_xgboost_policy',
                          'interval_validity':('clean primary calibration only; independent coverage not measured'
                                               if request.scenario=='clean' else 'fault input intervals unavailable; coverage not validated')}
            if not np.isfinite(prediction).all():
                raise RuntimeError('Synthetic forecast unavailable')
            sensor=self.sensors.index(request.sensor_id)
            filled,observed,_=self.statistics.fill(inputs)
            history=[{'index':index-11+i,'relative_minutes':(i-11)*5,
                      'timestamp':stamp(t),'value':float(v) if np.isfinite(v) else None,
                      'usable':bool(o),'filled_value':float(z)} for i,(t,v,o,z) in enumerate(zip(times,values[:,sensor],observed[:,sensor],filled[:,sensor]))]
            selected_model='xgboost_selected' if decision['choice']=='primary' else 'seasonal_historical_average'
            forecasts=[]
            for i,h in enumerate(HORIZONS):
                value=float(prediction[i,sensor])
                supported=request.policy=='xgboost' and request.scenario=='clean'
                reason=('clean_primary_calibration_not_independent_validation' if supported else
                        'fault_input_coverage_not_validated' if request.scenario!='clean' else 'controller_or_fallback_intervals_not_validated')
                interval={'status':'calibration_derived' if supported else 'unavailable',
                          'available':supported,'lower':value-self.widths[i] if supported else None,
                          'upper':value+self.widths[i] if supported else None,'nominal_coverage':.9 if supported else None,'reason':reason}
                forecasts.append({'horizon_steps':h,'horizon_minutes':h*5,'timestamp':stamp(times[-1]+h*STEP),
                                  'prediction':value,'selected_model':selected_model,'interval':interval})
            return {'contract_version':'1.1','data_origin':self.origin,'mode':'synthetic_historical_replay','units':self.units,
                    'sensor_id':request.sensor_id,'sensor_label':f'Synthetic sensor {sensor+1}',
                    'index':index,'timestamp':stamp(times[-1]),'scenario':request.scenario,
                    'playback':{'previous_index':index-1 if index>self.FIRST else None,
                                'next_index':index+1 if index<self.LAST else None,
                                'first_index':self.FIRST,'last_index':self.LAST},
                    'policy':request.policy,'joint_decision_scope':'all four sensors and all three horizons',
                    'decision':decision,'history':history,'forecasts':forecasts,'fault_summary':fault_summary}
