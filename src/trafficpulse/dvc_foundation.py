"""Non-training, local synthetic reproducibility checks. No model imports."""
import hashlib
import json
from pathlib import Path
import shutil

BASE = 'artifacts/dvc_foundation'
INVENTORY = 'configs/reviewed_synthetic_artifacts.json'
CONFIG = 'configs/dvc_foundation.json'
GROUPS = {
    'artifacts/synthetic_xgboost': {'horizon_3.json', 'horizon_6.json', 'horizon_12.json',
                                 'preprocessing.json', 'comparison.json', 'experiment_config.json'},
    'artifacts/synthetic_uncertainty/phase3b-synthetic-split-conformal-v1':
        {'calibration_summary.json', 'experiment_config.json'},
    'artifacts/synthetic_controller/phase3c-synthetic-joint-controller-v1':
        {'controller_comparison.json', 'synthetic_decisions.json', 'experiment_config.json'},
}
EXPECTED = {f'{folder}/{name}' for folder, names in GROUPS.items() for name in names}


def safe_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or '\\' in relative or ':' in relative:
        raise ValueError('Require an explicit relative workspace path')
    result = root / path
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError('Path escapes workspace')
    for parent in [result, *result.parents]:
        if parent == root:
            break
        if parent.is_symlink() or parent.is_junction():
            raise ValueError('Linked paths are not allowed')
    return result


def load_inventory(root):
    inventory = json.loads((root / INVENTORY).read_text(encoding='utf-8'))
    items = inventory['artifacts']
    if (inventory['data_origin'] != 'synthetic_generated_not_real_traffic' or
            len(items) != 11 or {x['path'] for x in items} != EXPECTED):
        raise ValueError('Require exactly eleven reviewed synthetic artifacts')
    return items


def verify_artifacts(root, *, snapshot=False):
    items = load_inventory(root)
    for item in items:
        relative = f"{BASE}/reviewed_bundle/{item['path']}" if snapshot else item['path']
        data = safe_path(root, relative).read_bytes()
        if len(data) != item['size_bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError(f"Reviewed artifact differs: {item['path']}")
    if snapshot:
        folder = safe_path(root, f'{BASE}/reviewed_bundle')
        entries = list(folder.rglob('*'))
        if any(p.is_symlink() or p.is_junction() for p in entries):
            raise ValueError('Linked snapshot entries are not allowed')
        actual = {p.relative_to(folder).as_posix() for p in entries if p.is_file()}
        if actual != EXPECTED:
            raise ValueError('Unexpected file in reviewed snapshot')
    return items


def snapshot_artifacts(root):
    """Explicit one-time import; never runs from dvc repro or overwrites a snapshot."""
    items = verify_artifacts(root)
    destination = safe_path(root, f'{BASE}/reviewed_bundle')
    if destination.exists():
        verify_artifacts(root, snapshot=True)
        return
    destination.mkdir(parents=True)
    for item in items:
        target = safe_path(root, f"{BASE}/reviewed_bundle/{item['path']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe_path(root, item['path']), target)
    verify_artifacts(root, snapshot=True)
    verify_artifacts(root)


def load_settings(root):
    settings = json.loads((root / CONFIG).read_text(encoding='utf-8'))
    if settings != {'schema_version': 1, 'data_origin': 'synthetic_generated_not_real_traffic',
                    'fixture': {'days': 28, 'sensors': 4, 'seed': 1729, 'stop_rows': 5644},
                    'max_threads': 4, 'timeout_seconds': 1800, 'allow_training': False,
                    'allow_calibration': False, 'allow_controller_selection': False, 'allow_final_test': False}:
        raise ValueError('Only the reviewed training-prefix-only foundation is permitted')
    return settings


def prefix_summary(root):
    import numpy as np
    from threadpoolctl import threadpool_limits, threadpool_info
    from .synthetic import make_synthetic
    settings = load_settings(root)
    with threadpool_limits(limits=4):
        series = make_synthetic(**settings['fixture'])
        if any(p['num_threads'] > 4 for p in threadpool_info()):
            raise RuntimeError('Numerical thread cap exceeded')
        return {'data_origin': series.origin, 'units': series.units, 'partition': 'train',
                'rows': len(series.values), 'sensors': len(series.sensors), 'raw_rows': [0, 5644],
                'first_timestamp': str(series.timestamps[0]), 'last_timestamp': str(series.timestamps[-1]),
                'values_sha256': hashlib.sha256(series.values.tobytes()).hexdigest(),
                'timestamps_sha256': hashlib.sha256(series.timestamps.tobytes()).hexdigest(),
                'zeros': int((series.values == 0).sum()), 'missing': int(np.isnan(series.values).sum()),
                'final_test_access': False, 'training_performed': False}


def write_json(root, name, value):
    if name not in ('training_prefix.json', 'verification.json'):
        raise ValueError('Unknown foundation output')
    path = safe_path(root, f'{BASE}/{name}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def verify_stage(root):
    validate_pipeline(root)
    load_settings(root)
    items = verify_artifacts(root, snapshot=True)
    prefix = json.loads(safe_path(root, f'{BASE}/training_prefix.json').read_text(encoding='utf-8'))
    if prefix != prefix_summary(root):
        raise ValueError('Training prefix is not reproducible')
    deps = WORKFLOW_DEPS
    source_hashes = {p: hashlib.sha256(safe_path(root, p).read_bytes()).hexdigest()
                     for p in deps if not p.startswith('artifacts/')}
    write_json(root, 'verification.json', {'data_origin': prefix['data_origin'],
               'artifact_count': len(items), 'artifacts': items, 'source_sha256': source_hashes,
               'prefix_values_sha256': prefix['values_sha256'], 'model_refitted': False,
               'recalibrated': False, 'controller_retuned': False, 'final_test_access': False,
               'mlflow_access': False, 'scope': 'snapshot integrity and training-prefix reproducibility only'})


def validate_pipeline(root):
    """Fail closed before DVC executes a changed command or unsafe output path."""
    load_settings(root)
    pipeline = json.loads((root/'dvc.yaml').read_text(encoding='utf-8'))
    stages = pipeline['stages']
    actions = {'reviewed_bundle': ('snapshot', 'reviewed_bundle'),
               'training_prefix': ('prefix', 'training_prefix.json'),
               'verify_reviewed_bundle': ('verify', 'verification.json')}
    if set(stages) != set(actions):
        raise ValueError('Unexpected pipeline stages')
    for name, (action, output) in actions.items():
        stage = stages[name]
        if (stage['cmd'] != f'".venv/Scripts/python.exe" scripts/run_dvc_foundation.py {action}' or
                stage['outs'] != [f'{BASE}/{output}'] or stage.get('frozen', False) != (name == 'reviewed_bundle')):
            raise ValueError('Unsafe command/output or unfrozen artifact import')
        if set(stage) != ({'cmd', 'deps', 'outs', 'frozen'} if name == 'reviewed_bundle' else {'cmd', 'deps', 'outs', 'params'}):
            raise ValueError('Unexpected pipeline options')
        for dep in stage['deps']:
            if dep not in WORKFLOW_DEPS and dep not in (f'{BASE}/reviewed_bundle', f'{BASE}/training_prefix.json'):
                raise ValueError('Unexpected dependency')
            safe_path(root, dep)
    if set(stages['verify_reviewed_bundle']['deps']) != set(WORKFLOW_DEPS) | {f'{BASE}/reviewed_bundle', f'{BASE}/training_prefix.json'}:
        raise ValueError('Missing workflow dependency')
    if set(stages['reviewed_bundle']['deps']) != {INVENTORY, 'scripts/run_dvc_foundation.py', 'src/trafficpulse/dvc_foundation.py'}:
        raise ValueError('Missing snapshot import dependency')
    prefix_deps = {'scripts/run_dvc_foundation.py', 'src/trafficpulse/dvc_foundation.py',
                   'src/trafficpulse/synthetic.py', 'src/trafficpulse/__init__.py', CONFIG,
                   'configs/phase6a-environment.txt', 'pyproject.toml'}
    if set(stages['training_prefix']['deps']) != prefix_deps:
        raise ValueError('Missing prefix dependency')
    for name in ('training_prefix', 'verify_reviewed_bundle'):
        if stages[name]['params'] != [{CONFIG: list(load_settings(root))}]:
            raise ValueError('Missing safety/fixture parameters')


WORKFLOW_DEPS = ['src/trafficpulse/__init__.py',
 'src/trafficpulse/baselines.py',
 'src/trafficpulse/data.py',
 'src/trafficpulse/synthetic.py',
 'src/trafficpulse/metrics.py',
 'src/trafficpulse/faults.py',
 'src/trafficpulse/reliability.py',
 'src/trafficpulse/uncertainty.py',
 'src/trafficpulse/xgboost_baseline.py',
 'src/trafficpulse/tracking.py',
 'src/trafficpulse/experiment.py',
 'src/trafficpulse/xgboost_experiment.py',
 'src/trafficpulse/fault_experiment.py',
 'src/trafficpulse/uncertainty_experiment.py',
 'src/trafficpulse/controller_experiment.py',
 'src/trafficpulse/dvc_foundation.py',
 'scripts/run_synthetic_baselines.py',
 'scripts/run_local_dvc.py',
 'scripts/run_dvc_foundation.py',
 'configs/evaluation_protocol.json',
 'configs/phase2b_synthetic_xgboost.json',
 'configs/phase3a_synthetic_faults.json',
 'configs/phase3b_synthetic_uncertainty.json',
 'configs/phase3c_synthetic_controller.json',
 'configs/reviewed_synthetic_artifacts.json',
 'configs/dvc_foundation.json',
 'configs/phase6a-environment.txt',
 'pyproject.toml']
