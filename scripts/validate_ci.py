"""Offline JSON-as-YAML and selected GitHub workflow safety checks, not actionlint."""
import json
from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    'actions/checkout': '3d3c42e5aac5ba805825da76410c181273ba90b1',
    'actions/setup-python': '6d4fd22b20465ccc0f7354476588e8df7c9ca0e0',
    'actions/setup-node': '949feb2413d6458794dcd2491c4babbbce0c15c1',
}
COMMANDS = ['python -m pip install --disable-pip-version-check --only-binary=:all: -r configs/ci-requirements.txt',
            'python -m pip check', 'python scripts/validate_ci.py', 'python scripts/run_ci_checks.py',
            'npm ci --ignore-scripts --no-audit --no-fund', 'npm test', 'npm run build']


def validate(workflow):
    if any(not re.fullmatch(r'[0-9a-f]{40}',sha) for sha in PINS.values()):
        raise ValueError('Action references must be full lowercase hexadecimal SHAs')
    if workflow['on'] != ['push', 'pull_request'] or workflow['permissions'] != {'contents':'read'}:
        raise ValueError('Only unprivileged push/PR checks are permitted')
    if set(workflow['jobs']) != {'checks'}:
        raise ValueError('Unexpected CI jobs')
    job = workflow['jobs']['checks']
    if job['runs-on'] != 'ubuntu-24.04' or job['timeout-minutes'] != 30:
        raise ValueError('Unexpected platform/resource limits')
    if set(job) != {'runs-on', 'timeout-minutes', 'steps'}:
        raise ValueError('Unexpected job services/permissions/configuration')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMEXPR_MAX_THREADS',
                'UV_THREADPOOL_SIZE','GOMAXPROCS','RAYON_NUM_THREADS'):
        if workflow['env'][key] != '4': raise ValueError('Missing four-thread cap')
    refs = [s for s in job['steps'] if 'uses' in s]
    if {s['uses'] for s in refs} != {f'{name}@{sha}' for name,sha in PINS.items()} or len(refs)!=3:
        raise ValueError('Unreviewed or mutable action reference')
    if refs[0]['with'] != {'persist-credentials':False,'fetch-depth':1,'lfs':False,'submodules':False}:
        raise ValueError('Checkout must not persist authentication or fetch external data')
    if refs[1]['with'] != {'python-version':'3.12.10'} or refs[2]['with'] != {'node-version':'22.18.0'}:
        raise ValueError('Unexpected toolchain')
    commands = [s for s in job['steps'] if 'run' in s]
    for step in job['steps']:
        allowed = {'name','uses','with'} if 'uses' in step else {'name','run','working-directory'}
        if not set(step) <= allowed:
            raise ValueError('Unexpected step options may bypass checks or mask failure')
    if [s['run'] for s in commands] != COMMANDS:
        raise ValueError('Unexpected CI command or upload')
    for i, step in enumerate(commands):
        if step.get('working-directory') != ('frontend' if i>=4 else None):
            raise ValueError('Incorrect command working directory')
    text = json.dumps(workflow).lower()
    if any(word in text for word in ('secrets.', 'pull_request_target', 'upload-artifact', 'dvc push', 'git push', 'dagshub.com')):
        raise ValueError('Unexpected privilege/remote payload operation')


def main():
    workflow = json.loads((ROOT/'.github/workflows/ci.yml').read_text(encoding='utf-8'))
    validate(workflow)
    project = tomllib.loads((ROOT/'pyproject.toml').read_text(encoding='utf-8'))['project']
    pins = (ROOT/'configs/ci-requirements.txt').read_text().splitlines()
    for item in pins:
        if item and not item.startswith('#') and not re.fullmatch(r'[A-Za-z0-9_.-]+==[0-9][A-Za-z0-9_.+-]*',item):
            raise ValueError('CI requirements must use exact versions and no URLs')
    api = [p for p in project['optional-dependencies']['backend'] if p.startswith(('fastapi==', 'httpx=='))]
    for pin in project['dependencies'] + api + project['optional-dependencies']['xgboost'][1:]:
        if pin not in pins: raise ValueError(f'CI dependency mismatch: {pin}')
    print('Offline workflow structure, commands, action-pin format and dependency pins validated.')


if __name__ == '__main__': main()
