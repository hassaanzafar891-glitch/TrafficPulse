"""Container packaging only: derive serving dependencies and check mounted files."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tomllib

REPORTS = (
    'reports/generated/phase2b_synthetic_xgboost.json',
    'reports/generated/phase3b_synthetic_uncertainty.json',
    'reports/generated/phase3c_synthetic_controller.json',
)


def requirements(manifest, environment):
    project = tomllib.loads(Path(manifest).read_text(encoding='utf-8'))['project']
    candidates = (project['dependencies'] + project['optional-dependencies']['backend']
                  + project['optional-dependencies']['xgboost'])
    wanted = {'numpy', 'threadpoolctl', 'fastapi', 'uvicorn', 'xgboost', 'psutil'}
    selected = {}
    for requirement in candidates:
        name, separator, version = requirement.partition('==')
        if name in wanted:
            if separator != '==' or not version or name in selected:
                raise ValueError('Serving dependencies require unique exact pins')
            # Same Python import/version, CPU-only Linux distribution: never add NCCL.
            selected[name] = f'{"xgboost-cpu" if name == "xgboost" else name}=={version}'
    if set(selected) != wanted:
        raise ValueError('Serving dependency manifest is incomplete')
    # XGBRegressor requires sklearn even for restore/predict; reuse reviewed pins.
    for name in ('scikit-learn', 'scipy'):
        matches = [line for line in Path(environment).read_text().splitlines()
                   if line.startswith(name+'==')]
        if len(matches) != 1:
            raise ValueError('Missing reviewed sklearn/SciPy serving pin')
        selected[name] = matches[0]
    return [selected[name] for name in sorted(selected)]


def check_runtime(root):
    inventory = json.loads((root/'configs/reviewed_synthetic_artifacts.json').read_text())
    if inventory['data_origin'] != 'synthetic_generated_not_real_traffic' or len(inventory['artifacts']) != 11:
        raise ValueError('Require the reviewed eleven-file synthetic inventory')
    for item in inventory['artifacts']:
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'artifacts':
            raise ValueError('Unexpected runtime inventory path')
        file = root/relative
        if not file.is_file():
            raise ValueError(f'Missing required synthetic runtime file: {relative.as_posix()}')
        data = file.read_bytes()
        if len(data) != item['size_bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError(f'Synthetic artifact integrity mismatch: {relative.as_posix()}')
    for relative in REPORTS:
        file = root/relative
        if not file.is_file():
            raise ValueError(f'Missing required synthetic runtime report: {relative}')
        try:
            json.loads(file.read_text(encoding='utf-8'))
        except (ValueError, UnicodeError):
            raise ValueError('Synthetic runtime report is not valid JSON') from None


def main():
    if len(sys.argv) == 4 and sys.argv[1] == 'requirements':
        print('\n'.join(requirements(sys.argv[2], sys.argv[3])))
        return
    if sys.argv[1:] != ['serve']:
        raise ValueError('Choose requirements or serve')
    try:
        check_runtime(Path('/app'))
    except (ValueError, OSError, KeyError):
        print('TrafficPulse startup blocked: required reviewed synthetic runtime files '
              'are missing, unreadable or inconsistent. Check the 14 read-only mounts.', file=sys.stderr)
        raise SystemExit(1) from None
    os.execvp('python', ['python', '-m', 'uvicorn', 'trafficpulse.api.app:create_app',
                        '--factory', '--host', '0.0.0.0', '--port', '8000', '--workers', '1'])


if __name__ == '__main__':
    main()
