"""Guarded non-training foundation runner, bounded to four threads/1800 seconds."""
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        def guard(event, args):
            if event == 'socket.connect':
                raise RuntimeError('Foundation stages are offline')
            if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
                path = Path(os.fsdecode(args[0])).resolve()
                forbidden = [root/'data', root/'reports/local', root/'artifacts/synthetic_mlflow']
                if any(path == p or p in path.parents for p in forbidden):
                    raise RuntimeError('Restricted data/MLflow access blocked')
        sys.addaudithook(guard)
        sys.path.insert(0, str(root/'src'))
        from trafficpulse.dvc_foundation import prefix_summary, write_json, verify_stage, snapshot_artifacts
        action = sys.argv[2]
        if action == 'prefix':
            write_json(root, 'training_prefix.json', prefix_summary(root))
        elif action == 'verify':
            verify_stage(root)
        elif action == 'snapshot':
            snapshot_artifacts(root)
        else:
            raise ValueError('Unknown foundation action')
        return 0
    if len(sys.argv) != 2 or sys.argv[1] not in ('prefix', 'verify', 'snapshot'):
        raise ValueError('Choose prefix, verify or explicit snapshot import')
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMEXPR_MAX_THREADS'):
        env[key] = '4'
    env['DO_NOT_TRACK'] = 'true'
    try:
        return subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', sys.argv[1]],
                              cwd=root, env=env, timeout=1800, check=False).returncode
    except subprocess.TimeoutExpired:
        return 124


if __name__ == '__main__':
    raise SystemExit(main())
