"""Explicit artifact-free CI selection; never discover the entire legacy suite."""
import ipaddress
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SUITES = ('test_baselines.WindowTests', 'test_baselines.MetricTests', 'test_tracking',
          'test_dagshub_readiness', 'test_dvc_foundation', 'test_api_offline', 'test_ci_foundation')


def check_fixture_scope(days=28, sensors=4, seed=1729, *, stop_rows=None):
    if days == 28 and (stop_rows is None or stop_rows > 5644):
        raise RuntimeError('CI cannot construct the reserved synthetic suffix')


def guard(event, args):
    if event == 'socket.connect':
        address = args[1]
        if not isinstance(address, tuple) or not ipaddress.ip_address(address[0]).is_loopback:
            raise RuntimeError('External network disabled during CI tests')
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])).resolve()
        forbidden = ('data', 'artifacts', 'reports/local', '.dvc/cache', '.dvc/tmp')
        if any(path == ROOT/p or ROOT/p in path.parents for p in forbidden):
            raise RuntimeError('CI cannot access private data, artifacts, stores or caches')

# The open audit event omits dir_fd. Normalize descriptor-relative os.open calls
# before the event so the guard checks the actual target, not cwd / basename.
_OS_OPEN = os.open


def descriptor_aware_open(path, flags, mode=0o777, *, dir_fd=None):
    if dir_fd is not None and not Path(os.fsdecode(path)).is_absolute():
        try:
            base = Path(os.readlink(f'/proc/self/fd/{dir_fd}'))
        except (OSError, ValueError, NotImplementedError) as exc:
            raise RuntimeError('CI cannot resolve directory-relative open safely') from None
        if not base.is_absolute() or str(base).endswith(' (deleted)'):
            raise RuntimeError('CI cannot resolve directory-relative open safely')
        path = base / os.fsdecode(path)
    return _OS_OPEN(path, flags, mode, dir_fd=dir_fd)


def main():
    if sys.argv[1:] == ['--worker']:
        os.open = descriptor_aware_open
        sys.addaudithook(guard)
        sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
        # Test-created dummy artifact files stay under this disposable directory.
        (ROOT/'reports/tmp').mkdir(parents=True, exist_ok=True)
        import trafficpulse.synthetic as synthetic
        original = synthetic.make_synthetic
        def prefix_only(*args, **kwargs):
            check_fixture_scope(*args, **kwargs)
            return original(*args, **kwargs)
        synthetic.make_synthetic = prefix_only
        import unittest
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=4):
            suite = unittest.defaultTestLoader.loadTestsFromNames(SUITES)
            result = unittest.TextTestRunner(verbosity=2).run(suite)
        if any(name in sys.modules for name in ('xgboost', 'mlflow', 'dvc')):
            raise RuntimeError('Lightweight CI unexpectedly imported an experiment/client dependency')
        return 0 if result.wasSuccessful() else 1
    if sys.argv[1:]:
        raise ValueError('No custom test selection allowed; use the explicit CI suites')
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMEXPR_MAX_THREADS'):
        env[key] = '4'
    env.update(DO_NOT_TRACK='true', MLFLOW_ENABLE_TELEMETRY='false', PYTHONHASHSEED='0')
    try:
        return subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'],
                              cwd=ROOT, env=env, timeout=1800, check=False).returncode
    except subprocess.TimeoutExpired:
        return 124


if __name__ == '__main__': raise SystemExit(main())
