"""Bounded local DVC commands; repository trust is scoped to this process only."""
import os
from pathlib import Path
import subprocess
import sys

COMMANDS = {
    'cache-bundle': ['commit', '--force', 'reviewed_bundle'],
    'repro': ['repro'],
    'force-repro': ['repro', '--force'],
    'status': ['status'],
    'dag': ['dag', '--mermaid'],
    'checkout-bundle': ['checkout', 'reviewed_bundle'],
}


def main():
    root = Path(__file__).resolve().parents[1]
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        def offline(event, args):
            if event == 'socket.connect':
                raise RuntimeError('Local DVC commands cannot connect to the network')
        sys.addaudithook(offline)
        import pygit2
        local = root/'reports/tmp/dvc-git-trust'
        local.mkdir(parents=True, exist_ok=True)
        (local/'.gitconfig').write_text('[safe]\n\tdirectory = ' + root.as_posix() + '\n', encoding='utf-8')
        # DVC/libgit2 ignores Git CLI -c; use a process-local, one-repository trust list.
        pygit2.settings.search_path[pygit2.enums.ConfigLevel.GLOBAL] = str(local)
        sys.path.insert(0,str(root/"src"))
        from trafficpulse.dvc_foundation import validate_pipeline
        validate_pipeline(root)
        from dvc.repo import Repo
        with Repo(str(root)) as repo:
            if repo.config.get("remote") or repo.config.get("core", {}).get("remote"):
                raise RuntimeError("Remote configuration is forbidden for this foundation")
        from dvc.cli import main as dvc_main
        return dvc_main(COMMANDS[sys.argv[2]])
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        raise ValueError('Choose a documented local DVC command')
    env = os.environ.copy()
    env.update({'DVC_NO_ANALYTICS': '1', 'DO_NOT_TRACK': 'true', 'PYTHONHASHSEED': '0'})
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMEXPR_MAX_THREADS'):
        env[key] = '4'
    try:
        return subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', sys.argv[1]],
                              cwd=root, env=env, timeout=1800, check=False).returncode
    except subprocess.TimeoutExpired:
        return 124


if __name__ == '__main__':
    raise SystemExit(main())
