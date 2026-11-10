"""Run only the synthetic fixture in a child process with a hard wall-clock cap."""
import os
from pathlib import Path
import subprocess
import sys


def synthetic_worker(root, experiment_module="trafficpulse.experiment"):
    """Fail closed on real-data file access or any network connection."""
    forbidden = [(root / "data").resolve(), (root / "reports/local").resolve()]

    def audit(event, args):
        if event == "socket.connect":
            raise RuntimeError("Network connections are disabled for synthetic smoke runs")
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            candidate = Path(os.fsdecode(args[0])).resolve()
            if any(candidate == p or p in candidate.parents for p in forbidden):
                raise RuntimeError("Real-data/local detailed audit access is disabled")

    sys.addaudithook(audit)
    sys.path.insert(0, str(root / "src"))
    if experiment_module not in ("trafficpulse.experiment", "trafficpulse.xgboost_experiment", "trafficpulse.fault_experiment", "trafficpulse.uncertainty_experiment", "trafficpulse.controller_experiment"):
        raise ValueError("Only synthetic experiment modules are allowed")
    from importlib import import_module
    import_module(experiment_module).main()


def main():
    root = Path(__file__).resolve().parents[1]
    xgboost = "--xgboost" in sys.argv
    faults = "--faults" in sys.argv
    uncertainty = "--uncertainty" in sys.argv
    controller = "--controller" in sys.argv
    if sum((xgboost,faults,uncertainty,controller))>1:
        raise ValueError("Choose one synthetic experiment at a time")
    if xgboost:
        sys.argv.remove("--xgboost")
    if faults:
        sys.argv.remove("--faults")
    if uncertainty:
        sys.argv.remove("--uncertainty")
    if controller:
        sys.argv.remove("--controller")
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        sys.argv.pop(1)
        module="trafficpulse.controller_experiment" if controller else ("trafficpulse.uncertainty_experiment" if uncertainty else ("trafficpulse.fault_experiment" if faults else ("trafficpulse.xgboost_experiment" if xgboost else "trafficpulse.experiment")))
        synthetic_worker(root,module)
        return 0
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
        env[key] = "4"
    env["PYTHONPATH"] = str(root / "src")
    env["MLFLOW_ENABLE_TELEMETRY"] = "false"
    env["DO_NOT_TRACK"] = "true"
    try:
        flags = ["--controller"] if controller else (["--uncertainty"] if uncertainty else (["--faults"] if faults else (["--xgboost"] if xgboost else [])))
        return subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", *flags, *sys.argv[1:]],
                              cwd=root, env=env, timeout=1800, check=False).returncode
    except subprocess.TimeoutExpired:
        print("Synthetic run exceeded 1800 seconds and was terminated", file=sys.stderr)
        return 124


if __name__ == "__main__":
    raise SystemExit(main())
