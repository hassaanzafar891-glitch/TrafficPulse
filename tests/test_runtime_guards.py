"""Verify timeout and synthetic-worker access guards without touching raw files."""
import importlib.util
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "synthetic_launcher", Path(__file__).resolve().parents[1]/"scripts/run_synthetic_baselines.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class RuntimeGuardTests(unittest.TestCase):
    def test_worker_blocks_data_reports_and_network(self):
        root=Path(__file__).resolve().parents[1]
        with patch.object(launcher.sys,'addaudithook') as install, patch('trafficpulse.experiment.main'):
            launcher.synthetic_worker(root)
        guard=install.call_args.args[0]
        for path in (root/'data/raw/forbidden.h5',root/'reports/local/forbidden.json'):
            with self.assertRaises(RuntimeError):
                guard('open',(str(path),'r',0))
        with self.assertRaises(RuntimeError):
            guard('socket.connect',(None,('example.invalid',443)))
        guard('open',(str(root/'reports/generated/synthetic.json'),'w',0))

    def test_timeout_and_thread_environment_are_enforced(self):
        with patch.object(launcher.sys,'argv',['runner']), patch.object(launcher.subprocess,'run',side_effect=subprocess.TimeoutExpired('synthetic',1800)) as run:
            self.assertEqual(launcher.main(),124)
        self.assertEqual(run.call_args.kwargs['timeout'],1800)
        env=run.call_args.kwargs['env']
        self.assertEqual(env['OMP_NUM_THREADS'],'4')
        self.assertEqual(env['OPENBLAS_NUM_THREADS'],'4')
