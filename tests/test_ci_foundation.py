"""Offline workflow safety checks; never execute installation or remote actions."""
import copy
import importlib.util
import json
import os
import tempfile
from unittest.mock import patch
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    value=importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


class CiFoundationTests(unittest.TestCase):
    def setUp(self):
        self.validator=module('validate_ci')
        self.workflow=json.loads((ROOT/'.github/workflows/ci.yml').read_text(encoding='utf-8'))

    def test_workflow_commands_and_dependency_pins_validate_offline(self):
        self.validator.main()

    def test_privileged_trigger_write_permissions_and_upload_are_rejected(self):
        for field,value in [('on',['pull_request_target']),('permissions',{'contents':'write'})]:
            changed=copy.deepcopy(self.workflow);changed[field]=value
            with self.assertRaises(ValueError):self.validator.validate(changed)
        changed=copy.deepcopy(self.workflow)
        changed['jobs']['checks']['steps'].append({'run':'git push'})
        with self.assertRaises(ValueError):self.validator.validate(changed)
        for bypass in ({'continue-on-error':True}, {'if':False}):
            changed=copy.deepcopy(self.workflow)
            changed['jobs']['checks']['steps'][3].update(bypass)
            with self.assertRaises(ValueError):self.validator.validate(changed)

    def test_action_mutability_and_missing_thread_caps_are_rejected(self):
        changed=copy.deepcopy(self.workflow)
        changed['jobs']['checks']['steps'][0]['uses']='actions/checkout@main'
        with self.assertRaises(ValueError):self.validator.validate(changed)
        changed=copy.deepcopy(self.workflow);changed['env']['OMP_NUM_THREADS']='8'
        with self.assertRaises(ValueError):self.validator.validate(changed)

    def test_descriptor_relative_cleanup_preserves_protected_path_guard(self):
        runner = module('run_ci_checks')
        # Real cleanup uses dir_fd on Linux and the portable path on Windows.
        with tempfile.TemporaryDirectory(dir=ROOT/'reports/tmp') as location:
            base = Path(location)
            for name in ('artifacts', 'data', 'reports/local', '.dvc/cache', '.dvc/tmp'):
                target = base/name
                target.mkdir(parents=True, exist_ok=True)
                (target/'dummy.txt').write_text('test-only', encoding='utf-8')
            # Emulate the Linux descriptor audit path on every platform.
            def audited_open(path, flags, mode, *, dir_fd):
                runner.guard('open', (path, None, flags))
                return 123
            with patch.object(runner, '_OS_OPEN', side_effect=audited_open):
                with patch.object(runner.os, 'readlink', return_value=str(base)):
                    self.assertEqual(runner.descriptor_aware_open('artifacts', os.O_RDONLY, dir_fd=99), 123)
                with patch.object(runner.os, 'readlink', return_value=str(ROOT)):
                    for name in ('data', 'artifacts', 'reports/local', '.dvc/cache', '.dvc/tmp'):
                        with self.assertRaises(RuntimeError):
                            runner.descriptor_aware_open(name, os.O_RDONLY, dir_fd=99)
                with patch.object(runner.os, 'readlink', side_effect=OSError):
                    with self.assertRaises(RuntimeError):
                        runner.descriptor_aware_open('artifacts', os.O_RDONLY, dir_fd=99)
            for name in ('data', 'artifacts', 'reports/local', '.dvc/cache', '.dvc/tmp'):
                with self.assertRaises(RuntimeError):
                    runner.guard('open', (str(ROOT/name), None, os.O_RDONLY))
        self.assertFalse(base.exists())

    def test_ci_cannot_select_artifact_tests_or_full_reserved_fixture(self):
        runner=module('run_ci_checks')
        self.assertNotIn('test_api',runner.SUITES)
        self.assertNotIn('test_xgboost_baseline',runner.SUITES)
        self.assertNotIn('test_baselines.SyntheticExperimentTests',runner.SUITES)
        with self.assertRaises(RuntimeError):runner.check_fixture_scope()
        with self.assertRaises(RuntimeError):runner.check_fixture_scope(stop_rows=6451)
        runner.check_fixture_scope(stop_rows=5644)


if __name__ == '__main__':unittest.main()
