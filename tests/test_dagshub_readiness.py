"""Offline policy/ignore checks only; no credentials, clients, remote or experiments."""
import configparser
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads((ROOT/'configs/dagshub_readiness.json').read_text(encoding='utf-8'))

    def test_policy_is_disabled_and_has_no_destination(self):
        self.assertEqual(self.policy['default_action'], 'deny')
        for key in ('network_enabled', 'remote_write_approved', 'automatic_artifact_upload',
                    'automatic_run_migration', 'automatic_model_registration',
                    'automatic_metrics_upload', 'automatic_parameter_upload'):
            self.assertIs(self.policy[key], False)
        for key in ('owner', 'repository', 'visibility', 'authorized_account', 'mlflow_endpoint', 'dvc_endpoint'):
            self.assertIsNone(self.policy[key])

    def test_metrics_need_scientific_context_and_separate_write_gates(self):
        self.assertEqual(self.policy['data_origin'], 'synthetic_generated_not_real_traffic')
        self.assertIn('not_independent_validation', self.policy['required_context'])
        self.assertIn('partition_role', self.policy['required_context'])
        self.assertIn('metrics_payload', self.policy['approval_gates'])
        self.assertIn('artifact_manifest', self.policy['approval_gates'])
        self.assertIn('specific_remote_write', self.policy['approval_gates'])
        self.assertIn('destination_and_visibility', self.policy['approval_gates'])
        self.assertIn('authorized_account_and_permissions', self.policy['approval_gates'])
        self.assertNotIn('config', self.policy['candidate_scalar_parameters'])
        self.assertNotIn('fault_config', self.policy['candidate_scalar_parameters'])

    def test_sensitive_and_reserved_categories_default_to_no_export(self):
        for category in ('credentials', 'raw_datasets', 'restricted_sensor_metadata', 'unverified_rights_data',
                         'final_test_observations_or_scores', 'sqlite_and_sidecars', 'raw_predictions_or_residuals'):
            self.assertIn(category, self.policy['automatic_export_blocked_categories'])
        self.assertTrue(all(p.startswith('artifacts/dvc_foundation/') for p in
                            self.policy['candidate_dvc_outputs_require_separate_approval']))

    def test_dvc_stays_local_and_credential_templates_remain_empty(self):
        config = configparser.ConfigParser(); config.read(ROOT/'.dvc/config')
        self.assertFalse(any(s.startswith('remote ') for s in config.sections()))
        self.assertFalse(config.has_option('core', 'remote'))
        self.assertEqual(config['cache']['type'], 'copy')
        template = dict(line.split('=', 1) for line in (ROOT/'.env.example').read_text().splitlines()
                        if '=' in line and not line.startswith('#'))
        for key in self.policy['credential_env_names_for_later_review']:
            self.assertEqual(template[key], '')

    def test_ignore_rules_cover_secrets_data_outputs_and_sqlite_sidecars(self):
        paths = ['.env', '.env.production', 'credentials.json', '.dvc/config.local',
                 'data/raw/any-name.csv', 'graph_sensor_ids.txt', 'graph_sensor_locations.csv',
                 'distances_la_2012.csv', 'artifacts/model.json', 'reports/generated/run.json',
                 'reports/local/audit.json', '.dvc/cache/object', '.venv/installed.py',
                 'frontend/node_modules/package.json', 'frontend/dist/index.html']
        paths += [f'local{ext}{suffix}' for ext in ('.db', '.sqlite', '.sqlite3')
                  for suffix in ('', '-wal', '-shm', '-journal')]
        with patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')):
            result = subprocess.run(['git', '-c', f'safe.directory={ROOT.as_posix()}', 'check-ignore', '--stdin', '-z'],
                                    cwd=ROOT, input=('\0'.join(paths)+'\0').encode(),
                                    capture_output=True, timeout=30, check=True)
        self.assertEqual(set(result.stdout.decode().strip('\0').split('\0')), set(paths))


if __name__ == '__main__': unittest.main()
