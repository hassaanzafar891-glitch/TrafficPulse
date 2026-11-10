"""No forecasting fits, calibration, controller selection or reserved observations."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from trafficpulse.dvc_foundation import (BASE, CONFIG, INVENTORY, EXPECTED, WORKFLOW_DEPS,
    load_settings, prefix_summary, safe_path, snapshot_artifacts, validate_pipeline, verify_artifacts)

ROOT = Path(__file__).resolve().parents[1]


class FoundationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/'reports/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in (CONFIG, INVENTORY, 'dvc.yaml'):
            target = self.root/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT/name).read_bytes())

    def save(self, name, data):
        (self.root/name).write_text(json.dumps(data), encoding='utf-8')

    def seed_artifacts(self):
        inventory = json.loads((self.root/INVENTORY).read_text())
        for item in inventory['artifacts']:
            # Test-only files, not source models or external datasets.
            content = json.dumps({'synthetic_test_only': item['path']}).encode()
            target = self.root/item['path']; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            item.update(size_bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
        self.save(INVENTORY, inventory)

    def test_snapshot_is_exact_and_does_not_overwrite_reviewed_inputs(self):
        self.seed_artifacts()
        before = {name: (self.root/name).read_bytes() for name in EXPECTED}
        snapshot_artifacts(self.root); snapshot_artifacts(self.root)
        self.assertEqual(len(verify_artifacts(self.root, snapshot=True)), 11)
        self.assertEqual(before, {name: (self.root/name).read_bytes() for name in EXPECTED})

    def test_corrupt_snapshot_and_extra_files_are_rejected(self):
        self.seed_artifacts(); snapshot_artifacts(self.root)
        extra = self.root/BASE/'reviewed_bundle/secret.db'; extra.write_bytes(b'test')
        with self.assertRaises(ValueError): verify_artifacts(self.root, snapshot=True)
        extra.unlink()
        target = self.root/BASE/'reviewed_bundle'/next(iter(EXPECTED)); target.write_bytes(b'bad')
        with self.assertRaises(ValueError): snapshot_artifacts(self.root)

    def test_restricted_inventory_path_is_rejected_before_reading(self):
        inventory = json.loads((self.root/INVENTORY).read_text())
        inventory['artifacts'][0]['path'] = 'data/raw/restricted.h5'
        self.save(INVENTORY, inventory)
        with self.assertRaises(ValueError): verify_artifacts(self.root)

    def test_path_escape_and_unsafe_output_are_rejected(self):
        for path in ('../escape', 'E:/other', 'data\\raw', '/absolute'):
            with self.assertRaises(ValueError): safe_path(self.root, path)
        pipeline = json.loads((self.root/'dvc.yaml').read_text())
        pipeline['stages']['training_prefix']['outs'] = ['artifacts/synthetic_xgboost']
        self.save('dvc.yaml', pipeline)
        with self.assertRaises(ValueError): validate_pipeline(self.root)

    def test_dependency_removal_or_training_command_cannot_run(self):
        pipeline = json.loads((self.root/'dvc.yaml').read_text())
        pipeline['stages']['verify_reviewed_bundle']['deps'].remove('src/trafficpulse/tracking.py')
        self.save('dvc.yaml', pipeline)
        with self.assertRaises(ValueError): validate_pipeline(self.root)
        pipeline = json.loads((ROOT/'dvc.yaml').read_text())
        pipeline['stages']['training_prefix']['cmd'] = 'python scripts/run_synthetic_baselines.py --xgboost'
        self.save('dvc.yaml', pipeline)
        with self.assertRaises(ValueError): validate_pipeline(self.root)

    def test_all_safety_parameters_and_snapshot_freeze_are_required(self):
        for field in ('allow_training', 'allow_calibration', 'allow_controller_selection', 'allow_final_test'):
            settings = json.loads((ROOT/CONFIG).read_text()); settings[field] = True
            self.save(CONFIG, settings)
            with self.assertRaises(ValueError): load_settings(self.root)
        self.save(CONFIG, json.loads((ROOT/CONFIG).read_text()))
        pipeline = json.loads((self.root/'dvc.yaml').read_text())
        pipeline['stages']['reviewed_bundle']['frozen'] = False
        self.save('dvc.yaml', pipeline)
        with self.assertRaises(ValueError): validate_pipeline(self.root)

    def test_training_prefix_is_deterministic_and_stops_before_reserved_data(self):
        first = prefix_summary(self.root); second = prefix_summary(self.root)
        self.assertEqual(first, second)
        self.assertEqual(first['raw_rows'], [0, 5644])
        self.assertEqual(first['rows'], 5644)
        self.assertEqual(first['last_timestamp'], '2024-01-20T14:15:00.000000000')
        self.assertFalse(first['training_performed']); self.assertFalse(first['final_test_access'])

    def test_real_pipeline_covers_all_workflows_and_only_safe_outputs(self):
        validate_pipeline(ROOT)
        for module in ('experiment', 'xgboost_experiment', 'fault_experiment', 'uncertainty_experiment', 'controller_experiment', 'tracking'):
            self.assertIn(f'src/trafficpulse/{module}.py', WORKFLOW_DEPS)
        self.assertTrue(all((ROOT/p).is_file() for p in WORKFLOW_DEPS))


if __name__ == '__main__': unittest.main()
