"""Test tracking endpoint isolation without requiring a server or real data."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from trafficpulse.tracking import log_local_run


class TrackingTests(unittest.TestCase):
    def test_local_uri_overrides_inherited_remote_and_logs_no_arrays(self):
        mlflow=MagicMock()
        mlflow.get_experiment_by_name.return_value=None
        mlflow.create_experiment.return_value='1'
        mlflow.start_run.return_value.__enter__.return_value=SimpleNamespace(info=SimpleNamespace(run_id='synthetic-run'))
        result={'data_origin':'synthetic_generated_not_real_traffic',
                'parameters':{'seed':1729},'metrics':{'persistence':{'3':{'mae':1.,'rmse':2.,'valid_target_count':4}}}}
        with TemporaryDirectory() as location, patch.dict('sys.modules',{'mlflow':mlflow}), patch.dict('os.environ',{'MLFLOW_TRACKING_URI':'https://invalid.example'}):
            outcome=log_local_run(result,Path(location))
        self.assertTrue(outcome['tracking_uri'].startswith('sqlite:///'))
        mlflow.set_tracking_uri.assert_called_once_with(outcome['tracking_uri'])
        mlflow.set_registry_uri.assert_called_once_with(outcome['tracking_uri'])
        self.assertEqual(mlflow.log_metric.call_count,3)
        mlflow.log_artifact.assert_not_called()

    def test_uncertainty_metrics_log_defined_values_only(self):
        mlflow=MagicMock()
        mlflow.get_experiment_by_name.return_value=SimpleNamespace(experiment_id='1')
        mlflow.start_run.return_value.__enter__.return_value=SimpleNamespace(info=SimpleNamespace(run_id='synthetic-run'))
        result={'data_origin':'synthetic_generated_not_real_traffic','parameters':{},
                'metrics':{'calibration':{'3':{'coverage':.9,'average_interval_width':4.,
                    'valid_target_count':10,'calibration_count':10,'calibrated_half_width':2.},
                    '6':{'coverage':None,'average_interval_width':None,'valid_target_count':0,
                         'calibration_count':0,'calibrated_half_width':None}}}}
        with TemporaryDirectory() as location, patch.dict('sys.modules',{'mlflow':mlflow}):
            log_local_run(result,Path(location))
        self.assertEqual(mlflow.log_metric.call_count,7)
        mlflow.log_metric.assert_any_call('calibration_h3_coverage',.9)
        self.assertFalse(any(call.args[1] is None for call in mlflow.log_metric.call_args_list))
