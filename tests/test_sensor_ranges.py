import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.simulink_bridge import (
    SENSOR_COLUMNS,
    evaluate_simulink_run,
    get_sensor_range_warnings,
    load_training_sensor_ranges,
)


class SensorRangeTests(unittest.TestCase):
    def test_bounds_exclude_test_rows(self):
        df = pd.DataFrame({sensor: np.arange(50, dtype=float)
                           for sensor in SENSOR_COLUMNS})
        df['machine_failure'] = [0, 1] * 25
        training, testing = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df['machine_failure']
        )
        df.loc[testing.index[0], SENSOR_COLUMNS] = -10000
        df.loc[testing.index[1], SENSOR_COLUMNS] = 10000
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'processed.csv'
            df.to_csv(path, index=False)
            ranges = load_training_sensor_ranges(path)
        self.assertEqual(set(ranges), set(SENSOR_COLUMNS))
        for sensor in SENSOR_COLUMNS:
            self.assertEqual(ranges[sensor],
                             (training[sensor].min(), training[sensor].max()))

    def test_values_on_boundaries_do_not_warn(self):
        ranges = {sensor: (10.0, 20.0) for sensor in SENSOR_COLUMNS}
        for value in [10.0, 15.0, 20.0]:
            with self.subTest(value=value):
                self.assertEqual(get_sensor_range_warnings(
                    dict.fromkeys(SENSOR_COLUMNS, value), ranges), [])

    def test_warns_for_each_sensor_below_or_above_range(self):
        ranges = {sensor: (10.0, 20.0) for sensor in SENSOR_COLUMNS}
        for sensor in SENSOR_COLUMNS:
            for value in [9.0, 21.0]:
                with self.subTest(sensor=sensor, value=value):
                    sample = dict.fromkeys(SENSOR_COLUMNS, 15.0)
                    sample[sensor] = value
                    original = sample.copy()
                    self.assertEqual(get_sensor_range_warnings(sample, ranges), [{
                        'sensor': sensor, 'value': value,
                        'minimum': 10.0, 'maximum': 20.0,
                    }])
                    self.assertEqual(sample, original)

    def test_mat_warnings_are_separate_and_preserve_prediction_results(self):
        matrix = np.array([
            [0, 1500, 310, 40, 300, 50],
            [50, 1500, 325, 40, 300, 50],
        ], dtype=float)
        ranges = {
            'rotational_speed_rpm': (1168, 2886),
            'process_temperature_k': (305.7, 313.8),
            'torque_nm': (3.8, 76.6),
            'air_temperature_k': (295.3, 304.5),
            'tool_wear_min': (0, 253),
        }
        prediction = pd.DataFrame({'failure_probability': [0.02, 0.02],
                                   'failure_predicted': [0, 0], 'is_anomaly': [0, 0]})
        with tempfile.TemporaryDirectory() as directory:
            for transpose in [False, True]:
                with self.subTest(transpose=transpose):
                    path = Path(directory) / 'telemetry.mat'
                    with h5py.File(path, 'w') as mat:
                        mat['telemetry_data'] = matrix.T if transpose else matrix
                    output = io.StringIO()
                    with patch('src.simulink_bridge.load_training_sensor_ranges',
                               return_value=ranges), patch(
                        'src.simulink_bridge.predict_batch',
                        return_value=prediction,
                    ) as predictor, contextlib.redirect_stdout(output):
                        result = evaluate_simulink_run(str(path))
                    np.testing.assert_array_equal(result[SENSOR_COLUMNS], matrix[:, 1:])
                    self.assertEqual(result['failure_prob'].tolist(), [0.02, 0.02])
                    self.assertEqual(result['is_anomaly'].tolist(), [0, 0])
                    predictor.assert_called_once()
                    np.testing.assert_array_equal(
                        predictor.call_args.args[0][SENSOR_COLUMNS], matrix[:, 1:])
                    self.assertIn('não são falhas previstas nem anomalias detectadas',
                                  output.getvalue())
                    self.assertIn('Amostra 1: process_temperature_k = 325', output.getvalue())
                    self.assertIn('[305.7, 313.8]', output.getvalue())
                    self.assertNotIn('Amostra 0:', output.getvalue())


if __name__ == '__main__':
    unittest.main()
