import contextlib
import hashlib
import io
import os
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import Mock, patch

import h5py
import joblib
import numpy as np
import pandas as pd

from src.data_loader import load_raw_data
from src.features import create_features, save_processed_data
from src.model_store import MODEL_FILENAMES, load_models, publish_models
from src.predict import predict_batch, predict_telemetry
from src.simulink_bridge import evaluate_simulink_run, read_simulink_mat
from src.train import train_models
from src.validation import FEATURE_COLUMNS, ROOT_DIR, SENSOR_COLUMNS


def sample_frame():
    return pd.DataFrame([dict(zip(SENSOR_COLUMNS, [1500, 310, 40, 300, 50]))])


class ValidationTests(unittest.TestCase):
    def test_invalid_sensor_inputs_are_rejected_before_loading_models(self):
        with patch('src.predict.load_models') as loader:
            for value in [None, '50', 'invalid', np.nan, np.inf, -np.inf,
                          True, 1 + 2j, [50], 1e100]:
                with self.subTest(value=value), self.assertRaises(ValueError):
                    sample = sample_frame().iloc[0].to_dict()
                    sample['torque_nm'] = value
                    predict_telemetry(sample)
            for sample in [{}, [], {'torque_nm': 5}]:
                with self.subTest(sample=sample), self.assertRaises(ValueError):
                    predict_telemetry(sample)
            loader.assert_not_called()

    def test_feature_calculation_does_not_mutate_input_or_invent_physical_limits(self):
        sample = sample_frame().astype(float)
        sample.loc[0, 'rotational_speed_rpm'] = 0
        sample.loc[0, 'torque_nm'] = -1
        original = sample.copy(deep=True)
        result = create_features(sample)
        pd.testing.assert_frame_equal(sample, original)
        self.assertEqual(result.temp_difference_k.iloc[0], 10)
        self.assertEqual(result.power_estimate.iloc[0], 0)
        self.assertAlmostEqual(result.torque_speed_ratio.iloc[0], -1 / 1e-5)

    def test_csv_rejects_missing_and_duplicate_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.csv'
            for contents in ['', 'Torque [Nm],Torque [Nm]\n40,41\n', 'unrelated\n1\n']:
                path.write_text(contents)
                with self.subTest(contents=contents), self.assertRaises(ValueError):
                    load_raw_data(path)

    def test_processed_data_and_labels_validated_before_fit(self):
        valid = pd.concat([create_features(sample_frame())] * 50, ignore_index=True)
        valid['machine_failure'] = [0] * 40 + [1] * 10
        cases = []
        for labels in [[0] * 50, [1] * 50, [0] * 49 + [1],
                       [0] * 10 + [1] * 40, [0] * 49 + [2]]:
            frame = valid.copy()
            frame['machine_failure'] = labels
            cases.append(frame)
        invalid = valid.copy()
        invalid.loc[0, 'power_estimate'] += 100
        cases.append(invalid)
        with tempfile.TemporaryDirectory() as directory, patch(
            'src.train.RandomForestClassifier'
        ) as classifier:
            path = Path(directory) / 'processed.csv'
            for frame in cases:
                frame.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    train_models(path, Path(directory) / 'models')
            classifier.assert_not_called()
            self.assertFalse((Path(directory) / 'models').exists())

    def test_invalid_mat_structure_and_time(self):
        good = np.array([[0, 1500, 310, 40, 300, 50],
                         [1, 1500, 310, 40, 300, 50]], dtype=float)
        backwards = good.copy()
        backwards[:, 0] = [1, 0]
        nonfinite = good.copy()
        nonfinite[0, 2] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.mat'
            for data in [np.ones(6), np.ones((2, 5)), np.ones((6, 6)),
                         np.empty((0, 6)), backwards, nonfinite,
                         np.full((2, 6), b'text')]:
                with self.subTest(shape=data.shape):
                    with h5py.File(path, 'w') as mat:
                        mat['telemetry_data'] = data
                    with self.assertRaises(ValueError):
                        read_simulink_mat(path)
            with h5py.File(path, 'w') as mat:
                mat['wrong_name'] = good
            with self.assertRaisesRegex(ValueError, 'telemetry_data'):
                read_simulink_mat(path)


class ModelPublicationTests(unittest.TestCase):
    def test_legacy_loading_and_successful_publication_preserve_old_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for model, name in zip(('legacy_rf', 'legacy_if'), MODEL_FILENAMES):
                joblib.dump(model, root / name)
            old = {name: (root / name).read_bytes() for name in MODEL_FILENAMES}
            self.assertEqual(load_models(root), ('legacy_rf', 'legacy_if'))
            publish_models('new_rf', 'new_if', root)
            self.assertEqual(load_models(root), ('new_rf', 'new_if'))
            for name in MODEL_FILENAMES:
                self.assertEqual((root / name).read_bytes(), old[name])

    def test_second_write_failure_leaves_previous_pair_available(self):
        with tempfile.TemporaryDirectory() as directory:
            publish_models('old_rf', 'old_if', directory)
            real_dump = joblib.dump
            calls = []

            def fail_second(model, path):
                calls.append(path)
                if len(calls) == 2:
                    raise OSError('simulated disk error')
                return real_dump(model, path)

            with patch('src.model_store.joblib.dump', side_effect=fail_second):
                with self.assertRaises(OSError):
                    publish_models('new_rf', 'new_if', directory)
            self.assertEqual(load_models(directory), ('old_rf', 'old_if'))

    def test_manifest_switch_failure_leaves_previous_pair_available(self):
        with tempfile.TemporaryDirectory() as directory:
            publish_models('old_rf', 'old_if', directory)
            replace = os.replace

            def fail_manifest(source, destination):
                if Path(destination).name == 'current.json':
                    raise OSError('simulated publication error')
                return replace(source, destination)

            with patch('src.model_store.os.replace', side_effect=fail_manifest):
                with self.assertRaises(OSError):
                    publish_models('new_rf', 'new_if', directory)
            self.assertEqual(load_models(directory), ('old_rf', 'old_if'))

    def test_reader_keeps_one_generation_during_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            publish_models('old_rf', 'old_if', directory)
            real_load = joblib.load
            triggered = False

            def publish_between_reads(path):
                nonlocal triggered
                result = real_load(path)
                if not triggered:
                    triggered = True
                    publish_models('new_rf', 'new_if', directory)
                return result

            with patch('src.model_store.joblib.load', side_effect=publish_between_reads):
                self.assertEqual(load_models(directory), ('old_rf', 'old_if'))
            self.assertEqual(load_models(directory), ('new_rf', 'new_if'))


class PredictionRegressionTests(unittest.TestCase):
    def test_threshold_classifies_before_rounding_and_loads_models_once(self):
        classifier, detector = Mock(), Mock()
        classifier.predict_proba.return_value = np.array([
            [0.5, 0.5], [0.50004, 0.49996], [0.49996, 0.50004]])
        detector.predict.return_value = np.array([1, 1, -1])
        samples = pd.concat([sample_frame()] * 3, ignore_index=True)
        with patch('src.predict.load_models', return_value=(classifier, detector)) as loader:
            result = predict_batch(samples)
        loader.assert_called_once()
        classifier.predict_proba.assert_called_once()
        detector.predict.assert_called_once()
        self.assertEqual(result.failure_probability.tolist(), [0.5, 0.5, 0.5])
        self.assertEqual(result.failure_predicted.tolist(), [1, 0, 1])

    def test_bridge_counts_classification_instead_of_rounded_probability(self):
        predicted = pd.DataFrame({'failure_probability': [0.5],
                                  'failure_predicted': [1], 'is_anomaly': [0]})
        output = io.StringIO()
        with patch('src.simulink_bridge.read_simulink_mat', return_value=sample_frame()), patch(
            'src.simulink_bridge.predict_batch', return_value=predicted
        ), contextlib.redirect_stdout(output):
            result = evaluate_simulink_run('telemetry_normal_torque_teste.mat')
        self.assertEqual(result.failure_predicted.sum(), 1)
        self.assertIn('Alertas detetados: 1', output.getvalue())

    def test_real_mat_batch_matches_legacy_row_predictions(self):
        classifier, detector = load_models()
        for filename, count in [('telemetry_normal_torque_teste.mat', 58),
                                ('telemetry_fault_corrigido.mat', 60)]:
            with self.subTest(filename=filename):
                path = ROOT_DIR / filename
                self.assertTrue(path.exists(), f'Arquivo de regressão ausente: {path}')
                before = hashlib.sha256(path.read_bytes()).hexdigest()
                raw = read_simulink_mat(path)
                expected = []
                for _, row in raw.iterrows():
                    X = create_features(pd.DataFrame([row]))[FEATURE_COLUMNS]
                    probability = float(classifier.predict_proba(X)[0, 1])
                    expected.append((round(probability, 4), int(probability >= 0.5),
                                     int(detector.predict(X)[0] == -1)))
                with contextlib.redirect_stdout(io.StringIO()):
                    result = evaluate_simulink_run(filename)
                self.assertEqual(len(result), count)
                np.testing.assert_array_equal(
                    result[['failure_prob', 'failure_predicted', 'is_anomaly']], expected)
                self.assertEqual(result.failure_predicted.sum(), 9)
                self.assertEqual(result.is_anomaly.sum(), 0)
                with h5py.File(path) as mat:
                    times = np.array(mat['telemetry_data'])[:, 0]
                self.assertEqual(result.loc[times >= 50, 'failure_predicted'].sum(), 0)
                self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_default_paths_work_outside_project(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                self.assertEqual(load_raw_data().shape, (10000, 14))
                self.assertEqual(predict_telemetry(sample_frame().iloc[0].to_dict())['failure_predicted'], 0)
                self.assertEqual(len(read_simulink_mat('telemetry_normal_torque_teste.mat')), 58)
            finally:
                os.chdir(previous)

    def test_main_entrypoint_without_retraining_or_overwriting_data(self):
        original_save = save_processed_data
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'processed.csv'
            with patch('src.train.train_models') as trainer, patch(
                'src.features.save_processed_data',
                side_effect=lambda df, path: original_save(df, destination),
            ), contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(str(ROOT_DIR / 'main.py'), run_name='__main__')
            self.assertEqual(pd.read_csv(destination).shape, (10000, 18))
            trainer.assert_called_once_with('data/processed/ai4i2020_processed.csv', 'models')

    def test_training_flow_preserves_parameters_and_evaluates_before_publication(self):
        frame = pd.concat([create_features(sample_frame())] * 50, ignore_index=True)
        frame['machine_failure'] = [0] * 40 + [1] * 10
        classifier, detector = Mock(), Mock()
        classifier.predict_proba.return_value = np.tile([0.8, 0.2], (10, 1))
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'training.csv'
            frame.to_csv(data, index=False)
            with patch('src.train.RandomForestClassifier', return_value=classifier) as rf, patch(
                'src.train.IsolationForest', return_value=detector
            ) as iso, patch('src.train.publish_models') as publish, contextlib.redirect_stdout(io.StringIO()):
                train_models(data, directory)
            rf.assert_called_once_with(n_estimators=100, class_weight='balanced', random_state=42)
            iso.assert_called_once_with(contamination=0.2, random_state=42)
            self.assertEqual(classifier.fit.call_args.args[0].shape, (40, 9))
            classifier.fit.assert_called_once()
            detector.fit.assert_called_once()
            publish.assert_called_once_with(classifier, detector, directory)

    def test_real_training_uses_only_synthetic_temporary_artifacts(self):
        raw = pd.concat([sample_frame()] * 50, ignore_index=True)
        raw['torque_nm'] = np.linspace(30, 55, 50)
        frame = create_features(raw)
        frame['machine_failure'] = [0] * 40 + [1] * 10
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.csv'
            destination = Path(directory) / 'models'
            frame.to_csv(path, index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                train_models(path, destination)
            classifier, detector = load_models(destination)
            self.assertEqual(classifier.n_estimators, 100)
            self.assertEqual(classifier.class_weight, 'balanced')
            self.assertEqual(detector.contamination, 0.2)
            result = predict_batch(raw, destination)
            self.assertEqual(len(result), 50)
            self.assertTrue((destination / 'current.json').exists())


if __name__ == '__main__':
    unittest.main()
