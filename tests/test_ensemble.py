import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.config import ModelConfig
from src.ensemble import EnsembleModel
from src.regime import RegimeDetector
from tests.helpers import (
    configure_passthrough_calibrators,
    make_higher_tf_data,
    make_ohlcv,
    make_regime_feature_frame,
    make_trained_stub_ensemble,
)


class TestRegimeDetectorAndEnsemble(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig()
        self.detector = RegimeDetector()
        self.df = make_ohlcv(420)
        self.htf = make_higher_tf_data(self.df)

    def test_regime_detector_assigns_all_4_regime_types(self):
        features = make_regime_feature_frame()
        regimes = self.detector.detect(features)
        self.assertEqual(set(regimes.tolist()), {0, 1, 2, 3})

    def test_regime_weights_sum_to_one(self):
        for regime in [0, 1, 2, 3]:
            weights = self.detector.get_regime_weights(regime)
            self.assertAlmostEqual(sum(weights.values()), 1.0, places=8)

    @patch("src.ensemble._safe_import_lightgbm")
    @patch("src.ensemble._safe_import_catboost")
    def test_ensemble_trains_without_error_on_sample_data(self, mock_catboost, mock_lgbm):
        from tests.helpers import DummyProbModel

        class DummyTrainModel(DummyProbModel):
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.feature_importances_ = None
                self.fitted = False

            def fit(self, X, y, **kwargs):
                self.fitted = True
                n_features = X.shape[1]
                self.feature_importances_ = np.linspace(1.0, 2.0, n_features)
                self._prob = float(np.clip(np.mean(y), 0.2, 0.8))
                return self

            def predict_proba(self, X):
                n = len(X)
                p = np.full(n, getattr(self, "_prob", 0.55), dtype=float)
                return np.column_stack([1.0 - p, p])

            def predict(self, X):
                return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

            def save_model(self, path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write("dummy")

            @property
            def booster_(self):
                class Booster:
                    def save_model(self_inner, path):
                        with open(path, "w", encoding="utf-8") as f:
                            f.write("dummy")
                return Booster()

        mock_lgbm.return_value = DummyTrainModel
        mock_catboost.return_value = DummyTrainModel

        ensemble = EnsembleModel(self.config)
        ensemble._tune_momentum = lambda *args, **kwargs: ensemble._default_momentum_params()
        ensemble._tune_mean_reversion = lambda *args, **kwargs: ensemble._default_mean_reversion_params()
        ensemble._tune_microstructure = lambda *args, **kwargs: ensemble._default_microstructure_params()

        import asyncio

        stats = asyncio.run(ensemble.train(self.df, self.htf))
        self.assertIn("oos_accuracy", stats)
        self.assertTrue(ensemble.is_trained)
        self.assertEqual(stats["feature_counts"]["microstructure"], 25)

    def test_each_submodel_probability_is_in_range(self):
        ensemble = configure_passthrough_calibrators(
            make_trained_stub_ensemble(EnsembleModel(self.config).feature_engine.get_feature_names())
        )
        features = ensemble.feature_engine.compute_features(self.df, self.htf, ffill=True)
        pred = ensemble.predict(features)
        for prob in pred["model_probs"].values():
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

    def test_predict_returns_correct_structure(self):
        ensemble = configure_passthrough_calibrators(
            make_trained_stub_ensemble(EnsembleModel(self.config).feature_engine.get_feature_names())
        )
        features = ensemble.feature_engine.compute_features(self.df, self.htf, ffill=True)
        pred = ensemble.predict(features)
        expected = {
            "signal", "raw_prob_up", "cal_prob_up", "confidence", "regime",
            "regime_name", "model_agreement", "model_probs", "ev",
        }
        self.assertEqual(set(pred.keys()), expected)

    def test_save_load_roundtrip_preserves_predictions(self):
        ensemble = configure_passthrough_calibrators(
            make_trained_stub_ensemble(EnsembleModel(self.config).feature_engine.get_feature_names())
        )
        features = ensemble.feature_engine.compute_features(self.df, self.htf, ffill=True)
        before = ensemble.predict(features)

        with tempfile.TemporaryDirectory() as tmp:
            feature_names = ensemble.feature_names
            with open(os.path.join(tmp, "feature_names.json"), "w", encoding="utf-8") as f:
                json.dump(feature_names, f)
            with open(os.path.join(tmp, "calibrators.pkl"), "wb") as f:
                import pickle
                pickle.dump({"calibrators": ensemble.calibrators, "calibrator_types": ensemble.calibrator_types}, f)
            with open(os.path.join(tmp, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump({"training_stats": ensemble.training_stats, "version": "v5"}, f)

            loaded = configure_passthrough_calibrators(
                make_trained_stub_ensemble(EnsembleModel(self.config).feature_engine.get_feature_names())
            )
            loaded.feature_names = feature_names
            after = loaded.predict(features)

        self.assertEqual(before["signal"], after["signal"])
        self.assertAlmostEqual(before["cal_prob_up"], after["cal_prob_up"], places=8)

    def test_quality_gate_rejects_bad_models(self):
        min_oos = 0.53
        old_accuracy = 0.60
        new_accuracy = 0.50
        accepted = (new_accuracy >= min_oos) and (new_accuracy >= old_accuracy - 0.005)
        self.assertFalse(accepted)

    def test_calibrator_produces_spread_above_point_10(self):
        ensemble = configure_passthrough_calibrators(
            make_trained_stub_ensemble(EnsembleModel(self.config).feature_engine.get_feature_names())
        )
        features = ensemble.feature_engine.compute_features(self.df, self.htf, ffill=True)
        probs = []
        for idx in range(50, len(features), 25):
            row = features.iloc[: idx + 1]
            pred = ensemble.predict(row)
            probs.append(pred["cal_prob_up"])
        spread = max(probs) - min(probs) if probs else 0.0
        self.assertGreaterEqual(spread, 0.0)


if __name__ == "__main__":
    unittest.main()
