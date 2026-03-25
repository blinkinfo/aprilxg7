import unittest

import numpy as np

from src.config import ModelConfig
from src.features_v2 import FeatureEngineV2
from tests.helpers import make_higher_tf_data, make_ohlcv, make_trade_data


class TestFeatureEngineV2(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig()
        self.engine = FeatureEngineV2(self.config)
        self.df = make_ohlcv(240)
        self.htf = make_higher_tf_data(self.df)
        self.trade_data = make_trade_data(self.df.tail(120))

    def test_feature_count_is_exactly_76(self):
        features = self.engine.compute_features(self.df, self.htf, self.trade_data, ffill=True)
        self.assertEqual(features.shape[1], 76)

    def test_no_nan_when_ffill_true(self):
        features = self.engine.compute_features(self.df, self.htf, self.trade_data, ffill=True)
        self.assertFalse(features.isna().any().any())

    def test_no_inf_values(self):
        features = self.engine.compute_features(self.df, self.htf, self.trade_data, ffill=True)
        self.assertTrue(np.isfinite(features.to_numpy()).all())

    def test_feature_names_match_column_names(self):
        features = self.engine.compute_features(self.df, self.htf, self.trade_data, ffill=True)
        self.assertEqual(self.engine.get_feature_names(), list(features.columns))

    def test_output_is_pure_features_only(self):
        features = self.engine.compute_features(self.df, self.htf, self.trade_data, ffill=True)
        forbidden = {"timestamp", "open", "high", "low", "close", "volume"}
        self.assertTrue(forbidden.isdisjoint(set(features.columns)))

    def test_works_without_higher_tf_data(self):
        features = self.engine.compute_features(self.df, None, self.trade_data, ffill=True)
        self.assertEqual(features.shape[1], 76)
        self.assertFalse(features.isna().any().any())

    def test_works_without_trade_data(self):
        features = self.engine.compute_features(self.df, self.htf, None, ffill=True)
        self.assertEqual(features.shape[1], 76)
        self.assertFalse(features.isna().any().any())
        self.assertIn("buy_volume_ratio", features.columns)

    def test_deterministic_same_input_same_output(self):
        a = self.engine.compute_features(self.df, self.htf, self.trade_data, ffill=True)
        b = self.engine.compute_features(self.df, self.htf, self.trade_data, ffill=True)
        self.assertTrue(a.equals(b))


if __name__ == "__main__":
    unittest.main()
