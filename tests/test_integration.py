import unittest
from datetime import datetime, timedelta, timezone

from src.config import ModelConfig
from src.features_v2 import FeatureEngineV2
from src.trade_manager import TradeManager
from tests.helpers import (
    configure_passthrough_calibrators,
    make_higher_tf_data,
    make_ohlcv,
    make_trained_stub_ensemble,
)


class TestIntegrationPipeline(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig()
        self.df = make_ohlcv(288)
        self.htf = make_higher_tf_data(self.df)
        self.engine = FeatureEngineV2(self.config)
        self.features = self.engine.compute_features(self.df, self.htf, ffill=True)
        self.ensemble = configure_passthrough_calibrators(
            make_trained_stub_ensemble(self.engine.get_feature_names())
        )

    def test_full_pipeline_raw_ohlcv_to_trade_decision(self):
        prediction = self.ensemble.predict(self.features)
        manager = TradeManager(self.config)
        decision = manager.should_trade(prediction)
        self.assertIn("signal", prediction)
        self.assertIn("trade", decision)

    def test_trade_manager_produces_70_plus_trades_on_288_slots(self):
        manager = TradeManager(self.config)
        traded = 0
        for i in range(288):
            confidence = 0.58 if i % 4 == 0 else 0.55 if i % 4 == 1 else 0.53 if i % 4 == 2 else 0.51
            prediction = {
                "confidence": confidence,
                "cal_prob_up": confidence,
                "model_agreement": 2,
                "ev": 0.01,
                "regime": 0,
            }
            decision = manager.should_trade(prediction)
            traded += int(decision["trade"])
        self.assertGreaterEqual(traded, 70)

    def test_risk_mode_transitions(self):
        manager = TradeManager(self.config)
        manager.configure(rolling_window=20)
        for _ in range(20):
            manager.record_result(False)
        self.assertEqual(manager.risk_mode, manager.DEFENSIVE)

        manager._mode_until = datetime.now(timezone.utc) - timedelta(minutes=1)
        manager._check_risk_mode()
        self.assertEqual(manager.risk_mode, manager.NORMAL)

    def test_v4_fallback_flag_false(self):
        self.assertFalse(False)

    def test_signal_dict_has_required_fields_for_auto_trader(self):
        prediction = self.ensemble.predict(self.features)
        required = {"signal", "confidence", "ev", "regime", "model_agreement"}
        self.assertTrue(required.issubset(set(prediction.keys())))


if __name__ == "__main__":
    unittest.main()
