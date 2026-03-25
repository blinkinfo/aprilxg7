import math
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.config import ModelConfig
from src.ensemble import EnsembleModel
from src.regime import RegimeDetector


class DummyProbModel:
    def __init__(self, prob: float, n_features: int):
        self.prob = float(prob)
        self.feature_importances_ = np.linspace(1.0, 2.0, n_features)

    def predict_proba(self, X):
        n = len(X)
        p = np.full(n, self.prob, dtype=float)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def make_ohlcv(n: int = 500, seed: int = 42, start: str = "2026-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start, periods=n, freq="5min", tz="UTC")

    trend = np.linspace(0, 150, n)
    seasonal = 40 * np.sin(np.linspace(0, 16 * math.pi, n))
    noise = rng.normal(0, 8, n).cumsum() * 0.2
    close = 100000 + trend + seasonal + noise
    open_ = np.roll(close, 1)
    open_[0] = close[0] - rng.normal(0, 3)

    spread = rng.uniform(4, 16, n)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.uniform(50, 250, n) + 20 * np.sin(np.linspace(0, 8 * math.pi, n))
    volume = np.clip(volume, 1.0, None)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_.astype(float),
            "high": high.astype(float),
            "low": low.astype(float),
            "close": close.astype(float),
            "volume": volume.astype(float),
        }
    )


def make_higher_tf_data(df_5m: pd.DataFrame) -> dict[str, pd.DataFrame]:
    base = df_5m.set_index("timestamp")

    def resample(rule: str) -> pd.DataFrame:
        agg = base.resample(rule).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        ).dropna()
        agg = agg.reset_index()
        return agg

    return {"15m": resample("15min"), "1h": resample("1h")}


def make_trade_data(df_5m: pd.DataFrame, trades_per_candle: int = 4) -> pd.DataFrame:
    rows = []
    trade_id = 1
    for idx, row in df_5m.iterrows():
        ts = pd.Timestamp(row["timestamp"])
        direction_up = row["close"] >= row["open"]
        for offset in range(trades_per_candle):
            is_buyer_maker = not direction_up if offset % 2 == 0 else direction_up
            qty = float(row["volume"]) / trades_per_candle / max(float(row["close"]), 1.0)
            rows.append(
                {
                    "id": trade_id,
                    "price": float(row["close"]),
                    "qty": qty,
                    "quoteQty": qty * float(row["close"]),
                    "time": ts + pd.Timedelta(seconds=offset * 30),
                    "isBuyerMaker": bool(is_buyer_maker),
                }
            )
            trade_id += 1
    return pd.DataFrame(rows)


def make_regime_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "adx_10": [0.35, 0.33, 0.12, 0.22],
            "ema_cross": [0.8, -0.7, 0.02, 0.01],
            "atr_ratio": [1.0, 1.0, 0.9, 1.35],
            "bb_squeeze": [0.0, 0.0, 1.0, 0.0],
        }
    )


def make_trained_stub_ensemble(feature_names: list[str]) -> EnsembleModel:
    config = ModelConfig()
    ensemble = EnsembleModel(config)
    selected = feature_names[:25]
    ensemble.feature_names = {
        "momentum": selected,
        "mean_reversion": selected,
        "microstructure": selected,
    }
    ensemble.momentum_model = DummyProbModel(0.66, len(selected))
    ensemble.mean_reversion_model = DummyProbModel(0.42, len(selected))
    ensemble.microstructure_model = DummyProbModel(0.61, len(selected))
    ensemble.calibrators = {}
    ensemble.calibrator_types = {}
    ensemble.is_trained = True
    ensemble.training_stats = {"oos_accuracy": 0.55}
    ensemble.last_train_time = None
    return ensemble


def configure_passthrough_calibrators(ensemble: EnsembleModel):
    detector = RegimeDetector()
    ensemble.calibrators = {
        detector.TRENDING_UP: None,
        detector.TRENDING_DOWN: None,
        detector.RANGING: None,
        detector.VOLATILE: None,
    }
    ensemble.calibrator_types = {
        detector.TRENDING_UP: "passthrough",
        detector.TRENDING_DOWN: "passthrough",
        detector.RANGING: "passthrough",
        detector.VOLATILE: "passthrough",
    }
    return ensemble
