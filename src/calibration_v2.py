"""Regime-conditional probability calibration.

Phase 3 of the AprilXG V5 multi-model ensemble upgrade.

Instead of one global isotonic calibrator (which collapsed all probs to ~50.4%),
fits separate calibrators per regime. Also uses Platt scaling as a fallback
when a regime has too few calibration samples.

Calibrator selection:
- If regime has >= 100 samples in CAL split: Isotonic regression
- If regime has 30-99 samples: Platt scaling (logistic regression)
- If regime has < 30 samples: No calibration (pass-through raw probability)
"""
import json
import logging
import os
import pickle
from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

# Thresholds for calibrator selection
ISOTONIC_MIN_SAMPLES = 100
PLATT_MIN_SAMPLES = 30

# Probability clipping range
PROB_CLIP_MIN = 0.01
PROB_CLIP_MAX = 0.99


class CalibratorV2:
    """Regime-conditional probability calibration.

    Instead of one global isotonic calibrator (which collapsed all probs to ~50.4%),
    fits separate calibrators per regime. Also uses Platt scaling as a fallback
    when a regime has too few calibration samples.

    Calibrator selection:
    - If regime has >= 100 samples in CAL split: Isotonic regression
    - If regime has 30-99 samples: Platt scaling (logistic regression)
    - If regime has < 30 samples: No calibration (pass-through raw probability)
    """

    def __init__(self):
        self.calibrators: dict[int, Optional[object]] = {}  # {regime_id: fitted calibrator}
        self.calibrator_types: dict[int, str] = {}  # {regime_id: "isotonic" | "platt" | "passthrough"}
        self.is_fitted: bool = False
        self._fit_stats: dict[int, dict] = {}  # Per-regime fitting statistics

    def fit(
        self,
        raw_probs: np.ndarray,
        true_labels: np.ndarray,
        regimes: np.ndarray,
    ):
        """Fit per-regime calibrators.

        Args:
            raw_probs: Raw model P(UP) probabilities, shape (n,)
            true_labels: True binary labels (1=UP, 0=DOWN), shape (n,)
            regimes: Regime labels (0-3) for each sample, shape (n,)
        """
        raw_probs = np.asarray(raw_probs, dtype=np.float64).ravel()
        true_labels = np.asarray(true_labels, dtype=np.int32).ravel()
        regimes = np.asarray(regimes, dtype=np.int32).ravel()

        if len(raw_probs) != len(true_labels) or len(raw_probs) != len(regimes):
            raise ValueError(
                f"Length mismatch: raw_probs={len(raw_probs)}, "
                f"true_labels={len(true_labels)}, regimes={len(regimes)}"
            )

        self.calibrators = {}
        self.calibrator_types = {}
        self._fit_stats = {}

        for regime_id in range(4):  # 0=TRENDING_UP, 1=TRENDING_DOWN, 2=RANGING, 3=VOLATILE
            mask = regimes == regime_id
            n_regime = int(mask.sum())

            regime_probs = raw_probs[mask]
            regime_labels = true_labels[mask]

            stats = {
                "n_samples": n_regime,
                "calibrator_type": "passthrough",
                "raw_prob_mean": float(np.mean(regime_probs)) if n_regime > 0 else 0.0,
                "raw_prob_std": float(np.std(regime_probs)) if n_regime > 0 else 0.0,
                "label_mean": float(np.mean(regime_labels)) if n_regime > 0 else 0.0,
            }

            if n_regime >= ISOTONIC_MIN_SAMPLES:
                # Isotonic regression — best calibration with enough data
                calibrator = IsotonicRegression(
                    y_min=PROB_CLIP_MIN,
                    y_max=PROB_CLIP_MAX,
                    out_of_bounds="clip",
                )
                calibrator.fit(regime_probs, regime_labels)
                self.calibrators[regime_id] = calibrator
                self.calibrator_types[regime_id] = "isotonic"
                stats["calibrator_type"] = "isotonic"

                # Verify spread was preserved
                cal_probs = calibrator.predict(regime_probs)
                stats["cal_prob_min"] = float(np.min(cal_probs))
                stats["cal_prob_max"] = float(np.max(cal_probs))
                stats["cal_prob_mean"] = float(np.mean(cal_probs))
                stats["cal_prob_std"] = float(np.std(cal_probs))
                stats["cal_spread"] = float(np.max(cal_probs) - np.min(cal_probs))

                logger.info(
                    f"CalibratorV2 [regime={regime_id}]: isotonic on {n_regime} samples, "
                    f"spread={stats['cal_spread']:.4f}, "
                    f"range=[{stats['cal_prob_min']:.3f}, {stats['cal_prob_max']:.3f}]"
                )

            elif n_regime >= PLATT_MIN_SAMPLES:
                # Platt scaling — logistic regression on raw probabilities
                calibrator = LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    solver="lbfgs",
                )
                calibrator.fit(regime_probs.reshape(-1, 1), regime_labels)
                self.calibrators[regime_id] = calibrator
                self.calibrator_types[regime_id] = "platt"
                stats["calibrator_type"] = "platt"

                # Verify spread
                cal_probs = calibrator.predict_proba(regime_probs.reshape(-1, 1))[:, 1]
                cal_probs = np.clip(cal_probs, PROB_CLIP_MIN, PROB_CLIP_MAX)
                stats["cal_prob_min"] = float(np.min(cal_probs))
                stats["cal_prob_max"] = float(np.max(cal_probs))
                stats["cal_prob_mean"] = float(np.mean(cal_probs))
                stats["cal_prob_std"] = float(np.std(cal_probs))
                stats["cal_spread"] = float(np.max(cal_probs) - np.min(cal_probs))

                logger.info(
                    f"CalibratorV2 [regime={regime_id}]: platt on {n_regime} samples, "
                    f"spread={stats['cal_spread']:.4f}, "
                    f"range=[{stats['cal_prob_min']:.3f}, {stats['cal_prob_max']:.3f}]"
                )

            else:
                # Too few samples — pass through raw probability
                self.calibrators[regime_id] = None
                self.calibrator_types[regime_id] = "passthrough"
                stats["calibrator_type"] = "passthrough"
                stats["cal_spread"] = float(np.max(regime_probs) - np.min(regime_probs)) if n_regime > 1 else 0.0

                logger.info(
                    f"CalibratorV2 [regime={regime_id}]: passthrough ({n_regime} samples, "
                    f"below threshold of {PLATT_MIN_SAMPLES})"
                )

            self._fit_stats[regime_id] = stats

        self.is_fitted = True
        logger.info(
            f"CalibratorV2 fitted: {dict(self.calibrator_types)}"
        )

    def calibrate(self, raw_prob: float, regime: int) -> float:
        """Calibrate a single probability.

        Args:
            raw_prob: Raw model P(UP) probability
            regime: Regime label (0-3)

        Returns:
            Calibrated probability, clipped to [0.01, 0.99]
        """
        if not self.is_fitted:
            logger.warning("CalibratorV2 not fitted — returning clipped raw probability")
            return float(np.clip(raw_prob, PROB_CLIP_MIN, PROB_CLIP_MAX))

        cal_type = self.calibrator_types.get(regime, "passthrough")
        calibrator = self.calibrators.get(regime)

        if cal_type == "passthrough" or calibrator is None:
            return float(np.clip(raw_prob, PROB_CLIP_MIN, PROB_CLIP_MAX))

        elif cal_type == "isotonic":
            cal_prob = float(calibrator.predict(np.array([raw_prob]))[0])
            return float(np.clip(cal_prob, PROB_CLIP_MIN, PROB_CLIP_MAX))

        elif cal_type == "platt":
            cal_prob = float(
                calibrator.predict_proba(np.array([[raw_prob]]))[0, 1]
            )
            return float(np.clip(cal_prob, PROB_CLIP_MIN, PROB_CLIP_MAX))

        else:
            logger.warning(f"Unknown calibrator type '{cal_type}' for regime {regime}")
            return float(np.clip(raw_prob, PROB_CLIP_MIN, PROB_CLIP_MAX))

    def calibrate_batch(self, raw_probs: np.ndarray, regimes: np.ndarray) -> np.ndarray:
        """Calibrate a batch of probabilities.

        Args:
            raw_probs: Raw probabilities, shape (n,)
            regimes: Regime labels, shape (n,)

        Returns:
            Calibrated probabilities, shape (n,)
        """
        raw_probs = np.asarray(raw_probs, dtype=np.float64).ravel()
        regimes = np.asarray(regimes, dtype=np.int32).ravel()

        cal_probs = np.empty_like(raw_probs)
        for i in range(len(raw_probs)):
            cal_probs[i] = self.calibrate(raw_probs[i], int(regimes[i]))

        return cal_probs

    def get_stats(self) -> dict:
        """Return calibration statistics per regime.

        Returns:
            Dict with per-regime stats including sample count, calibrator type,
            probability spread, and mean values.
        """
        if not self.is_fitted:
            return {"fitted": False}

        regime_names = {
            0: "TRENDING_UP",
            1: "TRENDING_DOWN",
            2: "RANGING",
            3: "VOLATILE",
        }

        stats = {"fitted": True, "regimes": {}}
        for regime_id in range(4):
            name = regime_names.get(regime_id, f"UNKNOWN({regime_id})")
            regime_stat = self._fit_stats.get(regime_id, {})
            stats["regimes"][name] = {
                "calibrator_type": self.calibrator_types.get(regime_id, "unknown"),
                "n_samples": regime_stat.get("n_samples", 0),
                "spread": regime_stat.get("cal_spread", 0.0),
                "raw_prob_mean": regime_stat.get("raw_prob_mean", 0.0),
                "cal_prob_mean": regime_stat.get("cal_prob_mean", regime_stat.get("raw_prob_mean", 0.0)),
            }

        # Overall spread (across all regimes)
        all_spreads = [
            self._fit_stats[r].get("cal_spread", 0.0)
            for r in range(4)
            if r in self._fit_stats
        ]
        stats["overall_max_spread"] = max(all_spreads) if all_spreads else 0.0
        stats["overall_min_spread"] = min(all_spreads) if all_spreads else 0.0

        return stats

    def save(self, path: str):
        """Save calibrator to disk.

        Args:
            path: File path (e.g., 'data/ensemble_model/calibrators_v2.pkl')
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        data = {
            "calibrators": self.calibrators,
            "calibrator_types": self.calibrator_types,
            "fit_stats": self._fit_stats,
            "is_fitted": self.is_fitted,
            "version": "v2",
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"CalibratorV2 saved to {path}")

    @classmethod
    def load(cls, path: str) -> "CalibratorV2":
        """Load calibrator from disk.

        Args:
            path: File path (e.g., 'data/ensemble_model/calibrators_v2.pkl')

        Returns:
            Loaded CalibratorV2 instance.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        cal = cls()
        cal.calibrators = data["calibrators"]
        cal.calibrator_types = data["calibrator_types"]
        cal._fit_stats = data.get("fit_stats", {})
        cal.is_fitted = data.get("is_fitted", True)

        logger.info(
            f"CalibratorV2 loaded from {path}: {dict(cal.calibrator_types)}"
        )
        return cal
