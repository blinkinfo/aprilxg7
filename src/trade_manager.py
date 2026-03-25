"""Tiered trade frequency system with session risk management.

Phase 3 of the AprilXG V5 multi-model ensemble upgrade.

Manages trade frequency and session risk:
- Tiered confidence system ensures 70+ trades/day while maintaining quality.
- Session risk monitor temporarily tightens filters during losing streaks.

Tiers (evaluated in order — first match wins):
    Tier 1 (High):   cal_prob >= 0.57  -> Trade (highest conviction)
    Tier 2 (Medium): cal_prob >= 0.54  -> Trade (good conviction)
    Tier 3 (Base):   cal_prob >= 0.52 AND model_agreement >= 2  -> Trade

Session Risk:
    Tracks rolling accuracy over last 20 decided trades.
    If rolling accuracy drops below 48%, enter CAUTIOUS mode:
    - Tier 3 disabled (only Tier 1 + 2 trade)
    - Stays cautious for 30 minutes, then reverts
    If rolling accuracy drops below 42%, enter DEFENSIVE mode:
    - Only Tier 1 trades (cal_prob >= 0.57)
    - Stays defensive for 60 minutes, then reverts
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from .config import ModelConfig

logger = logging.getLogger(__name__)

# Default tier thresholds
DEFAULT_TIER1_THRESHOLD = 0.57
DEFAULT_TIER2_THRESHOLD = 0.54
DEFAULT_TIER3_THRESHOLD = 0.52
DEFAULT_TIER3_MIN_AGREEMENT = 2

# Default risk thresholds
DEFAULT_CAUTIOUS_ACCURACY = 0.48
DEFAULT_DEFENSIVE_ACCURACY = 0.42
DEFAULT_CAUTIOUS_DURATION_MIN = 30
DEFAULT_DEFENSIVE_DURATION_MIN = 60
DEFAULT_ROLLING_WINDOW = 20


class TradeManager:
    """Manages trade frequency and session risk.

    Tiered confidence system ensures 70+ trades/day while maintaining quality.
    Session risk monitor temporarily tightens filters during losing streaks.

    Tiers (evaluated in order — first match wins):
        Tier 1 (High):   cal_prob >= 0.57  -> Trade (highest conviction)
        Tier 2 (Medium): cal_prob >= 0.54  -> Trade (good conviction)
        Tier 3 (Base):   cal_prob >= 0.52 AND model_agreement >= 2  -> Trade

    Session Risk:
        Tracks rolling accuracy over last 20 decided trades.
        If rolling accuracy drops below 48%, enter CAUTIOUS mode:
        - Tier 3 disabled (only Tier 1 + 2 trade)
        - Stays cautious for 30 minutes, then reverts
        If rolling accuracy drops below 42%, enter DEFENSIVE mode:
        - Only Tier 1 trades (cal_prob >= 0.57)
        - Stays defensive for 60 minutes, then reverts
    """

    # Risk modes
    NORMAL = "NORMAL"
    CAUTIOUS = "CAUTIOUS"
    DEFENSIVE = "DEFENSIVE"

    def __init__(self, config: ModelConfig):
        self.config = config
        self.risk_mode: str = self.NORMAL
        self._mode_until: Optional[datetime] = None  # datetime when mode reverts to NORMAL
        self._recent_results: list[bool] = []  # list of bools, True=WIN
        self._rolling_window: int = DEFAULT_ROLLING_WINDOW

        # Tier thresholds (can be overridden via EnsembleConfig in Phase 4)
        self._tier1_threshold: float = DEFAULT_TIER1_THRESHOLD
        self._tier2_threshold: float = DEFAULT_TIER2_THRESHOLD
        self._tier3_threshold: float = DEFAULT_TIER3_THRESHOLD
        self._tier3_min_agreement: int = DEFAULT_TIER3_MIN_AGREEMENT

        # Risk thresholds
        self._cautious_accuracy: float = DEFAULT_CAUTIOUS_ACCURACY
        self._defensive_accuracy: float = DEFAULT_DEFENSIVE_ACCURACY
        self._cautious_duration: timedelta = timedelta(minutes=DEFAULT_CAUTIOUS_DURATION_MIN)
        self._defensive_duration: timedelta = timedelta(minutes=DEFAULT_DEFENSIVE_DURATION_MIN)

        # Trade counters for stats
        self._total_trades: int = 0
        self._total_skips: int = 0
        self._tier_counts: dict[int, int] = {1: 0, 2: 0, 3: 0}
        self._mode_transitions: list[dict] = []

    def configure(
        self,
        tier1_threshold: float = DEFAULT_TIER1_THRESHOLD,
        tier2_threshold: float = DEFAULT_TIER2_THRESHOLD,
        tier3_threshold: float = DEFAULT_TIER3_THRESHOLD,
        tier3_min_agreement: int = DEFAULT_TIER3_MIN_AGREEMENT,
        cautious_accuracy: float = DEFAULT_CAUTIOUS_ACCURACY,
        defensive_accuracy: float = DEFAULT_DEFENSIVE_ACCURACY,
        cautious_duration_minutes: int = DEFAULT_CAUTIOUS_DURATION_MIN,
        defensive_duration_minutes: int = DEFAULT_DEFENSIVE_DURATION_MIN,
        rolling_window: int = DEFAULT_ROLLING_WINDOW,
    ):
        """Configure trade manager thresholds.

        Called during Phase 4 integration to apply EnsembleConfig values.
        """
        self._tier1_threshold = tier1_threshold
        self._tier2_threshold = tier2_threshold
        self._tier3_threshold = tier3_threshold
        self._tier3_min_agreement = tier3_min_agreement
        self._cautious_accuracy = cautious_accuracy
        self._defensive_accuracy = defensive_accuracy
        self._cautious_duration = timedelta(minutes=cautious_duration_minutes)
        self._defensive_duration = timedelta(minutes=defensive_duration_minutes)
        self._rolling_window = rolling_window

        logger.info(
            f"TradeManager configured: tiers=[{tier1_threshold}, {tier2_threshold}, "
            f"{tier3_threshold} (agree>={tier3_min_agreement})], "
            f"risk=[cautious<{cautious_accuracy}, defensive<{defensive_accuracy}], "
            f"window={rolling_window}"
        )

    def should_trade(self, prediction: dict) -> dict:
        """Decide whether to trade based on prediction and current risk mode.

        Args:
            prediction: Dict from EnsembleModel.predict() containing:
                - confidence: float (max of cal_prob_up, 1-cal_prob_up)
                - cal_prob_up: float
                - model_agreement: int
                - ev: float
                - regime: int

        Returns:
            {
                "trade": bool,
                "tier": int or None,     # 1, 2, 3, or None if no trade
                "reason": str,           # why trade/skip
                "risk_mode": str,        # NORMAL, CAUTIOUS, DEFENSIVE
                "rolling_accuracy": float or None,
            }
        """
        # Check if risk mode should revert to NORMAL
        self._check_risk_mode()

        cal_prob_up = prediction.get("cal_prob_up", 0.5)
        confidence = prediction.get("confidence", 0.5)
        model_agreement = prediction.get("model_agreement", 0)
        ev = prediction.get("ev", 0.0)

        # The directional probability (how far from 0.5)
        # confidence = max(cal_prob_up, 1 - cal_prob_up), already computed in predict()

        rolling_acc = self._get_rolling_accuracy()

        # Evaluate tiers based on current risk mode
        tier = None
        trade = False
        reason = ""

        # Tier 1: High conviction — always active in all modes
        if confidence >= self._tier1_threshold:
            tier = 1
            trade = True
            reason = f"Tier 1 (High): confidence {confidence:.3f} >= {self._tier1_threshold}"

        # Tier 2: Medium conviction — active in NORMAL and CAUTIOUS
        elif confidence >= self._tier2_threshold:
            if self.risk_mode == self.DEFENSIVE:
                tier = None
                trade = False
                reason = (
                    f"Tier 2 blocked: DEFENSIVE mode (confidence {confidence:.3f} >= "
                    f"{self._tier2_threshold} but only Tier 1 allowed)"
                )
            else:
                tier = 2
                trade = True
                reason = f"Tier 2 (Medium): confidence {confidence:.3f} >= {self._tier2_threshold}"

        # Tier 3: Base — active only in NORMAL mode, requires model agreement
        elif confidence >= self._tier3_threshold and model_agreement >= self._tier3_min_agreement:
            if self.risk_mode != self.NORMAL:
                tier = None
                trade = False
                mode_reason = (
                    "only Tier 1 allowed" if self.risk_mode == self.DEFENSIVE
                    else "only Tier 1+2 allowed"
                )
                reason = (
                    f"Tier 3 blocked: {self.risk_mode} mode ({mode_reason}), "
                    f"confidence={confidence:.3f}, agreement={model_agreement}"
                )
            else:
                tier = 3
                trade = True
                reason = (
                    f"Tier 3 (Base): confidence {confidence:.3f} >= {self._tier3_threshold} "
                    f"AND agreement {model_agreement} >= {self._tier3_min_agreement}"
                )

        else:
            # Below all thresholds
            tier = None
            trade = False
            if confidence < self._tier3_threshold:
                reason = f"Skip: confidence {confidence:.3f} < {self._tier3_threshold} (below all tiers)"
            else:
                reason = (
                    f"Skip: confidence {confidence:.3f} >= {self._tier3_threshold} but "
                    f"agreement {model_agreement} < {self._tier3_min_agreement}"
                )

        # Update counters
        if trade:
            self._total_trades += 1
            if tier is not None:
                self._tier_counts[tier] = self._tier_counts.get(tier, 0) + 1
        else:
            self._total_skips += 1

        result = {
            "trade": trade,
            "tier": tier,
            "reason": reason,
            "risk_mode": self.risk_mode,
            "rolling_accuracy": rolling_acc,
        }

        if trade:
            logger.info(
                f"TRADE: {reason} | mode={self.risk_mode} | "
                f"rolling_acc={f'{rolling_acc:.1%}' if rolling_acc is not None else 'N/A'}"
            )
        else:
            logger.debug(
                f"SKIP: {reason} | mode={self.risk_mode} | "
                f"rolling_acc={f'{rolling_acc:.1%}' if rolling_acc is not None else 'N/A'}"
            )

        return result

    def record_result(self, won: bool):
        """Record a trade result for rolling accuracy tracking.

        Args:
            won: True if trade was a win, False if loss.
        """
        self._recent_results.append(won)

        # Keep only the rolling window
        if len(self._recent_results) > self._rolling_window * 2:
            # Keep some extra history but trim to prevent unbounded growth
            self._recent_results = self._recent_results[-self._rolling_window * 2:]

        # Check if risk mode needs to change
        rolling_acc = self._get_rolling_accuracy()
        if rolling_acc is not None:
            old_mode = self.risk_mode
            now = datetime.now(timezone.utc)

            if rolling_acc < self._defensive_accuracy:
                # DEFENSIVE: very bad streak
                if self.risk_mode != self.DEFENSIVE:
                    self.risk_mode = self.DEFENSIVE
                    self._mode_until = now + self._defensive_duration
                    self._mode_transitions.append({
                        "from": old_mode,
                        "to": self.DEFENSIVE,
                        "rolling_accuracy": rolling_acc,
                        "timestamp": now.isoformat(),
                    })
                    logger.warning(
                        f"RISK MODE: {old_mode} -> DEFENSIVE | "
                        f"rolling_acc={rolling_acc:.1%} < {self._defensive_accuracy:.1%} | "
                        f"reverts at {self._mode_until.isoformat()}"
                    )

            elif rolling_acc < self._cautious_accuracy:
                # CAUTIOUS: moderate losing streak
                if self.risk_mode == self.NORMAL:
                    self.risk_mode = self.CAUTIOUS
                    self._mode_until = now + self._cautious_duration
                    self._mode_transitions.append({
                        "from": old_mode,
                        "to": self.CAUTIOUS,
                        "rolling_accuracy": rolling_acc,
                        "timestamp": now.isoformat(),
                    })
                    logger.warning(
                        f"RISK MODE: {old_mode} -> CAUTIOUS | "
                        f"rolling_acc={rolling_acc:.1%} < {self._cautious_accuracy:.1%} | "
                        f"reverts at {self._mode_until.isoformat()}"
                    )

        result_str = "WIN" if won else "LOSS"
        logger.info(
            f"Trade result: {result_str} | "
            f"rolling_acc={f'{rolling_acc:.1%}' if rolling_acc is not None else 'N/A'} | "
            f"mode={self.risk_mode} | "
            f"results_count={len(self._recent_results)}"
        )

    def _get_rolling_accuracy(self) -> Optional[float]:
        """Calculate rolling accuracy over the last N decided trades.

        Returns:
            Float accuracy or None if not enough data.
        """
        if len(self._recent_results) < self._rolling_window:
            return None

        recent = self._recent_results[-self._rolling_window:]
        return sum(recent) / len(recent)

    def _check_risk_mode(self):
        """Update risk mode based on time expiry.

        If we're in CAUTIOUS or DEFENSIVE mode and the duration has elapsed,
        revert to NORMAL.
        """
        if self.risk_mode == self.NORMAL:
            return

        if self._mode_until is not None:
            now = datetime.now(timezone.utc)
            if now >= self._mode_until:
                old_mode = self.risk_mode
                self.risk_mode = self.NORMAL
                self._mode_until = None
                self._mode_transitions.append({
                    "from": old_mode,
                    "to": self.NORMAL,
                    "rolling_accuracy": self._get_rolling_accuracy(),
                    "timestamp": now.isoformat(),
                    "reason": "duration_expired",
                })
                logger.info(
                    f"RISK MODE: {old_mode} -> NORMAL (duration expired)"
                )

    def get_stats(self) -> dict:
        """Return trade manager statistics.

        Returns:
            Dict with trade counts, tier distribution, risk mode info,
            and rolling accuracy.
        """
        rolling_acc = self._get_rolling_accuracy()

        return {
            "risk_mode": self.risk_mode,
            "mode_until": self._mode_until.isoformat() if self._mode_until else None,
            "rolling_accuracy": rolling_acc,
            "rolling_window": self._rolling_window,
            "results_count": len(self._recent_results),
            "total_trades": self._total_trades,
            "total_skips": self._total_skips,
            "trade_rate": (
                self._total_trades / (self._total_trades + self._total_skips)
                if (self._total_trades + self._total_skips) > 0
                else 0.0
            ),
            "tier_distribution": dict(self._tier_counts),
            "mode_transitions": len(self._mode_transitions),
            "recent_transitions": self._mode_transitions[-5:] if self._mode_transitions else [],
            "thresholds": {
                "tier1": self._tier1_threshold,
                "tier2": self._tier2_threshold,
                "tier3": self._tier3_threshold,
                "tier3_min_agreement": self._tier3_min_agreement,
                "cautious_accuracy": self._cautious_accuracy,
                "defensive_accuracy": self._defensive_accuracy,
            },
        }

    def reset(self):
        """Reset all state (for testing or session restart)."""
        self.risk_mode = self.NORMAL
        self._mode_until = None
        self._recent_results = []
        self._total_trades = 0
        self._total_skips = 0
        self._tier_counts = {1: 0, 2: 0, 3: 0}
        self._mode_transitions = []
        logger.info("TradeManager reset")
