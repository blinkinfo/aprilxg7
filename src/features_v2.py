"""V5 Feature Engine — 76 microstructure features for 5-minute BTC prediction.

Phase 1 of the AprilXG V5 multi-model ensemble upgrade.
All features are normalized (percentage, z-score, or ratio). No raw prices.
Uses numpy vectorized operations throughout — no Python loops over rows.

Feature Groups:
  1. Price Action Microstructure (12)
  2. Multi-Scale Momentum (16)
  3. Volatility & Regime (10)
  4. Volume Profile (10)
  5. Order Flow Proxy (6)
  6. Trend Indicators (8)
  7. Time/Session Features (6)
  8. Higher Timeframe Context (8)
  Total: 76 features
"""
import logging

import numpy as np
import pandas as pd

from .config import ModelConfig

logger = logging.getLogger(__name__)


class FeatureEngineV2:
    """V5 feature engine — 70+ features optimized for 5-minute BTC prediction."""

    # Ordered list of all 76 feature names (deterministic)
    FEATURE_NAMES: list[str] = [
        # Group 1: Price Action Microstructure (12)
        "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "candle_direction",
        "consec_up", "consec_down", "hl_range_pct", "gap_pct",
        "close_position", "real_body_pct", "candle_pattern_3", "candle_pattern_5",
        # Group 2: Multi-Scale Momentum (16)
        "return_1", "return_2", "return_3", "return_5",
        "return_8", "return_13", "return_21", "return_34",
        "rsi_3", "rsi_5", "rsi_8", "rsi_14",
        "roc_3", "roc_5", "momentum_accel", "momentum_jerk",
        # Group 3: Volatility & Regime (10)
        "atr_5", "atr_14", "atr_ratio", "bb_width_10",
        "bb_position_10", "bb_squeeze", "std_5", "std_20",
        "volatility_ratio", "range_expansion",
        # Group 4: Volume Profile (10)
        "volume_sma_5", "volume_sma_20", "volume_change", "vwap_deviation",
        "obv_roc_5", "obv_roc_14", "volume_price_corr_10", "mfi_5",
        "mfi_14", "volume_direction",
        # Group 5: Order Flow Proxy (6)
        "buy_volume_ratio", "volume_delta", "delta_pct",
        "cvd_5", "cvd_divergence", "trade_intensity",
        # Group 6: Trend Indicators (8)
        "ema_9", "ema_21", "ema_cross", "ema_cross_signal",
        "macd_5_13", "macd_signal_5_13", "macd_hist_5_13", "adx_10",
        # Group 7: Time/Session Features (6)
        "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
        "is_asian_session", "is_us_session",
        # Group 8: Higher Timeframe Context (8)
        "htf_15m_rsi_5", "htf_15m_return_3", "htf_15m_atr_ratio", "htf_15m_adx",
        "htf_1h_rsi_5", "htf_1h_return_3", "htf_1h_atr_ratio", "htf_1h_trend",
    ]

    def __init__(self, config: ModelConfig):
        self.config = config

    def compute_features(
        self,
        df: pd.DataFrame,
        higher_tf_data: dict[str, pd.DataFrame] | None = None,
        trade_data: pd.DataFrame | None = None,
        ffill: bool = False,
    ) -> pd.DataFrame:
        """Compute all V5 features.

        Args:
            df: 5m OHLCV DataFrame (columns: timestamp, open, high, low, close, volume)
            higher_tf_data: Optional dict of higher TF DataFrames {"15m": df, "1h": df}
            trade_data: Optional recent trades DataFrame for volume delta features
            ffill: If True, forward-fill NaN instead of dropping

        Returns:
            DataFrame with feature columns only (no OHLCV, no timestamp)
        """
        if df.empty or len(df) < 50:
            logger.warning("Insufficient data for feature computation")
            return pd.DataFrame()

        feat = df.copy()
        close = feat["close"]
        open_ = feat["open"]
        high = feat["high"]
        low = feat["low"]
        volume = feat["volume"]

        # =====================================================================
        # Group 1: Price Action Microstructure (12 features)
        # =====================================================================
        hl_range = high - low
        feat["body_ratio"] = (close - open_).abs() / (hl_range + 1e-10)
        feat["upper_wick_ratio"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / (hl_range + 1e-10)
        feat["lower_wick_ratio"] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / (hl_range + 1e-10)
        feat["candle_direction"] = np.sign(close - open_)
        feat["hl_range_pct"] = hl_range / close * 100
        feat["gap_pct"] = (open_ - close.shift(1)) / (close.shift(1) + 1e-10) * 100
        feat["close_position"] = (close - low) / (hl_range + 1e-10)
        feat["real_body_pct"] = (close - open_).abs() / close * 100

        # Consecutive up/down counts
        direction = feat["candle_direction"]
        consec_up = pd.Series(0.0, index=feat.index)
        consec_down = pd.Series(0.0, index=feat.index)
        # Vectorized consecutive count using groupby on changes
        is_up = (direction > 0).astype(int)
        is_down = (direction < 0).astype(int)
        # Group consecutive runs
        up_groups = (is_up != is_up.shift(1)).cumsum()
        down_groups = (is_down != is_down.shift(1)).cumsum()
        consec_up = is_up.groupby(up_groups).cumsum().astype(float)
        consec_down = is_down.groupby(down_groups).cumsum().astype(float)
        feat["consec_up"] = consec_up
        feat["consec_down"] = consec_down

        # Candle pattern encodings (base-3: map -1->0, 0->1, 1->2)
        mapped_dir = direction.map({-1: 0, 0: 1, 1: 2}).fillna(1).astype(int)
        feat["candle_pattern_3"] = (
            mapped_dir.shift(2).fillna(1).astype(int) * 9
            + mapped_dir.shift(1).fillna(1).astype(int) * 3
            + mapped_dir
        ).astype(float)
        feat["candle_pattern_5"] = (
            mapped_dir.shift(4).fillna(1).astype(int) * 81
            + mapped_dir.shift(3).fillna(1).astype(int) * 27
            + mapped_dir.shift(2).fillna(1).astype(int) * 9
            + mapped_dir.shift(1).fillna(1).astype(int) * 3
            + mapped_dir
        ).astype(float)

        # =====================================================================
        # Group 2: Multi-Scale Momentum (16 features)
        # =====================================================================
        for period in [1, 2, 3, 5, 8, 13, 21, 34]:
            feat[f"return_{period}"] = close.pct_change(period)

        feat["rsi_3"] = self._rsi(close, 3)
        feat["rsi_5"] = self._rsi(close, 5)
        feat["rsi_8"] = self._rsi(close, 8)
        feat["rsi_14"] = self._rsi(close, 14)

        feat["roc_3"] = (close - close.shift(3)) / (close.shift(3) + 1e-10) * 100
        feat["roc_5"] = (close - close.shift(5)) / (close.shift(5) + 1e-10) * 100

        feat["momentum_accel"] = feat["return_1"] - feat["return_1"].shift(1)
        feat["momentum_jerk"] = feat["momentum_accel"] - feat["momentum_accel"].shift(1)

        # =====================================================================
        # Group 3: Volatility & Regime (10 features)
        # =====================================================================
        atr_5_raw = self._atr(high, low, close, 5)
        atr_14_raw = self._atr(high, low, close, 14)
        feat["atr_5"] = atr_5_raw / close * 100
        feat["atr_14"] = atr_14_raw / close * 100
        feat["atr_ratio"] = feat["atr_5"] / (feat["atr_14"] + 1e-10)

        bb_mid_10 = close.rolling(10).mean()
        bb_std_10 = close.rolling(10).std()
        bb_upper_10 = bb_mid_10 + 2 * bb_std_10
        bb_lower_10 = bb_mid_10 - 2 * bb_std_10
        feat["bb_width_10"] = (bb_upper_10 - bb_lower_10) / (bb_mid_10 + 1e-10) * 100
        feat["bb_position_10"] = (close - bb_lower_10) / (bb_upper_10 - bb_lower_10 + 1e-10)

        bb_width_quantile_20 = feat["bb_width_10"].rolling(50).quantile(0.2)
        feat["bb_squeeze"] = (feat["bb_width_10"] < bb_width_quantile_20).astype(float)

        pct_change = close.pct_change()
        feat["std_5"] = pct_change.rolling(5).std() * 100
        feat["std_20"] = pct_change.rolling(20).std() * 100
        feat["volatility_ratio"] = feat["std_5"] / (feat["std_20"] + 1e-10)

        hl_range_pct = feat["hl_range_pct"]
        feat["range_expansion"] = hl_range_pct / (hl_range_pct.rolling(20).mean() + 1e-10)

        # =====================================================================
        # Group 4: Volume Profile (10 features)
        # =====================================================================
        feat["volume_sma_5"] = volume / (volume.rolling(5).mean() + 1e-10)
        feat["volume_sma_20"] = volume / (volume.rolling(20).mean() + 1e-10)

        prev_volume = volume.shift(1)
        feat["volume_change"] = (volume - prev_volume) / (prev_volume + 1e-10)
        feat["volume_change"] = feat["volume_change"].replace([np.inf, -np.inf], 0.0)

        # VWAP: rolling 60-candle (5 hour) window as session proxy
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).rolling(60).sum() / (volume.rolling(60).sum() + 1e-10)
        feat["vwap_deviation"] = (close - vwap) / (close + 1e-10) * 100

        # OBV (On-Balance Volume)
        obv_direction = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
        obv = (volume * obv_direction).cumsum()
        obv_series = pd.Series(obv, index=feat.index)
        feat["obv_roc_5"] = obv_series.pct_change(5)
        feat["obv_roc_14"] = obv_series.pct_change(14)
        # Guard against inf from zero OBV
        feat["obv_roc_5"] = feat["obv_roc_5"].replace([np.inf, -np.inf], 0.0)
        feat["obv_roc_14"] = feat["obv_roc_14"].replace([np.inf, -np.inf], 0.0)

        feat["volume_price_corr_10"] = close.pct_change().rolling(10).corr(volume.pct_change())

        feat["mfi_5"] = self._mfi(high, low, close, volume, 5)
        feat["mfi_14"] = self._mfi(high, low, close, volume, 14)

        feat["volume_direction"] = volume * feat["candle_direction"]

        # =====================================================================
        # Group 5: Order Flow Proxy (6 features)
        # =====================================================================
        if trade_data is not None and not trade_data.empty:
            # Use actual trade data for buy/sell split
            buy_vol, sell_vol, trade_count = self._aggregate_trade_data(
                trade_data, feat
            )
        else:
            # Estimate buy/sell split from OHLCV
            buy_frac = np.where(
                (high.values != low.values),
                (close.values - low.values) / (high.values - low.values),
                0.5,
            )
            buy_vol = pd.Series(volume.values * buy_frac, index=feat.index)
            sell_vol = pd.Series(volume.values * (1 - buy_frac), index=feat.index)
            trade_count = pd.Series(np.nan, index=feat.index)

        total_vol = buy_vol + sell_vol
        feat["buy_volume_ratio"] = buy_vol / (total_vol + 1e-10)
        feat["volume_delta"] = buy_vol - sell_vol
        feat["delta_pct"] = feat["volume_delta"] / (total_vol + 1e-10) * 100

        feat["cvd_5"] = feat["volume_delta"].rolling(5).sum()
        cvd_sign = np.sign(feat["cvd_5"])
        return_sign = np.sign(feat["return_5"])
        feat["cvd_divergence"] = (cvd_sign != return_sign).astype(float)

        # Trade intensity: trades per minute (5-min candles = 5 minutes)
        if trade_count.notna().any():
            feat["trade_intensity"] = trade_count / 5.0
        else:
            feat["trade_intensity"] = 0.0

        # =====================================================================
        # Group 6: Trend Indicators (8 features)
        # =====================================================================
        ema_9 = close.ewm(span=9, adjust=False).mean()
        ema_21 = close.ewm(span=21, adjust=False).mean()
        # Store as normalized distance from close (no raw prices)
        feat["ema_9"] = (close - ema_9) / (close + 1e-10) * 100
        feat["ema_21"] = (close - ema_21) / (close + 1e-10) * 100
        feat["ema_cross"] = (ema_9 - ema_21) / (close + 1e-10) * 100
        ema_cross_prev = ((ema_9.shift(1) - ema_21.shift(1)) / (close.shift(1) + 1e-10) * 100)
        feat["ema_cross_signal"] = (np.sign(feat["ema_cross"]) != np.sign(ema_cross_prev)).astype(float)

        # MACD optimized for 5m: fast=5, slow=13, signal=4
        ema_fast = close.ewm(span=5, adjust=False).mean()
        ema_slow = close.ewm(span=13, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal_line = macd_line.ewm(span=4, adjust=False).mean()
        feat["macd_5_13"] = macd_line / (close + 1e-10) * 100
        feat["macd_signal_5_13"] = macd_signal_line / (close + 1e-10) * 100
        feat["macd_hist_5_13"] = (macd_line - macd_signal_line) / (close + 1e-10) * 100

        feat["adx_10"] = self._adx(high, low, close, 10) / 100.0

        # =====================================================================
        # Group 7: Time/Session Features (6 features)
        # =====================================================================
        if "timestamp" in feat.columns:
            ts = pd.to_datetime(feat["timestamp"])
            hour = ts.dt.hour
            dow = ts.dt.dayofweek
        else:
            # Fallback: use index if it's a DatetimeIndex
            hour = pd.Series(feat.index.hour if hasattr(feat.index, 'hour') else 0, index=feat.index)
            dow = pd.Series(feat.index.dayofweek if hasattr(feat.index, 'dayofweek') else 0, index=feat.index)

        feat["hour_sin"] = np.sin(2 * np.pi * hour.values / 24)
        feat["hour_cos"] = np.cos(2 * np.pi * hour.values / 24)
        feat["day_of_week_sin"] = np.sin(2 * np.pi * dow.values / 7)
        feat["day_of_week_cos"] = np.cos(2 * np.pi * dow.values / 7)
        feat["is_asian_session"] = ((hour.values >= 0) & (hour.values < 8)).astype(float)
        feat["is_us_session"] = ((hour.values >= 13) & (hour.values < 21)).astype(float)

        # =====================================================================
        # Group 8: Higher Timeframe Context (8 features)
        # =====================================================================
        self._compute_htf_features(feat, higher_tf_data)

        # =====================================================================
        # Select feature columns only and handle NaN
        # =====================================================================
        result = feat[self.FEATURE_NAMES].copy()

        # Replace any remaining inf values
        result = result.replace([np.inf, -np.inf], np.nan)

        if ffill:
            result = result.ffill().bfill()
        else:
            result = result.dropna()

        if result.empty:
            logger.warning("All rows dropped after NaN handling")

        return result

    def get_feature_names(self) -> list[str]:
        """Return ordered list of all feature names."""
        return list(self.FEATURE_NAMES)

    # =========================================================================
    # Static helper methods — self-contained TA implementations (no ta-lib)
    # =========================================================================

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        """Compute RSI (Relative Strength Index), normalized 0-100."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Compute Average True Range."""
        high_low = high - low
        high_close = (high - close.shift(1)).abs()
        low_close = (low - close.shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(period).mean()

    @staticmethod
    def _mfi(high: pd.Series, low: pd.Series, close: pd.Series,
             volume: pd.Series, period: int) -> pd.Series:
        """Compute Money Flow Index, normalized 0-100."""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        pos_flow = money_flow.where(typical_price > typical_price.shift(1), 0.0)
        neg_flow = money_flow.where(typical_price < typical_price.shift(1), 0.0)
        pos_mf = pos_flow.rolling(period).sum()
        neg_mf = neg_flow.rolling(period).sum()
        mfi_ratio = pos_mf / (neg_mf + 1e-10)
        return 100 - (100 / (1 + mfi_ratio))

    @staticmethod
    def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Compute Average Directional Index (0-100 scale)."""
        # True Range
        high_low = high - low
        high_close = (high - close.shift(1)).abs()
        low_close = (low - close.shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        # Smoothed with rolling mean
        atr = true_range.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-10))

        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
        adx = dx.rolling(period).mean()
        return adx

    @staticmethod
    def _aggregate_trade_data(
        trade_data: pd.DataFrame,
        feat: pd.DataFrame,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Aggregate raw trade data into per-candle buy/sell volumes.

        Args:
            trade_data: DataFrame with columns: time, qty, isBuyerMaker
            feat: Main feature DataFrame (used for index alignment)

        Returns:
            Tuple of (buy_volume, sell_volume, trade_count) Series aligned to feat.index
        """
        # Default to estimation if trade_data lacks required columns
        required_cols = {"time", "qty", "isBuyerMaker"}
        if not required_cols.issubset(trade_data.columns):
            buy_frac = np.where(
                (feat["high"].values != feat["low"].values),
                (feat["close"].values - feat["low"].values) / (feat["high"].values - feat["low"].values),
                0.5,
            )
            buy_vol = pd.Series(feat["volume"].values * buy_frac, index=feat.index)
            sell_vol = pd.Series(feat["volume"].values * (1 - buy_frac), index=feat.index)
            return buy_vol, sell_vol, pd.Series(np.nan, index=feat.index)

        td = trade_data.copy()
        td["time"] = pd.to_datetime(td["time"], utc=True)

        # isBuyerMaker=True means the buyer is the maker, so the trade
        # is a sell (taker sold). isBuyerMaker=False means taker bought.
        td["buy_qty"] = td["qty"].where(~td["isBuyerMaker"], 0.0)
        td["sell_qty"] = td["qty"].where(td["isBuyerMaker"], 0.0)

        # Resample to 5-minute buckets
        td = td.set_index("time")
        resampled = td.resample("5min").agg({
            "buy_qty": "sum",
            "sell_qty": "sum",
            "qty": "count",  # trade count
        }).rename(columns={"qty": "trade_count"})

        # Align to feat index
        if "timestamp" in feat.columns:
            align_index = pd.to_datetime(feat["timestamp"])
        else:
            align_index = feat.index

        buy_vol = resampled["buy_qty"].reindex(align_index, method="ffill").fillna(0.0)
        sell_vol = resampled["sell_qty"].reindex(align_index, method="ffill").fillna(0.0)
        trade_count = resampled["trade_count"].reindex(align_index, method="ffill").fillna(0.0)

        buy_vol.index = feat.index
        sell_vol.index = feat.index
        trade_count.index = feat.index

        return buy_vol, sell_vol, trade_count

    def _compute_htf_features(
        self,
        feat: pd.DataFrame,
        higher_tf_data: dict[str, pd.DataFrame] | None,
    ) -> None:
        """Compute higher timeframe features and add them to feat in-place.

        Expects higher_tf_data keys: "15m" and/or "1h".
        Reindexes HTF features to 5m index using forward-fill.
        """
        htf_specs = {
            "15m": {
                "rsi_col": "htf_15m_rsi_5",
                "return_col": "htf_15m_return_3",
                "atr_ratio_col": "htf_15m_atr_ratio",
                "extra_col": "htf_15m_adx",
                "extra_type": "adx",
            },
            "1h": {
                "rsi_col": "htf_1h_rsi_5",
                "return_col": "htf_1h_return_3",
                "atr_ratio_col": "htf_1h_atr_ratio",
                "extra_col": "htf_1h_trend",
                "extra_type": "trend",
            },
        }

        for tf_label, spec in htf_specs.items():
            if higher_tf_data and tf_label in higher_tf_data:
                tf_df = higher_tf_data[tf_label]
                if tf_df.empty or len(tf_df) < 20:
                    self._fill_htf_nan(feat, spec)
                    continue

                htf_close = tf_df["close"]
                htf_high = tf_df["high"]
                htf_low = tf_df["low"]

                htf_feats = pd.DataFrame(index=tf_df.index)
                htf_feats[spec["rsi_col"]] = self._rsi(htf_close, 5)
                htf_feats[spec["return_col"]] = htf_close.pct_change(3)

                atr_5 = self._atr(htf_high, htf_low, htf_close, 5)
                atr_14 = self._atr(htf_high, htf_low, htf_close, 14)
                htf_feats[spec["atr_ratio_col"]] = atr_5 / (atr_14 + 1e-10)

                if spec["extra_type"] == "adx":
                    htf_feats[spec["extra_col"]] = self._adx(htf_high, htf_low, htf_close, 10) / 100.0
                else:  # trend
                    ema9 = htf_close.ewm(span=9, adjust=False).mean()
                    ema21 = htf_close.ewm(span=21, adjust=False).mean()
                    htf_feats[spec["extra_col"]] = np.sign(ema9 - ema21)

                # Reindex to 5m index using ffill
                try:
                    htf_reindexed = htf_feats.reindex(feat.index, method="ffill")
                except (ValueError, TypeError):
                    try:
                        combined = htf_feats.reindex(
                            htf_feats.index.union(feat.index)
                        ).ffill()
                        htf_reindexed = combined.reindex(feat.index)
                    except Exception as e:
                        logger.warning(f"HTF reindex failed for {tf_label}: {e}")
                        self._fill_htf_nan(feat, spec)
                        continue

                for col in htf_reindexed.columns:
                    feat[col] = htf_reindexed[col].values[:len(feat)]
            else:
                # No HTF data — fill with NaN (will be handled by ffill/dropna)
                self._fill_htf_nan(feat, spec)

    @staticmethod
    def _fill_htf_nan(feat: pd.DataFrame, spec: dict) -> None:
        """Fill HTF feature columns with neutral defaults when data is unavailable.

        Uses neutral values instead of NaN so that ffill/bfill can handle the
        all-missing case (all-NaN columns cannot be forward-filled).
        Neutral values: RSI=50 (midpoint), returns=0, ATR ratio=1, ADX/trend=0.
        """
        defaults = {
            "rsi_col": 50.0,       # RSI midpoint (no directional bias)
            "return_col": 0.0,     # No return
            "atr_ratio_col": 1.0,  # Neutral ATR ratio
            "extra_col": 0.0,      # No trend / no ADX signal
        }
        for key in ["rsi_col", "return_col", "atr_ratio_col", "extra_col"]:
            feat[spec[key]] = defaults[key]
