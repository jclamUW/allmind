from __future__ import annotations

import config as config
import numpy as np
import pandas as pd

from src.modules.indicator import Indicator
from typing import Dict, List


_NAME: str = "sr"
_DESCRIPTIONS: List[str] = ["roll_max",
                            "roll_min",
                            "dist_to_resistance",
                            "dist_to_support",
                            "range_frac",
                            "range",
                            "touch_count_resistance",
                            "touch_count_support",
                            "near_resistance",
                            "near_support",
                            "bounce_ratio_support",
                            "bounce_ratio_resistance"]
_TOUCH_TOLERANCE: float = 0.05


class Sr(Indicator):

    def __init__(self):
        super().__init__(name=_NAME, descriptions=_DESCRIPTIONS)

    def build_features_1_tf(self, sdf: pd.DataFrame) -> pd.DataFrame:
        """
        Create sdf with the same index, but containing only SR feature columns.
        """
        if sdf is None or sdf.empty:
            raise RuntimeError("Valid sdf must be provided.")

        if "close" not in sdf.columns:
            raise ValueError("sdf must contain 'close' column.")
        close = pd.to_numeric(sdf["close"], errors="coerce").astype(float)

        idx = sdf.index
        if not isinstance(idx, pd.DatetimeIndex):
            raise ValueError("sdf.index must be a DatetimeIndex.")

        high = pd.to_numeric(sdf["high"], errors="coerce").astype(float) if "high" in sdf.columns else close.copy()
        low = pd.to_numeric(sdf["low"], errors="coerce").astype(float) if "low" in sdf.columns else close.copy()

        cols: Dict[str, pd.Series] = {}
        periods: List[int] = sorted({int(x) for x in config.PERIODS})
        for p in periods:
            p = int(p)
            prefix = f"{self.name}__p={p}"

            roll_max = high.rolling(window=p, min_periods=1).max()      # rolling extrema on the lookback window
            roll_min = low.rolling(window=p, min_periods=1).min()
            roll_range = roll_max - roll_min
            dist_res = roll_max - close                                 # distances from current close to the boundary
            dist_sup = close - roll_min

            # compute boolean array where price is within tolerance of max/min
            tolerance = (roll_range.abs() * _TOUCH_TOLERANCE).fillna(0.0)
            near_tolerance = (roll_range.abs() * _TOUCH_TOLERANCE * 2).fillna(0.0)
            is_touch_res = (dist_res.abs() <= tolerance) & (tolerance > 0)
            is_touch_sup = (dist_sup.abs() <= tolerance) & (tolerance > 0)
            touch_count_res = is_touch_res.rolling(window=p, min_periods=1).sum().fillna(0).astype("Int64")
            touch_count_sup = is_touch_sup.rolling(window=p, min_periods=1).sum().fillna(0).astype("Int64")

            # momentum at the boundary: how often price reversed after touching (simple proxy)
            # compute sign of price change after touches in the next bar (1 if bounced up after touching support, -1 if dropped after touching resistance)
            # compute a simple "bounce_ratio" as mean of next bar sign conditioned on touch within window
            # keep as numeric, possibly NaN if insufficient
            # mean close_diff after recent touches (rolling apply could be heavy; use rolling sum of positive events)
            close_diff = close.shift(-1) - close
            bounced_after_res = ((close_diff < 0) & is_touch_res).rolling(window=p, min_periods=1).sum()
            bounced_after_sup = ((close_diff > 0) & is_touch_sup).rolling(window=p, min_periods=1).sum()
            touched_res_total = is_touch_res.rolling(window=p, min_periods=1).sum().fillna(0)
            touched_sup_total = is_touch_sup.rolling(window=p, min_periods=1).sum().fillna(0)

            cols[f"{prefix}__roll_max"] = roll_max
            cols[f"{prefix}__roll_min"] = roll_min
            cols[f"{prefix}__range"] = roll_range
            cols[f"{prefix}__dist_to_resistance"] = dist_res
            cols[f"{prefix}__dist_to_support"] = dist_sup
            cols[f"{prefix}__range_frac"] = _safe_div(a=roll_range, b=roll_min).fillna(0.0)
            cols[f"{prefix}__touch_count_resistance"] = touch_count_res
            cols[f"{prefix}__touch_count_support"] = touch_count_sup
            cols[f"{prefix}__near_resistance"] = ((dist_res >= 0) & (dist_res <= near_tolerance)).astype("Int64")
            cols[f"{prefix}__near_support"] = ((dist_sup >= 0) & (dist_sup <= near_tolerance)).astype("Int64")
            cols[f"{prefix}__bounce_ratio_resistance"] = _safe_div(a=bounced_after_res, b=touched_res_total).fillna(0.0)
            cols[f"{prefix}__bounce_ratio_support"] = _safe_div(a=bounced_after_sup, b=touched_sup_total).fillna(0.0)

        # replace inf with NaNs
        result = pd.DataFrame(cols, index=idx)
        result = result.replace([np.inf, -np.inf], np.nan)
        return result


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = a / b.replace(0, np.nan)
    return result
