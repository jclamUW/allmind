from __future__ import annotations

import config as config
import numpy as np
import pandas as pd

from src.modules.indicator import Indicator
from typing import Dict, List


_NAME: str = "vwap"
_DESCRIPTIONS: List[str] = ["price_minus", "price_div", "streak_above", "slope"]


class Vwap(Indicator):

    def __init__(self):
        super().__init__(name=_NAME, descriptions=_DESCRIPTIONS)

    def build_features_1_tf(self, sdf: pd.DataFrame) -> pd.DataFrame:
        """
        Create sdf with the same index, but containing only VWAP feature columns.
        """
        if sdf is None or sdf.empty:
            raise RuntimeError("Valid sdf must be provided.")

        if "close" not in sdf.columns:
            raise ValueError("sdf must contain 'close' column.")
        close = pd.to_numeric(sdf["close"], errors="coerce").astype(float)

        idx = sdf.index
        if not isinstance(idx, pd.DatetimeIndex):
            raise ValueError("sdf.index must be a DatetimeIndex.")

        # volume: build with the same index to avoid misaligned Series objects
        if "volume" in sdf.columns:
            vol = pd.to_numeric(sdf["volume"], errors="coerce").astype(float).fillna(0.0)
        else:
            vol = pd.Series(0.0, index=idx, dtype=float)

        cols: Dict[str, pd.Series] = {}
        periods: List[int] = sorted({int(x) for x in config.PERIODS})
        for p in periods:
            p = int(p)
            prefix = f"{self.name}__p={p}"

            # rolling numerator and denominator for VWAP: sum(tp * vol) / sum(vol)
            tp = _get_typical_price(sdf)
            num = (tp * vol).rolling(window=p, min_periods=1).sum()
            den = vol.rolling(window=p, min_periods=1).sum()
            with np.errstate(divide="ignore", invalid="ignore"):
                vwap = (num / den.replace(0, np.nan)).astype(float)

            diff = vwap - vwap.shift(1)

            cols[prefix] = vwap
            cols[f"{prefix}__diff"] = diff
            cols[f"{prefix}__slope"] = diff         # slope (redundant with diff but explicit)
            cols[f"{prefix}__price_minus"] = close - vwap
            cols[f"{prefix}__price_div"] = (close / vwap.replace(0, np.nan)).astype(float)

            # streak above vwap: consecutive closes > vwap, resets on <=
            mask = (close > vwap).astype(int)
            streak_above = mask.groupby((mask != mask.shift()).cumsum()).cumsum() * mask
            cols[f"{prefix}__streak_above"] = streak_above.fillna(0).astype("Int64")

        # replace inf with NaNs
        result = pd.DataFrame(cols, index=idx)
        result = result.replace([np.inf, -np.inf], np.nan)
        return result


def _get_typical_price(sdf: pd.DataFrame) -> pd.Series:
    """
    Compute typical price = (high + low + close) / 3.
    If high/low missing, falls back to close.
    """
    if sdf is None or sdf.empty:
        return pd.Series(dtype=float)

    idx = sdf.index
    if all(col in sdf.columns for col in ("high", "low", "close")):
        hp = pd.to_numeric(sdf["high"], errors="coerce").astype(float)
        lp = pd.to_numeric(sdf["low"], errors="coerce").astype(float)
        cl = pd.to_numeric(sdf["close"], errors="coerce").astype(float)
        return ((hp + lp + cl) / 3.0).reindex(idx)
    return pd.to_numeric(sdf["close"], errors="coerce").astype(float).reindex(idx)
