from __future__ import annotations

import config as config
import numpy as np
import pandas as pd

from src.modules.indicator import Indicator
from typing import Dict, List


_NAME: str = "sma"
_DESCRIPTIONS: List[str] = ["price_minus", "price_div", "streak_above"]


class Sma(Indicator):

    def __init__(self):
        super().__init__(name=_NAME, descriptions=_DESCRIPTIONS)

    def build_features_1_tf(self, sdf: pd.DataFrame) -> pd.DataFrame:
        """
        Create sdf with the same index, but containing only SMA feature columns.
        Handles logic where candle streaks above the SMA is a good BUY indicator.
        """
        if sdf is None or sdf.empty:
            raise RuntimeError("Valid sdf must be provided.")

        if "close" not in sdf.columns:
            raise ValueError("sdf must contain 'close' column.")
        close = pd.to_numeric(sdf["close"], errors="coerce").astype(float)

        idx = sdf.index
        if not isinstance(idx, pd.DatetimeIndex):
            raise ValueError("sdf.index must be a DatetimeIndex.")

        cols: Dict[str, pd.Series] = {}
        periods: List[int] = sorted({int(x) for x in config.PERIODS})
        for p in periods:
            p = int(p)
            prefix = f"{self.name}__p={p}"

            sma = close.rolling(window=p, min_periods=1).mean()

            cols[prefix] = sma
            cols[f"{prefix}__diff"] = sma - sma.shift(1)
            cols[f"{prefix}__price_minus"] = close - sma
            cols[f"{prefix}__price_div"] = close / sma.replace(0, np.nan)

            # streak above SMA: consecutive closes > EMA; resets on <=
            mask = (close > sma).astype(int)
            streak_above = mask.groupby((mask != mask.shift()).cumsum()).cumsum() * mask
            cols[f"{prefix}__streak_above"] = streak_above.fillna(0).astype("Int64")

        # replace inf with NaNs
        result = pd.DataFrame(cols, index=idx)
        result = result.replace([np.inf, -np.inf], np.nan)
        return result
