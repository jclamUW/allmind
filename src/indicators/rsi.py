from __future__ import annotations

import config as config
import numpy as np
import pandas as pd

from src.modules.indicator import Indicator
from typing import Dict, List


_NAME: str = "rsi"
_DESCRIPTIONS: List[str] = ["dist_to_overbought",
                            "dist_to_oversold",
                            "near_overbought",
                            "near_oversold",
                            "is_overbought",
                            "is_oversold",
                            "dist50"]
_RSI_OVERBOUGHT = 70
_RSI_OVERSOLD = 30
_RSI_MARGIN = 5


class Rsi(Indicator):

    def __init__(self):
        super().__init__(name=_NAME, descriptions=_DESCRIPTIONS)

    def build_features_1_tf(self, sdf: pd.DataFrame) -> pd.DataFrame:
        """
        Create sdf with the same index, but containing only RSI feature columns.
        Handles logic where RSI approaching overbought is a good SELL indicator; approaching oversold is a good BUY indicator.
        """
        if sdf is None or sdf.empty:
            raise RuntimeError("Valid sdf must be provided.")

        if "close" not in sdf.columns:
            raise ValueError("sdf must contain 'close' column.")
        close = pd.to_numeric(sdf["close"], errors="coerce").astype(float)

        idx = sdf.index
        if not isinstance(idx, pd.DatetimeIndex):
            raise ValueError("sdf.index must be a DatetimeIndex.")

        # compute price changes once (used for all periods)
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)

        cols: Dict[str, pd.Series] = {}
        periods: List[int] = sorted({int(x) for x in config.PERIODS})
        for p in periods:
            p = int(p)
            prefix = f"{self.name}__p={p}"

            # average gains and losses (simple rolling mean)
            up_avg = up.rolling(window=p, min_periods=1).mean()
            down_avg = down.rolling(window=p, min_periods=1).mean()

            # RSI calculation (safe against divide-by-zero)
            with np.errstate(divide="ignore", invalid="ignore"):
                rs = up_avg / down_avg.replace(0, np.nan)
                rsi = 100 - (100 / (1 + rs))

            cols[prefix] = rsi
            cols[f"{prefix}__diff"] = rsi - rsi.shift(1)
            cols[f"{prefix}__dist50"] = rsi - 50
            cols[f"{prefix}__is_overbought"] = (rsi >= _RSI_OVERBOUGHT).astype("Int64")
            cols[f"{prefix}__is_oversold"] = (rsi <= _RSI_OVERSOLD).astype("Int64")
            cols[f"{prefix}__near_overbought"] = ((rsi >= (_RSI_OVERBOUGHT - _RSI_MARGIN)) & (rsi < _RSI_OVERBOUGHT)).astype("Int64")
            cols[f"{prefix}__near_oversold"] = ((rsi <= (_RSI_OVERSOLD + _RSI_MARGIN)) & (rsi > _RSI_OVERSOLD)).astype("Int64")
            cols[f"{prefix}__dist_to_overbought"] = _RSI_OVERBOUGHT - rsi       # positive when below _RSI_OVERBOUGHT
            cols[f"{prefix}__dist_to_oversold"] = rsi - _RSI_OVERSOLD           # positive when above _RSI_OVERSOLD

        # replace inf with NaNs
        result = pd.DataFrame(cols, index=idx)
        result = result.replace([np.inf, -np.inf], np.nan)
        return result
