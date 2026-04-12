"""
Optional module to store trades made by a Model.
"""

import config as config
import numpy as np
import pandas as pd
import traceback

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class Trade:
    buy_ts: str
    buy_price: float
    sell_ts: str
    sell_price: float
    profit: float

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts a Trade object to a dictionary.
        """
        return {
            "buy_ts": self.buy_ts,
            "buy_price": round(self.buy_price, config.DECIMALS_ROUNDING),
            "sell_ts": self.sell_ts,
            "sell_price": round(self.sell_price, config.DECIMALS_ROUNDING),
            "profit": round(self.profit, config.DECIMALS_ROUNDING),
        }


def get_trades(sdf: pd.DataFrame, y_pred: np.ndarray, val_idx: int) -> List[Trade]:
    """
    Returns a list of Trade objects from the sdf.
    Trades are created using positional alignment.
    """
    trades: List[Trade] = []

    if sdf is None or sdf.empty:
        return trades

    if "close" not in sdf.columns:
        raise ValueError("sdf must contain 'close' column.")

    if val_idx < 0:
        raise ValueError(f"val_idx {val_idx} must be >= 0.")

    preds = np.asarray(y_pred)
    for rel_pos, pred in enumerate(preds):
        if not _is_buy(pred):
            continue

        buy_idx = val_idx + int(rel_pos)
        sell_idx = buy_idx + int(config.HORIZON)
        if buy_idx < 0 or sell_idx >= len(sdf):
            continue

        # ensure close values exist and are finite
        buy_price_raw = sdf.iloc[buy_idx].get("close", None)
        sell_price_raw = sdf.iloc[sell_idx].get("close", None)
        try:
            buy_price = float(buy_price_raw)
            sell_price = float(sell_price_raw)
        except Exception:
            traceback.print_exc()
            raise

        # skip trade due to non-finite prices
        if not (np.isfinite(buy_price) and np.isfinite(sell_price)):
            continue

        buy_ts = pd.Timestamp(sdf.index[buy_idx]).strftime("%Y-%m-%d %H:%M")
        sell_ts = pd.Timestamp(sdf.index[sell_idx]).strftime("%Y-%m-%d %H:%M")
        profit = sell_price - buy_price

        trades.append(
            Trade(
                buy_ts=buy_ts,
                buy_price=buy_price,
                sell_ts=sell_ts,
                sell_price=sell_price,
                profit=profit)
        )

    if config.PRINT_DEBUG:
        _verbose(trades=trades)

    return trades


def _verbose(trades: List[Trade]) -> None:
    """
    Prints the trade information of all trades in the list.
    """
    trades_df = pd.DataFrame([trade.to_dict() for trade in trades])
    print(f"Trades ({len(trades_df)}):")
    if not trades_df.empty:
        print(trades_df)


def _is_buy(pred: Any) -> bool:
    """
    Determine whether a prediction indicates a buy.
    Handles:
      - binary labels (0/1)
      - probability scores in [0,1] (thresholded by config.MIN_PROB_UP_TO_BUY)
    """
    try:
        if hasattr(pred, "__iter__") and not isinstance(pred, (str, bytes)):
            pred_arr = np.asarray(pred).ravel()
            if pred_arr.size == 0:
                return False
            pred_val = pred_arr[0]
        else:
            pred_val = pred

        if np.issubdtype(np.asarray(pred_val).dtype, np.floating):
            if not np.isfinite(pred_val):
                return False
            return float(pred_val) >= float(config.MIN_PROB_UP_TO_BUY)
        else:
            return int(pred_val) == 1
    except Exception:
        traceback.print_exc()
        raise
