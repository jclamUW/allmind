"""
Application global configuration.
"""

from src.indicators.ema import Ema
from src.indicators.rsi import Rsi
from src.indicators.sma import Sma
from src.indicators.sr import Sr
from src.indicators.vwap import Vwap
from src.modules.indicator import Indicator
from typing import Final, List, Tuple


# exchange symbols/timeframes (ccxt)
EXCHANGE_NAME: Final[str] = "kraken"
INDICATORS: Final[List[Indicator]] = [Ema(), Rsi(), Sma(), Sr(), Vwap()]
SYMBOLS: Final[Tuple[str, ...]] = ("BTC/USD", "ETH/USD", "XRP/CAD")
TFS: Final[Tuple[str, ...]] = ("1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w")


# ml/training parameters
HORIZON: Final[int] = 15                            # dependent on PRIMARY_TF
MAX_INDICATOR_TRAINING_ATTEMPTS: Final[int] = 5
VALIDATION_FRAC: Final[float] = 0.25                # size as fraction of dataset

PRIMARY_TF: Final[str] = "4h"
SECONDARY_TFS: Final[Tuple[str, ...]] = ("1h", "1d")
PERIODS: Final[Tuple[int, ...]] = (10, 20, 30, 40, 50, 60, 80, 100)

MIN_EXP_PROFIT_FRAC: Final[float] = 0.05            # expected profit as fraction (ex. 0.10 == 10%)
MIN_PROB_UP_TO_BUY: Final[float] = 0.7              # buy if model p(up) >= this
MIN_PROB_UP_TO_SELL: Final[float] = 0.3             # sell if model p(up) <= this

DECIMALS_ROUNDING: Final[int] = 3
PRINT_DEBUG: Final[bool] = False


def _validate_config() -> None:
    if not (0.0 < VALIDATION_FRAC < 1.0):
        raise ValueError("VALIDATION_FRAC must be between 0 and 1.")

    if PRIMARY_TF not in TFS:
        raise ValueError(f"PRIMARY_TF {PRIMARY_TF!r} not in TFS.")
    for tf in SECONDARY_TFS:
        if tf not in TFS:
            raise ValueError(f"SECONDARY_TFS contains unknown timeframe {tf!r}.")
    if PRIMARY_TF in SECONDARY_TFS:
        raise ValueError(f"PRIMARY_TF {PRIMARY_TF!r} cannot be in SECONDARY_TFS {SECONDARY_TFS!r}.")

    if any(p <= 0 for p in PERIODS):
        raise ValueError(f"All periods must be positive integers.")

    if not (0.0 <= MIN_EXP_PROFIT_FRAC):
        raise ValueError("MIN_EXP_PROFIT_FRAC must be >= 0.")
    if not (0.0 <= MIN_PROB_UP_TO_BUY <= 1.0):
        raise ValueError("MIN_PROB_UP_TO_BUY must be between 0 and 1.")
    if not (0.0 <= MIN_PROB_UP_TO_SELL <= 1.0):
        raise ValueError("MIN_PROB_UP_TO_SELL must be between 0 and 1.")
    if MIN_PROB_UP_TO_SELL >= MIN_PROB_UP_TO_BUY:
        raise ValueError("MIN_PROB_UP_TO_SELL must be lower than MIN_PROB_UP_TO_BUY.")


# run checks on import
_validate_config()
