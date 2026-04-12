"""
Module to handles all external market data access (historical and live candles).
Only source of market data for the entire project.
"""

from __future__ import annotations

import ccxt
import pandas as pd
import config as config
import src.utils as utils
import time
import traceback

from dataclasses import asdict, dataclass
from typing import Any, Iterable, List, Optional, Tuple


_COLS_REQUIRED: Tuple[str, ...] = ("ts", "open", "high", "low", "close", "volume")
_COLS_SANITIZED: Tuple[str, ...] = _COLS_REQUIRED + ("datetime",)


@dataclass(frozen=True)
class Candle:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float


def get_candles_from_csv(symbol: str, tf: str) -> pd.DataFrame:
    """
    Loads the corresponding saved candles csv and build/sanitize the sdf.
    """
    candles_csv = utils.get_csv(symbol=symbol, tf=tf)
    if not candles_csv.exists():
        raise RuntimeError(f"Invalid candles csv for: {symbol} {tf}.")

    try:
        rdf = pd.read_csv(candles_csv)
    except Exception:
        traceback.print_exc()
        raise

    return _build_candles(rdf=rdf)


def sync() -> None:
    """
    Incremental/rolling updates to keep ML data consistent and preserve history.
    Fetch new candles only if data already exists; bootstrap otherwise.
    """
    for symbol in config.SYMBOLS:
        for tf in config.TFS:
            candles_csv = utils.get_csv(symbol=symbol, tf=tf)

            if candles_csv.exists():
                candles_from_csv = get_candles_from_csv(symbol=symbol, tf=tf)

                last_ts: Optional[int] = None
                if not candles_from_csv.empty:
                    last_ts = int(candles_from_csv["ts"].iloc[-1])

                since = None if last_ts is None else int(last_ts) + 1
                exchange_candles = _get_exchange_candles(symbol=symbol, tf=tf, since=since)
                if not exchange_candles:
                    continue

                # filter out candles that are not strictly newer than last_ts
                new_candles = [exchange_candle for exchange_candle in exchange_candles if (last_ts is None) or (exchange_candle.ts > last_ts)]
                if not new_candles:
                    continue

                _update_csv(symbol=symbol, tf=tf, new_candles=new_candles)

                if config.PRINT_DEBUG:
                    print(f"Update {utils.get_csv(symbol=symbol, tf=tf)} - append {len(new_candles)} new candles since {since}.")
            else:
                exchange_candles = _get_exchange_candles(symbol=symbol, tf=tf, since=None)
                if not exchange_candles:
                    continue

                _save_csv(symbol=symbol, tf=tf, new_candles=exchange_candles)

                if config.PRINT_DEBUG:
                    print(f"Bootstrap {utils.get_csv(symbol=symbol, tf=tf)}.")


def _build_candles(rdf: pd.DataFrame) -> pd.DataFrame:
    """
    Turn a given raw df to a sanitized df.
    Makes sure all candle data in the df is consistent.
    """
    if (rdf is None) or rdf.empty:
        raise RuntimeError("Valid rdf must be provided.")

    missing = set(_COLS_REQUIRED) - set(rdf.columns)
    if missing:
        raise ValueError(f"rdf is missing required columns: {missing}.")

    rdf_copy = rdf.copy()
    rdf_copy["ts"] = pd.to_numeric(rdf_copy["ts"], errors="coerce")
    rdf_copy = rdf_copy.dropna(subset=["ts"])                                                   # drop rows with NaN ts
    rdf_copy["ts"] = rdf_copy["ts"].astype("int64")
    for col in ("open", "high", "low", "close", "volume"):
        rdf_copy[col] = pd.to_numeric(rdf_copy[col], errors="coerce").astype("float64")         # fill invalid to NaN (do not drop, keep alignment)
    rdf_copy = rdf_copy.drop_duplicates(subset=["ts"], keep="last")                             # dedupe
    rdf_copy = rdf_copy.sort_values("ts").reset_index(drop=True)                                # sort by ts
    rdf_copy["close"] = pd.to_numeric(rdf_copy["close"], errors="coerce").astype("float64")
    rdf_copy["datetime"] = pd.to_datetime(rdf_copy["ts"], unit="ms", utc=True)                  # build a timezone-aware datetime index from ts (ms)
    rdf_copy = rdf_copy.set_index("datetime", drop=False).sort_index()

    # enforce required numeric columns have no NaN (fail fast)
    if rdf_copy[["open", "high", "low", "close", "volume"]].isna().any().any():
        missing_cols = rdf_copy[["open", "high", "low", "close", "volume"]].isna().any()
        raise RuntimeError(f"sdf contains NaNs in required numeric columns: {missing_cols[missing_cols].index.tolist()}")

    for col in _COLS_SANITIZED:                                                                 # ensure sanitized columns exist
        if col not in rdf_copy.columns:
            rdf_copy[col] = pd.NA

    return rdf_copy.loc[:, list(_COLS_SANITIZED)]


def _get_exchange() -> ccxt.Exchange:
    """
    Initializes the ccxt exchange and returns it.
    """
    try:
        ccxt_exchange_cls = getattr(ccxt, config.EXCHANGE_NAME)
        result = ccxt_exchange_cls({"timeout": 10_000, "enableRateLimit": True})
        result.load_markets()
    except Exception:
        traceback.print_exc()
        raise
    return result


def _get_exchange_candle(raw_candle: List[Any]) -> Candle:
    """
    Converts a single ccxt ohlcv candle (row) into a Candle object.
    """
    if (not raw_candle) or (len(raw_candle) < len(Candle.__dataclass_fields__)):
        raise RuntimeError(f"raw_candle is invalid and cannot be converted to Candle object.")

    return Candle(
        ts=int(raw_candle[0]),
        open=float(raw_candle[1]),
        high=float(raw_candle[2]),
        low=float(raw_candle[3]),
        close=float(raw_candle[4]),
        volume=float(raw_candle[5]) if len(raw_candle) > 5 else 0.0)


def _get_exchange_candles(symbol: str, tf: str, since: Optional[int] = None) -> List[Candle]:
    """
    Gets a list of raw candles from the ccxt exchange.
    Converts them into Candle objects and returns the list.
    Sort by timestamp ascending and drop last incomplete candle.
    """
    exchange = _get_exchange()
    if not hasattr(exchange, "fetch_ohlcv"):
        raise RuntimeError(f"Exchange {getattr(exchange, 'id', 'unknown')} does not implement function fetch_ohlcv().")

    try:
        # limit is number of initial candles to fetch (if available)
        raw_candles = exchange.fetch_ohlcv(symbol=symbol, timeframe=tf, since=since, limit=3650)
        if not raw_candles:
            return []

        result: List[Candle] = []
        for raw_candle in raw_candles:
            candle = _get_exchange_candle(raw_candle=raw_candle)
            if candle is not None:
                result.append(candle)

        # drop last candle only if it is likely incomplete
        result.sort(key=lambda c: c.ts)
        if result:
            now_ms = int(time.time() * 1000)
            tf_ms = exchange.parse_timeframe(tf) * 1000
            last_open = result[-1].ts
            if now_ms < last_open + tf_ms:
                result = result[:-1]

        return result
    except Exception:
        traceback.print_exc()
        raise


def _save_csv(symbol: str, tf: str, new_candles: Iterable[Candle]) -> None:
    """
    Saves a collection of sanitized candles in a new csv file.
    """
    new_candles = _build_candles(rdf=pd.DataFrame([asdict(new_candle) for new_candle in new_candles]))

    utils.CANDLES_PATH.mkdir(parents=True, exist_ok=True)
    candles_csv = utils.get_csv(symbol=symbol, tf=tf)
    candles_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_candles_csv = candles_csv.with_name(candles_csv.name + ".tmp")
    new_candles.to_csv(tmp_candles_csv, index=False)
    tmp_candles_csv.replace(candles_csv)


def _update_csv(symbol: str, tf: str, new_candles: Iterable[Candle]) -> None:
    """
    Append new sanitized candles only to an existing csv file.
    """
    candles_from_csv = get_candles_from_csv(symbol=symbol, tf=tf)
    new_candles = _build_candles(rdf=pd.DataFrame([asdict(new_candle) for new_candle in new_candles]))

    if candles_from_csv.empty:
        result = new_candles.copy()
    else:
        frames = [frame for frame in (candles_from_csv, new_candles) if not frame.empty]
        if not frames:
            result = new_candles.copy()
        elif len(frames) == 1:
            result = frames[0].copy()
        else:
            result = pd.concat(frames, ignore_index=True)
            result = result.drop_duplicates(subset=["ts"], keep="last")
            result = result.sort_values("ts").reset_index(drop=True)

    candles_csv = utils.get_csv(symbol=symbol, tf=tf)
    candles_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_candles_csv = candles_csv.with_name(candles_csv.name + ".tmp")
    result.to_csv(tmp_candles_csv, index=False)
    tmp_candles_csv.replace(candles_csv)
