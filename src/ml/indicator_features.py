"""
Module to handle ml features for SINGLE indicators.
"""

from __future__ import annotations

import config as config
import pandas as pd
import src.utils as utils

from src.modules.indicator import Indicator
from typing import List


def build_features_all_tfs(primary_sdf: pd.DataFrame, indicator: Indicator) -> pd.DataFrame:
    """
    Using a primary_sdf, convert the given indicator's column names into canonical names (features).
    Resample and forward-fill all other tfs' dfs to the primary_sdf's index.
    Convert those column names into canonical names (features) as well.
    Return the new sdf.
    """
    if primary_sdf is None:
        raise ValueError("primary_sdf is required.")

    # prepare primary_sdf
    primary_sdf_copy = primary_sdf.copy()
    if not isinstance(primary_sdf_copy.index, pd.DatetimeIndex):
        if "datetime" in primary_sdf_copy.columns:
            primary_sdf_copy.index = pd.to_datetime(primary_sdf_copy["datetime"], utc=True)
        else:
            primary_sdf_copy.index = pd.to_datetime(primary_sdf_copy.index, utc=True)

    if primary_sdf_copy.index.tz is None:
        primary_sdf_copy.index = primary_sdf_copy.index.tz_localize("UTC")
    else:
        primary_sdf_copy.index = primary_sdf_copy.index.tz_convert("UTC")
    primary_sdf_copy = primary_sdf_copy.sort_index()

    if "close" not in primary_sdf_copy.columns:
        raise RuntimeError("primary_sdf must contain 'close' column.")

    # prepare parts
    parts: List[pd.DataFrame] = []

    # prepare config.PRIMARY_TF
    if not primary_sdf_copy.empty:
        features_1_tf = indicator.build_features_1_tf(sdf=primary_sdf_copy)
        features_1_tf = indicator.build_feature_names(sdf=features_1_tf, tf=config.PRIMARY_TF)
        features_1_tf = features_1_tf.reindex(primary_sdf_copy.index)
        parts.append(features_1_tf)

    # prepare config.SECONDARY_TFS
    for tf in config.SECONDARY_TFS:
        resampled_sdf = _resample(original_sdf=primary_sdf_copy, new_tf=tf)
        if resampled_sdf is None or resampled_sdf.empty:
            continue

        features_1_tf = indicator.build_features_1_tf(sdf=resampled_sdf)
        features_1_tf = features_1_tf.reindex(primary_sdf_copy.index).ffill()
        features_1_tf = indicator.build_feature_names(sdf=features_1_tf, tf=tf)
        parts.append(features_1_tf)

    # if no features could be created (ex. missing "close" entirely), return empty df
    if not parts:
        empty_sdf = pd.DataFrame(index=primary_sdf_copy.index)
        if "close" in primary_sdf_copy.columns:
            empty_sdf.insert(0, "close", pd.to_numeric(primary_sdf_copy["close"].reindex(primary_sdf_copy.index), errors="coerce").astype("float64"))
        return empty_sdf

    # concat horizontally, drop fully-NaN columns, fill remaining NaNs conservatively
    result = pd.concat(parts, axis=1)
    result = result.loc[:, ~result.columns.duplicated()].copy()
    result = result.loc[:, result.isna().mean() < 1.0]

    if "close" in primary_sdf_copy.columns:
        close_series = pd.to_numeric(primary_sdf_copy["close"].reindex(primary_sdf_copy.index), errors="coerce").astype("float64")
        if close_series.isna().any():
            raise RuntimeError("Primary 'close' series contains NaNs after coercion.")
        if "close" in result.columns:
            result.drop(columns=["close"], inplace=True)
        result.insert(0, "close", close_series)

    result = result.dropna(subset=["close"]).copy()
    if result.empty:
        raise RuntimeError("No valid 'close' values after feature generation.")
    if not isinstance(result.index, pd.DatetimeIndex):
        raise RuntimeError("Result must have a DatetimeIndex.")
    if result.index.tz is None:
        raise RuntimeError("Result index must be timezone-aware (UTC).")
    if config.HORIZON >= len(result):
        raise ValueError(f"Horizon={config.HORIZON} must be smaller than result length={len(result)}.")

    result["target"] = _get_target(sdf=result)

    return result


def get_feature_names(sdf: pd.DataFrame, indicator: Indicator) -> List[str]:
    """
    Gets the list of features (column names) only from an sdf.
    """
    if sdf is None or sdf.empty:
        return []

    prefix = f"{indicator.name}__"
    cols = [col for col in sdf.columns if col.startswith(prefix) and col not in ("close", "target")]
    if cols:
        return cols

    # fallback: choose numeric columns excluding "close" and "target"
    numeric_cols = [c for c in sdf.columns if pd.api.types.is_numeric_dtype(sdf[c])]
    return [col for col in numeric_cols if col not in ("close", "target")]


def _get_target(sdf: pd.DataFrame) -> pd.Series:
    """
    Returns the binary target (1 if close after HORIZON bars is higher than current close; 0 otherwise)
    Shifted so that the label at time t represents the price move from t -> t + HORIZON.
    """
    if "close" not in sdf.columns:
        raise ValueError("sdf must contain a 'close' column.")

    target = (sdf["close"].shift(-config.HORIZON) > sdf["close"]).astype(int)
    target.name = "target"
    return target


def _resample(original_sdf: pd.DataFrame, new_tf: str) -> pd.DataFrame:
    """
    Combines smaller ohlcv candles to bigger ones (higher timeframes).
    ex. use the data in a 1h tf table to build a new 4h tf table.
    """
    if original_sdf is None or original_sdf.empty:
        raise RuntimeError("Valid original_sdf must be provided.")

    original_sdf_copy = original_sdf.copy()
    if isinstance(original_sdf_copy.index, pd.DatetimeIndex):
        if original_sdf_copy.index.tz is None:
            original_sdf_copy.index = original_sdf_copy.index.tz_localize("UTC")
        else:
            original_sdf_copy.index = original_sdf_copy.index.tz_convert("UTC")
    else:
        if "datetime" in original_sdf_copy.columns:
            original_sdf_copy.index = pd.to_datetime(original_sdf_copy["datetime"], utc=True)
        else:
            raise ValueError("rdf does not have a DatetimeIndex nor a 'datetime' column.")

    original_sdf_copy = original_sdf_copy.sort_index()

    # aggregations to determine how lower tf should be grouped (column: function)
    # ex. higher tf open candle is the lower timeframes first candle
    agg = {"open": "first",
           "high": "max",
           "low": "min",
           "close": "last",
           "volume": "sum"}

    # only keep columns that exist in df
    cols = {}
    for col, func in agg.items():
        if col in original_sdf_copy.columns:
            cols[col] = func
    if not cols:
        return pd.DataFrame()

    # resample through aggregation
    resampled_sdf = original_sdf_copy.resample(utils.sanitize_tf(tf=new_tf)).agg(cols)
    if "close" in resampled_sdf.columns:
        resampled_sdf = resampled_sdf.dropna(subset=["close"], how="any")
    return resampled_sdf
