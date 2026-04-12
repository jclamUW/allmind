"""
Module to handle ml features for MULTIPLE indicators.
There are preliminary steps after individual indicator models have been trained and before the (unified) meta model is trained.
"""

from __future__ import annotations

import config as config
import numpy as np
import pandas as pd
import src.ml.indicator_features as indicator_features_mod
import src.ml.predict as predict_mod
import src.modules.model as model_mod
import src.modules.candle as candle_mod
import src.utils as utils
import traceback

from typing import Dict, List, Tuple


def get_feature_names(symbol: str, primary_tf: str) -> Dict[str, pd.Series]:
    """
    Produce features for all indicators.
    """
    result: Dict[str, pd.Series] = {}

    candles_csv = candle_mod.get_candles_from_csv(symbol=symbol, tf=primary_tf)
    if candles_csv is None or candles_csv.empty:
        raise RuntimeError(f"No csv for {symbol} {primary_tf}.")

    # canonical primary index (make tz-aware)
    primary_idx = candles_csv.index
    if not isinstance(primary_idx, pd.DatetimeIndex):
        if "datetime" in candles_csv.columns:
            primary_idx = pd.to_datetime(candles_csv["datetime"], utc=True)
        else:
            primary_idx = pd.to_datetime(primary_idx, utc=True)
    if primary_idx.tz is None:
        primary_idx = primary_idx.tz_localize("UTC")
    else:
        primary_idx = primary_idx.tz_convert("UTC")

    # build the multi-indicator sdf once (expensive), reuse below
    try:
        _, indicators_sdf = build_features_all_tfs(symbol=symbol, primary_tf=primary_tf)
    except Exception:
        traceback.print_exc()
        raise

    # make sure the raw df returned is reindexed to the canonical primary index and tz-aware
    if indicators_sdf.index.tz is None:
        indicators_sdf.index = indicators_sdf.index.tz_localize("UTC")
    else:
        indicators_sdf.index = indicators_sdf.index.tz_convert("UTC")
    indicators_sdf = indicators_sdf.reindex(primary_idx)

    for indicator in config.INDICATORS:
        # load saved primary model
        try:
            model_name = utils.get_model_name(symbol=symbol, tf=primary_tf, indicator=indicator)
            model = model_mod.load_model(model_name=model_name)
        except Exception:
            traceback.print_exc()
            raise

        # create a trimmed copy aligned to what indicator trainer would have used
        indicators_sdf_copy = indicators_sdf.copy()
        if len(indicators_sdf_copy) > config.HORIZON:
            indicators_sdf_copy = indicators_sdf_copy.iloc[:-config.HORIZON].copy()
        if indicators_sdf_copy.empty:
            raise RuntimeError("indicators_sdf_copy is empty.")

        # ensure columns the estimator expects exist (fill missing with zeros)
        features = list(model.manifest.features or [])
        missing = [c for c in features if c not in indicators_sdf_copy.columns]
        if missing:
            # build a single DataFrame holding all missing columns, then concat once
            missing_df = pd.DataFrame(0.0, index=indicators_sdf_copy.index, columns=missing)
            indicators_sdf_copy = pd.concat([indicators_sdf_copy, missing_df], axis=1)
            indicators_sdf_copy = indicators_sdf_copy.loc[:, ~indicators_sdf_copy.columns.duplicated()].copy()

        # prepare X only if features exist; otherwise mark we don't need to call estimator
        x = None
        need_predict = True
        if not features:
            # model expects zero features — produce NaN probability series (keeps shapes consistent)
            probs_arr = np.full(shape=(len(indicators_sdf_copy),), fill_value=np.nan)
            need_predict = False
        else:
            features = list(dict.fromkeys(features))
            indicators_sdf_copy = indicators_sdf_copy.loc[:, ~indicators_sdf_copy.columns.duplicated()].copy()
            x = indicators_sdf_copy.reindex(columns=features, fill_value=0.0).values
            if x.shape[1] != len(features):
                raise RuntimeError(f"Feature mismatch: x has {x.shape[1]} cols but expected {len(features)}")

        if config.PRINT_DEBUG:
            # ---------- DIAGNOSTIC: inspect the actual x passed to the model ----------
            print(f"--- DIAG: indicator={indicator.name} model={model_name} ---")
            print("features_list length:", len(features))
            print("unified_indicator_df.shape:", getattr(indicators_sdf_copy, "shape", None))
            print("x.shape:", getattr(x, "shape", None) if x is not None else "no x (no features)")
            if x is not None and getattr(x, "ndim", 0) == 2:
                try:
                    rows, cols = x.shape
                    if rows * cols <= 50000:
                        unique_rows = np.unique(x, axis=0)
                        print("unique rows count:", unique_rows.shape[0])
                        print("x row 0 sample (first 10 cols):", x[0][:10] if x.shape[1] > 0 else "no cols")
                        if unique_rows.shape[0] <= 5:
                            print("unique rows (up to 5):", unique_rows[:5])
                except Exception:
                    traceback.print_exc()
                    raise

                try:
                    col_nonzero_counts = (indicators_sdf_copy != 0).sum().head(10).to_dict()
                    print("non-zero counts (first 10 cols):", col_nonzero_counts)
                except Exception:
                    traceback.print_exc()
                    raise
            print(f"--- END DIAG: indicator={indicator.name} ---")
            # ------------------------------------------------------------------------

        # attempt prediction only when we actually need to (i.e., features exist)
        if need_predict:
            try:
                probs_arr = model.estimator.predict_proba(x)[:, 1]
            except Exception:
                try:
                    scores = model.estimator.decision_function(x)
                    probs_arr = 1.0 / (1.0 + np.exp(-scores))
                except Exception:
                    probs_arr = np.full(shape=(len(indicators_sdf_copy),), fill_value=np.nan)

        # series aligned to trimmed index, then reindex back to full primary index (last HORIZON become NaN)
        s_prob = pd.Series(data=probs_arr, index=indicators_sdf_copy.index, name=f"{indicator.name}__prob")
        s_prob = s_prob.reindex(primary_idx)
        s_conf = s_prob.subtract(0.5).abs()
        try:
            exp_profit_scalar = float(predict_mod.predict_avg_positive_profit(sdf=indicators_sdf_copy))
        except Exception:
            exp_profit_scalar = float(np.nan)
        s_exp_profit = pd.Series(data=exp_profit_scalar, index=indicators_sdf_copy.index, name=f"{indicator.name}__exp_profit")
        s_exp_profit = s_exp_profit.reindex(primary_idx)
        result[f"{indicator.name}__prob"] = s_prob
        result[f"{indicator.name}__conf"] = s_conf
        result[f"{indicator.name}__exp_profit"] = s_exp_profit

    return result


def build_features_all_tfs(symbol: str, primary_tf: str) -> Tuple[float, pd.DataFrame]:
    """
    Build x (1xn) using features as the authoritative ordering.
    Returns Tuple with information: (latest_close, unified_df).
    """
    candles_csv = candle_mod.get_candles_from_csv(symbol=symbol, tf=primary_tf)
    if candles_csv is None or candles_csv.empty:
        raise RuntimeError("No market data available for prediction.")

    indicators_sdf = _build_features_all_tfs(primary_sdf=candles_csv)
    if "close" not in indicators_sdf.columns and "close" in candles_csv.columns:
        indicators_sdf["close"] = candles_csv["close"].reindex(indicators_sdf.index)

    meta_model_name = utils.get_model_name(symbol=symbol, tf=primary_tf, indicator=None)
    if utils.get_model_folder(model_name=meta_model_name).exists():
        try:
            meta_model = model_mod.load_model(model_name=meta_model_name)
        except Exception:
            traceback.print_exc()
            raise

        missing_col = [col for col in meta_model.manifest.features if col not in indicators_sdf.columns]
        if missing_col:
            new_cols = {}
            for col in missing_col:
                if col == "close" and "close" in candles_csv.columns:
                    new_cols[col] = candles_csv["close"].reindex(indicators_sdf.index)
                else:
                    new_cols[col] = 0.0
            missing_df = pd.DataFrame(new_cols, index=indicators_sdf.index)
            indicators_sdf = pd.concat([indicators_sdf, missing_df], axis=1)

    # align to canonical primary index, ensure tz/sort, then forward-fill and impute zeros
    indicators_sdf = indicators_sdf.reindex(candles_csv.index)
    if indicators_sdf.index.tz is None:
        indicators_sdf.index = indicators_sdf.index.tz_localize("UTC")
    else:
        indicators_sdf.index = indicators_sdf.index.tz_convert("UTC")
    indicators_sdf = indicators_sdf.sort_index().ffill().fillna(0)

    if indicators_sdf.empty:
        raise RuntimeError("indicators_sdf is empty after sanitizing.")

    # make sure last_row is a single-row Series (not a multi-row DataFrame)
    last_row = indicators_sdf.iloc[-1]
    latest_close = float(last_row["close"]) if "close" in last_row.index else float(candles_csv["close"].iloc[-1])

    return latest_close, indicators_sdf


def _build_features_all_tfs(primary_sdf: pd.DataFrame) -> pd.DataFrame:
    """
    Takes primary_sdf and runs build_features_all_tfs() for all indicators.
    Collect all dfs and concatenate into one sdf.
    """
    if primary_sdf is None or primary_sdf.empty:
        raise RuntimeError("Valid primary_sdf must be provided.")

    parts: List[pd.DataFrame] = []
    for indicator in config.INDICATORS:
        parts.append(indicator_features_mod.build_features_all_tfs(primary_sdf=primary_sdf, indicator=indicator))
    if not parts:
        raise RuntimeError("No sdf created for indicators.")

    # always ensure we have close available if present in primary_sdf
    base_close = None
    if "close" in primary_sdf.columns:
        base_close = primary_sdf[["close"]].copy()

    result = pd.concat(parts, axis=1)

    # if multiple 'close' columns exist, consolidate them into a single 'close' series
    # take the last non-null value across duplicate 'close' columns per row; otherwise take the last column if bfill fails
    try:
        close_mask = [c == "close" for c in result.columns]
        if sum(close_mask) > 1:
            close_df = result.loc[:, close_mask]
            try:
                consolidated = close_df.bfill(axis=1).iloc[:, 0]
            except Exception:
                consolidated = close_df.iloc[:, -1]
            result = result.loc[:, [c for c in result.columns if c != "close"]]
            consolidated = consolidated.reindex(result.index)
            result.insert(0, "close", consolidated)
    except Exception:
        traceback.print_exc()
        raise

    if "close" not in result.columns and base_close is not None:
        result.insert(0, "close", base_close["close"].reindex(result.index))
    result = result.reindex(primary_sdf.index)
    result = result.ffill().fillna(0)
    result = result.loc[:, result.isna().mean() < 1.0]
    return result
