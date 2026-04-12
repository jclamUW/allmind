from __future__ import annotations

import numpy as np
import pandas as pd
import config as config
import src.modules.candle as candle_mod
import src.modules.model as model_mod
import src.ml.indicators_features as indicators_features_mod
import src.ml.predict as predict_mod
import src.utils as utils
import traceback

from typing import Dict, Tuple


def build_features(symbol: str, primary_tf: str) -> pd.DataFrame:
    """
    Build a meta training/inference df for primary_tf by collecting saved primary model predictions from all indicators.
    For each indicator we produce columns:
      - <indicator.name>__prob      : predicted probability of 'up' from that indicator's model (tf=primary_tf)
      - <indicator.name>__conf      : abs(p - 0.5)
      - <indicator.name>__exp_profit: historical mean positive-profit estimate (scalar, repeated)
    The returned df is indexed by the primary_tf candle datetimes (timezone-aware).
    Contains 'close' (if available) and 'target' aligned to config.HORIZON.
    """
    # load candles for primary_tf (index must be timezone-aware DatetimeIndex)
    candles_csv = candle_mod.get_candles_from_csv(symbol=symbol, tf=primary_tf)
    if candles_csv is None or candles_csv.empty:
        raise RuntimeError("candles_csv is empty or missing.")

    # how many rows will have valid primary model outputs (trainer trimmed last HORIZON rows during training)
    if len(candles_csv) - config.HORIZON <= 0:
        raise RuntimeError("Not enough rows in candles_csv to build meta model.")

    # build unified meta df with all indicators (this gives prob/conf/exp_profit series aligned to primary_idx)
    primary_idx = candles_csv.index
    indicators_features = indicators_features_mod.get_feature_names(symbol=symbol, primary_tf=primary_tf)
    result = pd.DataFrame(indicators_features, index=primary_idx)

    if "close" in candles_csv.columns:
        result["close"] = candles_csv["close"].reindex(primary_idx)

    # target aligned to HORIZON (indicator trainer's label)
    result["target"] = (candles_csv["close"].shift(-config.HORIZON) > candles_csv["close"]).astype(int)
    result = result.loc[:, result.isna().mean() < 1.0]
    if "target" in result.columns:
        result = result.dropna(subset=["target"])

    if "target" not in result.columns:
        raise RuntimeError("Result missing 'target' column.")

    return result


def get_feature_names(symbol: str, primary_tf: str) -> Dict[str, float]:
    """
    Get meta features (column names) list.
    """
    result: Dict[str, float] = {}
    per_tf_cache: Dict[str, Tuple[pd.DataFrame, float]] = {}

    # load meta model manifest (will raise if missing)
    meta_model_name = utils.get_model_name(symbol=symbol, tf=primary_tf, indicator=None)
    meta_model = model_mod.load_model(model_name=meta_model_name)

    # helper: map indicator name to Indicator object for loading indicator models
    indicator_by_name = {ind.name: ind for ind in config.INDICATORS}

    for feature in meta_model.manifest.features:
        if "tf=" in feature:
            tf = feature.split("tf=", 1)[1].split("__", 1)[0]
        else:
            tf = config.PRIMARY_TF

        # ---------- PROBABILITY ----------
        if feature.endswith("__prob"):
            indicator_name = feature.split("__", 1)[0]
            try:
                # ensure we have the indicators_sdf for this tf in cache
                if tf not in per_tf_cache:
                    latest, indicators_sdf = indicators_features_mod.build_features_all_tfs(symbol=symbol, primary_tf=tf)
                    per_tf_cache[tf] = (indicators_sdf, latest)
                indicators_sdf, latest = per_tf_cache[tf]

                # load the indicator model (must exist)
                indicator = indicator_by_name.get(indicator_name)
                if indicator is None:
                    raise RuntimeError(f"No indicator object with name {indicator_name} in config.INDICATORS")

                indicator_model_name = utils.get_model_name(symbol=symbol, tf=tf, indicator=indicator)
                indicator_model = None
                try:
                    indicator_model = model_mod.load_model(model_name=indicator_model_name)
                except Exception:
                    traceback.print_exc()
                    raise

                # build input x for the indicator model using the last row (defensively)
                if indicator_model is not None:
                    # use same logic as indicators_features_mod: last row aligned to indicator_model.manifest.features
                    last_row = indicators_sdf.iloc[-1]
                    required_cols = list(indicator_model.manifest.features or [])
                    row_values = []
                    for col in required_cols:
                        if col in last_row.index:
                            row_values.append(last_row[col])
                        else:
                            row_values.append(0.0)

                    x_row = np.asarray(row_values, dtype=float).reshape(1, -1)
                    prob = float(predict_mod.predict_proba_from_estimator(model=indicator_model, x=x_row))
                else:
                    prob = 0.5
                result[feature] = prob
                continue
            except Exception:
                traceback.print_exc()
                raise

        # ---------- CONFIDENCE ----------
        if feature.endswith("__conf"):
            pkey = feature.replace("__conf", "__prob")
            result[feature] = abs(result.get(pkey, 0.5) - 0.5)
            continue

        # ---------- EXPECTED PROFIT ----------
        if feature.endswith("__exp_profit"):
            indicator_name = feature.split("__", 1)[0]
            try:
                # make sure we have the per-tf df and latest
                if tf not in per_tf_cache:
                    latest, indicators_sdf = indicators_features_mod.build_features_all_tfs(symbol=symbol, primary_tf=tf)
                    per_tf_cache[tf] = (indicators_sdf, latest)
                indicators_sdf, latest = per_tf_cache[tf]

                # mean_up: estimate mean upward price move in absolute price units
                mean_up = predict_mod.get_avg_price_increase_from_history(sdf=indicators_sdf)
                # if latest is not provided (NaN or 0), fallback to last close from df or 1.0
                if latest is None or latest == 0:
                    if "close" in indicators_sdf.columns and not indicators_sdf["close"].empty:
                        latest = float(indicators_sdf["close"].iloc[-1])
                    else:
                        latest = 1.0
                result[feature] = (mean_up / latest) if latest != 0 else 0.0
                continue
            except Exception:
                traceback.print_exc()
                raise

        # default fallback value
        result[feature] = result.get(feature, 0.0)

    return result
