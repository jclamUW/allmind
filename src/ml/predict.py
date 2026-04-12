"""
Prediction functions that verify saved models.
Holds all predict functions (only source of truth for predictions in the project).
"""

from __future__ import annotations

import config as config
import numpy as np
import pandas as pd
import src.ml.indicators_features as indicators_features_mod
import src.ml.meta_features as meta_features_mod
import src.modules.model as model_mod
import src.utils as utils
import traceback

from sklearn.pipeline import Pipeline
from typing import Optional, Tuple


def evaluate_present_and_future(symbol: str, primary_tf: str) -> None:
    """
    Determines if it is currently a good buy or sell point.
    Gives a potential future buy and sell price and the probability.
    """
    # prepare authoritative features for model
    latest_close, indicators_features_sdf = indicators_features_mod.build_features_all_tfs(symbol=symbol, primary_tf=primary_tf)

    # get probability (possibly via meta model)
    meta_prob = _predict_meta_prob(symbol=symbol, primary_tf=primary_tf)

    avg_price_increase = get_avg_price_increase_from_history(sdf=indicators_features_sdf)
    avg_price_decrease = get_avg_price_decrease_from_history(sdf=indicators_features_sdf)
    avg_exp_return = meta_prob * avg_price_increase + (1.0 - meta_prob) * avg_price_decrease
    avg_exp_return_frac = avg_exp_return / (float(latest_close) if latest_close != 0 else 1.0)

    is_buy_now = (meta_prob >= config.MIN_PROB_UP_TO_BUY) and (avg_exp_return_frac >= config.MIN_EXP_PROFIT_FRAC)
    is_sell_now = (meta_prob <= config.MIN_PROB_UP_TO_SELL) and (avg_exp_return_frac <= -config.MIN_EXP_PROFIT_FRAC)

    current_s, current_r = _get_current_sr_range(features_sdf=indicators_features_sdf, latest_close=latest_close)
    next_res = _get_next_res(features_sdf=indicators_features_sdf, latest_close=current_r if current_r is not None else latest_close)

    if config.PRINT_DEBUG:
        print("evaluating...")
        print(f"meta_prob={round(meta_prob, config.DECIMALS_ROUNDING)}   avg_price_increase={round(avg_price_increase, config.DECIMALS_ROUNDING)}   avg_price_decrease={round(avg_price_decrease, config.DECIMALS_ROUNDING)}   avg_exp_return={round(avg_exp_return, config.DECIMALS_ROUNDING)}")

    lines = [
        f"Symbol:                               {symbol} ({primary_tf})",
        f"Current analysis:                     {'BUY' if is_buy_now else 'SELL' if is_sell_now else 'HOLD'}",
        f"Next entry/exit:                      ({current_s} - {current_r}) --> {next_res} ({round(meta_prob * 100, config.DECIMALS_ROUNDING)}% chance of price increase)",
        f"Average historic expected returns:    {round(avg_exp_return_frac * 100, config.DECIMALS_ROUNDING)}%"
    ]
    utils.save_analysis(lines)


def get_avg_price_decrease_from_history(sdf: pd.DataFrame) -> float:
    """
    Returns the average dollar amount (future_close - close) over periods where future_close <= close (negative or zero).
    """
    if sdf is None or sdf.empty or "close" not in sdf.columns:
        return 0.0

    closes = sdf["close"].astype(float)
    if len(closes) <= config.HORIZON:
        return 0.0

    future_close = closes.shift(-config.HORIZON)
    profits = future_close - closes
    profits_negative = profits[(future_close <= closes)].dropna()
    return float(profits_negative.mean()) if not profits_negative.empty else 0.0


def get_avg_price_increase_from_history(sdf: pd.DataFrame) -> float:
    """
    Returns the average dollar amount (future_close - close) over periods where future_close > close.
    """
    if sdf is None or sdf.empty or "close" not in sdf.columns:
        return 0.0

    closes = sdf["close"].astype(float)
    if len(closes) <= config.HORIZON:
        return 0.0

    future_close = closes.shift(-config.HORIZON)
    profits = future_close - closes
    profits_positive = profits[(future_close > closes)].dropna()
    return float(profits_positive.mean()) if not profits_positive.empty else 0.0


def predict_avg_positive_profit(sdf: pd.DataFrame) -> float:
    """
    Predicts the average positive profit as a fraction.

    Used by indicators_features.py
    """
    if sdf is None or sdf.empty or "close" not in sdf.columns:
        return 0.0
    closes = pd.to_numeric(sdf["close"], errors="coerce").astype(float)
    if len(closes) <= config.HORIZON:
        return 0.0
    future_close = closes.shift(-config.HORIZON)
    profit = (future_close - closes) / closes.replace(0, np.nan)
    profit_positive = profit[(future_close > closes)].dropna()
    if profit_positive.empty:
        return 0.0
    return float(profit_positive.mean())


def predict_proba_from_estimator(model: model_mod.Model, x: np.ndarray) -> float:
    """
    Given a Model (model_mod.Model), return probability (0.0 - 1.0) of class 1 (price up).

    Robust strategy:
      - Ensure x is an array with shape (n_samples, n_features).
      - Try model.estimator.predict_proba(x) first.
      - If that fails and estimator is a Pipeline but final estimator supports predict_proba/decision_function/predict,
        run the pipeline's transformers (all but last step) to produce transformed X, then call the final estimator's method.
      - Try decision_function -> sigmoid, or predict -> deterministic 0/1.
      - Raise a clear RuntimeError if none of the above works.
    """
    model_name = getattr(getattr(model, "manifest", None), "model_name", repr(model))

    try:
        # ensure numpy array shape
        x_arr = np.asarray(x)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)

        est = getattr(model, "estimator", None)
        if est is None:
            raise RuntimeError(f"Model {model_name} has no estimator attribute.")

        # direct predict_proba on estimator (works for Pipeline if final exposes it)
        if hasattr(est, "predict_proba"):
            try:
                probs = est.predict_proba(x_arr)
                if getattr(probs, "ndim", 0) == 2 and probs.shape[1] >= 2:
                    cls_idx = None
                    classes = getattr(est, "classes_", None)
                    if classes is not None:
                        try:
                            cls_idx = list(classes).index(1)
                        except ValueError:
                            cls_idx = probs.shape[1] - 1
                    if cls_idx is None:
                        cls_idx = probs.shape[1] - 1
                    prob = float(probs[0, cls_idx])
                else:
                    prob = float(probs[0, -1])
                prob = max(0.0, min(1.0, prob))

                print(f"predict_proba_from_estimator (predict_proba) for model {model_name}: {round(prob, config.DECIMALS_ROUNDING)}")

                return prob
            except Exception:
                traceback.print_exc()
                raise

        # if estimator is Pipeline, try to use final step explicitly
        if isinstance(est, Pipeline):
            if len(est.steps) >= 1:
                final_name, final_est = est.steps[-1]
                transformers_pipeline = est[:-1]  # all but final step
                transformed_x: Optional[np.ndarray] = None
                try:
                    if hasattr(transformers_pipeline, "transform"):
                        transformed_x = transformers_pipeline.transform(x_arr)
                    else:
                        transformed_x = x_arr
                except Exception:
                    traceback.print_exc()
                    raise

                # final estimator predict_proba
                if hasattr(final_est, "predict_proba") and transformed_x is not None:
                    try:
                        probs = final_est.predict_proba(transformed_x)
                        if getattr(probs, "ndim", 0) == 2 and probs.shape[1] >= 2:
                            cls_idx = None
                            classes = getattr(final_est, "classes_", None)
                            if classes is not None:
                                try:
                                    cls_idx = list(classes).index(1)
                                except ValueError:
                                    cls_idx = probs.shape[1] - 1
                            if cls_idx is None:
                                cls_idx = probs.shape[1] - 1
                            prob = float(probs[0, cls_idx])
                        else:
                            prob = float(probs[0, -1])
                        return max(0.0, min(1.0, prob))
                    except Exception:
                        traceback.print_exc()
                        raise

                # final estimator decision_function -> sigmoid
                if hasattr(final_est, "decision_function") and transformed_x is not None:
                    try:
                        scores = final_est.decision_function(transformed_x)
                        score0 = float(scores[0]) if getattr(scores, "ndim", 0) == 0 else float(scores[0])
                        prob = 1.0 / (1.0 + np.exp(-score0))
                        return max(0.0, min(1.0, prob))
                    except Exception:
                        traceback.print_exc()
                        raise

                # final estimator predict -> deterministic 0/1
                if hasattr(final_est, "predict") and transformed_x is not None:
                    try:
                        pred = final_est.predict(transformed_x)
                        pred0 = int(pred[0])
                        return 1.0 if pred0 == 1 else 0.0
                    except Exception:
                        traceback.print_exc()
                        raise

        # try decision_function on top-level estimator (non-pipeline)
        if hasattr(est, "decision_function"):
            try:
                scores = est.decision_function(x_arr)
                score0 = float(scores[0]) if getattr(scores, "ndim", 0) == 0 else float(scores[0])
                prob = 1.0 / (1.0 + np.exp(-score0))
                return max(0.0, min(1.0, prob))
            except Exception:
                traceback.print_exc()
                raise

        # try predict on top-level estimator
        if hasattr(est, "predict"):
            try:
                pred = est.predict(x_arr)
                pred0 = int(pred[0])
                return 1.0 if pred0 == 1 else 0.0
            except Exception:
                traceback.print_exc()
                raise

        # nothing usable found
        raise RuntimeError(f"Estimator for model {model_name} provides no usable prediction methods (no predict_proba, decision_function or predict).")
    except Exception:
        traceback.print_exc()
        raise


def _get_current_sr_range(features_sdf: pd.DataFrame, latest_close: float) -> Tuple[float | None, float | None]:
    """
    Gets the current support/resistance levels.
    """
    levels = _get_sr_levels(features_sdf)
    if not levels:
        return None, None

    min_gap_frac = 0.05
    floor = latest_close * (1.0 + min_gap_frac)
    below = latest_close
    above = None

    for level in levels:
        if level < latest_close:
            below = level
        elif level >= floor and above is None:
            above = level
            break

    return below, above


def _get_next_res(features_sdf: pd.DataFrame, latest_close: float) -> float | None:
    """
    Gets the next closest resistance.
    """
    levels = _get_sr_levels(features_sdf)
    if not levels:
        return None

    min_gap_frac = 0.05
    floor = latest_close * (1.0 + min_gap_frac)
    found_current = False

    for level in levels:
        if level >= floor:
            if found_current:
                return level
            found_current = True

    return None


def _get_sr_levels(features_sdf: pd.DataFrame) -> list[float]:
    """
    Gets all support/resistance levels.
    """
    if features_sdf is None or features_sdf.empty:
        return []

    row = features_sdf.iloc[-1]
    levels = []

    for col in features_sdf.columns:
        if not (col.startswith("sr__") and "__roll_max" in col):
            continue

        val = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
        if pd.notna(val):
            levels.append(float(val))

    return sorted(set(levels))


def _predict_meta_prob(symbol: str, primary_tf: str) -> float:
    """
    Build the meta features and return probability.
    """
    try:
        meta_model_name = utils.get_model_name(symbol=symbol, tf=primary_tf, indicator=None)
        try:
            meta_model = model_mod.load_model(model_name=meta_model_name)
        except Exception:
            traceback.print_exc()
            raise

        meta_manifest = meta_model.manifest
        if not meta_manifest.features:
            raise RuntimeError("meta_manifest missing features.")

        meta_features = meta_features_mod.get_feature_names(symbol=symbol, primary_tf=primary_tf)
        x = [float(meta_features.get(col, 0.0)) for col in meta_manifest.features]
        x_arr = np.array(x).reshape(1, -1)

        # use meta estimator to predict probability
        return predict_proba_from_estimator(model=meta_model, x=x_arr)
    except Exception:
        traceback.print_exc()
        raise
