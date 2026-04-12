"""
Module to handle ml training for individual indicators.
Trains one indicator against multiple tfs.
"""

from __future__ import annotations

import config as config
import os
import pandas as pd
import src.ml.indicator_features as indicator_features_mod
import src.modules.candle as candle_mod
import src.modules.manifest as manifest_mod
import src.modules.model as model_mod
import src.modules.trade as trade_mod
import src.utils as utils
import traceback
import warnings

from src.modules.indicator import Indicator
from typing import Any, Optional, Tuple


warnings.filterwarnings("ignore", module="sklearn")
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning")


_N_ESTIMATORS: int = 300
_N_JOBS: int = 1


def train(symbol: str, tfs: Tuple[str, ...], indicator: Indicator) -> None:
    """
    Trains and saves the best models for each of the given multiple tfs.
    """
    for tf in tfs:
        try:
            _train_1_tf(symbol=symbol, tf=tf, indicator=indicator)
        except Exception:
            traceback.print_exc()
            raise


def _train_1_tf(symbol: str, tf: str, indicator: Indicator) -> None:
    """
    Creates a model with the given symbol, tf, and indicator.
    Trains the model against an indicator_features_sdf (fully prepared).
    Repeats the process config.MAX_INDICATOR_TRAINING_ATTEMPTS times and saves the model with the best metrics to a json.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model_name = utils.get_model_name(symbol=symbol, tf=tf, indicator=indicator)

    candles_from_csv = candle_mod.get_candles_from_csv(symbol=symbol, tf=tf)
    if candles_from_csv is None or candles_from_csv.empty:
        raise RuntimeError(f"No candle data returned from {symbol} {tf}.")

    # drop rows with any remaining NaNs
    indicator_sdf = indicator_features_mod.build_features_all_tfs(primary_sdf=candles_from_csv, indicator=indicator)
    if indicator_sdf is None:
        raise RuntimeError(f"get_indicator_sdf() returned None for {model_name}.")

    # ensure primary 'close' is present and numeric in the resulting sdf
    if "close" in indicator_sdf.columns:
        try:
            close_series = pd.to_numeric(indicator_sdf["close"], errors="coerce").astype("float64")
        except Exception:
            traceback.print_exc()
            raise

        if close_series.isna().any():
            raise RuntimeError(f"Primary 'close' series contains NaNs after coercion for {model_name}.")
        indicator_sdf["close"] = close_series
    else:
        raise RuntimeError(f"indicator_sdf for {model_name} missing 'close' column.")

    # drop rows that contain NaNs after warmup trim (expected to be few)
    indicator_sdf = indicator_sdf.dropna(how="any")

    if "target" not in indicator_sdf.columns:
        raise RuntimeError(f"indicator_sdf for {model_name} missing 'target' column.")
    if indicator_sdf.empty:
        raise RuntimeError("No rows remain after dropping NaNs post-warmup.")
    if config.HORIZON >= len(indicator_sdf):
        raise ValueError(f"Horizon={config.HORIZON} must be smaller than indicator_sdf length={len(indicator_sdf)}.")
    indicator_sdf = indicator_sdf.iloc[:-config.HORIZON].copy()

    # coerce features to numeric and validate
    indicator_features = indicator_features_mod.get_feature_names(sdf=indicator_sdf, indicator=indicator)
    if not indicator_features:
        raise RuntimeError(f"No features extracted for indicator {indicator.name} on {symbol} {tf}.")
    duplicate_features = pd.Index(indicator_features)[pd.Index(indicator_features).duplicated()].tolist()
    if duplicate_features:
        raise RuntimeError(f"Duplicate feature names for {model_name}: {duplicate_features}")

    model_name = utils.get_model_name(symbol=symbol, tf=tf, indicator=indicator)
    try:
        x_series = indicator_sdf.loc[:, indicator_features].copy()
    except Exception:
        traceback.print_exc()
        raise

    if list(x_series.columns) != list(indicator_features):
        raise RuntimeError(f"Feature order/duplication mismatch for {model_name}: x_series has {x_series.shape[1]} cols, indicator_features has {len(indicator_features)}.")

    # ensure target is integer 0/1 array
    y_series = pd.to_numeric(indicator_sdf["target"], errors="coerce")
    if y_series.isna().any():
        raise RuntimeError(f"Target column contains NaNs for {model_name}.")

    x = x_series.values                 # ML inputs (numpy)
    y = y_series.astype(int).values     # ML outputs (numpy)

    if len(x) != len(y):
        raise RuntimeError(f"Feature/label length mismatch for {model_name}: x={len(x)} y={len(y)}.")

    # index where validation begins
    val_idx = int(len(x) * (1 - config.VALIDATION_FRAC))
    if val_idx <= 0 or val_idx >= len(x):
        raise RuntimeError(f"Invalid split (val_idx={val_idx}) for data length={len(x)}. Adjust config.VALIDATION_FRAC or provide more data.")
    x_train = x[:val_idx]
    x_val = x[val_idx:]
    y_train = y[:val_idx]
    y_val = y[val_idx:]

    # ensure training labels contain at least two classes
    unique_train_labels = set(int(v) for v in pd.Series(y_train).astype(int).unique())
    if len(unique_train_labels) < 2:
        raise RuntimeError(f"Not enough class diversity in training labels for {model_name}. Labels found: {unique_train_labels}.")

    best_estimator: Optional[Any] = None
    best_manifest: manifest_mod.Manifest = None

    # run multiple independent training attempts
    for train_attempt in range(1, config.MAX_INDICATOR_TRAINING_ATTEMPTS + 1):
        # build model (pipeline) and train
        estimator = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=_N_ESTIMATORS, n_jobs=_N_JOBS))])
        estimator.fit(x_train, y_train)

        # evaluate
        accuracy, precision, recall, score = model_mod.evaluate_model(estimator=estimator, x_val=x_val, y_val=y_val)
        manifest = manifest_mod.Manifest(model_name=model_name, features=indicator_features, accuracy=accuracy, precision=precision, recall=recall, score=score)

        if best_manifest is None or manifest.is_better(best_manifest):
            best_estimator = estimator
            best_manifest = manifest

        if config.PRINT_DEBUG:
            print(f"Attempt {train_attempt}/{config.MAX_INDICATOR_TRAINING_ATTEMPTS} for {model_name} - metrics: accuracy={accuracy}, precision={precision}, recall={recall}, score={score}.")

    if best_estimator is None or best_manifest is None:
        raise RuntimeError("Training finished but no model was produced.")

    # build and save model with the best estimator and manifest
    model_mod.save_model(estimator=best_estimator, manifest=best_manifest)

    # OPTIONAL: print best estimator's trades (predictions on the validation slice)
    try:
        y_pred_best = best_estimator.predict(x_val)
        trade_mod.get_trades(sdf=indicator_sdf, y_pred=y_pred_best, val_idx=val_idx)
    except Exception:
        traceback.print_exc()
        raise
