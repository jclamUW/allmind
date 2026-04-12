from __future__ import annotations

import math
import os
import config as config
import src.ml.meta_features as meta_features_mod
import src.modules.manifest as manifest_mod
import src.modules.model as model_mod
import src.utils as utils
import traceback
import warnings

from typing import Generator, List, Tuple


warnings.filterwarnings("ignore", module="sklearn")
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning")


_N_SPLITS: int = 5


def train(symbol: str, primary_tf: str) -> None:
    """
    Trains and saves a meta model using the meta_sdf.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import ParameterGrid

    meta_model_name = utils.get_model_name(symbol=symbol, tf=primary_tf, indicator=None)

    # build meta dataset
    meta_sdf = meta_features_mod.build_features(symbol=symbol, primary_tf=primary_tf)
    meta_sdf = meta_sdf.dropna(how="any")
    if meta_sdf.empty:
        raise RuntimeError("meta_sdf empty after dropping NaNs.")

    # reset index to ensure positional slicing (walk-forward uses integer positions)
    meta_sdf = meta_sdf.reset_index(drop=True)

    features = [feature for feature in meta_sdf.columns if feature != "target"]
    x = meta_sdf[features].values
    y = meta_sdf["target"].values.astype(int)

    # ensure we can do at least one walk-forward split
    splits = list(_walk_forward(n_rows=len(meta_sdf)))
    if not splits:
        raise RuntimeError("Not enough rows to create walk-forward splits. Adjust VAL_FRAC or N_SPLITS in config.")

    best_estimator = None
    best_score = -math.inf
    best_manifest = None
    best_param = None

    # simple grid for C regularization
    params = {"clf__C": [0.01, 0.1, 1.0, 10.0]}
    for param in ParameterGrid(params):
        accs: List[float] = []
        precs: List[float] = []
        recs: List[float] = []
        scores: List[float] = []

        for train_idx, val_idx in splits:
            x_train = x[train_idx]
            y_train = y[train_idx]
            x_val = x[val_idx]
            y_val = y[val_idx]

            estimator = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=param["clf__C"], max_iter=2000))])
            estimator.fit(x_train, y_train)
            acc, prec, rec, score = model_mod.evaluate_model(estimator=estimator, x_val=x_val, y_val=y_val)
            accs.append(float(acc))
            precs.append(float(prec))
            recs.append(float(rec))
            scores.append(float(score))

        avg_acc = round(float(sum(accs) / len(accs)) if accs else 0.0, config.DECIMALS_ROUNDING)
        avg_prec = round(float(sum(precs) / len(precs)) if precs else 0.0, config.DECIMALS_ROUNDING)
        avg_rec = round(float(sum(recs) / len(recs)) if recs else 0.0, config.DECIMALS_ROUNDING)
        avg_score = round(float(sum(scores) / len(scores)) if scores else ((avg_acc + avg_prec + avg_rec) / 3.0), config.DECIMALS_ROUNDING)

        if avg_score > best_score:
            best_score = avg_score
            best_manifest = manifest_mod.Manifest(model_name=meta_model_name, features=list(features), accuracy=avg_acc, precision=avg_prec, recall=avg_rec, score=avg_score)
            best_param = param.copy()

     # after grid search, ensure we found something
    if best_param is None or best_manifest is None:
        raise RuntimeError("No meta parameter configuration produced a valid model.")

    # build final estimator with the best param and fit on all data
    best_estimator = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=best_param["clf__C"], max_iter=2000))])
    best_estimator.fit(x, y)

    # save best model
    model_mod.save_model(estimator=best_estimator, manifest=best_manifest)

    # persist the meta training CSV into the newly-saved model folder (so it isn't overwritten)
    try:
        meta_csv = utils.get_meta_csv(meta_model_name=meta_model_name)
        meta_csv.parent.mkdir(parents=True, exist_ok=True)
        meta_sdf.to_csv(meta_csv, index=False)
    except Exception:
        traceback.print_exc()
        raise


def _walk_forward(n_rows: int) -> Generator[Tuple[List[int], List[int]], None, None]:
    """
    Generate (idx_train, idx_true) index ranges for walk-forward.
    """
    if not isinstance(n_rows, int) or n_rows < 1:
        return

    min_train = max(2, int(n_rows * (1.0 - config.VALIDATION_FRAC)))
    remaining = n_rows - min_train
    if remaining <= 0:
        return
    test_size = max(1, int(remaining / _N_SPLITS))
    starts: List[Tuple[int, int, int, int]] = []

    for i in range(_N_SPLITS):
        train_end = min_train + i * test_size
        start_true = train_end
        end_true = min(n_rows, start_true + test_size)
        if start_true >= end_true:
            break
        starts.append((0, train_end, start_true, end_true))
    for (a, b, c, d) in starts:
        idx_train = list(range(a, b))
        idx_true = list(range(c, d))
        yield idx_train, idx_true
