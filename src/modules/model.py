"""
Module to store the actual trained estimator capable of making predictions.
"""

from __future__ import annotations

import config as config
import joblib
import json
import numpy as np
import shutil
import src.modules.manifest as manifest_mod
import src.utils as utils
import traceback

from dataclasses import dataclass
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from typing import Any, Tuple


@dataclass
class Model:
    estimator: Any
    manifest: manifest_mod.Manifest


def evaluate_model(estimator: Pipeline, x_val: np.ndarray, y_val: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Evaluates the given model from the estimator.
    Coerce inputs and outputs to arrays and return (accuracy, precision, recall, and score) numbers.
    """
    y_pred = estimator.predict(x_val)

    # ensure 1-D arrays (sklearn can be picky if passed column vectors)
    y_val = np.asarray(y_val).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_val.shape[0] != y_pred.shape[0]:
        raise ValueError("y_val and y_pred must have the same length.")

    # convert y_val to integer labels if they are floats but already 0/1-ish
    threshold = 0.5
    if np.issubdtype(y_val.dtype, np.floating):
        if np.isfinite(y_val).all() and ((y_val >= 0.0) & (y_val <= 1.0)).all():
            y_val = (y_val >= threshold).astype(int)
        else:
            raise ValueError("y_val contains floats that are not probabilities in [0,1]. Provide integer labels or probability floats.")
    y_val = _verify_binary_labels(arr=y_val)

    # handle y_pred: if float probabilities, threshold; otherwise cast to int
    if np.issubdtype(y_pred.dtype, np.floating):
        if not np.isfinite(y_pred).all():
            raise ValueError("y_pred contains non-finite floats.")

        if ((y_pred >= 0.0) & (y_pred <= 1.0)).all():
            y_pred = (y_pred >= threshold).astype(int)
        else:
            raise ValueError("y_pred contains float values that are not probabilities in [0,1]. Provide integer labels or probability floats.")
    else:
        y_pred = y_pred.astype(int)
    y_pred = _verify_binary_labels(arr=y_pred)

    accuracy = float(accuracy_score(y_val, y_pred))
    precision = float(precision_score(y_val, y_pred, zero_division=0))
    recall = float(recall_score(y_val, y_pred, zero_division=0))
    score = (accuracy + precision + recall) / 3.0

    return round(accuracy, config.DECIMALS_ROUNDING), round(precision, config.DECIMALS_ROUNDING), round(recall, config.DECIMALS_ROUNDING), round(score, config.DECIMALS_ROUNDING)


def load_model(model_name: str) -> Model:
    """
    Load the corresponding Model object.
    """
    estimator_joblib = utils.get_estimator_joblib(model_name=model_name)
    if not estimator_joblib.exists():
        raise RuntimeError(f"estimator joblib does not exist for model {model_name}.")

    payload = joblib.load(estimator_joblib)
    if payload is None:
        raise RuntimeError(f"Loaded payload is empty for model {model_name}.")

    estimator = payload
    if not any(hasattr(estimator, m) for m in ("predict_proba", "decision_function", "predict")):
        raise RuntimeError(f"Loaded estimator for model {model_name} provides no usable prediction methods (needs predict_proba, decision_function or predict).")

    manifest = manifest_mod.get_manifest_from_json(model_name=model_name)
    if hasattr(estimator, "n_features_in_") and manifest.features:
        if estimator.n_features_in_ != len(manifest.features):
            raise RuntimeError(f"Feature mismatch in loaded model {model_name}: estimator expects {estimator.n_features_in_} but manifest has {len(manifest.features)}")

    return Model(estimator=estimator, manifest=manifest)


def save_model(estimator: Any, manifest: manifest_mod.Manifest) -> None:
    """
    Save new model.
    Will only replace an existing saved model if the new manifest.score is strictly greater.
    """
    if not manifest:
        raise ValueError("manifest is invalid.")

    utils.MODELS_PATH.mkdir(parents=True, exist_ok=True)
    model_folder = utils.get_model_folder(model_name=manifest.model_name)
    model_exists = model_folder.exists()

    # build temporary model folder
    tmp_model_folder = model_folder.parent / f".tmp__{manifest.model_name}"
    if tmp_model_folder.exists():
        shutil.rmtree(tmp_model_folder)
    tmp_model_folder.mkdir(parents=True, exist_ok=True)

    # save temporary estimator joblib inside tmp_model_folder
    tmp_estimator_path = tmp_model_folder / utils.ESTIMATOR_FILENAME
    tmp_estimator_joblib = tmp_estimator_path.with_name(tmp_estimator_path.name + ".tmp")
    try:
        joblib.dump(estimator, tmp_estimator_joblib)
        tmp_estimator_joblib.replace(tmp_estimator_path)
    except Exception:
        # cleanup and re-raise useful error
        try:
            if tmp_model_folder.exists():
                shutil.rmtree(tmp_model_folder)
        except Exception:
            traceback.print_exc()
            raise
        traceback.print_exc()
        raise

    # save temporary manifest json inside tmp_model_folder
    tmp_manifest_path = tmp_model_folder / utils.MANIFEST_FILENAME
    tmp_manifest_json = tmp_manifest_path.with_name(tmp_manifest_path.name + ".tmp")
    try:
        tmp_manifest_json.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
        tmp_manifest_json.replace(tmp_manifest_path)
    except Exception:
        try:
            if tmp_model_folder.exists():
                shutil.rmtree(tmp_model_folder)
        except Exception:
            traceback.print_exc()
            raise
        traceback.print_exc()
        raise

    # if no existing model, move tmp_model_folder into place
    if not model_exists:
        tmp_model_folder.rename(model_folder)
        print(f"New model {manifest.model_name} saved (score={manifest.score}). No previous model.")
        return

    # load existing manifest
    try:
        existing_manifest = manifest_mod.get_manifest_from_json(manifest.model_name)
    except Exception:
        traceback.print_exc()
        raise

    # replace old model with new; else discard candidate
    if manifest.is_better(other=existing_manifest):
        try:
            if model_folder.exists():
                shutil.rmtree(model_folder)
        except Exception:
            traceback.print_exc()
            raise
        tmp_model_folder.rename(model_folder)
        print(f"New model {manifest.model_name} saved (score={manifest.score} > previous={existing_manifest.score}).")
    else:
        try:
            if tmp_model_folder.exists():
                shutil.rmtree(tmp_model_folder)
        except Exception:
            traceback.print_exc()
            raise
        print(f"New model {manifest.model_name} not saved (score={manifest.score} <= previous={existing_manifest.score}).")


def _verify_binary_labels(arr: np.ndarray) -> np.ndarray:
    """
    Verify array contains only 0/1 integer labels.
    Safety check.
    """
    arr = np.asarray(arr).ravel()
    if arr.size == 0:
        raise ValueError("arr is empty.")
    if not np.isfinite(arr).all():
        raise ValueError("arr contains non-finite values.")
    if np.issubdtype(arr.dtype, np.floating):
        raise ValueError("arr has floats. Expected integer 0/1 labels at this stage.")

    try:
        arr = arr.astype(int)
    except Exception:
        traceback.print_exc()
        raise

    unique = set(np.unique(arr))
    if not unique.issubset({0, 1}):
        raise ValueError(f"arr contains non-binary label values: {unique}. Expected only {{0,1}}.")
    return arr
