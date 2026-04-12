"""
Centralized module to hold shared functions for consistent string names, files, and folders.
Keep consistent naming structure throughout the entire application.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.modules.indicator import Indicator


CANDLES_PATH: Path = Path("property/candles")
META_MODELS_PATH: Path = Path("property/meta_models")
MODELS_PATH: Path = Path("property/models")
ANALYSIS_PATH: Path = Path("ANALYSIS.txt")

ESTIMATOR_FILENAME: str = "estimator.joblib"    # predicts
MANIFEST_FILENAME: str = "manifest.json"        # explains
META_CSV_FILENAME: str = "meta.csv"


def get_csv(symbol: str, tf: str) -> Path:
    """
    Returns the corresponding candles csv if it exists.
    """
    return CANDLES_PATH / _sanitize_str(str=symbol) / f"tf={tf}.csv"


def get_estimator_joblib(model_name: str) -> Path:
    """
    Returns the corresponding estimator joblib if it exists.
    """
    return get_model_folder(model_name=model_name) / ESTIMATOR_FILENAME


def get_manifest_json(model_name: str) -> Path:
    """
    Returns the corresponding manifest json if it exists.
    """
    return get_model_folder(model_name=model_name) / MANIFEST_FILENAME


def get_meta_csv(meta_model_name: str) -> Path:
    """
    Returns the corresponding meta model csv if it exists.
    """
    return META_MODELS_PATH / meta_model_name / META_CSV_FILENAME


def get_model_folder(model_name: str) -> Path:
    """
    Returns the corresponding model folder if it exists.
    """
    sanitized_model_name = _sanitize_str(str=model_name)
    if sanitized_model_name.endswith("__meta"):
        return META_MODELS_PATH / sanitized_model_name
    return MODELS_PATH / sanitized_model_name


def get_model_name(symbol: str, tf: str, indicator: Optional[Indicator]) -> str:
    """
    Returns the corresponding model name.
    If indicator is None, get meta model where tf is the primary tf.
    """
    if not indicator:
        return f"{_sanitize_str(str=symbol)}__tf={tf}__meta"
    return f"{_sanitize_str(str=symbol)}__tf={tf}__{indicator.name}"


def sanitize_tf(tf: str) -> str:
    """
    Converts a raw ccxt compatible tf (e.g. '1m','4h','1d') to the pandas offset alias.
    """
    sanitized_tf = tf.strip().lower()
    if len(sanitized_tf) >= 2 and sanitized_tf[:-1].isdigit():
        num = sanitized_tf[:-1]
        unit = sanitized_tf[-1]
        if unit == "m":
            return f"{num}min"
        if unit == "h":
            return f"{num}h"
        if unit == "d":
            return f"{num}d"
        if unit == "w":
            return f"{num}W"
    return sanitized_tf


def save_analysis(lines: list[str]) -> None:
    """
    Append an entry to ANALYSIS.txt.
    """
    ANALYSIS_PATH.touch(exist_ok=True)  # ensure file exists

    now = datetime.now().strftime("%m-%d-%Y %H:%M:%S")

    with open(ANALYSIS_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n========== {now} ==========\n")
        for line in lines:
            f.write(line + "\n")


def _sanitize_str(str: str) -> str:
    """
    Converts strings (ex. model_name or symbol) to a file friendly one.
    """
    return (
        str
        .replace("/", "-")
        .replace("\\", "-")
        .replace("..", "")
        .replace(" ", "_")
        .replace(":", "-")
    )
