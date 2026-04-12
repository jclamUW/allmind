"""
Module to contain a description/metadata of a trained model.
"""

from __future__ import annotations

import json
import numpy as np
import config as config
import src.utils as utils
import traceback

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Manifest:
    model_name: str
    features: List[str]
    accuracy: float         # market trend prediction correctness frac
    precision: float        # BUY signal correctness frac
    recall: float           # real upward moves caught correctness frac
    score: float            # the average of accuracy, precision, and recall

    def is_better(self, other: Optional[Manifest]) -> bool:
        """
        Compare based on score.
        """
        if other is None:
            return True
        return float(self.score) > float(other.score)

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts a Manifest object to a dictionary.
        Used for writing to json.
        """
        return {
            "model_name": self.model_name,
            "features": list(self.features or []),
            "accuracy": float(self.accuracy),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "score": float(self.score),
        }


def get_manifest_from_json(model_name: str) -> Manifest:
    """
    Loads the corresponding manifest json and converts it to a Manifest object.
    """
    manifest_json = utils.get_manifest_json(model_name=model_name)
    if not manifest_json.exists():
        raise RuntimeError(f"Invalid manifest json for model {model_name}.")

    try:
        return _to_manifest(json.loads(manifest_json.read_text(encoding="utf-8")))
    except Exception:
        traceback.print_exc()
        raise


def _to_manifest(data: Dict[str, Any]) -> Manifest:
    """
    Converts a dictionary to a Manifest object.
    Used for json data. 
    """
    return Manifest(
        model_name=data["model_name"],
        features=list(data.get("features", []) if "features" in data and data["features"] else []),
        accuracy=float(data.get("accuracy", 0.0)),
        precision=float(data.get("precision", 0.0)),
        recall=float(data.get("recall", 0.0)),
        score=float(data.get("score", 0.0)),
    )
