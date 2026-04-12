from __future__ import annotations

import pandas as pd
import src.utils as utils

from abc import abstractmethod
from typing import Dict, List, Optional


class Indicator:
    def __init__(self, name: str, descriptions: List[str]):
        self.name = name
        self.descriptions = descriptions

    @abstractmethod
    def build_features_1_tf(self, sdf: pd.DataFrame) -> pd.DataFrame:
        pass

    def build_feature_names(self, sdf: pd.DataFrame, tf: str) -> pd.DataFrame:
        """
        Returns a sdf with any given indicator's column names converted into canonical names (features) for one tf.
        """
        features: Dict[str, str] = {}
        for col in list(sdf.columns):
            if col.startswith(self.name):
                parts = col.split("__")

                p: Optional[int] = None
                for part in parts:
                    if part.startswith("p="):
                        p = int(part.split("=", 1)[1])

                if p is None:
                    raise ValueError(f"Feature column {col} for indicator {self.name} missing 'p=' period information.")

                description: Optional[str] = None
                for d in self.descriptions:
                    if f"__{d}" in col:
                        description = d
                        break

                diff = "__diff" in col

                feature = self._build_feature_name(p=p, description=description, diff=diff, tf=tf)
            else:
                feature = f"{col}__tf={tf}"
            features[col] = feature

        return sdf.rename(columns=features)

    def _build_feature_name(self, p: int, description: Optional[str], diff: bool, tf: str) -> str:
        """
        Only source of truth for all canonical feature (columns) names.
        Format: <indicator.name>__p=<p>__<description>__diff__tf=<tf>
        """
        parts = [self.name, f"p={p}"]
        if description:
            parts.append(description)
        if diff:
            parts.append("diff")
        parts.append(f"tf={utils.sanitize_tf(tf=tf)}")
        return "__".join(parts)
