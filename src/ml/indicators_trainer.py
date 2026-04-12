"""
Module to handle ml training for multiple indicator.
Trains all indicators against multiple tfs.
"""

from __future__ import annotations

import config as config
import src.ml.indicator_trainer as indicator_trainer_mod

from typing import Tuple


def train(symbol: str) -> None:
    """
    Train all indicator models.
    """
    tfs: Tuple[str, ...] = (config.PRIMARY_TF,) + tuple(config.SECONDARY_TFS)
    for indicator in config.INDICATORS:
        indicator_trainer_mod.train(symbol=symbol, tfs=tfs, indicator=indicator)
