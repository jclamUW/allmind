from __future__ import annotations

import config as config
import os
import src.ml.indicators_trainer as indicators_trainer_mod
import src.ml.meta_trainer as meta_trainer_mod
import src.ml.predict as predict_mod
import src.modules.candle as candle_mod
import src.utils as utils
import sys
import traceback


def _hard_stop(exctype, value, tb):
    traceback.print_exception(exctype, value, tb)
    os._exit(1)


sys.excepthook = _hard_stop


def main() -> int:
    utils.CANDLES_PATH.mkdir(parents=True, exist_ok=True)
    utils.META_MODELS_PATH.mkdir(parents=True, exist_ok=True)
    utils.MODELS_PATH.mkdir(parents=True, exist_ok=True)

    try:
        # update all candle csvs
        candle_mod.sync()

        for symbol in config.SYMBOLS:
            # train all indicator models
            indicators_trainer_mod.train(symbol=symbol)

            # train meta model
            meta_trainer_mod.train(symbol=symbol, primary_tf=config.PRIMARY_TF)

            # evaluate
            try:
                predict_mod.evaluate_present_and_future(symbol=symbol, primary_tf=config.PRIMARY_TF)
            except Exception:
                traceback.print_exc()
                raise
    except Exception:
        traceback.print_exc()
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
