import logging
from typing import Dict, List, Any

import numpy as np  # noqa
import pandas as pd  # noqa
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import IntParameter, IStrategy, merge_informative_pair  # noqa


logger = logging.getLogger(__name__)


class FreqaiExampleHybridStrategy(IStrategy):
    """
    Example of a hybrid FreqAI strat, designed to illustrate how a user may employ
    FreqAI to bolster a typical Freqtrade strategy.
    """
    minimal_roi: Dict[str, float] = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.04,
    }

    plot_config: Dict[str, Any] = {
        "main_plot": {
            "tema": {},
        },
        "subplots": {
            "MACD": {
                "macd": {"color": "blue"},
                "macdsignal": {"color": "orange"},
            },
            "RSI": {
                "rsi": {"color": "red"},
            },
            "Up_or_down": {
                "&s-up_or_down": {"color": "green"},
            },
        },
    }

    process_only_new_candles: bool = True
    stoploss: float = -0.05
    use_exit_signal: bool = True
    startup_candle_count: int = 30
    can_short: bool = True

    buy_rsi: IntParameter = IntParameter(low=1, high=50, default=30, space="buy", optimize=True, load=True)
    sell_rsi: IntParameter = IntParameter(low=50, high=100, default=70, space="sell", optimize=True, load=True)
    short_rsi: IntParameter = IntParameter(low=51, high=100, default=70, space="sell", optimize=True, load=True)
    exit_short_rsi: IntParameter = IntParameter(low=1, high=50, default=30, space="buy", optimize=True, load=True)

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: Dict[str, Any], **kwargs
    ) -> DataFrame:
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-sma-period"] = ta.SMA(dataframe, timeperiod=period)
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=period, stds=2.2
        )
        dataframe["bb_lowerband-period"] = bollinger["lower"]
        dataframe["bb_middleband-period"] = bollinger["mid"]
        dataframe["bb_upperband-period"] = bollinger["upper"]

        dataframe["%-bb_width-period"] = (
            dataframe["bb_upperband-period"] - dataframe["bb_lowerband-period"]
        ) / dataframe["bb_middleband-period"]
        dataframe["%-close-bb_lower-period"] = dataframe["close"] / dataframe["bb_lowerband-period"]

        dataframe["%-roc-period"] = ta.ROC(dataframe, timeperiod=period)

        dataframe["%-relative_volume-period"] = (
            dataframe["volume"] / dataframe["volume"].rolling(period).mean()
        )

        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: Dict[str, Any], **kwargs
    ) -> DataFrame:
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: Dict[str, Any], **kwargs
    ) -> DataFrame:
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict[str, Any], **kwargs) -> DataFrame:
        self.freqai.class_names = ["down", "up"]
        dataframe["&s-up_or_down"] = np.where(
            dataframe["close"].shift(-50) > dataframe["close"], "up", "down"
        )

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)

        dataframe["rsi"] = ta.RSI(dataframe)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]
        dataframe["bb_percent"] = (dataframe["close"] - dataframe["bb_lowerband"]) / (
            dataframe["bb_upperband"] - dataframe["bb_lowerband"]
        )
        dataframe["bb_width"] = (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe[
            "bb_middleband"
        ]

        dataframe["tema"] = ta.TEMA(dataframe, timeperiod=9)

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        df.loc[
            (
                (qtpylib.crossed_above(df["rsi"], self.buy_rsi.value))
                & (df["tema"] <= df["bb_middleband"])
                & (df["tema"] > df["tema"].shift(1))
                & (df["volume"] > 0)
                & (df["do_predict"] == 1)
                & (df["&s-up_or_down"] == "up")
            ),
            "enter_long",
        ] = 1

        df.loc[
            (
                (qtpylib.crossed_above(df["rsi"], self.short_rsi.value))
                & (df["tema"] > df["bb_middleband"])
                & (df["tema"] < df["tema"].shift(1))
                & (df["volume"] > 0)
                & (df["do_predict"] == 1)
                & (df["&s-up_or_down"] == "down")
            ),
            "enter_short",
        ] = 1

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        df.loc[
            (
                (qtpylib.crossed_above(df["rsi"], self.sell_rsi.value))
                & (df["tema"] > df["bb_middleband"])
                & (df["tema"] < df["tema"].shift(1))
                & (df["volume"] > 0)
            ),
            "exit_long",
        ] = 1

        df.loc[
            (
                (qtpylib.crossed_above(df["rsi"], self.exit_short_rsi.value))
                & (df["tema"] <= df["bb_middleband"])
                & (df["tema"] > df["tema"].shift(1))
                & (df["volume"] > 0)
            ),
            "exit_short",
        ] = 1

        return df
