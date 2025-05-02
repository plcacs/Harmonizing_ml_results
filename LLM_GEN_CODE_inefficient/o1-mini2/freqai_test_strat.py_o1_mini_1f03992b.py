import logging
from functools import reduce
from typing import Any, Dict

import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy


logger = logging.getLogger(__name__)


class freqai_test_strat(IStrategy):
    """
    Test strategy - used for testing freqAI functionalities.
    DO not use in production.
    """

    minimal_roi: Dict[str, float] = {"0": 0.1, "240": -1}

    plot_config: Dict[str, Any] = {
        "main_plot": {},
        "subplots": {
            "prediction": {"prediction": {"color": "blue"}},
            "target_roi": {
                "target_roi": {"color": "brown"},
            },
            "do_predict": {
                "do_predict": {"color": "brown"},
            },
        },
    }

    process_only_new_candles: bool = True
    stoploss: float = -0.05
    use_exit_signal: bool = True
    startup_candle_count: int = 300
    can_short: bool = False

    linear_roi_offset: DecimalParameter = DecimalParameter(
        0.00, 0.02, default=0.005, space="sell", optimize=False, load=True
    )
    max_roi_time_long: IntParameter = IntParameter(
        0, 800, default=400, space="sell", optimize=False, load=True
    )

    freqai_info: Dict[str, Any]

    def feature_engineering_expand_all(
        self,
        dataframe: DataFrame,
        period: int,
        metadata: Dict[str, Any],
        **kwargs: Any
    ) -> DataFrame:
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)

        return dataframe

    def feature_engineering_expand_basic(
        self,
        dataframe: DataFrame,
        metadata: Dict[str, Any],
        **kwargs: Any
    ) -> DataFrame:
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]

        return dataframe

    def feature_engineering_standard(
        self,
        dataframe: DataFrame,
        metadata: Dict[str, Any],
        **kwargs: Any
    ) -> DataFrame:
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour

        return dataframe

    def set_freqai_targets(
        self,
        dataframe: DataFrame,
        metadata: Dict[str, Any],
        **kwargs: Any
    ) -> DataFrame:
        label_period_candles: int = self.freqai_info["feature_parameters"]["label_period_candles"]
        dataframe["&-s_close"] = (
            dataframe["close"]
            .shift(-label_period_candles)
            .rolling(label_period_candles)
            .mean()
            / dataframe["close"]
            - 1
        )

        return dataframe

    def populate_indicators(
        self,
        dataframe: DataFrame,
        metadata: Dict[str, Any]
    ) -> DataFrame:
        self.freqai_info = self.config["freqai"]

        dataframe = self.freqai.start(dataframe, metadata, self)

        dataframe["target_roi"] = dataframe["&-s_close_mean"] + dataframe["&-s_close_std"] * 1.25
        dataframe["sell_roi"] = dataframe["&-s_close_mean"] - dataframe["&-s_close_std"] * 1.25
        return dataframe

    def populate_entry_trend(
        self,
        df: DataFrame,
        metadata: Dict[str, Any]
    ) -> DataFrame:
        enter_long_conditions: list[Any] = [
            df["do_predict"] == 1,
            df["&-s_close"] > df["target_roi"]
        ]

        if all(enter_long_conditions):
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
            ] = (1, "long")

        enter_short_conditions: list[Any] = [
            df["do_predict"] == 1,
            df["&-s_close"] < df["sell_roi"]
        ]

        if all(enter_short_conditions):
            df.loc[
                reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
            ] = (1, "short")

        return df

    def populate_exit_trend(
        self,
        df: DataFrame,
        metadata: Dict[str, Any]
    ) -> DataFrame:
        exit_long_conditions: list[Any] = [
            df["do_predict"] == 1,
            df["&-s_close"] < df["sell_roi"] * 0.25
        ]
        if all(exit_long_conditions):
            df.loc[
                reduce(lambda x, y: x & y, exit_long_conditions), "exit_long"
            ] = 1

        exit_short_conditions: list[Any] = [
            df["do_predict"] == 1,
            df["&-s_close"] > df["target_roi"] * 0.25
        ]
        if all(exit_short_conditions):
            df.loc[
                reduce(lambda x, y: x & y, exit_short_conditions), "exit_short"
            ] = 1

        return df
