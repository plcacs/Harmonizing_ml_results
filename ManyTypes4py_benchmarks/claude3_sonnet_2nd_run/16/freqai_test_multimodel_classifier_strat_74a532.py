import logging
from functools import reduce
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import talib.abstract as ta
from pandas import DataFrame
from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy

logger = logging.getLogger(__name__)

class freqai_test_multimodel_classifier_strat(IStrategy):
    """
    Test strategy - used for testing freqAI multimodel functionalities.
    DO not use in production.
    """
    minimal_roi: Dict[str, float] = {'0': 0.1, '240': -1}
    plot_config: Dict[str, Dict] = {'main_plot': {}, 'subplots': {'prediction': {'prediction': {'color': 'blue'}}, 'target_roi': {'target_roi': {'color': 'brown'}}, 'do_predict': {'do_predict': {'color': 'brown'}}}}
    process_only_new_candles: bool = True
    stoploss: float = -0.05
    use_exit_signal: bool = True
    startup_candle_count: int = 300
    can_short: bool = False
    linear_roi_offset: DecimalParameter = DecimalParameter(0.0, 0.02, default=0.005, space='sell', optimize=False, load=True)
    max_roi_time_long: IntParameter = IntParameter(0, 800, default=400, space='sell', optimize=False, load=True)

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int, metadata: Dict, **kwargs) -> DataFrame:
        dataframe['%-rsi-period'] = ta.RSI(dataframe, timeperiod=period)
        dataframe['%-mfi-period'] = ta.MFI(dataframe, timeperiod=period)
        dataframe['%-adx-period'] = ta.ADX(dataframe, timeperiod=period)
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        dataframe['%-pct-change'] = dataframe['close'].pct_change()
        dataframe['%-raw_volume'] = dataframe['volume']
        dataframe['%-raw_price'] = dataframe['close']
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        dataframe['%-day_of_week'] = dataframe['date'].dt.dayofweek
        dataframe['%-hour_of_day'] = dataframe['date'].dt.hour
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        dataframe['&s-up_or_down'] = np.where(dataframe['close'].shift(-50) > dataframe['close'], 'up', 'down')
        dataframe['&s-up_or_down2'] = np.where(dataframe['close'].shift(-50) > dataframe['close'], 'up2', 'down2')
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        self.freqai_info = self.config['freqai']
        dataframe = self.freqai.start(dataframe, metadata, self)
        dataframe['target_roi'] = dataframe['&-s_close_mean'] + dataframe['&-s_close_std'] * 1.25
        dataframe['sell_roi'] = dataframe['&-s_close_mean'] - dataframe['&-s_close_std'] * 1.25
        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: Dict) -> DataFrame:
        enter_long_conditions: List[Any] = [df['do_predict'] == 1, df['&-s_close'] > df['target_roi']]
        if enter_long_conditions:
            df.loc[reduce(lambda x, y: x & y, enter_long_conditions), ['enter_long', 'enter_tag']] = (1, 'long')
        enter_short_conditions: List[Any] = [df['do_predict'] == 1, df['&-s_close'] < df['sell_roi']]
        if enter_short_conditions:
            df.loc[reduce(lambda x, y: x & y, enter_short_conditions), ['enter_short', 'enter_tag']] = (1, 'short')
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: Dict) -> DataFrame:
        exit_long_conditions: List[Any] = [df['do_predict'] == 1, df['&-s_close'] < df['sell_roi'] * 0.25]
        if exit_long_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_conditions), 'exit_long'] = 1
        exit_short_conditions: List[Any] = [df['do_predict'] == 1, df['&-s_close'] > df['target_roi'] * 0.25]
        if exit_short_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_short_conditions), 'exit_short'] = 1
        return df
