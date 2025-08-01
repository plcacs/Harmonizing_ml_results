import logging
from functools import reduce
import numpy as np
import talib.abstract as ta
from pandas import DataFrame
from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy

logger = logging.getLogger(__name__)

class freqai_test_classifier(IStrategy):
    """
    Test strategy - used for testing freqAI functionalities.
    DO not use in production.
    """
    minimal_roi: dict = {'0': 0.1, '240': -1}
    plot_config: dict = {'main_plot': {}, 'subplots': {'prediction': {'prediction': {'color': 'blue'}}, 'target_roi': {'target_roi': {'color': 'brown'}}, 'do_predict': {'do_predict': {'color': 'brown'}}}}
    process_only_new_candles: bool = True
    stoploss: float = -0.05
    use_exit_signal: bool = True
    startup_candle_count: int = 300
    can_short: bool = False
    linear_roi_offset: DecimalParameter = DecimalParameter(0.0, 0.02, default=0.005, space='sell', optimize=False, load=True)
    max_roi_time_long: IntParameter = IntParameter(0, 800, default=400, space='sell', optimize=False, load=True)

    def informative_pairs(self) -> list:
        whitelist_pairs: list = self.dp.current_whitelist()
        corr_pairs: list = self.config['freqai']['feature_parameters']['include_corr_pairlist']
        informative_pairs: list = []
        for tf in self.config['freqai']['feature_parameters']['include_timeframes']:
            for pair in whitelist_pairs:
                informative_pairs.append((pair, tf))
            for pair in corr_pairs:
                if pair in whitelist_pairs:
                    continue
                informative_pairs.append((pair, tf))
        return informative_pairs

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int, metadata: dict, **kwargs) -> DataFrame:
        dataframe['%-rsi-period'] = ta.RSI(dataframe, timeperiod=period)
        dataframe['%-mfi-period'] = ta.MFI(dataframe, timeperiod=period)
        dataframe['%-adx-period'] = ta.ADX(dataframe, timeperiod=period)
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        dataframe['%-pct-change'] = dataframe['close'].pct_change()
        dataframe['%-raw_volume'] = dataframe['volume']
        dataframe['%-raw_price'] = dataframe['close']
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        dataframe['%-day_of_week'] = dataframe['date'].dt.dayofweek
        dataframe['%-hour_of_day'] = dataframe['date'].dt.hour
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        self.freqai.class_names = ['down', 'up']
        dataframe['&s-up_or_down'] = np.where(dataframe['close'].shift(-100) > dataframe['close'], 'up', 'down')
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self.freqai_info = self.config['freqai']
        dataframe = self.freqai.start(dataframe, metadata, self)
        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        enter_long_conditions: list = [df['&s-up_or_down'] == 'up']
        if enter_long_conditions:
            df.loc[reduce(lambda x, y: x & y, enter_long_conditions), ['enter_long', 'enter_tag']] = (1, 'long')
        enter_short_conditions: list = [df['&s-up_or_down'] == 'down']
        if enter_short_conditions:
            df.loc[reduce(lambda x, y: x & y, enter_short_conditions), ['enter_short', 'enter_tag']] = (1, 'short')
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        return df
