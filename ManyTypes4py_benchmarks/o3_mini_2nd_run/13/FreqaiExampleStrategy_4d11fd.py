import logging
from functools import reduce
from typing import Any, Dict
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib
from freqtrade.strategy import IStrategy

logger = logging.getLogger(__name__)


class FreqaiExampleStrategy(IStrategy):
    """
    Example strategy showing how the user connects their own
    IFreqaiModel to the strategy.

    Warning! This is a showcase of functionality,
    which means that it is designed to show various functions of FreqAI
    and it runs on all computers. We use this showcase to help users
    understand how to build a strategy, and we use it as a benchmark
    to help debug possible problems.

    This means this is *not* meant to be run live in production.
    """
    minimal_roi: Dict[str, float] = {'0': 0.1, '240': -1}
    plot_config: Dict[str, Any] = {'main_plot': {}, 'subplots': {'&-s_close': {'&-s_close': {'color': 'blue'}}, 'do_predict': {'do_predict': {'color': 'brown'}}}}
    process_only_new_candles: bool = True
    stoploss: float = -0.05
    use_exit_signal: bool = True
    startup_candle_count: int = 40
    can_short: bool = True

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int, metadata: Dict[str, Any], **kwargs: Any) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        This function will automatically expand the defined features on the config defined
        `indicator_periods_candles`, `include_timeframes`, `include_shifted_candles`, and
        `include_corr_pairs`. In other words, a single feature defined in this function
        will automatically expand to a total of
        `indicator_periods_candles` * `include_timeframes` * `include_shifted_candles` *
        `include_corr_pairs` numbers of features added to the model.

        All features must be prepended with `%` to be recognized by FreqAI internals.
        """
        dataframe['%-rsi-period'] = ta.RSI(dataframe, timeperiod=period)
        dataframe['%-mfi-period'] = ta.MFI(dataframe, timeperiod=period)
        dataframe['%-adx-period'] = ta.ADX(dataframe, timeperiod=period)
        dataframe['%-sma-period'] = ta.SMA(dataframe, timeperiod=period)
        dataframe['%-ema-period'] = ta.EMA(dataframe, timeperiod=period)
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=period, stds=2.2)
        dataframe['bb_lowerband-period'] = bollinger['lower']
        dataframe['bb_middleband-period'] = bollinger['mid']
        dataframe['bb_upperband-period'] = bollinger['upper']
        dataframe['%-bb_width-period'] = (dataframe['bb_upperband-period'] - dataframe['bb_lowerband-period']) / dataframe['bb_middleband-period']
        dataframe['%-close-bb_lower-period'] = dataframe['close'] / dataframe['bb_lowerband-period']
        dataframe['%-roc-period'] = ta.ROC(dataframe, timeperiod=period)
        dataframe['%-relative_volume-period'] = dataframe['volume'] / dataframe['volume'].rolling(period).mean()
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: Dict[str, Any], **kwargs: Any) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        This function will automatically expand the defined features on the config defined
        `include_timeframes`, `include_shifted_candles`, and `include_corr_pairs`.
        In other words, a single feature defined in this function
        will automatically expand to a total of
        `include_timeframes` * `include_shifted_candles` * `include_corr_pairs`
        numbers of features added to the model.
        """
        dataframe['%-pct-change'] = dataframe['close'].pct_change()
        dataframe['%-raw_volume'] = dataframe['volume']
        dataframe['%-raw_price'] = dataframe['close']
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: Dict[str, Any], **kwargs: Any) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        This optional function will be called once with the dataframe of the base timeframe.
        This is the final function to be called, which means that the dataframe entering this
        function will contain all the features and columns created by all other
        freqai_feature_engineering_* functions.
        """
        dataframe['%-day_of_week'] = dataframe['date'].dt.dayofweek
        dataframe['%-hour_of_day'] = dataframe['date'].dt.hour
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict[str, Any], **kwargs: Any) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        Required function to set the targets for the model.
        All targets must be prepended with `&` to be recognized by the FreqAI internals.
        """
        dataframe['&-s_close'] = (
            dataframe['close'].shift(-self.freqai_info['feature_parameters']['label_period_candles'])
            .rolling(self.freqai_info['feature_parameters']['label_period_candles'])
            .mean() / dataframe['close'] - 1
        )
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)
        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        enter_long_conditions = [df['do_predict'] == 1, df['&-s_close'] > 0.01]
        if enter_long_conditions:
            df.loc[reduce(lambda x, y: x & y, enter_long_conditions), ['enter_long', 'enter_tag']] = (1, 'long')
        enter_short_conditions = [df['do_predict'] == 1, df['&-s_close'] < -0.01]
        if enter_short_conditions:
            df.loc[reduce(lambda x, y: x & y, enter_short_conditions), ['enter_short', 'enter_tag']] = (1, 'short')
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        exit_long_conditions = [df['do_predict'] == 1, df['&-s_close'] < 0]
        if exit_long_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_conditions), 'exit_long'] = 1
        exit_short_conditions = [df['do_predict'] == 1, df['&-s_close'] > 0]
        if exit_short_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_short_conditions), 'exit_short'] = 1
        return df

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: Any,
        entry_tag: str,
        side: str,
        **kwargs: Any
    ) -> bool:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()
        if side == 'long':
            if rate > last_candle['close'] * (1 + 0.0025):
                return False
        elif rate < last_candle['close'] * (1 - 0.0025):
            return False
        return True