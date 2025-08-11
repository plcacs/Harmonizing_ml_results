"""
The strategies here are minimal strategies designed to fail loading in certain conditions.
They are not operational, and don't aim to be.
"""
from datetime import datetime
from pandas import DataFrame
from freqtrade.persistence.trade_model import Order
from freqtrade.strategy.interface import IStrategy

class TestStrategyNoImplements(IStrategy):

    def populate_indicators(self, dataframe: Union[pandas.DataFrame, dict], metadata: Union[pandas.DataFrame, dict]) -> Union[str, tuple[str], dict]:
        return super().populate_indicators(dataframe, metadata)

class TestStrategyNoImplementSell(TestStrategyNoImplements):

    def populate_entry_trend(self, dataframe: Union[pandas.DataFrame, dict], metadata: Union[pandas.DataFrame, dict]) -> Union[str, list[str], bool]:
        return super().populate_entry_trend(dataframe, metadata)

class TestStrategyImplementCustomSell(TestStrategyNoImplementSell):

    def populate_exit_trend(self, dataframe: Union[pandas.DataFrame, dict], metadata: Union[pandas.DataFrame, dict]) -> Union[datetime.datetime, None, bool, float]:
        return super().populate_exit_trend(dataframe, metadata)

    def custom_sell(self, pair: Union[float, str, datetime.datetime.timedelta], trade: Union[float, str, datetime.datetime.timedelta], current_time: Union[float, str, datetime.datetime.timedelta], current_rate: Union[float, str, datetime.datetime.timedelta], current_profit: Union[float, str, datetime.datetime.timedelta], **kwargs) -> bool:
        return False

class TestStrategyImplementBuyTimeout(TestStrategyNoImplementSell):

    def populate_exit_trend(self, dataframe: Union[pandas.DataFrame, dict], metadata: Union[pandas.DataFrame, dict]) -> Union[datetime.datetime, None, bool, float]:
        return super().populate_exit_trend(dataframe, metadata)

    def check_buy_timeout(self, pair: Union[float, str, datetime.datetime.timedelta], trade: Union[float, str, datetime.datetime.timedelta], order: Union[float, str, datetime.datetime.timedelta], current_time: Union[float, str, datetime.datetime.timedelta], **kwargs) -> bool:
        return False

class TestStrategyImplementSellTimeout(TestStrategyNoImplementSell):

    def populate_exit_trend(self, dataframe: Union[pandas.DataFrame, dict], metadata: Union[pandas.DataFrame, dict]) -> Union[datetime.datetime, None, bool, float]:
        return super().populate_exit_trend(dataframe, metadata)

    def check_sell_timeout(self, pair: Union[float, str, datetime.datetime.timedelta], trade: Union[float, str, datetime.datetime.timedelta], order: Union[float, str, datetime.datetime.timedelta], current_time: Union[float, str, datetime.datetime.timedelta], **kwargs) -> bool:
        return False