"""
The strategies here are minimal strategies designed to fail loading in certain conditions.
They are not operational, and don't aim to be.
"""
from datetime import datetime
from pandas import DataFrame
from freqtrade.persistence.trade_model import Order
from typing import Dict, Any
from freqtrade.strategy.interface import IStrategy

class TestStrategyNoImplements(IStrategy):
    """
    Minimal strategy without implementing any methods.
    """
    def populate_indicators(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        return super().populate_indicators(dataframe, metadata)

class TestStrategyNoImplementSell(TestStrategyNoImplements):
    """
    Minimal strategy without implementing the sell method.
    """
    def populate_entry_trend(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        return super().populate_entry_trend(dataframe, metadata)

class TestStrategyImplementCustomSell(TestStrategyNoImplementSell):
    """
    Minimal strategy that implements a custom sell method.
    """
    def populate_exit_trend(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        return super().populate_exit_trend(dataframe, metadata)

    def custom_sell(self, pair: str, trade: Order, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Dict[str, Any]) -> bool:
        return False

class TestStrategyImplementBuyTimeout(TestStrategyNoImplementSell):
    """
    Minimal strategy that implements a buy timeout method.
    """
    def populate_exit_trend(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        return super().populate_exit_trend(dataframe, metadata)

    def check_buy_timeout(self, pair: str, trade: Order, order: Order, current_time: datetime, **kwargs: Dict[str, Any]) -> bool:
        return False

class TestStrategyImplementSellTimeout(TestStrategyNoImplementSell):
    """
    Minimal strategy that implements a sell timeout method.
    """
    def populate_exit_trend(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        return super().populate_exit_trend(dataframe, metadata)

    def check_sell_timeout(self, pair: str, trade: Order, order: Order, current_time: datetime, **kwargs: Dict[str, Any]) -> bool:
        return False
