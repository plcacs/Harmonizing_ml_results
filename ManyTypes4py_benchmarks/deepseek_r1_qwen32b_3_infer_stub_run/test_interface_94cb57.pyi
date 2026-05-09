import logging
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pytest
from pandas import DataFrame
from freqtrade.configuration import TimeRange
from freqtrade.constants import CUSTOM_TAG_MAX_LENGTH
from freqtrade.data.dataprovider import DataProvider
from freqtrade.data.history import load_data
from freqtrade.enums import ExitCheckTuple, ExitType, HyperoptState, SignalDirection
from freqtrade.exceptions import OperationalException, StrategyError
from freqtrade.optimize.hyperopt_tools import HyperoptStateContainer
from freqtrade.optimize.space import SKDecimal
from freqtrade.persistence import PairLocks, Trade
from freqtrade.resolvers import StrategyResolver
from freqtrade.strategy.hyper import detect_parameters
from freqtrade.strategy.parameters import (
    BaseParameter,
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
)
from freqtrade.util import dt_now

class StrategyTestV3:
    def __init__(self, config: Dict[str, Any]) -> None:
        ...
    
    def get_entry_signal(
        self,
        pair: str,
        timeframe: str,
        data: DataFrame,
        enter_tag: Optional[str] = None
    ) -> Union[Tuple[SignalDirection, Optional[str]], Tuple[None, None]]:
        ...
    
    def get_exit_signal(
        self,
        pair: str,
        timeframe: str,
        data: DataFrame,
        exit_tag: Optional[str] = None
    ) -> Union[Tuple[bool, bool, Optional[str]], Tuple[None, None]]:
        ...
    
    def analyze_pair(self, pair: str) -> None:
        ...
    
    def assert_df(
        self,
        dataframe: Optional[DataFrame],
        len_dataframe: int,
        last_close: float,
        last_date: datetime
    ) -> None:
        ...
    
    def disable_dataframe_checks(self) -> None:
        ...
    
    def populate_indicators(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        ...
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        ...
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        ...
    
    def advise_entry(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        ...
    
    def advise_exit(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        ...
    
    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        side: str,
        entry_tag: Optional[str] = None
    ) -> float:
        ...
    
    def is_pair_locked(self, pair: str, candle_date: Optional[datetime] = None) -> bool:
        ...
    
    def lock_pair(self, pair: str, until: datetime, reason: Optional[str] = None) -> None:
        ...
    
    def unlock_pair(self, pair: str, reason: Optional[str] = None) -> None:
        ...
    
    def unlock_reason(self, reason: str) -> None:
        ...
    
    def process_only_new_candles(self) -> bool:
        ...
    
    def gather_informative_pairs(self) -> List[Tuple[str, str]]:
        ...
    
    def get_latest_candle(
        self,
        pair: str,
        timeframe: str,
        data: Optional[DataFrame]
    ) -> Union[Tuple[None, None], Tuple[SignalDirection, Optional[str]]]:
        ...
    
    def get_signal(
        self,
        pair: str,
        timeframe: str,
        data: Optional[DataFrame]
    ) -> Union[Tuple[None, None], Tuple[SignalDirection, Optional[str]]]:
        ...
    
    def min_roi_reached(
        self,
        trade: Trade,
        current_profit: float,
        current_time: datetime
    ) -> bool:
        ...
    
    def ft_stoploss_reached(
        self,
        current_rate: float,
        trade: Trade,
        current_time: datetime,
        current_profit: float,
        force_stoploss: float,
        high: Optional[float] = None
    ) -> ExitCheckTuple:
        ...
    
    def should_exit(
        self,
        trade: Trade,
        current_rate: float,
        current_time: datetime,
        enter: bool,
        exit_: bool,
        low: Optional[float] = None,
        high: Optional[float] = None
    ) -> List[ExitCheckTuple]:
        ...
    
    def custom_exit(
        self,
        trade: Trade,
        current_rate: float,
        current_profit: float,
        current_time: datetime
    ) -> Union[bool, str, None]:
        ...
    
    def _analyze_ticker_internal(
        self,
        dataframe: DataFrame,
        metadata: Dict[str, Any]
    ) -> DataFrame:
        ...
    
    def analyze_ticker(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> None:
        ...
    
    def ignore_expired_candle(
        self,
        latest_date: datetime,
        current_time: datetime,
        timeframe_seconds: int,
        enter: bool
    ) -> bool:
        ...
    
    def advise_all_indicators(self, data: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
        ...
    
    def disable_dataframe_checks(self) -> None:
        ...
    
    def get_signal_empty(
        self,
        pair: str,
        timeframe: str,
        data: Optional[DataFrame]
    ) -> Tuple[None, None]:
        ...
    
    def get_signal_exception_valueerror(
        self,
        pair: str,
        timeframe: str,
        data: Optional[DataFrame]
    ) -> Tuple[None, None]:
        ...
    
    def get_signal_old_dataframe(
        self,
        pair: str,
        timeframe: str,
        data: Optional[DataFrame]
    ) -> Tuple[None, None]:
        ...
    
    def get_signal_no_sell_column(
        self,
        pair: str,
        timeframe: str,
        data: Optional[DataFrame]
    ) -> Tuple[SignalDirection, None]:
        ...
    
    def get_signal_exception_valueerror(
        self,
        pair: str,
        timeframe: str,
        data: Optional[DataFrame]
    ) -> Tuple[None, None]:
        ...
    
    def get_signal_old_dataframe(
        self,
        pair: str,
        timeframe: str,
        data: Optional[DataFrame]
    ) -> Tuple[None, None]:
        ...
    
    def get_signal_no_sell_column(
        self,
        pair: str,
        timeframe: str,
        data: Optional[DataFrame]
    ) -> Tuple[SignalDirection, None]:
        ...
    
    def get_signal_exception_valueerror(
        self,
        pair: str,
        timeframe: str,
        data: Optional[DataFrame]
    ) -> Tuple[None, None]:
        ...
    
    def get_signal_old_dataframe(
        self,
        pair: str,
        timeframe: str,
        data: Optional[DataFrame]
    ) -> Tuple[None, None]:
        ...
    
    def get_signal_no_sell_column(
        self,
        pair: str,
        timeframe: str,
        data: Optional[DataFrame]
    ) -> Tuple[SignalDirection, None]:
        ...
    
    def get_signal_exception_valueerror(
        self,
        pair: str,
        timeframe: str,
        data: Optional[DataFrame]
    ) -> Tuple[None, None]:
        ...
    
    def get_signal_old_dataframe(
        self,
        pair: str,
        timeframe: str,
        data: Optional[DataFrame]
    ) -> Tuple[None, None]:
        ...
    
    def get_signal_no_sell_column(
        self,
        pair: str,
        timeframe: str,
        data: Optional[DataFrame]
    ) -> Tuple[SignalDirection, None]:
        ...
    
    def get_signal_exception_valueerror(
        self,
        pair: str,
        timeframe: str,
        data: Optional[DataFrame]
    ) -> Tuple[None, None]:
        ...
    
    def get_signal_old_dataframe(
        self,
        pair: str,
        timeframe: str,
        data: Optional[DataFrame]
    ) -> Tuple[None, None]:
        ...
    
    def get_signal_no_sell_column(
        self,
        pair: str,
        timeframe: str,
        data: Optional[DataFrame]
    ) -> Tuple[SignalDirection, None]:
        ...