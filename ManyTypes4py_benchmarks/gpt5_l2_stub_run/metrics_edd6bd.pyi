from datetime import datetime
import logging
from typing import Dict, Optional, Tuple, Union

from pandas import DataFrame, Timestamp

logger: logging.Logger = ...

def calculate_market_change(data: Dict[str, DataFrame], column: str = 'close') -> float: ...
def combine_dataframes_by_column(data: Dict[str, DataFrame], column: str = 'close') -> DataFrame: ...
def combined_dataframes_with_rel_mean(
    data: Dict[str, DataFrame],
    fromdt: Union[datetime, Timestamp],
    todt: Union[datetime, Timestamp],
    column: str = 'close',
) -> DataFrame: ...
def combine_dataframes_with_mean(data: Dict[str, DataFrame], column: str = 'close') -> DataFrame: ...
def create_cum_profit(df: DataFrame, trades: DataFrame, col_name: str, timeframe: str) -> DataFrame: ...
def _calc_drawdown_series(
    profit_results: DataFrame, *, date_col: str, value_col: str, starting_balance: float
) -> DataFrame: ...
def calculate_underwater(
    trades: DataFrame, *, date_col: str = 'close_date', value_col: str = 'profit_ratio', starting_balance: float = 0.0
) -> DataFrame: ...

class DrawDownResult:
    drawdown_abs: float
    high_date: Optional[Union[datetime, Timestamp]]
    low_date: Optional[Union[datetime, Timestamp]]
    high_value: float
    low_value: float
    relative_account_drawdown: float
    def __init__(
        self,
        drawdown_abs: float = 0.0,
        high_date: Optional[Union[datetime, Timestamp]] = None,
        low_date: Optional[Union[datetime, Timestamp]] = None,
        high_value: float = 0.0,
        low_value: float = 0.0,
        relative_account_drawdown: float = 0.0,
    ) -> None: ...

def calculate_max_drawdown(
    trades: DataFrame,
    *,
    date_col: str = 'close_date',
    value_col: str = 'profit_abs',
    starting_balance: float = 0,
    relative: bool = False,
) -> DrawDownResult: ...
def calculate_csum(trades: DataFrame, starting_balance: float = 0) -> Tuple[float, float]: ...
def calculate_cagr(days_passed: float, starting_balance: float, final_balance: float) -> float: ...
def calculate_expectancy(trades: DataFrame) -> Tuple[float, float]: ...
def calculate_sortino(
    trades: DataFrame,
    min_date: Optional[Union[datetime, Timestamp]],
    max_date: Optional[Union[datetime, Timestamp]],
    starting_balance: float,
) -> float: ...
def calculate_sharpe(
    trades: DataFrame,
    min_date: Optional[Union[datetime, Timestamp]],
    max_date: Optional[Union[datetime, Timestamp]],
    starting_balance: float,
) -> float: ...
def calculate_calmar(
    trades: DataFrame,
    min_date: Optional[Union[datetime, Timestamp]],
    max_date: Optional[Union[datetime, Timestamp]],
    starting_balance: float,
) -> float: ...