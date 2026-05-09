from __future__ import annotations
from datetime import datetime
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    overload,
)

logger: Any = ...

@dataclass
class DrawDownResult:
    drawdown_abs: float = ...
    high_date: datetime = ...
    low_date: datetime = ...
    high_value: float = ...
    low_value: float = ...
    relative_account_drawdown: float = ...

def calculate_market_change(data: dict[str, pd.DataFrame], column: str = 'close') -> float: ...

def combine_dataframes_by_column(data: dict[str, pd.DataFrame], column: str = 'close') -> pd.DataFrame: ...

def combined_dataframes_with_rel_mean(
    data: dict[str, pd.DataFrame],
    fromdt: datetime,
    todt: datetime,
    column: str = 'close'
) -> pd.DataFrame: ...

def combine_dataframes_with_mean(data: dict[str, pd.DataFrame], column: str = 'close') -> pd.DataFrame: ...

def create_cum_profit(
    df: pd.DataFrame,
    trades: pd.DataFrame,
    col_name: str,
    timeframe: str
) -> pd.DataFrame: ...

def _calc_drawdown_series(
    profit_results: pd.DataFrame,
    date_col: str,
    value_col: str,
    starting_balance: float
) -> pd.DataFrame: ...

def calculate_underwater(
    trades: pd.DataFrame,
    date_col: str = 'close_date',
    value_col: str = 'profit_ratio',
    starting_balance: float = 0.0
) -> pd.DataFrame: ...

def calculate_max_drawdown(
    trades: pd.DataFrame,
    date_col: str = 'close_date',
    value_col: str = 'profit_abs',
    starting_balance: float = 0,
    relative: bool = False
) -> DrawDownResult: ...

def calculate_csum(trades: pd.DataFrame, starting_balance: float = 0) -> tuple[float, float]: ...

def calculate_cagr(
    days_passed: int,
    starting_balance: float,
    final_balance: float
) -> float: ...

def calculate_expectancy(trades: pd.DataFrame) -> tuple[float, float]: ...

def calculate_sortino(
    trades: pd.DataFrame,
    min_date: datetime,
    max_date: datetime,
    starting_balance: float
) -> float: ...

def calculate_sharpe(
    trades: pd.DataFrame,
    min_date: datetime,
    max_date: datetime,
    starting_balance: float
) -> float: ...

def calculate_calmar(
    trades: pd.DataFrame,
    min_date: datetime,
    max_date: datetime,
    starting_balance: float
) -> float: ...