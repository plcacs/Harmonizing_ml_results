from typing import Any, Literal
import numpy as np
from pandas import DataFrame, Series, concat, to_datetime
from freqtrade.constants import BACKTEST_BREAKDOWNS, DATETIME_PRINT_FORMAT
from freqtrade.data.metrics import calculate_cagr, calculate_calmar, calculate_csum, calculate_expectancy, calculate_market_change, calculate_max_drawdown, calculate_sharpe, calculate_sortino
from freqtrade.ft_types import BacktestResultType
from freqtrade.util import decimals_per_coin, fmt_coin, get_dry_run_wallet

def generate_trade_signal_candles(preprocessed_df: dict[str, DataFrame], bt_results: dict[str, Any], date_col: str) -> dict[str, DataFrame]:
    ...

def generate_rejected_signals(preprocessed_df: dict[str, DataFrame], rejected_dict: dict[str, list[tuple]]) -> dict[str, DataFrame]:
    ...

def _generate_result_line(result: DataFrame, starting_balance: float, first_column: str) -> dict[str, Any]:
    ...

def calculate_trade_volume(trades_dict: list[dict[str, Any]]) -> float:
    ...

def generate_pair_metrics(pairlist: list[str], stake_currency: str, starting_balance: float, results: DataFrame, skip_nan: bool = False) -> list[dict[str, Any]]:
    ...

def generate_tag_metrics(tag_type: Union[str, list[str]], starting_balance: float, results: DataFrame, skip_nan: bool = False) -> list[dict[str, Any]]:
    ...

def generate_strategy_comparison(bt_stats: dict[str, DataFrame]) -> list[dict[str, Any]]:
    ...

def _get_resample_from_period(period: str) -> str:
    ...

def generate_periodic_breakdown_stats(trade_list: Union[list[dict[str, Any]], DataFrame], period: str) -> list[dict[str, Any]]:
    ...

def generate_all_periodic_breakdown_stats(trade_list: Union[list[dict[str, Any]], DataFrame]) -> dict[str, list[dict[str, Any]]]:
    ...

def calc_streak(dataframe: DataFrame) -> tuple[int, int]:
    ...

def generate_trading_stats(results: DataFrame) -> dict[str, Any]:
    ...

def generate_daily_stats(results: DataFrame) -> dict[str, Any]:
    ...

def generate_strategy_stats(pairlist: list[str], strategy: str, content: dict[str, Any], min_date: datetime, max_date: datetime, market_change: float, is_hyperopt: bool = False) -> dict[str, Any]:
    ...

def generate_backtest_stats(btdata: dict[str, DataFrame], all_results: dict[str, dict[str, Any]], min_date: datetime, max_date: datetime) -> dict[str, Any]:
    ...
