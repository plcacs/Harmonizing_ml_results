"""
Helpers when analyzing backtest data
"""
import logging
import zipfile
from copy import copy
from datetime import datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union
import numpy as np
import pandas as pd
from freqtrade.constants import LAST_BT_RESULT_FN, IntOrInf
from freqtrade.exceptions import ConfigurationError, OperationalException
from freqtrade.ft_types import BacktestHistoryEntryType, BacktestResultType
from freqtrade.persistence import LocalTrade, Trade

logger: logging.Logger = ...
BT_DATA_COLUMNS: list[str] = ...

def get_latest_optimize_filename(directory: Union[str, Path], variant: str) -> str:
    ...

def get_latest_backtest_filename(directory: Union[str, Path]) -> str:
    ...

def get_latest_hyperopt_filename(directory: Union[str, Path]) -> str:
    ...

def get_latest_hyperopt_file(directory: Union[str, Path], predef_filename: Optional[str] = None) -> Path:
    ...

def load_backtest_metadata(filename: Union[str, Path]) -> Optional[dict]:
    ...

def load_backtest_stats(filename: Union[str, Path]) -> dict:
    ...

def load_and_merge_backtest_result(strategy_name: str, filename: Union[str, Path], results: dict) -> None:
    ...

def _get_backtest_files(dirname: Path) -> List[Path]:
    ...

def _extract_backtest_result(filename: Union[str, Path]) -> List[dict]:
    ...

def get_backtest_result(filename: Union[str, Path]) -> List[dict]:
    ...

def get_backtest_resultlist(dirname: Path) -> List[dict]:
    ...

def delete_backtest_result(file_abs: Path) -> None:
    ...

def update_backtest_metadata(filename: Union[str, Path], strategy: str, content: dict) -> None:
    ...

def get_backtest_market_change(filename: Union[str, Path], include_ts: bool = True) -> pd.DataFrame:
    ...

def find_existing_backtest_stats(dirname: Union[str, Path], run_ids: dict, min_backtest_date: Optional[datetime] = None) -> dict:
    ...

def _load_backtest_data_df_compatibility(df: pd.DataFrame) -> pd.DataFrame:
    ...

def load_backtest_data(filename: Union[str, Path], strategy: Optional[str] = None) -> pd.DataFrame:
    ...

def load_file_from_zip(zip_path: Union[str, Path], filename: str) -> bytes:
    ...

def load_backtest_analysis_data(backtest_dir: Union[str, Path], name: str) -> Optional[Any]:
    ...

def load_rejected_signals(backtest_dir: Union[str, Path]) -> Optional[Any]:
    ...

def load_signal_candles(backtest_dir: Union[str, Path]) -> Optional[Any]:
    ...

def load_exit_signal_candles(backtest_dir: Union[str, Path]) -> Optional[Any]:
    ...

def analyze_trade_parallelism(results: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    ...

def evaluate_result_multi(results: pd.DataFrame, timeframe: str, max_open_trades: int) -> pd.DataFrame:
    ...

def trade_list_to_dataframe(trades: List[Trade]) -> pd.DataFrame:
    ...

def load_trades_from_db(db_url: str, strategy: Optional[str] = None) -> pd.DataFrame:
    ...

def load_trades(source: Literal['DB', 'file'], db_url: str, exportfilename: str, no_trades: bool = False, strategy: Optional[str] = None) -> pd.DataFrame:
    ...

def extract_trades_of_period(dataframe: pd.DataFrame, trades: pd.DataFrame, date_index: bool = False) -> pd.DataFrame:
    ...