import datetime
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, Optional, Union, overload

import numpy as np
import pandas as pd
from freqtrade.constants import IntOrInf
from freqtrade.ft_types import BacktestHistoryEntryType, BacktestResultType
from freqtrade.persistence import LocalTrade, Trade

logger: Any = ...

BT_DATA_COLUMNS: list[str] = ...

def get_latest_optimize_filename(
    directory: Union[str, Path], variant: Literal["backtest", "hyperopt"]
) -> str: ...

def get_latest_backtest_filename(directory: Union[str, Path]) -> str: ...

def get_latest_hyperopt_filename(directory: Union[str, Path]) -> str: ...

def get_latest_hyperopt_file(
    directory: Union[str, Path], predef_filename: Optional[str] = None
) -> Path: ...

def load_backtest_metadata(filename: Union[str, Path]) -> dict[str, Any]: ...

def load_backtest_stats(filename: Union[str, Path]) -> Union[dict[str, Any], list[Any]]: ...

def load_and_merge_backtest_result(
    strategy_name: str,
    filename: Union[str, Path],
    results: dict[str, dict[str, Any]],
) -> None: ...

def _get_backtest_files(dirname: Path) -> list[Path]: ...

def _extract_backtest_result(
    filename: Path,
) -> list[dict[str, Union[str, int, float, None]]]: ...

def get_backtest_result(filename: Union[str, Path]) -> list[dict[str, Union[str, int, float, None]]]: ...

def get_backtest_resultlist(dirname: Union[str, Path]) -> list[dict[str, Union[str, int, float, None]]]: ...

def delete_backtest_result(file_abs: Path) -> None: ...

def update_backtest_metadata(
    filename: Union[str, Path],
    strategy: str,
    content: dict[str, Any],
) -> None: ...

def get_backtest_market_change(
    filename: Path, include_ts: bool = True
) -> pd.DataFrame: ...

def find_existing_backtest_stats(
    dirname: Union[str, Path],
    run_ids: dict[str, str],
    min_backtest_date: Optional[datetime.datetime] = None,
) -> dict[str, Any]: ...

def _load_backtest_data_df_compatibility(df: pd.DataFrame) -> pd.DataFrame: ...

def load_backtest_data(
    filename: Union[str, Path], strategy: Optional[str] = None
) -> pd.DataFrame: ...

def load_file_from_zip(zip_path: Path, filename: str) -> bytes: ...

def load_backtest_analysis_data(
    backtest_dir: Path, name: str
) -> Optional[Any]: ...

def load_rejected_signals(backtest_dir: Path) -> Optional[Any]: ...

def load_signal_candles(backtest_dir: Path) -> Optional[Any]: ...

def load_exit_signal_candles(backtest_dir: Path) -> Optional[Any]: ...

def analyze_trade_parallelism(
    results: pd.DataFrame, timeframe: str
) -> pd.DataFrame: ...

def evaluate_result_multi(
    results: pd.DataFrame,
    timeframe: str,
    max_open_trades: Union[int, IntOrInf],
) -> pd.DataFrame: ...

def trade_list_to_dataframe(trades: list[Union[Trade, LocalTrade]]) -> pd.DataFrame: ...

def load_trades_from_db(
    db_url: str, strategy: Optional[str] = None
) -> pd.DataFrame: ...

@overload
def load_trades(
    source: Literal["DB"],
    db_url: str,
    exportfilename: Union[str, Path],
    no_trades: bool = False,
    strategy: Optional[str] = None,
) -> pd.DataFrame: ...

@overload
def load_trades(
    source: Literal["file"],
    db_url: str,
    exportfilename: Union[str, Path],
    no_trades: bool = False,
    strategy: Optional[str] = None,
) -> pd.DataFrame: ...

def load_trades(
    source: str,
    db_url: str,
    exportfilename: Union[str, Path],
    no_trades: bool = False,
    strategy: Optional[str] = None,
) -> pd.DataFrame: ...

def extract_trades_of_period(
    dataframe: pd.DataFrame,
    trades: pd.DataFrame,
    date_index: bool = False,
) -> pd.DataFrame: ...