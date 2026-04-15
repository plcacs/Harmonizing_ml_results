import logging
from pathlib import Path
from typing import Any, Optional, Union, List, Dict

import pandas as pd
from freqtrade.configuration import TimeRange
from freqtrade.constants import Config
from freqtrade.exceptions import ConfigurationError, OperationalException

logger: logging.Logger = ...

def _process_candles_and_indicators(
    pairlist: List[str],
    strategy_name: str,
    trades: pd.DataFrame,
    signal_candles: Dict[str, Dict[str, pd.DataFrame]],
    date_col: str = 'open_date'
) -> Dict[str, Dict[str, pd.DataFrame]]: ...

def _analyze_candles_and_indicators(
    pair: str,
    trades: pd.DataFrame,
    signal_candles: pd.DataFrame,
    date_col: str = 'open_date'
) -> pd.DataFrame: ...

def _do_group_table_output(
    bigdf: pd.DataFrame,
    glist: List[str],
    csv_path: Path,
    to_csv: bool = False
) -> None: ...

def _do_rejected_signals_output(
    rejected_signals_df: pd.DataFrame,
    to_csv: bool = False,
    csv_path: Optional[Path] = None
) -> None: ...

def _select_rows_within_dates(
    df: pd.DataFrame,
    timerange: Optional[TimeRange] = None,
    df_date_col: str = 'date'
) -> pd.DataFrame: ...

def _select_rows_by_tags(
    df: pd.DataFrame,
    enter_reason_list: List[str],
    exit_reason_list: List[str]
) -> pd.DataFrame: ...

def prepare_results(
    analysed_trades: Dict[str, Dict[str, pd.DataFrame]],
    stratname: str,
    enter_reason_list: List[str],
    exit_reason_list: List[str],
    timerange: Optional[TimeRange] = None
) -> pd.DataFrame: ...

def print_results(
    res_df: pd.DataFrame,
    exit_df: pd.DataFrame,
    analysis_groups: List[str],
    indicator_list: List[str],
    entry_only: bool,
    exit_only: bool,
    csv_path: Path,
    rejected_signals: Optional[pd.DataFrame] = None,
    to_csv: bool = False
) -> None: ...

def _merge_dfs(
    entry_df: pd.DataFrame,
    exit_df: pd.DataFrame,
    available_inds: List[str],
    entry_only: bool,
    exit_only: bool
) -> pd.DataFrame: ...

def _print_table(
    df: pd.DataFrame,
    sortcols: Optional[List[str]] = None,
    *,
    show_index: bool = False,
    name: Optional[str] = None,
    to_csv: bool = False,
    csv_path: Path
) -> None: ...

def process_entry_exit_reasons(config: Config) -> None: ...

def _generate_dfs(
    pairlist: List[str],
    enter_reason_list: List[str],
    exit_reason_list: List[str],
    signal_candles: Dict[str, Dict[str, pd.DataFrame]],
    strategy_name: str,
    timerange: Optional[TimeRange],
    trades: pd.DataFrame,
    date_col: str
) -> pd.DataFrame: ...