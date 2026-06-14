import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from freqtrade.configuration import TimeRange
from freqtrade.constants import Config

logger: logging.Logger

def _process_candles_and_indicators(
    pairlist: list[str],
    strategy_name: str,
    trades: pd.DataFrame,
    signal_candles: dict[str, dict[str, pd.DataFrame]],
    date_col: str = "open_date",
) -> dict[str, dict[str, pd.DataFrame]]: ...

def _analyze_candles_and_indicators(
    pair: str,
    trades: pd.DataFrame,
    signal_candles: pd.DataFrame,
    date_col: str = "open_date",
) -> pd.DataFrame: ...

def _do_group_table_output(
    bigdf: pd.DataFrame,
    glist: list[str],
    csv_path: Path | str,
    to_csv: bool = False,
) -> None: ...

def _do_rejected_signals_output(
    rejected_signals_df: pd.DataFrame,
    to_csv: bool = False,
    csv_path: Optional[Path | str] = None,
) -> None: ...

def _select_rows_within_dates(
    df: pd.DataFrame,
    timerange: Optional[TimeRange] = None,
    df_date_col: str = "date",
) -> pd.DataFrame: ...

def _select_rows_by_tags(
    df: pd.DataFrame,
    enter_reason_list: list[str],
    exit_reason_list: list[str],
) -> pd.DataFrame: ...

def prepare_results(
    analysed_trades: dict[str, dict[str, pd.DataFrame]],
    stratname: str,
    enter_reason_list: list[str],
    exit_reason_list: list[str],
    timerange: Optional[TimeRange] = None,
) -> pd.DataFrame: ...

def print_results(
    res_df: pd.DataFrame,
    exit_df: Optional[pd.DataFrame],
    analysis_groups: Optional[list[str]],
    indicator_list: list[str],
    entry_only: bool,
    exit_only: bool,
    csv_path: Path | str,
    rejected_signals: Optional[pd.DataFrame] = None,
    to_csv: bool = False,
) -> None: ...

def _merge_dfs(
    entry_df: pd.DataFrame,
    exit_df: Optional[pd.DataFrame],
    available_inds: list[str],
    entry_only: bool,
    exit_only: bool,
) -> pd.DataFrame: ...

def _print_table(
    df: pd.DataFrame,
    sortcols: Optional[list[str]] = None,
    *,
    show_index: bool = False,
    name: Optional[str] = None,
    to_csv: bool = False,
    csv_path: Path | str,
) -> None: ...

def process_entry_exit_reasons(config: Config) -> None: ...

def _generate_dfs(
    pairlist: list[str],
    enter_reason_list: list[str],
    exit_reason_list: list[str],
    signal_candles: dict[str, dict[str, pd.DataFrame]],
    strategy_name: str,
    timerange: Optional[TimeRange],
    trades: pd.DataFrame,
    date_col: str,
) -> pd.DataFrame: ...