import logging
from pathlib import Path
from typing import Any

import pandas as pd

from freqtrade.configuration import TimeRange
from freqtrade.constants import Config

logger: logging.Logger

def _process_candles_and_indicators(
    pairlist: list[str],
    strategy_name: str,
    trades: pd.DataFrame,
    signal_candles: dict[str, dict[str, pd.DataFrame]],
    date_col: str = 'open_date',
) -> dict[str, dict[str, pd.DataFrame]]: ...

def _analyze_candles_and_indicators(
    pair: str,
    trades: pd.DataFrame,
    signal_candles: pd.DataFrame,
    date_col: str = 'open_date',
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
    csv_path: Path | str | None = None,
) -> None: ...

def _select_rows_within_dates(
    df: pd.DataFrame,
    timerange: TimeRange | None = None,
    df_date_col: str = 'date',
) -> pd.DataFrame: ...

def _select_rows_by_tags(
    df: pd.DataFrame,
    enter_reason_list: list[str] | None,
    exit_reason_list: list[str] | None,
) -> pd.DataFrame: ...

def prepare_results(
    analysed_trades: dict[str, dict[str, pd.DataFrame]],
    stratname: str,
    enter_reason_list: list[str],
    exit_reason_list: list[str],
    timerange: TimeRange | None = None,
) -> pd.DataFrame: ...

def print_results(
    res_df: pd.DataFrame,
    exit_df: pd.DataFrame | None,
    analysis_groups: list[str] | None,
    indicator_list: list[str] | None,
    entry_only: bool,
    exit_only: bool,
    csv_path: Path | str,
    rejected_signals: pd.DataFrame | None = None,
    to_csv: bool = False,
) -> None: ...

def _merge_dfs(
    entry_df: pd.DataFrame,
    exit_df: pd.DataFrame | None,
    available_inds: list[str],
    entry_only: bool,
    exit_only: bool,
) -> pd.DataFrame: ...

def _print_table(
    df: pd.DataFrame,
    sortcols: list[str] | None = None,
    *,
    show_index: bool = False,
    name: str | None = None,
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
    timerange: TimeRange | None,
    trades: pd.DataFrame,
    date_col: str,
) -> pd.DataFrame: ...