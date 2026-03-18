```python
import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import pandas as pd
from freqtrade.configuration import TimeRange
from freqtrade.constants import Config
from freqtrade.exceptions import ConfigurationError, OperationalException

logger: logging.Logger = ...

def _process_candles_and_indicators(
    pairlist: Sequence[str],
    strategy_name: str,
    trades: pd.DataFrame,
    signal_candles: dict[str, dict[str, pd.DataFrame]],
    date_col: str = 'open_date'
) -> dict[str, dict[str, pd.DataFrame]]: ...

def _analyze_candles_and_indicators(
    pair: str,
    trades: pd.DataFrame,
    signal_candles: pd.DataFrame,
    date_col: str = 'open_date'
) -> pd.DataFrame: ...

def _do_group_table_output(
    bigdf: pd.DataFrame,
    glist: Sequence[str],
    csv_path: Union[str, Path],
    to_csv: bool = False
) -> None: ...

def _do_rejected_signals_output(
    rejected_signals_df: pd.DataFrame,
    to_csv: bool = False,
    csv_path: Optional[Union[str, Path]] = None
) -> None: ...

def _select_rows_within_dates(
    df: pd.DataFrame,
    timerange: Optional[TimeRange] = None,
    df_date_col: str = 'date'
) -> pd.DataFrame: ...

def _select_rows_by_tags(
    df: pd.DataFrame,
    enter_reason_list: Optional[Sequence[str]],
    exit_reason_list: Optional[Sequence[str]]
) -> pd.DataFrame: ...

def prepare_results(
    analysed_trades: dict[str, dict[str, pd.DataFrame]],
    stratname: str,
    enter_reason_list: Sequence[str],
    exit_reason_list: Sequence[str],
    timerange: Optional[TimeRange] = None
) -> pd.DataFrame: ...

def print_results(
    res_df: pd.DataFrame,
    exit_df: pd.DataFrame,
    analysis_groups: Sequence[str],
    indicator_list: Sequence[str],
    entry_only: bool,
    exit_only: bool,
    csv_path: Union[str, Path],
    rejected_signals: Optional[pd.DataFrame] = None,
    to_csv: bool = False
) -> None: ...

def _merge_dfs(
    entry_df: pd.DataFrame,
    exit_df: pd.DataFrame,
    available_inds: Sequence[str],
    entry_only: bool,
    exit_only: bool
) -> pd.DataFrame: ...

def _print_table(
    df: pd.DataFrame,
    sortcols: Optional[Sequence[str]] = None,
    *,
    show_index: bool = False,
    name: Optional[str] = None,
    to_csv: bool = False,
    csv_path: Union[str, Path]
) -> None: ...

def process_entry_exit_reasons(config: Config) -> None: ...

def _generate_dfs(
    pairlist: Sequence[str],
    enter_reason_list: Sequence[str],
    exit_reason_list: Sequence[str],
    signal_candles: dict[str, dict[str, pd.DataFrame]],
    strategy_name: str,
    timerange: TimeRange,
    trades: pd.DataFrame,
    date_col: str
) -> pd.DataFrame: ...
```