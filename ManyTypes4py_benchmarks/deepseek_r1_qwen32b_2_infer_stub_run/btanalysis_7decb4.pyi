"""
Stub file for 'btanalysis_7decb4' module
"""

from __future__ import annotations
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    TYPE_CHECKING,
)
from pathlib import Path
from datetime import datetime
import pandas as pd
from freqtrade.trade import Trade

if TYPE_CHECKING:
    from freqtrade.optimize.backtest_result import BacktestResult

BT_DATA_COLUMNS = List[str]

def get_latest_optimize_filename(directory: Union[str, Path], variant: Literal['backtest', 'hyperopt']) -> str:
    ...

def get_latest_backtest_filename(directory: Union[str, Path]) -> str:
    ...

def get_latest_hyperopt_filename(directory: Union[str, Path]) -> str:
    ...

def get_latest_hyperopt_file(directory: Union[str, Path], predef_filename: Optional[str] = None) -> Path:
    ...

def load_backtest_metadata(filename: Union[str, Path]) -> Optional[Dict]:
    ...

def load_backtest_stats(filename: Union[str, Path]) -> Dict:
    ...

def load_and_merge_backtest_result(strategy_name: str, filename: Union[str, Path], results: Dict) -> None:
    ...

def _get_backtest_files(dirname: Union[str, Path]) -> List[Path]:
    ...

def _extract_backtest_result(filename: Union[str, Path]) -> List[Dict]:
    ...

def get_backtest_result(filename: Union[str, Path]) -> List[Dict]:
    ...

def get_backtest_resultlist(dirname: Union[str, Path]) -> List[Dict]:
    ...

def delete_backtest_result(file_abs: Union[str, Path]) -> None:
    ...

def update_backtest_metadata(filename: Union[str, Path], strategy: str, content: Dict) -> None:
    ...

def get_backtest_market_change(filename: Union[str, Path], include_ts: bool = True) -> pd.DataFrame:
    ...

def find_existing_backtest_stats(
    dirname: Union[str, Path],
    run_ids: Dict[str, str],
    min_backtest_date: Optional[datetime] = None
) -> Dict:
    ...

def load_backtest_data(filename: Union[str, Path], strategy: Optional[str] = None) -> pd.DataFrame:
    ...

def _load_backtest_data_df_compatibility(df: pd.DataFrame) -> pd.DataFrame:
    ...

def load_file_from_zip(zip_path: Union[str, Path], filename: str) -> bytes:
    ...

def load_backtest_analysis_data(backtest_dir: Union[str, Path], name: str) -> Union[Any, None]:
    ...

def load_rejected_signals(backtest_dir: Union[str, Path]) -> Union[Any, None]:
    ...

def load_signal_candles(backtest_dir: Union[str, Path]) -> Union[Any, None]:
    ...

def load_exit_signal_candles(backtest_dir: Union[str, Path]) -> Union[Any, None]:
    ...

def analyze_trade_parallelism(results: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    ...

def evaluate_result_multi(results: pd.DataFrame, timeframe: str, max_open_trades: int) -> pd.DataFrame:
    ...

def trade_list_to_dataframe(trades: List[Trade]) -> pd.DataFrame:
    ...

def load_trades_from_db(db_url: str, strategy: Optional[str] = None) -> pd.DataFrame:
    ...

def load_trades(
    source: Literal['DB', 'file'],
    db_url: str,
    exportfilename: str,
    no_trades: bool = False,
    strategy: Optional[str] = None
) -> pd.DataFrame:
    ...

def extract_trades_of_period(
    dataframe: pd.DataFrame,
    trades: pd.DataFrame,
    date_index: bool = False
) -> pd.DataFrame:
    ...