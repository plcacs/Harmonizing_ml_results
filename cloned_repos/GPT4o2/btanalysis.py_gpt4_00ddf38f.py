```python
from typing import Union, Optional, List, Dict

def get_latest_optimize_filename(directory: Union[Path, str], variant: str) -> str:
    ...

def get_latest_backtest_filename(directory: Union[Path, str]) -> str:
    ...

def get_latest_hyperopt_filename(directory: Union[Path, str]) -> str:
    ...

def get_latest_hyperopt_file(directory: Union[Path, str], predef_filename: Optional[str] = None) -> Path:
    ...

def load_backtest_metadata(filename: Union[Path, str]) -> Dict[str, Any]:
    ...

def load_backtest_stats(filename: Union[Path, str]) -> BacktestResultType:
    ...

def load_and_merge_backtest_result(strategy_name: str, filename: Path, results: Dict[str, Any]):
    ...

def _get_backtest_files(dirname: Path) -> List[Path]:
    ...

def _extract_backtest_result(filename: Path) -> List[BacktestHistoryEntryType]:
    ...

def get_backtest_result(filename: Path) -> List[BacktestHistoryEntryType]:
    ...

def get_backtest_resultlist(dirname: Path) -> List[BacktestHistoryEntryType]:
    ...

def delete_backtest_result(file_abs: Path):
    ...

def update_backtest_metadata(filename: Path, strategy: str, content: Dict[str, Any]):
    ...

def get_backtest_market_change(filename: Path, include_ts: bool = True) -> pd.DataFrame:
    ...

def find_existing_backtest_stats(
    dirname: Union[Path, str], run_ids: Dict[str, str], min_backtest_date: Optional[datetime] = None
) -> Dict[str, Any]:
    ...

def _load_backtest_data_df_compatibility(df: pd.DataFrame) -> pd.DataFrame:
    ...

def load_backtest_data(filename: Union[Path, str], strategy: Optional[str] = None) -> pd.DataFrame:
    ...

def load_file_from_zip(zip_path: Path, filename: str) -> bytes:
    ...

def load_backtest_analysis_data(backtest_dir: Path, name: str):
    ...

def load_rejected_signals(backtest_dir: Path):
    ...

def load_signal_candles(backtest_dir: Path):
    ...

def load_exit_signal_candles(backtest_dir: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    ...

def analyze_trade_parallelism(results: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    ...

def evaluate_result_multi(
    results: pd.DataFrame, timeframe: str, max_open_trades: IntOrInf
) -> pd.DataFrame:
    ...

def trade_list_to_dataframe(trades: List[Union[Trade, LocalTrade]]) -> pd.DataFrame:
    ...

def load_trades_from_db(db_url: str, strategy: Optional[str] = None) -> pd.DataFrame:
    ...

def load_trades(
    source: str,
    db_url: str,
    exportfilename: Path,
    no_trades: bool = False,
    strategy: Optional[str] = None,
) -> pd.DataFrame:
    ...

def extract_trades_of_period(
    dataframe: pd.DataFrame, trades: pd.DataFrame, date_index: bool = False
) -> pd.DataFrame:
    ...
```