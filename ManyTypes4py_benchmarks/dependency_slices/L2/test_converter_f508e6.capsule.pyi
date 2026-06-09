from typing import Any

# === Internal dependency: freqtrade.configuration.timerange ===
class TimeRange:
    def __init__(self, starttype: str | None = ..., stoptype: str | None = ..., startts: int = ..., stopts: int = ...) -> Any: ...

# === Internal dependency: freqtrade.data.converter ===
# re-export: from freqtrade.data.converter.converter import convert_ohlcv_format
# re-export: from freqtrade.data.converter.converter import ohlcv_fill_up_missing_data
# re-export: from freqtrade.data.converter.converter import ohlcv_to_dataframe
# re-export: from freqtrade.data.converter.converter import reduce_dataframe_footprint
# re-export: from freqtrade.data.converter.converter import trim_dataframe
# re-export: from freqtrade.data.converter.trade_converter import convert_trades_format
# re-export: from freqtrade.data.converter.trade_converter import convert_trades_to_ohlcv
# re-export: from freqtrade.data.converter.trade_converter import trades_df_remove_duplicates
# re-export: from freqtrade.data.converter.trade_converter import trades_dict_to_list
# re-export: from freqtrade.data.converter.trade_converter import trades_to_ohlcv

# === Internal dependency: freqtrade.data.history ===
# re-export: from .history_utils import get_timerange
# re-export: from .history_utils import load_data
# re-export: from .history_utils import load_pair_history
# re-export: from .history_utils import validate_backtest_data

# === Internal dependency: freqtrade.data.history.datahandlers.IDataHandler ===
create_dir_if_needed: Any

# === Internal dependency: freqtrade.enums.CandleType ===
FUTURES: Any
MARK: Any
SPOT: Any

# === Internal dependency: freqtrade.exchange ===
# re-export: from freqtrade.exchange.exchange_utils_timeframe import timeframe_to_minutes
# re-export: from freqtrade.exchange.exchange_utils_timeframe import timeframe_to_seconds

# === Third-party dependency: numpy ===
# Used symbols: float32, float64

# === Third-party dependency: pandas ===
# Used symbols: DataFrame, Timestamp, concat, to_timedelta

# === Third-party dependency: pandas.testing ===
# Used symbols: assert_frame_equal

# === Third-party dependency: pytest ===
# Used symbols: mark, raises

# === Internal dependency: tests.conftest ===
def log_has(line, logs) -> Any: ...
def log_has_re(line, logs) -> Any: ...
def generate_trades_history(n_rows, start_date: datetime | None = ..., days = ...) -> Any: ...
def generate_test_data(timeframe: str, size: int, start: str = ..., random_seed = ...) -> Any: ...

# === Internal dependency: tests.data.test_history ===
def _clean_test_file(file: Path) -> None: ...