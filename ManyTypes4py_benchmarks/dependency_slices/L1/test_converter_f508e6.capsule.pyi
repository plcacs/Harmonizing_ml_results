from typing import Any

# === Internal dependency: freqtrade.configuration.timerange ===
class TimeRange:
    def __init__(self, starttype=..., stoptype=..., startts=..., stopts=...): ...

# === Internal dependency: freqtrade.data.converter ===
from freqtrade.data.converter.converter import convert_ohlcv_format
from freqtrade.data.converter.converter import ohlcv_fill_up_missing_data
from freqtrade.data.converter.converter import ohlcv_to_dataframe
from freqtrade.data.converter.converter import reduce_dataframe_footprint
from freqtrade.data.converter.converter import trim_dataframe
from freqtrade.data.converter.trade_converter import convert_trades_format
from freqtrade.data.converter.trade_converter import convert_trades_to_ohlcv
from freqtrade.data.converter.trade_converter import trades_df_remove_duplicates
from freqtrade.data.converter.trade_converter import trades_dict_to_list
from freqtrade.data.converter.trade_converter import trades_to_ohlcv

# === Internal dependency: freqtrade.data.history ===
from .history_utils import get_timerange
from .history_utils import load_data
from .history_utils import load_pair_history
from .history_utils import validate_backtest_data

# === Internal dependency: freqtrade.data.history.datahandlers.IDataHandler ===
create_dir_if_needed: Any

# === Internal dependency: freqtrade.enums.CandleType ===
FUTURES: Any
MARK: Any
SPOT: Any

# === Internal dependency: freqtrade.exchange ===
from freqtrade.exchange.exchange_utils_timeframe import timeframe_to_minutes
from freqtrade.exchange.exchange_utils_timeframe import timeframe_to_seconds

# === Third-party dependency: numpy ===
# Used symbols: float32, float64

# === Third-party dependency: pandas ===
# Used symbols: DataFrame, Timestamp, concat, to_timedelta

# === Third-party dependency: pandas.testing ===
# Used symbols: assert_frame_equal

# === Third-party dependency: pytest ===
# Used symbols: mark, raises

# === Internal dependency: tests.conftest ===
def log_has(line, logs): ...
def log_has_re(line, logs): ...
def generate_trades_history(n_rows, start_date=..., days=...): ...
def generate_test_data(timeframe, size, start=..., random_seed=...): ...

# === Internal dependency: tests.data.test_history ===
def _clean_test_file(file): ...