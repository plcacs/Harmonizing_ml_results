# === Internal dependency: freqtrade.constants ===
LAST_BT_RESULT_FN = '.last_result.json'

# === Internal dependency: freqtrade.exceptions ===
class OperationalException(FreqtradeException): ...
class ConfigurationError(OperationalException): ...

# === Internal dependency: freqtrade.exchange ===
from freqtrade.exchange.exchange_utils_timeframe import timeframe_to_resample_freq

# === Internal dependency: freqtrade.ft_types ===
from freqtrade.ft_types.backtest_result_type import BacktestHistoryEntryType
from freqtrade.ft_types.backtest_result_type import BacktestResultType

# === Internal dependency: freqtrade.misc ===
def file_dump_json(filename, data, is_zip=..., log=...): ...
def json_load(datafile): ...

# === Internal dependency: freqtrade.optimize.backtest_caching ===
def get_backtest_metadata_filename(filename): ...

# === Internal dependency: freqtrade.persistence ===
from freqtrade.persistence.models import init_db
from freqtrade.persistence.trade_model import Trade

# === Third-party dependency: joblib ===
# Used symbols: load

# === Third-party dependency: numpy ===
# Used symbols: int64, repeat

# === Third-party dependency: pandas ===
# Used symbols: DataFrame, Series, concat, date_range, read_feather, to_datetime