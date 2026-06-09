from typing import Any

# === Internal dependency: freqtrade.constants ===
LAST_BT_RESULT_FN: str

# === Internal dependency: freqtrade.exceptions ===
class OperationalException(FreqtradeException): ...
class ConfigurationError(OperationalException): ...

# === Internal dependency: freqtrade.exchange ===
# re-export: from freqtrade.exchange.exchange_utils_timeframe import timeframe_to_resample_freq

# === Internal dependency: freqtrade.ft_types ===
# re-export: from freqtrade.ft_types.backtest_result_type import BacktestHistoryEntryType
# re-export: from freqtrade.ft_types.backtest_result_type import BacktestResultType

# === Internal dependency: freqtrade.misc ===
def file_dump_json(filename: Path, data: Any, is_zip: bool = ..., log: bool = ...) -> None: ...
def json_load(datafile: TextIO) -> Any: ...

# === Internal dependency: freqtrade.optimize.backtest_caching ===
def get_backtest_metadata_filename(filename: Path | str) -> Path: ...

# === Internal dependency: freqtrade.persistence ===
# re-export: from freqtrade.persistence.models import init_db
# re-export: from freqtrade.persistence.trade_model import Trade

# === Third-party dependency: joblib ===
# Used symbols: load

# === Third-party dependency: numpy ===
# Used symbols: int64, repeat

# === Third-party dependency: pandas ===
# Used symbols: DataFrame, Series, concat, date_range, read_feather, to_datetime