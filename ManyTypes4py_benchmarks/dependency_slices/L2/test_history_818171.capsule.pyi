from typing import Any

# === Internal dependency: freqtrade.configuration.TimeRange ===
parse_timerange: Any

# === Internal dependency: freqtrade.constants ===
DATETIME_PRINT_FORMAT: str

# === Internal dependency: freqtrade.data.converter ===
# re-export: from freqtrade.data.converter.converter import ohlcv_to_dataframe

# === Internal dependency: freqtrade.data.history ===
# re-export: from .datahandlers import get_datahandler

# === Internal dependency: freqtrade.data.history.datahandlers.jsondatahandler ===
class JsonDataHandler(IDataHandler):
    ...
class JsonGzDataHandler(JsonDataHandler):

# === Internal dependency: freqtrade.data.history.history_utils ===
def load_pair_history(pair: str, timeframe: str, datadir: Path, *, timerange: TimeRange | None = ..., fill_up_missing: bool = ..., drop_incomplete: bool = ..., startup_candles: int = ..., data_format: str | None = ..., data_handler: IDataHandler | None = ..., candle_type: CandleType = ...) -> DataFrame: ...
def load_data(datadir: Path, timeframe: str, pairs: list[str], *, timerange: TimeRange | None = ..., fill_up_missing: bool = ..., startup_candles: int = ..., fail_without_data: bool = ..., data_format: str = ..., candle_type: CandleType = ..., user_futures_funding_rate: int | None = ...) -> dict[str, DataFrame]: ...
def refresh_data(*, datadir: Path, timeframe: str, pairs: list[str], exchange: Exchange, data_format: str | None = ..., timerange: TimeRange | None = ..., candle_type: CandleType) -> None: ...
def _load_cached_data_for_updating(pair: str, timeframe: str, timerange: TimeRange | None, data_handler: IDataHandler, candle_type: CandleType, prepend: bool = ...) -> tuple[DataFrame, int | None, int | None]: ...
def _download_pair_history(pair: str, *, datadir: Path, exchange: Exchange, timeframe: str = ..., new_pairs_days: int = ..., data_handler: IDataHandler | None = ..., timerange: TimeRange | None = ..., candle_type: CandleType, erase: bool = ..., prepend: bool = ..., pair_candles: DataFrame | None = ...) -> bool: ...
def refresh_backtest_ohlcv_data(exchange: Exchange, *, pairs: list[str], timeframes: list[str], datadir: Path, trading_mode: str, timerange: TimeRange | None = ..., new_pairs_days: int = ..., erase: bool = ..., data_format: str | None = ..., prepend: bool = ..., progress_tracker: CustomProgress | None = ..., candle_types: list[CandleType] | None = ..., no_parallel_download: bool = ...) -> list[str]: ...
def _download_trades_history(exchange: Exchange, pair: str, *, new_pairs_days: int = ..., timerange: TimeRange | None = ..., data_handler: IDataHandler, trading_mode: TradingMode) -> bool: ...
def refresh_backtest_trades_data(exchange: Exchange, pairs: list[str], datadir: Path, timerange: TimeRange, trading_mode: TradingMode, new_pairs_days: int = ..., erase: bool = ..., data_format: str = ..., progress_tracker: CustomProgress | None = ...) -> list[str]: ...
def get_timerange(data: dict[str, DataFrame]) -> tuple[datetime, datetime]: ...
def validate_backtest_data(data: DataFrame, pair: str, min_date: datetime, max_date: datetime, timeframe_min: int) -> bool: ...

# === Internal dependency: freqtrade.enums.CandleType ===
SPOT: Any
from_string: Any

# === Internal dependency: freqtrade.enums.TradingMode ===
SPOT: Any

# === Internal dependency: freqtrade.exchange ===
# re-export: from freqtrade.exchange.exchange_utils_timeframe import timeframe_to_minutes

# === Internal dependency: freqtrade.misc ===
def file_dump_json(filename: Path, data: Any, is_zip: bool = ..., log: bool = ...) -> None: ...

# === Internal dependency: freqtrade.resolvers ===
# re-export: from freqtrade.resolvers.strategy_resolver import StrategyResolver

# === Internal dependency: freqtrade.util ===
# re-export: from freqtrade.util.datetime_helpers import dt_ts
# re-export: from freqtrade.util.datetime_helpers import dt_utc

# === Third-party dependency: pandas ===
# Used symbols: DataFrame

# === Third-party dependency: pandas.testing ===
# Used symbols: assert_frame_equal

# === Third-party dependency: pytest ===
# Used symbols: mark, raises

# === Internal dependency: tests.conftest ===
def log_has(line, logs) -> Any: ...
def log_has_re(line, logs) -> Any: ...
def patch_exchange(mocker, api_mock = ..., exchange = ..., mock_markets = ..., mock_supported_modes = ...) -> None: ...
def get_patched_exchange(mocker, config, api_mock = ..., exchange = ..., mock_markets = ..., mock_supported_modes = ...) -> Exchange: ...
CURRENT_TEST_STRATEGY: str
EXMS: str