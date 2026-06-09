from typing import Any

# === Internal dependency: freqtrade.configuration.TimeRange ===
parse_timerange: Any

# === Internal dependency: freqtrade.constants ===
DATETIME_PRINT_FORMAT = '%Y-%m-%d %H:%M:%S'

# === Internal dependency: freqtrade.data.converter ===
from freqtrade.data.converter.converter import ohlcv_to_dataframe

# === Internal dependency: freqtrade.data.history ===
from .datahandlers import get_datahandler

# === Internal dependency: freqtrade.data.history.datahandlers.jsondatahandler ===
class JsonDataHandler(IDataHandler):
    ...
class JsonGzDataHandler(JsonDataHandler):

# === Internal dependency: freqtrade.data.history.history_utils ===
def load_pair_history(pair, timeframe, datadir, *, timerange=..., fill_up_missing=..., drop_incomplete=..., startup_candles=..., data_format=..., data_handler=..., candle_type=...): ...
def load_data(datadir, timeframe, pairs, *, timerange=..., fill_up_missing=..., startup_candles=..., fail_without_data=..., data_format=..., candle_type=..., user_futures_funding_rate=...): ...
def refresh_data(*, datadir, timeframe, pairs, exchange, data_format=..., timerange=..., candle_type): ...
def _load_cached_data_for_updating(pair, timeframe, timerange, data_handler, candle_type, prepend=...): ...
def _download_pair_history(pair, *, datadir, exchange, timeframe=..., new_pairs_days=..., data_handler=..., timerange=..., candle_type, erase=..., prepend=..., pair_candles=...): ...
def refresh_backtest_ohlcv_data(exchange, *, pairs, timeframes, datadir, trading_mode, timerange=..., new_pairs_days=..., erase=..., data_format=..., prepend=..., progress_tracker=..., candle_types=..., no_parallel_download=...): ...
def _download_trades_history(exchange, pair, *, new_pairs_days=..., timerange=..., data_handler, trading_mode): ...
def refresh_backtest_trades_data(exchange, pairs, datadir, timerange, trading_mode, new_pairs_days=..., erase=..., data_format=..., progress_tracker=...): ...
def get_timerange(data): ...
def validate_backtest_data(data, pair, min_date, max_date, timeframe_min): ...

# === Internal dependency: freqtrade.enums.CandleType ===
SPOT: Any
from_string: Any

# === Internal dependency: freqtrade.enums.TradingMode ===
SPOT: Any

# === Internal dependency: freqtrade.exchange ===
from freqtrade.exchange.exchange_utils_timeframe import timeframe_to_minutes

# === Internal dependency: freqtrade.misc ===
def file_dump_json(filename, data, is_zip=..., log=...): ...

# === Internal dependency: freqtrade.resolvers ===
from freqtrade.resolvers.strategy_resolver import StrategyResolver

# === Internal dependency: freqtrade.util ===
from freqtrade.util.datetime_helpers import dt_ts
from freqtrade.util.datetime_helpers import dt_utc

# === Third-party dependency: pandas ===
# Used symbols: DataFrame

# === Third-party dependency: pandas.testing ===
# Used symbols: assert_frame_equal

# === Third-party dependency: pytest ===
# Used symbols: mark, raises

# === Internal dependency: tests.conftest ===
def log_has(line, logs): ...
def log_has_re(line, logs): ...
def patch_exchange(mocker, api_mock=..., exchange=..., mock_markets=..., mock_supported_modes=...): ...
def get_patched_exchange(mocker, config, api_mock=..., exchange=..., mock_markets=..., mock_supported_modes=...): ...
CURRENT_TEST_STRATEGY = 'StrategyTestV3'
EXMS = 'freqtrade.exchange.exchange.Exchange'