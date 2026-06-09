from typing import Any

# === Internal dependency: freqtrade.commands.optimize_commands ===
def setup_optimize_configuration(args, method): ...
def start_backtesting(args): ...

# === Internal dependency: freqtrade.configuration.TimeRange ===
parse_timerange: Any

# === Internal dependency: freqtrade.constants ===
BACKTEST_CACHE_AGE = ['none', 'day', 'week', 'month']

# === Internal dependency: freqtrade.data.btanalysis ===
def evaluate_result_multi(results, timeframe, max_open_trades): ...
BT_DATA_COLUMNS = ['pair', 'stake_amount', 'max_stake_amount', 'amount', 'open_date', 'close_date', 'open_rate', 'close_rate', ...]

# === Internal dependency: freqtrade.data.converter ===
from freqtrade.data.converter.converter import clean_ohlcv_dataframe
from freqtrade.data.converter.converter import ohlcv_fill_up_missing_data

# === Internal dependency: freqtrade.data.dataprovider ===
class DataProvider: ...

# === Internal dependency: freqtrade.data.history ===
from .history_utils import get_timerange
from .history_utils import load_data
from .history_utils import load_pair_history

# === Internal dependency: freqtrade.enums.CandleType ===
FUTURES: Any

# === Internal dependency: freqtrade.enums.ExitType ===
ROI: Any
STOP_LOSS: Any

# === Internal dependency: freqtrade.enums.RunMode ===
BACKTEST: Any

# === Internal dependency: freqtrade.exceptions ===
class OperationalException(FreqtradeException): ...
class DependencyException(FreqtradeException): ...

# === Internal dependency: freqtrade.exchange ===
from freqtrade.exchange.exchange_utils_timeframe import timeframe_to_next_date
from freqtrade.exchange.exchange_utils_timeframe import timeframe_to_prev_date

# === Internal dependency: freqtrade.optimize.backtest_caching ===
def get_strategy_run_id(strategy): ...
def get_backtest_metadata_filename(filename): ...

# === Internal dependency: freqtrade.optimize.backtesting ===
class Backtesting:
    def __init__(self, config, exchange=...): ...
    def _set_strategy(self, strategy): ...
    def load_bt_data_detail(self): ...
    def check_abort(self): ...
    def _check_trade_exit(self, trade, row, current_time): ...
    def _enter_trade(self, pair, row, direction, stake_amount=..., trade=..., requested_rate=..., requested_stake=..., entry_tag1=...): ...
    def backtest(self, processed, start_date, end_date): ...
    def start(self): ...

# === Internal dependency: freqtrade.persistence ===
from freqtrade.persistence.trade_model import LocalTrade
from freqtrade.persistence.trade_model import Trade

# === Internal dependency: freqtrade.resolvers ===
from freqtrade.resolvers.strategy_resolver import StrategyResolver

# === Internal dependency: freqtrade.util.datetime_helpers ===
def dt_utc(year, month, day, hour=..., minute=..., second=..., microsecond=...): ...

# === Third-party dependency: numpy ===
# Used symbols: sin, where, zeros

# === Third-party dependency: pandas ===
# Used symbols: DataFrame, Timestamp, testing, to_datetime

# === Third-party dependency: pytest ===
# Used symbols: approx, mark, raises

# === Internal dependency: tests.conftest ===
def log_has(line, logs): ...
def log_has_re(line, logs): ...
def get_args(args): ...
def generate_test_data(timeframe, size, start=..., random_seed=...): ...
def patched_configuration_load_config_file(mocker, config): ...
def patch_exchange(mocker, api_mock=..., exchange=..., mock_markets=..., mock_supported_modes=...): ...
CURRENT_TEST_STRATEGY = 'StrategyTestV3'
EXMS = 'freqtrade.exchange.exchange.Exchange'