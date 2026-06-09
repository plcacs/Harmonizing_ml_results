from typing import Any

# === Internal dependency: freqtrade.commands.optimize_commands ===
def setup_optimize_configuration(args: dict[str, Any], method: RunMode) -> dict[str, Any]: ...
def start_backtesting(args: dict[str, Any]) -> None: ...

# === Internal dependency: freqtrade.configuration.TimeRange ===
parse_timerange: Any

# === Internal dependency: freqtrade.constants ===
BACKTEST_CACHE_AGE: Any

# === Internal dependency: freqtrade.data.btanalysis ===
def evaluate_result_multi(results: pd.DataFrame, timeframe: str, max_open_trades: IntOrInf) -> pd.DataFrame: ...
BT_DATA_COLUMNS: Any

# === Internal dependency: freqtrade.data.converter ===
# re-export: from freqtrade.data.converter.converter import clean_ohlcv_dataframe
# re-export: from freqtrade.data.converter.converter import ohlcv_fill_up_missing_data

# === Internal dependency: freqtrade.data.dataprovider ===
class DataProvider: ...

# === Internal dependency: freqtrade.data.history ===
# re-export: from .history_utils import get_timerange
# re-export: from .history_utils import load_data
# re-export: from .history_utils import load_pair_history

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
# re-export: from freqtrade.exchange.exchange_utils_timeframe import timeframe_to_next_date
# re-export: from freqtrade.exchange.exchange_utils_timeframe import timeframe_to_prev_date

# === Internal dependency: freqtrade.optimize.backtest_caching ===
def get_strategy_run_id(strategy) -> str: ...
def get_backtest_metadata_filename(filename: Path | str) -> Path: ...

# === Internal dependency: freqtrade.optimize.backtesting ===
class Backtesting:
    def __init__(self, config: Config, exchange: Exchange | None = ...) -> None: ...
    def _set_strategy(self, strategy: IStrategy) -> Any: ...
    def load_bt_data_detail(self) -> None: ...
    def check_abort(self) -> Any: ...
    def _check_trade_exit(self, trade: LocalTrade, row: tuple, current_time: datetime) -> LocalTrade | None: ...
    def _enter_trade(self, pair: str, row: tuple, direction: LongShort, stake_amount: float | None = ..., trade: LocalTrade | None = ..., requested_rate: float | None = ..., requested_stake: float | None = ..., entry_tag1: str | None = ...) -> LocalTrade | None: ...
    def backtest(self, processed: dict, start_date: datetime, end_date: datetime) -> dict[str, Any]: ...
    def start(self) -> None: ...

# === Internal dependency: freqtrade.persistence ===
# re-export: from freqtrade.persistence.trade_model import LocalTrade
# re-export: from freqtrade.persistence.trade_model import Trade

# === Internal dependency: freqtrade.resolvers ===
# re-export: from freqtrade.resolvers.strategy_resolver import StrategyResolver

# === Internal dependency: freqtrade.util.datetime_helpers ===
def dt_utc(year: int, month: int, day: int, hour: int = ..., minute: int = ..., second: int = ..., microsecond: int = ...) -> datetime: ...

# === Third-party dependency: numpy ===
# Used symbols: sin, where, zeros

# === Third-party dependency: pandas ===
# Used symbols: DataFrame, Timestamp, testing, to_datetime

# === Third-party dependency: pytest ===
# Used symbols: approx, mark, raises

# === Internal dependency: tests.conftest ===
def log_has(line, logs) -> Any: ...
def log_has_re(line, logs) -> Any: ...
def get_args(args) -> Any: ...
def generate_test_data(timeframe: str, size: int, start: str = ..., random_seed = ...) -> Any: ...
def patched_configuration_load_config_file(mocker, config) -> None: ...
def patch_exchange(mocker, api_mock = ..., exchange = ..., mock_markets = ..., mock_supported_modes = ...) -> None: ...
CURRENT_TEST_STRATEGY: str
EXMS: str