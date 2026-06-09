from typing import Any

# === Third-party dependency: filelock ===
# Used symbols: Timeout

# === Internal dependency: freqtrade.commands.optimize_commands ===
def setup_optimize_configuration(args, method): ...
def start_hyperopt(args): ...

# === Internal dependency: freqtrade.data.history ===
from .history_utils import load_data

# === Internal dependency: freqtrade.enums.ExitType ===
FORCE_EXIT: Any
ROI: Any
STOP_LOSS: Any

# === Internal dependency: freqtrade.enums.RunMode ===
HYPEROPT: Any

# === Internal dependency: freqtrade.exceptions ===
class OperationalException(FreqtradeException): ...

# === Internal dependency: freqtrade.optimize.hyperopt.Hyperopt ===
get_lock_filename: Any

# === Internal dependency: freqtrade.optimize.hyperopt.hyperopt_auto ===
class HyperOptAuto(IHyperOpt): ...

# === Internal dependency: freqtrade.optimize.hyperopt_tools ===
class HyperoptTools:
    ...

# === Internal dependency: freqtrade.optimize.optimize_reports ===
from freqtrade.optimize.optimize_reports.optimize_reports import generate_strategy_stats

# === Internal dependency: freqtrade.optimize.space ===
from .decimalspace import SKDecimal

# === Internal dependency: freqtrade.strategy ===
from freqtrade.strategy.parameters import IntParameter

# === Internal dependency: freqtrade.util ===
from freqtrade.util.datetime_helpers import dt_utc

# === Third-party dependency: pandas ===
# Used symbols: DataFrame

# === Third-party dependency: pytest ===
# Used symbols: mark, raises

# === Unresolved dependency: skopt.space ===
# Used unresolved symbols: Integer

# === Internal dependency: tests.conftest ===
def log_has(line, logs): ...
def log_has_re(line, logs): ...
def get_args(args): ...
def patched_configuration_load_config_file(mocker, config): ...
def patch_exchange(mocker, api_mock=..., exchange=..., mock_markets=..., mock_supported_modes=...): ...
def get_markets(): ...
CURRENT_TEST_STRATEGY = 'StrategyTestV3'
EXMS = 'freqtrade.exchange.exchange.Exchange'