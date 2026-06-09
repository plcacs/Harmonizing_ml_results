from typing import Any

# === Internal dependency: freqtrade.configuration.TimeRange ===
parse_timerange: Any

# === Internal dependency: freqtrade.constants ===
CUSTOM_TAG_MAX_LENGTH = 255

# === Internal dependency: freqtrade.data.dataprovider ===
class DataProvider:
    def __init__(self, config, exchange, pairlists=..., rpc=...): ...

# === Internal dependency: freqtrade.data.history ===
from .history_utils import load_data

# === Internal dependency: freqtrade.enums ===
from freqtrade.enums.signaltype import SignalDirection

# === Internal dependency: freqtrade.enums.ExitType ===
CUSTOM_EXIT: Any
EXIT_SIGNAL: Any
LIQUIDATION: Any
NONE: Any
ROI: Any
STOP_LOSS: Any
TRAILING_STOP_LOSS: Any

# === Internal dependency: freqtrade.enums.HyperoptState ===
INDICATORS: Any
OPTIMIZE: Any

# === Internal dependency: freqtrade.exceptions ===
class OperationalException(FreqtradeException): ...
class StrategyError(FreqtradeException): ...

# === Internal dependency: freqtrade.optimize.hyperopt_tools ===
class HyperoptStateContainer: ...

# === Internal dependency: freqtrade.optimize.space ===
from .decimalspace import SKDecimal

# === Internal dependency: freqtrade.persistence ===
from freqtrade.persistence.pairlock_middleware import PairLocks
from freqtrade.persistence.trade_model import Trade

# === Internal dependency: freqtrade.resolvers ===
from freqtrade.resolvers.strategy_resolver import StrategyResolver

# === Internal dependency: freqtrade.strategy.hyper ===
def detect_parameters(obj, category): ...

# === Internal dependency: freqtrade.strategy.parameters ===
class BaseParameter(ABC): ...
class IntParameter(NumericParameter):
    def __init__(self, low, high=..., *, default, space=..., optimize=..., load=..., **kwargs): ...
    def get_space(self, name): ...
    def range(self): ...
class RealParameter(NumericParameter):
class DecimalParameter(NumericParameter):
    def __init__(self, low, high=..., *, default, decimals=..., space=..., optimize=..., load=..., **kwargs): ...
class CategoricalParameter(BaseParameter):
    def __init__(self, categories, *, default=..., space=..., optimize=..., load=..., **kwargs): ...
class BooleanParameter(CategoricalParameter):
    def __init__(self, *, default=..., space=..., optimize=..., load=..., **kwargs): ...

# === Internal dependency: freqtrade.util ===
from freqtrade.util.datetime_helpers import dt_now

# === Third-party dependency: pandas ===
# Used symbols: DataFrame

# === Third-party dependency: pytest ===
# Used symbols: mark, raises

# === Unresolved dependency: skopt.space ===
# Used unresolved symbols: Categorical, Integer, Real

# === Internal dependency: tests.conftest ===
def log_has(line, logs): ...
def log_has_re(line, logs): ...
CURRENT_TEST_STRATEGY = 'StrategyTestV3'
TRADE_SIDES = ('long', 'short')

# === Internal dependency: tests.strategy.strats.strategy_test_v3 ===
class StrategyTestV3(IStrategy):
    def populate_indicators(self, dataframe, metadata): ...