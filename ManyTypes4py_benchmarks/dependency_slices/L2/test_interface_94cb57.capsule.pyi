from typing import Any

# === Internal dependency: freqtrade.configuration.TimeRange ===
parse_timerange: Any

# === Internal dependency: freqtrade.constants ===
CUSTOM_TAG_MAX_LENGTH: int

# === Internal dependency: freqtrade.data.dataprovider ===
class DataProvider:
    def __init__(self, config: Config, exchange: Exchange | None, pairlists = ..., rpc: RPCManager | None = ...) -> None: ...

# === Internal dependency: freqtrade.data.history ===
# re-export: from .history_utils import load_data

# === Internal dependency: freqtrade.enums ===
# re-export: from freqtrade.enums.signaltype import SignalDirection

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
# re-export: from .decimalspace import SKDecimal

# === Internal dependency: freqtrade.persistence ===
# re-export: from freqtrade.persistence.pairlock_middleware import PairLocks
# re-export: from freqtrade.persistence.trade_model import Trade

# === Internal dependency: freqtrade.resolvers ===
# re-export: from freqtrade.resolvers.strategy_resolver import StrategyResolver

# === Internal dependency: freqtrade.strategy.hyper ===
def detect_parameters(obj: HyperStrategyMixin | type[HyperStrategyMixin], category: str) -> Iterator[tuple[str, BaseParameter]]: ...

# === Internal dependency: freqtrade.strategy.parameters ===
class BaseParameter(ABC): ...
class IntParameter(NumericParameter):
    def __init__(self, low: int | Sequence[int], high: int | None = ..., *, default: int, space: str | None = ..., optimize: bool = ..., load: bool = ..., **kwargs) -> Any: ...
    def get_space(self, name: str) -> 'Integer': ...
    def range(self) -> Any: ...
class RealParameter(NumericParameter):
    def __init__(self, low: float | Sequence[float], high: float | None = ..., *, default: float, space: str | None = ..., optimize: bool = ..., load: bool = ..., **kwargs) -> Any: ...
    def get_space(self, name: str) -> 'Real': ...
class DecimalParameter(NumericParameter):
    def __init__(self, low: float | Sequence[float], high: float | None = ..., *, default: float, decimals: int = ..., space: str | None = ..., optimize: bool = ..., load: bool = ..., **kwargs) -> Any: ...
    def get_space(self, name: str) -> 'SKDecimal': ...
class CategoricalParameter(BaseParameter):
    def __init__(self, categories: Sequence[Any], *, default: Any | None = ..., space: str | None = ..., optimize: bool = ..., load: bool = ..., **kwargs) -> Any: ...
    def get_space(self, name: str) -> 'Categorical': ...
class BooleanParameter(CategoricalParameter):
    def __init__(self, *, default: Any | None = ..., space: str | None = ..., optimize: bool = ..., load: bool = ..., **kwargs) -> Any: ...

# === Internal dependency: freqtrade.util ===
# re-export: from freqtrade.util.datetime_helpers import dt_now

# === Third-party dependency: pandas ===
# Used symbols: DataFrame

# === Third-party dependency: pytest ===
# Used symbols: mark, raises

# === Unresolved dependency: skopt.space ===
# Used unresolved symbols: Categorical, Integer, Real

# === Internal dependency: tests.conftest ===
def log_has(line, logs) -> Any: ...
def log_has_re(line, logs) -> Any: ...
CURRENT_TEST_STRATEGY: str
TRADE_SIDES: Any

# === Internal dependency: tests.strategy.strats.strategy_test_v3 ===
class StrategyTestV3(IStrategy):
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame: ...