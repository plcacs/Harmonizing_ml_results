from typing import Any

# === Internal dependency: freqtrade.constants ===
DATETIME_PRINT_FORMAT: str

# === Internal dependency: freqtrade.data.converter ===
# re-export: from freqtrade.data.converter.converter import trim_dataframes

# === Internal dependency: freqtrade.data.history ===
# re-export: from .history_utils import get_timerange

# === Internal dependency: freqtrade.data.metrics ===
def calculate_market_change(data: dict[str, pd.DataFrame], column: str = ...) -> float: ...

# === Internal dependency: freqtrade.enums.HyperoptState ===
DATALOAD: Any
INDICATORS: Any
OPTIMIZE: Any

# === Internal dependency: freqtrade.exceptions ===
class OperationalException(FreqtradeException): ...

# === Internal dependency: freqtrade.misc ===
def deep_merge_dicts(source, destination, allow_null_overrides: bool = ...) -> Any: ...
def round_dict(d, n) -> Any: ...

# === Internal dependency: freqtrade.optimize.backtesting ===
class Backtesting:
    def __init__(self, config: Config, exchange: Exchange | None = ...) -> None: ...

# === Internal dependency: freqtrade.optimize.hyperopt.hyperopt_auto ===
class HyperOptAuto(IHyperOpt):
    ...

# === Internal dependency: freqtrade.optimize.hyperopt.hyperopt_logger ===
def logging_mp_setup(log_queue: Queue, verbosity: int) -> Any: ...
def logging_mp_handle(q: Queue) -> Any: ...

# === Internal dependency: freqtrade.optimize.hyperopt_loss.hyperopt_loss_interface ===
class IHyperOptLoss(ABC): ...

# === Internal dependency: freqtrade.optimize.hyperopt_tools ===
class HyperoptStateContainer: ...
class HyperoptTools:
    ...

# === Internal dependency: freqtrade.optimize.optimize_reports ===
# re-export: from freqtrade.optimize.optimize_reports.optimize_reports import generate_strategy_stats

# === Internal dependency: freqtrade.optimize.space ===
# re-export: from .decimalspace import SKDecimal
# re-export: from .optunaspaces import DimensionProtocol
# re-export: from .optunaspaces import ft_CategoricalDistribution
# re-export: from .optunaspaces import ft_FloatDistribution
# re-export: from .optunaspaces import ft_IntDistribution

# === Internal dependency: freqtrade.resolvers.hyperopt_resolver ===
class HyperOptLossResolver(IResolver):
    ...

# === Internal dependency: freqtrade.util ===
# re-export: from freqtrade.util.datetime_helpers import dt_now

# === Internal dependency: freqtrade.util.dry_run_wallet ===
def get_dry_run_wallet(config: Config) -> int | float: ...

# === Third-party dependency: joblib ===
# Used symbols: delayed, dump, load, wrap_non_picklable_objects

# === Third-party dependency: joblib.externals ===
# Used symbols: cloudpickle

# === Unresolved dependency: optuna ===
# Used unresolved symbols: create_study, distributions, samplers

# === Unresolved dependency: optuna.exceptions ===
# Used unresolved symbols: ExperimentalWarning

# === Unresolved dependency: optuna.terminator ===
# Used unresolved symbols: BestValueStagnationEvaluator, Terminator

# === Third-party dependency: pandas ===
# Used symbols: DataFrame