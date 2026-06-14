import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pandas import DataFrame
from skopt import Optimizer
from skopt.space import Dimension

from freqtrade.constants import Config
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.optimize.hyperopt.hyperopt_auto import HyperOptAuto
from freqtrade.optimize.hyperopt_loss.hyperopt_loss_interface import IHyperOptLoss

logger: logging.Logger
MAX_LOSS: int

class HyperOptimizer:
    buy_space: list[Dimension]
    sell_space: list[Dimension]
    protection_space: list[Dimension]
    roi_space: list[Dimension]
    stoploss_space: list[Dimension]
    trailing_space: list[Dimension]
    max_open_trades_space: list[Dimension]
    dimensions: list[Dimension]
    config: Config
    backtesting: Backtesting
    pairlist: list[str]
    analyze_per_epoch: bool
    custom_hyperopt: HyperOptAuto
    custom_hyperoptloss: IHyperOptLoss
    calculate_loss: Any
    data_pickle_file: Path
    market_change: float
    min_date: datetime
    max_date: datetime
    timerange: Any

    def __init__(self, config: Config) -> None: ...
    def prepare_hyperopt(self) -> None: ...
    def get_strategy_name(self) -> str: ...
    def hyperopt_pickle_magic(self, bases: tuple[type, ...]) -> None: ...
    def _get_params_dict(self, dimensions: list[Dimension], raw_params: list[Any]) -> dict[str, Any]: ...
    def _get_params_details(self, params: dict[str, Any]) -> dict[str, Any]: ...
    def _get_no_optimize_details(self) -> dict[str, Any]: ...
    def init_spaces(self) -> None: ...
    def assign_params(self, params_dict: dict[str, Any], category: str) -> None: ...
    def generate_optimizer(self, raw_params: list[Any]) -> dict[str, Any]: ...
    def _get_results_dict(
        self,
        backtesting_results: dict[str, Any],
        min_date: datetime,
        max_date: datetime,
        params_dict: dict[str, Any],
        processed: dict[str, DataFrame],
    ) -> dict[str, Any]: ...
    def get_optimizer(
        self,
        cpu_count: int,
        random_state: int,
        initial_points: int,
        model_queue_size: int,
    ) -> Optimizer: ...
    def advise_and_trim(self, data: dict[str, DataFrame]) -> dict[str, DataFrame]: ...
    def prepare_hyperopt_data(self) -> None: ...