from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional
from pandas import DataFrame
from skopt import Optimizer
from skopt.space import Dimension
from freqtrade.constants import Config
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.optimize.hyperopt.hyperopt_auto import HyperOptAuto
from freqtrade.optimize.hyperopt_loss.hyperopt_loss_interface import IHyperOptLoss

MAX_LOSS: int

class HyperOptimizer:
    buy_space: List[Dimension]
    sell_space: List[Dimension]
    protection_space: List[Dimension]
    roi_space: List[Dimension]
    stoploss_space: List[Dimension]
    trailing_space: List[Dimension]
    max_open_trades_space: List[Dimension]
    dimensions: List[Dimension]
    config: Config
    backtesting: Backtesting
    pairlist: List[str]
    analyze_per_epoch: bool
    custom_hyperopt: HyperOptAuto
    custom_hyperoptloss: IHyperOptLoss
    calculate_loss: Any
    data_pickle_file: Any
    market_change: float
    min_date: datetime
    max_date: datetime
    timerange: Any

    def __init__(self, config: Config) -> None: ...
    def prepare_hyperopt(self) -> None: ...
    def get_strategy_name(self) -> str: ...
    def hyperopt_pickle_magic(self, bases: Any) -> None: ...
    def _get_params_dict(self, dimensions: List[Dimension], raw_params: List[Any]) -> Dict[str, Any]: ...
    def _get_params_details(self, params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]: ...
    def _get_no_optimize_details(self) -> Dict[str, Dict[str, Any]]: ...
    def init_spaces(self) -> None: ...
    def assign_params(self, params_dict: Dict[str, Any], category: str) -> None: ...
    def generate_optimizer(self, raw_params: List[Any]) -> Dict[str, Any]: ...
    def _get_results_dict(self, backtesting_results: Dict[str, Any], min_date: datetime, max_date: datetime, params_dict: Dict[str, Any], processed: Dict[str, DataFrame]) -> Dict[str, Any]: ...
    def get_optimizer(self, cpu_count: int, random_state: Optional[int], initial_points: int, model_queue_size: int) -> Optimizer: ...
    def advise_and_trim(self, data: Dict[str, DataFrame]) -> Dict[str, DataFrame]: ...
    def prepare_hyperopt_data(self) -> None: ...