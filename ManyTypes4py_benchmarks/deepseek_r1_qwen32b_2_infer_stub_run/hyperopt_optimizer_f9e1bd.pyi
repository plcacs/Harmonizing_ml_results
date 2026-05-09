"""
Stub file for hyperopt_optimizer_f9e1bd module
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from skopt import Optimizer
from pandas import DataFrame

MAX_LOSS: float
logger: logging.Logger

class HyperOptimizer:
    """
    HyperoptOptimizer class
    """
    buy_space: List[Any]
    sell_space: List[Any]
    protection_space: List[Any]
    roi_space: List[Any]
    stoploss_space: List[Any]
    trailing_space: List[Any]
    max_open_trades_space: List[Any]
    dimensions: List[Any]
    config: Dict[str, Any]
    backtesting: Backtesting
    pairlist: List[str]
    analyze_per_epoch: bool
    custom_hyperopt: HyperOptAuto
    custom_hyperoptloss: IHyperOptLoss
    data_pickle_file: str
    market_change: float

    def __init__(self, config: dict) -> None:
        ...

    def prepare_hyperopt(self) -> None:
        ...

    def get_strategy_name(self) -> str:
        ...

    def hyperopt_pickle_magic(self, bases: Any) -> None:
        ...

    def _get_params_dict(self, dimensions: List[Any], raw_params: List[Any]) -> Dict[str, Any]:
        ...

    def _get_params_details(self, params: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def _get_no_optimize_details(self) -> Dict[str, Any]:
        ...

    def init_spaces(self) -> None:
        ...

    def assign_params(self, params_dict: Dict[str, Any], category: str) -> None:
        ...

    def generate_optimizer(self, raw_params: List[Any]) -> Dict[str, Any]:
        ...

    def _get_results_dict(self, backtesting_results: Dict[str, Any], min_date: datetime, max_date: datetime, params_dict: Dict[str, Any], processed: Any) -> Dict[str, Any]:
        ...

    def get_optimizer(self, cpu_count: int, random_state: int, initial_points: int, model_queue_size: int) -> Optimizer:
        ...

    def advise_and_trim(self, data: Any) -> DataFrame:
        ...

    def prepare_hyperopt_data(self) -> None:
        ...