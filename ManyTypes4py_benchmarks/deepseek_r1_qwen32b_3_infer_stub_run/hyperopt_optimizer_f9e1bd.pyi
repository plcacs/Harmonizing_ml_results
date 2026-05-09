"""
Stub file for hyperopt_optimizer_f9e1bd module
"""

from __future__ import annotations
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from skopt import Optimizer
from skopt.space import Dimension
from pandas import DataFrame
from freqtrade.enums import HyperoptState
from freqtrade.optimize.hyperopt_loss.hyperopt_loss_interface import IHyperOptLoss

class HyperOptimizer:
    """
    HyperoptOptimizer class
    """
    buy_space: List[Dimension]
    sell_space: List[Dimension]
    protection_space: List[Dimension]
    roi_space: List[Dimension]
    stoploss_space: List[Dimension]
    trailing_space: List[Dimension]
    max_open_trades_space: List[Dimension]
    dimensions: List[Dimension]
    config: Dict[str, Any]
    backtesting: Any
    pairlist: List[str]
    analyze_per_epoch: bool
    custom_hyperopt: HyperOptAuto
    custom_hyperoptloss: IHyperOptLoss
    calculate_loss: Any
    data_pickle_file: Any
    market_change: float

    def __init__(self, config: Dict[str, Any]) -> None:
        ...

    def prepare_hyperopt(self) -> None:
        ...

    def get_strategy_name(self) -> str:
        ...

    def hyperopt_pickle_magic(self, bases: Any) -> None:
        ...

    def _get_params_dict(self, dimensions: List[Dimension], raw_params: List[float]) -> Dict[str, Any]:
        ...

    def _get_params_details(self, params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        ...

    def _get_no_optimize_details(self) -> Dict[str, Dict[str, Any]]:
        ...

    def init_spaces(self) -> None:
        ...

    def assign_params(self, params_dict: Dict[str, Any], category: str) -> None:
        ...

    def generate_optimizer(self, raw_params: List[float]) -> Dict[str, Union[float, Dict[str, Any], str, int]]:
        ...

    def _get_results_dict(self, backtesting_results: Dict[str, Any], min_date: datetime, max_date: datetime, params_dict: Dict[str, Any], processed: DataFrame) -> Dict[str, Union[float, Dict[str, Any], str, int]]:
        ...

    def get_optimizer(self, cpu_count: int, random_state: int, initial_points: int, model_queue_size: int) -> Optimizer:
        ...

    def advise_and_trim(self, data: DataFrame) -> DataFrame:
        ...

    def prepare_hyperopt_data(self) -> None:
        ...