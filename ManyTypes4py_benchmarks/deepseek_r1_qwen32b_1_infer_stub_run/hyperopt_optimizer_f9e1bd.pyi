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
    cast,
)
from skopt import Optimizer
from skopt.space import Dimension
from pandas import DataFrame
from freqtrade.enums import HyperoptState
from freqtrade.optimize.hyperopt.hyperopt_auto import HyperOptAuto
from freqtrade.optimize.hyperopt_loss.hyperopt_loss_interface import IHyperOptLoss
from freqtrade.util.dry_run_wallet import get_dry_run_wallet

MAX_LOSS: int = ...

class HyperOptimizer:
    """
    HyperoptOptimizer class
    """
    def __init__(self, config: dict) -> None:
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

    def generate_optimizer(self, raw_params: List[float]) -> Dict[str, Any]:
        ...

    def _get_results_dict(
        self,
        backtesting_results: Dict[str, Any],
        min_date: datetime,
        max_date: datetime,
        params_dict: Dict[str, Any],
        processed: Dict[str, DataFrame],
    ) -> Dict[str, Any]:
        ...

    def get_optimizer(
        self,
        cpu_count: int,
        random_state: int,
        initial_points: int,
        model_queue_size: int,
    ) -> Optimizer:
        ...

    def advise_and_trim(self, data: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
        ...

    def prepare_hyperopt_data(self) -> None:
        ...