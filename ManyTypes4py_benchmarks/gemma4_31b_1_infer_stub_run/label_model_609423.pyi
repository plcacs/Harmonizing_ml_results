import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, DefaultDict, Dict, List, NamedTuple, Optional, Set, Tuple, Union, overload
from snorkel.labeling.analysis import LFAnalysis
from snorkel.labeling.model.base_labeler import BaseLabeler
from snorkel.labeling.model.logger import Logger
from snorkel.types import Config
from snorkel.utils.config_utils import merge_config
from snorkel.utils.lr_schedulers import LRSchedulerConfig
from snorkel.utils.optimizers import OptimizerConfig

Metrics = Dict[str, float]

class TrainConfig(Config):
    n_epochs: int
    lr: float
    l2: float
    optimizer: str
    optimizer_config: OptimizerConfig
    lr_scheduler: str
    lr_scheduler_config: LRSchedulerConfig
    prec_init: Union[int, float, np.ndarray, List[float], torch.Tensor]
    seed: int
    log_freq: int
    mu_eps: Optional[float]

class LabelModelConfig(Config):
    verbose: bool
    device: str

class _CliqueData(NamedTuple):
    start_index: int
    end_index: int
    max_cliques: Set[int]

class LabelModel(nn.Module, BaseLabeler):
    cardinality: int
    config: LabelModelConfig
    seed: int
    n: int
    m: int
    t: int
    c_data: Dict[int, _CliqueData]
    c_tree: Any
    d: int
    O: torch.Tensor
    mask: torch.Tensor
    _prec_init: torch.Tensor
    mu_init: torch.Tensor
    mu: nn.Parameter
    p: np.ndarray
    P: torch.Tensor
    coverage: np.ndarray
    train_config: TrainConfig
    logger: Logger
    optimizer: optim.Optimizer
    lr_scheduler: Optional[optim.lr_scheduler._LRScheduler]
    warmup_steps: int
    warmup_scheduler: Optional[optim.lr_scheduler._LRScheduler]
    running_loss: float
    running_examples: int

    def __init__(self, cardinality: int = 2, **kwargs: Any) -> None: ...

    def _create_L_ind(self, L: np.ndarray) -> np.ndarray: ...

    def _get_augmented_label_matrix(self, L: np.ndarray, higher_order: bool = False) -> np.ndarray: ...

    def _build_mask(self) -> None: ...

    def _generate_O(self, L: np.ndarray, higher_order: bool = False) -> None: ...

    def _init_params(self) -> None: ...

    def _get_conditional_probs(self, mu: np.ndarray) -> np.ndarray: ...

    def get_conditional_probs(self) -> np.ndarray: ...

    def get_weights(self) -> np.ndarray: ...

    def predict_proba(self, L: np.ndarray) -> np.ndarray: ...

    @overload
    def predict(self, L: np.ndarray, return_probs: bool = False, tie_break_policy: str = 'abstain') -> np.ndarray: ...
    @overload
    def predict(self, L: np.ndarray, return_probs: bool = True, tie_break_policy: str = 'abstain') -> Tuple[np.ndarray, np.ndarray]: ...

    def score(self, L: np.ndarray, Y: np.ndarray, metrics: List[str] = ['accuracy'], tie_break_policy: str = 'abstain') -> Dict[str, float]: ...

    def _loss_l2(self, l2: Union[int, float, np.ndarray] = 0) -> torch.Tensor: ...

    def _loss_mu(self, l2: Union[int, float, np.ndarray] = 0) -> torch.Tensor: ...

    def _set_class_balance(self, class_balance: Optional[Union[List[float], np.ndarray]], Y_dev: Optional[np.ndarray] = None) -> None: ...

    def _set_constants(self, L: np.ndarray) -> None: ...

    def _create_tree(self) -> None: ...

    def _execute_logging(self, loss: torch.Tensor) -> Dict[str, float]: ...

    def _set_logger(self) -> None: ...

    def _set_optimizer(self) -> None: ...

    def _set_lr_scheduler(self) -> None: ...

    def _set_warmup_scheduler(self) -> None: ...

    def _update_lr_scheduler(self, step: int) -> None: ...

    def _clamp_params(self) -> None: ...

    def _break_col_permutation_symmetry(self) -> None: ...

    def fit(
        self, 
        L_train: np.ndarray, 
        Y_dev: Optional[np.ndarray] = None, 
        class_balance: Optional[Union[List[float], np.ndarray]] = None, 
        progress_bar: bool = True, 
        **kwargs: Any
    ) -> None: ...