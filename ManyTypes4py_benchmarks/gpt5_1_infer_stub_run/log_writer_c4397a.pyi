from typing import Any, DefaultDict, Deque, Dict, List, Mapping, Optional, Set
import logging
import torch
from torch.utils.hooks import RemovableHandle
from allennlp.data import TensorDict
from allennlp.training.callbacks.callback import TrainerCallback
from allennlp.training.gradient_descent_trainer import GradientDescentTrainer

logger: logging.Logger = ...

class LogWriterCallback(TrainerCallback):
    trainer: GradientDescentTrainer
    _summary_interval: int
    _distribution_interval: Optional[int]
    _batch_size_interval: Optional[int]
    _should_log_parameter_statistics: bool
    _should_log_learning_rate: bool
    _cumulative_batch_group_size: float
    _distribution_parameters: Optional[Set[str]]
    _module_hook_handles: List[RemovableHandle]
    _batch_loss_moving_average_count: int
    _batch_loss_moving_sum: DefaultDict[str, float]
    _batch_loss_moving_items: DefaultDict[str, Deque[float]]
    _param_updates: Optional[Dict[str, torch.Tensor]]

    def __init__(
        self,
        serialization_dir: str,
        summary_interval: int = ...,
        distribution_interval: Optional[int] = ...,
        batch_size_interval: Optional[int] = ...,
        should_log_parameter_statistics: bool = ...,
        should_log_learning_rate: bool = ...,
        batch_loss_moving_average_count: int = ...,
    ) -> None: ...
    def log_scalars(self, scalars: Mapping[str, float | int], log_prefix: str = ..., epoch: Optional[int] = ...) -> None: ...
    def log_tensors(self, tensors: Mapping[str, torch.Tensor], log_prefix: str = ..., epoch: Optional[int] = ...) -> None: ...
    def log_inputs(self, inputs: List[TensorDict], log_prefix: str = ...) -> None: ...
    def close(self) -> None: ...
    def on_start(self, trainer: GradientDescentTrainer, is_primary: bool = ..., **kwargs: Any) -> None: ...
    def on_batch(
        self,
        trainer: GradientDescentTrainer,
        batch_inputs: List[TensorDict],
        batch_outputs: Any,
        batch_metrics: Dict[str, float],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_primary: bool = ...,
        batch_grad_norm: Optional[float] = ...,
        **kwargs: Any,
    ) -> None: ...
    def on_epoch(
        self,
        trainer: GradientDescentTrainer,
        metrics: Dict[str, Any],
        epoch: int,
        is_primary: bool = ...,
        **kwargs: Any,
    ) -> None: ...
    def on_end(
        self,
        trainer: GradientDescentTrainer,
        metrics: Optional[Dict[str, Any]] = ...,
        epoch: Optional[int] = ...,
        is_primary: bool = ...,
        **kwargs: Any,
    ) -> None: ...
    def log_batch(
        self,
        batch_grad_norm: Optional[float],
        metrics: Dict[str, float],
        batch_group: List[TensorDict],
        param_updates: Optional[Dict[str, torch.Tensor]],
        batch_number: int,
    ) -> None: ...
    def log_epoch(self, train_metrics: Dict[str, float | int], val_metrics: Dict[str, float | int], epoch: int) -> None: ...
    def _should_log_distributions_next_batch(self) -> bool: ...
    def _should_log_distributions_this_batch(self) -> bool: ...
    def _enable_activation_logging(self) -> None: ...
    def _should_log_this_batch(self) -> bool: ...
    def _log_activation_distribution(self, outputs: Any, module_name: str) -> None: ...
    def _log_parameter_and_gradient_statistics(self, batch_grad_norm: Optional[float] = ...) -> None: ...
    def _log_learning_rates(self) -> None: ...
    def _log_distributions(self) -> None: ...
    def _log_gradient_updates(self, param_updates: Dict[str, torch.Tensor]) -> None: ...