```python
from collections.abc import MutableMapping
from typing import (
    Any,
    DefaultDict,
    Deque,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
import torch
from allennlp.data import TensorDict
from allennlp.training.callbacks.callback import TrainerCallback

logger: logging.Logger = ...

class LogWriterCallback(TrainerCallback):
    _summary_interval: int
    _distribution_interval: Optional[int]
    _batch_size_interval: Optional[int]
    _should_log_parameter_statistics: bool
    _should_log_learning_rate: bool
    _cumulative_batch_group_size: int
    _distribution_parameters: Optional[Set[str]]
    _module_hook_handles: List[Any]
    _batch_loss_moving_average_count: int
    _batch_loss_moving_sum: DefaultDict[str, float]
    _batch_loss_moving_items: DefaultDict[str, Deque[float]]
    _param_updates: Optional[Dict[str, torch.Tensor]]
    trainer: Any

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

    def log_scalars(
        self,
        scalars: Dict[str, Union[int, float]],
        log_prefix: str = ...,
        epoch: Optional[int] = None,
    ) -> None: ...

    def log_tensors(
        self,
        tensors: Dict[str, torch.Tensor],
        log_prefix: str = ...,
        epoch: Optional[int] = None,
    ) -> None: ...

    def log_inputs(
        self,
        inputs: List[TensorDict],
        log_prefix: str = ...,
    ) -> None: ...

    def close(self) -> None: ...

    def on_start(
        self,
        trainer: Any,
        is_primary: bool = ...,
        **kwargs: Any,
    ) -> None: ...

    def on_batch(
        self,
        trainer: Any,
        batch_inputs: List[TensorDict],
        batch_outputs: Dict[str, Any],
        batch_metrics: Dict[str, Any],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_primary: bool = ...,
        batch_grad_norm: Optional[float] = None,
        **kwargs: Any,
    ) -> None: ...

    def on_epoch(
        self,
        trainer: Any,
        metrics: Dict[str, Any],
        epoch: int,
        is_primary: bool = ...,
        **kwargs: Any,
    ) -> None: ...

    def on_end(
        self,
        trainer: Any,
        metrics: Optional[Dict[str, Any]] = None,
        epoch: Optional[int] = None,
        is_primary: bool = ...,
        **kwargs: Any,
    ) -> None: ...

    def log_batch(
        self,
        batch_grad_norm: Optional[float],
        metrics: Dict[str, Any],
        batch_group: List[TensorDict],
        param_updates: Optional[Dict[str, torch.Tensor]],
        batch_number: int,
    ) -> None: ...

    def log_epoch(
        self,
        train_metrics: Dict[str, Any],
        val_metrics: Dict[str, Any],
        epoch: int,
    ) -> None: ...

    def _should_log_distributions_next_batch(self) -> bool: ...

    def _should_log_distributions_this_batch(self) -> bool: ...

    def _enable_activation_logging(self) -> None: ...

    def _should_log_this_batch(self) -> bool: ...

    def _log_activation_distribution(
        self,
        outputs: Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor], Dict[str, torch.Tensor]],
        module_name: str,
    ) -> None: ...

    def _log_parameter_and_gradient_statistics(
        self,
        batch_grad_norm: Optional[float] = None,
    ) -> None: ...

    def _log_learning_rates(self) -> None: ...

    def _log_distributions(self) -> None: ...

    def _log_gradient_updates(
        self,
        param_updates: Dict[str, torch.Tensor],
    ) -> None: ...
```