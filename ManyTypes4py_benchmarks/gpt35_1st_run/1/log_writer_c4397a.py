from collections import deque, defaultdict
import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING, Union, Deque, Set
import torch
from allennlp.data import TensorDict
from allennlp.nn.util import tiny_value_of_dtype
from allennlp.training.callbacks.callback import TrainerCallback
from allennlp.training.util import get_train_and_validation_metrics, get_batch_size

if TYPE_CHECKING:
    from allennlp.training.gradient_descent_trainer import GradientDescentTrainer

logger = logging.getLogger(__name__)

class LogWriterCallback(TrainerCallback):
    def __init__(self, serialization_dir: str, summary_interval: int = 100, distribution_interval: Optional[int] = None, batch_size_interval: Optional[int] = None, should_log_parameter_statistics: bool = True, should_log_learning_rate: bool = False, batch_loss_moving_average_count: int = 100) -> None:
        ...

    def log_scalars(self, scalars: Dict[str, Union[int, float]], log_prefix: str = '', epoch: Optional[int] = None) -> None:
        ...

    def log_tensors(self, tensors: Dict[str, torch.Tensor], log_prefix: str = '', epoch: Optional[int] = None) -> None:
        ...

    def log_inputs(self, inputs: Any, log_prefix: str = '') -> None:
        ...

    def close(self) -> None:
        ...

    def on_start(self, trainer: 'GradientDescentTrainer', is_primary: bool = True, **kwargs: Any) -> None:
        ...

    def on_batch(self, trainer: 'GradientDescentTrainer', batch_inputs: Any, batch_outputs: Any, batch_metrics: Dict[str, Any], epoch: int, batch_number: int, is_training: bool, is_primary: bool = True, batch_grad_norm: Optional[float] = None, **kwargs: Any) -> None:
        ...

    def on_epoch(self, trainer: 'GradientDescentTrainer', metrics: Dict[str, Any], epoch: int, is_primary: bool = True, **kwargs: Any) -> None:
        ...

    def on_end(self, trainer: 'GradientDescentTrainer', metrics: Optional[Dict[str, Any]] = None, epoch: Optional[int] = None, is_primary: bool = True, **kwargs: Any) -> None:
        ...

    def log_batch(self, batch_grad_norm: Optional[float], metrics: Dict[str, Any], batch_group: Any, param_updates: Optional[Dict[str, torch.Tensor]], batch_number: int) -> None:
        ...

    def log_epoch(self, train_metrics: Dict[str, Any], val_metrics: Dict[str, Any], epoch: int) -> None:
        ...

    def _should_log_distributions_next_batch(self) -> bool:
        ...

    def _should_log_distributions_this_batch(self) -> bool:
        ...

    def _enable_activation_logging(self) -> None:
        ...

    def _should_log_this_batch(self) -> bool:
        ...

    def _log_activation_distribution(self, outputs: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]], module_name: str) -> None:
        ...

    def _log_parameter_and_gradient_statistics(self, batch_grad_norm: Optional[float] = None) -> None:
        ...

    def _log_learning_rates(self) -> None:
        ...

    def _log_distributions(self) -> None:
        ...

    def _log_gradient_updates(self, param_updates: Dict[str, torch.Tensor]) -> None:
        ...
