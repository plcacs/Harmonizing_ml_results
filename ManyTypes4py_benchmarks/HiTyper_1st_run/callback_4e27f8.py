from typing import List, Dict, Any, Optional, TYPE_CHECKING
import torch
from allennlp.common import Registrable
from allennlp.data import TensorDict
if TYPE_CHECKING:
    from allennlp.training.gradient_descent_trainer import GradientDescentTrainer

class TrainerCallback(Registrable):
    """
    A general callback object that handles multiple events.

    This class has `on_backward`, `on_batch`, `on_epoch`, and `on_end` methods, corresponding to
    each callback type. Each one receives the state of the wrapper object as `self`.
    This enables easier state sharing between related callbacks.

    Also, this callback type is instantiated with `serialization_dir` and `on_start` is called
    with the trainer instance as an argument. This might be handy in case of callback logging
    and saving its own files next to the config/checkpoints/logs/etc.
    """

    def __init__(self, serialization_dir: str) -> None:
        self.serialization_dir = serialization_dir
        self.trainer = None

    def on_start(self, trainer: Union[bool, tuple[typing.Union[bool,float]]], is_primary: bool=True, **kwargs) -> None:
        """
        This callback hook is called before the training is started.
        """
        self.trainer = trainer

    def on_backward(self, trainer: bool, batch_outputs: bool, backward_called: bool, **kwargs) -> bool:
        """
        This callback hook performs backpropagation and allows for gradient manipulation.
        `backward_called` indicates if `loss.backward` has been called prior to this callback.
        `on_backward` should return `True` if and only if `loss.backward` is called in its body.
        """
        return False

    def on_batch(self, trainer: bool, batch_inputs: bool, batch_outputs: bool, batch_metrics: bool, epoch: bool, batch_number: bool, is_training: bool, is_primary: bool=True, batch_grad_norm: Union[None, bool]=None, **kwargs) -> None:
        """
        This callback hook is called after the end of each batch.
        """
        pass

    def on_epoch(self, trainer: Union[bool, list[bytes]], metrics: Union[bool, list[bytes]], epoch: Union[bool, list[bytes]], is_primary: bool=True, **kwargs) -> None:
        """
        This callback hook is called after the end of each epoch.
        """
        pass

    def on_end(self, trainer: Union[bool, typing.Iterable[allennlp.data.instance.Instance]], metrics: Union[None, bool, typing.Iterable[allennlp.data.instance.Instance]]=None, epoch: Union[None, bool, typing.Iterable[allennlp.data.instance.Instance]]=None, is_primary: bool=True, **kwargs) -> None:
        """
        This callback hook is called after the final training epoch.
        """
        pass

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: Union[dict, list[typing.Callable], dict[str, typing.Any]]) -> None:
        pass
TrainerCallback.register('null')(TrainerCallback)