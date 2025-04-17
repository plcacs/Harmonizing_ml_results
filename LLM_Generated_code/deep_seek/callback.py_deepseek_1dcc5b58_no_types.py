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

    def __init__(self, serialization_dir):
        self.serialization_dir = serialization_dir
        self.trainer: Optional['GradientDescentTrainer'] = None

    def on_start(self, trainer, is_primary=True, **kwargs):
        """
        This callback hook is called before the training is started.
        """
        self.trainer = trainer

    def on_backward(self, trainer, batch_outputs, backward_called, **kwargs):
        """
        This callback hook performs backpropagation and allows for gradient manipulation.
        `backward_called` indicates if `loss.backward` has been called prior to this callback.
        `on_backward` should return `True` if and only if `loss.backward` is called in its body.
        """
        return False

    def on_batch(self, trainer, batch_inputs, batch_outputs, batch_metrics,
        epoch, batch_number, is_training, is_primary=True, batch_grad_norm=
        None, **kwargs):
        """
        This callback hook is called after the end of each batch.
        """
        pass

    def on_epoch(self, trainer, metrics, epoch, is_primary=True, **kwargs):
        """
        This callback hook is called after the end of each epoch.
        """
        pass

    def on_end(self, trainer, metrics=None, epoch=None, is_primary=True, **
        kwargs):
        """
        This callback hook is called after the final training epoch.
        """
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


TrainerCallback.register('null')(TrainerCallback)
