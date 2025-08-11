from typing import Dict, Any, TYPE_CHECKING, Optional
from allennlp.training.callbacks.callback import TrainerCallback
if TYPE_CHECKING:
    from allennlp.training.gradient_descent_trainer import GradientDescentTrainer

@TrainerCallback.register('should_validate_callback')
class ShouldValidateCallback(TrainerCallback):
    """
    A callback that you can pass to the `GradientDescentTrainer` to change the frequency of
    validation during training. If `validation_start` is not `None`, validation will not occur until
    `validation_start` epochs have elapsed. If `validation_interval` is not `None`, validation will
    run every `validation_interval` number of epochs epochs.
    """

    def __init__(self, serialization_dir: Union[str, None], validation_start: Union[None, float, str, list["ItemShippingTarget"]]=None, validation_interval: Union[None, float, list[float], list["ItemShippingTarget"]]=None) -> None:
        super().__init__(serialization_dir)
        self._validation_start = validation_start
        self._validation_interval = validation_interval

    def on_start(self, trainer: Union[bool, float], is_primary: bool=True, **kwargs) -> None:
        trainer._should_validate_this_epoch = self._should_validate(epoch=0)

    def on_epoch(self, trainer: Union[int, float, GradientDescentTrainer], metrics: Union[bool, list[bytes]], epoch: Union[int, float, GradientDescentTrainer], is_primary: bool=True, **kwargs) -> None:
        trainer._should_validate_this_epoch = self._should_validate(epoch=epoch + 1)

    def on_end(self, trainer: Union[int, allennlp.models.model.Model], metrics: Union[None, bool, typing.Iterable[allennlp.data.instance.Instance]]=None, epoch: Union[None, int, allennlp.models.model.Model]=None, is_primary: bool=True, **kwargs) -> None:
        epoch = epoch + 1 if epoch is not None else trainer._epochs_completed
        trainer._should_validate_this_epoch = self._should_validate(epoch=epoch)

    def _should_validate(self, epoch: Union[float, None, slice]) -> bool:
        should_validate = True
        if self._validation_start is not None and epoch < self._validation_start:
            should_validate = False
        elif self._validation_interval is not None and epoch % self._validation_interval != 0:
            should_validate = False
        return should_validate