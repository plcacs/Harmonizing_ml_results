from typing import Any, Dict, List
import torch
from torch.optim.optimizer import Optimizer as TorchOptimizer
from torch.optim.lr_scheduler import _LRScheduler
from allennlp.common.checks import ConfigurationError
from allennlp.common.registrable import Registrable
from allennlp.training.optimizers import Optimizer
from allennlp.training.scheduler import Scheduler
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)


class LearningRateScheduler(Scheduler, Registrable):
    def __init__(self, optimizer: TorchOptimizer, last_epoch: int = -1) -> None:
        super().__init__(optimizer, "lr", last_epoch)

    def get_values(self) -> List[float]:
        raise NotImplementedError


class _PyTorchLearningRateSchedulerWrapper(LearningRateScheduler):
    def __init__(self, lr_scheduler: _LRScheduler) -> None:
        self.lr_scheduler = lr_scheduler

    def get_values(self) -> List[float]:
        return self.lr_scheduler.get_last_lr()

    def step(self, metric: Any = None) -> None:
        self.lr_scheduler.step()

    def state_dict(self) -> Dict[str, Any]:
        return self.lr_scheduler.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.lr_scheduler.load_state_dict(state_dict)


class _PyTorchLearningRateSchedulerWithMetricsWrapper(_PyTorchLearningRateSchedulerWrapper):
    def step(self, metric: Any = None) -> None:
        if metric is None:
            raise ConfigurationError(
                "This learning rate scheduler requires a validation metric to compute the schedule and therefore must be used with a validation dataset."
            )
        self.lr_scheduler.step(metric)


@LearningRateScheduler.register("constant")
class ConstantLearningRateScheduler(_PyTorchLearningRateSchedulerWrapper):
    """
    Registered as a `LearningRateScheduler` with name "constant".  The
    "optimizer" argument does not get an entry in a configuration file for the
    object.
    """

    def __init__(self, optimizer: TorchOptimizer, last_epoch: int = -1) -> None:
        lr_scheduler = get_constant_schedule(optimizer=optimizer, last_epoch=last_epoch)
        super().__init__(lr_scheduler)


@LearningRateScheduler.register("constant_with_warmup")
class ConstantWithWarmupLearningRateScheduler(_PyTorchLearningRateSchedulerWrapper):
    """
    Registered as a `LearningRateScheduler` with name "constant_with_warmup".
    The "optimizer" argument does not get an entry in a configuration file for
    the object.
    """

    def __init__(self, optimizer: TorchOptimizer, num_warmup_steps: int, last_epoch: int = -1) -> None:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=num_warmup_steps, last_epoch=last_epoch
        )
        super().__init__(lr_scheduler)


@LearningRateScheduler.register("cosine_with_warmup")
class CosineWithWarmupLearningRateScheduler(_PyTorchLearningRateSchedulerWrapper):
    """
    Registered as a `LearningRateScheduler` with name "cosine_with_warmup".  The "optimizer" argument does
    not get an entry in a configuration file for the object.
    """

    def __init__(
        self,
        optimizer: TorchOptimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
    ) -> None:
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            last_epoch=last_epoch,
        )
        super().__init__(lr_scheduler)


@LearningRateScheduler.register("cosine_hard_restarts_with_warmup")
class CosineHardRestartsWithWarmupLearningRateScheduler(_PyTorchLearningRateSchedulerWrapper):
    """
    Registered as a `LearningRateScheduler` with name
    "cosine_hard_restarts_with_warmup". The "optimizer" argument does not get an
    entry in a configuration file for the object.
    """

    def __init__(
        self,
        optimizer: TorchOptimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: int = 1,
        last_epoch: int = -1,
    ) -> None:
        lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            last_epoch=last_epoch,
        )
        super().__init__(lr_scheduler)