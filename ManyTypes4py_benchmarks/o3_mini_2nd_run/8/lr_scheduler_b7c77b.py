import logging
from math import sqrt
from typing import Any, Dict, List, Optional, Tuple, Type
import torch
import sockeye.constants as C
from sockeye.utils import check_condition

logger = logging.getLogger(__name__)


class LearningRateScheduler:
    """
    Learning rate scheduler base class.
    """

    def __init__(self, optimizer: Optional[torch.optim.Optimizer] = None, base_lr: float = 1.0, warmup: int = 0) -> None:
        self.optimizer: Optional[torch.optim.Optimizer] = optimizer
        self.base_lr: float = base_lr
        check_condition(warmup >= 0, 'warmup needs to be >= 0.')
        self.warmup: int = warmup
        self._t: int = 0
        self._last_lr: Optional[List[float]] = None

    def __call__(self, optimizer: torch.optim.Optimizer) -> "LearningRateScheduler":
        """
        DeepSpeed compatibility method: associate otherwise initialized learning
        rate scheduler with an optimizer.
        """
        assert self.optimizer is None, 'This learning rate scheduler is already associated with an optimizer.'
        self.optimizer = optimizer
        return self

    def __repr__(self) -> str:
        return self.__class__.__name__

    def state_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != 'optimizer'}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)

    def get_lr(self) -> List[float]:
        """
        Get the learning rate for the current step for each param group.
        """
        raise NotImplementedError()

    def get_last_lr(self) -> List[float]:
        """
        Get the last computed learning rate for each param group.
        """
        assert self._last_lr is not None, '`get_last_lr()` cannot be called before `get_lr()`'
        return self._last_lr

    def step(self, t: Optional[int] = None) -> None:
        """
        Increment or specify the time step (update number) and recompute the
        learning rate for each param group by calling `get_lr()`.
        """
        assert self.optimizer is not None, 'This learning rate scheduler is not associated with an optimizer.'
        if t is None:
            t = self._t + 1
        self._t = t
        for group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            group['lr'] = lr
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _warmup(self, t: int) -> float:
        """
        Returns linearly increasing fraction of base_lr.
        """
        if not self.warmup:
            return self.base_lr
        return self.base_lr * min(1.0, t / self.warmup)


class AdaptiveLearningRateScheduler(LearningRateScheduler):
    """
    Learning rate scheduler that implements new_evaluation_result and adaptively adjusts the learning rate.
    """

    def new_evaluation_result(self, has_improved: bool) -> bool:
        """
        Returns true if the parameters should be reset to the ones with the best validation score.

        :param has_improved: Whether the model improved on held-out validation data.
        :return: True if parameters should be reset to the ones with best validation score.
        """
        return False


class LearningRateSchedulerInvSqrtDecay(LearningRateScheduler):
    """
    Learning rate schedule: lr / sqrt(max(t, warmup_steps)).
    """

    def get_lr(self) -> List[float]:
        warm_lr: float = self._warmup(self._t)
        warmup_steps: int = max(1, self.warmup)
        lr: float = warm_lr / sqrt(max(self._t, warmup_steps))
        return [lr for _ in self.optimizer.param_groups]  # type: ignore


class LearningRateSchedulerLinearDecay(LearningRateScheduler):
    """
    Learning rate schedule: lr * (1 - t / total_steps)
    Step grows until it reaches decay_steps then remains constant.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, base_lr: float, total_steps: int, warmup: int = 0) -> None:
        super().__init__(optimizer, base_lr, warmup)
        check_condition(total_steps >= 0, 'total_steps need to be >= 0.')
        self.total_steps: int = total_steps

    def get_lr(self) -> List[float]:
        warm_lr: float = self._warmup(self._t)
        bounded_t: int = min(max(self._t, 1), self.total_steps)
        lr: float = warm_lr * (1 - bounded_t / self.total_steps)
        return [lr for _ in self.optimizer.param_groups]  # type: ignore


class LearningRateSchedulerPlateauReduce(AdaptiveLearningRateScheduler):
    """
    Lower the learning rate as soon as the validation score plateaus.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, base_lr: float, reduce_factor: float, reduce_num_not_improved: int, warmup: int = 0) -> None:
        super().__init__(optimizer, base_lr, warmup)
        self.lr: float = base_lr
        check_condition(0.0 < reduce_factor < 1, 'reduce_factor should be between (0, 1).')
        self.reduce_factor: float = reduce_factor
        self.reduce_num_not_improved: int = reduce_num_not_improved
        self.num_not_improved: int = 0
        self.warmed_up: bool = not self.warmup > 0
        logger.info("Will reduce the learning rate by a factor of %.2f whenever the validation score doesn't improve %d times.", reduce_factor, reduce_num_not_improved)

    def __repr__(self) -> str:
        return ('LearningRateSchedulerPlateauReduce(reduce_factor=%.2f, reduce_num_not_improved=%d, num_not_improved=%d, '
                'base_lr=%s, lr=%s, warmup=%d, warmed_up=%s)' % (self.reduce_factor, self.reduce_num_not_improved, self.num_not_improved, self.base_lr, self.lr, self.warmup, self.warmed_up))

    def new_evaluation_result(self, has_improved: bool) -> bool:
        """
        Returns true if the parameters should be reset to the ones with the best validation score.
        """
        if has_improved:
            self.num_not_improved = 0
        else:
            self.num_not_improved += 1
            if self.num_not_improved >= self.reduce_num_not_improved and self.reduce_factor < 1.0 and self.warmed_up:
                old_lr: float = self.lr
                self.lr *= self.reduce_factor
                logger.info('%d checkpoints since improvement or rate scaling, lowering learning rate: %1.2e -> %1.2e', self.num_not_improved, old_lr, self.lr)
                self.num_not_improved = 0
                return True
        return False

    def get_lr(self) -> List[float]:
        lr: float = self._warmup(self._t) if self.warmup > 0 and self._t <= self.warmup else self.lr
        if self._t == self.warmup:
            self.warmed_up = True
        return [lr for _ in self.optimizer.param_groups]  # type: ignore


def get_lr_scheduler(
    scheduler_type: Optional[str],
    base_learning_rate: float,
    learning_rate_reduce_factor: Optional[float],
    learning_rate_reduce_num_not_improved: Optional[int],
    learning_rate_warmup: int = 0,
    max_updates: Optional[int] = None
) -> Tuple[Optional[Type[LearningRateScheduler]], Dict[str, Any]]:
    """
    Get learning rate scheduler class and kwargs.
    """
    if scheduler_type is None or scheduler_type == C.LR_SCHEDULER_NONE:
        return (None, {})
    if scheduler_type == C.LR_SCHEDULER_INV_SQRT_DECAY:
        return (LearningRateSchedulerInvSqrtDecay, {'base_lr': base_learning_rate, 'warmup': learning_rate_warmup})
    if scheduler_type == C.LR_SCHEDULER_LINEAR_DECAY:
        check_condition(max_updates is not None, 'The total number of training updates (--max-updates) must be specified when using the linear decay learning rate scheduler.')
        return (LearningRateSchedulerLinearDecay, {'base_lr': base_learning_rate, 'total_steps': max_updates, 'warmup': learning_rate_warmup})
    if scheduler_type == C.LR_SCHEDULER_PLATEAU_REDUCE:
        check_condition(learning_rate_reduce_factor is not None, 'learning_rate_reduce_factor needed for %s scheduler' % C.LR_SCHEDULER_PLATEAU_REDUCE)
        check_condition(learning_rate_reduce_num_not_improved is not None, 'learning_rate_reduce_num_not_improved needed for %s scheduler' % C.LR_SCHEDULER_PLATEAU_REDUCE)
        if learning_rate_reduce_factor >= 1.0:
            logger.warning('Not using %s learning rate scheduling: learning_rate_reduce_factor == 1.0', C.LR_SCHEDULER_PLATEAU_REDUCE)
            return (None, {})
        return (LearningRateSchedulerPlateauReduce, {
            'base_lr': base_learning_rate,
            'reduce_factor': learning_rate_reduce_factor,
            'reduce_num_not_improved': learning_rate_reduce_num_not_improved,
            'warmup': learning_rate_warmup
        })
    raise ValueError('Unknown learning rate scheduler type %s.' % scheduler_type)