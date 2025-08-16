from typing import Dict, Any, List, Tuple, Optional
import torch
from allennlp.common.lazy import Lazy
from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import LearningRateScheduler

@LearningRateScheduler.register('combined')
class CombinedLearningRateScheduler(LearningRateScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, schedulers: List[Tuple[int, Lazy[LearningRateScheduler]], num_steps_per_epoch: Optional[int] = None, last_epoch: int = -1) -> None:
        super().__init__(optimizer, last_epoch=last_epoch)
        self.num_steps_per_epoch: Optional[int] = num_steps_per_epoch
        self.schedulers: List[Tuple[int, Lazy[LearningRateScheduler]]] = schedulers
        self._last_epoch_updated: int = -2
        self._current_scheduler: Optional[LearningRateScheduler] = None
        self._current_scheduler_first_epoch: Optional[int] = None

    @property
    def current_scheduler(self) -> Optional[LearningRateScheduler]:
        ...

    def state_dict(self) -> Dict[str, Any]:
        ...

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        ...

    def get_values(self) -> None:
        ...

    def step_batch(self, batch_num_total: Optional[int] = None) -> None:
        ...

    def step(self, metric: Any = None) -> None:
        ...
