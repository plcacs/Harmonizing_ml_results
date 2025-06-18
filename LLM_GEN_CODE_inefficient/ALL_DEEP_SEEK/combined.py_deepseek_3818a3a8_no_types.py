from typing import Dict, Any, List, Tuple, Optional
import torch
from allennlp.common.lazy import Lazy
from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import LearningRateScheduler

@LearningRateScheduler.register('combined')
class CombinedLearningRateScheduler(LearningRateScheduler):

    def __init__(self, optimizer, schedulers, num_steps_per_epoch=None, last_epoch=-1):
        super().__init__(optimizer, last_epoch=last_epoch)
        self.num_steps_per_epoch: Optional[int] = num_steps_per_epoch
        self.schedulers: List[Tuple[int, Lazy[LearningRateScheduler]]] = schedulers
        self._last_epoch_updated: int = -2
        self._current_scheduler: Optional[LearningRateScheduler] = None
        self._current_scheduler_first_epoch: Optional[int] = None
        self.current_scheduler

    @property
    def current_scheduler(self):
        if self._last_epoch_updated != self.last_epoch:
            current_epoch: int = self.last_epoch + 1
            scheduler_first_epoch: int = 0
            scheduler_last_epoch: int = -1
            for scheduler_epochs, lazy_scheduler in self.schedulers:
                scheduler_last_epoch += scheduler_epochs
                if current_epoch == scheduler_first_epoch or (self._current_scheduler_first_epoch != scheduler_first_epoch and scheduler_first_epoch <= current_epoch <= scheduler_last_epoch):
                    for group in self.optimizer.param_groups:
                        group[self._initial_param_group_field] = group[self.param_group_field]
                    self._current_scheduler = lazy_scheduler.construct(optimizer=self.optimizer, num_epochs=scheduler_epochs, num_steps_per_epoch=self.num_steps_per_epoch)
                    self._current_scheduler_first_epoch = scheduler_first_epoch
                    break
                scheduler_first_epoch = scheduler_last_epoch + 1
            else:
                if current_epoch > scheduler_last_epoch:
                    self._current_scheduler = None
        self._last_epoch_updated = self.last_epoch
        return self._current_scheduler

    def state_dict(self):
        current_scheduler: Optional[LearningRateScheduler] = self.current_scheduler
        return {'last_epoch': self.last_epoch, 'num_steps_per_epoch': self.num_steps_per_epoch, 'current_scheduler': None if current_scheduler is None else current_scheduler.state_dict()}

    def load_state_dict(self, state_dict):
        self.last_epoch: int = state_dict['last_epoch']
        self.num_steps_per_epoch: Optional[int] = state_dict['num_steps_per_epoch']
        if self.current_scheduler is not None:
            assert state_dict['current_scheduler'] is not None
            self.current_scheduler.load_state_dict(state_dict['current_scheduler'])

    def get_values(self):
        raise NotImplementedError

    def step_batch(self, batch_num_total=None):
        if self.current_scheduler is not None:
            self.current_scheduler.step_batch(batch_num_total)

    def step(self, metric=None):
        self.last_epoch += 1
        self.metric: Optional[float] = metric
        if self.current_scheduler is not None:
            self.current_scheduler.step(metric)