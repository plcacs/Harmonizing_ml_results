from typing import Dict, Any, List, Tuple, Optional
import torch
from allennlp.common.lazy import Lazy
from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import LearningRateScheduler


@LearningRateScheduler.register('combined')
class CombinedLearningRateScheduler(LearningRateScheduler):
    """
    This `LearningRateScheduler` can be used to apply an arbitrary number of other schedulers
    one after the other.

    These schedulers are defined though the `schedulers` parameter, which takes
    a list of `Tuple[int, Lazy[LearningRateScheduler]]`. The first field of the
    tuple, the `int`, specifies how many epochs the corresponding scheduler will
     be used before the next scheduler takes its place.

    While it usually makes sense for the sum

    ```python
    sum(n_epochs for (n_epochs, _) in schedulers)
    ```

    to equal the total number of training epochs, it is not a requirement.
    If training continues beyond the last defined scheduler, both `step()` and
    `step_batch()` will be a no-op. In effect, this causes the learning rate to
    stay constant.

    # Example

    Config for using the `CombinedLearningRateScheduler` Learning Rate Scheduler
    with the following arguments:

    * Use [`PolynomialDecay`](
    https://docs.allennlp.org/main/api/training/learning_rate_schedulers/polynomial_decay/
    ) for the first `15` epochs.
    * Use [`NoamLR`](
    https://docs.allennlp.org/main/api/training/learning_rate_schedulers/noam/
    ) for the next `15` epochs.
    * Use a constant LR for the remaining epochs.

    ```json
    {
        ...
       "trainer":{
            ...
            "learning_rate_scheduler": {
                "type": "combined",
                "schedulers": [
                    [
                        15, {
                            "type": "polynomial_decay",
                            "power": 2,
                            "warmup_steps": 50,
                            "end_learning_rate": 1e-10
                        }
                    ],
                    [
                        15, {
                            "type": "noam",
                            "warmup_steps": 1,
                            "model_size": 128,
                            "factor": 0.5
                        }
                    ]
                ]
            },
            ...
       }
    }
    ```
    Note that you do NOT pass a `optimizer` key to the Learning rate scheduler.
    """

    def __init__(self, optimizer, schedulers, num_steps_per_epoch=None,
        last_epoch=-1):
        super().__init__(optimizer, last_epoch=last_epoch)
        self.num_steps_per_epoch = num_steps_per_epoch
        self.schedulers = schedulers
        self._last_epoch_updated = -2
        self._current_scheduler: Optional[LearningRateScheduler] = None
        self._current_scheduler_first_epoch: Optional[int] = None
        self.current_scheduler

    @property
    def current_scheduler(self):
        if self._last_epoch_updated != self.last_epoch:
            current_epoch = self.last_epoch + 1
            scheduler_first_epoch, scheduler_last_epoch = 0, -1
            for scheduler_epochs, lazy_scheduler in self.schedulers:
                scheduler_last_epoch += scheduler_epochs
                if (current_epoch == scheduler_first_epoch or self.
                    _current_scheduler_first_epoch != scheduler_first_epoch and
                    scheduler_first_epoch <= current_epoch <=
                    scheduler_last_epoch):
                    for group in self.optimizer.param_groups:
                        group[self._initial_param_group_field] = group[self
                            .param_group_field]
                    self._current_scheduler = lazy_scheduler.construct(
                        optimizer=self.optimizer, num_epochs=
                        scheduler_epochs, num_steps_per_epoch=self.
                        num_steps_per_epoch)
                    self._current_scheduler_first_epoch = scheduler_first_epoch
                    break
                scheduler_first_epoch = scheduler_last_epoch + 1
            else:
                if current_epoch > scheduler_last_epoch:
                    self._current_scheduler = None
        self._last_epoch_updated = self.last_epoch
        return self._current_scheduler

    def state_dict(self):
        current_scheduler = self.current_scheduler
        return {'last_epoch': self.last_epoch, 'num_steps_per_epoch': self.
            num_steps_per_epoch, 'current_scheduler': None if 
            current_scheduler is None else current_scheduler.state_dict()}

    def load_state_dict(self, state_dict):
        self.last_epoch = state_dict['last_epoch']
        self.num_steps_per_epoch = state_dict['num_steps_per_epoch']
        if self.current_scheduler is not None:
            assert state_dict['current_scheduler'] is not None
            self.current_scheduler.load_state_dict(state_dict[
                'current_scheduler'])

    def get_values(self):
        """
        This should never be called directly.
        """
        raise NotImplementedError

    def step_batch(self, batch_num_total=None):
        if self.current_scheduler is not None:
            self.current_scheduler.step_batch(batch_num_total)

    def step(self, metric=None):
        self.last_epoch += 1
        self.metric = metric
        if self.current_scheduler is not None:
            self.current_scheduler.step(metric)
