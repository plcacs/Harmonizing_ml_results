import glob
from typing import Tuple, Optional, Set, Union, Dict, Any
import logging
import os
import re
import time
import torch
import torch.distributed as dist
from allennlp.common import Registrable
from allennlp.common.util import is_distributed
from allennlp.nn import util as nn_util
from allennlp.training.trainer import Trainer, TrainerCheckpoint

logger = logging.getLogger(__name__)

class Checkpointer(Registrable):
    default_implementation = 'default'

    def __init__(self, serialization_dir: str, save_completed_epochs: bool = True, save_every_num_seconds: Optional[int] = None, save_every_num_batches: Optional[int] = None, keep_most_recent_by_count: Optional[int] = 2, keep_most_recent_by_age: Optional[int] = None) -> None:
        self._serialization_dir = str(serialization_dir)
        self._save_completed_epochs = save_completed_epochs
        self._save_every_num_seconds = save_every_num_seconds
        self._save_every_num_batches = save_every_num_batches
        self._keep_most_recent_by_count = keep_most_recent_by_count
        self._keep_most_recent_by_age = keep_most_recent_by_age
        self._last_save_time = time.time()
        self._last_save_num_epochs_completed = 0
        self._last_save_num_batches_in_epoch_completed = 0
        self._rank = 0 if not is_distributed() else dist.get_rank()
        self.state_is_sharded = False
        if is_distributed() and save_every_num_seconds is not None:
            raise ValueError("Checkointer parameter 'save_every_num_seconds' is not supported in distributed training")

    @property
    def _is_primary(self) -> bool:
        return self._rank == 0

    def _model_state_path(self, epochs_completed: int, batches_in_epoch_completed: int) -> str:
        path = os.path.join(self._serialization_dir, f'model_state_e{epochs_completed}_b{batches_in_epoch_completed}')
        if self.state_is_sharded:
            return path + f'_w{self._rank}.th'
        else:
            return path + '.th'

    def _training_state_path(self, epochs_completed: int, batches_in_epoch_completed: int) -> str:
        path = os.path.join(self._serialization_dir, f'training_state_e{epochs_completed}_b{batches_in_epoch_completed}')
        if self.state_is_sharded:
            return path + f'_w{self._rank}.th'
        else:
            return path + '.th'

    _model_state_file_re = re.compile(r'(.*[/\\\\])?model_state_e(\d+)_b(\d+)(_w\d+)?\.th$')
    _training_state_file_re = re.compile(r'(.*[/\\\\])?training_state_e(\d+)_b(\d+)(_w\d+)?\.th$')

    @classmethod
    def _parse_model_state_path(cls, path: str) -> Optional[Tuple[int, int]]:
        match = cls._model_state_file_re.match(str(path))
        if match is None:
            return None
        else:
            try:
                return (int(match.group(2)), int(match.group(3)))
            except ValueError:
                return None

    @classmethod
    def _parse_training_state_path(cls, path: str) -> Optional[Tuple[int, int]]:
        match = cls._training_state_file_re.match(str(path))
        if match is None:
            return None
        else:
            try:
                return (int(match.group(2)), int(match.group(3)))
            except ValueError:
                return None

    def _find_all_checkpoints(self) -> Set[Tuple[int, int]]:
        checkpoints = set()
        pattern = f'model_state_e*_b*_w{self._rank}.th' if self.state_is_sharded else 'model_state_e*_b*.th'
        for model_state_file in glob.iglob(os.path.join(self._serialization_dir, pattern)):
            point_in_time = self._parse_model_state_path(model_state_file)
            if point_in_time is None:
                continue
            else:
                checkpoints.add(point_in_time)
        return checkpoints

    def _remove_checkpoint(self, epochs_completed: int, batches_in_epoch_completed: int) -> None:
        for state_name in ('model_state', 'training_state'):
            pattern = f'{state_name}_e{epochs_completed}_b{batches_in_epoch_completed}*.th'
            for fname in glob.iglob(os.path.join(self._serialization_dir, pattern)):
                os.remove(fname)

    def maybe_save_checkpoint(self, trainer: Trainer, num_epochs_completed: int, num_batches_in_epoch_completed: int) -> bool:
        end_of_epoch = num_batches_in_epoch_completed == 0
        if num_epochs_completed == self._last_save_num_epochs_completed:
            last_save_num_batches_in_epoch_completed = self._last_save_num_batches_in_epoch_completed
        else:
            last_save_num_batches_in_epoch_completed = 0
        should_save = end_of_epoch and self._save_completed_epochs or (self._save_every_num_seconds is not None and time.time() - self._last_save_time >= self._save_every_num_seconds) or (self._save_every_num_batches is not None and num_batches_in_epoch_completed - last_save_num_batches_in_epoch_completed >= self._save_every_num_batches)
        if should_save:
            self.save_checkpoint(trainer)
            return True
        return False

    def save_checkpoint(self, trainer: Trainer) -> None:
        if self._serialization_dir is None:
            return
        tcps = trainer.get_checkpoint_state()
        if tcps is None:
            assert not self._is_primary and (not self.state_is_sharded)
            return
        epochs_completed = tcps.trainer_state['epochs_completed']
        batches_in_epoch_completed = tcps.trainer_state['batches_in_epoch_completed']
        model_state_path = self._model_state_path(epochs_completed, batches_in_epoch_completed)
        if not os.path.isfile(model_state_path):
            torch.save(tcps.model_state, model_state_path)
        trainer_state_path = self._training_state_path(epochs_completed, batches_in_epoch_completed)
        if not os.path.isfile(trainer_state_path):
            torch.save(tcps.trainer_state, trainer_state_path)
        self._last_save_time = time.time()
        self._last_save_num_epochs_completed = epochs_completed
        self._last_save_num_batches_in_epoch_completed = batches_in_epoch_completed
        if self._is_primary and (self._keep_most_recent_by_age is not None or self._keep_most_recent_by_count is not None):
            checkpoints = list(self._find_all_checkpoints())
            checkpoints.sort(reverse=True)
            if self._keep_most_recent_by_count is not None:
                checkpoints_to_keep = set(checkpoints[:self._keep_most_recent_by_count])
            else:
                checkpoints_to_keep = set()
            now = time.time()
            if self._keep_most_recent_by_age is not None:
                for checkpoint in checkpoints:
                    checkpoint_mtime = max((os.path.getmtime(n) for n in [self._model_state_path(*checkpoint), self._training_state_path(*checkpoint)]))
                    if now - checkpoint_mtime <= self._keep_most_recent_by_age:
                        checkpoints_to_keep.add(checkpoint)
            for checkpoint in checkpoints:
                if checkpoint not in checkpoints_to_keep:
                    self._remove_checkpoint(*checkpoint)

    def find_latest_checkpoint(self) -> Optional[Tuple[str, str]]:
        checkpoints = self._find_all_checkpoints()
        if len(checkpoints) <= 0:
            return None
        last_checkpoint = max(checkpoints)
        return (self._model_state_path(*last_checkpoint), self._training_state_path(*last_checkpoint))

    def load_checkpoint(self) -> Optional[TrainerCheckpoint]:
        latest_checkpoint = self.find_latest_checkpoint()
        if latest_checkpoint is None:
            return None
        model_path, training_state_path = latest_checkpoint
        model_state = torch.load(model_path, map_location=nn_util.device_mapping(-1))
        training_state = torch.load(training_state_path, map_location=nn_util.device_mapping(-1))
        return TrainerCheckpoint(model_state, training_state)

Checkpointer.register('default')(Checkpointer)
