import logging
import os
from typing import Optional, Dict, Any, List, Union, Tuple, TYPE_CHECKING
import torch
from allennlp.common import Params
from allennlp.training.callbacks.callback import TrainerCallback
from allennlp.training.callbacks.log_writer import LogWriterCallback
if TYPE_CHECKING:
    from allennlp.training.gradient_descent_trainer import GradientDescentTrainer
logger = logging.getLogger(__name__)

@TrainerCallback.register('wandb')
class WandBCallback(LogWriterCallback):

    def __init__(self, serialization_dir, summary_interval=100, distribution_interval=None, batch_size_interval=None, should_log_parameter_statistics=True, should_log_learning_rate=False, project=None, entity=None, group=None, name=None, notes=None, tags=None, watch_model=True, files_to_save=('config.json', 'out.log'), wandb_kwargs=None):
        if 'WANDB_API_KEY' not in os.environ:
            logger.warning("Missing environment variable 'WANDB_API_KEY' required to authenticate to Weights & Biases.")
        super().__init__(serialization_dir, summary_interval=summary_interval, distribution_interval=distribution_interval, batch_size_interval=batch_size_interval, should_log_parameter_statistics=should_log_parameter_statistics, should_log_learning_rate=should_log_learning_rate)
        self._watch_model: bool = watch_model
        self._files_to_save: Tuple[str, ...] = files_to_save
        self._run_id: Optional[str] = None
        self._wandb_kwargs: Dict[str, Any] = dict(dir=os.path.abspath(serialization_dir), project=project, entity=entity, group=group, name=name, notes=notes, config=Params.from_file(os.path.join(serialization_dir, 'config.json')).as_dict(), tags=tags, anonymous='allow', **wandb_kwargs or {})

    def log_scalars(self, scalars, log_prefix='', epoch=None):
        self._log(scalars, log_prefix=log_prefix, epoch=epoch)

    def log_tensors(self, tensors, log_prefix='', epoch=None):
        self._log({k: self.wandb.Histogram(v.cpu().data.numpy().flatten()) for k, v in tensors.items()}, log_prefix=log_prefix, epoch=epoch)

    def _log(self, dict_to_log, log_prefix='', epoch=None):
        if log_prefix:
            dict_to_log = {f'{log_prefix}/{k}': v for k, v in dict_to_log.items()}
        if epoch is not None:
            dict_to_log['epoch'] = epoch
        self.wandb.log(dict_to_log, step=self.trainer._total_batches_completed)

    def on_start(self, trainer, is_primary=True, **kwargs: Any):
        super().on_start(trainer, is_primary=is_primary, **kwargs)
        if not is_primary:
            return None
        import wandb
        self.wandb = wandb
        if self._run_id is None:
            self._run_id = self.wandb.util.generate_id()
        self.wandb.init(id=self._run_id, **self._wandb_kwargs)
        for fpath in self._files_to_save:
            self.wandb.save(os.path.join(self.serialization_dir, fpath), base_path=self.serialization_dir)
        if self._watch_model:
            self.wandb.watch(self.trainer.model)

    def close(self):
        super().close()
        self.wandb.finish()

    def state_dict(self):
        return {'run_id': self._run_id}

    def load_state_dict(self, state_dict):
        self._wandb_kwargs['resume'] = 'auto'
        self._run_id = state_dict['run_id']