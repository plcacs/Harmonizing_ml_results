"""
Code for training
"""
import glob
import logging
import os
import pickle
import random
import shutil
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Iterable, Tuple, Union, Set, cast
import numpy as np
import torch
import torch.distributed
try:
    import apex.amp
except ImportError:
    pass
from . import average
from . import checkpoint_decoder
from . import constants as C
from . import data_io
from . import loss
from . import lr_scheduler
from . import model
from . import optimizers
from . import utils
from . import vocab
from .config import Config
logger = logging.getLogger(__name__)

class ModelWithLoss(torch.nn.Module):
    """
    Wraps a SockeyeModel and its Losses in a single module. The SockeyeModel
    can be JIT traced (ScriptModule).

    :param model: SockeyeModel (untraced or traced).
    :param losses: List of Loss objects.

    :return: Tuple of summed loss, list of loss values, and list of number of
             samples.
    """

    def __init__(self, model: torch.nn.Module, losses: List[loss.Loss]) -> None:
        super().__init__()
        self.model = model
        self.losses = losses

    def forward(self, source: torch.Tensor, source_length: torch.Tensor, target: torch.Tensor, target_length: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        model_outputs = self.model(source, source_length, target, target_length)
        if utils.using_deepspeed():
            model_outputs = {output_name: output.to(torch.float32) for output_name, output in model_outputs.items()}
        loss_outputs = [loss_function(model_outputs, labels) for loss_function in self.losses]
        loss_values, num_samples = zip(*loss_outputs)
        sum_losses = sum(loss_values) if len(loss_values) > 1 else loss_values[0]
        return (sum_losses, loss_values, num_samples)

@dataclass
class TrainerConfig(Config):
    max_checkpoints: Optional[int] = None
    min_samples: Optional[int] = None
    max_samples: Optional[int] = None
    min_updates: Optional[int] = None
    max_updates: Optional[int] = None
    min_epochs: Optional[int] = None
    max_epochs: Optional[int] = None
    max_seconds: Optional[int] = None
    update_interval: int = 1
    stop_training_on_decoder_failure: bool = False
    no_reload_on_learning_rate_reduce: bool = False

class TrainState:
    """
    Stores the state an EarlyStoppingTrainer instance.
    """
    __slots__ = ['num_not_improved', 'epoch', 'checkpoint', 'best_checkpoint', 'batches', 'updates', 'samples', 'metrics', 'start_tic', '_tic_last_time_elapsed', '_time_elapsed', 'early_stopping_metric', 'best_metric', 'best_metric_history', 'best_checkpoint', 'converged', 'diverged']

    def __init__(self, early_stopping_metric: str) -> None:
        self.num_not_improved: int = 0
        self.epoch: int = 0
        self.checkpoint: int = 0
        self.best_checkpoint: int = 0
        self.batches: int = 0
        self.updates: int = 0
        self.samples: int = 0
        self.metrics: List[Dict[str, Any]] = []
        self.start_tic: float = time.time()
        self._tic_last_time_elapsed: float = self.start_tic
        self._time_elapsed: float = 0.0
        self.early_stopping_metric: str = early_stopping_metric
        self.best_metric: float = C.METRIC_WORST[early_stopping_metric]
        self.best_metric_history: deque = deque([self.best_metric])
        self.best_checkpoint: int = 0
        self.converged: bool = False
        self.diverged: bool = False

    def save(self, fname: str) -> None:
        """
        Saves this training state to fname.
        """
        self.update_time_elapsed()
        assert len(self.metrics) == self.checkpoint
        with open(fname, 'wb') as fp:
            pickle.dump(self, fp)

    @staticmethod
    def load(fname: str) -> 'TrainState':
        """
        Loads a training state from fname.
        """
        with open(fname, 'rb') as fp:
            state = pickle.load(fp)
            state._tic_last_time_elapsed = time.time()
            assert len(state.metrics) == state.checkpoint
            return state

    def update_time_elapsed(self) -> None:
        current_time = time.time()
        self._time_elapsed += current_time - self._tic_last_time_elapsed
        self._tic_last_time_elapsed = current_time

    @property
    def time_elapsed(self) -> float:
        return self._time_elapsed

    def __getstate__(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        for k, v in state.items():
            setattr(self, k, v)

class EarlyStoppingTrainer:

    def __init__(self, config: TrainerConfig, optimizer_config: optimizers.OptimizerConfig, sockeye_model: model.SockeyeModel, model_object: ModelWithLoss, optimizer: torch.optim.Optimizer, lr_scheduler: Optional[lr_scheduler.LearningRateScheduler], zero_grad_kwargs: Dict[str, Any], loss_functions: List[loss.Loss], device: torch.device, using_amp: bool = False, using_apex_amp: bool = False, custom_metrics_logger: Optional[Callable[[Dict[str, Any], Optional[int]], None]] = None, checkpoint_callback: Optional[Callable[[int], None]] = None) -> None:
        self.config = config
        self.optimizer_config = optimizer_config
        self.sockeye_model = sockeye_model
        self.model_object = model_object
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.zero_grad_kwargs = zero_grad_kwargs
        self.loss_functions = loss_functions
        self.device = device
        self.using_amp = using_amp
        if using_amp:
            self._scaler = torch.cuda.amp.GradScaler()
        self.using_apex_amp = using_apex_amp
        self.state: Optional[TrainState] = None
        self._speedometer = Speedometer(frequency=C.MEASURE_SPEED_EVERY, auto_reset=False)
        self._custom_metrics_logger = custom_metrics_logger
        self._tflogger = TensorboardLogger(logdir=os.path.join(self.config.output_dir, C.TENSORBOARD_NAME))
        self.checkpoint_callback = checkpoint_callback

    def fit(self, train_iter: data_io.BaseDataIter, validation_iter: data_io.BaseDataIter, checkpoint_decoder: Optional[checkpoint_decoder.CheckpointDecoder] = None) -> TrainState:
        logger.info("Early stopping by optimizing '%s'", self.config.early_stopping_metric)
        if utils.is_primary_worker() and self.config.early_stopping_metric in C.METRICS_REQUIRING_DECODER:
            utils.check_condition(checkpoint_decoder is not None, '%s requires CheckpointDecoder' % self.config.early_stopping_metric)
        resume_training = os.path.exists(self.training_state_dirname)
        if resume_training:
            logger.info("Found partial training in '%s'. Resuming from saved state.", self.training_state_dirname)
            self._load_training_state(train_iter)
        else:
            self.state = TrainState(self.config.early_stopping_metric)
            if utils.is_primary_worker():
                self.sockeye_model.save_config(self.config.output_dir)
                self.sockeye_model.save_version(self.config.output_dir)
            self._save_params(use_checkpoint=False)
            logger.info('Training started.')
        tic = time.time()
        if self.config.max_checkpoints is not None:
            self.config.max_updates = self.state.updates + self.config.max_checkpoints * self.config.checkpoint_interval
            logger.info('Resetting max_updates to %d + %d * %d = %d in order to implement stopping after (an additional) %d checkpoints.', self.state.updates, self.config.max_checkpoints, self.config.checkpoint_interval, self.config.max_updates, self.config.max_checkpoints)
        checkpoint_up_to_date = resume_training
        while True:
            if self.config.max_epochs is not None and self.state.epoch == self.config.max_epochs:
                logger.info('Maximum # of epochs (%s) reached.', self.config.max_epochs)
                if not checkpoint_up_to_date:
                    time_cost = time.time() - tic
                    self._create_checkpoint(checkpoint_decoder, time_cost, train_iter, validation_iter)
                break
            if self.config.max_updates is not None and self.state.updates == self.config.max_updates:
                logger.info('Maximum # of updates (%s) reached.', self.config.max_updates)
                if not checkpoint_up_to_date:
                    time_cost = time.time() - tic
                    self._create_checkpoint(checkpoint_decoder, time_cost, train_iter, validation_iter)
                break
            if self.config.max_samples is not None and self.state.samples >= self.config.max_samples:
                logger.info('Maximum # of samples (%s) reached', self.config.max_samples)
                if not checkpoint_up_to_date:
                    time_cost = time.time() - tic
                    self._create_checkpoint(checkpoint_decoder, time_cost, train_iter, validation_iter)
                break
            did_grad_step = self._step(batch=train_iter.next())
            checkpoint_up_to_date = checkpoint_up_to_date and (not did_grad_step)
            if not train_iter.iter_next():
                self.state.epoch += 1
                train_iter.reset()
            if self.state.updates > 0 and self.state.batches % (self.config.checkpoint_interval * self.config.update_interval) == 0:
                time_cost = time.time() - tic
                self._create_checkpoint(checkpoint_decoder, time_cost, train_iter, validation_iter)
                checkpoint_up_to_date = True
                if self.config.max_seconds is not None and self.state.time_elapsed >= self.config.max_seconds:
                    logger.info('Maximum # of seconds (%s) reached. Training ran for %d seconds.', self.config.max_seconds, self.state.time_elapsed)
                    break
                if self.state.converged or self.state.diverged:
                    break
                tic = time.time()
        logger.info('Training finished%s. Best checkpoint: %d. Best validation %s: %.6f', ', can be continued later' if not self.state.converged else '', self.state.best_checkpoint, self.state.early_stopping_metric, self.state.best_metric)
        if utils.is_primary_worker():
            self._cleanup(keep_training_state=True)
        return self.state

    def _create_checkpoint(self, checkpoint_decoder: Optional[checkpoint_decoder.CheckpointDecoder], time_cost: float, train_iter: data_io.BaseDataIter, validation_iter: data_io.BaseDataIter) -> None:
        """
        Creates a checkpoint, which will update self.state.converged/self.state.diverged, evaluate validation
        metrics and update the best known parameters accordingly.
        """
        self.state.checkpoint += 1
        train_metrics = [lf.metric for lf in self.loss_functions]
        logger.info('Checkpoint [%d]\tUpdates=%d Epoch=%d Samples=%d Time-cost=%.3f Updates/sec=%.3f', self.state.checkpoint, self.state.updates, self.state.epoch, self.state.samples, time_cost, self.config.checkpoint_interval / time_cost)
        logger.info('Checkpoint [%d]\t%s', self.state.checkpoint, '\t'.join(('Train-%s' % str(metric) for metric in train_metrics)))
        val_metrics = self._evaluate(self.state.checkpoint, validation_iter, checkpoint_decoder)
        has_improved = self._determine_improvement(val_metrics)
        self.state.converged = self._determine_convergence()
        self.state.diverged = self._determine_divergence(val_metrics)
        self._adjust_learning_rate(has_improved)
        if utils.is_primary_worker():
            self._write_and_log_metrics(train_metrics=train_metrics, val_metrics=val_metrics)
        self._save_training_state(train_iter)
        self._save_params(use_checkpoint=True)
        if utils.is_primary_worker():
            if has_improved:
                self._update_best_params()
                if not utils.using_deepspeed():
                    self._save_optimizer_state(self.best_optimizer_state_fname)
                    self._save_lr_scheduler(self.best_lr_scheduler_fname)
        for metric in train_metrics:
            metric.reset()
        if self.checkpoint_callback:
            self.checkpoint_callback(self.state.checkpoint)

    def _forward_backward(self, batch: data_io.Batch, is_update_batch: bool = True) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Performs forward-backward pass on a batch.

        :param batch: Current data batch.
        :param is_update_batch: Whether this is the final batch before updating
                                weights.
        :return: List loss values.
        """
        batch = batch.load(device=self.device)
        with torch.cuda.amp.autocast(cache_enabled=False) if self.using_amp else utils.no_context():
            sum_losses, loss_values, num_samples = self.model_object(batch.source, batch.source_length, batch.target, batch.target_length, batch.labels)
        if utils.using_deepspeed():
            self.model_object.backward(sum_losses)
        else:
            if self.config.update_interval > 1:
                sum_losses = sum_losses / self.config.update_interval
            if self.using_amp:
                sum_losses = self._scaler.scale(sum_losses)
            if self.using_apex_amp:
                with apex.amp.scale_loss(sum_losses, self.optimizer, delay_unscale=not is_update_batch) as scaled_sum_losses:
                    scaled_sum_losses.backward()
            else:
                sum_losses.backward()
        return (loss_values, num_samples)

    def _step(self, batch: data_io.Batch) -> bool:
        self.state.batches += 1
        self.state.samples += batch.samples
        is_update_batch = self.state.batches % self.config.update_interval == 0
        self.state.updates += 1 if is_update_batch else 0
        with self.model_object.model.no_sync() if utils.is_distributed() and (not is_update_batch) and (not utils.using_deepspeed()) else utils.no_context():
            loss_values, num_samples = self._forward_backward(batch, is_update_batch)
        for loss_func, loss_value, num_samples in zip(self.loss_functions, loss_values, num_samples):
            loss_func.metric.update(loss_value.item(), num_samples.item())
        if utils.using_deepspeed():
            self.model_object.step()
        elif is_update_batch:
            if self.using_amp:
                self._scaler.unscale_(self.optimizer)
            if self.optimizer_config.gradient_clipping_type == C.GRADIENT_CLIPPING_TYPE_ABS:
                torch.nn.utils.clip_grad.clip_grad_value_(self.sockeye_model.parameters(), self.optimizer_config.gradient_clipping_threshold)
            elif self.optimizer_config.gradient_clipping_type == C.GRADIENT_CLIPPING_TYPE_NORM:
                torch.nn.utils.clip_grad.clip_grad_norm_(self.sockeye_model.parameters(), self.optimizer_config.gradient_clipping_threshold)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.using_amp:
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(**self.zero_grad_kwargs)
        self._speedometer(self.state.epoch, self.state.batches, self.state.updates, batch.samples, batch.tokens, (lf.metric for lf in self.loss_functions))
        return is_update_batch

    def _evaluate(self, checkpoint: int, data_iter: data_io.BaseDataIter, checkpoint_decoder: Optional[checkpoint_decoder.CheckpointDecoder]) -> List[loss.LossMetric]:
        """
        Computes loss(es) on validation data and returns their metrics.
        :param data_iter: Validation data iterator.
        :return: List of validation metrics, same order as self.loss_functions.
        """
        self.sockeye_model.eval()
        data_iter.reset()
        val_metrics = [lf.create_metric() for lf in self.loss_functions]
        for batch in data_iter:
            batch = batch.load(device=self.device)
            with torch.inference_mode():
                outputs = self.sockeye_model(batch.source, batch.source_length, batch.target, batch.target_length)
                outputs = {name: output.to(torch.float32) for name, output in outputs.items()}
                loss_outputs = [loss_function(outputs, batch.labels) for loss_function in self.loss_functions]
            for loss_metric, (loss_value, num_samples) in zip(val_metrics, loss_outputs):
                loss_metric.update(loss_value.item(), num_samples.item())
        if utils.is_primary