import glob
import logging
import os
import pickle
import random
import shutil
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Iterable, Tuple, Union, Set
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
logger: logging.Logger = logging.getLogger(__name__)

class ModelWithLoss(torch.nn.Module):
    def __init__(self, model: Any, losses: List[Callable]):
        super().__init__()
        self.model = model
        self.losses = losses

    def forward(self, source: torch.Tensor, source_length: torch.Tensor, target: torch.Tensor, target_length: torch.Tensor, labels: torch.Tensor) -> Tuple[float, List[float], List[int]]:
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
    __slots__ = ['num_not_improved', 'epoch', 'checkpoint', 'best_checkpoint', 'batches', 'updates', 'samples', 'metrics', 'start_tic', '_tic_last_time_elapsed', '_time_elapsed', 'early_stopping_metric', 'best_metric', 'best_metric_history', 'best_checkpoint', 'converged', 'diverged']

    def __init__(self, early_stopping_metric: str):
        self.num_not_improved = 0
        self.epoch = 0
        self.checkpoint = 0
        self.best_checkpoint = 0
        self.batches = 0
        self.updates = 0
        self.samples = 0
        self.metrics = []
        self.start_tic = time.time()
        self._tic_last_time_elapsed = self.start_tic
        self._time_elapsed = 0.0
        self.early_stopping_metric = early_stopping_metric
        self.best_metric = C.METRIC_WORST[early_stopping_metric]
        self.best_metric_history = deque([self.best_metric])
        self.best_checkpoint = 0
        self.converged = False
        self.diverged = False

    def save(self, fname: str) -> None:
        self.update_time_elapsed()
        assert len(self.metrics) == self.checkpoint
        with open(fname, 'wb') as fp:
            pickle.dump(self, fp)

    @staticmethod
    def load(fname: str) -> 'TrainState':
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
    def __init__(self, config: TrainerConfig, optimizer_config: Any, sockeye_model: Any, model_object: Any, optimizer: Any, lr_scheduler: Any, zero_grad_kwargs: Dict[str, Any], loss_functions: List[Callable], device: torch.device, using_amp: bool = False, using_apex_amp: bool = False, custom_metrics_logger: Optional[Callable] = None, checkpoint_callback: Optional[Callable] = None):
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
        self.state = None
        self._speedometer = Speedometer(frequency=C.MEASURE_SPEED_EVERY, auto_reset=False)
        self._custom_metrics_logger = custom_metrics_logger
        self._tflogger = TensorboardLogger(logdir=os.path.join(self.config.output_dir, C.TENSORBOARD_NAME))
        self.checkpoint_callback = checkpoint_callback

    def fit(self, train_iter: Any, validation_iter: Any, checkpoint_decoder: Optional[Callable] = None) -> TrainState:
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

    def _create_checkpoint(self, checkpoint_decoder: Optional[Callable], time_cost: float, train_iter: Any, validation_iter: Any) -> None:
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

    def _forward_backward(self, batch: Any, is_update_batch: bool = True) -> Tuple[List[float], List[int]]:
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

    def _step(self, batch: Any) -> bool:
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

    def _evaluate(self, checkpoint: int, data_iter: Any, checkpoint_decoder: Optional[Callable]) -> List[Any]:
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
        if utils.is_primary_worker():
            decoder_metrics = {}
            if checkpoint_decoder is not None:
                output_name = os.path.join(self.config.output_dir, C.DECODE_OUT_NAME.format(checkpoint=checkpoint))
                decoder_metrics = checkpoint_decoder.decode_and_evaluate(output_name=output_name)
            for metric_name, metric_value in decoder_metrics.items():
                assert metric_name not in val_metrics, 'Duplicate validation metric %s' % metric_name
                metric = loss.LossMetric(name=metric_name)
                metric.update(metric_value, num_samples=1)
                val_metrics.append(metric)
        if utils.is_distributed():
            val_metrics = utils.broadcast_object(val_metrics)
        logger.info('Checkpoint [%d]\t%s', self.state.checkpoint, '\t'.join(('Validation-%s' % str(lm) for lm in val_metrics)))
        self.sockeye_model.train()
        return val_metrics

    def _determine_improvement(self, val_metrics: List[Any]) -> bool:
        value = None
        value_is_better = False
        for val_metric in val_metrics:
            if val_metric.name == self.config.early_stopping_metric:
                value = val_metric.get()
                value_is_better = utils.metric_value_is_better(value, self.state.best_metric, self.config.early_stopping_metric)
                if value_is_better:
                    logger.info('Validation-%s improved to %f (delta=%f).', self.config.early_stopping_metric, value, abs(value - self.state.best_metric))
                    self.state.best_metric = value
                    self.state.best_checkpoint = self.state.checkpoint
                    self.state.num_not_improved = 0
        assert value is not None, 'Early stopping metric %s not found in validation metrics.' % self.config.early_stopping_metric
        if not value_is_better:
            self.state.num_not_improved += 1
            logger.info('Validation-%s has not improved for %d checkpoints, best so far: %f', self.config.early_stopping_metric, self.state.num_not_improved, self.state.best_metric)
        self.state.best_metric_history.append(self.state.best_metric)
        if self.config.max_num_checkpoint_not_improved is not None and len(self.state.best_metric_history) > self.config.max_num_checkpoint_not_improved + 1:
            self.state.best_metric_history.popleft()
        return value_is_better

    def _determine_convergence(self) -> bool:
        if self.config.min_samples is not None and self.state.samples < self.config.min_samples:
            logger.info('Minimum number of samples (%d) not reached yet: %d', self.config.min_samples, self.state.samples)
            return False
        if self.config.min_updates is not None and self.state.updates < self.config.min_updates:
            logger.info('Minimum number of updates (%d) not reached yet: %d', self.config.min_updates, self.state.updates)
            return False
        if self.config.min_epochs is not None and self.state.epoch < self.config.min_epochs:
            logger.info('Minimum number of epochs (%d) not reached yet: %d', self.config.min_epochs, self.state.epoch)
            return False
        if self.config.max_num_checkpoint_not_improved is not None and 0 <= self.config.max_num_checkpoint_not_improved and (self.state.checkpoint >= self.config.max_num_checkpoint_not_improved):
            window_improvement = 0.0
            if utils.is_primary_worker():
                window_improvement = abs(self.state.best_metric - self.state.best_metric_history[0])
            if utils.is_distributed():
                window_improvement = utils.broadcast_object(window_improvement)
            if window_improvement <= self.config.checkpoint_improvement_threshold:
                logger.info('Maximum number of not improved checkpoints reached: improvement %f <= %f over %d checkpoints', window_improvement, self.config.checkpoint_improvement_threshold, self.config.max_num_checkpoint_not_improved)
                return True
            else:
                logger.info('Sufficient improvement to continue: %f > %f over %d checkpoints', window_improvement, self.config.checkpoint_improvement_threshold, self.config.max_num_checkpoint_not_improved)
        return False

    def _determine_divergence(self, val_metrics: List[Any]) -> bool:
        last_ppl