# Copyright 2017--2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

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
from typing import Any, Callable, Dict, List, Optional, Iterable, Tuple, Union, Set

import numpy as np
import torch
import torch.distributed

# Optional imports. Import errors are not an issue because these modules are
# only used when certain settings are activated. We check that these modules
# can be imported before activating the settings.
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

    def forward(self, source: torch.Tensor,
                source_length: torch.Tensor,
                target: torch.Tensor,
                target_length: torch.Tensor,
                labels: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor,
                                                          List[torch.Tensor],
                                                          List[torch.Tensor]]:
        model_outputs = self.model(source, source_length, target, target_length)
        if utils.using_deepspeed():
            # Guarantee model outputs are float32 before computing losses.
            # Computing losses in DeepSpeed float16 mode can lead to overflow.
            model_outputs = {output_name: output.to(torch.float32) for (output_name, output) in model_outputs.items()}
        loss_outputs = [loss_function(model_outputs, labels) for loss_function in self.losses]
        loss_values, num_samples = zip(*loss_outputs)
        sum_losses = sum(loss_values) if len(loss_values) > 1 else loss_values[0]
        return sum_losses, loss_values, num_samples  # type: ignore


@dataclass
class TrainerConfig(Config):
    output_dir: str
    early_stopping_metric: str
    max_params_files_to_keep: int
    keep_initializations: bool
    max_params_files_to_cache: int
    cache_strategy: str
    cache_metric: str
    checkpoint_interval: int
    max_num_checkpoint_not_improved: int
    checkpoint_improvement_threshold: float
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

    __slots__ = ['num_not_improved', 'epoch', 'checkpoint', 'best_checkpoint', 'batches', 'updates', 'samples',
                 'metrics', 'start_tic', '_tic_last_time_elapsed', '_time_elapsed', 'early_stopping_metric',
                 'best_metric', 'best_metric_history', 'best_checkpoint', 'converged', 'diverged']

    def __init__(self, early_stopping_metric: str) -> None:
        self.num_not_improved = 0
        self.epoch = 0
        self.checkpoint = 0
        self.best_checkpoint = 0
        self.batches = 0
        self.updates = 0
        self.samples = 0
        # stores dicts of metric names & values for each checkpoint
        self.metrics = []  # type: List[Dict]
        self.start_tic = time.time()
        self._tic_last_time_elapsed = self.start_tic
        self._time_elapsed = 0.0
        self.early_stopping_metric = early_stopping_metric
        self.best_metric = C.METRIC_WORST[early_stopping_metric]
        # List of the last N best metrics, used for threshold-based stopping
        self.best_metric_history = deque([self.best_metric])
        self.best_checkpoint = 0
        self.converged = False
        self.diverged = False

    def save(self, fname: str) -> None:
        """
        Saves this training state to fname.
        """
        self.update_time_elapsed()
        assert len(self.metrics) == self.checkpoint
        with open(fname, "wb") as fp:
            pickle.dump(self, fp)

    @staticmethod
    def load(fname: str) -> 'TrainState':
        """
        Loads a training state from fname.
        """
        with open(fname, "rb") as fp:
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

    def __init__(self,
                 config: TrainerConfig,
                 optimizer_config: optimizers.OptimizerConfig,
                 sockeye_model: model.SockeyeModel,
                 model_object: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: Optional[lr_scheduler.LearningRateScheduler],
                 zero_grad_kwargs: Dict[str, Any],
                 loss_functions: List[loss.Loss],
                 device: torch.device,
                 using_amp: bool = False,
                 using_apex_amp: bool = False,
                 custom_metrics_logger: Optional[Callable] = None,
                 checkpoint_callback: Optional[Callable] = None) -> None:
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
        self.state = None  # type: Optional[TrainState]
        self._speedometer = Speedometer(frequency=C.MEASURE_SPEED_EVERY, auto_reset=False)
        self._custom_metrics_logger = custom_metrics_logger
        self._tflogger = TensorboardLogger(logdir=os.path.join(self.config.output_dir, C.TENSORBOARD_NAME))
        self.checkpoint_callback = checkpoint_callback

    def fit(self,
            train_iter: data_io.BaseParallelSampleIter,
            validation_iter: data_io.BaseParallelSampleIter,
            checkpoint_decoder: Optional[checkpoint_decoder.CheckpointDecoder] = None) -> TrainState:
        logger.info("Early stopping by optimizing '%s'", self.config.early_stopping_metric)

        if utils.is_primary_worker() and self.config.early_stopping_metric in C.METRICS_REQUIRING_DECODER:
            utils.check_condition(checkpoint_decoder is not None,
                                  "%s requires CheckpointDecoder" % self.config.early_stopping_metric)

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
            logger.info("Training started.")

        tic = time.time()

        if self.config.max_checkpoints is not None:
            self.config.max_updates = self.state.updates + self.config.max_checkpoints * self.config.checkpoint_interval
            logger.info("Resetting max_updates to %d + %d * %d = %d in order to implement stopping "
                        "after (an additional) %d checkpoints.",
                        self.state.updates,
                        self.config.max_checkpoints,
                        self.config.checkpoint_interval,
                        self.config.max_updates,
                        self.config.max_checkpoints)

        # At the start of training, the checkpoint is only up to date if it has
        # just been loaded (resuming training with an existing model directory).
        checkpoint_up_to_date = resume_training
        while True:
            if self.config.max_epochs is not None and self.state.epoch == self.config.max_epochs:
                logger.info("Maximum # of epochs (%s) reached.", self.config.max_epochs)
                if not checkpoint_up_to_date:
                    time_cost = time.time() - tic
                    self._create_checkpoint(checkpoint_decoder, time_cost, train_iter, validation_iter)
                break

            if self.config.max_updates is not None and self.state.updates == self.config.max_updates:
                logger.info("Maximum # of updates (%s) reached.", self.config.max_updates)
                if not checkpoint_up_to_date:
                    time_cost = time.time() - tic
                    self._create_checkpoint(checkpoint_decoder, time_cost, train_iter, validation_iter)
                break

            if self.config.max_samples is not None and self.state.samples >= self.config.max_samples:
                logger.info("Maximum # of samples (%s) reached", self.config.max_samples)
                if not checkpoint_up_to_date:
                    time_cost = time.time() - tic
                    self._create_checkpoint(checkpoint_decoder, time_cost, train_iter, validation_iter)
                break

            did_grad_step = self._step(batch=train_iter.next())
            checkpoint_up_to_date = checkpoint_up_to_date and not did_grad_step

            if not train_iter.iter_next():
                self.state.epoch += 1
                train_iter.reset()

            if self.state.updates > 0 and self.state.batches % (
                    self.config.checkpoint_interval * self.config.update_interval) == 0:
                time_cost = time.time() - tic
                self._create_checkpoint(checkpoint_decoder, time_cost, train_iter, validation_iter)
                checkpoint_up_to_date = True

                if self.config.max_seconds is not None and self.state.time_elapsed >= self.config.max_seconds:
                    logger.info("Maximum # of seconds (%s) reached. Training ran for %d seconds.",
                                self.config.max_seconds, self.state.time_elapsed)
                    break

                if self.state.converged or self.state.diverged:
                    break

                tic = time.time()

        logger.info("Training finished%s. Best checkpoint: %d. Best validation %s: %.6f",
                    ", can be continued later" if not self.state.converged else "",
                    self.state.best_checkpoint, self.state.early_stopping_metric, self.state.best_metric)

        # Always keep the training state to allow continuing training with
        # different stopping criteria
        if utils.is_primary_worker():
            self._cleanup(keep_training_state=True)

        return self.state

    def _create_checkpoint(self, checkpoint_decoder: checkpoint_decoder.CheckpointDecoder, time_cost: float,
                           train_iter: data_io.BaseParallelSampleIter,
                           validation_iter: data_io.BaseParallelSampleIter) -> None:
        """
        Creates a checkpoint, which will update self.state.converged/self.state.diverged, evaluate validation
        metrics and update the best known parameters accordingly.
        """
        self.state.checkpoint += 1
        train_metrics = [lf.metric for lf in self.loss_functions]
        logger.info("Checkpoint [%d]\tUpdates=%d Epoch=%d Samples=%d Time-cost=%.3f Updates/sec=%.3f",
                    self.state.checkpoint, self.state.updates, self.state.epoch,
                    self.state.samples, time_cost, self.config.checkpoint_interval / time_cost)
        logger.info('Checkpoint [%d]\t%s', self.state.checkpoint,
                    "\t".join("Train-%s" % str(metric) for metric in train_metrics))

        val_metrics = self._evaluate(self.state.checkpoint, validation_iter, checkpoint_decoder)

        has_improved = self._determine_improvement(val_metrics)
        self.state.converged = self._determine_convergence()
        self.state.diverged = self._determine_divergence(val_metrics)
        self._adjust_learning_rate(has_improved)
        if utils.is_primary_worker():
            self._write_and_log_metrics(train_metrics=train_metrics, val_metrics=val_metrics)
        # When using DeepSpeed, all workers participate in saving the training
        # state and model parameters. Otherwise these methods are a no-op for
        # secondary workers.
        self._save_training_state(train_iter)
        self._save_params(use_checkpoint=True)
        if utils.is_primary_worker():
            if has_improved:
                self._update_best_params()
                if not utils.using_deepspeed():
                    # DeepSpeed mode does not support checkpoint reloading
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
        with torch.cuda.amp.autocast(cache_enabled=False) if self.using_amp else utils.no_context():  # type: ignore
            # Forward + loss
            sum_losses, loss_values, num_samples = self.model_object(batch.source, batch.source_length,
                                                                     batch.target, batch.target_length, batch.labels)
        # Backward
        if utils.using_deepspeed():
            # DeepSpeed backward. DeepSpeed handles all loss scaling.
            self.model_object.backward(sum_losses)  # type: ignore
        else:
            if self.config.update_interval > 1:
                # Scale loss by number of batches per update
                # TODO(mdenkows): We currently give equal weight to every batch
                # in every update but batches have subtly different sizes
                # (different numbers of padding tokens). Consider normalizing by
                # relative batch size.
                sum_losses = sum_losses / self.config.update_interval
            if self.using_amp:
                # PyTorch AMP loss scaling
                sum_losses = self._scaler.scale(sum_losses)
            if self.using_apex_amp:
                # Apex AMP loss scaling
                with apex.amp.scale_loss(sum_losses, self.optimizer,
                                         delay_unscale=not is_update_batch) as scaled_sum_losses:
                    # Apex AMP backward
                    scaled_sum_losses.backward()
            else:
                # PyTorch (with/without AMP) backward
                sum_losses.backward()  # type: ignore
        return loss_values, num_samples

    def _step(self, batch: data_io.Batch) -> bool:
        self.state.batches += 1
        self.state.samples += batch.samples
        # We accumulate gradients over N=update_interval batches before running
        # the optimizer to update model weights. Every Nth batch is an update
        # batch.
        is_update_batch = self.state.batches % self.config.update_interval == 0
        self.state.updates += 1 if is_update_batch else 0

        # Forward/loss/backward (compute gradients). In distributed mode,
        # workers accumulate gradients locally for N-1 batches (no_sync), then
        # average the accumulated gradients across workers during the update
        # batch.
        with (self.model_object.model.no_sync() if utils.is_distributed() and not is_update_batch  # type: ignore
        and not utils.using_deepspeed() else utils.no_context()):
            loss_values, num_samples = self._forward_backward(batch, is_update_batch)

        for loss_func, loss_value, num_samples in zip(self.loss_functions, loss_values, num_samples):
            loss_func.metric.update(loss_value.item(), num_samples.item())

        if utils.using_deepspeed():
            self.model_object.step()  # type: ignore
        elif is_update_batch:
            if self.using_amp:
                self._scaler.unscale_(self.optimizer)
            # Clip gradients
            if self.optimizer_config.gradient_clipping_type == C.GRADIENT_CLIPPING_TYPE_ABS:
                torch.nn.utils.clip_grad.clip_grad_value_(self.sockeye_model.parameters(),
                                                          self.optimizer_config.gradient_clipping_threshold)
            elif self.optimizer_config.gradient_clipping_type == C.GRADIENT_CLIPPING_TYPE_NORM:
                torch.nn.utils.clip_grad.clip_grad_norm_(self.sockeye_model.parameters(),
                                                         self.optimizer_config.gradient_clipping_threshold)
            # Set learning rate for current step
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            # Update weights and reset gradients
            if self.using_amp:
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(**self.zero_grad_kwargs)

        self._speedometer(self.state.epoch, self.state.batches,
                          self.state.updates, batch.samples, batch.tokens, (lf.metric for lf in self.loss_functions))
        return is_update_batch

    def _evaluate(self, checkpoint: int, data_iter,
                  checkpoint_decoder: Optional[checkpoint_decoder.CheckpointDecoder]) -> List[loss.LossMetric]:
        """
        Computes loss(es) on validation data and returns their metrics.
        :param data_iter: Validation data iterator.
        :return: List of validation metrics, same order as self.loss_functions.
        """
        # Switch model to eval mode (disable dropout, etc.) to score validation
        # set and run checkpoint decoder.
        self.sockeye_model.eval()

        data_iter.reset()
        val_metrics = [lf.create_metric() for lf in self.loss_functions]
        for batch in data_iter:
            batch = batch.load(device=self.device)
            with torch.inference_mode():
                # Forward: run SockeyeModel directly. The traced model may not
                # fully support switching between train and eval modes depending
                # how much Python logic is used in the various submodules.
                outputs = self.sockeye_model(batch.source, batch.source_length, batch.target, batch.target_length)
                # Guarantee model outputs are float32 before computing losses
                outputs = {name: output.to(torch.float32) for (name, output) in outputs.items()}
                # Loss
                loss_outputs = [loss_function(outputs, batch.labels) for loss_function in self.loss_functions]
                # Update validation metrics for batch
            for loss_metric, (loss_value, num_samples) in zip(val_metrics, loss_outputs):
                loss_metric.update(loss_value.item(), num_samples.item())

        if utils.is_primary_worker():
            # Primary worker runs checkpoint decoder
            decoder_metrics = {}  # type: Dict[str, float]
            if checkpoint_decoder is not None:
                output_name = os.path.join(self.config.output_dir, C.DECODE_OUT_NAME.format(checkpoint=checkpoint))
                decoder_metrics = checkpoint_decoder.decode_and_evaluate(output_name=output_name)
            # Add decoder metrics (if any) to validation metrics
            for metric_name, metric_value in decoder_metrics.items():
                assert metric_name not in val_metrics, "Duplicate validation metric %s" % metric_name
                metric = loss.LossMetric(name=metric_name)
                metric.update(metric_value, num_samples=1)
                val_metrics.append(metric)
        if utils.is_distributed():
            # Primary worker's evaluation is authoritative
            val_metrics = utils.broadcast_object(val_metrics)

        logger.info('Checkpoint [%d]\t%s',
                    self.state.checkpoint, "\t".join("Validation-%s" % str(lm) for lm in val_metrics))

        # Switch model back to train mode to continue training
        self.sockeye_model.train()
        return val_metrics

    def _determine_improvement(self, val_metrics: List[loss.LossMetric]) -> bool:
        """
        Determines whether early stopping metric on validation data improved and updates best value and checkpoint in
        the state.
        :param val_metrics: Validation metrics.
        :return: Whether model has improved on held-out data since last checkpoint.
        """
        value = None
        value_is_better = False
        for val_metric in val_metrics:
            if val_metric.name == self.config.early_stopping_metric:
                value = val_metric.get()
                value_is_better = utils.metric_value_is_better(value,
                                                               self.state.best_metric,
                                                               self.config.early_stopping_metric)
                if value_is_better:
                    logger.info("Validation-%s improved to %f (delta=%f).", self.config.early_stopping_metric,
                                value, abs(value - self.state.best_metric))
                    self.state.best_metric = value
                    self.state.best_checkpoint = self.state.checkpoint
                    self.state.num_not_improved = 0
        assert value is not None, "Early stopping metric %s not found in validation metrics." % self.config.early_stopping_metric
        if not value_is_better:
            self.state.num_not_improved += 1
            logger.info("Validation-%s has not improved for %d checkpoints, best so far: %f",
                        self.config.early_stopping_metric, self.state.num_not_improved, self.state.best_metric)
        # Update best metric history
        self.state.best_metric_history.append(self.state.best_metric)
        if (self.config.max_num_checkpoint_not_improved is not None
                and len(self.state.best_metric_history) > self.config.max_num_checkpoint_not_improved + 1):
            self.state.best_metric_history.popleft()

        return value_is_better

    def _determine_convergence(self) -> bool:
        """
        True if model has converged w.r.t early stopping criteria (patience).
        Order: first check required minimums (samples, updates, epochs), then
        check early stopping criteria (checkpoints not improved).
        """
        if self.config.min_samples is not None and self.state.samples < self.config.min_samples:
            logger.info("Minimum number of samples (%d) not reached yet: %d",
                        self.config.min_samples, self.state.samples)
            return False

        if self.config.min_updates is not None and self.state.updates < self.config.min_updates:
            logger.info("Minimum number of updates (%d) not reached yet: %d",
                        self.config.min_updates, self.state.updates)
            return False

        if self.config.min_epochs is not None and self.state.epoch < self.config.min_epochs:
            logger.info("Minimum number of epochs (%d) not reached yet: %d",
                        self.config.min_epochs, self.state.epoch)
            return False

        if (self.config.max_num_checkpoint_not_improved is not None
                and 0 <= self.config.max_num_checkpoint_not_improved
                and self.state.checkpoint >= self.config.max_num_checkpoint_not_improved):
            # In distrubted mode, the primary worker makes the authoritative
            # calculation of improvement over the window for evaluating stopping
            window_improvement = 0.
            if utils.is_primary_worker():
                window_improvement = abs(self.state.best_metric - self.state.best_metric_history[0])
            if utils.is_distributed():
                window_improvement = utils.broadcast_object(window_improvement)

            # <= to correctly handle threshold == 0
            if window_improvement <= self.config.checkpoint_improvement_threshold:
                logger.info("Maximum number of not improved checkpoints reached: "
                            "improvement %f <= %f over %d checkpoints", window_improvement,
                            self.config.checkpoint_improvement_threshold, self.config.max_num_checkpoint_not_improved)
                return True
            else:
                logger.info("Sufficient improvement to continue: %f > %f over %d checkpoints", window_improvement,
                            self.config.checkpoint_improvement_threshold, self.config.max_num_checkpoint_not_improved)

        return False

    def _determine_divergence(self, val_metrics: List[loss.LossMetric]) -> bool:
        """
        True if last perplexity is infinite or >2*target_vocab_size.
        """
        # (5) detect divergence with respect to the perplexity value at the last checkpoint
        last_ppl = float('nan')
        for metric in val_metrics:
            if metric.name == C.PERPLEXITY:
                last_ppl = metric.get()
                break
        # using a double of uniform distribution's value as a threshold
        if not np.isfinite(last_ppl) or last_ppl > 2 * self.sockeye_model.config.vocab_target_size:
            logger.warning("Model optimization diverged. Last checkpoint's perplexity: %f", last_ppl)
            return True
        return False

    def _adjust_learning_rate(self, has_improved: bool) -> None:
        """
        Adjusts the optimizer learning rate if required and logs it.
        """
        lr = self.optimizer_config.lr
        if self.lr_scheduler is not None:
            if issubclass(type(self.lr_scheduler), lr_scheduler.AdaptiveLearningRateScheduler):
                lr_adjusted = self.lr_scheduler.new_evaluation_result(has_improved)  # type: ignore
            else:
                lr_adjusted = False
            if lr_adjusted and not has_improved and not self.config.no_reload_on_learning_rate_reduce:
                logger.info("Loading model parameters and optimizer states from best checkpoint: %d",
                            self.state.best_checkpoint)
                if os.path.exists(self.best_params_fname):
                    self.sockeye_model.load_parameters(filename=self.best_params_fname, device=self.device)
                if os.path.exists(self.best_optimizer_state_fname):
                    self._load_optimizer_state(self.best_optimizer_state_fname)
            # Assume same learning rate for all param groups
            lr = self.lr_scheduler.get_last_lr()[0]
        logger.info("Checkpoint [%d]\tLearning-rate=%.6f", self.state.checkpoint, lr)

    def _write_and_log_metrics(self,
                               train_metrics: Iterable[loss.LossMetric],
                               val_metrics: Iterable[loss.LossMetric]) -> None:
        """
        Updates metrics for current checkpoint.
        Writes all metrics to the metrics file, optionally logs to tensorboard, and sends metrics to custom logger.
        """
        data = {"epoch": self.state.epoch,
                "learning-rate": (self.optimizer_config.lr if self.lr_scheduler is None
                                  else self.lr_scheduler.get_last_lr()[0]),
                "time-elapsed": self.state.time_elapsed,
                "max-gpu-memory": torch.cuda.max_memory_allocated(self.device),
                "converged": self.state.converged,
                "diverged": self.state.diverged}

        for metric in train_metrics:
            data["%s-train" % metric.name] = metric.get()
        for metric in val_metrics:
            data["%s-val" % metric.name] = metric.get()

        self.state.metrics.append(data)
        utils.write_metrics_file(self.state.metrics, self.metrics_fname)

        self._tflogger.log_metrics(metrics=data, checkpoint=self.state.checkpoint)
        safe_custom_metrics_logger(logging_function=self._custom_metrics_logger,
                                   metrics=data,
                                   global_step=self.state.checkpoint)

    def _update_best_params(self) -> None:
        """
        Updates the params.best link to the latest best parameter file.
        """
        actual_best_params_fname = C.PARAMS_NAME % self.state.best_checkpoint
        if os.path.lexists(self.best_params_fname):
            os.remove(self.best_params_fname)
        utils.fault_tolerant_symlink(actual_best_params_fname, self.best_params_fname)
        logger.info("'%s' now points to '%s'", self.best_params_fname, actual_best_params_fname)

    def _save_params(self, use_checkpoint: bool = False) -> None:
        """
        Saves model parameters at current checkpoint and optionally cleans up
        older parameter files to save disk space.

        :param use_checkpoint: When using DeepSpeed, copy files from the latest
                               checkpoint instead of creating a new checkpoint.
        """
        if utils.using_deepspeed():
            # Copy or create a DeepSpeed checkpoint that can be used to generate
            # a regular Sockeye parameter file at the end of training.
            if use_checkpoint:
                if utils.is_primary_worker():
                    shutil.copytree(src=os.path.join(self.training_state_dirname, C.TRAINING_STATE_DEEPSPEED),
                                    dst=self.current_params_fname)
            else:
                if utils.is_primary_worker() and not os.path.exists(self.current_params_fname):
                    os.mkdir(self.current_params_fname)
                torch.distributed.barrier()
                # All workers save their local shards of the float32 parameters.
                self.model_object.save_checkpoint(self.current_params_fname)  # type: ignore
        elif utils.is_primary_worker():
            self.sockeye_model.save_parameters(self.current_params_fname)
        if utils.is_primary_worker():
            # With or without DeepSpeed
            cleanup_params_files(self.config.output_dir, self.config.max_params_files_to_keep, self.state.checkpoint,
                                 self.state.best_checkpoint, self.config.keep_initializations,
                                 self.config.max_params_files_to_cache, self.config.cache_metric,
                                 self.config.cache_strategy)

    def _save_optimizer_state(self, fname: str) -> None:
        torch.save(self.optimizer.state_dict(), fname)
        logger.info('Saved optimizer state to "%s"', fname)

    def _load_optimizer_state(self, fname: str) -> None:
        self.optimizer.load_state_dict(torch.load(fname, map_location=self.device))
        logger.info('Loaded optimizer state from "%s"', fname)

    def _save_lr_scheduler(self, fname: str) -> None:
        if self.lr_scheduler is not None:
            torch.save(self.lr_scheduler.state_dict(), fname)
            logger.info("Saved '%s' to '%s'", self.lr_scheduler, fname)

    def _load_lr_scheduler(self, fname: str) -> None:
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(torch.load(fname))
            logger.info("Loaded '%s' from '%s'", self.lr_scheduler, fname)

    def _save_training_state(self, train_iter: data_io.BaseParallelSampleIter) -> None:
        """
        Saves current training state.
        """
        # Create temporary directory for storing the state of the optimization process
        training_state_dirname = os.path.join(self.config.output_dir, C.TRAINING_STATE_TEMP_DIRNAME)
        if utils.is_primary_worker() and not os.path.exists(training_state_dirname):
            os.mkdir(training_state_dirname)
        if utils.is_distributed():
            torch.distributed.barrier()

        if utils.using_deepspeed():
            # DeepSpeed saves parameters, optimizer state, and learning rate
            # scheduler in a single checkpoint file. All workers need to call
            # `save_checkpoint()`.
            self.model_object.save_checkpoint(os.path.join(training_state_dirname,  # type: ignore
                                                           C.TRAINING_STATE_DEEPSPEED))
        elif utils.is_primary_worker():
            # Otherwise, only the primary worker saves the following.
            # (1) Parameters: link current file
            params_base_fname = C.PARAMS_NAME % self.state.checkpoint
            params_file = os.path.join(training_state_dirname, C.TRAINING_STATE_PARAMS_NAME)
            if os.path.exists(params_file):
                os.unlink(params_file)
            utils.fault_tolerant_symlink(os.path.join("..", params_base_fname), params_file)

            # (2) Optimizer state
            opt_state_fname = os.path.join(training_state_dirname, C.OPT_STATE_LAST)
            self._save_optimizer_state(opt_state_fname)

            # (3) lr_scheduler
            lr_scheduler_fname = os.path.join(training_state_dirname, C.LR_SCHEDULER_LAST)
            self._save_lr_scheduler(lr_scheduler_fname)

        # Secondary workers are done. Only the primary worker runs everything
        # beyond this point.
        if not utils.is_primary_worker():
            return

        # (4