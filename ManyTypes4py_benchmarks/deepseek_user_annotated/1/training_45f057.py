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
        self.metrics: List[Dict] = []
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
        self.state: Optional[TrainState] = None
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
        #