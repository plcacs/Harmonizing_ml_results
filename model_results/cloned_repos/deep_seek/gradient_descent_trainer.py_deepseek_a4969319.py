```python
import datetime
import glob
import logging
import math
import os
import re
import time
import warnings
from typing import Optional, Union, List, Dict, Tuple, Any, Type, cast

import torch
from torch.cuda import amp
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
from torch.cuda.amp.grad_scaler import OptState

from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.common import util as common_util, Tqdm, Lazy
from allennlp.common.file_utils import hardlink_or_copy
from allennlp.data.data_loaders.data_loader import DataLoader, TensorDict
from allennlp.models.model import Model
from allennlp.nn.parallel import DdpAccelerator, DdpWrappedModel, TorchDdpAccelerator
from allennlp.nn.util import dist_reduce_sum
from allennlp.training.callbacks import ConsoleLoggerCallback
from allennlp.training.callbacks.confidence_checks import ConfidenceChecksCallback
from allennlp.training.callbacks.backward import MixedPrecisionBackwardCallback
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import LearningRateScheduler
from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.momentum_schedulers.momentum_scheduler import MomentumScheduler
from allennlp.training.moving_average import MovingAverage
from allennlp.training.optimizers import Optimizer
from allennlp.training.trainer import Trainer, TrainerCheckpoint
from allennlp.training.callbacks import TrainerCallback
from allennlp.training import util as training_util

logger = logging.getLogger(__name__)


@Trainer.register("gradient_descent", constructor="from_partial_objects")
class GradientDescentTrainer(Trainer):
    def __init__(
        self,
        model: Model,
        optimizer: torch.optim.Optimizer,
        data_loader: DataLoader,
        patience: Optional[int] = None,
        validation_metric: Union[str, List[str]] = "-loss",
        validation_data_loader: Optional[DataLoader] = None,
        num_epochs: int = 20,
        serialization_dir: Optional[Union[str, os.PathLike]] = None,
        checkpointer: Optional[Checkpointer] = None,
        cuda_device: Optional[Union[int, torch.device]] = None,
        grad_norm: Union[float, bool] = False,
        grad_clipping: Optional[float] = None,
        learning_rate_scheduler: Optional[LearningRateScheduler] = None,
        momentum_scheduler: Optional[MomentumScheduler] = None,
        moving_average: Optional[MovingAverage] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        distributed: bool = False,
        local_rank: int = 0,
        world_size: int = 1,
        num_gradient_accumulation_steps: int = 1,
        use_amp: bool = False,
        enable_default_callbacks: bool = True,
        run_confidence_checks: bool = True,
        grad_scaling: bool = True,
        ddp_wrapped_model: Optional[DdpWrappedModel] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            serialization_dir=serialization_dir,
            cuda_device=cuda_device,
            distributed=distributed,
            local_rank=local_rank,
            world_size=world_size,
        )

        if "run_sanity_checks" in kwargs:
            warnings.warn(
                "'run_sanity_checks' is deprecated, please use 'run_confidence_checks' instead.",
                DeprecationWarning,
            )
            run_confidence_checks = kwargs["run_sanity_checks"]

        self.model = model

        self.data_loader = data_loader
        self.data_loader.set_target_device(self.cuda_device)
        self._validation_data_loader = validation_data_loader
        if self._validation_data_loader is not None:
            self._validation_data_loader.set_target_device(self.cuda_device)
        self.optimizer = optimizer

        if patience is None:
            if validation_data_loader is not None:
                logger.warning(
                    "You provided a validation dataset but patience was set to None, "
                    "meaning that early stopping is disabled"
                )
        elif (not isinstance(patience, int)) or patience <= 0:
            raise ConfigurationError(
                '{} is an invalid value for "patience": it must be a positive integer '
                "or None (if you want to disable early stopping)".format(patience)
            )

        self._metric_tracker = MetricTracker(validation_metric, patience)

        self._num_epochs = num_epochs

        self._checkpointer = checkpointer

        self._grad_norm = grad_norm
        self._grad_clipping = grad_clipping

        self._learning_rate_scheduler = learning_rate_scheduler
        self._momentum_scheduler = momentum_scheduler
        self._moving_average = moving_average

        self._callbacks = callbacks or []
        default_callbacks = list(DEFAULT_CALLBACKS) if enable_default_callbacks else []

        if run_confidence_checks:
            default_callbacks.append(ConfidenceChecksCallback)
        for callback_cls in default_callbacks:
            for callback in self._callbacks:
                if callback.__class__ == callback_cls:
                    break
            else:
                self._callbacks.append(callback_cls(self._serialization_dir))

        self._num_gradient_accumulation_steps = num_gradient_accumulation_steps

        self._ddp_wrapped_model = ddp_wrapped_model
        if distributed:
            if ddp_wrapped_model is None:
                raise ValueError("trainer requires 'ddp_wrapped_model' for distributed training")
            if self._checkpointer is not None:
                self._checkpointer.state_is_sharded = ddp_wrapped_model.is_sharded

        self._scaler: Optional[amp.GradScaler] = None
        self._use_amp = use_amp
        if self._use_amp:
            if self.cuda_device == torch.device("cpu"):
                raise ValueError("Using AMP requires a cuda device")
            if grad_scaling:
                if self._ddp_wrapped_model is None:
                    self._scaler = amp.GradScaler()
                else:
                    self._scaler = self._ddp_wrapped_model.init_grad_scaler()

        self._epochs_completed: int = 0
        self._start_after_epochs_completed: int = 0
        self._batches_in_epoch_completed: int = 0
        self._start_after_batches_in_epoch_completed: int = 0
        self._best_model_filename: Optional[str] = None
        self._should_validate_this_epoch: bool = True
        self._total_batches_completed: int = 0

    @property
    def _pytorch_model(self) -> Model:
        if self._ddp_wrapped_model is None:
            return self.model
        return self._ddp_wrapped_model.model

    def clip_gradient(self) -> None:
        if self._grad_clipping is not None:
            if self._scaler is not None:
                optimizer_state = self._scaler._per_optimizer_states[id(self.optimizer)]
                if optimizer_state["stage"] is not OptState.UNSCALED:
                    self._scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_value_(
                [p for p in self.model.parameters() if p.grad is not None], self._grad_clipping
            )

    def rescale_gradients(self) -> Optional[float]:
        if not isinstance(self._grad_norm, bool):
            if self._scaler is not None:
                self._scaler.unscale_(self.optimizer)
            if self._ddp_wrapped_model is not None:
                return self._ddp_wrapped_model.clip_grad_norm_(self._grad_norm).item()
            else:
                parameters_to_clip = [p for p in self.model.parameters() if p.grad is not None]
                return clip_grad_norm_(parameters_to_clip, self._grad_norm).item()
        elif self._grad_norm:
            parameters_to_clip = [p for p in self.model.parameters() if p.grad is not None]
            return torch.norm(
                torch.stack([torch.norm(p.grad.detach()) for p in parameters_to_clip])
            ).item()
        else:
            return None

    def batch_outputs(self, batch: TensorDict, for_training: bool) -> Dict[str, torch.Tensor]:
        output_dict = self._pytorch_model(**batch)

        if for_training:
            try:
                assert "loss" in output_dict
                regularization_penalty = self.model.get_regularization_penalty()

                if regularization_penalty is not None:
                    output_dict["reg_loss"] = regularization_penalty
                    output_dict["loss"] += regularization_penalty

            except AssertionError:
                if for_training:
                    raise RuntimeError(
                        "The model you are trying to optimize does not contain a"
                        " 'loss' key in the output of model.forward(inputs)."
                    )

        return output_dict

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
        cpu_memory_usage = []
        for worker, memory in common_util.peak_cpu_memory().items():
            cpu_memory_usage.append((worker, memory))
            logger.info(f"Worker {worker} memory usage: {common_util.format_size(memory)}")
        gpu_memory_usage = []
        for gpu, memory in common_util.peak_gpu_memory().items():
            gpu_memory_usage.append((gpu, memory))
            logger.info(f"GPU {gpu} memory usage: {common_util.format_size(memory)}")

        regularization_penalty = self.model.get_regularization_penalty()

        train_loss = 0.0
        train_reg_loss: Optional[float] = None if regularization_penalty is None else 0.0
        batch_reg_loss: Optional[float] = None if regularization_penalty is None else 0.0

        self._pytorch_model.train()

        batch_generator = iter(self.data_loader)
        batch_group_generator = common_util.lazy_groups_of(
            batch_generator, self._num_gradient_accumulation_steps
        )

        logger.info("Training")

        num_training_batches: Union[int, float]
        try:
            len_data_loader = len(self.data_loader)
            num_training_batches = math.ceil(
                len_data_loader / self._num_gradient_accumulation_steps
            )
        except TypeError:
            num_training_batches = float("inf")

        if self._primary:
            batch_group_generator_tqdm = Tqdm.tqdm(
                batch_group_generator, total=num_training_batches
            )
        else:
            batch_group_generator_tqdm = batch_group_generator

        done_early = False
        for batch_group in batch_group_generator_tqdm:
            if done_early:
                break

            if self._epochs_completed < self._start_after_epochs_completed or (
                self._epochs_completed == self._start_after_epochs_completed
                and self._batches_in_epoch_completed < self._start_after_batches_in_epoch_completed
            ):
                self._batches_in_epoch_completed += 1
                self._total_batches_completed += 1
                continue

            self.optimizer.zero_grad()

            batch_loss = 0.0
            batch_group_outputs = []
            for batch in batch_group:
                if self._distributed:
                    done = torch.tensor(0, device=self.cuda_device)
                    torch.distributed.all_reduce(done, torch.distributed.ReduceOp.SUM)
                    if done.item() > 0:
                        done_early = True
                        logger.warning(
                            f"Worker {torch.distributed.get_rank()} finishing training early! "
                            "This implies that there is an imbalance in your training "
                            "data across the workers and that some amount of it will be "
                            "ignored. A small amount of this is fine, but a major imbalance "
                            "should be avoided. Note: This warning will appear unless your "
                            "data is perfectly balanced."
                        )
                        break

                with amp.autocast(self._use_amp):
                    batch_outputs = self.batch_outputs(batch, for_training=True)
                    batch_group_outputs.append(batch_outputs)
                    loss = batch_outputs["loss"]
                    reg_loss = batch_outputs.get("reg_loss")
                    if torch.isnan(loss):
                        raise ValueError("nan loss encountered")
                    loss = loss / len(batch_group)

                    batch_loss += loss.item()
                    if reg_loss is not None:
                        reg_loss = reg_loss / len(batch_group)
                        batch_reg_loss = reg_loss.item()
                        train_reg_loss += batch_reg_loss  # type: ignore

                backward_called = False
                for callback in self._callbacks:
                    backward_called |= callback.on_backward(self, batch_outputs, backward_called)
                if not backward_called:
                    if self._scaler is not None:
                        MixedPrecisionBackwardCallback(self._serialization_dir).on_backward(
                            self, batch_outputs, backward_called
                        )
                    else:
                        loss.backward()

            if len(batch_group_outputs) <= 0:
                continue

            train_loss += batch_loss

            batch_grad_norm = self.rescale_gradients()
            self.clip_gradient()

            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step_batch(self._total_batches_completed + 1)
            if self._momentum_scheduler:
                self._momentum_scheduler.step_batch(self._total_batches_completed + 1)

            if self._scaler is not None:
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                self.optimizer.step()

            if self._moving_average is not None:
                self._moving_average.apply(self._total_batches_completed + 1)

            self._batches_in_epoch_completed += 1
            self._total_batches_completed += 1

            metrics = training_util.get_metrics(
                self.model,
                train_loss,
                train_reg_loss,
                batch_loss,
                batch_reg_loss,
                self._batches_in_epoch_completed,
            )

            for callback in self._callbacks:
                callback.on_batch(
                    self,
                    batch_group,
                    batch_group_outputs,
                    metrics,
                    epoch,
                    self._batches_in_epoch_completed,
                    is_training=True,
                    is_primary=self._primary,
                    batch_grad_norm=batch_grad_norm,
                )

            if self._primary:
                description = training_util.description_from_metrics(metrics)
                batch_group_generator_tqdm.set_description(description, refresh=False)

            if self._checkpointer is not None:
                self._checkpointer.maybe_save_checkpoint(
                    self, self._epochs_completed, self._batches_in_epoch_completed
                )

        if self._distributed and not done_early:
            done = torch.tensor(1, device=self.cuda_device)
            torch.distributed.all_reduce(done, torch.distributed.ReduceOp.SUM)
            assert done.item()

        if self._distributed:
            dist.barrier()

        if self._epochs_completed < self._start_after_epochs_completed or (
            self._epochs_completed == self._start_after_epochs_completed
            and self._batches_in_epoch_completed - 1 < self._start_after_batches_in_epoch_completed
        ):
            metrics = {}
        else:
            train_loss = dist_reduce_sum(train_loss)
            num_batches = dist_reduce_sum(self._batches_in_epoch_completed)
            if train_reg_loss is not None:
                train_reg_loss = dist_reduce_sum(train_reg_loss)

            metrics = training_util.get_metrics(
                self.model,
                train_loss,
                train_reg_loss,
                batch_loss=None,
                batch_reg_loss=None,
                num_batches=num_batches,
                reset=True,
            )

        for (worker, memory) in cpu_memory_usage:
            metrics["worker_" + str(worker) + "_memory_MB"] = memory / (1024 * 1024)
        for (gpu_num, memory) in gpu_memory_usage:
            metrics["gpu_" + str(gpu_num) + "_memory_MB"] = memory / (1024 * 1024)
        return metrics

    def _validation_loss(self, epoch: int) -> Tuple[float, Optional[float], int]:
        logger.info("Validating")

        self._pytorch_model.eval()

        if self._moving_average is not None:
            self._moving_average.assign_average_value()
        try:
            if self._validation_data_loader is not None:
                validation_data_loader = self._validation_data_loader
            else:
                raise ConfigurationError(
                    "Validation results cannot be calculated without a validation_data_loader"
                )

            regularization_penalty = self.model.get_regularization_penalty()

            if self._primary:
                val_generator_tqdm = Tqdm.tqdm(validation_data_loader)
            else:
                val_generator_tqdm = validation_data_loader

            batches_this_epoch = 0
            val_loss = 0.0
            val_batch_loss = 0.0
            val_reg_loss: Optional[float] = None if regularization_penalty is None else 0.0
            val_batch_reg_loss: Optional[float] = None if regularization_penalty is None else 0.0
            done_early = False
            for batch in val_generator_tqdm:
                if self._distributed:
                    done = torch.tensor(0, device=self.cuda_device)
                    torch.distributed.all_reduce(done, torch.distributed.ReduceOp.SUM)
                   