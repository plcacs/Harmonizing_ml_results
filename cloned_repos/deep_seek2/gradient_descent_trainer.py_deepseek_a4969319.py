import datetime
import glob
import logging
import math
import os
import re
import time
import warnings
from typing import Optional, Union, List, Dict, Tuple, Any, Type

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

        self._checkpointer: Optional[Checkpointer] = checkpointer

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
    def _pytorch_model(self) -> torch.nn.Module:
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
        train_reg_loss = None if regularization_penalty is None else 0.0
        batch_reg_loss = None if regularization_penalty is None else 0.0

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
            logger.info(
                f"Worker {torch.distributed.get_rank()} completed its entire epoch (training)."
            )
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
            val_reg_loss = None if regularization_penalty is None else 0.0
            val_batch_reg_loss = None if regularization_penalty is None else 0.0
            done_early = False
            for batch in val_generator_tqdm:
                if self._distributed:
                    done = torch.tensor(0, device=self.cuda_device)
                    torch.distributed.all_reduce(done, torch.distributed.ReduceOp.SUM)
                    if done.item() > 0:
                        done_early = True
                        logger.warning(
                            f"Worker {torch.distributed.get_rank()} finishing validation early! "
                            "This implies that there is an imbalance in your validation "
                            "data across the workers and that some amount of it will be "
                            "ignored. A small amount of this is fine, but a major imbalance "
                            "should be avoided. Note: This warning will appear unless your "
                            "data is perfectly balanced."
                        )
                        break

                with amp.autocast(self._use_amp):
                    batch_outputs = self.batch_outputs(batch, for_training=False)
                    loss = batch_outputs.get("loss")
                    reg_loss = batch_outputs.get("reg_loss")
                    if loss is not None:
                        batches_this_epoch += 1
                        val_batch_loss = loss.item()
                        val_loss += val_batch_loss
                        if reg_loss is not None:
                            val_batch_reg_loss = reg_loss.item()
                            val_reg_loss += val_batch_reg_loss  # type: ignore

                val_metrics = training_util.get_metrics(
                    self.model,
                    val_loss,
                    val_reg_loss,
                    val_batch_loss,
                    val_batch_reg_loss,
                    batches_this_epoch,
                )

                description = training_util.description_from_metrics(val_metrics)
                if self._primary:
                    val_generator_tqdm.set_description(description, refresh=False)

                for callback in self._callbacks:
                    callback.on_batch(
                        self,
                        [batch],
                        [batch_outputs],
                        val_metrics,
                        epoch,
                        batches_this_epoch,
                        is_training=False,
                        is_primary=self._primary,
                    )

            if self._distributed and not done_early:
                logger.warning(
                    f"Worker {torch.distributed.get_rank()} completed its entire epoch (validation)."
                )
                done = torch.tensor(1, device=self.cuda_device)
                torch.distributed.all_reduce(done, torch.distributed.ReduceOp.SUM)
                assert done.item()

            return val_loss, val_reg_loss, batches_this_epoch
        finally:
            if self._moving_average is not None:
                self._moving_average.restore()

    def train(self) -> Dict[str, Any]:
        try:
            self._maybe_restore_checkpoint()
        except RuntimeError as e:
            configuration_error = ConfigurationError(
                f"Could not recover training from the checkpoint in {self._serialization_dir}. "
                "Did you mean to output to a different serialization directory or delete the "
                "existing serialization directory?"
            )
            configuration_error.__cause__ = e
            raise configuration_error

        for callback in self._callbacks:
            callback.on_start(self, is_primary=self._primary)

        epoch = None
        metrics = None

        try:
            metrics, epoch = self._try_train()
            return metrics
        finally:
            if self._primary:
                self._finalize_best_model_state()
            for callback in self._callbacks:
                callback.on_end(self, metrics=metrics, epoch=epoch, is_primary=self._primary)

    def _try_train(self) -> Tuple[Dict[str, Any], int]:
        logger.info("Beginning training.")

        val_metrics: Dict[str, float] = {}
        metrics: Dict[str, Any] = {}
        training_start_time = None

        metrics["best_epoch"] = self._metric_tracker.best_epoch
        for key, value in self._metric_tracker.best_epoch_metrics.items():
            metrics["best_validation_" + key] = value

        for epoch in range(self._num_epochs):
            epoch_start_time = time.time()
            train_metrics = self._train_epoch(epoch)

            if self._epochs_completed < self._start_after_epochs_completed:
                self._epochs_completed += 1
                self._batches_in_epoch_completed = 0
                continue
            if training_start_time is None:
                training_start_time = epoch_start_time

            for key, value in train_metrics.items():
                if key.startswith("gpu_") and key.endswith("_memory_MB"):
                    metrics["peak_" + key] = max(metrics.get("peak_" + key, 0), value)
                elif key.startswith("worker_") and key.endswith("_memory_MB"):
                    metrics["peak_" + key] = max(metrics.get("peak_" + key, 0), value)

            this_epoch_val_metric: Optional[float] = None
            if self._should_validate_this_epoch and self._validation_data_loader is not None:
                with torch.no_grad():
                    val_loss, val_reg_loss, num_batches = self._validation_loss(epoch)

                    if self._distributed:
                        dist.barrier()

                    val_loss = dist_reduce_sum(val_loss)
                    num_batches = dist_reduce_sum(num_batches)
                    if val_reg_loss is not None:
                        val_reg_loss = dist_reduce_sum(val_reg_loss)

                    val_metrics = training_util.get_metrics(
                        self.model,
                        val_loss,
                        val_reg_loss,
                        batch_loss=None,
                        batch_reg_loss=None,
                        num_batches=num_batches,
                        reset=True,
                    )

                    this_epoch_val_metric = self._metric_tracker.combined_score(val_metrics)
                    self._metric_tracker.add_metrics(val_metrics)

            training_elapsed_time = time.time() - training_start_time
            metrics["training_duration"] = str(datetime.timedelta(seconds=training_elapsed_time))
            metrics["epoch"] = epoch

            for key, value in train_metrics.items():
                metrics["training_" + key] = value
            for key, value in val_metrics.items():
                metrics["validation_" + key] = value

            if self._should_validate_this_epoch and self._metric_tracker.is_best_so_far():
                metrics["best_epoch"] = epoch
                for key, value in val_metrics.items():
                    metrics["best_validation_" + key] = value

                self._metric_tracker.best_epoch_metrics = val_metrics

            if self._serialization_dir and self._primary:
                common_util.dump_metrics(
                    os.path.join(self._serialization_dir, f"metrics_epoch_{epoch}.json"),
                    metrics,
                )

            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step(this_epoch_val_metric)
            if self._momentum_scheduler:
                self._momentum_scheduler.step(this_epoch_val_metric)
            for callback in self._callbacks:
                callback.on_epoch(self, metrics=metrics, epoch=epoch, is_primary=self._primary)

            self._epochs_completed += 1
            self._batches_in_epoch_completed = 0

            checkpoint_saved = False
            if self._checkpointer is not None:
                checkpoint_saved = self._checkpointer.maybe_save_checkpoint(
                    self, self._epochs_completed, self._batches_in_epoch_completed
                )

                if self._distributed:
                    dist.barrier()

            if (
                self._should_validate_this_epoch
                and self._serialization_dir
                and self._metric_tracker.is_best_so_far()
            ):
                should_save_model_state: bool
                if self._ddp_wrapped_model is not None and self._ddp_wrapped_model.is_sharded:
                    self._best_model_filename = os.path.join(
                        self._serialization_dir, f"best_w{self._rank}.th"
                    )
                    should_save_model_state = True
                else:
                    self._best_model_filename = os.path.join(self._serialization_dir, "best.th")
                    should_save_model_state = self._primary

                if should_save_model_state:
                    if self._moving_average is None:
                        if self._checkpointer is not None and checkpoint_saved:
                            last_checkpoint = self._checkpointer.find_latest_checkpoint()
                            assert last_checkpoint is not None
                            model_state_file, _ = last_checkpoint
                            if os.path.exists(self._best_model_filename):
                                os.remove(self._best_model_filename)
                            hardlink_or_copy(model_state_file, self._best_model_filename)
                        else:
                            self._save_model_state(self._best_model_filename)
                    else:
                        self._moving_average.assign_average_value()
                        try:
                            self._save_model_state(self._best_model_filename)
                        finally:
                            self._moving_average.restore()

            if self._distributed:
                dist.barrier()

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", datetime.timedelta(seconds=epoch_elapsed_time))

            if self._metric_tracker.should_stop_early():
                logger.info("Ran out of patience. Stopping training.")
                break

            if epoch < self._num_epochs - 1:
                time_per_epoch = training_elapsed_time / (
                    (epoch + 1) - self._start_after_epochs_completed
                )
                estimated_time_remaining = (
                    time_per_epoch * self._num_epochs
                ) - training_elapsed_time
                formatted_time = str(datetime.timedelta(seconds=int(estimated_time_remaining)))
                logger.info("Estimated training time remaining: %s", formatted_time)
        else:
            epoch = self._num_epochs - 1

        if self._best_model_filename is None or self._metric_tracker.is_best_so_far():
            self._finalize_model()
        else:
            self._load_model_state(self._best_model_filename)

        return metrics, epoch

    def _save_model_state(self, path: str) -> None:
        if self._ddp_wrapped_model is not None:
            torch.save(self._ddp_wrapped_model.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), path)

    def _load_model_state(self, path: str) -> None:
        device = torch.device("cpu")
        if self._ddp_wrapped_model is not None:
            self._ddp_wrapped_model.load_state_dict(torch.load(path, map_location=device))
        else:
            self._pytorch_model.load_state_dict(torch.load(path, map_location=device))

    def _finalize_model(self) -> None:
        if self._moving_average is not None:
            self._moving_average.assign_average_value()

    def _finalize_best_model_state(self) -> None:
        if (
            self._serialization_dir
            and self._ddp_wrapped_model is not None
            and self._ddp_wrapped_model.is_sharded
        ):
            logger.info("Consolidating sharded model states")
            sharded_model_state_files = list(
                glob.iglob(os.path.join(self._serialization_dir, "best_w*.th"))
            full_model_state = self._ddp_wrapped_model.consolidate_sharded_state(
                sharded_model_state_files
            )
            self._best_model_filename = os.path.join(self._serialization_dir, "best.th")
            torch.save(full_model_state, self._best_model_filename)

    def get_checkpoint_state(self) -> Optional[TrainerCheckpoint]:
        if self._distributed:
            assert self._ddp_wrapped_model is not None
            if self._ddp_wrapped_model.is_sharded or self._primary:
                model_state = self._ddp_wrapped_model.state_dict()
            else:
                return None
        else:
            model_state = self.model.state_dict()

        training_states = {
            "version": 1,
            "metric_tracker": self._metric_tracker.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "callbacks": [cb.state_dict() for cb in self._callbacks],
            "epochs_completed": self._epochs_completed,
            "batches_in_epoch_completed": self._batches_in_epoch_completed,
            "best_model_filename": self._best_model_filename,
        }

        if self._learning_rate_scheduler is not None:
            training_states["learning_rate_scheduler"] = self._learning_rate_scheduler.state_dict()
        if self._momentum_scheduler is not None:
            training_states["momentum_scheduler"] = self._momentum_scheduler.state_dict()
        if self._moving_average is not None:
            training_states["moving_average"] = self._moving_average.state_dict()

        return TrainerCheckpoint(model_state, training_states)

    def _maybe_restore_checkpoint(self) -> None:
        if self._checkpointer is None:
            return

        state = self._checkpointer.load_checkpoint()
        if state is None:
            self._start_after_epochs_completed = 0
            self._start_after_batches_in_epoch_completed = 0
            self._best_model_filename = None
            return

        model_state, training_state = state
        if training_state["version"] != 1:
            raise ValueError(
                f"This version of {self.__class__.__name__} only supports checkpoints of version 1. "
                f"Found version {training_state['version']}"
            )

        if self._distributed:
            assert self._ddp_wrapped_model is not None
            self._ddp_wrapped_model.load_state_dict(model_state)
        else:
            self._pytorch_model.load_state_dict(model_state)

        self._metric_tracker.load_state_dict(training_state["metric_tracker"])
        self.optimizer.load_state_dict(training_state["optimizer"])

        for cb, state_dict in zip(self._callbacks, training_state["callbacks"]):
            cb.load_state_dict(state_dict)

        if self._learning_rate_scheduler is not None:
            self._learning_rate_scheduler.load_state_dict(training_state["learning_rate_scheduler"])
        if self._momentum_scheduler is not None:
            self._momentum_scheduler.load_state_dict(training_state["momentum_scheduler"])
        if self._moving_average is not None:
            self._moving_average.load_state_dict(training_state["moving_average"])

        self._start_after_epochs_completed = training_state["epochs_completed"]
        self._start_after_batches_in_epoch_completed = training_state["batches_in_epoch_completed"]
        self._best_model_filename = training_state["best_model_filename"]

    @classmethod
    def from_partial_objects(
        cls,
        model: Model,
        serialization_dir: str,
        data_loader: DataLoader,
        validation_data_loader: Optional[DataLoader] = None,
        local_rank: int = 0,
        patience: Optional[int] = None,
        validation_metric: Union[str, List[str]] = "-loss",
        num_epochs: int = 20,
        cuda_device: Optional[Union[int, torch.device]] = None,
        grad_norm: Union[float, bool] = False,
        grad_clipping: Optional[float] = None,
        distributed: bool = False,
        world_size: int = 1,
        num_gradient_accumulation_steps: int = 1,
        use_amp: bool = False,
        no_grad: Optional[List[str]] = None,
        optimizer: Lazy[Optimizer] = Lazy(Optimizer.default),
        learning_rate_scheduler: Optional[Lazy[LearningRateScheduler]] = None,
        momentum_scheduler: Optional[Lazy[MomentumScheduler]] = None,
        moving_average: Optional[Lazy[MovingAverage]] = None,
        checkpointer: Optional[Lazy[Checkpointer]] = Lazy(Checkpointer),
        callbacks: Optional[List[Lazy[TrainerCallback]]] = None,
        enable_default_callbacks: bool = True,
        run_confidence_checks: bool = True,
        grad_scaling: bool = True,
        ddp_accelerator: Optional[DdpAccelerator] = None,
        **kwargs: Any,
    ) -> Trainer:
        if cuda_device is None:
            from torch import cuda

            if cuda.device_count() > 0:
                cuda_device = 0
            else:
                cuda_device = -1

        check_for_gpu(cuda_device)
        ddp_wrapped_model: Optional[DdpWrappedModel] = None
        if distributed:
            if ddp_accelerator is None:
                ddp_accelerator = TorchDdpAccelerator(cuda_device=cuda_device)
            model, ddp_wrapped_model = ddp_accelerator.wrap_model(model)
        else:
            if cuda_device >= 0:
                model = model.cuda(cuda_device)

        pytorch_model = model if ddp_wrapped_model is None else ddp_wrapped_model.model

        if no_grad:
            for name, parameter in pytorch_model.named_parameters():
                if any(re.search(regex, name) for regex in no_grad):
                    parameter.requires_grad_(False)

        parameters = [[n, p] for n, p in pytorch_model.named_parameters() if p.requires_grad]
        optimizer_ = optimizer.construct(model_parameters=parameters)

        common_util.log_frozen_and_tunable_parameter_names(pytorch_model)

        batches_per_epoch: Optional[int]
        try:
            batches_per_epoch = len(data_loader)
            batches_per_epoch = math.ceil(batches_per_epoch / num_gradient_accumulation_steps)
        except TypeError:
            batches_per_epoch = None

        moving_average_ = (
            None if moving_average is None else moving_average.