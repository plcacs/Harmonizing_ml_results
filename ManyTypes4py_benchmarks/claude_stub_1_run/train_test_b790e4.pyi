```pyi
import argparse
import logging
from collections import OrderedDict, Counter
from typing import Optional, List, Dict, Any, Callable, Set

import pytest
import torch
from allennlp.version import VERSION
from allennlp.commands.train import Train, train_model, train_model_from_args, TrainModel
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase, cpu_or_gpu
from allennlp.data import Vocabulary
from allennlp.data.data_loaders import TensorDict
from allennlp.models import load_archive, Model
from allennlp.models.archival import CONFIG_NAME
from allennlp.training import TrainerCallback, GradientDescentTrainer
from allennlp.training.learning_rate_schedulers import ExponentialLearningRateScheduler, LearningRateScheduler

SEQUENCE_TAGGING_DATA_PATH: str
SEQUENCE_TAGGING_SHARDS_PATH: str

class TrainingDataLoggerOnBatchCallback(TrainerCallback):
    def on_batch(
        self,
        trainer: Any,
        batch_inputs: Any,
        batch_outputs: Any,
        batch_metrics: Any,
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_primary: bool = ...,
        **kwargs: Any
    ) -> None: ...

class TrainingDeviceLoggerOnBatchCallback(TrainerCallback):
    def on_batch(
        self,
        trainer: Any,
        batch_inputs: Any,
        batch_outputs: Any,
        batch_metrics: Any,
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_primary: bool = ...,
        **kwargs: Any
    ) -> None: ...

class TrainingPrimaryCheckCallback(TrainerCallback):
    def on_start(self, trainer: Any, is_primary: bool = ..., **kwargs: Any) -> None: ...

class TestTrain(AllenNlpTestCase):
    DEFAULT_PARAMS: Params
    def test_train_model(self) -> None: ...
    @cpu_or_gpu
    def test_detect_gpu(self) -> None: ...
    @cpu_or_gpu
    def test_force_gpu(self) -> None: ...
    @cpu_or_gpu
    def test_force_cpu(self) -> None: ...
    @cpu_or_gpu
    def test_train_model_distributed(self) -> None: ...
    @pytest.mark.parametrize('max_instances', [1, 2, 3, 4, None])
    @pytest.mark.parametrize('grad_acc', [None, 2])
    @pytest.mark.parametrize('batch_size', [1, 2, 3])
    def test_train_model_distributed_with_gradient_accumulation(
        self, max_instances: Optional[int], grad_acc: Optional[int], batch_size: int
    ) -> None: ...
    @cpu_or_gpu
    @pytest.mark.parametrize('max_instances_in_memory', [None, 10])
    def test_train_model_distributed_with_sharded_reader(
        self, max_instances_in_memory: Optional[int]
    ) -> None: ...
    @cpu_or_gpu
    @pytest.mark.parametrize('max_instances_in_memory', [None, 10])
    def test_train_model_distributed_without_sharded_reader(
        self, max_instances_in_memory: Optional[int]
    ) -> None: ...
    def test_distributed_raises_error_with_no_gpus(self) -> None: ...
    def test_train_saves_all_keys_in_config(self) -> None: ...
    def test_error_is_throw_when_cuda_device_is_not_available(self) -> None: ...
    def test_train_with_test_set(self) -> None: ...
    def test_train_number_of_steps(self) -> None: ...
    def test_train_args(self) -> None: ...
    def test_train_model_can_instantiate_from_params(self) -> None: ...
    def test_train_can_fine_tune_model_from_archive(self) -> None: ...
    def test_train_nograd_regex(self) -> None: ...

class TestDryRun(AllenNlpTestCase):
    params: Params
    def setup_method(self) -> None: ...
    def test_dry_run_doesnt_overwrite_vocab(self) -> None: ...
    def test_dry_run_makes_vocab(self) -> None: ...
    def test_dry_run_with_extension(self) -> None: ...
    def test_dry_run_without_extension(self) -> None: ...
    def test_make_vocab_args(self) -> None: ...
    def test_warn_validation_loader_batches_per_epoch(self) -> None: ...
```