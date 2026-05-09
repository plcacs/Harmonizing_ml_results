import copy
import glob
import json
import os
import time
from typing import Any, Dict, List, Optional, Union
import math
import pytest
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.common.testing import AllenNlpTestCase, requires_gpu, requires_multi_gpu
from allennlp.data import Vocabulary, Instance, Token
from allennlp.data.dataset_readers import DatasetReader
from allennlp.models import Model
from allennlp.training import GradientDescentTrainer, Checkpointer
from allennlp.training.callbacks import TrainerCallback
from allennlp.training.learning_rate_schedulers import ExponentialLearningRateScheduler
from allennlp.training.momentum_schedulers import MomentumScheduler

class FakeDatasetReader(DatasetReader):
    def __init__(self, total_instances: int, batch_size: int) -> None: ...
    def _read(self, file_path: str) -> Any: ...
    def text_to_instance(self, index: int, field_type: str) -> Instance: ...

class FakeModel(Model):
    def __init__(self, vocab: Vocabulary) -> None: ...
    def forward(self, **kwargs: Any) -> Dict[str, torch.Tensor]: ...

class TrainerTestBase(AllenNlpTestCase):
    def setup_method(self) -> None: ...
    # Other attributes and methods are inferred but not explicitly shown in the stub

class ZeroGradientsBackwardCallback(TrainerCallback):
    def on_backward(self, trainer: GradientDescentTrainer, batch_outputs: Dict[str, Any], backward_called: bool, **kwargs: Any) -> bool: ...

class TestTrainer(TrainerTestBase):
    def test_trainer_can_run(self) -> None: ...
    def test_train_zero_gradients(self) -> None: ...
    # Other test methods are inferred but not explicitly shown in the stub

@requires_gpu
class TestAmpTrainer(TrainerTestBase):
    @pytest.mark.parametrize('grad_norm, num_gradient_accumulation_steps', [(None, 1), (1.0, 1), (1.0, 2)])
    def test_trainer_can_run_amp(self, grad_norm: Optional[float], num_gradient_accumulation_steps: int) -> None: ...

class TestSparseClipGrad(AllenNlpTestCase):
    def test_sparse_clip_grad(self) -> None: ...