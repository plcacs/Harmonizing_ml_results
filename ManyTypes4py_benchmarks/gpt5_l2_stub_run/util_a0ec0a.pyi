from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, Optional, Tuple, Set, Union, Counter as _TypingCounter

import torch
from torch.optim import Optimizer

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import dump_metrics, sanitize, int_to_device
from allennlp.data import DataLoader, Vocabulary, Batch, Instance  # noqa: F401
from allennlp.data.dataset_readers import DatasetReader
from allennlp.models.archival import CONFIG_NAME  # noqa: F401
from allennlp.models.model import Model
from allennlp.nn import util as nn_util  # noqa: F401

logger: logging.Logger

class HasBeenWarned:
    tqdm_ignores_underscores: bool

def move_optimizer_to_cuda(optimizer: Optimizer) -> None: ...
def get_batch_size(batch: Union[torch.Tensor, Dict[Any, Any]]) -> int: ...
def time_to_str(timestamp: Union[int, float]) -> str: ...
def str_to_time(time_str: str) -> datetime.datetime: ...
def data_loaders_from_params(
    params: Params,
    train: bool = ...,
    validation: bool = ...,
    test: bool = ...,
    serialization_dir: Optional[str] = ...,
) -> Dict[str, DataLoader]: ...
def create_serialization_dir(params: Params, serialization_dir: str, recover: bool, force: bool) -> None: ...
def enable_gradient_clipping(model: Model, grad_clipping: Optional[float]) -> None: ...
def rescale_gradients(model: Model, grad_norm: Optional[float] = ...) -> Optional[torch.Tensor]: ...
def get_metrics(
    model: Model,
    total_loss: float,
    total_reg_loss: Optional[float],
    batch_loss: Optional[float],
    batch_reg_loss: Optional[float],
    num_batches: int,
    reset: bool = ...,
) -> Dict[str, float]: ...
def get_train_and_validation_metrics(metrics: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]: ...
def evaluate(
    model: Model,
    data_loader: DataLoader,
    cuda_device: Union[int, torch.device] = ...,
    batch_weight_key: Optional[str] = ...,
    output_file: Optional[str] = ...,
    predictions_output_file: Optional[str] = ...,
) -> Dict[str, Any]: ...
def description_from_metrics(metrics: Dict[str, float]) -> str: ...
def make_vocab_from_params(params: Params, serialization_dir: str, print_statistics: bool = ...) -> Vocabulary: ...
def ngrams(tensor: torch.Tensor, ngram_size: int, exclude_indices: Set[int]) -> _TypingCounter[Tuple[int, ...]]: ...
def get_valid_tokens_mask(tensor: torch.Tensor, exclude_indices: Set[int]) -> torch.Tensor: ...