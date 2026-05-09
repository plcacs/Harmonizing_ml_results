"""
Stub file for 'util_a0ec0a' module
"""

from typing import Any, Dict, Iterable, Optional, Union, Tuple, Set, List, Optional, Any, Counter
from torch import Tensor
from torch.optim import Optimizer
from torch.nn import Module
from allennlp.common.params import Params
from allennlp.data import DataLoader, Instance, Vocabulary, Batch
from allennlp.models.model import Model
from datetime import datetime
import torch
from collections import Counter

class HasBeenWarned:
    tqdm_ignores_underscores: bool

def move_optimizer_to_cuda(optimizer: Optimizer) -> None:
    ...

def get_batch_size(batch: Union[torch.Tensor, Dict]) -> int:
    ...

def time_to_str(timestamp: float) -> str:
    ...

def str_to_time(time_str: str) -> datetime:
    ...

def data_loaders_from_params(params: Params, train: bool = True, validation: bool = True, test: bool = True, serialization_dir: Optional[str] = None) -> Dict[str, DataLoader]:
    ...

def create_serialization_dir(params: Params, serialization_dir: str, recover: bool, force: bool) -> None:
    ...

def enable_gradient_clipping(model: Module, grad_clipping: Optional[float]) -> None:
    ...

def rescale_gradients(model: Module, grad_norm: Optional[float]) -> Optional[float]:
    ...

def get_metrics(model: Model, total_loss: float, total_reg_loss: Optional[float], batch_loss: Optional[float], batch_reg_loss: Optional[float], num_batches: int, reset: bool = False) -> Dict[str, Any]:
    ...

def get_train_and_validation_metrics(metrics: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ...

def evaluate(model: Model, data_loader: DataLoader, cuda_device: Union[int, torch.device] = -1, batch_weight_key: Optional[str] = None, output_file: Optional[str] = None, predictions_output_file: Optional[str] = None) -> Dict[str, Any]:
    ...

def description_from_metrics(metrics: Dict[str, Any]) -> str:
    ...

def make_vocab_from_params(params: Params, serialization_dir: str, print_statistics: bool = False) -> Vocabulary:
    ...

def ngrams(tensor: Tensor, ngram_size: int, exclude_indices: Iterable[int]) -> Counter[Tuple[int, ...]]:
    ...

def get_valid_tokens_mask(tensor: Tensor, exclude_indices: Iterable[int]) -> Tensor:
    ...