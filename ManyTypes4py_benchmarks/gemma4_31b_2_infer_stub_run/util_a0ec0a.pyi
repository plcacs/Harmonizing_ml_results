import datetime
import logging
import os
import torch
from typing import Any, Dict, Iterable, Optional, Union, Tuple, Set, List, overload
from collections import Counter
from allennlp.common.params import Params
from allennlp.data import Vocabulary, DataLoader
from allennlp.data.dataset_readers import DatasetReader
from allennlp.models.model import Model

logger: logging.Logger ...

class HasBeenWarned:
    tqdm_ignores_underscores: bool ...

def move_optimizer_to_cuda(optimizer: Any) -> None ...

def get_batch_size(batch: Union[torch.Tensor, Dict[str, Any]]) -> int ...

def time_to_str(timestamp: float) -> str ...

def str_to_time(time_str: str) -> datetime.datetime ...

def data_loaders_from_params(
    params: Params,
    train: bool = True,
    validation: bool = True,
    test: bool = True,
    serialization_dir: Optional[str] = None,
) -> Dict[str, DataLoader] ...

def create_serialization_dir(
    params: Params,
    serialization_dir: str,
    recover: bool,
    force: bool,
) -> None ...

def enable_gradient_clipping(model: Model, grad_clipping: Optional[float]) -> None ...

def rescale_gradients(model: Model, grad_norm: Optional[float] = None) -> Optional[float] ...

def get_metrics(
    model: Model,
    total_loss: float,
    total_reg_loss: Optional[float],
    batch_loss: Optional[float],
    batch_reg_loss: Optional[float],
    num_batches: int,
    reset: bool = False,
) -> Dict[str, Any] ...

def get_train_and_validation_metrics(metrics: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]] ...

def evaluate(
    model: Model,
    data_loader: DataLoader,
    cuda_device: Union[int, torch.device] = -1,
    batch_weight_key: Optional[str] = None,
    output_file: Optional[str] = None,
    predictions_output_file: Optional[str] = None,
) -> Dict[str, Any] ...

def description_from_metrics(metrics: Dict[str, Any]) -> str ...

def make_vocab_from_params(
    params: Params,
    serialization_dir: str,
    print_statistics: bool = False,
) -> Vocabulary ...

def ngrams(
    tensor: torch.Tensor,
    ngram_size: int,
    exclude_indices: Iterable[int],
) -> Counter[Tuple[int, ...]] ...

def get_valid_tokens_mask(tensor: torch.Tensor, exclude_indices: Iterable[int]) -> torch.Tensor ...