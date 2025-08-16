import logging
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset
from freqtrade.freqai.torch.PyTorchDataConvertor import PyTorchDataConvertor
from freqtrade.freqai.torch.PyTorchTrainerInterface import PyTorchTrainerInterface
from .datasets import WindowDataset

logger: logging.Logger = logging.getLogger(__name__)

class PyTorchModelTrainer(PyTorchTrainerInterface):

    def __init__(self, model: nn.Module, optimizer: Optimizer, criterion: nn.Module, device: str, data_convertor: PyTorchDataConvertor, model_meta_data: Dict[str, Any] = {}, window_size: int = 1, tb_logger: Optional[Any] = None, **kwargs: Any) -> None:
        ...

    def fit(self, data_dictionary: Dict[str, Any], splits: List[str]) -> None:
        ...

    @torch.no_grad()
    def estimate_loss(self, data_loader_dictionary: Dict[str, DataLoader], split: str) -> None:
        ...

    def create_data_loaders_dictionary(self, data_dictionary: Dict[str, Any], splits: List[str]) -> Dict[str, DataLoader]:
        ...

    def calc_n_epochs(self, n_obs: int) -> int:
        ...

    def save(self, path: str) -> None:
        ...

    def load(self, path: str) -> 'PyTorchModelTrainer':
        ...

    def load_from_checkpoint(self, checkpoint: Dict[str, Any]) -> 'PyTorchModelTrainer':
        ...

class PyTorchTransformerTrainer(PyTorchModelTrainer):

    def create_data_loaders_dictionary(self, data_dictionary: Dict[str, Any], splits: List[str]) -> Dict[str, DataLoader]:
        ...
