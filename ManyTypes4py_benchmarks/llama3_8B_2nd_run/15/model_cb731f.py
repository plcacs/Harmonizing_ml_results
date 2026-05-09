import copy
import logging
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import cast, Dict, List, Optional, Tuple, Union
import numpy as np
import torch as pt
from sockeye import __version__
from . import constants as C
from . import data_io
from . import decoder
from . import encoder
from . import layers
from . import transformer
from . import utils
from . import vocab
from .config import Config
from .encoder import FactorConfig
from .layers import LengthRatioConfig
from . import nvs
from sockeye.knn import KNNConfig

@dataclass
class ModelConfig(Config):
    ...

class SockeyeModel(pt.nn.Module):
    ...

    def __init__(self, config: ModelConfig, inference_only: bool = False, clamp_to_dtype: bool = False, train_decoder_only: bool = False, forward_pass_cache_size: int = 0):
        ...

    def set_inference_only(self, inference_only: bool):
        ...

    def cast(self, dtype: Union[pt.dtype, str]):
        ...

    def save_config(self, folder: str):
        ...

    def load_config(self, fname: str) -> ModelConfig:
        ...

    def save_parameters(self, fname: str):
        ...

    def load_parameters(self, filename: str, device: pt.device, allow_missing: bool = False, ignore_extra: bool = False):
        ...

    def load_knn_index(self, knn_index_folder: str):
        ...

    @staticmethod
    def save_version(folder: str):
        ...

    @staticmethod
    def initialize_parameters(module: pt.nn.Module):
        ...

def load_model(model_folder: str, device: pt.device, dtype: Optional[pt.dtype] = None, clamp_to_dtype: bool = False, checkpoint: Optional[str] = None, inference_only: bool = False, train_decoder_only: bool = False, allow_missing: bool = False, forward_pass_cache_size: int = 0, knn_index: Optional[str] = None) -> Tuple[SockeyeModel, List[vocab.Vocab], List[vocab.Vocab]]:
    ...

def load_models(device: pt.device, model_folders: List[str], checkpoints: Optional[List[str]] = None, dtype: Optional[pt.dtype] = None, clamp_to_dtype: bool = False, inference_only: bool = False, train_decoder_only: bool = False, allow_missing: bool = False, forward_pass_cache_size: int = 0, knn_index: Optional[str] = None) -> Tuple[List[SockeyeModel], List[vocab.Vocab], List[vocab.Vocab]]:
    ...
