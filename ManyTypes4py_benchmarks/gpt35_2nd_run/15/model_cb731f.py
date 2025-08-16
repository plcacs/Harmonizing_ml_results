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
try:
    import faiss
    import faiss.contrib.torch_utils
except:
    pass
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig(Config):
    config_data: str
    vocab_source_size: int
    vocab_target_size: int
    config_embed_source: str
    config_embed_target: str
    config_encoder: str
    config_decoder: str
    config_length_task: Optional[str] = None
    weight_tying_type: str = C.WEIGHT_TYING_SRC_TRG_SOFTMAX
    lhuc: bool = False
    dtype: str = C.DTYPE_FP32
    neural_vocab_selection: Optional[bool] = None
    neural_vocab_selection_block_loss: bool = False

class SockeyeModel(pt.nn.Module):
    config: ModelConfig
    inference_only: bool
    clamp_to_dtype: bool

    def __init__(self, config: ModelConfig, inference_only: bool = False, clamp_to_dtype: bool = False, train_decoder_only: bool = False, forward_pass_cache_size: int = 0) -> None:
        ...

    def set_inference_only(self, inference_only: bool) -> None:
        ...

    def cast(self, dtype: Union[str, pt.dtype]) -> None:
        ...

    def state_structure(self) -> str:
        ...

    def encode(self, inputs: pt.Tensor, valid_length: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        ...

    def encode_and_initialize(self, inputs: pt.Tensor, valid_length: pt.Tensor, constant_length_ratio: float = 0.0) -> Tuple[pt.Tensor, pt.Tensor, Optional[pt.Tensor]]:
        ...

    def _embed_and_encode(self, source: pt.Tensor, source_length: pt.Tensor, target: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor, pt.Tensor, Optional[pt.Tensor]]:
        ...

    def decode_step(self, step_input: pt.Tensor, states: List[pt.Tensor], vocab_slice_ids: Optional[List[int]] = None) -> Tuple[pt.Tensor, Optional[pt.Tensor], List[pt.Tensor], List[pt.Tensor]]:
        ...

    def forward(self, source: pt.Tensor, source_length: pt.Tensor, target: pt.Tensor, target_length: pt.Tensor) -> Dict[str, pt.Tensor]:
        ...

    def get_decoder_states(self, source: pt.Tensor, source_length: pt.Tensor, target: pt.Tensor, target_length: pt.Tensor) -> pt.Tensor:
        ...

    def predict_output_length(self, source_encoded: pt.Tensor, source_encoded_length: pt.Tensor, constant_length_ratio: float = 0.0) -> pt.Tensor:
        ...

    def save_config(self, folder: str) -> None:
        ...

    @staticmethod
    def load_config(fname: str) -> ModelConfig:
        ...

    def save_parameters(self, fname: str) -> None:
        ...

    def load_parameters(self, filename: str, device: pt.device, allow_missing: bool = False, ignore_extra: bool = False) -> None:
        ...

    def set_parameters(self, new_params: Dict[str, pt.Tensor], allow_missing: bool = True, ignore_extra: bool = False) -> None:
        ...

    def load_knn_index(self, knn_index_folder: str) -> None:
        ...

    @staticmethod
    def save_version(folder: str) -> None:
        ...

    def _get_embeddings(self) -> Tuple[pt.nn.Embedding, pt.nn.Embedding, Optional[pt.Tensor]]:
        ...

    @property
    def num_source_factors(self) -> int:
        ...

    @property
    def num_target_factors(self) -> int:
        ...

    @property
    def target_factor_configs(self) -> List[FactorConfig]:
        ...

    @property
    def training_max_observed_len_source(self) -> int:
        ...

    @property
    def training_max_observed_len_target(self) -> int:
        ...

    @property
    def max_supported_len_source(self) -> int:
        ...

    @property
    def max_supported_len_target(self) -> int:
        ...

    @property
    def length_ratio_mean(self) -> float:
        ...

    @property
    def length_ratio_std(self) -> float:
        ...

    @property
    def output_layer_vocab_size(self) -> int:
        ...

    @property
    def eop_id(self) -> int:
        ...

    def _cache_wrapper(self, class_func: callable) -> callable:
        ...

class _DecodeStep(pt.nn.Module):
    def __init__(self, embedding_target: pt.nn.Embedding, decoder: pt.nn.Module, output_layer: pt.nn.Module, factor_output_layers: pt.nn.ModuleList, knn: Optional[layers.KNN] = None) -> None:
        ...

    def forward(self, step_input: pt.Tensor, states: List[pt.Tensor], vocab_slice_ids: Optional[List[int]] = None) -> Tuple[pt.Tensor, pt.Tensor, List[pt.Tensor]]:
        ...

def initialize_parameters(module: SockeyeModel) -> None:
    ...

def load_model(model_folder: str, device: pt.device, dtype: Optional[Union[str, pt.dtype]] = None, clamp_to_dtype: bool = False, checkpoint: Optional[int] = None, inference_only: bool = False, train_decoder_only: bool = False, allow_missing: bool = False, forward_pass_cache_size: int = 0, knn_index: Optional[str] = None) -> Tuple[SockeyeModel, List[vocab.Vocab], List[vocab.Vocab]]:
    ...

def load_models(device: pt.device, model_folders: List[str], checkpoints: Optional[List[Optional[int]] = None, dtype: Optional[Union[str, pt.dtype]] = None, clamp_to_dtype: bool = False, inference_only: bool = False, train_decoder_only: bool = False, allow_missing: bool = False, forward_pass_cache_size: int = 0, knn_index: Optional[str] = None) -> Tuple[List[SockeyeModel], List[vocab.Vocab], List[vocab.Vocab]]:
    ...
