import logging
import os
from os import PathLike
import re
from typing import Dict, List, Set, Type, Optional, Union
import numpy
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params, remove_keys_from_params
from allennlp.common.registrable import Registrable
from allennlp.data import Instance, Vocabulary
from allennlp.data.batch import Batch
from allennlp.nn import util
from allennlp.nn.module import Module
from allennlp.nn.parallel import DdpAccelerator
from allennlp.nn.regularizers import RegularizerApplicator
logger: logging.Logger = logging.getLogger(__name__)
_DEFAULT_WEIGHTS: str = 'best.th'

class Model(Module, Registrable):
    _warn_for_unseparable_batches: Set[str] = set()
    default_predictor: Optional[Type] = None

    def __init__(self, vocab: Vocabulary, regularizer: Optional[RegularizerApplicator] = None, serialization_dir: Optional[str] = None, ddp_accelerator: Optional[DdpAccelerator] = None) -> None:
        super().__init__()
        self.vocab: Vocabulary = vocab
        self._regularizer: Optional[RegularizerApplicator] = regularizer
        self.serialization_dir: Optional[str] = serialization_dir
        self.ddp_accelerator: Optional[DdpAccelerator] = ddp_accelerator

    def get_regularization_penalty(self) -> Optional[torch.Tensor]:
        ...

    def get_parameters_for_histogram_logging(self) -> List[str]:
        ...

    def get_parameters_for_histogram_tensorboard_logging(self) -> List[str]:
        ...

    def forward(self, *inputs: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        ...

    def forward_on_instance(self, instance: Instance) -> Dict[str, torch.Tensor]:
        ...

    def forward_on_instances(self, instances: List[Instance]) -> List[Dict[str, Union[numpy.ndarray, torch.Tensor]]]:
        ...

    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ...

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        ...

    def _get_prediction_device(self) -> int:
        ...

    def _maybe_warn_for_unseparable_batches(self, output_key: str) -> None:
        ...

    @classmethod
    def _load(cls, config: Params, serialization_dir: str, weights_file: Optional[str] = None, cuda_device: int = -1) -> 'Model':
        ...

    @classmethod
    def load(cls, config: Params, serialization_dir: str, weights_file: Optional[str] = None, cuda_device: int = -1) -> 'Model':
        ...

    def extend_embedder_vocab(self, embedding_sources_mapping: Optional[Dict[str, str]] = None) -> None:
        ...

    @classmethod
    def from_archive(cls, archive_file: str, vocab: Optional[Vocabulary] = None) -> 'Model':
        ...

def remove_weights_related_keys_from_params(params: Params, keys: List[str] = ['pretrained_file', 'initializer']) -> None:
    ...

def remove_pretrained_embedding_params(params: Params) -> None:
    ...
