```pyi
import json
import logging
import warnings
from typing import Any, Dict, List, Union, Optional
import numpy
import torch
from torch.nn.modules import Dropout
from allennlp.common import FromParams
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.batch import Batch
from allennlp.data.fields import TextField
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper, ELMoTokenCharactersIndexer
from allennlp.modules.elmo_lstm import ElmoLstm
from allennlp.modules.highway import Highway
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.nn.util import add_sentence_boundary_token_ids, get_device_of, remove_sentence_boundaries

logger: logging.Logger

class Elmo(torch.nn.Module, FromParams):
    _elmo_lstm: Any
    _has_cached_vocab: bool
    _keep_sentence_boundaries: bool
    _dropout: Dropout
    _scalar_mixes: List[ScalarMix]
    
    def __init__(
        self,
        options_file: str,
        weight_file: str,
        num_output_representations: int,
        requires_grad: bool = ...,
        do_layer_norm: bool = ...,
        dropout: float = ...,
        vocab_to_cache: Optional[List[str]] = ...,
        keep_sentence_boundaries: bool = ...,
        scalar_mix_parameters: Optional[List[float]] = ...,
        module: Optional[torch.nn.Module] = ...,
    ) -> None: ...
    def get_output_dim(self) -> int: ...
    def forward(
        self,
        inputs: torch.Tensor,
        word_inputs: Optional[torch.Tensor] = ...,
    ) -> Dict[str, Union[List[torch.Tensor], torch.Tensor]]: ...

def batch_to_ids(batch: List[List[str]]) -> torch.Tensor: ...

class _ElmoCharacterEncoder(torch.nn.Module):
    _options: Dict[str, Any]
    _weight_file: str
    output_dim: int
    requires_grad: bool
    _beginning_of_sentence_characters: torch.Tensor
    _end_of_sentence_characters: torch.Tensor
    _char_embedding_weights: torch.nn.Parameter
    _convolutions: List[torch.nn.Conv1d]
    _highways: Highway
    _projection: torch.nn.Linear
    
    def __init__(
        self,
        options_file: str,
        weight_file: str,
        requires_grad: bool = ...,
    ) -> None: ...
    def get_output_dim(self) -> int: ...
    def forward(self, inputs: torch.Tensor) -> Dict[str, Any]: ...
    def _load_weights(self) -> None: ...
    def _load_char_embedding(self) -> None: ...
    def _load_cnn_weights(self) -> None: ...
    def _load_highway(self) -> None: ...
    def _load_projection(self) -> None: ...

class _ElmoBiLm(torch.nn.Module):
    _token_embedder: _ElmoCharacterEncoder
    _requires_grad: bool
    _word_embedding: Optional[Any]
    _bos_embedding: Optional[torch.Tensor]
    _eos_embedding: Optional[torch.Tensor]
    _elmo_lstm: ElmoLstm
    num_layers: int
    
    def __init__(
        self,
        options_file: str,
        weight_file: str,
        requires_grad: bool = ...,
        vocab_to_cache: Optional[List[str]] = ...,
    ) -> None: ...
    def get_output_dim(self) -> int: ...
    def forward(
        self,
        inputs: torch.Tensor,
        word_inputs: Optional[torch.Tensor] = ...,
    ) -> Dict[str, Any]: ...
    def create_cached_cnn_embeddings(self, tokens: List[str]) -> None: ...
```