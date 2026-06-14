import logging
from typing import Any, Dict, List, Optional, Union

import torch
from allennlp.common import FromParams
from allennlp.modules.elmo_lstm import ElmoLstm
from allennlp.modules.highway import Highway
from allennlp.modules.scalar_mix import ScalarMix
from torch.nn.modules import Dropout

logger: logging.Logger

class Elmo(torch.nn.Module, FromParams):
    _elmo_lstm: Union[_ElmoBiLm, torch.nn.Module]
    _has_cached_vocab: bool
    _keep_sentence_boundaries: bool
    _dropout: Dropout
    _scalar_mixes: List[ScalarMix]

    def __init__(
        self,
        options_file: Optional[str],
        weight_file: Optional[str],
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
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]: ...

def batch_to_ids(batch: List[List[str]]) -> torch.Tensor: ...

class _ElmoCharacterEncoder(torch.nn.Module):
    _options: Dict[str, Any]
    _weight_file: str
    output_dim: int
    requires_grad: bool
    _char_embedding_weights: torch.nn.Parameter
    _convolutions: List[torch.nn.Conv1d]
    _highways: Highway
    _projection: torch.nn.Linear
    _beginning_of_sentence_characters: torch.Tensor
    _end_of_sentence_characters: torch.Tensor

    def __init__(
        self,
        options_file: str,
        weight_file: str,
        requires_grad: bool = ...,
    ) -> None: ...
    def get_output_dim(self) -> int: ...
    def forward(
        self, inputs: torch.Tensor
    ) -> Dict[str, torch.Tensor]: ...
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
    ) -> Dict[str, Union[List[torch.Tensor], torch.Tensor]]: ...
    def create_cached_cnn_embeddings(self, tokens: List[str]) -> None: ...