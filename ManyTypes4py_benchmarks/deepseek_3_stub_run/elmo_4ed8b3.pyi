import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from allennlp.common import FromParams
from allennlp.data.instance import Instance
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.fields import TextField
from allennlp.modules.highway import Highway
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.modules.token_embedders import Embedding

class Elmo(torch.nn.Module, FromParams):
    def __init__(
        self,
        options_file: str,
        weight_file: str,
        num_output_representations: int,
        requires_grad: bool = False,
        do_layer_norm: bool = False,
        dropout: float = 0.5,
        vocab_to_cache: Optional[List[str]] = None,
        keep_sentence_boundaries: bool = False,
        scalar_mix_parameters: Optional[List[float]] = None,
        module: Optional[torch.nn.Module] = None
    ) -> None: ...
    def get_output_dim(self) -> int: ...
    def forward(
        self,
        inputs: torch.Tensor,
        word_inputs: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]: ...

def batch_to_ids(batch: List[List[str]]) -> torch.Tensor: ...

class _ElmoCharacterEncoder(torch.nn.Module):
    def __init__(
        self,
        options_file: str,
        weight_file: str,
        requires_grad: bool = False
    ) -> None: ...
    def get_output_dim(self) -> int: ...
    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]: ...
    def _load_weights(self) -> None: ...
    def _load_char_embedding(self) -> None: ...
    def _load_cnn_weights(self) -> None: ...
    def _load_highway(self) -> None: ...
    def _load_projection(self) -> None: ...

class _ElmoBiLm(torch.nn.Module):
    def __init__(
        self,
        options_file: str,
        weight_file: str,
        requires_grad: bool = False,
        vocab_to_cache: Optional[List[str]] = None
    ) -> None: ...
    def get_output_dim(self) -> int: ...
    def forward(
        self,
        inputs: torch.Tensor,
        word_inputs: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[List[torch.Tensor], torch.Tensor]]: ...
    def create_cached_cnn_embeddings(self, tokens: List[str]) -> None: ...