```python
import torch
import numpy
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.nn.modules import Dropout
from allennlp.common import FromParams
from allennlp.data.instance import Instance
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.batch import Batch
from allennlp.data.fields import TextField
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.modules.elmo_lstm import ElmoLstm
from allennlp.modules.highway import Highway
from allennlp.modules.scalar_mix import ScalarMix

class Elmo(torch.nn.Module, FromParams):
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
        module: Optional[torch.nn.Module] = ...
    ) -> None: ...
    def get_output_dim(self) -> int: ...
    def forward(
        self,
        inputs: torch.Tensor,
        word_inputs: Optional[torch.Tensor] = ...
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]: ...

def batch_to_ids(batch: List[List[str]]) -> torch.Tensor: ...

class _ElmoCharacterEncoder(torch.nn.Module):
    def __init__(
        self,
        options_file: str,
        weight_file: str,
        requires_grad: bool = ...
    ) -> None: ...
    def get_output_dim(self) -> int: ...
    def forward(
        self,
        inputs: torch.Tensor
    ) -> Dict[str, torch.Tensor]: ...
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
        requires_grad: bool = ...,
        vocab_to_cache: Optional[List[str]] = ...
    ) -> None: ...
    def get_output_dim(self) -> int: ...
    def forward(
        self,
        inputs: torch.Tensor,
        word_inputs: Optional[torch.Tensor] = ...
    ) -> Dict[str, Any]: ...
    def create_cached_cnn_embeddings(self, tokens: List[str]) -> None: ...
```