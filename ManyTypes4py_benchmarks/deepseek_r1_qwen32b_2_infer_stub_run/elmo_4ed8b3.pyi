import json
import logging
import warnings
from typing import Any, Dict, List, Optional, Union
import numpy
import torch
from torch import Tensor
from torch.nn.modules import Dropout
from allennlp.common import FromParams
from allennlp.data.instance import Instance
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.batch import Batch
from allennlp.data.fields import TextField
from allennlp.modules.highway import Highway
from allennlp.modules.scalar_mix import ScalarMix

logger: logging.Logger = ...

class Elmo(torch.nn.Module, FromParams):
    def __init__(self, options_file: str, weight_file: str, num_output_representations: int, requires_grad: bool = False, do_layer_norm: bool = False, dropout: float = 0.5, vocab_to_cache: Optional[List[str]] = None, keep_sentence_boundaries: bool = False, scalar_mix_parameters: Optional[List[float]] = None, module: Optional[torch.nn.Module] = None) -> None: ...
    
    def get_output_dim(self) -> int: ...
    
    def forward(self, inputs: Tensor, word_inputs: Optional[Tensor] = None) -> Dict[str, Union[List[Tensor], Tensor]]: ...

def batch_to_ids(batch: List[List[str]]) -> Tensor: ...

class _ElmoCharacterEncoder(torch.nn.Module):
    def __init__(self, options_file: str, weight_file: str, requires_grad: bool = False) -> None: ...
    
    def get_output_dim(self) -> int: ...
    
    def forward(self, inputs: Tensor) -> Dict[str, Union[Tensor, Tensor]]: ...

class _ElmoBiLm(torch.nn.Module):
    def __init__(self, options_file: str, weight_file: str, requires_grad: bool = False, vocab_to_cache: Optional[List[str]] = None) -> None: ...
    
    def get_output_dim(self) -> int: ...
    
    def forward(self, inputs: Tensor, word_inputs: Optional[Tensor] = None) -> Dict[str, Union[List[Tensor], Tensor]]: ...
    
    def create_cached_cnn_embeddings(self, tokens: List[str]) -> None: ...