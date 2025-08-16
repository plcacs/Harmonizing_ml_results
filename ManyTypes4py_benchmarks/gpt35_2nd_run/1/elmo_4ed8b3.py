import json
import logging
import warnings
from typing import Any, Dict, List, Union
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
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    import h5py
logger = logging.getLogger(__name__)

class Elmo(torch.nn.Module, FromParams):
    def __init__(self, options_file: str, weight_file: str, num_output_representations: int, requires_grad: bool = False, do_layer_norm: bool = False, dropout: float = 0.5, vocab_to_cache: List[str] = None, keep_sentence_boundaries: bool = False, scalar_mix_parameters: List[float] = None, module: torch.nn.Module = None) -> None:
    
    def get_output_dim(self) -> int:
    
    def forward(self, inputs: torch.Tensor, word_inputs: torch.Tensor = None) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
    
def batch_to_ids(batch: List[List[str]]) -> torch.Tensor:

class _ElmoCharacterEncoder(torch.nn.Module):
    def __init__(self, options_file: str, weight_file: str, requires_grad: bool = False) -> None:
    
    def get_output_dim(self) -> int:
    
    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:

class _ElmoBiLm(torch.nn.Module):
    def __init__(self, options_file: str, weight_file: str, requires_grad: bool = False, vocab_to_cache: List[str] = None) -> None:
    
    def get_output_dim(self) -> int:
    
    def forward(self, inputs: torch.Tensor, word_inputs: torch.Tensor = None) -> Dict[str, Union[List[torch.Tensor], torch.BoolTensor]]:

    def create_cached_cnn_embeddings(self, tokens: List[str]) -> None:
