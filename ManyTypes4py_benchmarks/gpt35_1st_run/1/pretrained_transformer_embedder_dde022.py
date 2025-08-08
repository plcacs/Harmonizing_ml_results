import logging
import math
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.nn.util import batched_index_select
from transformers import XLNetConfig
logger: logging.Logger = logging.getLogger(__name__)

@TokenEmbedder.register('pretrained_transformer')
class PretrainedTransformerEmbedder(TokenEmbedder):
    def __init__(self, model_name: str, *, max_length: Optional[int] = None, sub_module: Optional[str] = None, train_parameters: bool = True, eval_mode: bool = False, last_layer_only: bool = True, override_weights_file: Optional[str] = None, override_weights_strip_prefix: Optional[str] = None, reinit_modules: Optional[Union[int, Tuple[int, ...], Tuple[str, ...]]] = None, load_weights: bool = True, gradient_checkpointing: Optional[bool] = None, tokenizer_kwargs: Optional[Dict[str, Any]] = None, transformer_kwargs: Optional[Dict[str, Any]] = None):
    def train(self, mode: bool = True) -> 'PretrainedTransformerEmbedder':
    def get_output_dim(self) -> int:
    def _number_of_token_type_embeddings(self) -> int:
    def forward(self, token_ids: torch.LongTensor, mask: torch.BoolTensor, type_ids: Optional[torch.LongTensor] = None, segment_concat_mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:
    def _fold_long_sequences(self, token_ids: torch.LongTensor, mask: torch.BoolTensor, type_ids: Optional[torch.LongTensor] = None) -> Tuple[torch.LongTensor, torch.BoolTensor, Optional[torch.LongTensor]]:
    def _unfold_long_sequences(self, embeddings: torch.FloatTensor, mask: torch.BoolTensor, batch_size: int, num_segment_concat_wordpieces: int) -> torch.FloatTensor:
