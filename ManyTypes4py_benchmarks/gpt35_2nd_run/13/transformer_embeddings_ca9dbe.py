from typing import Optional, TYPE_CHECKING
import torch
from allennlp.common import FromParams
from allennlp.modules.transformer.layer_norm import LayerNorm
from allennlp.modules.transformer.transformer_module import TransformerModule
if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig

class Embeddings(TransformerModule, FromParams):
    def __init__(self, embeddings: torch.nn.ModuleDict, embedding_size: int, dropout: float, layer_norm_eps: float = 1e-12) -> None:
        ...

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        ...

class ImageFeatureEmbeddings(Embeddings):
    def __init__(self, feature_size: int, embedding_size: int, dropout: float = 0.0) -> None:
        ...

class TransformerEmbeddings(Embeddings):
    def __init__(self, vocab_size: int, embedding_size: int, pad_token_id: int = 0, max_position_embeddings: int = 512, position_pad_token_id: Optional[int] = None, type_vocab_size: int = 2, dropout: float = 0.1, layer_norm_eps: float = 1e-12, output_size: Optional[int] = None) -> None:
        ...

    def forward(self, input_ids: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        ...

    @classmethod
    def _from_config(cls, config: PretrainedConfig, **kwargs) -> 'TransformerEmbeddings':
        ...
