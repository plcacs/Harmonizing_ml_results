from typing import Optional, Dict, List, Tuple, Union, Any, TypeVar, Type
import torch
from allennlp.common import FromParams
from allennlp.modules.transformer.layer_norm import LayerNorm
from allennlp.modules.transformer.transformer_module import TransformerModule
if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig

T = TypeVar('T', bound='Embeddings')

class Embeddings(TransformerModule, FromParams):
    def __init__(
        self,
        embeddings: torch.nn.ModuleDict,
        embedding_size: int,
        dropout: float,
        layer_norm_eps: float = 1e-12
    ) -> None:
        super().__init__()
        for name, embedding_layer in embeddings.named_children():
            if isinstance(embedding_layer, torch.nn.Embedding):
                assert embedding_layer.embedding_dim == embedding_size
            elif isinstance(embedding_layer, torch.nn.Linear):
                assert embedding_layer.out_features == embedding_size
            else:
                raise TypeError('Layer "{}" must be of type `torch.nn.Embedding` or `torch.nn.Linear`.'.format(name))
        self.embeddings = embeddings
        self.layer_norm = LayerNorm(embedding_size, eps=layer_norm_eps)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        assert len(inputs) == len(self.embeddings)
        outputs: List[torch.Tensor] = []
        for i, layer in enumerate(self.embeddings.children()):
            outputs.append(layer(inputs[i]))
        outputs_sum: torch.Tensor = sum(outputs)
        outputs_norm: torch.Tensor = self.layer_norm(outputs_sum)
        outputs_dropout: torch.Tensor = self.dropout(outputs_norm)
        return outputs_dropout

class ImageFeatureEmbeddings(Embeddings):
    def __init__(
        self,
        feature_size: int,
        embedding_size: int,
        dropout: float = 0.0
    ) -> None:
        image_embeddings: torch.nn.Linear = torch.nn.Linear(feature_size, embedding_size)
        location_embeddings: torch.nn.Linear = torch.nn.Linear(4, embedding_size, bias=False)
        embeddings: torch.nn.ModuleDict = torch.nn.ModuleDict({
            'image_embeddings': image_embeddings,
            'location_embeddings': location_embeddings
        })
        super().__init__(embeddings, embedding_size, dropout)

class TransformerEmbeddings(Embeddings):
    _pretrained_relevant_module: List[str] = ['embeddings', 'bert.embeddings', 'roberta.embeddings']
    _pretrained_mapping: Dict[str, str] = {
        'LayerNorm': 'layer_norm',
        'word_embeddings': 'embeddings.word_embeddings',
        'position_embeddings': 'embeddings.position_embeddings',
        'token_type_embeddings': 'embeddings.token_type_embeddings',
        'albert.embeddings.LayerNorm': 'layer_norm',
        'albert.embeddings.word_embeddings': 'embeddings.word_embeddings',
        'albert.embeddings.position_embeddings': 'embeddings.position_embeddings',
        'albert.embeddings.token_type_embeddings': 'embeddings.token_type_embeddings',
        'albert.encoder.embedding_hidden_mapping_in': 'linear_transform'
    }
    _pretrained_ignore: List[str] = [
        '^albert\\.pooler\\..*',
        '^albert\\.encoder\\.albert_layer_groups\\..*',
        '^predictions\\.*'
    ]

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        pad_token_id: int = 0,
        max_position_embeddings: int = 512,
        position_pad_token_id: Optional[int] = None,
        type_vocab_size: int = 2,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        output_size: Optional[int] = None
    ) -> None:
        embedding_dict: Dict[str, torch.nn.Module] = {}
        word_embeddings: torch.nn.Embedding = torch.nn.Embedding(
            vocab_size, embedding_size, padding_idx=pad_token_id
        )
        embedding_dict['word_embeddings'] = word_embeddings
        if max_position_embeddings > 0:
            position_embeddings: torch.nn.Embedding = torch.nn.Embedding(
                max_position_embeddings, embedding_size, padding_idx=position_pad_token_id
            )
            embedding_dict['position_embeddings'] = position_embeddings
        if type_vocab_size > 0:
            token_type_embeddings: torch.nn.Embedding = torch.nn.Embedding(
                type_vocab_size, embedding_size
            )
            embedding_dict['token_type_embeddings'] = token_type_embeddings
        embeddings: torch.nn.ModuleDict = torch.nn.ModuleDict(embedding_dict)
        super().__init__(embeddings, embedding_size, dropout, layer_norm_eps=layer_norm_eps)
        if output_size:
            self.linear_transform: torch.nn.Linear = torch.nn.Linear(embedding_size, output_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        input_shape: torch.Size = input_ids.size()
        device: torch.device = input_ids.device
        seq_length: int = input_shape[1]
        embedding_inputs: List[torch.Tensor] = [input_ids]
        if attention_mask is None:
            attention_mask = input_ids != self.embeddings['word_embeddings'].padding_idx
        if 'position_embeddings' in self.embeddings:
            if position_ids is None:
                padding_idx = self.embeddings['position_embeddings'].padding_idx
                if padding_idx is None:
                    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
                    position_ids = position_ids.unsqueeze(0).expand(input_shape)
                else:
                    position_ids = torch.arange(seq_length, dtype=torch.long, device=device) + 1
                    position_ids = position_ids.unsqueeze(0).expand(input_shape) * attention_mask
                    position_ids += padding_idx
            embedding_inputs.append(position_ids)
        if 'token_type_embeddings' in self.embeddings:
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            embedding_inputs.append(token_type_ids)
        embeddings: torch.Tensor = super().forward(*embedding_inputs)
        if hasattr(self, 'linear_transform'):
            embeddings = self.linear_transform(embeddings)
        return embeddings

    @classmethod
    def _from_config(
        cls: Type[T],
        config: 'PretrainedConfig',
        **kwargs: Any
    ) -> T:
        final_kwargs: Dict[str, Any] = {
            'vocab_size': config.vocab_size,
            'pad_token_id': config.pad_token_id,
            'max_position_embeddings': config.max_position_embeddings,
            'type_vocab_size': config.type_vocab_size,
            'layer_norm_eps': config.layer_norm_eps
        }
        if hasattr(config, 'embedding_size'):
            final_kwargs['embedding_size'] = config.embedding_size
            final_kwargs['output_size'] = config.hidden_size
        else:
            final_kwargs['embedding_size'] = config.hidden_size
        if config.model_type == 'roberta':
            final_kwargs['position_pad_token_id'] = config.pad_token_id
        final_kwargs.update(**kwargs)
        return cls(**final_kwargs)
