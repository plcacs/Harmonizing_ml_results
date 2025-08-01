from abc import abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union, Any
import torch as pt
import sockeye.constants as C
from . import config
from . import layers
from . import transformer

def get_transformer_encoder(config: Union[transformer.TransformerConfig], inference_only: bool = False, dtype: Optional[pt.dtype] = None, clamp_to_dtype: bool = False) -> 'TransformerEncoder':
    return TransformerEncoder(config=config, inference_only=inference_only, dtype=dtype, clamp_to_dtype=clamp_to_dtype)

get_encoder: Any = get_transformer_encoder
EncoderConfig = Union[transformer.TransformerConfig]

class Encoder(pt.nn.Module):
    """
    Generic encoder interface.
    """

    @abstractmethod
    def get_num_hidden(self) -> int:
        """
        :return: The representation size of this encoder.
        """
        raise NotImplementedError()

    def get_encoded_seq_len(self, seq_len: int) -> int:
        """
        :return: The size of the encoded sequence.
        """
        return seq_len

    def get_max_seq_len(self) -> Optional[int]:
        """
        :return: The maximum length supported by the encoder if such a restriction exists.
        """
        return None

@dataclass
class FactorConfig(config.Config):
    pass

@dataclass
class EmbeddingConfig(config.Config):
    num_factors: int = field(init=False)
    factor_configs: Optional[List[FactorConfig]] = None
    allow_sparse_grad: bool = False

    def __post_init__(self) -> None:
        self.num_factors = 1
        if self.factor_configs is not None:
            self.num_factors += len(self.factor_configs)

class Embedding(Encoder):
    """
    Thin wrapper around PyTorch's Embedding op.

    :param config: Embedding config.
    :param embedding: pre-existing embedding Module.
    :param dtype: Torch data type for parameters.
    """

    def __init__(self, config: EmbeddingConfig, embedding: Optional[pt.nn.Embedding] = None, dtype: Optional[pt.dtype] = None) -> None:
        super().__init__()
        self.config = config
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = pt.nn.Embedding(self.config.vocab_size, self.config.num_embed, sparse=self.config.allow_sparse_grad, dtype=dtype)
        self.num_factors: int = self.config.num_factors
        self.factor_embeds: pt.nn.ModuleList = pt.nn.ModuleList()
        self.factor_combinations: List[str] = []
        if self.config.factor_configs is not None:
            for i, fc in enumerate(self.config.factor_configs, 1):
                if fc.share_embedding:
                    factor_embed = self.embedding
                else:
                    factor_embed = pt.nn.Embedding(fc.vocab_size, fc.num_embed, sparse=self.config.allow_sparse_grad, dtype=dtype)
                self.factor_embeds.append(factor_embed)
                self.factor_combinations.append(fc.combine)
        self.dropout: pt.nn.Dropout = pt.nn.Dropout(p=self.config.dropout)

    def forward(self, data: pt.Tensor) -> pt.Tensor:
        primary_data = data[:, :, 0]
        embedded = self.embedding(primary_data)
        if self.num_factors > 1:
            average_factors_embeds: List[pt.Tensor] = []
            concat_factors_embeds: List[pt.Tensor] = []
            sum_factors_embeds: List[pt.Tensor] = []
            for i, (factor_embedding, factor_combination) in enumerate(zip(self.factor_embeds, self.factor_combinations), 1):
                factor_data = data[:, :, i]
                factor_embedded = factor_embedding(factor_data)
                if factor_combination == C.FACTORS_COMBINE_CONCAT:
                    concat_factors_embeds.append(factor_embedded)
                elif factor_combination == C.FACTORS_COMBINE_SUM:
                    sum_factors_embeds.append(factor_embedded)
                elif factor_combination == C.FACTORS_COMBINE_AVERAGE:
                    average_factors_embeds.append(factor_embedded)
                else:
                    raise ValueError(f'Unknown combine value for factors: {factor_combination}')
            if average_factors_embeds:
                embedded = pt.mean(pt.stack([embedded] + average_factors_embeds, dim=0), dim=0)
            if sum_factors_embeds:
                for sum_factor_embed in sum_factors_embeds:
                    embedded = embedded + sum_factor_embed
            if concat_factors_embeds:
                embedded = pt.cat([embedded] + concat_factors_embeds, dim=2)
        if self.dropout is not None:
            embedded = self.dropout(embedded)
        return embedded

    def get_num_hidden(self) -> int:
        """
        Return the representation size of this encoder.
        """
        return self.config.num_embed

class TransformerEncoder(Encoder):
    """
    Non-recurrent encoder based on the transformer architecture in:

    Attention Is All You Need, Figure 1 (left)
    Vaswani et al. (https://arxiv.org/pdf/1706.03762.pdf).

    :param config: Configuration for transformer encoder.
    """

    def __init__(self, config: transformer.TransformerConfig, inference_only: bool = False, dtype: Optional[pt.dtype] = None, clamp_to_dtype: bool = False) -> None:
        pt.nn.Module.__init__(self)
        self.config = config
        self.dropout: pt.nn.Dropout = pt.nn.Dropout(p=config.dropout_prepost)
        self.pos_embedding: layers.PositionalEmbeddings = layers.PositionalEmbeddings(weight_type=self.config.positional_embedding_type, num_embed=self.config.model_size, max_seq_len=self.config.max_seq_len_source, scale_up_input=True, scale_down_positions=False, dtype=dtype)
        self.layers: pt.nn.ModuleList = pt.nn.ModuleList((transformer.TransformerEncoderBlock(config, inference_only=inference_only, dtype=dtype, clamp_to_dtype=clamp_to_dtype) for _ in range(config.num_layers)))
        self.final_process: transformer.TransformerProcessBlock = transformer.TransformerProcessBlock(sequence=config.preprocess_sequence, dropout=config.dropout_prepost, num_hidden=self.config.model_size, dtype=dtype, clamp_to_dtype=clamp_to_dtype)

    def forward(self, data: pt.Tensor, valid_length: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        data = self.pos_embedding(data)
        if self.dropout is not None:
            data = self.dropout(data)
        _, max_len, __ = data.size()
        single_head_att_mask: pt.Tensor = layers.prepare_source_length_mask(valid_length, self.config.attention_heads, max_length=max_len, expand=False)
        att_mask: pt.Tensor = single_head_att_mask.unsqueeze(1).expand(-1, self.config.attention_heads, -1).reshape((-1, max_len)).unsqueeze(1)
        att_mask = att_mask.expand(-1, max_len, -1)
        data = data.transpose(1, 0)
        for layer in self.layers:
            data = layer(data, att_mask=att_mask)
        data = self.final_process(data)
        data = data.transpose(1, 0)
        return (data, valid_length, single_head_att_mask)

    def get_num_hidden(self) -> int:
        """
        Return the representation size of this encoder.
        """
        return self.config.model_size
