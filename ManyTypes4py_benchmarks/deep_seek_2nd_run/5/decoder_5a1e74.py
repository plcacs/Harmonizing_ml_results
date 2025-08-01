"""
Decoders for sequence-to-sequence models.
"""
import logging
from abc import abstractmethod
from itertools import islice
from typing import Dict, List, Optional, Tuple, Type, Union, Any, TypeVar, Generic
import torch as pt
from . import constants as C
from . import layers
from . import transformer
from .transformer import TransformerConfig

logger = logging.getLogger(__name__)
DecoderConfig = Union[TransformerConfig]

T = TypeVar('T')

def get_decoder(config: DecoderConfig, inference_only: bool = False, dtype: Optional[pt.dtype] = None, clamp_to_dtype: bool = False) -> 'Decoder':
    return Decoder.get_decoder(config=config, inference_only=inference_only, dtype=dtype, clamp_to_dtype=clamp_to_dtype)

class Decoder(pt.nn.Module):
    """
    Generic decoder interface.
    """
    __registry: Dict[Type[DecoderConfig], Type['Decoder']] = {}

    @classmethod
    def register(cls, config_type: Type[DecoderConfig]) -> Any:
        """
        Registers decoder type for configuration. Suffix is appended to decoder prefix.
        """
        def wrapper(target_cls: Type['Decoder']) -> Type['Decoder']:
            cls.__registry[config_type] = target_cls
            return target_cls
        return wrapper

    @classmethod
    def get_decoder(cls, config: DecoderConfig, inference_only: bool, dtype: Optional[pt.dtype] = None, clamp_to_dtype: bool = False) -> 'Decoder':
        """
        Creates decoder based on config type.
        """
        config_type = type(config)
        if config_type not in cls.__registry:
            raise ValueError('Unsupported decoder configuration %s' % config_type.__name__)
        decoder_cls = cls.__registry[config_type]
        return decoder_cls(config=config, inference_only=inference_only, dtype=dtype, clamp_to_dtype=clamp_to_dtype)

    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def set_inference_only(self, inference_only: bool) -> None:
        raise NotImplementedError()

    @abstractmethod
    def state_structure(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def init_state_from_encoder(self, encoder_outputs: pt.Tensor, encoder_valid_length: Optional[pt.Tensor] = None, target_embed: Optional[pt.Tensor] = None) -> List[pt.Tensor]:
        raise NotImplementedError()

    @abstractmethod
    def decode_seq(self, inputs: pt.Tensor, states: List[pt.Tensor]) -> pt.Tensor:
        """
        Decodes a sequence of embedded target words.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_num_hidden(self) -> int:
        raise NotImplementedError()

@Decoder.register(TransformerConfig)
class TransformerDecoder(Decoder):
    """
    Transformer decoder as in Vaswani et al, 2017: Attention is all you need.
    """
    def __init__(self, config: TransformerConfig, inference_only: bool = False, dtype: Optional[pt.dtype] = None, clamp_to_dtype: bool = False) -> None:
        Decoder.__init__(self)
        pt.nn.Module.__init__(self)
        self.config = config
        self.pos_embedding = layers.PositionalEmbeddings(weight_type=self.config.positional_embedding_type, num_embed=self.config.model_size, max_seq_len=self.config.max_seq_len_target, scale_up_input=True, scale_down_positions=False, dtype=dtype)
        self.autoregressive_mask = transformer.AutoRegressiveMask()
        self.layers = pt.nn.ModuleList((transformer.TransformerDecoderBlock(config, inference_only=inference_only, dtype=dtype, clamp_to_dtype=clamp_to_dtype) for _ in range(config.num_layers)))
        self.final_process = transformer.TransformerProcessBlock(sequence=config.preprocess_sequence, dropout=config.dropout_prepost, num_hidden=self.config.model_size, dtype=dtype, clamp_to_dtype=clamp_to_dtype)
        self.dropout = pt.nn.Dropout(p=self.config.dropout_prepost)
        self.set_inference_only(inference_only)

    def set_inference_only(self, inference_only: bool) -> None:
        self.inference_only = inference_only
        for layer in self.layers:
            layer.set_inference_only(inference_only)

    def state_structure(self) -> str:
        structure = ''
        if self.inference_only:
            structure += C.STEP_STATE + C.MASK_STATE + C.ENCODER_STATE * self.config.num_layers
        else:
            structure += C.STEP_STATE + C.ENCODER_STATE + C.MASK_STATE
        total_num_states = sum((layer.num_state_tensors for layer in self.layers))
        structure += C.DECODER_STATE * total_num_states
        return structure

    def init_state_from_encoder(self, encoder_outputs: pt.Tensor, encoder_valid_length: Optional[pt.Tensor] = None, target_embed: Optional[pt.Tensor] = None) -> List[pt.Tensor]:
        source_max_len = encoder_outputs.size()[1]
        source_mask = layers.prepare_source_length_mask(encoder_valid_length, self.config.attention_heads, source_max_len, mask_prepended_tokens=self.config.block_prepended_cross_attention)
        if target_embed is None:
            steps = pt.zeros_like(encoder_valid_length[:, :1])
            source_mask = source_mask.view(-1, self.config.attention_heads, 1, source_max_len)
        else:
            target_length = target_embed.size()[1]
            steps = pt.arange(0, target_length, device=target_embed.device).unsqueeze(0)
            source_mask = source_mask.expand(-1, target_length, -1)
            source_mask = source_mask.view(-1, self.config.attention_heads, target_length, source_max_len)
        if self.inference_only:
            states = [steps, source_mask]
            encoder_outputs_t = encoder_outputs.transpose(1, 0)
            for layer in self.layers:
                enc_att_kv = layer.enc_attention.ff_kv(encoder_outputs_t)
                states.append(enc_att_kv)
        else:
            states = [steps, encoder_outputs.transpose(1, 0), source_mask]
        _batch_size = encoder_outputs.size()[0]
        _device = encoder_outputs.device
        _dtype = encoder_outputs.dtype
        dummy_autoregr_states = [pt.zeros(layer.get_states_shape(_batch_size), device=_device, dtype=_dtype) for layer in self.layers for _ in range(layer.num_state_tensors)]
        states += dummy_autoregr_states
        return states

    def decode_seq(self, inputs: pt.Tensor, states: List[pt.Tensor]) -> pt.Tensor:
        outputs, _ = self.forward(inputs, states)
        return outputs

    def forward(self, step_input: pt.Tensor, states: List[pt.Tensor]) -> Tuple[pt.Tensor, List[pt.Tensor]]:
        target_mask = None
        if self.inference_only:
            steps, source_mask, *other = states
            source_encoded = None
            enc_att_kv = other[:self.config.num_layers]
            autoregr_states = other[self.config.num_layers:]
        else:
            if any((layer.needs_mask for layer in self.layers)):
                target_mask = self.autoregressive_mask(step_input)
            steps, source_encoded, source_mask, *autoregr_states = states
            enc_att_kv = [None for _ in range(self.config.num_layers)]
        if any((layer.num_state_tensors > 1 for layer in self.layers)):
            states_iter = iter(autoregr_states)
            autoregr_states = [list(islice(states_iter, 0, layer.num_state_tensors)) for layer in self.layers]
        batch, heads, target_max_len, source_max_len = source_mask.size()
        source_mask_view = source_mask.view(batch * heads, target_max_len, source_max_len)
        target = self.pos_embedding(step_input, steps)
        target = target.transpose(1, 0)
        target = self.dropout(target)
        new_autoregr_states = []
        for layer, layer_autoregr_state, layer_enc_att_kv in zip(self.layers, autoregr_states, enc_att_kv):
            target, new_layer_autoregr_state = layer(target=target, target_mask=target_mask, source=source_encoded, source_mask=source_mask_view, autoregr_states=layer_autoregr_state, enc_att_kv=layer_enc_att_kv)
            new_autoregr_states += [*new_layer_autoregr_state]
        target = self.final_process(target)
        target = target.transpose(1, 0)
        steps = steps + 1
        if self.inference_only:
            encoder_attention_keys_values = states[2:2 + self.config.num_layers]
            new_states = [steps, states[1]] + encoder_attention_keys_values + new_autoregr_states
        else:
            new_states = [steps, states[1], states[2]] + new_autoregr_states
        return (target, new_states)

    def get_num_hidden(self) -> int:
        return self.config.model_size
