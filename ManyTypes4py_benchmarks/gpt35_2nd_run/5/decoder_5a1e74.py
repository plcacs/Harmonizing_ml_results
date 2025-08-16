from typing import Dict, List, Optional, Tuple, Type, Union
import torch as pt

class Decoder(pt.nn.Module):
    __registry: Dict[Type, Type] = {}

    @classmethod
    def register(cls, config_type: Type) -> Type:
        def wrapper(target_cls: Type) -> Type:
            cls.__registry[config_type] = target_cls
            return target_cls
        return wrapper

    @classmethod
    def get_decoder(cls, config: DecoderConfig, inference_only: bool, dtype: Optional[pt.dtype] = None, clamp_to_dtype: bool = False) -> 'Decoder':
        ...

    @abstractmethod
    def __init__(self) -> None:
        ...

    @abstractmethod
    def set_inference_only(self, inference_only: bool) -> None:
        ...

    @abstractmethod
    def state_structure(self) -> str:
        ...

    @abstractmethod
    def init_state_from_encoder(self, encoder_outputs: pt.Tensor, encoder_valid_length: Optional[pt.Tensor] = None, target_embed: Optional[pt.Tensor] = None) -> List[pt.Tensor]:
        ...

    @abstractmethod
    def decode_seq(self, inputs: pt.Tensor, states: List[pt.Tensor]) -> pt.Tensor:
        ...

    @abstractmethod
    def get_num_hidden(self) -> int:
        ...

@Decoder.register(TransformerConfig)
class TransformerDecoder(Decoder):
    def __init__(self, config: TransformerConfig, inference_only: bool = False, dtype: Optional[pt.dtype] = None, clamp_to_dtype: bool = False) -> None:
        ...

    def set_inference_only(self, inference_only: bool) -> None:
        ...

    def state_structure(self) -> str:
        ...

    def init_state_from_encoder(self, encoder_outputs: pt.Tensor, encoder_valid_length: Optional[pt.Tensor] = None, target_embed: Optional[pt.Tensor] = None) -> List[pt.Tensor]:
        ...

    def decode_seq(self, inputs: pt.Tensor, states: List[pt.Tensor]) -> pt.Tensor:
        ...

    def forward(self, step_input: pt.Tensor, states: List[pt.Tensor]) -> Tuple[pt.Tensor, List[pt.Tensor]]:
        ...

    def get_num_hidden(self) -> int:
        ...
