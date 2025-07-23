from typing import Dict, Optional, List, Any, Union, Tuple
import torch
import torch.nn.functional
from allennlp.data.fields.field import Field
from allennlp.nn import util

def _tensorize(x: Union[List[int], torch.Tensor]) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x)
    return x

class TransformerTextField(Field[torch.Tensor]):
    """
    A `TransformerTextField` is a collection of several tensors that are are a representation of text,
    tokenized and ready to become input to a transformer.

    The naming pattern of the tensors follows the pattern that's produced by the huggingface tokenizers,
    and expected by the huggingface transformers.
    """
    __slots__ = ['input_ids', 'token_type_ids', 'attention_mask', 'special_tokens_mask', 'offsets_mapping', 'padding_token_id']

    def __init__(
        self,
        input_ids: Union[List[int], torch.Tensor],
        token_type_ids: Optional[Union[List[int], torch.Tensor]] = None,
        attention_mask: Optional[Union[List[int], torch.Tensor]] = None,
        special_tokens_mask: Optional[Union[List[int], torch.Tensor]] = None,
        offsets_mapping: Optional[Union[List[Tuple[int, int]], torch.Tensor]] = None,
        padding_token_id: int = 0
    ) -> None:
        self.input_ids = _tensorize(input_ids)
        self.token_type_ids = None if token_type_ids is None else _tensorize(token_type_ids)
        self.attention_mask = None if attention_mask is None else _tensorize(attention_mask)
        self.special_tokens_mask = None if special_tokens_mask is None else _tensorize(special_tokens_mask)
        self.offsets_mapping = None if offsets_mapping is None else _tensorize(offsets_mapping)
        self.padding_token_id = padding_token_id

    def get_padding_lengths(self) -> Dict[str, int]:
        return {name: getattr(self, name).shape[-1] for name in self.__slots__ if isinstance(getattr(self, name), torch.Tensor)}

    def as_tensor(self, padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:
        result: Dict[str, torch.Tensor] = {}
        for name, padding_length in padding_lengths.items():
            tensor = getattr(self, name)
            if len(tensor.shape) > 1:
                tensor = tensor.squeeze(0)
            result[name] = torch.nn.functional.pad(tensor, (0, padding_length - tensor.shape[-1]), value=self.padding_token_id if name == 'input_ids' else 0)
        if 'attention_mask' not in result:
            result['attention_mask'] = torch.tensor([True] * self.input_ids.shape[-1] + [False] * (padding_lengths['input_ids'] - self.input_ids.shape[-1]), dtype=torch.bool)
        return result

    def empty_field(self) -> 'TransformerTextField':
        return TransformerTextField(torch.LongTensor(), padding_token_id=self.padding_token_id)

    def batch_tensors(self, tensor_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        result = util.batch_tensor_dicts(tensor_list)
        result = {name: t.to(torch.int64) if t.dtype == torch.int32 else t for name, t in result.items()}
        return result

    def human_readable_repr(self) -> Dict[str, str]:

        def format_item(x: torch.Tensor) -> str:
            return str(x.item())

        def readable_tensor(t: torch.Tensor) -> str:
            if t.shape[-1] <= 16:
                return '[' + ', '.join(map(format_item, t)) + ']'
            else:
                return '[' + ', '.join(map(format_item, t[:8])) + ', ..., ' + ', '.join(map(format_item, t[-8:])) + ']'
        return {name: readable_tensor(getattr(self, name)) for name in self.__slots__ if isinstance(getattr(self, name), torch.Tensor)}

    def __len__(self) -> int:
        return len(self.input_ids)
