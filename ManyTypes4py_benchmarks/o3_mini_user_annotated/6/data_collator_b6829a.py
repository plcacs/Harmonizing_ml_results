from typing import List, Any

from transformers.data.data_collator import DataCollatorForLanguageModeling
from allennlp.common import Registrable, cached_transformers
from allennlp.data.batch import Batch
from allennlp.data.data_loaders.data_loader import TensorDict
from allennlp.data.instance import Instance


def allennlp_collate(instances: List[Instance]) -> TensorDict:
    """
    This is the default function used to turn a list of `Instance`s into a `TensorDict`
    batch.
    """
    batch: Batch = Batch(instances)
    return batch.as_tensor_dict()


class DataCollator(Registrable):
    """
    This class is similar with `DataCollator` in [Transformers]
    (https://github.com/huggingface/transformers/blob/master/src/transformers/data/data_collator.py)
    Allow to do some dynamic operations for tensor in different batches.
    This method runs before each epoch to convert `List[Instance]` to `TensorDict`.
    """

    default_implementation = "allennlp"

    def __call__(self, instances: List[Instance]) -> TensorDict:
        raise NotImplementedError


@DataCollator.register("allennlp")
class DefaultDataCollator(DataCollator):
    def __call__(self, instances: List[Instance]) -> TensorDict:
        return allennlp_collate(instances)


@DataCollator.register("language_model")
class LanguageModelingDataCollator(DataCollator):
    """
    Register as a `DataCollator` with name `language_model`.
    Used for language modeling.
    """

    def __init__(
        self,
        model_name: str,
        mlm: bool = True,
        mlm_probability: float = 0.15,
        filed_name: str = "source",
        namespace: str = "tokens",
    ) -> None:
        self._field_name: str = filed_name
        self._namespace: str = namespace
        tokenizer = cached_transformers.get_tokenizer(model_name)
        self._collator: DataCollatorForLanguageModeling = DataCollatorForLanguageModeling(
            tokenizer, mlm, mlm_probability
        )
        if hasattr(self._collator, "mask_tokens"):
            # For compatibility with transformers < 4.10
            self._mask_tokens = self._collator.mask_tokens
        else:
            self._mask_tokens = self._collator.torch_mask_tokens

    def __call__(self, instances: List[Instance]) -> TensorDict:
        tensor_dicts: TensorDict = allennlp_collate(instances)
        tensor_dicts = self.process_tokens(tensor_dicts)
        return tensor_dicts

    def process_tokens(self, tensor_dicts: TensorDict) -> TensorDict:
        inputs: Any = tensor_dicts[self._field_name][self._namespace]["token_ids"]
        inputs, labels = self._mask_tokens(inputs)
        tensor_dicts[self._field_name][self._namespace]["token_ids"] = inputs
        tensor_dicts[self._field_name][self._namespace]["labels"] = labels
        return tensor_dicts