from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import torch
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer
from allennlp.data.token_indexers.token_indexer import TokenIndexer, IndexedTokenList
logger = logging.getLogger(__name__)

@TokenIndexer.register('pretrained_transformer')
class PretrainedTransformerIndexer(TokenIndexer):
    def __init__(
        self,
        model_name: str,
        namespace: str = 'tags',
        max_length: Optional[int] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._namespace = namespace
        self._allennlp_tokenizer = PretrainedTransformerTokenizer(model_name, tokenizer_kwargs=tokenizer_kwargs)
        self._tokenizer = self._allennlp_tokenizer.tokenizer
        self._added_to_vocabulary = False
        self._num_added_start_tokens = len(self._allennlp_tokenizer.single_sequence_start_tokens)
        self._num_added_end_tokens = len(self._allennlp_tokenizer.single_sequence_end_tokens)
        self._max_length = max_length
        if self._max_length is not None:
            num_added_tokens = len(self._allennlp_tokenizer.tokenize('a')) - 1
            self._effective_max_length = self._max_length - num_added_tokens
            if self._effective_max_length <= 0:
                raise ValueError('max_length needs to be greater than the number of special tokens inserted.')

    def _add_encoding_to_vocabulary_if_needed(self, vocab: Vocabulary) -> None:
        if self._added_to_vocabulary:
            return
        vocab.add_transformer_vocab(self._tokenizer, self._namespace)
        self._added_to_vocabulary = True

    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]) -> None:
        pass

    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary) -> Dict[str, List[int]]:
        self._add_encoding_to_vocabulary_if_needed(vocabulary)
        indices, type_ids = self._extract_token_and_type_ids(tokens)
        output: Dict[str, List[Union[int, bool]]] = {
            'token_ids': indices,
            'mask': [True] * len(indices),
            'type_ids': type_ids or [0] * len(indices)
        }
        return self._postprocess_output(output)

    def indices_to_tokens(self, indexed_tokens: Dict[str, List[int]], vocabulary: Vocabulary) -> List[Token]:
        self._add_encoding_to_vocabulary_if_needed(vocabulary)
        token_ids = indexed_tokens['token_ids']
        type_ids = indexed_tokens.get('type_ids')
        return [
            Token(
                text=vocabulary.get_token_from_index(token_ids[i], self._namespace),
                text_id=token_ids[i],
                type_id=type_ids[i] if type_ids is not None else None
            )
            for i in range(len(token_ids))
        ]

    def _extract_token_and_type_ids(self, tokens: List[Token]) -> Tuple[List[int], List[int]]:
        indices: List[int] = []
        type_ids: List[int] = []
        for token in tokens:
            indices.append(token.text_id if token.text_id is not None else self._tokenizer.convert_tokens_to_ids(token.text))
            type_ids.append(token.type_id if token.type_id is not None else 0)
        return (indices, type_ids)

    def _postprocess_output(self, output: Dict[str, List[Union[int, bool]]]) -> Dict[str, List[Union[int, bool]]:
        if self._max_length is not None:
            indices = output['token_ids']
            type_ids = output.get('type_ids', [0] * len(indices))
            indices = indices[self._num_added_start_tokens:len(indices) - self._num_added_end_tokens]
            type_ids = type_ids[self._num_added_start_tokens:len(type_ids) - self._num_added_end_tokens]
            folded_indices = [indices[i:i + self._effective_max_length] for i in range(0, len(indices), self._effective_max_length)]
            folded_type_ids = [type_ids[i:i + self._effective_max_length] for i in range(0, len(type_ids), self._effective_max_length)]
            folded_indices = [self._tokenizer.build_inputs_with_special_tokens(segment) for segment in folded_indices]
            single_sequence_start_type_ids = [t.type_id for t in self._allennlp_tokenizer.single_sequence_start_tokens]
            single_sequence_end_type_ids = [t.type_id for t in self._allennlp_tokenizer.single_sequence_end_tokens]
            folded_type_ids = [single_sequence_start_type_ids + segment + single_sequence_end_type_ids for segment in folded_type_ids]
            assert all((len(segment_indices) == len(segment_type_ids) for segment_indices, segment_type_ids in zip(folded_indices, folded_type_ids)))
            indices = [i for segment in folded_indices for i in segment]
            type_ids = [i for segment in folded_type_ids for i in segment]
            output['token_ids'] = indices
            output['type_ids'] = type_ids
            output['segment_concat_mask'] = [True] * len(indices)
        return output

    def get_empty_token_list(self) -> Dict[str, List[Union[int, bool]]]:
        output: Dict[str, List[Union[int, bool]]] = {'token_ids': [], 'mask': [], 'type_ids': []}
        if self._max_length is not None:
            output['segment_concat_mask'] = []
        return output

    def as_padded_tensor_dict(
        self,
        tokens: Dict[str, List[Union[int, bool]]],
        padding_lengths: Dict[str, int]
    ) -> Dict[str, torch.Tensor]:
        tensor_dict: Dict[str, torch.Tensor] = {}
        for key, val in tokens.items():
            if key == 'type_ids':
                padding_value = 0
                mktensor = torch.LongTensor
            elif key == 'mask' or key == 'wordpiece_mask':
                padding_value = False
                mktensor = torch.BoolTensor
            elif len(val) > 0 and isinstance(val[0], bool):
                padding_value = False
                mktensor = torch.BoolTensor
            else:
                padding_value = self._tokenizer.pad_token_id
                if padding_value is None:
                    padding_value = 0
                mktensor = torch.LongTensor
            tensor = mktensor(pad_sequence_to_length(val, padding_lengths[key], default_value=lambda: padding_value))
            tensor_dict[key] = tensor
        return tensor_dict

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, PretrainedTransformerIndexer):
            for key in self.__dict__:
                if key == '_tokenizer':
                    continue
                if self.__dict__[key] != other.__dict__[key]:
                    return False
            return True
        return NotImplemented
