from typing import Dict, List, Any, Optional, Tuple
import logging
import torch
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import PretrainedTransformerIndexer, TokenIndexer
from allennlp.data.token_indexers.token_indexer import IndexedTokenList

logger = logging.getLogger(__name__)

@TokenIndexer.register('pretrained_transformer_mismatched')
class PretrainedTransformerMismatchedIndexer(TokenIndexer):
    def __init__(self, model_name: str, namespace: str = 'tags', max_length: Optional[int] = None, tokenizer_kwargs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._matched_indexer = PretrainedTransformerIndexer(model_name, namespace=namespace, max_length=max_length, tokenizer_kwargs=tokenizer_kwargs, **kwargs)
        self._allennlp_tokenizer = self._matched_indexer._allennlp_tokenizer
        self._tokenizer = self._matched_indexer._tokenizer
        self._num_added_start_tokens = self._matched_indexer._num_added_start_tokens
        self._num_added_end_tokens = self._matched_indexer._num_added_end_tokens

    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]) -> None:
        return self._matched_indexer.count_vocab_items(token, counter)

    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary) -> Dict[str, List[Any]]:
        self._matched_indexer._add_encoding_to_vocabulary_if_needed(vocabulary)
        wordpieces, offsets = self._allennlp_tokenizer.intra_word_tokenize([t.ensure_text() for t in tokens])
        offsets = [x if x is not None else (-1, -1) for x in offsets]
        output = {
            'token_ids': [t.text_id for t in wordpieces],
            'mask': [True] * len(tokens),
            'type_ids': [t.type_id for t in wordpieces],
            'offsets': offsets,
            'wordpiece_mask': [True] * len(wordpieces)
        }
        return self._matched_indexer._postprocess_output(output)

    def get_empty_token_list(self) -> IndexedTokenList:
        output = self._matched_indexer.get_empty_token_list()
        output['offsets'] = []
        output['wordpiece_mask'] = []
        return output

    def as_padded_tensor_dict(self, tokens: IndexedTokenList, padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:
        tokens = tokens.copy()
        padding_lengths = padding_lengths.copy()
        offsets_tokens = tokens.pop('offsets')
        offsets_padding_lengths = padding_lengths.pop('offsets')
        tensor_dict = self._matched_indexer.as_padded_tensor_dict(tokens, padding_lengths)
        tensor_dict['offsets'] = torch.LongTensor(pad_sequence_to_length(offsets_tokens, offsets_padding_lengths, default_value=lambda: (0, 0)))
        return tensor_dict

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, PretrainedTransformerMismatchedIndexer):
            for key in self.__dict__:
                if key == '_tokenizer':
                    continue
                if self.__dict__[key] != other.__dict__[key]:
                    return False
            return True
        return NotImplemented
