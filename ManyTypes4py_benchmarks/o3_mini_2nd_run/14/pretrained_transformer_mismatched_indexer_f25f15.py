from typing import Dict, List, Any, Optional, Callable, Union
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
    """
    Use this indexer when (for whatever reason) you are not using a corresponding
    `PretrainedTransformerTokenizer` on your input. We assume that you used a tokenizer that splits
    strings into words, while the transformer expects wordpieces as input. This indexer splits the
    words into wordpieces and flattens them out. You should use the corresponding
    `PretrainedTransformerMismatchedEmbedder` to embed these wordpieces and then pull out a single
    vector for each original word.

    Registered as a `TokenIndexer` with name "pretrained_transformer_mismatched".

    # Parameters

    model_name : `str`
        The name of the `transformers` model to use.
    namespace : `str`, optional (default=`tags`)
        We will add the tokens in the pytorch_transformer vocabulary to this vocabulary namespace.
        We use a somewhat confusing default value of `tags` so that we do not add padding or UNK
        tokens to this namespace, which would break on loading because we wouldn't find our default
        OOV token.
    max_length : `int`, optional (default = `None`)
        If positive, split the document into segments of this many tokens (including special tokens)
        before feeding into the embedder. The embedder embeds these segments independently and
        concatenate the results to get the original document representation. Should be set to
        the same value as the `max_length` option on the `PretrainedTransformerMismatchedEmbedder`.
    tokenizer_kwargs : `Dict[str, Any]`, optional (default = `None`)
        Dictionary with
        [additional arguments](https://github.com/huggingface/transformers/blob/155c782a2ccd103cf63ad48a2becd7c76a7d2115/transformers/tokenization_utils.py#L691)
        for `AutoTokenizer.from_pretrained`.
    """

    def __init__(
        self,
        model_name: str,
        namespace: str = 'tags',
        max_length: Optional[int] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._matched_indexer: PretrainedTransformerIndexer = PretrainedTransformerIndexer(
            model_name, namespace=namespace, max_length=max_length, tokenizer_kwargs=tokenizer_kwargs, **kwargs
        )
        self._allennlp_tokenizer = self._matched_indexer._allennlp_tokenizer
        self._tokenizer = self._matched_indexer._tokenizer
        self._num_added_start_tokens: int = self._matched_indexer._num_added_start_tokens
        self._num_added_end_tokens: int = self._matched_indexer._num_added_end_tokens

    def count_vocab_items(self, token: Token, counter: Dict[str, int]) -> None:
        return self._matched_indexer.count_vocab_items(token, counter)

    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary) -> Dict[str, Any]:
        self._matched_indexer._add_encoding_to_vocabulary_if_needed(vocabulary)
        # Convert tokens to text and perform intra-word tokenization.
        wordpieces, offsets = self._allennlp_tokenizer.intra_word_tokenize([t.ensure_text() for t in tokens])
        offsets = [x if x is not None else (-1, -1) for x in offsets]
        output: Dict[str, Any] = {
            'token_ids': [t.text_id for t in wordpieces],
            'mask': [True] * len(tokens),
            'type_ids': [t.type_id for t in wordpieces],
            'offsets': offsets,
            'wordpiece_mask': [True] * len(wordpieces)
        }
        return self._matched_indexer._postprocess_output(output)

    def get_empty_token_list(self) -> IndexedTokenList:
        output: IndexedTokenList = self._matched_indexer.get_empty_token_list()
        output['offsets'] = []
        output['wordpiece_mask'] = []
        return output

    def as_padded_tensor_dict(self, tokens: Dict[str, Any], padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:
        tokens = tokens.copy()
        padding_lengths = padding_lengths.copy()
        offsets_tokens = tokens.pop('offsets')
        offsets_padding_lengths: int = padding_lengths.pop('offsets')
        tensor_dict: Dict[str, torch.Tensor] = self._matched_indexer.as_padded_tensor_dict(tokens, padding_lengths)
        # pad_sequence_to_length expects a callable for default_value
        tensor_dict['offsets'] = torch.LongTensor(
            pad_sequence_to_length(offsets_tokens, offsets_padding_lengths, default_value=lambda: (0, 0))
        )
        return tensor_dict

    def __eq__(self, other: Any) -> Union[bool, Any]:
        if isinstance(other, PretrainedTransformerMismatchedIndexer):
            for key in self.__dict__:
                if key == '_tokenizer':
                    continue
                if self.__dict__[key] != other.__dict__[key]:
                    return False
            return True
        return NotImplemented
