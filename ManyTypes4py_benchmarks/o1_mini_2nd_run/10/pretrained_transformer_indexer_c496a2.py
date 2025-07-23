from typing import Dict, List, Optional, Tuple, Any
import logging
import torch
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer
from allennlp.data.token_indexers.token_indexer import TokenIndexer, IndexedTokenList

logger = logging.getLogger(__name__)

@TokenIndexer.register('pretrained_transformer')
class PretrainedTransformerIndexer(TokenIndexer):
    """
    This `TokenIndexer` assumes that Tokens already have their indexes in them (see `text_id` field).
    We still require `model_name` because we want to form allennlp vocabulary from pretrained one.
    This `Indexer` is only really appropriate to use if you've also used a
    corresponding :class:`PretrainedTransformerTokenizer` to tokenize your input.  Otherwise you'll
    have a mismatch between your tokens and your vocabulary, and you'll get a lot of UNK tokens.

    Registered as a `TokenIndexer` with name "pretrained_transformer".

    # Parameters

    model_name : `str`
        The name of the `transformers` model to use.
    namespace : `str`, optional (default=`tags`)
        We will add the tokens in the pytorch_transformer vocabulary to this vocabulary namespace.
        We use a somewhat confusing default value of `tags` so that we do not add padding or UNK
        tokens to this namespace, which would break on loading because we wouldn't find our default
        OOV token.
    max_length : `int`, optional (default = `None`)
        If not None, split the document into segments of this many tokens (including special tokens)
        before feeding into the embedder. The embedder embeds these segments independently and
        concatenate the results to get the original document representation. Should be set to
        the same value as the `max_length` option on the `PretrainedTransformerEmbedder`.
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
        self._namespace: str = namespace
        self._allennlp_tokenizer: PretrainedTransformerTokenizer = PretrainedTransformerTokenizer(model_name, tokenizer_kwargs=tokenizer_kwargs)
        self._tokenizer: Any = self._allennlp_tokenizer.tokenizer
        self._added_to_vocabulary: bool = False
        self._num_added_start_tokens: int = len(self._allennlp_tokenizer.single_sequence_start_tokens)
        self._num_added_end_tokens: int = len(self._allennlp_tokenizer.single_sequence_end_tokens)
        self._max_length: Optional[int] = max_length
        if self._max_length is not None:
            num_added_tokens: int = len(self._allennlp_tokenizer.tokenize('a')) - 1
            self._effective_max_length: int = self._max_length - num_added_tokens
            if self._effective_max_length <= 0:
                raise ValueError('max_length needs to be greater than the number of special tokens inserted.')

    def _add_encoding_to_vocabulary_if_needed(self, vocab: Vocabulary) -> None:
        """
        Copies tokens from 