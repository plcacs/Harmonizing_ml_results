from typing import Dict, List, Optional, Any
import itertools
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer, IndexedTokenList
_DEFAULT_VALUE = 'THIS IS A REALLY UNLIKELY VALUE THAT HAS TO BE A STRING'

@TokenIndexer.register('single_id')
class SingleIdTokenIndexer(TokenIndexer):
    def __init__(self, 
                 namespace: Optional[str] = 'tokens', 
                 lowercase_tokens: bool = False, 
                 start_tokens: Optional[List[str]] = None, 
                 end_tokens: Optional[List[str]] = None, 
                 feature_name: str = 'text', 
                 default_value: str = _DEFAULT_VALUE, 
                 token_min_padding_length: int = 0) -> None:
        super().__init__(token_min_padding_length)
        self.namespace = namespace
        self.lowercase_tokens = lowercase_tokens
        self._start_tokens = [Token(st) for st in start_tokens or []]
        self._end_tokens = [Token(et) for et in end_tokens or []]
        self._feature_name = feature_name
        self._default_value = default_value

    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]) -> None:
        if self.namespace is not None:
            text = self._get_feature_value(token)
            if self.lowercase_tokens:
                text = text.lower()
            counter[self.namespace][text] += 1

    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary) -> Dict[str, List[int]]:
        indices = []
        for token in itertools.chain(self._start_tokens, tokens, self._end_tokens):
            text = self._get_feature_value(token)
            if self.namespace is None:
                indices.append(text)
            else:
                if self.lowercase_tokens:
                    text = text.lower()
                indices.append(vocabulary.get_token_index(text, self.namespace))
        return {'tokens': indices}

    def get_empty_token_list(self) -> Dict[str, List[int]]:
        return {'tokens': []}

    def _get_feature_value(self, token: Token) -> str:
        text = getattr(token, self._feature_name)
        if text is None:
            if self._default_value is not _DEFAULT_VALUE:
                text = self._default_value
            else:
                raise ValueError(f'{token} did not have attribute {self._feature_name}. If you want to ignore this kind of error, give a default value in the constructor of this indexer.')
        return text

    def _to_params(self) -> Dict[str, Any]:
        return {'namespace': self.namespace, 
                'lowercase_tokens': self.lowercase_tokens, 
                'start_tokens': [t.text for t in self._start_tokens], 
                'end_tokens': [t.text for t in self._end_tokens], 
                'feature_name': self._feature_name, 
                'default_value': self._default_value, 
                'token_min_padding_length': self._token_min_padding_length}
