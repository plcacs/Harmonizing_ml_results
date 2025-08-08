import codecs
import copy
import logging
import os
import re
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Union, TYPE_CHECKING
from transformers import PreTrainedTokenizer
from allennlp.common import Registrable
from allennlp.common.file_utils import cached_path, FileLock
from allennlp.common.checks import ConfigurationError
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import namespace_match

if TYPE_CHECKING:
    from allennlp.data import instance as adi

logger: logging.Logger = logging.getLogger(__name__)
DEFAULT_NON_PADDED_NAMESPACES: Tuple[str, str] = ('*tags', '*labels')
DEFAULT_PADDING_TOKEN: str = '@@PADDING@@'
DEFAULT_OOV_TOKEN: str = '@@UNKNOWN@@'
NAMESPACE_PADDING_FILE: str = 'non_padded_namespaces.txt'
_NEW_LINE_REGEX: re.Pattern = re.compile('\\n|\\r\\n')

class _NamespaceDependentDefaultDict(defaultdict):
    def __init__(self, non_padded_namespaces: Iterable[str], padded_function: Callable[[], Any], non_padded_function: Callable[[], Any]) -> None:
        self._non_padded_namespaces: Set[str] = set(non_padded_namespaces)
        self._padded_function: Callable[[], Any] = padded_function
        self._non_padded_function: Callable[[], Any] = non_padded_function
        super().__init__()

    def __missing__(self, key: str) -> Any:
        if any((namespace_match(pattern, key) for pattern in self._non_padded_namespaces)):
            value = self._non_padded_function()
        else:
            value = self._padded_function()
        dict.__setitem__(self, key, value)
        return value

    def add_non_padded_namespaces(self, non_padded_namespaces: Iterable[str]) -> None:
        self._non_padded_namespaces.update(non_padded_namespaces)

class _TokenToIndexDefaultDict(_NamespaceDependentDefaultDict):
    def __init__(self, non_padded_namespaces: Iterable[str], padding_token: str, oov_token: str) -> None:
        super().__init__(non_padded_namespaces, lambda: {padding_token: 0, oov_token: 1}, lambda: {})

class _IndexToTokenDefaultDict(_NamespaceDependentDefaultDict):
    def __init__(self, non_padded_namespaces: Iterable[str], padding_token: str, oov_token: str) -> None:
        super().__init__(non_padded_namespaces, lambda: {0: padding_token, 1: oov_token}, lambda: {})

def _read_pretrained_tokens(embeddings_file_uri: str) -> List[str]:
    from allennlp.modules.token_embedders.embedding import EmbeddingsTextFile
    logger.info('Reading pretrained tokens from: %s', embeddings_file_uri)
    tokens: List[str] = []
    with EmbeddingsTextFile(embeddings_file_uri) as embeddings_file:
        for line_number, line in enumerate(Tqdm.tqdm(embeddings_file), start=1):
            token_end: int = line.find(' ')
            if token_end >= 0:
                token: str = line[:token_end]
                tokens.append(token)
            else:
                line_begin: str = line[:20] + '...' if len(line) > 20 else line
                logger.warning('Skipping line number %d: %s', line_number, line_begin)
    return tokens

class Vocabulary(Registrable):
    def __init__(self, counter: Optional[Dict[str, Dict[str, int]]] = None, min_count: Optional[Dict[str, int]] = None, max_vocab_size: Optional[Union[int, Dict[str, int]]] = None, non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES, pretrained_files: Optional[Dict[str, str]] = None, only_include_pretrained_words: bool = False, tokens_to_add: Optional[Dict[str, List[str]]] = None, min_pretrained_embeddings: Optional[Dict[str, int]] = None, padding_token: str = DEFAULT_PADDING_TOKEN, oov_token: str = DEFAULT_OOV_TOKEN) -> None:
        ...

    @classmethod
    def from_pretrained_transformer(cls, model_name: str, namespace: str = 'tokens', oov_token: Optional[str] = None) -> 'Vocabulary':
        ...

    @classmethod
    def from_instances(cls, instances: Iterable['adi.Instance'], min_count: Optional[Dict[str, int]] = None, max_vocab_size: Optional[Union[int, Dict[str, int]]] = None, non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES, pretrained_files: Optional[Dict[str, str]] = None, only_include_pretrained_words: bool = False, tokens_to_add: Optional[Dict[str, List[str]]] = None, min_pretrained_embeddings: Optional[Dict[str, int]] = None, padding_token: str = DEFAULT_PADDING_TOKEN, oov_token: str = DEFAULT_OOV_TOKEN) -> 'Vocabulary':
        ...

    @classmethod
    def from_files(cls, directory: str, padding_token: str = DEFAULT_PADDING_TOKEN, oov_token: str = DEFAULT_OOV_TOKEN) -> 'Vocabulary':
        ...

    @classmethod
    def from_files_and_instances(cls, instances: Iterable['adi.Instance'], directory: str, padding_token: str = DEFAULT_PADDING_TOKEN, oov_token: str = DEFAULT_OOV_TOKEN, min_count: Optional[Dict[str, int]] = None, max_vocab_size: Optional[Union[int, Dict[str, int]]] = None, non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES, pretrained_files: Optional[Dict[str, str]] = None, only_include_pretrained_words: bool = False, tokens_to_add: Optional[Dict[str, List[str]]] = None, min_pretrained_embeddings: Optional[Dict[str, int]] = None) -> 'Vocabulary':
        ...

    @classmethod
    def from_pretrained_transformer_and_instances(cls, instances: Iterable['adi.Instance'], transformers: Dict[str, str], min_count: Optional[Dict[str, int]] = None, max_vocab_size: Optional[Union[int, Dict[str, int]]] = None, non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES, pretrained_files: Optional[Dict[str, str]] = None, only_include_pretrained_words: bool = False, tokens_to_add: Optional[Dict[str, List[str]]] = None, min_pretrained_embeddings: Optional[Dict[str, int]] = None, padding_token: str = DEFAULT_PADDING_TOKEN, oov_token: str = DEFAULT_OOV_TOKEN) -> 'Vocabulary':
        ...

    @classmethod
    def empty(cls) -> 'Vocabulary':
        ...

    def add_transformer_vocab(self, tokenizer: PreTrainedTokenizer, namespace: str = 'tokens') -> None:
        ...

    def set_from_file(self, filename: str, is_padded: bool = True, oov_token: str = DEFAULT_OOV_TOKEN, namespace: str = 'tokens') -> None:
        ...

    def extend_from_instances(self, instances: Iterable['adi.Instance']) -> None:
        ...

    def extend_from_vocab(self, vocab: 'Vocabulary') -> None:
        ...

    def _extend(self, counter: Optional[Dict[str, Dict[str, int]]] = None, min_count: Optional[Dict[str, int]] = None, max_vocab_size: Optional[Union[int, Dict[str, int]]] = None, non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES, pretrained_files: Optional[Dict[str, str]] = None, only_include_pretrained_words: bool = False, tokens_to_add: Optional[Dict[str, List[str]]] = None, min_pretrained_embeddings: Optional[Dict[str, int]] = None) -> None:
        ...

    def __getstate__(self) -> Dict[str, Any]:
        ...

    def __setstate__(self, state: Dict[str, Any]) -> None:
        ...

    def save_to_files(self, directory: str) -> None:
        ...

    def is_padded(self, namespace: str) -> bool:
        ...

    def add_token_to_namespace(self, token: str, namespace: str = 'tokens') -> int:
        ...

    def add_tokens_to_namespace(self, tokens: List[str], namespace: str = 'tokens') -> List[int]:
        ...

    def get_index_to_token_vocabulary(self, namespace: str = 'tokens') -> Dict[int, str]:
        ...

    def get_token_to_index_vocabulary(self, namespace: str = 'tokens') -> Dict[str, int]:
        ...

    def get_token_index(self, token: str, namespace: str = 'tokens') -> int:
        ...

    def get_token_from_index(self, index: int, namespace: str = 'tokens') -> str:
        ...

    def get_vocab_size(self, namespace: str = 'tokens') -> int:
        ...

    def get_namespaces(self) -> Set[str]:
        ...

    def __eq__(self, other: Any) -> bool:
        ...

    def __str__(self) -> str:
        ...

    def __repr__(self) -> str:
        ...

    def print_statistics(self) -> None:
        ...
