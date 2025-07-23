"""
A Vocabulary maps strings to integers, allowing for strings to be mapped to an
out-of-vocabulary token.
"""
import codecs
import copy
import logging
import os
import re
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Union, TYPE_CHECKING, DefaultDict, Tuple, TypeVar
from transformers import PreTrainedTokenizer
from allennlp.common import Registrable
from allennlp.common.file_utils import cached_path, FileLock
from allennlp.common.checks import ConfigurationError
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import namespace_match

if TYPE_CHECKING:
    from allennlp.data import instance as adi

logger = logging.getLogger(__name__)
DEFAULT_NON_PADDED_NAMESPACES: Tuple[str, str] = ('*tags', '*labels')
DEFAULT_PADDING_TOKEN: str = '@@PADDING@@'
DEFAULT_OOV_TOKEN: str = '@@UNKNOWN@@'
NAMESPACE_PADDING_FILE: str = 'non_padded_namespaces.txt'
_NEW_LINE_REGEX: re.Pattern = re.compile('\\n|\\r\\n')

T = TypeVar('T')

class _NamespaceDependentDefaultDict(defaultdict):
    """
    [defaultdict] where the default value is dependent on the key that is passed.
    """
    def __init__(
        self,
        non_padded_namespaces: Iterable[str],
        padded_function: Callable[[], T],
        non_padded_function: Callable[[], T]
    ) -> None:
        self._non_padded_namespaces: Set[str] = set(non_padded_namespaces)
        self._padded_function: Callable[[], T] = padded_function
        self._non_padded_function: Callable[[], T] = non_padded_function
        super().__init__()

    def __missing__(self, key: str) -> T:
        if any((namespace_match(pattern, key) for pattern in self._non_padded_namespaces)):
            value = self._non_padded_function()
        else:
            value = self._padded_function()
        dict.__setitem__(self, key, value)
        return value

    def add_non_padded_namespaces(self, non_padded_namespaces: Iterable[str]) -> None:
        self._non_padded_namespaces.update(non_padded_namespaces)

class _TokenToIndexDefaultDict(_NamespaceDependentDefaultDict):
    def __init__(
        self,
        non_padded_namespaces: Iterable[str],
        padding_token: str,
        oov_token: str
    ) -> None:
        super().__init__(
            non_padded_namespaces,
            lambda: {padding_token: 0, oov_token: 1},
            lambda: {}
        )

class _IndexToTokenDefaultDict(_NamespaceDependentDefaultDict):
    def __init__(
        self,
        non_padded_namespaces: Iterable[str],
        padding_token: str,
        oov_token: str
    ) -> None:
        super().__init__(
            non_padded_namespaces,
            lambda: {0: padding_token, 1: oov_token},
            lambda: {}
        )

def _read_pretrained_tokens(embeddings_file_uri: str) -> List[str]:
    from allennlp.modules.token_embedders.embedding import EmbeddingsTextFile
    logger.info('Reading pretrained tokens from: %s', embeddings_file_uri)
    tokens: List[str] = []
    with EmbeddingsTextFile(embeddings_file_uri) as embeddings_file:
        for line_number, line in enumerate(Tqdm.tqdm(embeddings_file), start=1):
            token_end = line.find(' ')
            if token_end >= 0:
                token = line[:token_end]
                tokens.append(token)
            else:
                line_begin = line[:20] + '...' if len(line) > 20 else line
                logger.warning('Skipping line number %d: %s', line_number, line_begin)
    return tokens

class Vocabulary(Registrable):
    default_implementation: str = 'from_instances'

    def __init__(
        self,
        counter: Optional[Dict[str, Dict[str, int]]] = None,
        min_count: Optional[Dict[str, int]] = None,
        max_vocab_size: Optional[Union[int, Dict[str, int]]] = None,
        non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES,
        pretrained_files: Optional[Dict[str, str]] = None,
        only_include_pretrained_words: bool = False,
        tokens_to_add: Optional[Dict[str, List[str]]] = None,
        min_pretrained_embeddings: Optional[Dict[str, int]] = None,
        padding_token: Optional[str] = DEFAULT_PADDING_TOKEN,
        oov_token: Optional[str] = DEFAULT_OOV_TOKEN
    ) -> None:
        self._padding_token: str = padding_token if padding_token is not None else DEFAULT_PADDING_TOKEN
        self._oov_token: str = oov_token if oov_token is not None else DEFAULT_OOV_TOKEN
        self._non_padded_namespaces: Set[str] = set(non_padded_namespaces)
        self._token_to_index: _TokenToIndexDefaultDict = _TokenToIndexDefaultDict(
            self._non_padded_namespaces, self._padding_token, self._oov_token
        )
        self._index_to_token: _IndexToTokenDefaultDict = _IndexToTokenDefaultDict(
            self._non_padded_namespaces, self._padding_token, self._oov_token
        )
        self._retained_counter: Optional[Dict[str, Dict[str, int]]] = None
        self._extend(
            counter, min_count, max_vocab_size, non_padded_namespaces,
            pretrained_files, only_include_pretrained_words,
            tokens_to_add, min_pretrained_embeddings
        )

    @classmethod
    def from_pretrained_transformer(
        cls,
        model_name: str,
        namespace: str = 'tokens',
        oov_token: Optional[str] = None
    ) -> 'Vocabulary':
        from allennlp.common import cached_transformers
        tokenizer = cached_transformers.get_tokenizer(model_name)
        if oov_token is None:
            if hasattr(tokenizer, '_unk_token'):
                oov_token = tokenizer._unk_token
            elif hasattr(tokenizer, 'special_tokens_map'):
                oov_token = tokenizer.special_tokens_map.get('unk_token')
        vocab = cls(non_padded_namespaces=[namespace], oov_token=oov_token)
        vocab.add_transformer_vocab(tokenizer, namespace)
        return vocab

    @classmethod
    def from_instances(
        cls,
        instances: Iterable['adi.Instance'],
        min_count: Optional[Dict[str, int]] = None,
        max_vocab_size: Optional[Union[int, Dict[str, int]]] = None,
        non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES,
        pretrained_files: Optional[Dict[str, str]] = None,
        only_include_pretrained_words: bool = False,
        tokens_to_add: Optional[Dict[str, List[str]]] = None,
        min_pretrained_embeddings: Optional[Dict[str, int]] = None,
        padding_token: Optional[str] = DEFAULT_PADDING_TOKEN,
        oov_token: Optional[str] = DEFAULT_OOV_TOKEN
    ) -> 'Vocabulary':
        logger.info('Fitting token dictionary from dataset.')
        padding_token = padding_token if padding_token is not None else DEFAULT_PADDING_TOKEN
        oov_token = oov_token if oov_token is not None else DEFAULT_OOV_TOKEN
        namespace_token_counts: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))
        for instance in Tqdm.tqdm(instances, desc='building vocab'):
            instance.count_vocab_items(namespace_token_counts)
        return cls(
            counter=namespace_token_counts,
            min_count=min_count,
            max_vocab_size=max_vocab_size,
            non_padded_namespaces=non_padded_namespaces,
            pretrained_files=pretrained_files,
            only_include_pretrained_words=only_include_pretrained_words,
            tokens_to_add=tokens_to_add,
            min_pretrained_embeddings=min_pretrained_embeddings,
            padding_token=padding_token,
            oov_token=oov_token
        )

    @classmethod
    def from_files(
        cls,
        directory: str,
        padding_token: Optional[str] = DEFAULT_PADDING_TOKEN,
        oov_token: Optional[str] = DEFAULT_OOV_TOKEN
    ) -> 'Vocabulary':
        logger.info('Loading token dictionary from %s.', directory)
        padding_token = padding_token if padding_token is not None else DEFAULT_PADDING_TOKEN
        oov_token = oov_token if oov_token is not None else DEFAULT_OOV_TOKEN
        if not os.path.isdir(directory):
            base_directory = cached_path(directory, extract_archive=True)
            vocab_subdir = os.path.join(base_directory, 'vocabulary')
            if os.path.isdir(vocab_subdir):
                directory = vocab_subdir
            elif os.path.isdir(base_directory):
                directory = base_directory
            else:
                raise ConfigurationError(f'{directory} is neither a directory nor an archive')
        with FileLock(os.path.join(directory, '.lock'), read_only_ok=True):
            with codecs.open(os.path.join(directory, NAMESPACE_PADDING_FILE), 'r', 'utf-8') as namespace_file:
                non_padded_namespaces = [namespace_str.strip() for namespace_str in namespace_file]
            vocab = cls(non_padded_namespaces=non_padded_namespaces, padding_token=padding_token, oov_token=oov_token)
            for namespace_filename in os.listdir(directory):
                if namespace_filename == NAMESPACE_PADDING_FILE:
                    continue
                if namespace_filename.startswith('.'):
                    continue
                namespace = namespace_filename.replace('.txt', '')
                if any((namespace_match(pattern, namespace) for pattern in non_padded_namespaces)):
                    is_padded = False
                else:
                    is_padded = True
                filename = os.path.join(directory, namespace_filename)
                vocab.set_from_file(filename, is_padded, namespace=namespace, oov_token=oov_token)
        return vocab

    @classmethod
    def from_files_and_instances(
        cls,
        instances: Iterable['adi.Instance'],
        directory: str,
        padding_token: Optional[str] = DEFAULT_PADDING_TOKEN,
        oov_token: Optional[str] = DEFAULT_OOV_TOKEN,
        min_count: Optional[Dict[str, int]] = None,
        max_vocab_size: Optional[Union[int, Dict[str, int]]] = None,
        non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES,
        pretrained_files: Optional[Dict[str, str]] = None,
        only_include_pretrained_words: bool = False,
        tokens_to_add: Optional[Dict[str, List[str]]] = None,
        min_pretrained_embeddings: Optional[Dict[str, int]] = None
    ) -> 'Vocabulary':
        vocab = cls.from_files(directory, padding_token, oov_token)
        logger.info('Fitting token dictionary from dataset.')
        namespace_token_counts: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))
        for instance in Tqdm.tqdm(instances):
            instance.count_vocab_items(namespace_token_counts)
        vocab._extend(
            counter=namespace_token_counts,
            min_count=min_count,
            max_vocab_size=max_vocab_size,
            non_padded_namespaces=non_padded_namespaces,
            pretrained_files=pretrained_files,
            only_include_pretrained_words=only_include_pretrained_words,
            tokens_to_add=tokens_to_add,
            min_pretrained_embeddings=min_pretrained_embeddings
        )
        return vocab

    @classmethod
    def from_pretrained_transformer_and_instances(
        cls,
        instances: Iterable['adi.Instance'],
        transformers: Dict[str, str],
        min_count: Optional[Dict[str, int]] = None,
        max_vocab_size: Optional[Union[int, Dict[str, int]]] = None,
        non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES,
        pretrained_files: Optional[Dict[str, str]] = None,
        only_include_pretrained_words: bool = False,
        tokens_to_add: Optional[Dict[str, List[str]]] = None,
        min_pretrained_embeddings: Optional[Dict[str, int]] = None,
        padding_token: Optional[str] = DEFAULT_PADDING_TOKEN,
        oov_token: Optional[str] = DEFAULT_OOV_TOKEN
    ) -> 'Vocabulary':
        vocab = cls.from_instances(
            instances=instances,
            min_count=min_count,
            max_vocab_size=max_vocab_size,
            non_padded_namespaces=non_padded_namespaces,
            pretrained_files=pretrained_files,
            only_include_pretrained_words=only_include_pretrained_words,
            tokens_to_add=tokens_to_add,
            min_pretrained_embeddings=min_pretrained_embeddings,
            padding_token=padding_token,
            oov_token=oov_token
        )
        for namespace, model_name in transformers.items():
            transformer_vocab = cls.from_pretrained_transformer(model_name=model_name, namespace=namespace)
            vocab.extend_from_vocab(transformer_vocab)
        return vocab

    @classmethod
    def empty(cls) -> 'Vocabulary':
        return cls()

    def add_transformer_vocab(self, tokenizer: PreTrainedTokenizer, namespace: str = 'tokens') -> None:
        try:
            vocab_items = tokenizer.get_vocab().items()
        except NotImplementedError:
            vocab_items = ((tokenizer.convert_ids_to_tokens(idx), idx) for idx in range(tokenizer.vocab_size))
        for word, idx in vocab_items:
            self._token_to_index[namespace][word] = idx
            self._index_to_token[namespace][idx] = word
        self._non_padded_namespaces.add(namespace)

    def set_from_file(
        self,
        filename: str,
        is_padded: bool = True,
        oov_token: str = DEFAULT_OOV_TOKEN,
        namespace: str = 'tokens'
    ) -> None:
        if is_padded:
            self._token_to_index[namespace] = {self._padding_token: 0}
            self._index_to_token[namespace] = {0: self._padding_token}
        else:
            self._token_to_index[namespace] = {}
            self._index_to_token[namespace] = {}
        with codecs.open(filename, 'r', 'utf-8') as input_file:
            lines = _NEW_LINE_REGEX.split(input_file.read())
            if lines and lines[-1] == '':
                lines = lines[:-1]
            for i, line in enumerate(lines):
                index = i + 1 if is_padded else i
                token = line.replace('@@NEWLINE@@', '\n')
                if token == oov_token:
                    token = self._oov_token
                self._token_to_index[namespace][token] = index
                self._index_to_token[namespace][index] = token
        if is_padded:
            assert self._oov_token in self._token_to_index[namespace], 'OOV token not found!'

    def extend_from_instances(self, instances: Iterable['adi.Instance']) -> None:
        logger.info('Fitting token dictionary from dataset.')
        namespace_token_counts: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))
        for instance in Tqdm.tqdm(instances):
            instance.count_vocab_items(namespace_token_counts)
        self._extend(counter=namespace_token_counts)

    def extend_from_vocab(self, vocab: 'Vocabulary') -> None:
        self._non_padded_namespaces.update(vocab._non_padded_namespaces)
        self._token_to_index._non_padded_namespaces.update(vocab._non_padded_namespaces)
        self._index_to_token._non_padded_namespaces.update(vocab._non_padded_namespaces)
        for namespace in vocab.get_namespaces():
            for token in vocab.get_token_to_index_vocabulary(namespace):
                self.add_token_to_namespace(token, namespace)

    def _extend(
        self,
        counter: Optional[Dict[str, Dict[str, int]]] = None,
        min_count: Optional[Dict[str, int]] = None,
        max_vocab_size: Optional[Union[int, Dict[str, int]]] = None,
        non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES,
        pretrained_files: Optional[Dict[str, str]] = None,
        only_include_pretrained_words: bool = False,
        tokens_to_add: Optional[Dict[str, List[str]]] = None,
        min_pretrained_embeddings: Optional[Dict[str, int]] = None
    ) -> None:
        if min_count is not None:
            for key in min_count:
                if counter is not None and key not in counter or counter is None:
                    raise ConfigurationError(f"The key '{key}' is present in min_count but not in counter")
        if not isinstance(max_vocab_size, dict):
            int_max_vocab_size = max_vocab_size
            max_vocab_size = defaultdict(lambda: int_max_vocab_size)
        min_count = min_count or {}
        pretrained_files = pretrained_files or {}
        min_pretrained_embeddings = min_pretrained_embeddings or {}
        non_padded_namespaces = set(non_padded_namespaces)
        counter = counter or {}
        tokens_to_add = tokens_to_add or {}
        self._retained_counter = counter
        current_namespaces = {*self._token_to_index}
        extension_namespaces = {*counter, *tokens_to_add}
        for namespace in current_namespaces & extension_namespaces:
            original_padded = not any((namespace_match(pattern, namespace) for pattern in self._non_padded_names