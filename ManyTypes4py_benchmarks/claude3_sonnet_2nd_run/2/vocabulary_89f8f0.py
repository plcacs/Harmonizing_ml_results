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
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Union, TYPE_CHECKING, DefaultDict, Iterator, Tuple
from transformers import PreTrainedTokenizer
from allennlp.common import Registrable
from allennlp.common.file_utils import cached_path, FileLock
from allennlp.common.checks import ConfigurationError
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import namespace_match
if TYPE_CHECKING:
    from allennlp.data import instance as adi
logger = logging.getLogger(__name__)
DEFAULT_NON_PADDED_NAMESPACES = ('*tags', '*labels')
DEFAULT_PADDING_TOKEN = '@@PADDING@@'
DEFAULT_OOV_TOKEN = '@@UNKNOWN@@'
NAMESPACE_PADDING_FILE = 'non_padded_namespaces.txt'
_NEW_LINE_REGEX = re.compile('\\n|\\r\\n')

class _NamespaceDependentDefaultDict(defaultdict):
    """
    This is a [defaultdict]
    (https://docs.python.org/2/library/collections.html#collections.defaultdict) where the
    default value is dependent on the key that is passed.

    We use "namespaces" in the :class:`Vocabulary` object to keep track of several different
    mappings from strings to integers, so that we have a consistent API for mapping words, tags,
    labels, characters, or whatever else you want, into integers.  The issue is that some of those
    namespaces (words and characters) should have integers reserved for padding and
    out-of-vocabulary tokens, while others (labels and tags) shouldn't.  This class allows you to
    specify filters on the namespace (the key used in the `defaultdict`), and use different
    default values depending on whether the namespace passes the filter.

    To do filtering, we take a set of `non_padded_namespaces`.  This is a set of strings
    that are either matched exactly against the keys, or treated as suffixes, if the
    string starts with `*`.  In other words, if `*tags` is in `non_padded_namespaces` then
    `passage_tags`, `question_tags`, etc. (anything that ends with `tags`) will have the
    `non_padded` default value.

    # Parameters

    non_padded_namespaces : `Iterable[str]`
        A set / list / tuple of strings describing which namespaces are not padded.  If a namespace
        (key) is missing from this dictionary, we will use :func:`namespace_match` to see whether
        the namespace should be padded.  If the given namespace matches any of the strings in this
        list, we will use `non_padded_function` to initialize the value for that namespace, and
        we will use `padded_function` otherwise.
    padded_function : `Callable[[], Any]`
        A zero-argument function to call to initialize a value for a namespace that `should` be
        padded.
    non_padded_function : `Callable[[], Any]`
        A zero-argument function to call to initialize a value for a namespace that should `not` be
        padded.
    """

    def __init__(self, non_padded_namespaces: Iterable[str], padded_function: Callable[[], Any], non_padded_function: Callable[[], Any]) -> None:
        self._non_padded_namespaces: Set[str] = set(non_padded_namespaces)
        self._padded_function = padded_function
        self._non_padded_function = non_padded_function
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
            token_end = line.find(' ')
            if token_end >= 0:
                token = line[:token_end]
                tokens.append(token)
            else:
                line_begin = line[:20] + '...' if len(line) > 20 else line
                logger.warning('Skipping line number %d: %s', line_number, line_begin)
    return tokens

class Vocabulary(Registrable):
    """
    A Vocabulary maps strings to integers, allowing for strings to be mapped to an
    out-of-vocabulary token.

    Vocabularies are fit to a particular dataset, which we use to decide which tokens are
    in-vocabulary.

    Vocabularies also allow for several different namespaces, so you can have separate indices for
    'a' as a word, and 'a' as a character, for instance, and so we can use this object to also map
    tag and label strings to indices, for a unified :class:`~.fields.field.Field` API.  Most of the
    methods on this class allow you to pass in a namespace; by default we use the 'tokens'
    namespace, and you can omit the namespace argument everywhere and just use the default.

    This class is registered as a `Vocabulary` with four different names, which all point to
    different `@classmethod` constructors found in this class.  `from_instances` is registered as
    "from_instances", `from_files` is registered as "from_files", `from_files_and_instances` is
    registered as "extend", and `empty` is registered as "empty".  If you are using a configuration
    file to construct a vocabulary, you can use any of those strings as the "type" key in the
    configuration file to use the corresponding `@classmethod` to construct the object.
    "from_instances" is the default.  Look at the docstring for the `@classmethod` to see what keys
    are allowed in the configuration file (when there is an `instances` argument to the
    `@classmethod`, it will be passed in separately and does not need a corresponding key in the
    configuration file).

    # Parameters

    counter : `Dict[str, Dict[str, int]]`, optional (default=`None`)
        A collection of counts from which to initialize this vocabulary.  We will examine the
        counts and, together with the other parameters to this class, use them to decide which
        words are in-vocabulary.  If this is `None`, we just won't initialize the vocabulary with
        anything.

    min_count : `Dict[str, int]`, optional (default=`None`)
        When initializing the vocab from a counter, you can specify a minimum count, and every
        token with a count less than this will not be added to the dictionary.  These minimum
        counts are `namespace-specific`, so you can specify different minimums for labels versus
        words tokens, for example.  If a namespace does not have a key in the given dictionary, we
        will add all seen tokens to that namespace.

    max_vocab_size : `Union[int, Dict[str, int]]`, optional (default=`None`)
        If you want to cap the number of tokens in your vocabulary, you can do so with this
        parameter.  If you specify a single integer, every namespace will have its vocabulary fixed
        to be no larger than this.  If you specify a dictionary, then each namespace in the
        `counter` can have a separate maximum vocabulary size.  Any missing key will have a value
        of `None`, which means no cap on the vocabulary size.

    non_padded_namespaces : `Iterable[str]`, optional
        By default, we assume you are mapping word / character tokens to integers, and so you want
        to reserve word indices for padding and out-of-vocabulary tokens.  However, if you are
        mapping NER or SRL tags, or class labels, to integers, you probably do not want to reserve
        indices for padding and out-of-vocabulary tokens.  Use this field to specify which
        namespaces should `not` have padding and OOV tokens added.

        The format of each element of this is either a string, which must match field names
        exactly,  or `*` followed by a string, which we match as a suffix against field names.

        We try to make the default here reasonable, so that you don't have to think about this.
        The default is `("*tags", "*labels")`, so as long as your namespace ends in "tags" or
        "labels" (which is true by default for all tag and label fields in this code), you don't
        have to specify anything here.

    pretrained_files : `Dict[str, str]`, optional
        If provided, this map specifies the path to optional pretrained embedding files for each
        namespace. This can be used to either restrict the vocabulary to only words which appear
        in this file, or to ensure that any words in this file are included in the vocabulary
        regardless of their count, depending on the value of `only_include_pretrained_words`.
        Words which appear in the pretrained embedding file but not in the data are NOT included
        in the Vocabulary.

    min_pretrained_embeddings : `Dict[str, int]`, optional
        Specifies for each namespace a minimum number of lines (typically the
        most common words) to keep from pretrained embedding files, even for words not
        appearing in the data. By default the minimum number of lines to keep is 0.
        You can automatically include all lines for a namespace by setting the minimum number of lines
        to `-1`.

    only_include_pretrained_words : `bool`, optional (default=`False`)
        This defines the strategy for using any pretrained embedding files which may have been
        specified in `pretrained_files`.

        If `False`, we use an inclusive strategy and include both words in the `counter`
        that have a count of at least `min_count` and words from the pretrained file
        that are within the first `N` lines defined by `min_pretrained_embeddings`.

        If `True`, we use an exclusive strategy where words are only included in the `Vocabulary`
        if they are in the pretrained embedding file. Their count must also be at least `min_count`
        or they must be listed in the embedding file within the first `N` lines defined
        by `min_pretrained_embeddings`.

    tokens_to_add : `Dict[str, List[str]]`, optional (default=`None`)
        If given, this is a list of tokens to add to the vocabulary, keyed by the namespace to add
        the tokens to.  This is a way to be sure that certain items appear in your vocabulary,
        regardless of any other vocabulary computation.

    padding_token : `str`,  optional (default=`DEFAULT_PADDING_TOKEN`)
        If given, this the string used for padding.

    oov_token : `str`,  optional (default=`DEFAULT_OOV_TOKEN`)
        If given, this the string used for the out of vocabulary (OOVs) tokens.

    """
    default_implementation = 'from_instances'

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
        padding_token: str = DEFAULT_PADDING_TOKEN,
        oov_token: str = DEFAULT_OOV_TOKEN
    ) -> None:
        self._padding_token = padding_token if padding_token is not None else DEFAULT_PADDING_TOKEN
        self._oov_token = oov_token if oov_token is not None else DEFAULT_OOV_TOKEN
        self._non_padded_namespaces: Set[str] = set(non_padded_namespaces)
        self._token_to_index: _TokenToIndexDefaultDict = _TokenToIndexDefaultDict(self._non_padded_namespaces, self._padding_token, self._oov_token)
        self._index_to_token: _IndexToTokenDefaultDict = _IndexToTokenDefaultDict(self._non_padded_namespaces, self._padding_token, self._oov_token)
        self._retained_counter: Optional[Dict[str, Dict[str, int]]] = None
        self._extend(counter, min_count, max_vocab_size, non_padded_namespaces, pretrained_files, only_include_pretrained_words, tokens_to_add, min_pretrained_embeddings)

    @classmethod
    def from_pretrained_transformer(cls, model_name: str, namespace: str = 'tokens', oov_token: Optional[str] = None) -> 'Vocabulary':
        """
        Initialize a vocabulary from the vocabulary of a pretrained transformer model.
        If `oov_token` is not given, we will try to infer it from the transformer tokenizer.
        """
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
        padding_token: str = DEFAULT_PADDING_TOKEN,
        oov_token: str = DEFAULT_OOV_TOKEN
    ) -> 'Vocabulary':
        """
        Constructs a vocabulary given a collection of `Instances` and some parameters.
        We count all of the vocabulary items in the instances, then pass those counts
        and the other parameters, to :func:`__init__`.  See that method for a description
        of what the other parameters do.

        The `instances` parameter does not get an entry in a typical AllenNLP configuration file,
        but the other parameters do (if you want non-default parameters).
        """
        logger.info('Fitting token dictionary from dataset.')
        padding_token = padding_token if padding_token is not None else DEFAULT_PADDING_TOKEN
        oov_token = oov_token if oov_token is not None else DEFAULT_OOV_TOKEN
        namespace_token_counts: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))
        for instance in Tqdm.tqdm(instances, desc='building vocab'):
            instance.count_vocab_items(namespace_token_counts)
        return cls(counter=namespace_token_counts, min_count=min_count, max_vocab_size=max_vocab_size, non_padded_namespaces=non_padded_namespaces, pretrained_files=pretrained_files, only_include_pretrained_words=only_include_pretrained_words, tokens_to_add=tokens_to_add, min_pretrained_embeddings=min_pretrained_embeddings, padding_token=padding_token, oov_token=oov_token)

    @classmethod
    def from_files(cls, directory: str, padding_token: str = DEFAULT_PADDING_TOKEN, oov_token: str = DEFAULT_OOV_TOKEN) -> 'Vocabulary':
        """
        Loads a `Vocabulary` that was serialized either using `save_to_files` or inside
        a model archive file.

        # Parameters

        directory : `str`
            The directory or archive file containing the serialized vocabulary.
        """
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
        padding_token: str = DEFAULT_PADDING_TOKEN,
        oov_token: str = DEFAULT_OOV_TOKEN,
        min_count: Optional[Dict[str, int]] = None,
        max_vocab_size: Optional[Union[int, Dict[str, int]]] = None,
        non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES,
        pretrained_files: Optional[Dict[str, str]] = None,
        only_include_pretrained_words: bool = False,
        tokens_to_add: Optional[Dict[str, List[str]]] = None,
        min_pretrained_embeddings: Optional[Dict[str, int]] = None
    ) -> 'Vocabulary':
        """
        Extends an already generated vocabulary using a collection of instances.

        The `instances` parameter does not get an entry in a typical AllenNLP configuration file,
        but the other parameters do (if you want non-default parameters).  See `__init__` for a
        description of what the other parameters mean.
        """
        vocab = cls.from_files(directory, padding_token, oov_token)
        logger.info('Fitting token dictionary from dataset.')
        namespace_token_counts: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))
        for instance in Tqdm.tqdm(instances):
            instance.count_vocab_items(namespace_token_counts)
        vocab._extend(counter=namespace_token_counts, min_count=min_count, max_vocab_size=max_vocab_size, non_padded_namespaces=non_padded_namespaces, pretrained_files=pretrained_files, only_include_pretrained_words=only_include_pretrained_words, tokens_to_add=tokens_to_add, min_pretrained_embeddings=min_pretrained_embeddings)
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
        padding_token: str = DEFAULT_PADDING_TOKEN,
        oov_token: str = DEFAULT_OOV_TOKEN
    ) -> 'Vocabulary':
        """
        Construct a vocabulary given a collection of `Instance`'s and some parameters. Then extends
        it with generated vocabularies from pretrained transformers.

        Vocabulary from instances is constructed by passing parameters to :func:`from_instances`,
        and then updated by including merging in vocabularies from
        :func:`from_pretrained_transformer`. See other methods for full descriptions for what the
        other parameters do.

        The `instances` parameters does not get an entry in a typical AllenNLP configuration file,
        other parameters do (if you want non-default parameters).

        # Parameters

        transformers : `Dict[str, str]`
            Dictionary mapping the vocab namespaces (keys) to a transformer model name (value).
            Namespaces not included will be ignored.

        # Examples

        You can use this constructor by modifying the following example within your training
        configuration.

        