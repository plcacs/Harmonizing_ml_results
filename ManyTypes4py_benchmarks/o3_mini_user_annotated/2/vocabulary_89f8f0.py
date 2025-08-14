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
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Union, TYPE_CHECKING

from transformers import PreTrainedTokenizer

from allennlp.common import Registrable
from allennlp.common.file_utils import cached_path, FileLock
from allennlp.common.checks import ConfigurationError
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import namespace_match

if TYPE_CHECKING:
    from allennlp.data import instance as adi  # noqa

logger = logging.getLogger(__name__)

DEFAULT_NON_PADDED_NAMESPACES = ("*tags", "*labels")
DEFAULT_PADDING_TOKEN = "@@PADDING@@"
DEFAULT_OOV_TOKEN = "@@UNKNOWN@@"
NAMESPACE_PADDING_FILE = "non_padded_namespaces.txt"
_NEW_LINE_REGEX = re.compile(r"\n|\r\n")


class _NamespaceDependentDefaultDict(defaultdict):
    """
    This is a defaultdict where the default value is dependent on the key that is passed.
    """

    def __init__(
        self,
        non_padded_namespaces: Iterable[str],
        padded_function: Callable[[], Any],
        non_padded_function: Callable[[], Any],
    ) -> None:
        self._non_padded_namespaces: Set[str] = set(non_padded_namespaces)
        self._padded_function: Callable[[], Any] = padded_function
        self._non_padded_function: Callable[[], Any] = non_padded_function
        super().__init__()

    def __missing__(self, key: str) -> Any:
        if any(namespace_match(pattern, key) for pattern in self._non_padded_namespaces):
            value: Any = self._non_padded_function()
        else:
            value = self._padded_function()
        dict.__setitem__(self, key, value)
        return value

    def add_non_padded_namespaces(self, non_padded_namespaces: Set[str]) -> None:
        # add non_padded_namespaces which weren't already present
        self._non_padded_namespaces.update(non_padded_namespaces)


class _TokenToIndexDefaultDict(_NamespaceDependentDefaultDict):
    def __init__(self, non_padded_namespaces: Set[str], padding_token: str, oov_token: str) -> None:
        super().__init__(
            non_padded_namespaces, lambda: {padding_token: 0, oov_token: 1}, lambda: {}
        )


class _IndexToTokenDefaultDict(_NamespaceDependentDefaultDict):
    def __init__(self, non_padded_namespaces: Set[str], padding_token: str, oov_token: str) -> None:
        super().__init__(
            non_padded_namespaces, lambda: {0: padding_token, 1: oov_token}, lambda: {}
        )


def _read_pretrained_tokens(embeddings_file_uri: str) -> List[str]:
    # Moving this import to the top breaks everything (cycling import, I guess)
    from allennlp.modules.token_embedders.embedding import EmbeddingsTextFile

    logger.info("Reading pretrained tokens from: %s", embeddings_file_uri)
    tokens: List[str] = []
    with EmbeddingsTextFile(embeddings_file_uri) as embeddings_file:
        for line_number, line in enumerate(Tqdm.tqdm(embeddings_file), start=1):
            token_end = line.find(" ")
            if token_end >= 0:
                token = line[:token_end]
                tokens.append(token)
            else:
                line_begin = line[:20] + "..." if len(line) > 20 else line
                logger.warning("Skipping line number %d: %s", line_number, line_begin)
    return tokens


class Vocabulary(Registrable):
    """
    A Vocabulary maps strings to integers, allowing for strings to be mapped to an
    out-of-vocabulary token.
    """

    default_implementation = "from_instances"

    def __init__(
        self,
        counter: Optional[Dict[str, Dict[str, int]]] = None,
        min_count: Optional[Dict[str, int]] = None,
        max_vocab_size: Union[int, Dict[str, int], None] = None,
        non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES,
        pretrained_files: Optional[Dict[str, str]] = None,
        only_include_pretrained_words: bool = False,
        tokens_to_add: Optional[Dict[str, List[str]]] = None,
        min_pretrained_embeddings: Optional[Dict[str, int]] = None,
        padding_token: Optional[str] = DEFAULT_PADDING_TOKEN,
        oov_token: Optional[str] = DEFAULT_OOV_TOKEN,
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

        # Made an empty vocabulary, now extend it.
        self._extend(
            counter,
            min_count,
            max_vocab_size,
            non_padded_namespaces,
            pretrained_files,
            only_include_pretrained_words,
            tokens_to_add,
            min_pretrained_embeddings,
        )

    @classmethod
    def from_pretrained_transformer(
        cls, model_name: str, namespace: str = "tokens", oov_token: Optional[str] = None
    ) -> "Vocabulary":
        """
        Initialize a vocabulary from the vocabulary of a pretrained transformer model.
        If `oov_token` is not given, we will try to infer it from the transformer tokenizer.
        """
        from allennlp.common import cached_transformers

        tokenizer: PreTrainedTokenizer = cached_transformers.get_tokenizer(model_name)
        if oov_token is None:
            if hasattr(tokenizer, "_unk_token"):
                oov_token = tokenizer._unk_token  # type: ignore
            elif hasattr(tokenizer, "special_tokens_map"):
                oov_token = tokenizer.special_tokens_map.get("unk_token")
        vocab: Vocabulary = cls(non_padded_namespaces=[namespace], oov_token=oov_token)
        vocab.add_transformer_vocab(tokenizer, namespace)
        return vocab

    @classmethod
    def from_instances(
        cls,
        instances: Iterable["adi.Instance"],
        min_count: Optional[Dict[str, int]] = None,
        max_vocab_size: Union[int, Dict[str, int], None] = None,
        non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES,
        pretrained_files: Optional[Dict[str, str]] = None,
        only_include_pretrained_words: bool = False,
        tokens_to_add: Optional[Dict[str, List[str]]] = None,
        min_pretrained_embeddings: Optional[Dict[str, int]] = None,
        padding_token: Optional[str] = DEFAULT_PADDING_TOKEN,
        oov_token: Optional[str] = DEFAULT_OOV_TOKEN,
    ) -> "Vocabulary":
        """
        Constructs a vocabulary given a collection of `Instances` and some parameters.
        """
        logger.info("Fitting token dictionary from dataset.")
        padding_token = padding_token if padding_token is not None else DEFAULT_PADDING_TOKEN
        oov_token = oov_token if oov_token is not None else DEFAULT_OOV_TOKEN
        namespace_token_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for instance in Tqdm.tqdm(instances, desc="building vocab"):
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
            oov_token=oov_token,
        )

    @classmethod
    def from_files(
        cls,
        directory: Union[str, os.PathLike],
        padding_token: Optional[str] = DEFAULT_PADDING_TOKEN,
        oov_token: Optional[str] = DEFAULT_OOV_TOKEN,
    ) -> "Vocabulary":
        """
        Loads a `Vocabulary` that was serialized either using `save_to_files` or inside
        a model archive file.
        """
        logger.info("Loading token dictionary from %s.", directory)
        padding_token = padding_token if padding_token is not None else DEFAULT_PADDING_TOKEN
        oov_token = oov_token if oov_token is not None else DEFAULT_OOV_TOKEN

        if not os.path.isdir(directory):
            base_directory: str = cached_path(directory, extract_archive=True)
            vocab_subdir: str = os.path.join(base_directory, "vocabulary")
            if os.path.isdir(vocab_subdir):
                directory = vocab_subdir
            elif os.path.isdir(base_directory):
                directory = base_directory
            else:
                raise ConfigurationError(f"{directory} is neither a directory nor an archive")

        with FileLock(os.path.join(directory, ".lock"), read_only_ok=True):
            with codecs.open(
                os.path.join(directory, NAMESPACE_PADDING_FILE), "r", "utf-8"
            ) as namespace_file:
                non_padded_namespaces: List[str] = [namespace_str.strip() for namespace_str in namespace_file]

            vocab: Vocabulary = cls(
                non_padded_namespaces=non_padded_namespaces,
                padding_token=padding_token,
                oov_token=oov_token,
            )

            for namespace_filename in os.listdir(directory):
                if namespace_filename == NAMESPACE_PADDING_FILE:
                    continue
                if namespace_filename.startswith("."):
                    continue
                namespace: str = namespace_filename.replace(".txt", "")
                if any(namespace_match(pattern, namespace) for pattern in non_padded_namespaces):
                    is_padded: bool = False
                else:
                    is_padded = True
                filename: str = os.path.join(directory, namespace_filename)
                vocab.set_from_file(filename, is_padded, namespace=namespace, oov_token=oov_token)

        return vocab

    @classmethod
    def from_files_and_instances(
        cls,
        instances: Iterable["adi.Instance"],
        directory: str,
        padding_token: Optional[str] = DEFAULT_PADDING_TOKEN,
        oov_token: Optional[str] = DEFAULT_OOV_TOKEN,
        min_count: Optional[Dict[str, int]] = None,
        max_vocab_size: Union[int, Dict[str, int], None] = None,
        non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES,
        pretrained_files: Optional[Dict[str, str]] = None,
        only_include_pretrained_words: bool = False,
        tokens_to_add: Optional[Dict[str, List[str]]] = None,
        min_pretrained_embeddings: Optional[Dict[str, int]] = None,
    ) -> "Vocabulary":
        """
        Extends an already generated vocabulary using a collection of instances.
        """
        vocab: Vocabulary = cls.from_files(directory, padding_token, oov_token)
        logger.info("Fitting token dictionary from dataset.")
        namespace_token_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
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
            min_pretrained_embeddings=min_pretrained_embeddings,
        )
        return vocab

    @classmethod
    def from_pretrained_transformer_and_instances(
        cls,
        instances: Iterable["adi.Instance"],
        transformers: Dict[str, str],
        min_count: Optional[Dict[str, int]] = None,
        max_vocab_size: Union[int, Dict[str, int], None] = None,
        non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES,
        pretrained_files: Optional[Dict[str, str]] = None,
        only_include_pretrained_words: bool = False,
        tokens_to_add: Optional[Dict[str, List[str]]] = None,
        min_pretrained_embeddings: Optional[Dict[str, int]] = None,
        padding_token: Optional[str] = DEFAULT_PADDING_TOKEN,
        oov_token: Optional[str] = DEFAULT_OOV_TOKEN,
    ) -> "Vocabulary":
        """
        Construct a vocabulary given a collection of `Instance`'s and some parameters.
        Then extends it with generated vocabularies from pretrained transformers.
        """
        vocab: Vocabulary = cls.from_instances(
            instances=instances,
            min_count=min_count,
            max_vocab_size=max_vocab_size,
            non_padded_namespaces=non_padded_namespaces,
            pretrained_files=pretrained_files,
            only_include_pretrained_words=only_include_pretrained_words,
            tokens_to_add=tokens_to_add,
            min_pretrained_embeddings=min_pretrained_embeddings,
            padding_token=padding_token,
            oov_token=oov_token,
        )

        for namespace, model_name in transformers.items():
            transformer_vocab: Vocabulary = cls.from_pretrained_transformer(
                model_name=model_name, namespace=namespace
            )
            vocab.extend_from_vocab(transformer_vocab)

        return vocab

    @classmethod
    def empty(cls) -> "Vocabulary":
        """
        This method returns a bare vocabulary instantiated with `cls()`.
        """
        return cls()

    def add_transformer_vocab(
        self, tokenizer: PreTrainedTokenizer, namespace: str = "tokens"
    ) -> None:
        """
        Copies tokens from a transformer tokenizer's vocab into the given namespace.
        """
        try:
            vocab_items = tokenizer.get_vocab().items()
        except NotImplementedError:
            vocab_items = (
                (tokenizer.convert_ids_to_tokens(idx), idx) for idx in range(tokenizer.vocab_size)
            )

        for word, idx in vocab_items:
            self._token_to_index[namespace][word] = idx
            self._index_to_token[namespace][idx] = word

        self._non_padded_namespaces.add(namespace)

    def set_from_file(
        self,
        filename: str,
        is_padded: bool = True,
        oov_token: str = DEFAULT_OOV_TOKEN,
        namespace: str = "tokens",
    ) -> None:
        """
        If you already have a vocabulary file for a trained model somewhere,
        you can load that vocabulary file using this method.
        """
        if is_padded:
            self._token_to_index[namespace] = {self._padding_token: 0}
            self._index_to_token[namespace] = {0: self._padding_token}
        else:
            self._token_to_index[namespace] = {}
            self._index_to_token[namespace] = {}
        with codecs.open(filename, "r", "utf-8") as input_file:
            lines: List[str] = _NEW_LINE_REGEX.split(input_file.read())
            if lines and lines[-1] == "":
                lines = lines[:-1]
            for i, line in enumerate(lines):
                index: int = i + 1 if is_padded else i
                token: str = line.replace("@@NEWLINE@@", "\n")
                if token == oov_token:
                    token = self._oov_token
                self._token_to_index[namespace][token] = index
                self._index_to_token[namespace][index] = token
        if is_padded:
            assert self._oov_token in self._token_to_index[namespace], "OOV token not found!"

    def extend_from_instances(self, instances: Iterable["adi.Instance"]) -> None:
        logger.info("Fitting token dictionary from dataset.")
        namespace_token_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for instance in Tqdm.tqdm(instances):
            instance.count_vocab_items(namespace_token_counts)
        self._extend(counter=namespace_token_counts)

    def extend_from_vocab(self, vocab: "Vocabulary") -> None:
        """
        Adds all vocabulary items from all namespaces in the given vocabulary to this vocabulary.
        """
        self._non_padded_namespaces.update(vocab._non_padded_namespaces)
        self._token_to_index.add_non_padded_namespaces(vocab._non_padded_namespaces)
        self._index_to_token.add_non_padded_namespaces(vocab._non_padded_namespaces)
        for namespace in vocab.get_namespaces():
            for token in vocab.get_token_to_index_vocabulary(namespace):
                self.add_token_to_namespace(token, namespace)

    def _extend(
        self,
        counter: Optional[Dict[str, Dict[str, int]]] = None,
        min_count: Optional[Dict[str, int]] = None,
        max_vocab_size: Union[int, Dict[str, int], None] = None,
        non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES,
        pretrained_files: Optional[Dict[str, str]] = None,
        only_include_pretrained_words: bool = False,
        tokens_to_add: Optional[Dict[str, List[str]]] = None,
        min_pretrained_embeddings: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        This method is used for extending an already generated vocabulary.
        """
        if min_count is not None:
            for key in min_count:
                if counter is not None and key not in counter or counter is None:
                    raise ConfigurationError(
                        f"The key '{key}' is present in min_count but not in counter"
                    )

        if not isinstance(max_vocab_size, dict):
            int_max_vocab_size = max_vocab_size
            max_vocab_size = defaultdict(lambda: int_max_vocab_size)  # type: ignore
        min_count = min_count or {}
        pretrained_files = pretrained_files or {}
        min_pretrained_embeddings = min_pretrained_embeddings or {}
        non_padded_namespaces = set(non_padded_namespaces)
        counter = counter or {}
        tokens_to_add = tokens_to_add or {}

        self._retained_counter = counter
        current_namespaces: Set[str] = set(self._token_to_index.keys())
        extension_namespaces: Set[str] = set(counter.keys()) | set(tokens_to_add.keys())

        for namespace in current_namespaces & extension_namespaces:
            original_padded: bool = not any(
                namespace_match(pattern, namespace) for pattern in self._non_padded_namespaces
            )
            extension_padded: bool = not any(
                namespace_match(pattern, namespace) for pattern in non_padded_namespaces
            )
            if original_padded != extension_padded:
                raise ConfigurationError(
                    "Common namespace {} has conflicting ".format(namespace)
                    + "setting of padded = True/False. "
                    + "Hence extension cannot be done."
                )

        self._token_to_index.add_non_padded_namespaces(non_padded_namespaces)
        self._index_to_token.add_non_padded_namespaces(non_padded_namespaces)
        self._non_padded_namespaces.update(non_padded_namespaces)

        for namespace in counter:
            pretrained_set: Optional[Set[Any]] = None
            if namespace in pretrained_files:
                pretrained_list: List[str] = _read_pretrained_tokens(pretrained_files[namespace])
                min_embeddings: int = min_pretrained_embeddings.get(namespace, 0)
                if min_embeddings > 0 or min_embeddings == -1:
                    tokens_old: List[str] = tokens_to_add.get(namespace, [])
                    tokens_new: List[str] = (
                        pretrained_list if min_embeddings == -1 else pretrained_list[:min_embeddings]
                    )
                    tokens_to_add[namespace] = tokens_old + tokens_new
                pretrained_set = set(pretrained_list)
            token_counts: List[Any] = list(counter[namespace].items())
            token_counts.sort(key=lambda x: x[1], reverse=True)
            max_vocab: Optional[int]
            try:
                max_vocab = max_vocab_size[namespace]
            except KeyError:
                max_vocab = None
            if max_vocab:
                token_counts = token_counts[:max_vocab]
            for token, count in token_counts:
                if pretrained_set is not None:
                    if only_include_pretrained_words:
                        if token in pretrained_set and count >= min_count.get(namespace, 1):
                            self.add_token_to_namespace(token, namespace)
                    elif token in pretrained_set or count >= min_count.get(namespace, 1):
                        self.add_token_to_namespace(token, namespace)
                elif count >= min_count.get(namespace, 1):
                    self.add_token_to_namespace(token, namespace)

        for namespace, tokens in tokens_to_add.items():
            for token in tokens:
                self.add_token_to_namespace(token, namespace)

    def __getstate__(self) -> Dict[str, Any]:
        """
        Need to sanitize defaultdict and defaultdict-like objects
        by converting them to vanilla dicts when we pickle the vocabulary.
        """
        state: Dict[str, Any] = copy.copy(self.__dict__)
        state["_token_to_index"] = dict(state["_token_to_index"])
        state["_index_to_token"] = dict(state["_index_to_token"])

        if "_retained_counter" in state:
            state["_retained_counter"] = {
                key: dict(value) for key, value in state["_retained_counter"].items()
            }

        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        When unpickling, reload the plain dicts into our special DefaultDict subclasses.
        """
        self.__dict__ = copy.copy(state)
        self._token_to_index = _TokenToIndexDefaultDict(
            self._non_padded_namespaces, self._padding_token, self._oov_token
        )
        self._token_to_index.update(state["_token_to_index"])
        self._index_to_token = _IndexToTokenDefaultDict(
            self._non_padded_namespaces, self._padding_token, self._oov_token
        )
        self._index_to_token.update(state["_index_to_token"])

    def save_to_files(self, directory: str) -> None:
        """
        Persist this Vocabulary to files so it can be reloaded later.
        """
        os.makedirs(directory, exist_ok=True)
        if os.listdir(directory):
            logger.warning("vocabulary serialization directory %s is not empty", directory)

        with FileLock(os.path.join(directory, ".lock")):
            with codecs.open(
                os.path.join(directory, NAMESPACE_PADDING_FILE), "w", "utf-8"
            ) as namespace_file:
                for namespace_str in self._non_padded_namespaces:
                    print(namespace_str, file=namespace_file)

            for namespace, mapping in self._index_to_token.items():
                with codecs.open(
                    os.path.join(directory, namespace + ".txt"), "w", "utf-8"
                ) as token_file:
                    num_tokens: int = len(mapping)
                    start_index: int = 1 if mapping[0] == self._padding_token else 0
                    for i in range(start_index, num_tokens):
                        print(mapping[i].replace("\n", "@@NEWLINE@@"), file=token_file)

    def is_padded(self, namespace: str) -> bool:
        """
        Returns whether there are padding and OOV tokens added to the given namespace.
        """
        return self._index_to_token[namespace][0] == self._padding_token

    def add_token_to_namespace(self, token: str, namespace: str = "tokens") -> int:
        """
        Adds `token` to the index, if it is not already present.
        """
        if not isinstance(token, str):
            raise ValueError(
                "Vocabulary tokens must be strings, or saving and loading will break.  Got %s (with type %s)"
                % (repr(token), type(token))
            )
        if token not in self._token_to_index[namespace]:
            index: int = len(self._token_to_index[namespace])
            self._token_to_index[namespace][token] = index
            self._index_to_token[namespace][index] = token
            return index
        else:
            return self._token_to_index[namespace][token]

    def add_tokens_to_namespace(self, tokens: List[str], namespace: str = "tokens") -> List[int]:
        """
        Adds multiple `tokens` to the index, if they are not already present.
        """
        return [self.add_token_to_namespace(token, namespace) for token in tokens]

    def get_index_to_token_vocabulary(self, namespace: str = "tokens") -> Dict[int, str]:
        return self._index_to_token[namespace]

    def get_token_to_index_vocabulary(self, namespace: str = "tokens") -> Dict[str, int]:
        return self._token_to_index[namespace]

    def get_token_index(self, token: str, namespace: str = "tokens") -> int:
        try:
            return self._token_to_index[namespace][token]
        except KeyError:
            try:
                return self._token_to_index[namespace][self._oov_token]
            except KeyError:
                logger.error("Namespace: %s", namespace)
                logger.error("Token: %s", token)
                raise KeyError(
                    f"'{token}' not found in vocab namespace '{namespace}', and namespace "
                    f"does not contain the default OOV token ('{self._oov_token}')"
                )

    def get_token_from_index(self, index: int, namespace: str = "tokens") -> str:
        return self._index_to_token[namespace][index]

    def get_vocab_size(self, namespace: str = "tokens") -> int:
        return len(self._token_to_index[namespace])

    def get_namespaces(self) -> Set[str]:
        return set(self._index_to_token.keys())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __str__(self) -> str:
        base_string: str = "Vocabulary with namespaces:\n"
        non_padded_namespaces: str = f"\tNon Padded Namespaces: {self._non_padded_namespaces}\n"
        namespaces: List[str] = [
            f"\tNamespace: {name}, Size: {self.get_vocab_size(name)} \n"
            for name in self._index_to_token
        ]
        return " ".join([base_string, non_padded_namespaces] + namespaces)

    def __repr__(self) -> str:
        base_string: str = "Vocabulary with namespaces: "
        namespaces: List[str] = [
            f"{name}, Size: {self.get_vocab_size(name)} ||" for name in self._index_to_token
        ]
        non_padded_namespaces: str = f"Non Padded Namespaces: {self._non_padded_namespaces}"
        return " ".join([base_string] + namespaces + [non_padded_namespaces])

    def print_statistics(self) -> None:
        if self._retained_counter:
            logger.info(
                "Printed vocabulary statistics are only for the part of the vocabulary generated "
                "from instances. If vocabulary is constructed by extending saved vocabulary with "
                "dataset instances, the directly loaded portion won't be considered here."
            )
            print("\n\n----Vocabulary Statistics----\n")
            for namespace in self._retained_counter:
                tokens_with_counts: List[Any] = list(self._retained_counter[namespace].items())
                tokens_with_counts.sort(key=lambda x: x[1], reverse=True)
                print(f"\nTop 10 most frequent tokens in namespace '{namespace}':")
                for token, freq in tokens_with_counts[:10]:
                    print(f"\tToken: {token}\t\tFrequency: {freq}")
                tokens_with_counts.sort(key=lambda x: len(x[0]), reverse=True)
                print(f"\nTop 10 longest tokens in namespace '{namespace}':")
                for token, freq in tokens_with_counts[:10]:
                    print(f"\tToken: {token}\t\tlength: {len(token)}\tFrequency: {freq}")
                print(f"\nTop 10 shortest tokens in namespace '{namespace}':")
                for token, freq in reversed(tokens_with_counts[-10:]):
                    print(f"\tToken: {token}\t\tlength: {len(token)}\tFrequency: {freq}")
        else:
            logger.info(
                "Vocabulary statistics cannot be printed since "
                "dataset instances were not used for its construction."
            )


Vocabulary.register("from_pretrained_transformer", constructor="from_pretrained_transformer")(
    Vocabulary
)
Vocabulary.register(
    "from_pretrained_transformer_and_instances",
    constructor="from_pretrained_transformer_and_instances",
)(Vocabulary)
Vocabulary.register("from_instances", constructor="from_instances")(Vocabulary)
Vocabulary.register("from_files", constructor="from_files")(Vocabulary)
Vocabulary.register("extend", constructor="from_files_and_instances")(Vocabulary)
Vocabulary.register("empty", constructor="empty")(Vocabulary)
