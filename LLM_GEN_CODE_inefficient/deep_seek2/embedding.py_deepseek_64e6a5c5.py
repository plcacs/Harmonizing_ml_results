import io
import itertools
import logging
import re
import tarfile
import warnings
import zipfile
from typing import Any, cast, Iterator, NamedTuple, Optional, Sequence, Tuple, BinaryIO, List, Dict, Union

import numpy
import torch

from torch.nn.functional import embedding

from allennlp.common import Tqdm
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path, get_file_extension, is_url_or_existing_file
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.nn import util

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py

logger = logging.getLogger(__name__)


@TokenEmbedder.register("embedding")
class Embedding(TokenEmbedder):
    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: Optional[int] = None,
        projection_dim: Optional[int] = None,
        weight: Optional[torch.FloatTensor] = None,
        padding_index: Optional[int] = None,
        trainable: bool = True,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        vocab_namespace: str = "tokens",
        pretrained_file: Optional[str] = None,
        vocab: Optional[Vocabulary] = None,
    ) -> None:
        super().__init__()

        if num_embeddings is None and vocab is None:
            raise ConfigurationError(
                "Embedding must be constructed with either num_embeddings or a vocabulary."
            )

        _vocab_namespace: Optional[str] = vocab_namespace
        if num_embeddings is None:
            num_embeddings = vocab.get_vocab_size(_vocab_namespace)  # type: ignore
        else:
            _vocab_namespace = None  # type: ignore

        self.num_embeddings = num_embeddings
        self.padding_index = padding_index
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self._vocab_namespace = _vocab_namespace
        self._pretrained_file = pretrained_file

        self.output_dim = projection_dim or embedding_dim

        if weight is not None and pretrained_file:
            raise ConfigurationError(
                "Embedding was constructed with both a weight and a pretrained file."
            )

        elif pretrained_file is not None:
            if vocab is None:
                raise ConfigurationError(
                    "To construct an Embedding from a pretrained file, you must also pass a vocabulary."
                )

            weight = _read_pretrained_embeddings_file(
                pretrained_file, embedding_dim, vocab, vocab_namespace
            )
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)

        elif weight is not None:
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)

        else:
            weight = torch.FloatTensor(num_embeddings, embedding_dim)
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)
            torch.nn.init.xavier_uniform_(self.weight)

        if self.weight.size() != (num_embeddings, embedding_dim):
            raise ConfigurationError(
                "A weight matrix was passed with contradictory embedding shapes."
            )

        if self.padding_index is not None:
            self.weight.data[self.padding_index].fill_(0)

        if projection_dim:
            self._projection = torch.nn.Linear(embedding_dim, projection_dim)
        else:
            self._projection = None

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        original_size = tokens.size()
        tokens = util.combine_initial_dims(tokens)

        embedded = embedding(
            tokens,
            self.weight,
            padding_idx=self.padding_index,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

        embedded = util.uncombine_initial_dims(embedded, original_size)

        if self._projection:
            projection = self._projection
            for _ in range(embedded.dim() - 2):
                projection = TimeDistributed(projection)
            embedded = projection(embedded)
        return embedded

    def extend_vocab(
        self,
        extended_vocab: Vocabulary,
        vocab_namespace: Optional[str] = None,
        extension_pretrained_file: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> None:
        vocab_namespace = vocab_namespace or self._vocab_namespace
        if not vocab_namespace:
            logger.info(
                "Loading a model trained before embedding extension was implemented; "
                "pass an explicit vocab namespace if you want to extend the vocabulary."
            )
            return

        extended_num_embeddings = extended_vocab.get_vocab_size(vocab_namespace)
        if extended_num_embeddings == self.num_embeddings:
            return

        if extended_num_embeddings < self.num_embeddings:
            raise ConfigurationError(
                f"Size of namespace, {vocab_namespace} for extended_vocab is smaller than "
                f"embedding. You likely passed incorrect vocab or namespace for extension."
            )

        if extension_pretrained_file and is_url_or_existing_file(extension_pretrained_file):
            pass
        elif extension_pretrained_file:
            raise ConfigurationError(
                f"You passed pretrained embedding file {extension_pretrained_file} "
                f"for model_path {model_path} but it's not available."
            )
        elif is_url_or_existing_file(self._pretrained_file):
            extension_pretrained_file = self._pretrained_file
        elif self._pretrained_file is not None:
            logger.warning(
                f"Embedding at model_path, {model_path} cannot locate the pretrained_file. "
                f"Originally pretrained_file was at '{self._pretrained_file}'."
            )
        else:
            logger.info(
                "If you are fine-tuning and want to use a pretrained_file for "
                "embedding extension, please pass the mapping by --embedding-sources argument."
            )

        embedding_dim = self.weight.data.shape[-1]
        if not extension_pretrained_file:
            extra_num_embeddings = extended_num_embeddings - self.num_embeddings
            extra_weight = torch.FloatTensor(extra_num_embeddings, embedding_dim)
            torch.nn.init.xavier_uniform_(extra_weight)
        else:
            whole_weight = _read_pretrained_embeddings_file(
                extension_pretrained_file, embedding_dim, extended_vocab, vocab_namespace
            )
            extra_weight = whole_weight[self.num_embeddings :, :]

        device = self.weight.data.device
        extended_weight = torch.cat([self.weight.data, extra_weight.to(device)], dim=0)
        self.weight = torch.nn.Parameter(extended_weight, requires_grad=self.weight.requires_grad)
        self.num_embeddings = extended_num_embeddings


def _read_pretrained_embeddings_file(
    file_uri: str, embedding_dim: int, vocab: Vocabulary, namespace: str = "tokens"
) -> torch.FloatTensor:
    file_ext = get_file_extension(file_uri)
    if file_ext in [".h5", ".hdf5"]:
        return _read_embeddings_from_hdf5(file_uri, embedding_dim, vocab, namespace)

    return _read_embeddings_from_text_file(file_uri, embedding_dim, vocab, namespace)


def _read_embeddings_from_text_file(
    file_uri: str, embedding_dim: int, vocab: Vocabulary, namespace: str = "tokens"
) -> torch.FloatTensor:
    tokens_to_keep = set(vocab.get_index_to_token_vocabulary(namespace).values())
    vocab_size = vocab.get_vocab_size(namespace)
    embeddings: Dict[str, numpy.ndarray] = {}

    logger.info("Reading pretrained embeddings from file")

    with EmbeddingsTextFile(file_uri) as embeddings_file:
        for line in Tqdm.tqdm(embeddings_file):
            token = line.split(" ", 1)[0]
            if token in tokens_to_keep:
                fields = line.rstrip().split(" ")
                if len(fields) - 1 != embedding_dim:
                    logger.warning(
                        "Found line with wrong number of dimensions (expected: %d; actual: %d): %s",
                        embedding_dim,
                        len(fields) - 1,
                        line,
                    )
                    continue

                vector = numpy.asarray(fields[1:], dtype="float32")
                embeddings[token] = vector

    if not embeddings:
        raise ConfigurationError(
            "No embeddings of correct dimension found; you probably "
            "misspecified your embedding_dim parameter, or didn't "
            "pre-populate your Vocabulary"
        )

    all_embeddings = numpy.asarray(list(embeddings.values()))
    embeddings_mean = float(numpy.mean(all_embeddings))
    embeddings_std = float(numpy.std(all_embeddings))
    logger.info("Initializing pre-trained embedding layer")
    embedding_matrix = torch.FloatTensor(vocab_size, embedding_dim).normal_(
        embeddings_mean, embeddings_std
    )
    num_tokens_found = 0
    index_to_token = vocab.get_index_to_token_vocabulary(namespace)
    for i in range(vocab_size):
        token = index_to_token[i]

        if token in embeddings:
            embedding_matrix[i] = torch.FloatTensor(embeddings[token])
            num_tokens_found += 1
        else:
            logger.debug(
                "Token %s was not found in the embedding file. Initialising randomly.", token
            )

    logger.info(
        "Pretrained embeddings were found for %d out of %d tokens", num_tokens_found, vocab_size
    )

    return embedding_matrix


def _read_embeddings_from_hdf5(
    embeddings_filename: str, embedding_dim: int, vocab: Vocabulary, namespace: str = "tokens"
) -> torch.FloatTensor:
    with h5py.File(embeddings_filename, "r") as fin:
        embeddings = fin["embedding"][...]

    if list(embeddings.shape) != [vocab.get_vocab_size(namespace), embedding_dim]:
        raise ConfigurationError(
            "Read shape {0} embeddings from the file, but expected {1}".format(
                list(embeddings.shape), [vocab.get_vocab_size(namespace), embedding_dim]
            )
        )

    return torch.FloatTensor(embeddings)


def format_embeddings_file_uri(
    main_file_path_or_url: str, path_inside_archive: Optional[str] = None
) -> str:
    if path_inside_archive:
        return "({})#{}".format(main_file_path_or_url, path_inside_archive)
    return main_file_path_or_url


class EmbeddingsFileURI(NamedTuple):
    main_file_uri: str
    path_inside_archive: Optional[str] = None


def parse_embeddings_file_uri(uri: str) -> EmbeddingsFileURI:
    match = re.fullmatch(r"\((.*)\)#(.*)", uri)
    if match:
        fields = cast(Tuple[str, str], match.groups())
        return EmbeddingsFileURI(*fields)
    else:
        return EmbeddingsFileURI(uri, None)


class EmbeddingsTextFile(Iterator[str]):
    DEFAULT_ENCODING = "utf-8"

    def __init__(
        self, file_uri: str, encoding: str = DEFAULT_ENCODING, cache_dir: Optional[str] = None
    ) -> None:
        self.uri = file_uri
        self._encoding = encoding
        self._cache_dir = cache_dir
        self._archive_handle: Any = None

        main_file_uri, path_inside_archive = parse_embeddings_file_uri(file_uri)
        main_file_local_path = cached_path(main_file_uri, cache_dir=cache_dir)

        if zipfile.is_zipfile(main_file_local_path):
            self._open_inside_zip(main_file_uri, path_inside_archive)

        elif tarfile.is_tarfile(main_file_local_path):
            self._open_inside_tar(main_file_uri, path_inside_archive)

        else:
            if path_inside_archive:
                raise ValueError("Unsupported archive format: %s" + main_file_uri)

            extension = get_file_extension(main_file_uri)

            package = None
            if extension in [".txt", ".vec"]:
                package = io
            elif extension == ".gz":
                import gzip

                package = gzip
            elif extension == ".bz2":
                import bz2

                package = bz2
            elif extension == ".xz":
                import lzma

                package = lzma

            if package is None:
                logger.warning(
                    'The embeddings file has an unknown file extension "%s". '
                    "We will assume the file is an (uncompressed) text file",
                    extension,
                )
                package = io

            self._handle = package.open(  # type: ignore
                main_file_local_path, "rt", encoding=encoding
            )

        first_line = next(self._handle)
        self.num_tokens = EmbeddingsTextFile._get_num_tokens_from_first_line(first_line)
        if self.num_tokens:
            self._iterator = self._handle
        else:
            self._iterator = itertools.chain([first_line], self._handle)

    def _open_inside_zip(self, archive_path: str, member_path: Optional[str] = None) -> None:
        cached_archive_path = cached_path(archive_path, cache_dir=self._cache_dir)
        archive = zipfile.ZipFile(cached_archive_path, "r")
        if member_path is None:
            members_list = archive.namelist()
            member_path = self._get_the_only_file_in_the_archive(members_list, archive_path)
        member_path = cast(str, member_path)
        member_file = cast(BinaryIO, archive.open(member_path, "r"))
        self._handle = io.TextIOWrapper(member_file, encoding=self._encoding)
        self._archive_handle = archive

    def _open_inside_tar(self, archive_path: str, member_path: Optional[str] = None) -> None:
        cached_archive_path = cached_path(archive_path, cache_dir=self._cache_dir)
        archive = tarfile.open(cached_archive_path, "r")
        if member_path is None:
            members_list = archive.getnames()
            member_path = self._get_the_only_file_in_the_archive(members_list, archive_path)
        member_path = cast(str, member_path)
        member = archive.getmember(member_path)
        member_file = cast(BinaryIO, archive.extractfile(member))
        self._handle = io.TextIOWrapper(member_file, encoding=self._encoding)
        self._archive_handle = archive

    def read(self) -> str:
        return "".join(self._iterator)

    def readline(self) -> str:
        return next(self._iterator)

    def close(self) -> None:
        self._handle.close()
        if self._archive_handle:
            self._archive_handle.close()

    def __enter__(self) -> "EmbeddingsTextFile":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __iter__(self) -> "EmbeddingsTextFile":
        return self

    def __next__(self) -> str:
        return next(self._iterator)

    def __len__(self) -> Optional[int]:
        if self.num_tokens:
            return self.num_tokens
        raise AttributeError(
            "an object of type EmbeddingsTextFile implements `__len__` only if the underlying "
            "text file declares the number of tokens (i.e. the number of lines following)"
            "in the first line. That is not the case of this particular instance."
        )

    @staticmethod
    def _get_the_only_file_in_the_archive(members_list: Sequence[str], archive_path: str) -> str:
        if len(members_list) > 1:
            raise ValueError(
                "The archive %s contains multiple files, so you must select "
                "one of the files inside providing a uri of the type: %s."
                % (
                    archive_path,
                    format_embeddings_file_uri("path_or_url_to_archive", "path_inside_archive"),
                )
            )
        return members_list[0]

    @staticmethod
    def _get_num_tokens_from_first_line(line: str) -> Optional[int]:
        fields = line.split(" ")
        if 1 <= len(fields) <= 2:
            try:
                int_fields = [int(x) for x in fields]
            except ValueError:
                return None
            else:
                num_tokens = max(int_fields)
                logger.info(
                    "Recognized a header line in the embedding file with number of tokens: %d",
                    num_tokens,
                )
                return num_tokens
        return None
