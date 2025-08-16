import io
import itertools
import logging
import re
import tarfile
import warnings
import zipfile
from typing import Any, cast, Iterator, NamedTuple, Optional, Sequence, Tuple, BinaryIO
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
import h5py

logger: logging.Logger = logging.getLogger(__name__)

@TokenEmbedder.register('embedding')
class Embedding(TokenEmbedder):
    def __init__(self, embedding_dim: int, num_embeddings: Optional[int] = None, projection_dim: Optional[int] = None, weight: Optional[torch.FloatTensor] = None, padding_index: Optional[int] = None, trainable: bool = True, max_norm: Optional[float] = None, norm_type: float = 2.0, scale_grad_by_freq: bool = False, sparse: bool = False, vocab_namespace: Optional[str] = 'tokens', pretrained_file: Optional[str] = None, vocab: Optional[Vocabulary] = None) -> None:
        ...

    def get_output_dim(self) -> int:
        ...

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        ...

    def extend_vocab(self, extended_vocab: Vocabulary, vocab_namespace: Optional[str] = None, extension_pretrained_file: Optional[str] = None, model_path: Optional[str] = None) -> None:
        ...

def _read_pretrained_embeddings_file(file_uri: str, embedding_dim: int, vocab: Vocabulary, namespace: str = 'tokens') -> torch.FloatTensor:
    ...

def _read_embeddings_from_text_file(file_uri: str, embedding_dim: int, vocab: Vocabulary, namespace: str = 'tokens') -> torch.FloatTensor:
    ...

def _read_embeddings_from_hdf5(embeddings_filename: str, embedding_dim: int, vocab: Vocabulary, namespace: str = 'tokens') -> torch.FloatTensor:
    ...

def format_embeddings_file_uri(main_file_path_or_url: str, path_inside_archive: Optional[str] = None) -> str:
    ...

def parse_embeddings_file_uri(uri: str) -> EmbeddingsFileURI:
    ...

class EmbeddingsFileURI(NamedTuple):
    path_inside_archive: Optional[str] = None

class EmbeddingsTextFile(Iterator[str]):
    def __init__(self, file_uri: str, encoding: str = EmbeddingsTextFile.DEFAULT_ENCODING, cache_dir: Optional[str] = None) -> None:
        ...

    def _open_inside_zip(self, archive_path: str, member_path: Optional[str] = None) -> None:
        ...

    def _open_inside_tar(self, archive_path: str, member_path: Optional[str] = None) -> None:
        ...

    def read(self) -> str:
        ...

    def readline(self) -> str:
        ...

    def close(self) -> None:
        ...

    def __enter__(self) -> 'EmbeddingsTextFile':
        ...

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        ...

    def __iter__(self) -> 'EmbeddingsTextFile':
        ...

    def __next__(self) -> str:
        ...

    def __len__(self) -> int:
        ...

    @staticmethod
    def _get_the_only_file_in_the_archive(members_list: Sequence[str], archive_path: str) -> str:
        ...

    @staticmethod
    def _get_num_tokens_from_first_line(line: str) -> Optional[int]:
        ...
