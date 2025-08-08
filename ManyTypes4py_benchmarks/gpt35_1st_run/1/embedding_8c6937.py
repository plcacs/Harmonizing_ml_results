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
        self.num_embeddings: int
        self.padding_index: Optional[int]
        self.max_norm: Optional[float]
        self.norm_type: float
        self.scale_grad_by_freq: bool
        self.sparse: bool
        self._vocab_namespace: str
        self._pretrained_file: Optional[str]
        self.output_dim: int
        self.weight: torch.nn.Parameter

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        original_size: Tuple[int, ...] = tokens.size()
        tokens = util.combine_initial_dims(tokens)
        embedded = embedding(tokens, self.weight, padding_idx=self.padding_index, max_norm=self.max_norm, norm_type=self.norm_type, scale_grad_by_freq=self.scale_grad_by_freq, sparse=self.sparse)
        embedded = util.uncombine_initial_dims(embedded, original_size)
        if self._projection:
            projection = self._projection
            for _ in range(embedded.dim() - 2):
                projection = TimeDistributed(projection)
            embedded = projection(embedded)
        return embedded

    def extend_vocab(self, extended_vocab: Vocabulary, vocab_namespace: Optional[str] = None, extension_pretrained_file: Optional[str] = None, model_path: Optional[str] = None) -> None:
        pass

def _read_pretrained_embeddings_file(file_uri: str, embedding_dim: int, vocab: Vocabulary, namespace: str = 'tokens') -> torch.FloatTensor:
    pass

def _read_embeddings_from_text_file(file_uri: str, embedding_dim: int, vocab: Vocabulary, namespace: str = 'tokens') -> torch.FloatTensor:
    pass

def _read_embeddings_from_hdf5(embeddings_filename: str, embedding_dim: int, vocab: Vocabulary, namespace: str = 'tokens') -> torch.FloatTensor:
    pass

def format_embeddings_file_uri(main_file_path_or_url: str, path_inside_archive: Optional[str] = None) -> str:
    pass

def parse_embeddings_file_uri(uri: str) -> EmbeddingsFileURI:
    pass

class EmbeddingsFileURI(NamedTuple):
    path_inside_archive: Optional[str]

class EmbeddingsTextFile(Iterator[str]):
    def __init__(self, file_uri: str, encoding: str = EmbeddingsTextFile.DEFAULT_ENCODING, cache_dir: Optional[str] = None) -> None:
        self.uri: str
        self._encoding: str
        self._cache_dir: Optional[str]
        self._archive_handle: Optional[Union[zipfile.ZipFile, tarfile.TarFile]]
        self.num_tokens: Optional[int]
        self._handle: Union[io.TextIOWrapper, BinaryIO]
        self._iterator: Iterator[str]

    def _open_inside_zip(self, archive_path: str, member_path: Optional[str] = None) -> None:
        pass

    def _open_inside_tar(self, archive_path: str, member_path: Optional[str] = None) -> None:
        pass

    def read(self) -> str:
        pass

    def readline(self) -> str:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> 'EmbeddingsTextFile':
        pass

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    def __iter__(self) -> Iterator[str]:
        pass

    def __next__(self) -> str:
        pass

    def __len__(self) -> int:
        pass

    @staticmethod
    def _get_the_only_file_in_the_archive(members_list: List[str], archive_path: str) -> str:
        pass

    @staticmethod
    def _get_num_tokens_from_first_line(line: str) -> Optional[int]:
        pass
