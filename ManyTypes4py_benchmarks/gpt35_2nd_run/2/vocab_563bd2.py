import argparse
import json
import logging
import os
from collections import Counter
from itertools import chain, islice
from typing import Dict, Iterable, List, Optional, Tuple, Callable
from sockeye.log import setup_main_logger
from . import constants as C
from . import utils

logger: logging.Logger = logging.getLogger(__name__)
Vocab: Dict[str, int] = Dict[str, int]
InverseVocab: Dict[int, str] = Dict[int, str]

def count_tokens_for_path(path: str) -> Counter:
    ...

def build_from_paths(paths: List[str], num_words: Optional[int] = None, min_count: int = 1, pad_to_multiple_of: Optional[int] = None, mapper: Callable) -> Vocab:
    ...

def build_vocab(data: Iterable[str], num_words: Optional[int] = None, min_count: int = 1, pad_to_multiple_of: Optional[int] = None) -> Vocab:
    ...

def build_pruned_vocab(raw_vocab: Counter, num_words: Optional[int] = None, min_count: int = 1, pad_to_multiple_of: Optional[int] = None) -> Vocab:
    ...

def count_tokens(data: Iterable[str]) -> Counter:
    ...

def vocab_to_json(vocab: Vocab, path: str) -> None:
    ...

def is_valid_vocab(vocab: Vocab) -> bool:
    ...

def vocab_from_json(path: str, encoding: str = C.VOCAB_ENCODING) -> Vocab:
    ...

def save_source_vocabs(source_vocabs: List[Vocab], folder: str) -> None:
    ...

def save_target_vocabs(target_vocabs: List[Vocab], folder: str) -> None:
    ...

def _get_sorted_source_vocab_fnames(folder: str) -> List[str]:
    ...

def _get_sorted_target_vocab_fnames(folder: str) -> List[str]:
    ...

def load_source_vocabs(folder: str) -> List[Vocab]:
    ...

def load_target_vocabs(folder: str) -> List[Vocab]:
    ...

def load_or_create_vocab(data: Tuple[str], vocab_path: Optional[str], num_words: int, word_min_count: int, pad_to_multiple_of: Optional[int] = None, mapper: Callable) -> Vocab:
    ...

def load_or_create_vocabs(shard_source_paths: List[List[str]], shard_target_paths: List[List[str]], source_vocab_paths: List[str], source_factor_vocab_same_as_source: List[bool], target_vocab_paths: List[str], target_factor_vocab_same_as_target: List[bool], shared_vocab: bool, num_words_source: int, word_min_count_source: int, num_words_target: int, word_min_count_target: int, pad_to_multiple_of: Optional[int] = None, mapper: Callable) -> Tuple[List[Vocab], List[Vocab]]:
    ...

def reverse_vocab(vocab: Vocab) -> Dict[int, str]:
    ...

def get_ordered_tokens_from_vocab(vocab: Vocab) -> List[str]:
    ...

def are_identical(*vocabs: Vocab) -> bool:
    ...

def main() -> None:
    ...

def prepare_vocab(args: argparse.Namespace) -> None:
    ...

if __name__ == '__main__':
    main()
