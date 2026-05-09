import os
import random
from tempfile import TemporaryDirectory
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)
import numpy as np
import pytest
import torch
from sockeye import constants as C
from sockeye import data_io
from sockeye import utils
from sockeye import vocab
from sockeye.test_utils import tmp_digits_dataset
from sockeye.utils import SockeyeError, get_tokens, seed_rngs

def test_define_buckets(max_seq_len: int, step: int, expected_buckets: List[int]) -> None: ...
def test_define_parallel_buckets(
    max_seq_len_source: int,
    max_seq_len_target: int,
    bucket_width: int,
    bucket_scaling: bool,
    length_ratio: float,
    expected_buckets: List[Tuple[int, int]],
) -> None: ...
def test_get_bucket(buckets: List[int], length: int, expected_bucket: Optional[int]) -> None: ...
def test_tokens2ids(tokens: List[str], vocab: Dict[str, int], expected_ids: List[int]) -> None: ...
def test_strids2ids(tokens: List[str], expected_ids: List[int]) -> None: ...
def test_sequence_reader(
    sequences: List[str],
    use_vocab: bool,
    add_bos: bool,
    add_eos: bool,
) -> None: ...
def test_nonparallel_iter(
    source_iterables: List[List[List[int]]],
    target_iterables: List[List[List[int]]],
) -> None: ...
def test_not_source_token_parallel_iter(
    source_iterables: List[List[List[int]]],
    target_iterables: List[List[List[int]]],
) -> None: ...
def test_not_target_token_parallel_iter(
    source_iterables: List[List[List[int]]],
    target_iterables: List[List[List[int]]],
) -> None: ...
def test_parallel_iter(
    source_iterables: List[List[List[int]]],
    target_iterables: List[List[List[int]]],
    expected: List[Tuple[Tuple[List[List[int]], List[List[int]]], Tuple[List[List[int]], List[List[int]]]]],
) -> None: ...
def test_sample_based_define_bucket_batch_sizes() -> None: ...
def test_word_based_define_bucket_batch_sizes(
    length_ratio: float,
    batch_sentences_multiple_of: int,
    expected_batch_sizes: List[int],
) -> None: ...
def test_max_word_based_define_bucket_batch_sizes(
    length_ratio: float,
    batch_sentences_multiple_of: int,
    expected_batch_sizes: List[int],
) -> None: ...
def _get_random_bucketed_data(
    buckets: List[Tuple[int, int]],
    min_count: int,
    max_count: int,
    bucket_counts: Optional[List[Optional[int]]] = None,
    include_prepended_source_length: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], Optional[List[torch.Tensor]]]: ...
def test_parallel_data_set(include_prepended_source_length: bool) -> None: ...
def test_parallel_data_set_fill_up(include_prepended_source_length: bool) -> None: ...
def test_get_permutations() -> None: ...
def test_parallel_data_set_permute(include_prepended_source_length: bool) -> None: ...
def test_get_batch_indices() -> None: ...
def test_get_parallel_bucket(
    buckets: List[Tuple[int, int]],
    source_length: int,
    target_length: int,
    expected_bucket_index: Optional[int],
    expected_bucket: Optional[Tuple[int, int]],
) -> None: ...
def test_calculate_length_statistics(
    sources: List[List[List[int]]],
    targets: List[List[List[int]]],
    expected_num_sents: int,
    expected_mean: float,
    expected_std: float,
) -> None: ...
def test_non_parallel_calculate_length_statistics(
    sources: List[List[List[int]]],
    targets: List[List[List[int]]],
) -> None: ...
def test_get_training_data_iters(end_of_prepending_tag: Optional[str] = None) -> None: ...
def _data_batches_equal(db1: Any, db2: Any) -> bool: ...
def test_parallel_sample_iter() -> None: ...
def test_sharded_parallel_sample_iter() -> None: ...
def test_sharded_parallel_sample_iter_num_batches() -> None: ...
def test_sharded_and_parallel_iter_same_num_batches() -> None: ...
def test_create_target_and_shifted_label_sequences() -> None: ...

class ParallelDataSet:
    def __init__(
        self,
        source: List[torch.Tensor],
        target: List[torch.Tensor],
        prepended_source_length: Optional[List[torch.Tensor]] = None,
    ) -> None: ...
    def save(self, fname: str) -> None: ...
    @classmethod
    def load(cls, fname: str) -> 'ParallelDataSet': ...
    def fill_up(self, bucket_batch_sizes: List[Any]) -> 'ParallelDataSet': ...
    def get_bucket_counts(self) -> List[int]: ...
    def permute(self, permutations: List[torch.Tensor]) -> 'ParallelDataSet': ...

class ParallelSampleIter:
    def __init__(
        self,
        dataset: ParallelDataSet,
        buckets: List[Tuple[int, int]],
        batch_size: int,
        bucket_batch_sizes: List[Any],
    ) -> None: ...
    def iter_next(self) -> bool: ...
    def next(self) -> Any: ...
    def reset(self) -> None: ...
    def save_state(self, fname: str) -> None: ...
    def load_state(self, fname: str) -> None: ...

class ShardedParallelSampleIter(ParallelSampleIter):
    def __init__(
        self,
        shard_fnames: List[str],
        buckets: List[Tuple[int, int]],
        batch_size: int,
        bucket_batch_sizes: List[Any],
    ) -> None: ...