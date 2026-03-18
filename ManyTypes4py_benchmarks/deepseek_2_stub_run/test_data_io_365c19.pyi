```python
from typing import Any, Iterable, List, Optional, Tuple, Union
import numpy as np
import torch

class SequenceReader:
    def __init__(
        self,
        path: str,
        vocabulary: Optional[Any] = ...,
        add_bos: bool = ...,
        add_eos: bool = ...,
    ) -> None: ...
    def __iter__(self) -> Any: ...
    def __next__(self) -> Any: ...

class ParallelDataSet:
    source: List[torch.Tensor]
    target: List[torch.Tensor]
    prepended_source_length: Optional[List[torch.Tensor]]
    def __init__(
        self,
        source: List[torch.Tensor],
        target: List[torch.Tensor],
        prepended_source_length: Optional[List[torch.Tensor]] = ...,
    ) -> None: ...
    def save(self, fname: str, use_legacy_format: bool = ...) -> None: ...
    @staticmethod
    def load(fname: str) -> 'ParallelDataSet': ...
    def fill_up(self, bucket_batch_sizes: List[Any]) -> 'ParallelDataSet': ...
    def get_bucket_counts(self) -> List[int]: ...
    def permute(self, permutations: List[torch.Tensor]) -> 'ParallelDataSet': ...
    def __len__(self) -> int: ...

class Batch:
    source: torch.Tensor
    source_length: torch.Tensor
    target: torch.Tensor
    target_length: torch.Tensor
    labels: dict
    samples: int
    tokens: int

class ParallelSampleIter:
    def __init__(
        self,
        dataset: ParallelDataSet,
        buckets: List[Tuple[int, int]],
        batch_size: int,
        bucket_batch_sizes: List[Any],
    ) -> None: ...
    def reset(self) -> None: ...
    def iter_next(self) -> bool: ...
    def next(self) -> Batch: ...
    def save_state(self, fname: str) -> None: ...
    def load_state(self, fname: str) -> None: ...

class ShardedParallelSampleIter:
    def __init__(
        self,
        shard_fnames: List[str],
        buckets: List[Tuple[int, int]],
        batch_size: int,
        bucket_batch_sizes: List[Any],
    ) -> None: ...
    def reset(self) -> None: ...
    def iter_next(self) -> bool: ...
    def next(self) -> Batch: ...
    def save_state(self, fname: str) -> None: ...
    def load_state(self, fname: str) -> None: ...

class DataConfig:
    data_statistics: Any

def define_buckets(max_seq_len: int, step: int = ...) -> List[int]: ...

def define_parallel_buckets(
    max_seq_len_source: int,
    max_seq_len_target: int,
    bucket_width: int = ...,
    bucket_scaling: bool = ...,
    length_ratio: float = ...,
) -> List[Tuple[int, int]]: ...

def get_bucket(length: int, buckets: List[int]) -> Optional[int]: ...

def tokens2ids(tokens: List[str], vocab: dict) -> List[int]: ...

def strids2ids(tokens: List[str]) -> List[int]: ...

def parallel_iter(
    source_iterables: List[Iterable[Optional[List[int]]]],
    target_iterables: List[Iterable[Optional[List[int]]]],
) -> Iterable[Tuple[List[List[int]], List[List[int]]]]: ...

def define_bucket_batch_sizes(
    buckets: List[Tuple[int, int]],
    batch_size: int,
    batch_type: str,
    data_target_average_len: List[Optional[int]],
    batch_sentences_multiple_of: int = ...,
) -> List[Any]: ...

def get_permutations(bucket_counts: List[int]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]: ...

def get_batch_indices(
    dataset: ParallelDataSet,
    bucket_batch_sizes: List[Any],
) -> List[Tuple[int, int]]: ...

def get_parallel_bucket(
    buckets: List[Tuple[int, int]],
    source_length: int,
    target_length: int,
) -> Tuple[Optional[int], Optional[Tuple[int, int]]]: ...

def calculate_length_statistics(
    sources: List[List[List[int]]],
    targets: List[List[List[int]]],
    max_seq_len_source: int,
    max_seq_len_target: int,
) -> Any: ...

def get_training_data_iters(
    sources: List[str],
    targets: List[str],
    validation_sources: List[str],
    validation_targets: List[str],
    source_vocabs: List[Any],
    target_vocabs: List[Any],
    source_vocab_paths: List[Optional[str]],
    target_vocab_paths: List[Optional[str]],
    shared_vocab: bool,
    batch_size: int,
    batch_type: str,
    max_seq_len_source: int,
    max_seq_len_target: int,
    bucketing: bool,
    bucket_width: int,
    end_of_prepending_tag: Optional[str] = ...,
) -> Tuple[ParallelSampleIter, ParallelSampleIter, DataConfig, Any]: ...

def create_target_and_shifted_label_sequences(
    target_and_label: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]: ...
```