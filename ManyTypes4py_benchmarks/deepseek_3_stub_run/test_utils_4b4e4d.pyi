import gzip
import os
import re
from tempfile import TemporaryDirectory
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import numpy as np
import pytest
import torch as pt
from sockeye import constants as C

class SockeyeError(Exception):
    ...

class OnlineMeanAndVariance:
    mean: float
    variance: float
    def __init__(self) -> None: ...
    def update(self, sample: float) -> None: ...

def chunks(some_list: List[int], chunk_size: int) -> Iterator[List[int]]: ...

def check_condition(condition: bool, error_message: str) -> None: ...

def parse_version(version_string: str) -> Tuple[str, str, str]: ...

def check_version(version: str) -> None: ...

def metric_value_is_better(new: float, old: float, metric: str) -> bool: ...

def is_gzip_file(fname: str) -> bool: ...

def smart_open(fname: str, mode: str = ...) -> Iterator[str]: ...

def parse_metrics_line(line_num: int, line: str) -> Dict[str, Optional[Union[float, bool]]]: ...

def write_metrics_file(metrics: List[Dict[str, Optional[Union[float, bool]]]], metric_path: str) -> None: ...

def read_metrics_file(metric_path: str) -> List[Dict[str, Optional[Union[float, bool]]]]: ...

def adjust_first_step_masking(target_prefix: pt.Tensor, first_step_mask: pt.Tensor) -> pt.Tensor: ...

def count_seq_len(sample: str, count_type: str, replace_tokens: str) -> int: ...

def compute_isometric_score(
    hypothesis: str,
    hypothesis_score: float,
    source: str,
    metric: str,
    alpha: float
) -> float: ...

def update_dict_with_prefix_kv(
    dest: Dict[str, Any],
    prefix_kv: Dict[str, Any]
) -> None: ...

def fault_tolerant_symlink(src_fname: str, dst_fname: str) -> None: ...

def combine_means(
    sample_means: List[Optional[float]],
    num_sents: List[int]
) -> float: ...

def combine_stds(
    sample_stds: List[Optional[float]],
    sample_means: List[Optional[float]],
    num_sents: List[int]
) -> float: ...

def average_tensors(arrays: List[pt.Tensor]) -> pt.Tensor: ...

def get_tokens(line: str) -> Iterator[str]: ...