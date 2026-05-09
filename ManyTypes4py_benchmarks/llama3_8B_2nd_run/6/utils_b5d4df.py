import argparse
import binascii
import gzip
import itertools
import logging
import math
import multiprocessing
import os
import pprint
import random
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from itertools import starmap
from typing import Any, List, Iterator, Iterable, Tuple, Dict, Optional, Union, TypeVar

class SockeyeError(Exception):
    pass

def check_version(version: str) -> None:
    ...

def load_version(fname: str) -> str:
    ...

def parse_version(version_string: str) -> Tuple[str, str, str]:
    ...

def log_basic_info(args: argparse.Namespace) -> None:
    ...

def seed_rngs(seed: int) -> None:
    ...

def check_condition(condition: bool, error_message: str) -> None:
    ...

class OnlineMeanAndVariance:
    ...

def chunks(some_list: List[Any], n: int) -> Iterator[List[Any]]:
    ...

def get_tokens(line: str) -> Iterator[str]:
    ...

def is_gzip_file(filename: str) -> bool:
    ...

def smart_open(filename: str, mode: str = 'rt', ftype: str = 'auto', errors: str = 'replace') -> Any:
    ...

def combine_means(means: List[float], num_sents: List[int]) -> float:
    ...

def combine_stds(stds: List[float], means: List[float], num_sents: List[int]) -> float:
    ...

def average_tensors(tensors: List[Any]) -> Any:
    ...

def gen_prefix_masking(prefix: Any, vocab_size: int, dtype: Any) -> Tuple[Any, int]:
    ...

def shift_prefix_factors(prefix_factors: Any) -> Any:
    ...

def adjust_first_step_masking(target_prefix: Any, first_step_mask: Any) -> Any:
    ...

def parse_metrics_line(line_number: int, line: str) -> Dict[str, Any]:
    ...

def read_metrics_file(path: str) -> List[Dict[str, Any]]:
    ...

def write_metrics_file(metrics: List[Dict[str, Any]], path: str) -> None:
    ...

def get_validation_metric_points(model_path: str, metric: str) -> List[Tuple[float, int]]:
    ...

def grouper(iterable: Iterable[Any], size: int) -> Iterator[List[Any]]:
    ...

def metric_value_is_better(new: float, old: float, metric: str) -> bool:
    ...

_DTYPE_TO_STRING: Dict[Any, str] = {...}
_STRING_TO_TORCH_DTYPE: Dict[str, Any] = {...}
_STRING_TO_NUMPY_DTYPE: Dict[str, Any] = {...}

def dtype_to_str(dtype: Any) -> str:
    ...

def get_torch_dtype(dtype: str) -> Any:
    ...

def get_numpy_dtype(dtype: str) -> Any:
    ...

def log_parameters(model: Any) -> None:
    ...

@contextmanager
def no_context() -> None:
    ...

class SingleProcessPool:
    ...

def create_pool(max_processes: int) -> Any:
    ...

def update_dict(dest: Dict[str, Any], source: Dict[str, Any]) -> None:
    ...

def update_dict_with_prefix_kv(dest: Dict[str, Any], prefix_kv: Dict[str, Any]) -> None:
    ...

def is_distributed() -> bool:
    ...

def is_primary_worker() -> bool:
    ...

def get_local_rank() -> int:
    ...

T = TypeVar('T')

def broadcast_object(obj: T, src: int = 0) -> T:
    ...

def all_gather_object(obj: T) -> List[T]:
    ...

_using_deepspeed = False

def init_deepspeed() -> None:
    ...

def using_deepspeed() -> bool:
    ...

_faiss_checked = False

def check_import_faiss() -> None:
    ...

def count_seq_len(sample: str, count_type: str = 'char', replace_tokens: Optional[List[str]] = None) -> int:
    ...

def compute_isometric_score(hypothesis: str, hypothesis_score: float, source: str, isometric_metric: str = 'isometric-ratio', isometric_alpha: float = 0.5) -> float:
    ...

def init_device(args: argparse.Namespace) -> Any:
    ...

def fault_tolerant_symlink(src: str, dst: str, max_retries: int = 6) -> None:
    ...
