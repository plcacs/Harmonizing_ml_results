import os
import random
from tempfile import TemporaryDirectory
from typing import Optional, List, Tuple
import numpy as np
import pytest
import torch
from sockeye import constants as C
from sockeye import data_io
from sockeye import vocab
from sockeye.test_utils import tmp_digits_dataset
from sockeye.utils import SockeyeError, get_tokens, seed_rngs

seed_rngs(12)

define_bucket_tests: List[Tuple[int, int, List[int]]] = [(50, 10, [10, 20, 30, 40, 50]), (50, 20, [20, 40, 50]), (50, 50, [50]), (5, 10, [5]), (11, 5, [5, 10, 11]), (19, 10, [10, 19])]

@pytest.mark.parametrize('max_seq_len, step, expected_buckets', define_bucket_tests)
def test_define_buckets(max_seq_len: int, step: int, expected_buckets: List[int]) -> None:
    buckets = data_io.define_buckets(max_seq_len, step=step)
    assert buckets == expected_buckets

define_parallel_bucket_tests: List[Tuple[int, int, int, bool, float, List[Tuple[int, int]]]] = [(50, 50, 10, True, 1.0, [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)]), (50, 50, 10, True, 0.5, [(10, 5), (20, 10), (30, 15), (40, 20), (50, 25), (50, 30), (50, 35), (50, 40), (50, 45), (50, 50)]), (10, 10, 10, True, 0.1, [(10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10)]), (10, 5, 10, True, 0.01, [(10, 2), (10, 3), (10, 4), (10, 5)]), (50, 50, 10, True, 2.0, [(5, 10), (10, 20), (15, 30), (20, 40), (25, 50), (30, 50), (35, 50), (40, 50), (45, 50), (50, 50)]), (5, 10, 10, True, 10.0, [(2, 10), (3, 10), (4, 10), (5, 10)]), (5, 10, 10, True, 11.0, [(2, 10), (3, 10), (4, 10), (5, 10)]), (50, 50, 50, True, 0.5, [(50, 25), (50, 50)]), (50, 50, 50, True, 1.5, [(33, 50), (50, 50)]), (75, 75, 50, True, 1.5, [(33, 50), (66, 75), (75, 75)]), (50, 50, 8, False, 1.5, [(8, 8), (16, 16), (24, 24), (32, 32), (40, 40), (48, 48), (50, 50)]), (50, 75, 8, False, 1.5, [(8, 8), (16, 16), (24, 24), (32, 32), (40, 40), (48, 48), (50, 56), (50, 64), (50, 72), (50, 75)])]

@pytest.mark.parametrize('max_seq_len_source, max_seq_len_target, bucket_width, bucket_scaling, length_ratio, expected_buckets', define_parallel_bucket_tests)
def test_define_parallel_buckets(max_seq_len_source: int, max_seq_len_target: int, bucket_width: int, bucket_scaling: bool, length_ratio: float, expected_buckets: List[Tuple[int, int]]) -> None:
    buckets = data_io.define_parallel_buckets(max_seq_len_source, max_seq_len_target, bucket_width=bucket_width, bucket_scaling=bucket_scaling, length_ratio=length_ratio)
    assert buckets == expected_buckets

get_bucket_tests: List[Tuple[List[int], int, Optional[int]]] = [([10, 20, 30, 40, 50], 50, 50), ([10, 20, 30, 40, 50], 11, 20), ([10, 20, 30, 40, 50], 9, 10), ([10, 20, 30, 40, 50], 51, None), ([10, 20, 30, 40, 50], 1, 10), ([10, 20, 30, 40, 50], 0, 10), ([], 50, None)]

@pytest.mark.parametrize('buckets, length, expected_bucket', get_bucket_tests)
def test_get_bucket(buckets: List[int], length: int, expected_bucket: Optional[int]) -> None:
    bucket = data_io.get_bucket(length, buckets)
    assert bucket == expected_bucket

tokens2ids_tests: List[Tuple[List[str], dict, List[int]]] = [(['a', 'b', 'c'], {'a': 1, 'b': 0, 'c': 300, C.UNK_SYMBOL: 12}, [1, 0, 300]), (['a', 'x', 'c'], {'a': 1, 'b': 0, 'c': 300, C.UNK_SYMBOL: 12}, [1, 12, 300])]

@pytest.mark.parametrize('tokens, vocab, expected_ids', tokens2ids_tests)
def test_tokens2ids(tokens: List[str], vocab: dict, expected_ids: List[int]) -> None:
    ids = data_io.tokens2ids(tokens, vocab)
    assert ids == expected_ids

@pytest.mark.parametrize('tokens, expected_ids', [(['1', '2', '3', '0'], [1, 2, 3, 0]), ([], [])])
def test_strids2ids(tokens: List[str], expected_ids: List[int]) -> None:
    ids = data_io.strids2ids(tokens)
    assert ids == expected_ids
