# Copyright 2017--2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import os
import random
from tempfile import TemporaryDirectory
from typing import Optional, List, Tuple, Dict, Any, Set, Union, Iterable

import numpy as np
import pytest
import torch

from sockeye import constants as C
from sockeye import data_io
from sockeye import utils
from sockeye import vocab
from sockeye.test_utils import tmp_digits_dataset
from sockeye.utils import SockeyeError, get_tokens, seed_rngs

seed_rngs(12)

define_bucket_tests: List[Tuple[int, int, List[int]]] = [(50, 10, [10, 20, 30, 40, 50]),
                       (50, 20, [20, 40, 50]),
                       (50, 50, [50]),
                       (5, 10, [5]),
                       (11, 5, [5, 10, 11]),
                       (19, 10, [10, 19])]


@pytest.mark.parametrize("max_seq_len, step, expected_buckets", define_bucket_tests)
def test_define_buckets(max_seq_len: int, step: int, expected_buckets: List[int]) -> None:
    buckets = data_io.define_buckets(max_seq_len, step=step)
    assert buckets == expected_buckets


define_parallel_bucket_tests: List[Tuple[int, int, int, bool, float, List[Tuple[int, int]]] = [(50, 50, 10, True, 1.0, [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)]),
                                (50, 50, 10, True, 0.5,
                                 [(10, 5), (20, 10), (30, 15), (40, 20), (50, 25), (50, 30), (50, 35), (50, 40),
                                  (50, 45), (50, 50)]),
                                (10, 10, 10, True, 0.1,
                                 [(10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10)]),
                                (10, 5, 10, True, 0.01, [(10, 2), (10, 3), (10, 4), (10, 5)]),
                                (50, 50, 10, True, 2.0,
                                 [(5, 10), (10, 20), (15, 30), (20, 40), (25, 50), (30, 50), (35, 50), (40, 50),
                                  (45, 50), (50, 50)]),
                                (5, 10, 10, True, 10.0, [(2, 10), (3, 10), (4, 10), (5, 10)]),
                                (5, 10, 10, True, 11.0, [(2, 10), (3, 10), (4, 10), (5, 10)]),
                                (50, 50, 50, True, 0.5, [(50, 25), (50, 50)]),
                                (50, 50, 50, True, 1.5, [(33, 50), (50, 50)]),
                                (75, 75, 50, True, 1.5, [(33, 50), (66, 75), (75, 75)]),
                                (50, 50, 8, False, 1.5, [(8, 8), (16, 16), (24, 24), (32, 32), (40, 40), (48, 48),
                                                         (50, 50)]),
                                (50, 75, 8, False, 1.5, [(8, 8), (16, 16), (24, 24), (32, 32), (40, 40), (48, 48),
                                                         (50, 56), (50, 64), (50, 72), (50, 75)])]


@pytest.mark.parametrize("max_seq_len_source, max_seq_len_target, bucket_width, bucket_scaling, length_ratio,"
                         "expected_buckets", define_parallel_bucket_tests)
def test_define_parallel_buckets(max_seq_len_source: int, max_seq_len_target: int, bucket_width: int, bucket_scaling: bool, length_ratio: float,
                                 expected_buckets: List[Tuple[int, int]]) -> None:
    buckets = data_io.define_parallel_buckets(max_seq_len_source, max_seq_len_target, bucket_width=bucket_width,
                                              bucket_scaling=bucket_scaling, length_ratio=length_ratio)
    assert buckets == expected_buckets


get_bucket_tests: List[Tuple[List[int], int, Optional[int]]] = [([10, 20, 30, 40, 50], 50, 50),
                    ([10, 20, 30, 40, 50], 11, 20),
                    ([10, 20, 30, 40, 50], 9, 10),
                    ([10, 20, 30, 40, 50], 51, None),
                    ([10, 20, 30, 40, 50], 1, 10),
                    ([10, 20, 30, 40, 50], 0, 10),
                    ([], 50, None)]


@pytest.mark.parametrize("buckets, length, expected_bucket",
                         get_bucket_tests)
def test_get_bucket(buckets: List[int], length: int, expected_bucket: Optional[int]) -> None:
    bucket = data_io.get_bucket(length, buckets)
    assert bucket == expected_bucket


tokens2ids_tests: List[Tuple[List[str], Dict[str, int], List[int]]] = [(["a", "b", "c"], {"a": 1, "b": 0, "c": 300, C.UNK_SYMBOL: 12}, [1, 0, 300]),
                    (["a", "x", "c"], {"a": 1, "b": 0, "c": 300, C.UNK_SYMBOL: 12}, [1, 12, 300])]


@pytest.mark.parametrize("tokens, vocab, expected_ids", tokens2ids_tests)
def test_tokens2ids(tokens: List[str], vocab: Dict[str, int], expected_ids: List[int]) -> None:
    ids = data_io.tokens2ids(tokens, vocab)
    assert ids == expected_ids


@pytest.mark.parametrize("tokens, expected_ids", [(["1", "2", "3", "0"], [1, 2, 3, 0]), ([], [])])
def test_strids2ids(tokens: List[str], expected_ids: List[int]) -> None:
    ids = data_io.strids2ids(tokens)
    assert ids == expected_ids


sequence_reader_tests: List[Tuple[List[str], bool, bool, bool]] = [(["1 2 3", "2", "", "2 2 2"], False, False, False),
                         (["a b c", "c"], True, False, False),
                         (["a b c", ""], True, False, False),
                         (["a b c", "c"], True, True, True)]


@pytest.mark.parametrize("sequences, use_vocab, add_bos, add_eos", sequence_reader_tests)
def test_sequence_reader(sequences: List[str], use_vocab: bool, add_bos: bool, add_eos: bool) -> None:
    with TemporaryDirectory() as work_dir:
        path = os.path.join(work_dir, 'input')
        with open(path, 'w') as f:
            for sequence in sequences:
                print(sequence, file=f)

        vocabulary = vocab.build_vocab(sequences) if use_vocab else None

        reader = data_io.SequenceReader(path, vocabulary=vocabulary, add_bos=add_bos, add_eos=add_eos)

        read_sequences = [s for s in reader]
        assert len(read_sequences) == len(sequences)

        if vocabulary is None:
            with pytest.raises(SockeyeError) as e:
                data_io.SequenceReader(path, vocabulary=vocabulary, add_bos=True)
            assert str(e.value) == "Adding a BOS or EOS symbol requires a vocabulary"

            expected_sequences = [data_io.strids2ids(get_tokens(s)) if s else None for s in sequences]
            assert read_sequences == expected_sequences
        else:
            expected_sequences = [data_io.tokens2ids(get_tokens(s), vocabulary) if s else None for s in sequences]
            if add_bos:
                expected_sequences = [[vocabulary[C.BOS_SYMBOL]] + s if s else None for s in expected_sequences]
            if add_eos:
                expected_sequences = [s + [vocabulary[C.EOS_SYMBOL]] if s else None for s in expected_sequences]
            assert read_sequences == expected_sequences


@pytest.mark.parametrize("source_iterables, target_iterables",
                         [
                             (
                                     [[[0], [1, 1], [2], [3, 3, 3]], [[0], [1, 1], [2], [3, 3, 3]]],
                                     [[[0], [1]]]
                             ),
                             (
                                     [[[0], [1, 1]], [[0], [1, 1]]],
                                     [[[0], [1, 1], [2], [3, 3, 3]]]
                             ),
                             (
                                     [[[0], [1, 1]]],
                                     [[[0], [1, 1], [2], [3, 3, 3]]]
                             ),
                         ])
def test_nonparallel_iter(source_iterables: List[List[List[int]]], target_iterables: List[List[List[int]]]) -> None:
    with pytest.raises(SockeyeError) as e:
        list(data_io.parallel_iter(source_iterables, target_iterables))
    assert str(e.value) == "Different number of lines in source(s) and target(s) iterables."


@pytest.mark.parametrize("source_iterables, target_iterables",
                         [
                             (
                                     [[[0], [1, 1]], [[0], [1]]],
                                     [[[0], [1]]]
                             )
                         ])
def test_not_source_token_parallel_iter(source_iterables: List[List[List[int]]], target_iterables: List[List[List[int]]]) -> None:
    with pytest.raises(SockeyeError) as e:
        list(data_io.parallel_iter(source_iterables, target_iterables))
    assert str(e.value).startswith("Source sequences are not token-parallel")


@pytest.mark.parametrize("source_iterables, target_iterables",
                         [
                             (
                                     [[[0], [1]]],
                                     [[[0], [1, 1]], [[0], [1]]],
                             )
                         ])
def test_not_target_token_parallel_iter(source_iterables: List[List[List[int]]], target_iterables: List[List[List[int]]]) -> None:
    with pytest.raises(SockeyeError) as e:
        list(data_io.parallel_iter(source_iterables, target_iterables))
    assert str(e.value).startswith("Target sequences are not token-parallel")


@pytest.mark.parametrize("source_iterables, target_iterables, expected",
                         [
                             (
                                     [[[0], [1, 1]], [[0], [1, 1]]],
                                     [[[0], [1]]],
                                     [([[0], [0]], [[0]]), ([[1, 1], [1, 1]], [[1]])]
                             ),
                             (
                                     [[[0], None], [[0], None]],
                                     [[[0], [1]]],
                                     [([[0], [0]], [[0]])]
                             ),
                             (
                                     [[[0], [1, 1]], [[0], [1, 1]]],
                                     [[[0], None]],
                                     [([[0], [0]], [[0]])]
                             ),
                             (
                                     [[None, [1, 1]], [None, [1, 1]]],
                                     [[None, [1]]],
                                     [([[1, 1], [1, 1]], [[1]])]
                             ),
                             (
                                     [[None, [1]]],
                                     [[None, [1, 1]], [None, [1, 1]]],
                                     [([[1]], [[1, 1], [1, 1]])]
                             ),
                             (
                                     [[None, [1, 1]], [None, [1, 1]]],
                                     [[None, None]],
                                     []
                             )
                         ])
def test_parallel_iter(source_iterables: List[List[Optional[List[int]]]], target_iterables: List[List[Optional[List[int]]]], expected: List[Tuple[List[List[int]], List[List[int]]]]) -> None:
    assert list(data_io.parallel_iter(source_iterables, target_iterables)) == expected


def test_sample_based_define_bucket_batch_sizes() -> None:
    batch_type = C.BATCH_TYPE_SENTENCE
    batch_size = 32
    max_seq_len = 100
    buckets = data_io.define_parallel_buckets(max_seq_len, max_seq_len, 10, True, 1.5)
    bucket_batch_sizes = data_io.define_bucket_batch_sizes(buckets=buckets,
                                                           batch_size=batch_size,
                                                           batch_type=batch_type,
                                                           data_target_average_len=[None] * len(buckets))
    for bbs in bucket_batch_sizes:
        assert bbs.batch_size == batch_size
        assert bbs.average_target_words_per_batch == bbs.bucket[1] * batch_size


@pytest.mark.parametrize("length_ratio,batch_sentences_multiple_of,expected_batch_sizes", [
    # Reference batch sizes manually inspected for sanity.
    (0.5, 1, [200, 100, 67, 50, 40, 33, 29, 25, 22, 20]),
    (1.5, 1, [100, 50, 33, 25, 20, 20, 20, 20]),
    (1.5, 8, [96, 48, 32, 24, 16, 16, 16, 16])])
def test_word_based_define_bucket_batch_sizes(length_ratio: float, batch_sentences_multiple_of: int, expected_batch_sizes: List[int]) -> None:
    batch_type = C.BATCH_TYPE_WORD
    batch_size = 1000
    max_seq_len = 50
    buckets = data_io.define_parallel_buckets(max_seq_len, max_seq_len, 10, True, length_ratio)
    bucket_batch_sizes = data_io.define_bucket_batch_sizes(buckets=buckets,
                                                           batch_size=batch_size,
                                                           batch_type=batch_type,
                                                           data_target_average_len=[None] * len(buckets),
                                                           batch_sentences_multiple_of=batch_sentences_multiple_of)
    for bbs, expected_batch_size in zip(bucket_batch_sizes, expected_batch_sizes):
        assert bbs.batch_size == expected_batch_size
        expected_average_target_words_per_batch = expected_batch_size * bbs.bucket[1]
        assert bbs.average_target_words_per_batch == expected_average_target_words_per_batch


@pytest.mark.parametrize("length_ratio,batch_sentences_multiple_of,expected_batch_sizes", [
    # Reference batch sizes manually inspected for sanity.
    (0.5, 1, [200, 100, 66, 50, 40, 33, 28, 25, 22, 20]),
    (1.5, 1, [100, 50, 33, 25, 20, 20, 20, 20]),
    (1.5, 8, [96, 48, 32, 24, 16, 16, 16, 16])])
def test_max_word_based_define_bucket_batch_sizes(length_ratio: float, batch_sentences_multiple_of: int, expected_batch_sizes: List[int]) -> None:
    batch_type = C.BATCH_TYPE_MAX_WORD
    batch_size = 1000
    max_seq_len = 50
    buckets = data_io.define_parallel_buckets(max_seq_len, max_seq_len, 10, True, length_ratio)
    bucket_batch_sizes = data_io.define_bucket_batch_sizes(buckets=buckets,
                                                           batch_size=batch_size,
                                                           batch_type=batch_type,
                                                           data_target_average_len=[None] * len(buckets),
                                                           batch_sentences_multiple_of=batch_sentences_multiple_of)
    for bbs, expected_batch_size in zip(bucket_batch_sizes, expected_batch_sizes):
        assert bbs.batch_size == expected_batch_size
        expected_average_target_words_per_batch = expected_batch_size * bbs