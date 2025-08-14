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

"""
Implements data iterators and I/O related functions for sequence-to-sequence models.
"""
import bisect
import logging
import math
import multiprocessing.pool
import os
import pickle
import random
from abc import abstractmethod
from collections import OrderedDict
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import chain
from typing import Any, cast, Dict, Iterator, Iterable, List, Optional, Sequence, Sized, Tuple, Set, Union

import numpy as np
import torch
import torch.distributed

from . import config
from . import constants as C
from . import utils
from . import vocab
from .utils import check_condition, smart_open, get_tokens, OnlineMeanAndVariance, combine_means, combine_stds

logger = logging.getLogger(__name__)


def define_buckets(max_seq_len: int, step: int = 10) -> List[int]:
    """
    Returns a list of integers defining bucket boundaries.
    Bucket boundaries are created according to the following policy:
    We generate buckets with a step size of step until the final bucket fits max_seq_len.
    We then limit that bucket to max_seq_len (difference between semi-final and final bucket may be less than step).

    :param max_seq_len: Maximum bucket size.
    :param step: Distance between buckets.

    :return: List of bucket sizes.
    """
    buckets = list(range(step, max_seq_len + step, step))
    buckets[-1] = max_seq_len
    return buckets


def define_parallel_buckets(max_seq_len_source: int,
                            max_seq_len_target: int,
                            bucket_width: int = 10,
                            bucket_scaling: bool = True,
                            length_ratio: float = 1.0) -> List[Tuple[int, int]]:
    """
    Returns (source, target) buckets up to (max_seq_len_source, max_seq_len_target).  The longer side of the data uses
    steps of bucket_width while the shorter side uses steps scaled down by the average target/source length ratio.  If
    one side reaches its max_seq_len before the other, width of extra buckets on that side is fixed to that max_seq_len.

    :param max_seq_len_source: Maximum source bucket size.
    :param max_seq_len_target: Maximum target bucket size.
    :param bucket_width: Width of buckets on longer side.
    :param bucket_scaling: Scale bucket steps based on length ratio.
    :param length_ratio: Length ratio of data (target/source).
    """
    source_step_size = bucket_width
    target_step_size = bucket_width
    if bucket_scaling:
        if length_ratio >= 1.0:
            # target side is longer -> scale source
            source_step_size = max(1, int(round(bucket_width / length_ratio)))
        else:
            # source side is longer, -> scale target
            target_step_size = max(1, int(round(bucket_width * length_ratio)))
    source_buckets = define_buckets(max_seq_len_source, step=source_step_size)
    target_buckets = define_buckets(max_seq_len_target, step=target_step_size)
    # Extra buckets
    if len(source_buckets) < len(target_buckets):
        source_buckets += [source_buckets[-1] for _ in range(len(target_buckets) - len(source_buckets))]
    elif len(target_buckets) < len(source_buckets):
        target_buckets += [target_buckets[-1] for _ in range(len(source_buckets) - len(target_buckets))]
    # minimum bucket size is 2 (as we add BOS symbol to target side)
    source_buckets = [max(2, b) for b in source_buckets]
    target_buckets = [max(2, b) for b in target_buckets]
    parallel_buckets = list(zip(source_buckets, target_buckets))
    # deduplicate for return
    buckets = list(OrderedDict.fromkeys(parallel_buckets))
    buckets.sort()
    return buckets


def get_bucket(seq_len: int, buckets: List[int]) -> Optional[int]:
    """
    Given sequence length and a list of buckets, return corresponding bucket.

    :param seq_len: Sequence length.
    :param buckets: List of buckets.
    :return: Chosen bucket.
    """
    bucket_idx = bisect.bisect_left(buckets, seq_len)
    if bucket_idx == len(buckets):
        return None
    return buckets[bucket_idx]


@dataclass
class BucketBatchSize:
    bucket: Tuple[int, int]  # The corresponding bucket.
    batch_size: int  # Number of sequences in each batch.
    average_target_words_per_batch: float  # Approximate number of target non-padding tokens in each batch.


def define_bucket_batch_sizes(buckets: List[Tuple[int, int]],
                              batch_size: int,
                              batch_type: str,
                              data_target_average_len: List[Optional[float]],
                              batch_sentences_multiple_of: int = 1) -> List[BucketBatchSize]:
    """
    Compute bucket-specific batch sizes (sentences, average_target_words).

    If sentence batching: number of sentences is the same for each batch.

    If word batching: number of sentences for each batch is the number of words
    closest to the target batch size. Number of sentences is rounded to the
    nearest multiple of batch_sentences_multiple_of. Average target sentence
    length (non-padding symbols) is used for word number calculations.

    If max-word batching: number of sentences for each batch is set to the
    multiple of batch_sentences_multiple_of that is closest to batch_size
    without exceeding the value.

    :param buckets: Bucket list.
    :param batch_size: Batch size.
    :param batch_type: Type of batching.
    :param data_target_average_len: Optional average target length for each
        bucket.
    :param batch_sentences_multiple_of: Guarantee the number of sentences in
        each bucket's batch to a multiple of this value.
    """
    check_condition(len(data_target_average_len) == len(buckets),
                    "Must provide None or average target length for each bucket")
    data_target_average_len = list(data_target_average_len)
    bucket_batch_sizes = []  # type: List[BucketBatchSize]
    largest_total_num_words = 0

    for buck_idx, bucket in enumerate(buckets):
        # Target/label length with padding
        padded_seq_len = bucket[1]
        # Average target/label length excluding padding
        if data_target_average_len[buck_idx] is None:
            data_target_average_len[buck_idx] = padded_seq_len
        average_seq_len = data_target_average_len[buck_idx]

        # Batch size for each bucket is measured in sentences:
        # - word batching: convert average word-based size to number of
        #       sequences
        # - max-word batching: convert max word-based size to number of
        #       sequences
        # - sentence batching: use batch size directly
        if batch_type == C.BATCH_TYPE_WORD:
            check_condition(padded_seq_len <= batch_size, "Word batch size must cover sequence lengths for all"
                                                          " buckets: (%d > %d)" % (padded_seq_len, batch_size))
            # Multiple of minimum batch step closest to target number of words,
            # assuming each sentence is of average length
            batch_size_seq = batch_sentences_multiple_of * max(1, round((batch_size / average_seq_len) /
                                                                        batch_sentences_multiple_of))
        elif batch_type == C.BATCH_TYPE_MAX_WORD:
            check_condition(padded_seq_len <= batch_size,
                            'Word batch size must cover sequence lengths for all buckets: (%d > %d)'
                            % (padded_seq_len, batch_size))
            # Max number of sequences without exceeding batch size
            batch_size_seq = batch_size // padded_seq_len
            check_condition(batch_size_seq // batch_sentences_multiple_of > 0,
                            'Please increase the batch size to avoid the batch size being rounded down to 0.')
            # Round down to closest multiple
            batch_size_seq = (batch_size_seq // batch_sentences_multiple_of) * batch_sentences_multiple_of
        elif batch_type == C.BATCH_TYPE_SENTENCE:
            batch_size_seq = batch_size
        else:
            raise ValueError('Unknown batch type: %s' % batch_type)
        # Number of words here is an average of non-padding tokens
        batch_size_word = batch_size_seq * average_seq_len

        bucket_batch_sizes.append(BucketBatchSize(bucket, batch_size_seq, batch_size_word))
        # Track largest number of source or target word samples in a batch
        largest_total_num_words = max(largest_total_num_words, batch_size_seq * max(*bucket))

    return bucket_batch_sizes


def calculate_length_statistics(source_iterables: Sequence[Iterable[Any]],
                                target_iterables: Sequence[Iterable[Any]],
                                max_seq_len_source: int,
                                max_seq_len_target: int) -> 'LengthStatistics':
    """
    Returns mean and standard deviation of target-to-source length ratios of parallel corpus.

    :param source_iterables: Source sequence readers.
    :param target_iterables: Target sequence readers.
    :param max_seq_len_source: Maximum source sequence length.
    :param max_seq_len_target: Maximum target sequence length.
    :return: The number of sentences as well as the mean and standard deviation of target to source length ratios.
    """
    mean_and_variance = OnlineMeanAndVariance()

    for sources, targets in parallel_iter(source_iterables, target_iterables):
        source_len = len(sources[0])
        target_len = len(targets[0])
        if source_len > max_seq_len_source or target_len > max_seq_len_target:
            continue

        length_ratio = target_len / source_len
        mean_and_variance.update(length_ratio)

    return LengthStatistics(mean_and_variance.count, mean_and_variance.mean, mean_and_variance.std)


def analyze_sequence_lengths(sources: List[str],
                             targets: List[str],
                             vocab_sources: List[vocab.Vocab],
                             vocab_targets: List[vocab.Vocab],
                             max_seq_len_source: int,
                             max_seq_len_target: int) -> 'LengthStatistics':
    train_sources_sentences, train_targets_sentences = create_sequence_readers(sources, targets,
                                                                               vocab_sources, vocab_targets)

    length_statistics = calculate_length_statistics(train_sources_sentences, train_targets_sentences,
                                                    max_seq_len_source, max_seq_len_target)

    logger.info("%d sequences of maximum length (%d, %d) in '%s' and '%s'.",
                length_statistics.num_sents, max_seq_len_source, max_seq_len_target, sources[0], targets[0])
    logger.info("Mean training target/source length ratio: %.2f (+-%.2f)",
                length_statistics.length_ratio_mean,
                length_statistics.length_ratio_std)
    return length_statistics


def are_none(sequences: Sequence[Sized]) -> bool:
    """
    Returns True if all sequences are None.
    """
    if not sequences:
        return True
    return all(s is None for s in sequences)


def are_token_parallel(sequences: Sequence[Sized]) -> bool:
    """
    Returns True if all sequences in the list have the same length.
    """
    if not sequences or len(sequences) == 1:
            return True
    else:
        return all(len(s) == len(sequences[0]) for s in sequences)


class DataStatisticsAccumulator:

    def __init__(self,
                 buckets: List[Tuple[int, int]],
                 vocab_source: Optional[Dict[str, int]],
                 vocab_target: Dict[str, int],
                 length_ratio_mean: float,
                 length_ratio_std: float) -> None:
        self.buckets = buckets
        num_buckets = len(buckets)
        self.length_ratio_mean = length_ratio_mean
        self.length_ratio_std = length_ratio_std
        if vocab_source is not None:
            self.unk_id_source = vocab_source[C.UNK_SYMBOL]
            self.size_vocab_source = len(vocab_source)
        else:
            self.unk_id_source = None
            self.size_vocab_source = 0
        self.unk_id_target = vocab_target[C.UNK_SYMBOL]
        self.size_vocab_target = len(vocab_target)
        self.num_sents = 0
        self.num_discarded = 0
        self.num_tokens_source = 0
        self.num_tokens_target = 0
        self.num_unks_source = 0
        self.num_unks_target = 0
        self.max_observed_len_source = 0
        self.max_observed_len_target = 0
        self._mean_len_target_per_bucket = [OnlineMeanAndVariance() for _ in range(num_buckets)]
        self._length_ratio_per_bucket = [OnlineMeanAndVariance() for _ in range(num_buckets)]

    def sequence_pair(self,
                      source: List[int],
                      target: List[int],
                      bucket_idx: Optional[int]):
        if bucket_idx is None:
            self.num_discarded += 1
            return

        source_len = len(source)
        target_len = len(target)
        length_ratio = target_len / (source_len if source_len else 1.)

        self._mean_len_target_per_bucket[bucket_idx].update(target_len)
        self._length_ratio_per_bucket[bucket_idx].update(length_ratio)

        self.num_sents += 1
        self.num_tokens_source += source_len
        self.num_tokens_target += target_len
        self.max_observed_len_source = max(source_len, self.max_observed_len_source)
        self.max_observed_len_target = max(target_len, self.max_observed_len_target)

        if self.unk_id_source is not None:
            self.num_unks_source += source.count(self.unk_id_source)
        self.num_unks_target += target.count(self.unk_id_target)

    @property
    def mean_len_target_per_bucket(self) -> List[Optional[float]]:
        return [mean_and_variance.mean if mean_and_variance.count > 0 else None
                for mean_and_variance in self._mean_len_target_per_bucket]

    @property
    def length_ratio_stats_per_bucket(self) -> List[Tuple[Optional[float], Optional[float]]]:
        return [(mean_and_variance.mean, mean_and_variance.std) if mean_and_variance.count > 0 else (None, None)
                for mean_and_variance in self._length_ratio_per_bucket]

    @property
    def statistics(self) -> 'DataStatistics':
        num_sents_per_bucket = [mean_and_variance.count for mean_and_variance in self._mean_len_target_per_bucket]
        return DataStatistics(num_sents=self.num_sents,
                              num_discarded=self.num_discarded,
                              num_tokens_source=self.num_tokens_source,
                              num_tokens_target=self.num_tokens_target,
                              num_unks_source=self.num_unks_source,
                              num_unks_target=self.num_unks_target,
                              max_observed_len_source=self.max_observed_len_source,
                              max_observed_len_target=self.max_observed_len_target,
                              size_vocab_source=self.size_vocab_source,
                              size_vocab_target=self.size_vocab_target,
                              length_ratio_mean=self.length_ratio_mean,
                              length_ratio_std=self.length_ratio_std,
                              buckets=self.buckets,
                              num_sents_per_bucket=num_sents_per_bucket,
                              average_len_target_per_bucket=self.mean_len_target_per_bucket,
                              length_ratio_stats_per_bucket=self.length_ratio_stats_per_bucket)


def create_shards(source_fnames: List[str],
                  target_fnames: List[str],
                  num_shards: int,
                  output_prefix: str) -> Tuple[List[Tuple[Tuple[str, ...], Tuple[str, ...]]], bool]:
    """
    Assign source/target sentence pairs to shards at random.

    :param source_fnames: The path to the source text (and optional token-parallel factor files).
    :param target_fnames: The path to the target text (and optional token-parallel factor files).
    :param num_shards: The total number of shards.
    :param output_prefix: The prefix under which the shard files will be created.
    :return: List of tuples of source (and source factor) file names and target (and target factor) file names for each
             shard and a flag of whether the returned file names are temporary and can be deleted.
    """
    if num_shards == 1:
        return [(tuple(source_fnames), tuple(target_fnames))], True
    os.makedirs(output_prefix, exist_ok=True)
    sources_shard_fnames = [[os.path.join(output_prefix, C.SHARD_SOURCE % i) + ".%d" % f for i in range(num_shards)]
                            for f in range(len(source_fnames))]
    targets_shard_fnames = [[os.path.join(output_prefix, C.SHARD_TARGET % i) + ".%d" % f for i in range(num_shards)]
                            for f in range(len(target_fnames))]

    with ExitStack() as exit_stack:
        sources_shards = [[exit_stack.enter_context(smart_open(f, mode="wb")) for f in sources_shard_fnames[i]] for i in
                          range(len(source_fnames))]
        targets_shards = [[exit_stack.enter_context(smart_open(f, mode="wb")) for f in targets_shard_fnames[i]] for i in
                          range(len(target_fnames))]

        source_readers = [exit_stack.enter_context(smart_open(f, mode="rb")) for f in source_fnames]
        target_readers