#!/usr/bin/env python3
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
from typing import Any, cast, Dict, Iterator, Iterable, List, Optional, Sequence, Tuple, Set, Generator

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
    buckets: List[int] = list(range(step, max_seq_len + step, step))
    buckets[-1] = max_seq_len
    return buckets


def define_parallel_buckets(max_seq_len_source: int, max_seq_len_target: int, bucket_width: int = 10, bucket_scaling: bool = True, length_ratio: float = 1.0) -> List[Tuple[int, int]]:
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
    source_step_size: int = bucket_width
    target_step_size: int = bucket_width
    if bucket_scaling:
        if length_ratio >= 1.0:
            source_step_size = max(1, int(round(bucket_width / length_ratio)))
        else:
            target_step_size = max(1, int(round(bucket_width * length_ratio)))
    source_buckets: List[int] = define_buckets(max_seq_len_source, step=source_step_size)
    target_buckets: List[int] = define_buckets(max_seq_len_target, step=target_step_size)
    if len(source_buckets) < len(target_buckets):
        source_buckets += [source_buckets[-1] for _ in range(len(target_buckets) - len(source_buckets))]
    elif len(target_buckets) < len(source_buckets):
        target_buckets += [target_buckets[-1] for _ in range(len(source_buckets) - len(target_buckets))]
    source_buckets = [max(2, b) for b in source_buckets]
    target_buckets = [max(2, b) for b in target_buckets]
    parallel_buckets: List[Tuple[int, int]] = list(zip(source_buckets, target_buckets))
    buckets = list(OrderedDict.fromkeys(parallel_buckets))
    buckets.sort()
    return buckets


def get_bucket(seq_len: int, buckets: List[int]) -> Optional[int]:
    """
    Given sequence length and a list of buckets, return corresponding bucket.

    :param seq_len: Sequence length.
    :param buckets: List of buckets.
    :return: Chosen bucket or None if no bucket found.
    """
    bucket_idx: int = bisect.bisect_left(buckets, seq_len)
    if bucket_idx == len(buckets):
        return None
    return buckets[bucket_idx]


@dataclass
class BucketBatchSize:
    bucket: Tuple[int, int]
    batch_size: int
    batch_size_word: int


def define_bucket_batch_sizes(buckets: List[Tuple[int, int]], batch_size: int, batch_type: str, data_target_average_len: Sequence[Optional[float]], batch_sentences_multiple_of: int = 1) -> List[BucketBatchSize]:
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
    :param data_target_average_len: Optional average target length for each bucket.
    :param batch_sentences_multiple_of: Guarantee the number of sentences in each bucket's batch to a multiple of this value.
    """
    check_condition(len(data_target_average_len) == len(buckets), 'Must provide None or average target length for each bucket')
    data_target_average_len = list(data_target_average_len)
    bucket_batch_sizes: List[BucketBatchSize] = []
    largest_total_num_words: int = 0
    for buck_idx, bucket in enumerate(buckets):
        padded_seq_len: int = bucket[1]
        if data_target_average_len[buck_idx] is None:
            data_target_average_len[buck_idx] = float(padded_seq_len)
        average_seq_len: float = data_target_average_len[buck_idx]  # type: ignore
        if batch_type == C.BATCH_TYPE_WORD:
            check_condition(padded_seq_len <= batch_size, 'Word batch size must cover sequence lengths for all buckets: (%d > %d)' % (padded_seq_len, batch_size))
            batch_size_seq: int = batch_sentences_multiple_of * max(1, round(batch_size / average_seq_len / batch_sentences_multiple_of))
        elif batch_type == C.BATCH_TYPE_MAX_WORD:
            check_condition(padded_seq_len <= batch_size, 'Word batch size must cover sequence lengths for all buckets: (%d > %d)' % (padded_seq_len, batch_size))
            batch_size_seq = batch_size // padded_seq_len
            check_condition(batch_size_seq // batch_sentences_multiple_of > 0, 'Please increase the batch size to avoid the batch size being rounded down to 0.')
            batch_size_seq = (batch_size_seq // batch_sentences_multiple_of) * batch_sentences_multiple_of
        elif batch_type == C.BATCH_TYPE_SENTENCE:
            batch_size_seq = batch_size
        else:
            raise ValueError('Unknown batch type: %s' % batch_type)
        batch_size_word: float = batch_size_seq * average_seq_len
        bucket_batch_sizes.append(BucketBatchSize(bucket, batch_size_seq, int(batch_size_word)))
        largest_total_num_words = max(largest_total_num_words, batch_size_seq * max(*bucket))
    return bucket_batch_sizes


def calculate_length_statistics(source_iterables: Iterable[Any], target_iterables: Iterable[Any], max_seq_len_source: int, max_seq_len_target: int) -> "LengthStatistics":
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
        source_len: int = len(sources[0])
        target_len: int = len(targets[0])
        if source_len > max_seq_len_source or target_len > max_seq_len_target:
            continue
        length_ratio: float = target_len / source_len
        mean_and_variance.update(length_ratio)
    return LengthStatistics(mean_and_variance.count, mean_and_variance.mean, mean_and_variance.std)


def analyze_sequence_lengths(sources: List[str], targets: List[str], vocab_sources: List[Dict[str, int]], vocab_targets: List[Dict[str, int]], max_seq_len_source: int, max_seq_len_target: int) -> "LengthStatistics":
    train_sources_sentences, train_targets_sentences = create_sequence_readers(sources, targets, vocab_sources, vocab_targets)
    length_statistics: LengthStatistics = calculate_length_statistics(train_sources_sentences, train_targets_sentences, max_seq_len_source, max_seq_len_target)
    logger.info("%d sequences of maximum length (%d, %d) in '%s' and '%s'.", length_statistics.num_sents, max_seq_len_source, max_seq_len_target, sources[0], targets[0])
    logger.info('Mean training target/source length ratio: %.2f (+-%.2f)', length_statistics.length_ratio_mean, length_statistics.length_ratio_std)
    return length_statistics


def are_none(sequences: Sequence[Optional[Any]]) -> bool:
    """
    Returns True if all sequences are None.
    """
    if not sequences:
        return True
    return all((s is None for s in sequences))


def are_token_parallel(sequences: Sequence[List[Any]]) -> bool:
    """
    Returns True if all sequences in the list have the same length.
    """
    if not sequences or len(sequences) == 1:
        return True
    else:
        return all((len(s) == len(sequences[0]) for s in sequences))


class DataStatisticsAccumulator:
    def __init__(self, buckets: List[Tuple[int, int]], vocab_source: Optional[Dict[str, int]], vocab_target: Dict[str, int], length_ratio_mean: float, length_ratio_std: float) -> None:
        self.buckets: List[Tuple[int, int]] = buckets
        num_buckets: int = len(buckets)
        self.length_ratio_mean: float = length_ratio_mean
        self.length_ratio_std: float = length_ratio_std
        if vocab_source is not None:
            self.unk_id_source: int = vocab_source[C.UNK_SYMBOL]
            self.size_vocab_source: int = len(vocab_source)
        else:
            self.unk_id_source = None
            self.size_vocab_source = 0
        self.unk_id_target: int = vocab_target[C.UNK_SYMBOL]
        self.size_vocab_target: int = len(vocab_target)
        self.num_sents: int = 0
        self.num_discarded: int = 0
        self.num_tokens_source: int = 0
        self.num_tokens_target: int = 0
        self.num_unks_source: int = 0
        self.num_unks_target: int = 0
        self.max_observed_len_source: int = 0
        self.max_observed_len_target: int = 0
        self._mean_len_target_per_bucket: List[OnlineMeanAndVariance] = [OnlineMeanAndVariance() for _ in range(num_buckets)]
        self._length_ratio_per_bucket: List[OnlineMeanAndVariance] = [OnlineMeanAndVariance() for _ in range(num_buckets)]

    def sequence_pair(self, source: List[int], target: List[int], bucket_idx: Optional[int]) -> None:
        if bucket_idx is None:
            self.num_discarded += 1
            return
        source_len: int = len(source)
        target_len: int = len(target)
        length_ratio: float = target_len / (source_len if source_len else 1.0)
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
        return [mean_and_variance.mean if mean_and_variance.count > 0 else None for mean_and_variance in self._mean_len_target_per_bucket]

    @property
    def length_ratio_stats_per_bucket(self) -> List[Tuple[Optional[float], Optional[float]]]:
        return [(mean_and_variance.mean, mean_and_variance.std) if mean_and_variance.count > 0 else (None, None) for mean_and_variance in self._length_ratio_per_bucket]

    @property
    def statistics(self) -> "DataStatistics":
        num_sents_per_bucket: List[int] = [mean_and_variance.count for mean_and_variance in self._mean_len_target_per_bucket]
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


def create_shards(source_fnames: List[str], target_fnames: List[str], num_shards: int, output_prefix: str) -> Tuple[List[Tuple[Tuple[str, ...], Tuple[str, ...]]], bool]:
    """
    Assign source/target sentence pairs to shards at random.

    :param source_fnames: The path to the source text (and optional token-parallel factor files).
    :param target_fnames: The path to the target text (and optional token-parallel factor files).
    :param num_shards: The total number of shards.
    :param output_prefix: The prefix under which the shard files will be created.
    :return: List of tuples of source (and source factor) file names and target (and target factor) file names for each shard and a flag of whether the returned file names are temporary and can be deleted.
    """
    if num_shards == 1:
        return ([(tuple(source_fnames), tuple(target_fnames))], True)
    os.makedirs(output_prefix, exist_ok=True)
    sources_shard_fnames: List[List[str]] = [[os.path.join(output_prefix, C.SHARD_SOURCE % i) + '.%d' % f for i in range(num_shards)] for f in range(len(source_fnames))]
    targets_shard_fnames: List[List[str]] = [[os.path.join(output_prefix, C.SHARD_TARGET % i) + '.%d' % f for i in range(num_shards)] for f in range(len(target_fnames))]
    with ExitStack() as exit_stack:
        sources_shards = [[exit_stack.enter_context(smart_open(f, mode='wb')) for f in sources_shard_fnames[i]] for i in range(len(source_fnames))]
        targets_shards = [[exit_stack.enter_context(smart_open(f, mode='wb')) for f in targets_shard_fnames[i]] for i in range(len(target_fnames))]
        source_readers = [exit_stack.enter_context(smart_open(f, mode='rb')) for f in source_fnames]
        target_readers = [exit_stack.enter_context(smart_open(f, mode='rb')) for f in target_fnames]
        random_shard_iter = iter(lambda: random.randrange(num_shards), None)
        for (sources, targets), random_shard_index in zip(parallel_iter(source_readers, target_readers, True, False), random_shard_iter):
            random_shard_index = cast(int, random_shard_index)
            for i, line in enumerate(sources):
                file = sources_shards[i][random_shard_index]
                file.write(line)
            for i, line in enumerate(targets):
                file = targets_shards[i][random_shard_index]
                file.write(line)
    sources_shard_fnames_by_shards = zip(*sources_shard_fnames)
    targets_shard_fnames_by_shards = zip(*targets_shard_fnames)
    return (list(zip(sources_shard_fnames_by_shards, targets_shard_fnames_by_shards)), False)


def get_prepended_token_length(ids: List[int], eop_id: int) -> int:
    """
    Gets the length of prepended tokens before the end-of-prepending tag (inclusive).

    :param ids: List of token ids.
    :param eop_id: End-of-prepending tag id.
    :return: Length of prepended tokens.
    """
    if eop_id == C.INVALID_ID:
        return 0
    try:
        return ids.index(eop_id) + 1
    except ValueError:
        return 0


class RawParallelDatasetLoader:
    """
    Loads a data set of variable-length parallel source/target sequences into buckets of tensors.
    """
    def __init__(self, buckets: List[Tuple[int, int]], eos_id: int, pad_id: int, eop_id: int = C.INVALID_ID, skip_blanks: bool = True, dtype: str = 'int32', shift_target_factors: bool = C.TARGET_FACTOR_SHIFT) -> None:
        self.buckets: List[Tuple[int, int]] = buckets
        self.eos_id: int = eos_id
        self.pad_id: int = pad_id
        self.eop_id: int = eop_id
        self.skip_blanks: bool = skip_blanks
        self.dtype: str = dtype
        self.shift_target_factors: bool = shift_target_factors

    def load(self, source_iterables: Iterable[Any], target_iterables: Iterable[Any], num_samples_per_bucket: Sequence[int]) -> "ParallelDataSet":
        assert len(num_samples_per_bucket) == len(self.buckets)
        num_source_factors: int = len(source_iterables)  # assuming list of iterables, one per factor
        num_target_factors: int = len(target_iterables)
        data_source: List[np.ndarray] = [np.full((num_samples, source_len, num_source_factors), self.pad_id, dtype=self.dtype)
                                         for (source_len, _), num_samples in zip(self.buckets, num_samples_per_bucket)]
        data_target: List[np.ndarray] = [np.full((num_samples, target_len + 1, num_target_factors), self.pad_id, dtype=self.dtype)
                                         for (_, target_len), num_samples in zip(self.buckets, num_samples_per_bucket)]
        data_prepended_source_length: Optional[List[np.ndarray]] = (
            [np.full((num_samples,), 0, dtype=self.dtype) for num_samples in num_samples_per_bucket]
            if self.eop_id != C.INVALID_ID else None)
        bucket_sample_index: List[int] = [0 for _ in self.buckets]
        num_tokens_source: int = 0
        num_tokens_target: int = 0
        num_pad_source: int = 0
        num_pad_target: int = 0
        for sources, targets in parallel_iter(source_iterables, target_iterables, skip_blanks=self.skip_blanks):
            sources = [[] if stream is None else stream for stream in sources]
            targets = [[] if stream is None else stream for stream in targets]
            source_len: int = len(sources[0])
            target_len: int = len(targets[0])
            buck_index, buck = get_parallel_bucket(self.buckets, source_len, target_len)
            if buck is None:
                if self.skip_blanks:
                    continue
                else:
                    buck_index = len(self.buckets)
                    buck = self.buckets[buck_index]
            num_tokens_source += buck[0]
            num_tokens_target += buck[1]
            num_pad_source += buck[0] - source_len
            num_pad_target += buck[1] - target_len
            sample_index: int = bucket_sample_index[buck_index]
            for i, s in enumerate(sources):
                data_source[buck_index][sample_index, 0:source_len, i] = s
            for i, t in enumerate(targets):
                if i == 0 or not self.shift_target_factors:
                    t.append(self.eos_id)
                    data_target[buck_index][sample_index, 0:target_len + 1, i] = t
                else:
                    t.insert(0, C.BOS_ID)
                    data_target[buck_index][sample_index, 0:target_len + 1, i] = t
            if data_prepended_source_length is not None:
                data_prepended_source_length[buck_index][sample_index] = get_prepended_token_length(sources[0], self.eop_id)
            bucket_sample_index[buck_index] += 1
        data_source_tensors: List[torch.Tensor] = [torch.from_numpy(data) for data in data_source]
        data_target_tensors: List[torch.Tensor] = [torch.from_numpy(data) for data in data_target]
        data_prepended_source_length_tensors: Optional[List[torch.Tensor]] = (
            [torch.from_numpy(data) for data in data_prepended_source_length] if data_prepended_source_length is not None else None)
        if num_tokens_source > 0 and num_tokens_target > 0:
            logger.info('Created bucketed parallel data set. Introduced padding: source=%.1f%% target=%.1f%%)',
                        num_pad_source / num_tokens_source * 100,
                        num_pad_target / num_tokens_target * 100)
        return ParallelDataSet(data_source_tensors, data_target_tensors, data_prepended_source_length_tensors)


def get_num_shards(num_samples: int, samples_per_shard: int, min_num_shards: int) -> int:
    """
    Returns the number of shards.

    :param num_samples: Number of training data samples.
    :param samples_per_shard: Samples per shard.
    :param min_num_shards: Minimum number of shards.
    :return: Number of shards.
    """
    return max(int(math.ceil(num_samples / samples_per_shard)), min_num_shards)


def save_shard(shard_idx: int, data_loader: RawParallelDatasetLoader, shard_sources: List[str], shard_targets: List[str], source_vocabs: List[Dict[str, int]], target_vocabs: List[Dict[str, int]], length_ratio_mean: float, length_ratio_std: float, buckets: List[Tuple[int, int]], output_prefix: str, keep_tmp_shard_files: bool) -> "DataStatistics":
    """
    Load raw shard source and target data files, map to integers using the corresponding vocabularies,
    convert data into tensors and save to disk.
    Optionally it can delete the source/target files.

    :param shard_idx: The index of the shard.
    :param data_loader: A loader for loading parallel data from sources and target.
    :param shard_sources: A list of source file names.
    :param shard_targets: A list of target file names.
    :param source_vocabs: Source vocabulary (and optional source factor vocabularies).
    :param target_vocabs: Target vocabulary (and optional target factor vocabularies).
    :param length_ratio_mean: Mean length ratio.
    :param length_ratio_std: Standard deviation of length ratios.
    :param buckets: Bucket list.
    :param output_prefix: The prefix of the output file name.
    :param keep_tmp_shard_files: Keep the sources/target files when it is True otherwise delete them.
    :return: Shard statistics.
    """
    shard_stat_accumulator = DataStatisticsAccumulator(buckets, source_vocabs[0], target_vocabs[0], length_ratio_mean, length_ratio_std)
    sources_sentences, targets_sentences = create_sequence_readers(shard_sources, shard_targets, source_vocabs, target_vocabs)
    for sources, targets in parallel_iter(sources_sentences, targets_sentences):
        source_len: int = len(sources[0])
        target_len: int = len(targets[0])
        buck_idx, _ = get_parallel_bucket(buckets, source_len, target_len)
        shard_stat_accumulator.sequence_pair(sources[0], targets[0], buck_idx)
    shard_stats = shard_stat_accumulator.statistics
    dataset = data_loader.load(sources_sentences, targets_sentences, shard_stats.num_sents_per_bucket)
    shard_fname: str = os.path.join(output_prefix, C.SHARD_NAME % shard_idx)
    shard_stats.log()
    logger.info("Writing '%s'", shard_fname)
    dataset.save(shard_fname)
    if not keep_tmp_shard_files:
        for f in chain(shard_sources, shard_targets):
            os.remove(f)
    return shard_stat_accumulator.statistics


def get_eop_id(vocab: Dict[str, int], end_of_prepending_tag: Optional[str]) -> int:
    """
    Gets end-of-prepending tag id from the vocabulary.

    :param vocab: Vocabulary dictionary.
    :param end_of_prepending_tag: Tag indicating the end of prepended text.
    :return: End-of-prepending tag id.
    """
    eop_id: int = vocab.get(end_of_prepending_tag, C.INVALID_ID)  # type: ignore
    if end_of_prepending_tag is not None:
        check_condition(eop_id != C.INVALID_ID, f"The end-of-prepending tag '{end_of_prepending_tag}' is not found in the vocabulary.")
    return eop_id


def prepare_data(source_fnames: List[str],
                 target_fnames: List[str],
                 source_vocabs: List[Dict[str, int]],
                 target_vocabs: List[Dict[str, int]],
                 source_vocab_paths: List[str],
                 target_vocab_paths: List[str],
                 shared_vocab: bool,
                 max_seq_len_source: int,
                 max_seq_len_target: int,
                 bucketing: bool,
                 bucket_width: int,
                 num_shards: int,
                 output_prefix: str,
                 bucket_scaling: bool = True,
                 end_of_prepending_tag: Optional[str] = None,
                 keep_tmp_shard_files: bool = False,
                 pool: Optional[multiprocessing.pool.Pool] = None,
                 shards: Optional[List[Tuple[List[str], List[str]]]] = None) -> None:
    """
    :param shards: List of num_shards shards of parallel source and target tuples which in turn contain tuples to shard
                   data factor file paths.
    """
    logger.info('Preparing data.')
    vocab.save_source_vocabs(source_vocabs, output_prefix)
    vocab.save_target_vocabs(target_vocabs, output_prefix)
    stats_args = ((source_path, target_path, source_vocabs, target_vocabs, max_seq_len_source, max_seq_len_target) for source_path, target_path in shards)  # type: ignore
    length_stats = pool.starmap(analyze_sequence_lengths, stats_args)  # type: ignore
    shards_num_sents = [stat.num_sents for stat in length_stats]
    shards_mean = [stat.length_ratio_mean for stat in length_stats]
    shards_std = [stat.length_ratio_std for stat in length_stats]
    length_ratio_mean: float = combine_means(shards_mean, shards_num_sents)
    length_ratio_std: float = combine_stds(shards_std, shards_mean, shards_num_sents)
    length_statistics = LengthStatistics(sum(shards_num_sents), length_ratio_mean, length_ratio_std)
    check_condition(length_statistics.num_sents > 0, 'No training sequences found with length smaller or equal than the maximum sequence length.Consider increasing %s' % C.TRAINING_ARG_MAX_SEQ_LEN)
    buckets: List[Tuple[int, int]] = (define_parallel_buckets(max_seq_len_source, max_seq_len_target, bucket_width, bucket_scaling, length_statistics.length_ratio_mean)
                                      if bucketing else [(max_seq_len_source, max_seq_len_target)])
    logger.info('Buckets: %s', buckets)
    eop_id: int = get_eop_id(source_vocabs[0], end_of_prepending_tag)
    data_loader: RawParallelDatasetLoader = RawParallelDatasetLoader(buckets=buckets, eos_id=C.EOS_ID, pad_id=C.PAD_ID, eop_id=eop_id)
    args = ((shard_idx, data_loader, shard_sources, shard_targets, source_vocabs, target_vocabs,
             length_statistics.length_ratio_mean, length_statistics.length_ratio_std, buckets, output_prefix, keep_tmp_shard_files)
            for shard_idx, (shard_sources, shard_targets) in enumerate(shards))  # type: ignore
    per_shard_statistics = pool.starmap(save_shard, args)  # type: ignore
    shard_average_len = [shard_stats.average_len_target_per_bucket for shard_stats in per_shard_statistics]
    shard_num_sents = [shard_stats.num_sents_per_bucket for shard_stats in per_shard_statistics]
    num_sents_per_bucket = [sum(n) for n in zip(*shard_num_sents)]
    average_len_target_per_bucket: List[Optional[float]] = []
    for num_sents_bucket, average_len_bucket in zip(zip(*shard_num_sents), zip(*shard_average_len)):
        if all((avg is None for avg in average_len_bucket)):
            average_len_target_per_bucket.append(None)
        else:
            average_len_target_per_bucket.append(combine_means(average_len_bucket, shards_num_sents))
    shard_length_ratios = [shard_stats.length_ratio_stats_per_bucket for shard_stats in per_shard_statistics]
    length_ratio_stats_per_bucket: List[Tuple[Optional[float], Optional[float]]] = []
    for num_sents_bucket, len_ratios_bucket in zip(zip(*shard_num_sents), zip(*shard_length_ratios)):
        if all((all((x is None for x in ratio)) for ratio in len_ratios_bucket)):
            length_ratio_stats_per_bucket.append((None, None))
        else:
            shards_mean_local = [ratio[0] for ratio in len_ratios_bucket]  # renamed to avoid conflict
            ratio_mean = combine_means(shards_mean_local, num_sents_bucket)
            ratio_std = combine_stds([ratio[1] for ratio in len_ratios_bucket], shards_mean_local, num_sents_bucket)
            length_ratio_stats_per_bucket.append((ratio_mean, ratio_std))
    data_statistics = DataStatistics(num_sents=sum(shards_num_sents),
                                     num_discarded=sum((shard_stats.num_discarded for shard_stats in per_shard_statistics)),
                                     num_tokens_source=sum((shard_stats.num_tokens_source for shard_stats in per_shard_statistics)),
                                     num_tokens_target=sum((shard_stats.num_tokens_target for shard_stats in per_shard_statistics)),
                                     num_unks_source=sum((shard_stats.num_unks_source for shard_stats in per_shard_statistics)),
                                     num_unks_target=sum((shard_stats.num_unks_target for shard_stats in per_shard_statistics)),
                                     max_observed_len_source=max((shard_stats.max_observed_len_source for shard_stats in per_shard_statistics)),
                                     max_observed_len_target=max((shard_stats.max_observed_len_target for shard_stats in per_shard_statistics)),
                                     size_vocab_source=per_shard_statistics[0].size_vocab_source,
                                     size_vocab_target=per_shard_statistics[0].size_vocab_target,
                                     length_ratio_mean=length_ratio_mean,
                                     length_ratio_std=length_ratio_std,
                                     buckets=per_shard_statistics[0].buckets,
                                     num_sents_per_bucket=num_sents_per_bucket,
                                     average_len_target_per_bucket=average_len_target_per_bucket,
                                     length_ratio_stats_per_bucket=length_ratio_stats_per_bucket)
    data_statistics.log()
    data_info = DataInfo(sources=[os.path.abspath(fname) for fname in source_fnames],
                         targets=[os.path.abspath(fname) for fname in target_fnames],
                         source_vocabs=source_vocab_paths,
                         target_vocabs=target_vocab_paths,
                         shared_vocab=shared_vocab,
                         num_shards=num_shards)
    data_info_fname: str = os.path.join(output_prefix, C.DATA_INFO)
    logger.info("Writing data info to '%s'", data_info_fname)
    data_info.save(data_info_fname)
    config_data = DataConfig(data_statistics=data_statistics,
                             max_seq_len_source=max_seq_len_source,
                             max_seq_len_target=max_seq_len_target,
                             num_source_factors=len(source_fnames),
                             num_target_factors=len(target_fnames),
                             eop_id=eop_id)
    config_data_fname: str = os.path.join(output_prefix, C.DATA_CONFIG)
    logger.info("Writing data config to '%s'", config_data_fname)
    config_data.save(config_data_fname)
    version_file: str = os.path.join(output_prefix, C.PREPARED_DATA_VERSION_FILE)
    with open(version_file, 'w') as version_out:
        version_out.write(str(C.PREPARED_DATA_VERSION))


def get_data_statistics(source_readers: Optional[Iterable[Any]], target_readers: Iterable[Any], buckets: List[Tuple[int, int]], length_ratio_mean: float, length_ratio_std: float, source_vocabs: Optional[List[Dict[str, int]]], target_vocabs: List[Dict[str, int]]) -> "DataStatistics":
    data_stats_accumulator = DataStatisticsAccumulator(buckets, source_vocabs[0] if source_vocabs is not None else None, target_vocabs[0], length_ratio_mean, length_ratio_std)
    if source_readers is not None:
        for sources, targets in parallel_iter(source_readers, target_readers):
            buck_idx, _ = get_parallel_bucket(buckets, len(sources[0]), len(targets[0]))
            data_stats_accumulator.sequence_pair(sources[0], targets[0], buck_idx)
    else:
        for targets in target_readers:
            buck_idx, _ = get_target_bucket(buckets, len(targets[0]))
            data_stats_accumulator.sequence_pair([], targets[0], buck_idx)
    return data_stats_accumulator.statistics


def get_validation_data_iter(data_loader: RawParallelDatasetLoader,
                             validation_sources: List[str],
                             validation_targets: List[str],
                             buckets: List[Tuple[int, int]],
                             bucket_batch_sizes: List[BucketBatchSize],
                             source_vocabs: List[Dict[str, int]],
                             target_vocabs: List[Dict[str, int]],
                             max_seq_len_source: int,
                             max_seq_len_target: int,
                             batch_size: int,
                             permute: bool = False) -> "ParallelSampleIter":
    """
    Returns a ParallelSampleIter for the validation data.
    """
    logger.info('=================================')
    logger.info('Creating validation data iterator')
    logger.info('=================================')
    validation_length_statistics = analyze_sequence_lengths(validation_sources, validation_targets, source_vocabs, target_vocabs, max_seq_len_source, max_seq_len_target)
    check_condition(validation_length_statistics.num_sents > 0, 'No validation sequences found with length smaller or equal than the maximum sequence length.Consider increasing %s' % C.TRAINING_ARG_MAX_SEQ_LEN)
    validation_sources_sentences, validation_targets_sentences = create_sequence_readers(validation_sources, validation_targets, source_vocabs, target_vocabs)
    validation_data_statistics = get_data_statistics(validation_sources_sentences, validation_targets_sentences, buckets, validation_length_statistics.length_ratio_mean, validation_length_statistics.length_ratio_std, source_vocabs, target_vocabs)
    validation_data_statistics.log(bucket_batch_sizes)
    validation_data = data_loader.load(validation_sources_sentences, validation_targets_sentences, validation_data_statistics.num_sents_per_bucket).fill_up(bucket_batch_sizes)
    return ParallelSampleIter(data=validation_data,
                              buckets=buckets,
                              batch_size=batch_size,
                              bucket_batch_sizes=bucket_batch_sizes,
                              num_source_factors=len(validation_sources),
                              num_target_factors=len(validation_targets),
                              permute=permute)


def get_prepared_data_iters(prepared_data_dir: str,
                            validation_sources: List[str],
                            validation_targets: List[str],
                            shared_vocab: bool,
                            batch_size: int,
                            batch_type: str,
                            batch_sentences_multiple_of: int = 1,
                            permute: bool = True) -> Tuple["ShardedParallelSampleIter", "ParallelSampleIter", "DataConfig", List[Dict[str, int]], List[Dict[str, int]]]:
    logger.info('===============================')
    logger.info('Creating training data iterator')
    logger.info('===============================')
    version_file: str = os.path.join(prepared_data_dir, C.PREPARED_DATA_VERSION_FILE)
    with open(version_file) as version_in:
        version: int = int(version_in.read())
        check_condition(version in (C.PREPARED_DATA_VERSION, C.PREPARED_DATA_LEGACY_VERSION), 'The dataset %s was written in an incompatible format. Please rerun data preparation with this version of Sockeye.' % prepared_data_dir)
    info_file: str = os.path.join(prepared_data_dir, C.DATA_INFO)
    check_condition(os.path.exists(info_file), 'Could not find data info %s. Are you sure %s is a directory created with sockeye-prepare-data?' % (info_file, prepared_data_dir))
    data_info: DataInfo = cast(DataInfo, DataInfo.load(info_file))
    config_file: str = os.path.join(prepared_data_dir, C.DATA_CONFIG)
    check_condition(os.path.exists(config_file), 'Could not find data config %s. Are you sure %s is a directory created with sockeye-prepare-data?' % (config_file, prepared_data_dir))
    config_data: DataConfig = cast(DataConfig, DataConfig.load(config_file))
    shard_fnames: List[str] = [os.path.join(prepared_data_dir, C.SHARD_NAME % shard_idx) for shard_idx in range(data_info.num_shards)]
    for shard_fname in shard_fnames:
        check_condition(os.path.exists(shard_fname), 'Shard %s does not exist.' % shard_fname)
    check_condition(shared_vocab == data_info.shared_vocab, 'Shared vocabulary settings need to match these of the prepared data (e.g. for weight tying). Specify or omit %s consistently when training and preparing the data.' % C.VOCAB_ARG_SHARED_VOCAB)
    source_vocabs: List[Dict[str, int]] = vocab.load_source_vocabs(prepared_data_dir)
    target_vocabs: List[Dict[str, int]] = vocab.load_target_vocabs(prepared_data_dir)
    check_condition(len(source_vocabs) == len(data_info.sources), 'Wrong number of source vocabularies. Found %d, need %d.' % (len(source_vocabs), len(data_info.sources)))
    check_condition(len(target_vocabs) == len(data_info.targets), 'Wrong number of target vocabularies. Found %d, need %d.' % (len(target_vocabs), len(data_info.targets)))
    buckets: List[Tuple[int, int]] = config_data.data_statistics.buckets
    max_seq_len_source = config_data.max_seq_len_source
    max_seq_len_target = config_data.max_seq_len_target
    bucket_batch_sizes: List[BucketBatchSize] = define_bucket_batch_sizes(buckets, batch_size, batch_type, config_data.data_statistics.average_len_target_per_bucket, batch_sentences_multiple_of)
    config_data.data_statistics.log(bucket_batch_sizes)
    train_iter: ShardedParallelSampleIter = ShardedParallelSampleIter(shard_fnames, buckets, batch_size, bucket_batch_sizes,
                                                                      num_source_factors=len(data_info.sources),
                                                                      num_target_factors=len(data_info.targets),
                                                                      permute=permute)
    data_loader: RawParallelDatasetLoader = RawParallelDatasetLoader(buckets=buckets, eos_id=C.EOS_ID, pad_id=C.PAD_ID, eop_id=config_data.eop_id)
    validation_iter: ParallelSampleIter = get_validation_data_iter(data_loader=data_loader,
                                                                   validation_sources=validation_sources,
                                                                   validation_targets=validation_targets,
                                                                   buckets=buckets,
                                                                   bucket_batch_sizes=bucket_batch_sizes,
                                                                   source_vocabs=source_vocabs,
                                                                   target_vocabs=target_vocabs,
                                                                   max_seq_len_source=max_seq_len_source,
                                                                   max_seq_len_target=max_seq_len_target,
                                                                   batch_size=batch_size,
                                                                   permute=False)
    return (train_iter, validation_iter, config_data, source_vocabs, target_vocabs)


def get_training_data_iters(sources: List[str],
                            targets: List[str],
                            validation_sources: List[str],
                            validation_targets: List[str],
                            source_vocabs: List[Dict[str, int]],
                            target_vocabs: List[Dict[str, int]],
                            source_vocab_paths: List[str],
                            target_vocab_paths: List[str],
                            shared_vocab: bool,
                            batch_size: int,
                            batch_type: str,
                            max_seq_len_source: int,
                            max_seq_len_target: int,
                            bucketing: bool,
                            bucket_width: int,
                            bucket_scaling: bool = True,
                            end_of_prepending_tag: Optional[str] = None,
                            allow_empty: bool = False,
                            batch_sentences_multiple_of: int = 1,
                            permute: bool = True) -> Tuple["ParallelSampleIter", "ParallelSampleIter", DataConfig, DataInfo]:
    """
    Returns data iterators for training and validation data.

    :param sources: Path to source training data (with optional factor data paths).
    :param targets: Path to target training data (with optional factor data paths).
    :param validation_sources: Path to source validation data (with optional factor data paths).
    :param validation_targets: Path to target validation data (with optional factor data paths).
    :param source_vocabs: Source vocabulary and optional factor vocabularies.
    :param target_vocabs: Target vocabulary and optional factor vocabularies.
    :param source_vocab_paths: Path to source vocabularies.
    :param target_vocab_paths: Path to target vocabularies.
    :param shared_vocab: Whether the vocabularies are shared.
    :param batch_size: Batch size.
    :param batch_type: Method for sizing batches.
    :param max_seq_len_source: Maximum source sequence length.
    :param max_seq_len_target: Maximum target sequence length.
    :param bucketing: Whether to use bucketing.
    :param bucket_width: Size of buckets.
    :param bucket_scaling: Scale bucket steps based on source/target length ratio.
    :param end_of_prepending_tag: Tag indicating the end of prepended text.
    :param allow_empty: Unless True if no sentences are below or equal to the maximum length an exception is raised.
    :param batch_sentences_multiple_of: Round the number of sentences in each bucket's batch to a multiple of this value (word-based batching only).
    :param permute: Randomly shuffle the parallel data.
    :return: Tuple of (training data iterator, validation data iterator, data config).
    """
    logger.info('===============================')
    logger.info('Creating training data iterator')
    logger.info('===============================')
    length_statistics = analyze_sequence_lengths(sources, targets, source_vocabs, target_vocabs, max_seq_len_source, max_seq_len_target)
    if not allow_empty:
        check_condition(length_statistics.num_sents > 0, 'No training sequences found with length smaller or equal than the maximum sequence length.Consider increasing %s' % C.TRAINING_ARG_MAX_SEQ_LEN)
    buckets: List[Tuple[int, int]] = (define_parallel_buckets(max_seq_len_source, max_seq_len_target, bucket_width, bucket_scaling, length_statistics.length_ratio_mean)
                                      if bucketing else [(max_seq_len_source, max_seq_len_target)])
    sources_sentences, targets_sentences = create_sequence_readers(sources, targets, source_vocabs, target_vocabs)
    data_statistics = get_data_statistics(sources_sentences, targets_sentences, buckets, length_statistics.length_ratio_mean, length_statistics.length_ratio_std, source_vocabs, target_vocabs)
    bucket_batch_sizes = define_bucket_batch_sizes(buckets, batch_size, batch_type, data_statistics.average_len_target_per_bucket, batch_sentences_multiple_of)
    data_statistics.log(bucket_batch_sizes)
    eop_id: int = get_eop_id(source_vocabs[0], end_of_prepending_tag)
    data_loader: RawParallelDatasetLoader = RawParallelDatasetLoader(buckets=buckets, eos_id=C.EOS_ID, pad_id=C.PAD_ID, eop_id=eop_id)
    training_data = data_loader.load(sources_sentences, targets_sentences, data_statistics.num_sents_per_bucket).fill_up(bucket_batch_sizes)
    data_info = DataInfo(sources=sources, targets=targets, source_vocabs=source_vocab_paths, target_vocabs=target_vocab_paths, shared_vocab=shared_vocab, num_shards=1)
    config_data = DataConfig(data_statistics=data_statistics,
                             max_seq_len_source=max_seq_len_source,
                             max_seq_len_target=max_seq_len_target,
                             num_source_factors=len(sources),
                             num_target_factors=len(targets),
                             eop_id=eop_id)
    train_iter = ParallelSampleIter(data=training_data,
                                    buckets=buckets,
                                    batch_size=batch_size,
                                    bucket_batch_sizes=bucket_batch_sizes,
                                    num_source_factors=len(sources),
                                    num_target_factors=len(targets),
                                    permute=permute)
    validation_iter = get_validation_data_iter(data_loader=data_loader,
                                               validation_sources=validation_sources,
                                               validation_targets=validation_targets,
                                               buckets=buckets,
                                               bucket_batch_sizes=bucket_batch_sizes,
                                               source_vocabs=source_vocabs,
                                               target_vocabs=target_vocabs,
                                               max_seq_len_source=max_seq_len_source,
                                               max_seq_len_target=max_seq_len_target,
                                               batch_size=batch_size,
                                               permute=False)
    return (train_iter, validation_iter, config_data, data_info)


def get_scoring_data_iters(sources: List[str],
                           targets: List[str],
                           source_vocabs: List[Dict[str, int]],
                           target_vocabs: List[Dict[str, int]],
                           batch_size: int,
                           max_seq_len_source: int,
                           max_seq_len_target: int,
                           eop_id: int = C.INVALID_ID) -> "BatchedRawParallelSampleIter":
    """
    Returns a data iterator for scoring. The iterator loads data on demand,
    batch by batch, and does not skip any lines. Lines that are too long
    are truncated.

    :param sources: Path to source training data (with optional factor data paths).
    :param targets: Path to target training data (with optional factor data paths).
    :param source_vocabs: Source vocabulary and optional factor vocabularies.
    :param target_vocabs: Target vocabulary and optional factor vocabularies.
    :param batch_size: Batch size.
    :param max_seq_len_source: Maximum source sequence length.
    :param max_seq_len_target: Maximum target sequence length.
    :param eop_id: End-of-prepending tag id.
    :return: The scoring data iterator.
    """
    logger.info('==============================')
    logger.info('Creating scoring data iterator')
    logger.info('==============================')
    bucket: Tuple[int, int] = (max_seq_len_source, max_seq_len_target)
    data_loader: RawParallelDatasetLoader = RawParallelDatasetLoader(buckets=[bucket], eos_id=C.EOS_ID, pad_id=C.PAD_ID, eop_id=eop_id, skip_blanks=False)
    scoring_iter: BatchedRawParallelSampleIter = BatchedRawParallelSampleIter(data_loader=data_loader,
                                                                             sources=sources,
                                                                             targets=targets,
                                                                             source_vocabs=source_vocabs,
                                                                             target_vocabs=target_vocabs,
                                                                             bucket=bucket,
                                                                             batch_size=batch_size,
                                                                             max_lens=(max_seq_len_source, max_seq_len_target),
                                                                             num_source_factors=len(sources),
                                                                             num_target_factors=len(targets))
    return scoring_iter


@dataclass
class LengthStatistics(config.Config):
    num_sents: int
    length_ratio_mean: float
    length_ratio_std: float


@dataclass
class DataStatistics(config.Config):
    # Other attributes are defined dynamically in __init__ via the properties.
    length_ratio_stats_per_bucket: Any = None

    def log(self, bucket_batch_sizes: Optional[List[BucketBatchSize]] = None) -> None:
        logger.info('Tokens: source %d target %d', self.num_tokens_source, self.num_tokens_target)
        logger.info('Number of <unk> tokens: source %d target %d', self.num_unks_source, self.num_unks_target)
        if self.num_tokens_source > 0 and self.num_tokens_target > 0:
            logger.info('Vocabulary coverage: source %.0f%% target %.0f%%', (1 - self.num_unks_source / self.num_tokens_source) * 100, (1 - self.num_unks_target / self.num_tokens_target) * 100)
        logger.info('%d sequences across %d buckets', self.num_sents, len(self.num_sents_per_bucket))
        logger.info('%d sequences did not fit into buckets and were discarded', self.num_discarded)
        if bucket_batch_sizes is not None:
            describe_data_and_buckets(self, bucket_batch_sizes)


def describe_data_and_buckets(data_statistics: DataStatistics, bucket_batch_sizes: List[BucketBatchSize]) -> None:
    """
    Describes statistics across buckets
    """
    check_condition(len(bucket_batch_sizes) == len(data_statistics.buckets), 'Number of bucket batch sizes (%d) does not match number of buckets in statistics (%d).' % (len(bucket_batch_sizes), len(data_statistics.buckets)))
    for bucket_batch_size, num_seq, (lr_mean, lr_std) in zip(bucket_batch_sizes, data_statistics.num_sents_per_bucket, data_statistics.length_ratio_stats_per_bucket):
        if num_seq > 0:
            logger.info('Bucket %s: %d samples in %d batches of %d, ~%.1f target tokens/batch, trg/src length ratio: %.2f (+-%.2f)',
                        bucket_batch_size.bucket,
                        num_seq,
                        math.ceil(num_seq / bucket_batch_size.batch_size),
                        bucket_batch_size.batch_size,
                        bucket_batch_size.batch_size_word,
                        lr_mean,
                        lr_std)


@dataclass
class DataInfo(config.Config):
    sources: List[str] = None  # type: ignore
    targets: List[str] = None  # type: ignore
    source_vocabs: List[str] = None  # type: ignore
    target_vocabs: List[str] = None  # type: ignore
    shared_vocab: bool = False
    num_shards: int = 0

    @classmethod
    def load(cls, fname: str) -> "DataInfo":
        return cast(DataInfo, torch.load(fname))

    def save(self, fname: str) -> None:
        torch.save(self, fname)


@dataclass
class DataConfig(config.Config):
    data_statistics: DataStatistics = None  # type: ignore
    max_seq_len_source: int = 0
    max_seq_len_target: int = 0
    num_source_factors: int = 0
    num_target_factors: int = 0
    eop_id: int = C.INVALID_ID

    @classmethod
    def load(cls, fname: str) -> "DataConfig":
        return cast(DataConfig, torch.load(fname))

    def save(self, fname: str) -> None:
        torch.save(self, fname)


def read_content(path: str, limit: Optional[int] = None) -> Iterator[List[str]]:
    """
    Returns a list of tokens for each line in path up to a limit.

    :param path: Path to files containing sentences.
    :param limit: How many lines to read from path.
    :return: Iterator over lists of words.
    """
    with smart_open(path) as indata:
        for i, line in enumerate(indata):
            if limit is not None and i == limit:
                break
            yield list(get_tokens(line))


def tokens2ids(tokens: List[str], vocab: Dict[str, int]) -> List[int]:
    """
    Returns sequence of integer ids given a sequence of tokens and vocab.

    :param tokens: List of string tokens.
    :param vocab: Vocabulary (containing UNK symbol).
    :return: List of word ids.
    """
    return [vocab.get(w, vocab[C.UNK_SYMBOL]) for w in tokens]


def strids2ids(tokens: List[str]) -> List[int]:
    """
    Returns sequence of integer ids given a sequence of string ids.

    :param tokens: List of integer tokens.
    :return: List of word ids.
    """
    return list(map(int, tokens))


def ids2tokens(token_ids: List[int], vocab_inv: Dict[int, str], exclude_set: Set[int]) -> Generator[str, None, None]:
    """
    Transforms a list of token IDs into a list of words, excluding any IDs in `exclude_set`.

    :param token_ids: The list of token IDs.
    :param vocab_inv: The inverse vocabulary.
    :param exclude_set: The list of token IDs to exclude.
    :return: The list of words.
    """
    tokens = (vocab_inv[token] for token in token_ids)
    return (tok for token_id, tok in zip(token_ids, tokens) if token_id not in exclude_set)


class SequenceReader:
    """
    Reads sequence samples from path and (optionally) creates integer id sequences.
    Streams from disk, instead of loading all samples into memory.
    If vocab is None, the sequences in path are assumed to be integers coded as strings.
    Empty sequences are yielded as None.

    :param path: Path to read data from.
    :param vocabulary: Optional mapping from strings to integer ids.
    :param add_bos: Whether to add Beginning-Of-Sentence (BOS) symbol.
    :param limit: Read limit.
    """
    def __init__(self, path: str, vocabulary: Optional[Dict[str, int]] = None, add_bos: bool = False, add_eos: bool = False, limit: Optional[int] = None) -> None:
        self.path: str = path
        self.vocab: Optional[Dict[str, int]] = vocabulary
        self.bos_id: Optional[int] = None
        self.eos_id: Optional[int] = None
        if vocabulary is not None:
            assert vocab.is_valid_vocab(vocabulary)
            self.bos_id = C.BOS_ID
            self.eos_id = C.EOS_ID
        else:
            check_condition(not add_bos and (not add_eos), 'Adding a BOS or EOS symbol requires a vocabulary')
        self.add_bos: bool = add_bos
        self.add_eos: bool = add_eos
        self.limit: Optional[int] = limit

    def __iter__(self) -> Iterator[Optional[List[int]]]:
        for tokens in read_content(self.path, self.limit):
            if self.vocab is not None:
                sequence: List[int] = tokens2ids(tokens, self.vocab)
            else:
                sequence = strids2ids(tokens)
            if len(sequence) == 0:
                yield None
                continue
            if self.add_bos:
                sequence.insert(0, self.bos_id)  # type: ignore
            if self.add_eos:
                sequence.append(self.eos_id)  # type: ignore
            yield sequence


def create_sequence_readers(sources: List[str], targets: List[str], vocab_sources: List[Dict[str, int]], vocab_targets: List[Dict[str, int]]) -> Tuple[List[SequenceReader], List[SequenceReader]]:
    """
    Create source readers with EOS and target readers with BOS.

    :param sources: The file names of source data and factors.
    :param targets: The file name of the target data and factors.
    :param vocab_sources: The source vocabularies.
    :param vocab_targets: The target vocabularies.
    :return: The source sequence readers and the target reader.
    """
    source_sequence_readers: List[SequenceReader] = [SequenceReader(source, vocab, add_eos=True) for source, vocab in zip(sources, vocab_sources)]
    target_sequence_readers: List[SequenceReader] = [SequenceReader(target, vocab, add_bos=True) for target, vocab in zip(targets, vocab_targets)]
    return (source_sequence_readers, target_sequence_readers)


def parallel_iter(source_iterables: Iterable[Any], target_iterables: Iterable[Any], skip_blanks: bool = True, check_token_parallel: bool = True) -> Iterator[Tuple[List[Any], List[Any]]]:
    """
    Creates iterators over parallel iterables by calling iter() on the iterables
    and chaining to parallel_iterate(). The purpose of the separation is to allow
    the caller to save iterator state between calls, if desired.

    :param source_iterables: A list of source iterables.
    :param target_iterables: A target iterable.
    :param skip_blanks: Whether to skip empty target lines.
    :param check_token_parallel: Whether to check if the tokens are parallel or not.
    :return: Iterators over sources and target.
    """
    source_iterators: List[Iterator[Any]] = [iter(s) for s in source_iterables]
    target_iterators: List[Iterator[Any]] = [iter(t) for t in target_iterables]
    return parallel_iterate(source_iterators, target_iterators, skip_blanks, check_token_parallel)


def parallel_iterate(source_iterators: List[Iterator[Any]], target_iterators: List[Iterator[Any]], skip_blanks: bool = True, check_token_parallel: bool = True) -> Iterator[Tuple[List[Any], List[Any]]]:
    """
    Yields parallel source(s), target sequences from iterables.
    Checks for token parallelism in source sequences.
    Skips pairs where element in at least one iterable is None.
    Checks that all iterables have the same number of elements.
    Can optionally continue from an already-begun iterator.

    :param source_iterators: A list of source iterators.
    :param target_iterators: A list of target iterators.
    :param skip_blanks: Whether to skip empty target lines.
    :param check_token_parallel: Whether to check if the tokens are parallel or not.
    :return: Iterators over sources and target.
    """
    num_skipped: int = 0
    while True:
        try:
            sources: List[Any] = [next(source_iter) for source_iter in source_iterators]
            targets: List[Any] = [next(target_iter) for target_iter in target_iterators]
        except StopIteration:
            break
        if skip_blanks and (any((s is None for s in sources)) or any((t is None for t in targets))):
            num_skipped += 1
            continue
        if check_token_parallel:
            check_condition(are_none(sources) or are_token_parallel(sources), 'Source sequences are not token-parallel: %s' % str(sources))
            check_condition(are_none(targets) or are_token_parallel(targets), 'Target sequences are not token-parallel: %s' % str(targets))
        yield (sources, targets)
    if num_skipped > 0:
        logger.warning('Parallel reading of sequences skipped %d elements', num_skipped)
    check_condition(all((next(cast(Iterator[Any], s), None) is None for s in source_iterators)) and all((next(cast(Iterator[Any], t), None) is None for t in target_iterators)), 'Different number of lines in source(s) and target(s) iterables.')


def get_parallel_bucket(buckets: List[Tuple[int, int]], length_source: int, length_target: int) -> Tuple[Optional[int], Optional[Tuple[int, int]]]:
    """
    Returns bucket index and bucket from a list of buckets, given source and target length.
    Algorithm assumes buckets are sorted from shortest to longest.
    Returns (None, None) if no bucket fits.

    :param buckets: List of buckets, in sorted order, shortest to longest.
    :param length_source: Length of source sequence.
    :param length_target: Length of target sequence.
    :return: Tuple of (bucket index, bucket), or (None, None) if not fitting.
    """
    for j, (source_bkt, target_bkt) in enumerate(buckets):
        if source_bkt >= length_source and target_bkt >= length_target:
            return (j, (source_bkt, target_bkt))
    return (None, None)


def get_target_bucket(buckets: List[Tuple[int, int]], length_target: int) -> Tuple[Optional[int], Optional[Tuple[int, int]]]:
    """
    Returns bucket index and bucket from a list of buckets, given source and target length.
    Returns (None, None) if no bucket fits.

    :param buckets: List of buckets.
    :param length_target: Length of target sequence.
    :return: Tuple of (bucket index, bucket), or (None, None) if not fitting.
    """
    bucket_idx: Optional[int] = None
    bucket: Optional[Tuple[int, int]] = None
    for j, (source_bkt, target_bkt) in enumerate(buckets):
        if target_bkt >= length_target:
            bucket_idx, bucket = (j, (source_bkt, target_bkt))
            break
    return (bucket_idx, bucket)


class ParallelDataSet:
    """
    Bucketed parallel data set
    """
    def __init__(self, source: List[torch.Tensor], target: List[torch.Tensor], prepended_source_length: Optional[List[torch.Tensor]] = None) -> None:
        check_condition(len(source) == len(target), 'Number of buckets for source/target do not match: %d/%d.' % (len(source), len(target)))
        if prepended_source_length is not None:
            check_condition(len(source) == len(prepended_source_length), 'Number of buckets for source/prepended_source_length do not match: %d/%d.' % (len(source), len(prepended_source_length)))
        self.source: List[torch.Tensor] = source
        self.target: List[torch.Tensor] = target
        self.prepended_source_length: Optional[List[torch.Tensor]] = prepended_source_length

    def __len__(self) -> int:
        return len(self.source)

    def get_bucket_counts(self) -> List[int]:
        return [self.source[buck_idx].shape[0] for buck_idx in range(len(self))]

    def save(self, fname: str, use_legacy_format: bool = False) -> None:
        """
        Saves the dataset to disk using PyTorch's pickle/zip format. The option
        to use the legacy data format is included for testing backward
        compatibility and is not recommended for general use.
        """
        if use_legacy_format:
            check_condition(self.prepended_source_length is None, 'Cannot save data in legacy format when specifying prepended tokens')
            torch.save(self.source + self.target, fname)
        else:
            data_dict: Dict[str, Any] = {C.DATA_KEY_SOURCE: self.source, C.DATA_KEY_TARGET: self.target}
            if self.prepended_source_length is not None:
                data_dict[C.DATA_KEY_PREPENDED_SOURCE_LENGTH] = self.prepended_source_length
            torch.save(data_dict, fname)

    @staticmethod
    def load(fname: str) -> "ParallelDataSet":
        """
        Loads a dataset from a binary .npy file. When running in distributed
        mode, the data is sliced and each worker loads a different slice based
        on its rank.
        """
        data: Any = torch.load(fname)
        if isinstance(data, list):
            n: int = len(data) // 2
            source: List[torch.Tensor] = data[:n]
            target: List[torch.Tensor] = data[n:2 * n]
            prepended_source_length: Optional[List[torch.Tensor]] = None
        else:
            check_condition(isinstance(data, dict) and C.DATA_KEY_SOURCE in data and (C.DATA_KEY_TARGET in data), f'Unknown data format for file {fname}. Please rerun data preparation with this version of Sockeye.')
            source = data[C.DATA_KEY_SOURCE]
            target = data[C.DATA_KEY_TARGET]
            prepended_source_length = data.get(C.DATA_KEY_PREPENDED_SOURCE_LENGTH, None)
        if utils.is_distributed():
            split_index: int = torch.distributed.get_rank()
            total_splits: int = torch.distributed.get_world_size()
            i: float = split_index / total_splits
            j: float = (split_index + 1) / total_splits
            for k in range(len(source)):
                num_sentences: int = len(source[k])
                if num_sentences > 0:
                    num_copies: int = math.ceil(total_splits / num_sentences)
                    if num_copies > 1:
                        logger.info('Replicating bucket of %d sentence(s) %d times to cover %d splits.', num_sentences, num_copies, total_splits)
                        source[k] = torch.repeat_interleave(source[k], repeats=num_copies, dim=0)
                        target[k] = torch.repeat_interleave(target[k], repeats=num_copies, dim=0)
                        if prepended_source_length is not None:
                            prepended_source_length[k] = torch.repeat_interleave(prepended_source_length[k], repeats=num_copies, dim=0)
            source = [s[math.floor(i * s.shape[0]):math.floor(j * s.shape[0])] if s.shape[0] > 0 else s for s in source]
            target = [t[math.floor(i * t.shape[0]):math.floor(j * t.shape[0])] if t.shape[0] > 0 else t for t in target]
            if prepended_source_length is not None:
                prepended_source_length = [l[math.floor(i * l.shape[0]):math.floor(j * l.shape[0])] if l.shape[0] > 0 else l for l in prepended_source_length]
        assert len(source) == len(target)
        if prepended_source_length is not None:
            assert len(source) == len(prepended_source_length)
        return ParallelDataSet(source, target, prepended_source_length)

    def fill_up(self, bucket_batch_sizes: List[BucketBatchSize], seed: int = 42) -> "ParallelDataSet":
        """
        Returns a new dataset with buckets filled up.

        :param bucket_batch_sizes: Bucket batch sizes.
        :param seed: The random seed used for sampling sentences to fill up.
        :return: New dataset with buckets filled up to the next multiple of batch size
        """
        source = list(self.source)
        target = list(self.target)
        prepended_source_length = list(self.prepended_source_length) if self.prepended_source_length is not None else None
        rs = np.random.RandomState(seed)
        for bucket_idx in range(len(self)):
            bucket_batch_size: int = bucket_batch_sizes[bucket_idx].batch_size
            bucket_source: torch.Tensor = self.source[bucket_idx]
            bucket_target: torch.Tensor = self.target[bucket_idx]
            bucket_prepended_source_length: Optional[torch.Tensor] = self.prepended_source_length[bucket_idx] if self.prepended_source_length is not None else None
            num_samples: int = bucket_source.shape[0]
            target_num_samples: int = num_samples
            if num_samples % bucket_batch_size != 0:
                target_num_samples = num_samples + (bucket_batch_size - num_samples % bucket_batch_size)
            if utils.is_distributed():
                target_num_samples = max(utils.all_gather_object(target_num_samples))
            rest: int = target_num_samples - num_samples
            if rest > 0:
                desired_indices = torch.tensor(rs.randint(num_samples, size=rest))
                source[bucket_idx] = torch.cat((bucket_source, torch.index_select(bucket_source, 0, desired_indices)), dim=0)
                target[bucket_idx] = torch.cat((bucket_target, torch.index_select(bucket_target, 0, desired_indices)), dim=0)
                if prepended_source_length is not None:
                    assert bucket_prepended_source_length is not None
                    prepended_source_length[bucket_idx] = torch.cat((bucket_prepended_source_length, torch.index_select(bucket_prepended_source_length, 0, desired_indices)), dim=0)
        return ParallelDataSet(source, target, prepended_source_length)

    def permute(self, permutations: List[torch.Tensor]) -> "ParallelDataSet":
        """
        Permutes the data within each bucket. The permutation is received as an argument,
        allowing the data to be unpermuted (i.e., restored) later on.

        :param permutations: For each bucket, a permutation of the data within that bucket.
        :return: A new, permuted ParallelDataSet.
        """
        assert len(self) == len(permutations)
        source: List[torch.Tensor] = []
        target: List[torch.Tensor] = []
        prepended_source_length: Optional[List[torch.Tensor]] = [] if self.prepended_source_length is not None else None
        for buck_idx in range(len(self)):
            num_samples: int = self.source[buck_idx].shape[0]
            if num_samples:
                permutation: torch.Tensor = permutations[buck_idx]
                source.append(torch.index_select(self.source[buck_idx], 0, permutation))
                target.append(torch.index_select(self.target[buck_idx], 0, permutation))
                if self.prepended_source_length is not None:
                    assert prepended_source_length is not None
                    prepended_source_length.append(torch.index_select(self.prepended_source_length[buck_idx], 0, permutation))
            else:
                source.append(self.source[buck_idx])
                target.append(self.target[buck_idx])
                if self.prepended_source_length is not None:
                    assert prepended_source_length is not None
                    prepended_source_length.append(self.prepended_source_length[buck_idx])
        return ParallelDataSet(source, target, prepended_source_length)


def get_permutations(bucket_counts: List[int]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Returns the indices of a random permutation for each bucket and the corresponding inverse permutations that can
    restore the original order of the data if applied to the permuted data.

    :param bucket_counts: The number of elements per bucket.
    :return: For each bucket a permutation and inverse permutation is returned.
    """
    data_permutations: List[torch.Tensor] = []
    inverse_data_permutations: List[torch.Tensor] = []
    for num_samples in bucket_counts:
        if num_samples == 0:
            num_samples = 1
        data_permutation = torch.from_numpy(np.random.permutation(num_samples)).long()
        inverse_data_permutation = torch.empty(num_samples, dtype=torch.int64)
        inverse_data_permutation[data_permutation] = torch.arange(num_samples)
        data_permutations.append(data_permutation)
        inverse_data_permutations.append(inverse_data_permutation)
    return (data_permutations, inverse_data_permutations)


def get_batch_indices(data: ParallelDataSet, bucket_batch_sizes: List[BucketBatchSize]) -> List[Tuple[int, int]]:
    """
    Returns a list of index tuples that index into the bucket and the start index inside a bucket given
    the batch size for a bucket. These indices are valid for the given dataset.
    """
    idxs: List[Tuple[int, int]] = []
    for buck_idx, _ in enumerate(data.source):
        bucket: Tuple[int, int] = bucket_batch_sizes[buck_idx].bucket
        batch_size: int = bucket_batch_sizes[buck_idx].batch_size
        num_samples: int = data.source[buck_idx].shape[0]
        rest: int = num_samples % batch_size
        if rest > 0:
            logger.info('Ignoring %d samples from bucket %s with %d samples due to incomplete batch', rest, bucket, num_samples)
        idxs.extend([(buck_idx, j) for j in range(0, num_samples - batch_size + 1, batch_size)])
    return idxs


class BaseParallelSampleIter:
    """
    Base parallel sample iterator.

    :param buckets: The list of buckets.
    :param bucket_batch_sizes: A list, parallel to `buckets`, containing the number of samples in each bucket.
    :param num_source_factors: The number of source factors.
    :param num_target_factors: The number of target factors.
    :param permute: Randomly shuffle the parallel data.
    :param dtype: The data type.
    """
    def __init__(self, buckets: List[Tuple[int, int]], batch_size: int, bucket_batch_sizes: List[BucketBatchSize], num_source_factors: int = 1, num_target_factors: int = 1, permute: bool = True, dtype: str = 'int32') -> None:
        self.batch_size: int = batch_size
        self.buckets: List[Tuple[int, int]] = list(buckets)
        self.bucket_batch_sizes: List[BucketBatchSize] = bucket_batch_sizes
        self.num_source_factors: int = num_source_factors
        self.num_target_factors: int = num_target_factors
        self.permute: bool = permute
        self.dtype: str = dtype

    def __iter__(self) -> "BaseParallelSampleIter":
        return self

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def iter_next(self) -> bool:
        pass

    @abstractmethod
    def next(self) -> Any:
        pass

    def __next__(self) -> Any:
        return self.next()

    @abstractmethod
    def save_state(self, fname: str) -> None:
        pass

    @abstractmethod
    def load_state(self, fname: str) -> None:
        pass


class BatchedRawParallelSampleIter(BaseParallelSampleIter):
    """
    Goes through the raw data, loading only one batch at a time into memory.
    Used by the scorer. Iterates through the data in order, and therefore does
    not support bucketing.
    """
    def __init__(self, data_loader: RawParallelDatasetLoader, sources: List[str], targets: List[str], source_vocabs: List[Dict[str, int]], target_vocabs: List[Dict[str, int]], bucket: Tuple[int, int], batch_size: int, max_lens: Tuple[int, int], num_source_factors: int = 1, num_target_factors: int = 1, dtype: str = 'int32') -> None:
        super().__init__(buckets=[bucket], batch_size=batch_size, bucket_batch_sizes=[BucketBatchSize(bucket, batch_size, 0)], num_source_factors=num_source_factors, num_target_factors=num_target_factors, permute=False, dtype=dtype)
        self.data_loader: RawParallelDatasetLoader = data_loader
        self.sources_sentences, self.targets_sentences = create_sequence_readers(sources, targets, source_vocabs, target_vocabs)
        self.sources_iters: List[Iterator[Any]] = [iter(s) for s in self.sources_sentences]
        self.targets_iters: List[Iterator[Any]] = [iter(s) for s in self.targets_sentences]
        self.max_len_source, self.max_len_target = max_lens
        self.next_batch: Optional[Any] = None
        self.sentno: int = 1

    def reset(self) -> None:
        raise Exception('Not supported!')

    def iter_next(self) -> bool:
        sources_sentences: List[List[Any]] = [[] for _ in self.sources_sentences]
        targets_sentences: List[List[Any]] = [[] for _ in self.targets_sentences]
        num_read: int = 0
        for num_read, (sources, targets) in enumerate(parallel_iterate(self.sources_iters, self.targets_iters, skip_blanks=False), 1):
            source_len: int = 0 if sources[0] is None else len(sources[0])
            target_len: int = 0 if targets[0] is None else len(targets[0])
            if source_len > self.max_len_source:
                logger.debug('Trimming source sentence {} ({} -> {})'.format(self.sentno + num_read, source_len, self.max_len_source))
                sources = [source[0:self.max_len_source] for source in sources]
            if target_len > self.max_len_target:
                logger.debug('Trimming target sentence {} ({} -> {})'.format(self.sentno + num_read, target_len, self.max_len_target))
                targets = [target[0:self.max_len_target] for target in targets]
            for i, source in enumerate(sources):
                sources_sentences[i].append(source)
            for i, target in enumerate(targets):
                targets_sentences[i].append(target)
            if num_read == self.batch_size:
                break
        aux: int = int(self.sentno / 1000000)
        self.sentno += num_read
        if int(self.sentno / 1000000) != aux:
            logger.info('Processed {} lines'.format(self.sentno))
        if num_read == 0:
            self.next_batch = None
            return False
        dataset = self.data_loader.load(sources_sentences, targets_sentences, [num_read])
        source = dataset.source[0]
        target, label = create_target_and_shifted_label_sequences(dataset.target[0])
        prepended_source_length = dataset.prepended_source_length[0] if dataset.prepended_source_length is not None else None
        self.next_batch = create_batch_from_parallel_sample(source, target, label, prepended_source_length)
        return True

    def next(self) -> Any:
        if self.iter_next():
            return self.next_batch
        raise StopIteration

    def save_state(self, fname: str) -> None:
        raise NotImplementedError('Not supported!')

    def load_state(self, fname: str) -> None:
        raise NotImplementedError('Not supported!')


class ShardedParallelSampleIter(BaseParallelSampleIter):
    """
    Goes through the data one shard at a time. The memory consumption is limited by the memory consumption of the
    largest shard. The order in which shards are traversed is changed with each reset.
    """
    def __init__(self, shards_fnames: List[str], buckets: List[Tuple[int, int]], batch_size: int, bucket_batch_sizes: List[BucketBatchSize], num_source_factors: int = 1, num_target_factors: int = 1, permute: bool = True, dtype: str = 'int32') -> None:
        super().__init__(buckets=buckets, batch_size=batch_size, bucket_batch_sizes=bucket_batch_sizes, num_source_factors=num_source_factors, num_target_factors=num_target_factors, permute=permute, dtype=dtype)
        assert len(shards_fnames) > 0
        self.shards_fnames: List[str] = list(shards_fnames)
        self.shard_index: int = -1
        self.reset()

    def _load_shard(self) -> None:
        shard_fname: str = self.shards_fnames[self.shard_index]
        logger.info('Loading shard %s.', shard_fname)
        dataset = ParallelDataSet.load(self.shards_fnames[self.shard_index]).fill_up(self.bucket_batch_sizes, seed=self.shard_index)
        self.shard_iter: ParallelSampleIter = ParallelSampleIter(data=dataset, buckets=self.buckets, batch_size=self.batch_size, bucket_batch_sizes=self.bucket_batch_sizes, num_source_factors=self.num_source_factors, num_target_factors=self.num_target_factors, permute=self.permute)

    def reset(self) -> None:
        if len(self.shards_fnames) > 1:
            logger.info('Shuffling the shards.')
            if self.shard_index < 0:
                current_shard_fname: str = ''
            else:
                current_shard_fname = self.shards_fnames[self.shard_index]
            remaining_shards = [shard for shard in self.shards_fnames if shard != current_shard_fname]
            next_shard_fname = random.choice(remaining_shards)
            remaining_shards = [shard for shard in self.shards_fnames if shard != next_shard_fname]
            random.shuffle(remaining_shards)
            self.shards_fnames = [next_shard_fname] + remaining_shards
            if utils.is_distributed():
                self.shards_fnames = utils.broadcast_object(self.shards_fnames)
            self.shard_index = 0
            self._load_shard()
        else:
            if self.shard_index < 0:
                self.shard_index = 0
                self._load_shard()
            self.shard_iter.reset()

    def iter_next(self) -> bool:
        next_shard_index: int = self.shard_index + 1
        return self.shard_iter.iter_next() or next_shard_index < len(self.shards_fnames)

    def next(self) -> Any:
        if not self.shard_iter.iter_next():
            if self.shard_index < len(self.shards_fnames) - 1:
                self.shard_index += 1
                self._load_shard()
            else:
                raise StopIteration
        return self.shard_iter.next()

    def save_state(self, fname: str) -> None:
        with open(fname, 'wb') as fp:
            pickle.dump(self.shards_fnames, fp)
            pickle.dump(self.shard_index, fp)
        self.shard_iter.save_state(fname + '.sharditer')

    def load_state(self, fname: str) -> None:
        with open(fname, 'rb') as fp:
            self.shards_fnames = pickle.load(fp)
            self.shard_index = pickle.load(fp)
        self._load_shard()
        self.shard_iter.load_state(fname + '.sharditer')


class ParallelSampleIter(BaseParallelSampleIter):
    """
    Data iterator on a bucketed ParallelDataSet. Shuffles data at every reset and supports saving and loading the
    iterator state.
    """
    def __init__(self, data: ParallelDataSet, buckets: List[Tuple[int, int]], batch_size: int, bucket_batch_sizes: List[BucketBatchSize], num_source_factors: int = 1, num_target_factors: int = 1, permute: bool = True, dtype: str = 'int32') -> None:
        super().__init__(buckets=buckets, batch_size=batch_size, bucket_batch_sizes=bucket_batch_sizes, num_source_factors=num_source_factors, num_target_factors=num_target_factors, permute=permute, dtype=dtype)
        self.data: ParallelDataSet = ParallelDataSet(list(data.source), list(data.target), list(data.prepended_source_length) if data.prepended_source_length is not None else None)
        self.batch_indices: List[Tuple[int, int]] = get_batch_indices(self.data, bucket_batch_sizes)
        self.curr_batch_index: int = 0
        self.inverse_data_permutations: List[torch.Tensor] = [torch.arange(0, max(1, self.data.source[i].shape[0])) for i in range(len(self.data))]
        self.data_permutations: List[torch.Tensor] = [torch.arange(0, max(1, self.data.source[i].shape[0])) for i in range(len(self.data))]
        self.reset()

    def reset(self) -> None:
        """
        Resets and reshuffles the data.
        """
        self.curr_batch_index = 0
        if self.permute:
            if utils.is_primary_worker():
                random.shuffle(self.batch_indices)
            if utils.is_distributed():
                self.batch_indices = utils.broadcast_object(self.batch_indices)
            self.data = self.data.permute(self.inverse_data_permutations)
            self.data_permutations, self.inverse_data_permutations = get_permutations(self.data.get_bucket_counts())
            self.data = self.data.permute(self.data_permutations)

    def iter_next(self) -> bool:
        """
        True if iterator can return another batch
        """
        return self.curr_batch_index != len(self.batch_indices)

    def next(self) -> Any:
        if not self.iter_next():
            raise StopIteration
        i, j = self.batch_indices[self.curr_batch_index]
        self.curr_batch_index += 1
        batch_size: int = self.bucket_batch_sizes[i].batch_size
        source: torch.Tensor = self.data.source[i][j:j + batch_size]
        target, label = create_target_and_shifted_label_sequences(self.data.target[i][j:j + batch_size])
        prepended_source_length: Optional[torch.Tensor] = self.data.prepended_source_length[i][j:j + batch_size] if self.data.prepended_source_length is not None else None
        return create_batch_from_parallel_sample(source, target, label, prepended_source_length)

    def save_state(self, fname: str) -> None:
        with open(fname, 'wb') as fp:
            pickle.dump(self.batch_indices, fp)
            pickle.dump(self.curr_batch_index, fp)
            pickle.dump(self.inverse_data_permutations, fp)
            pickle.dump(self.data_permutations, fp)

    def load_state(self, fname: str) -> None:
        self.data = self.data.permute(self.inverse_data_permutations)
        with open(fname, 'rb') as fp:
            self.batch_indices = pickle.load(fp)
            self.curr_batch_index = pickle.load(fp)
            inverse_data_permutations = pickle.load(fp)
            data_permutations = pickle.load(fp)
        self.curr_batch_index -= 1
        self.inverse_data_permutations = []
        self.data_permutations = []
        for bucket in range(len(self.data)):
            self.inverse_data_permutations.append(inverse_data_permutations[bucket])
            self.data_permutations.append(data_permutations[bucket])
        self.data = self.data.permute(self.data_permutations)


@dataclass
class Batch:
    source: torch.Tensor
    source_length: torch.Tensor
    target: torch.Tensor
    target_length: torch.Tensor
    labels: Dict[str, torch.Tensor]
    samples: int
    tokens: int

    def load(self, device: torch.device) -> "Batch":
        source = self.source.to(device)
        source_length = self.source_length.to(device)
        target = self.target.to(device)
        target_length = self.target_length.to(device)
        labels = {name: label.to(device) for name, label in self.labels.items()}
        return Batch(source, source_length, target, target_length, labels, self.samples, self.tokens)


def create_target_and_shifted_label_sequences(target_and_label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the target and label sequence from a joint tensor of varying-length
    sequences including both <bos> and <eos>. Both tensors returned have input
    size of second dimension - 1.
    """
    target = target_and_label[:, :-1, :].clone()
    target[target == C.EOS_ID] = 0
    label = target_and_label[:, 1:, :]
    return (target, label)


def create_batch_from_parallel_sample(source: torch.Tensor, target: torch.Tensor, label: torch.Tensor, prepended_source_length: Optional[torch.Tensor] = None) -> Batch:
    """
    Creates a Batch instance from parallel data.

    :param source: Source tensor. Shape: (batch, max_source_length, num_source_factors).
    :param target: Target tensor. Shape: (batch, max_target_length, num_target_factors).
    :param label: Time-shifted label tensor. Shape: (batch, max_target_length, num_target_factors).
    :param prepended_source_length: Length of prepended source tokens tensor. Shape: (batch,).
    """
    source_words = source[:, :, 0]
    all_source_length = (source_words != C.PAD_ID).sum(dim=1)
    if prepended_source_length is None:
        prepended_source_length = torch.zeros_like(all_source_length)
    source_length = torch.stack((all_source_length, prepended_source_length), dim=1)
    target_words = target[:, :, 0]
    target_length = (target_words != C.PAD_ID).sum(dim=1)
    length_ratio = all_source_length / target_length
    samples, tokens, _ = source.shape
    tokens *= samples
    labels = {C.LENRATIO_LABEL_NAME: length_ratio}
    if label.shape[2] == 1:
        labels[C.TARGET_LABEL_NAME] = torch.squeeze(label, dim=2)
    else:
        primary_label, *factor_labels = (torch.squeeze(x, dim=2) for x in torch.split(label, 1, dim=2))
        labels[C.TARGET_LABEL_NAME] = primary_label
        labels.update({C.TARGET_FACTOR_LABEL_NAME % i: label for i, label in enumerate(factor_labels, 1)})
    return Batch(source, source_length, target, target_length, labels, samples, tokens)
