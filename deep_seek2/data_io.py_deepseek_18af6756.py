from typing import List, Tuple, Optional, Dict, Any, Iterable, Sequence, Set, Iterator, cast
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
    buckets = list(range(step, max_seq_len + step, step))
    buckets[-1] = max_seq_len
    return buckets

def define_parallel_buckets(max_seq_len_source: int,
                            max_seq_len_target: int,
                            bucket_width: int = 10,
                            bucket_scaling: bool = True,
                            length_ratio: float = 1.0) -> List[Tuple[int, int]]:
    source_step_size = bucket_width
    target_step_size = bucket_width
    if bucket_scaling:
        if length_ratio >= 1.0:
            source_step_size = max(1, int(round(bucket_width / length_ratio)))
        else:
            target_step_size = max(1, int(round(bucket_width * length_ratio)))
    source_buckets = define_buckets(max_seq_len_source, step=source_step_size)
    target_buckets = define_buckets(max_seq_len_target, step=target_step_size)
    if len(source_buckets) < len(target_buckets):
        source_buckets += [source_buckets[-1] for _ in range(len(target_buckets) - len(source_buckets))]
    elif len(target_buckets) < len(source_buckets):
        target_buckets += [target_buckets[-1] for _ in range(len(source_buckets) - len(target_buckets))]
    source_buckets = [max(2, b) for b in source_buckets]
    target_buckets = [max(2, b) for b in target_buckets]
    parallel_buckets = list(zip(source_buckets, target_buckets))
    buckets = list(OrderedDict.fromkeys(parallel_buckets))
    buckets.sort()
    return buckets

def get_bucket(seq_len: int, buckets: List[int]) -> Optional[int]:
    bucket_idx = bisect.bisect_left(buckets, seq_len)
    if bucket_idx == len(buckets):
        return None
    return buckets[bucket_idx]

@dataclass
class BucketBatchSize:
    bucket: Tuple[int, int]
    batch_size: int
    average_target_words_per_batch: float

def define_bucket_batch_sizes(buckets: List[Tuple[int, int]],
                              batch_size: int,
                              batch_type: str,
                              data_target_average_len: List[Optional[float]],
                              batch_sentences_multiple_of: int = 1) -> List[BucketBatchSize]:
    check_condition(len(data_target_average_len) == len(buckets),
                    "Must provide None or average target length for each bucket")
    data_target_average_len = list(data_target_average_len)
    bucket_batch_sizes = []  # type: List[BucketBatchSize]
    largest_total_num_words = 0

    for buck_idx, bucket in enumerate(buckets):
        padded_seq_len = bucket[1]
        if data_target_average_len[buck_idx] is None:
            data_target_average_len[buck_idx] = padded_seq_len
        average_seq_len = data_target_average_len[buck_idx]

        if batch_type == C.BATCH_TYPE_WORD:
            check_condition(padded_seq_len <= batch_size, "Word batch size must cover sequence lengths for all"
                                                          " buckets: (%d > %d)" % (padded_seq_len, batch_size))
            batch_size_seq = batch_sentences_multiple_of * max(1, round((batch_size / average_seq_len) /
                                                                        batch_sentences_multiple_of))
        elif batch_type == C.BATCH_TYPE_MAX_WORD:
            check_condition(padded_seq_len <= batch_size,
                            'Word batch size must cover sequence lengths for all buckets: (%d > %d)'
                            % (padded_seq_len, batch_size))
            batch_size_seq = batch_size // padded_seq_len
            check_condition(batch_size_seq // batch_sentences_multiple_of > 0,
                            'Please increase the batch size to avoid the batch size being rounded down to 0.')
            batch_size_seq = (batch_size_seq // batch_sentences_multiple_of) * batch_sentences_multiple_of
        elif batch_type == C.BATCH_TYPE_SENTENCE:
            batch_size_seq = batch_size
        else:
            raise ValueError('Unknown batch type: %s' % batch_type)
        batch_size_word = batch_size_seq * average_seq_len

        bucket_batch_sizes.append(BucketBatchSize(bucket, batch_size_seq, batch_size_word))
        largest_total_num_words = max(largest_total_num_words, batch_size_seq * max(*bucket))

    return bucket_batch_sizes

def calculate_length_statistics(source_iterables: Sequence[Iterable[Any]],
                                target_iterables: Sequence[Iterable[Any]],
                                max_seq_len_source: int,
                                max_seq_len_target: int) -> 'LengthStatistics':
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
    if not sequences:
        return True
    return all(s is None for s in sequences)

def are_token_parallel(sequences: Sequence[Sized]) -> bool:
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
    def statistics(self):
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
        target_readers = [exit_stack.enter_context(smart_open(f, mode="rb")) for f in target_fnames]

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

    return list(zip(sources_shard_fnames_by_shards, targets_shard_fnames_by_shards)), False

def get_prepended_token_length(ids: List[int], eop_id: int) -> int:
    if eop_id == C.INVALID_ID:
        return 0
    try:
        return ids.index(eop_id) + 1
    except ValueError:
        return 0

class RawParallelDatasetLoader:
    def __init__(self,
                 buckets: List[Tuple[int, int]],
                 eos_id: int,
                 pad_id: int,
                 eop_id: int = C.INVALID_ID,
                 skip_blanks: bool = True,
                 dtype: str = 'int32',
                 shift_target_factors: bool = C.TARGET_FACTOR_SHIFT) -> None:
        self.buckets = buckets
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.eop_id = eop_id
        self.skip_blanks = skip_blanks
        self.dtype = dtype
        self.shift_target_factors = shift_target_factors

    def load(self,
             source_iterables: Sequence[Iterable],
             target_iterables: Sequence[Iterable],
             num_samples_per_bucket: List[int]) -> 'ParallelDataSet':

        assert len(num_samples_per_bucket) == len(self.buckets)
        num_source_factors = len(source_iterables)
        num_target_factors = len(target_iterables)

        data_source = [np.full((num_samples, source_len, num_source_factors), self.pad_id, dtype=self.dtype)
                       for (source_len, _), num_samples in zip(self.buckets, num_samples_per_bucket)]
        data_target = [np.full((num_samples, target_len + 1, num_target_factors), self.pad_id, dtype=self.dtype)
                       for (_, target_len), num_samples in zip(self.buckets, num_samples_per_bucket)]
        data_prepended_source_length = \
            [np.full((num_samples,), 0, dtype=self.dtype) for num_samples in num_samples_per_bucket] \
                if self.eop_id != C.INVALID_ID else None

        bucket_sample_index = [0 for _ in self.buckets]

        num_tokens_source = 0
        num_tokens_target = 0
        num_pad_source = 0
        num_pad_target = 0

        for sources, targets in parallel_iter(source_iterables, target_iterables):
            sources = [[] if stream is None else stream for stream in sources]
            targets = [[] if stream is None else stream for stream in targets]
            source_len = len(sources[0])
            target_len = len(targets[0])
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

            sample_index = bucket_sample_index[buck_index]
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
                data_prepended_source_length[buck_index][sample_index] = get_prepended_token_length(sources[0],
                                                                                                    self.eop_id)

            bucket_sample_index[buck_index] += 1

        data_source_tensors = [torch.from_numpy(data) for data in data_source]
        data_target_tensors = [torch.from_numpy(data) for data in data_target]
        data_prepended_source_length_tensors = [torch.from_numpy(data) for data in data_prepended_source_length] \
            if data_prepended_source_length is not None else None

        if num_tokens_source > 0 and num_tokens_target > 0:
            logger.info("Created bucketed parallel data set. Introduced padding: source=%.1f%% target=%.1f%%)",
                        num_pad_source / num_tokens_source * 100,
                        num_pad_target / num_tokens_target * 100)

        return ParallelDataSet(data_source_tensors, data_target_tensors, data_prepended_source_length_tensors)

def get_num_shards(num_samples: int, samples_per_shard: int, min_num_shards: int) -> int:
    return max(int(math.ceil(num_samples / samples_per_shard)), min_num_shards)

def save_shard(shard_idx: int,
               data_loader: RawParallelDatasetLoader,
               shard_sources: List[str],
               shard_targets: List[str],
               source_vocabs: List[vocab.Vocab],
               target_vocabs: List[vocab.Vocab],
               length_ratio_mean: float,
               length_ratio_std: float,
               buckets: List[Tuple[int, int]],
               output_prefix: str,
               keep_tmp_shard_files: bool):
    shard_stat_accumulator = DataStatisticsAccumulator(buckets, source_vocabs[0], target_vocabs[0],
                                                       length_ratio_mean, length_ratio_std)

    sources_sentences, targets_sentences = create_sequence_readers(shard_sources, shard_targets, source_vocabs, target_vocabs)

    for sources, targets in parallel_iter(sources_sentences, targets_sentences):
        source_len = len(sources[0])
        target_len = len(targets[0])

        buck_idx, _ = get_parallel_bucket(buckets, source_len, target_len)
        shard_stat_accumulator.sequence_pair(sources[0], targets[0], buck_idx)

    shard_stats = shard_stat_accumulator.statistics

    dataset = data_loader.load(sources_sentences, targets_sentences, shard_stats.num_sents_per_bucket)
    shard_fname = os.path.join(output_prefix, C.SHARD_NAME % shard_idx)
    shard_stats.log()
    logger.info("Writing '%s'", shard_fname)
    dataset.save(shard_fname)

    if not keep_tmp_shard_files:
        for f in chain(shard_sources, shard_targets):
            os.remove(f)

    return shard_stat_accumulator.statistics

def get_eop_id(vocab: vocab.Vocab, end_of_prepending_tag: str) -> int:
    eop_id = vocab.get(end_of_prepending_tag, C.INVALID_ID)
    if end_of_prepending_tag is not None:
        check_condition(eop_id != C.INVALID_ID,
                        f"The end-of-prepending tag '{end_of_prepending_tag}' is not found in the vocabulary.")
    return eop_id

def prepare_data(source_fnames: List[str],
                 target_fnames: List[str],
                 source_vocabs: List[vocab.Vocab],
                 target_vocabs: List[vocab.Vocab],
                 source_vocab_paths: List[Optional[str]],
                 target_vocab_paths: List[Optional[str]],
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
                 pool: multiprocessing.pool.Pool = None,
                 shards: List[Tuple[Tuple[str, ...], Tuple[str, ...]]] = None):
    logger.info("Preparing data.")
    vocab.save_source_vocabs(source_vocabs, output_prefix)
    vocab.save_target_vocabs(target_vocabs, output_prefix)

    stats_args = ((source_path, target_path, source_vocabs, target_vocabs, max_seq_len_source, max_seq_len_target)
                  for source_path, target_path in shards)
    length_stats = pool.starmap(analyze_sequence_lengths, stats_args)
    shards_num_sents = [stat.num_sents for stat in length_stats]
    shards_mean = [stat.length_ratio_mean for stat in length_stats]
    shards_std = [stat.length_ratio_std for stat in length_stats]
    length_ratio_mean = combine_means(shards_mean, shards_num_sents)
    length_ratio_std = combine_stds(shards_std, shards_mean, shards_num_sents)
    length_statistics = LengthStatistics(sum(shards_num_sents), length_ratio_mean, length_ratio_std)

    check_condition(length_statistics.num_sents > 0,
                    "No training sequences found with length smaller or equal than the maximum sequence length."
                    "Consider increasing %s" % C.TRAINING_ARG_MAX_SEQ_LEN)

    buckets = define_parallel_buckets(max_seq_len_source, max_seq_len_target, bucket_width, bucket_scaling,
                                      length_statistics.length_ratio_mean) if bucketing else [(max_seq_len_source,
                                                                                               max_seq_len_target)]
    logger.info("Buckets: %s", buckets)

    eop_id = get_eop_id(source_vocabs[0], end_of_prepending_tag)
    data_loader = RawParallelDatasetLoader(buckets=buckets,
                                           eos_id=C.EOS_ID,
                                           pad_id=C.PAD_ID,
                                           eop_id=eop_id)

    args = ((shard_idx, data_loader, shard_sources, shard_targets, source_vocabs, target_vocabs,
             length_statistics.length_ratio_mean, length_statistics.length_ratio_std, buckets, output_prefix,
             keep_tmp_shard_files) for shard_idx, (shard_sources, shard_targets) in enumerate(shards))
    per_shard_statistics = pool.starmap(save_shard, args)

    shard_average_len = [shard_stats.average_len_target_per_bucket for shard_stats in per_shard_statistics]
    shard_num_sents = [shard_stats.num_sents_per_bucket for shard_stats in per_shard_statistics]
    num_sents_per_bucket = [sum(n) for n in zip(*shard_num_sents)]
    average_len_target_per_bucket = [] # type: List[Optional[float]]
    for num_sents_bucket, average_len_bucket in zip(zip(*shard_num_sents), zip(*shard_average_len)):
        if all(avg is None for avg in average_len_bucket):
            average_len_target_per_bucket.append(None)
        else:
            average_len_target_per_bucket.append(combine_means(average_len_bucket, shards_num_sents))

    shard_length_ratios = [shard_stats.length_ratio_stats_per_bucket for shard_stats in per_shard_statistics]
    length_ratio_stats_per_bucket = [] # type: Optional[List[Tuple[Optional[float], Optional[float]]]]
    for num_sents_bucket, len_ratios_bucket in zip(zip(*shard_num_sents), zip(*shard_length_ratios)):
        if all(all(x is None for x in ratio) for ratio in len_ratios_bucket):
            length_ratio_stats_per_bucket.append((None, None))
        else:
            shards_mean = [ratio[0] for ratio in len_ratios_bucket]
            ratio_mean = combine_means(shards_mean, num_sents_bucket)
            ratio_std = combine_stds([ratio[1] for ratio in len_ratios_bucket], shards_mean, num_sents_bucket)
            length_ratio_stats_per_bucket.append((ratio_mean, ratio_std))
    data_statistics = DataStatistics(
        num_sents=sum(shards_num_sents),
        num_discarded=sum(shard_stats.num_discarded for shard_stats in per_shard_statistics),
        num_tokens_source=sum(shard_stats.num_tokens_source for shard_stats in per_shard_statistics),
        num_tokens_target=sum(shard_stats.num_tokens_target for shard_stats in per_shard_statistics),
        num_unks_source=sum(shard_stats.num_unks_source for shard_stats in per_shard_statistics),
        num_unks_target=sum(shard_stats.num_unks_target for shard_stats in per_shard_statistics),
        max_observed_len_source=max(shard_stats.max_observed_len_source for shard_stats in per_shard_statistics),
        max_observed_len_target=max(shard_stats.max_observed_len_target for shard_stats in per_shard_statistics),
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
    data_info_fname = os.path.join(output_prefix, C.DATA_INFO)
    logger.info("Writing data info to '%s'", data_info_fname)
    data_info.save(data_info_fname)

    config_data = DataConfig(data_statistics=data_statistics,
                             max_seq_len_source=max_seq_len_source,
                             max_seq_len_target=max_seq_len_target,
                             num_source_factors=len(source_fnames),
                             num_target_factors=len(target_fnames),
                             eop_id=eop_id)
    config_data_fname = os.path.join(output_prefix, C.DATA_CONFIG)
    logger.info("Writing data config to '%s'", config_data_fname)
    config_data.save(config_data_fname)

    version_file = os.path.join(output_prefix, C.PREPARED_DATA_VERSION_FILE)

    with open(version_file, "w") as version_out:
        version_out.write(str(C.PREPARED_DATA_VERSION))

def get_data_statistics(source_readers: Optional[Sequence[Iterable]],
                        target_readers: Sequence[Iterable],
                        buckets: List[Tuple[int, int]],
                        length_ratio_mean: float,
                        length_ratio_std: float,
                        source_vocabs: Optional[List[vocab.Vocab]],
                        target_vocabs: List[vocab.Vocab]) -> 'DataStatistics':
    data_stats_accumulator = DataStatisticsAccumulator(buckets,
                                                       source_vocabs[0] if source_vocabs is not None else None,
                                                       target_vocabs[0],
                                                       length_ratio_mean,
                                                       length_ratio_std)

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
                             source_vocabs: List[vocab.Vocab],
                             target_vocabs: List[vocab.Vocab],
                             max_seq_len_source: int,
                             max_seq_len_target: int,
                             batch_size: int,
                             permute: bool = False) -> 'ParallelSampleIter':
    logger.info("=================================")
    logger.info("Creating validation data iterator")
    logger.info("=================================")
    validation_length_statistics = analyze_sequence_lengths(validation_sources, validation_targets,
                                                            source_vocabs, target_vocabs,
                                                            max_seq_len_source, max_seq_len_target)

    check_condition(validation_length_statistics.num_sents > 0,
                    "No validation sequences found with length smaller or equal than the maximum sequence length."
                    "Consider increasing %s" % C.TRAINING_ARG_MAX_SEQ_LEN)

    validation_sources_sentences, validation_targets_sentences = create_sequence_readers(validation_sources,
                                                                                         validation_targets,
                                                                                         source_vocabs, target_vocabs)

    validation_data_statistics = get_data_statistics(validation_sources_sentences,
                                                     validation_targets_sentences,
                                                     buckets,
                                                     validation_length_statistics.length_ratio_mean,
                                                     validation_length_statistics.length_ratio_std,
                                                     source_vocabs, target_vocabs)

    validation_data_statistics.log(bucket_batch_sizes)

    validation_data = data_loader.load(validation_sources_sentences, validation_targets_sentences,
                                       validation_data_statistics.num_sents_per_bucket).fill_up(bucket_batch_sizes)

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
                            permute: bool = True) -> Tuple['BaseParallelSampleIter',
                                                           'BaseParallelSampleIter',
                                                           'DataConfig', List[vocab.Vocab], List[vocab.Vocab]]:
    logger.info("===============================")
    logger.info("Creating training data iterator")
    logger.info("===============================")

    version_file = os.path.join(prepared_data_dir, C.PREPARED_DATA_VERSION_FILE)
    with open(version_file) as version_in:
        version = int(version_in.read())
        check_condition(version in (C.PREPARED_DATA_VERSION, C.PREPARED_DATA_LEGACY_VERSION),
                        "The dataset %s was written in an incompatible format. "
                        "Please rerun data preparation with this version of Sockeye." % prepared_data_dir)
    info_file = os.path.join(prepared_data_dir, C.DATA_INFO)
    check_condition(os.path.exists(info_file),
                    "Could not find data info %s. Are you sure %s is a directory created with "
                    "sockeye-prepare-data?" % (info_file, prepared_data_dir))
    data_info = cast(DataInfo, DataInfo.load(info_file))
    config_file = os.path.join(prepared_data_dir, C.DATA_CONFIG)
    check_condition(os.path.exists(config_file),
                    "Could not find data config %s. Are you sure %s is a directory created with "
                    "sockeye-prepare-data?" % (config_file, prepared_data_dir))
    config_data = cast(DataConfig, DataConfig.load(config_file))
    shard_fnames = [os.path.join(prepared_data_dir,
                                 C.SHARD_NAME % shard_idx) for shard_idx in range(data_info.num_shards)]
    for shard_fname in shard_fnames:
        check_condition(os.path.exists(shard_fname), "Shard %s does not exist." % shard_fname)

    check_condition(shared_vocab == data_info.shared_vocab, "Shared vocabulary settings need to match these "
                                                            "of the prepared data (e.g. for weight tying). "
                                                            "Specify or omit %s consistently when training "
                                                            "and preparing the data." % C.VOCAB_ARG_SHARED_VOCAB)

    source_vocabs = vocab.load_source_vocabs(prepared_data_dir)
    target_vocabs = vocab.load_target_vocabs(prepared_data_dir)

    check_condition(len(source_vocabs) == len(data_info.sources),
                    "Wrong number of source vocabularies. Found %d, need %d." % (len(source_vocabs),
                                                                                 len(data_info.sources)))
    check_condition(len(target_vocabs) == len(data_info.targets),
                    "Wrong number of target vocabularies. Found %d, need %d." % (len(target_vocabs),
                                                                                 len(data_info.targets)))

    buckets = config_data.data_statistics.buckets
    max_seq_len_source = config_data.max_seq_len_source
    max_seq_len_target = config_data.max_seq_len_target

    bucket_batch_sizes = define_bucket_batch_sizes(buckets,
                                                   batch_size,
                                                   batch_type,
                                                   config_data.data_statistics.average_len_target_per_bucket,
                                                   batch_sentences_multiple_of)

    config_data.data_statistics.log(bucket_batch_sizes)

    train_iter = ShardedParallelSampleIter(shard_fnames,
                                           buckets,
                                           batch_size,
                                           bucket_batch_sizes,
                                           num_source_factors=len(data_info.sources),
                                           num_target_factors=len(data