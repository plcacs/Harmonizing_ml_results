import os
import random
from tempfile import TemporaryDirectory
from typing import Optional, List, Tuple, Dict, Any
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

define_bucket_tests: List[Tuple[int, int, List[int]]] = [
    (50, 10, [10, 20, 30, 40, 50]),
    (50, 20, [20, 40, 50]),
    (50, 50, [50]),
    (5, 10, [5]),
    (11, 5, [5, 10, 11]),
    (19, 10, [10, 19]),
]

@pytest.mark.parametrize('max_seq_len, step, expected_buckets', define_bucket_tests)
def test_define_buckets(max_seq_len: int, step: int, expected_buckets: List[int]) -> None:
    buckets = data_io.define_buckets(max_seq_len, step=step)
    assert buckets == expected_buckets

define_parallel_bucket_tests: List[Tuple[int, int, int, bool, float, List[Tuple[int, int]]]] = [
    (50, 50, 10, True, 1.0, [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)]),
    (50, 50, 10, True, 0.5, [(10, 5), (20, 10), (30, 15), (40, 20), (50, 25), (50, 30), (50, 35), (50, 40), (50, 45), (50, 50)]),
    (10, 10, 10, True, 0.1, [(10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10)]),
    (10, 5, 10, True, 0.01, [(10, 2), (10, 3), (10, 4), (10, 5)]),
    (50, 50, 10, True, 2.0, [(5, 10), (10, 20), (15, 30), (20, 40), (25, 50), (30, 50), (35, 50), (40, 50), (45, 50), (50, 50)]),
    (5, 10, 10, True, 10.0, [(2, 10), (3, 10), (4, 10), (5, 10)]),
    (5, 10, 10, True, 11.0, [(2, 10), (3, 10), (4, 10), (5, 10)]),
    (50, 50, 50, True, 0.5, [(50, 25), (50, 50)]),
    (50, 50, 50, True, 1.5, [(33, 50), (50, 50)]),
    (75, 75, 50, True, 1.5, [(33, 50), (66, 75), (75, 75)]),
    (50, 50, 8, False, 1.5, [(8, 8), (16, 16), (24, 24), (32, 32), (40, 40), (48, 48), (50, 50)]),
    (50, 75, 8, False, 1.5, [(8, 8), (16, 16), (24, 24), (32, 32), (40, 40), (48, 48), (50, 56), (50, 64), (50, 72), (50, 75)]),
]

@pytest.mark.parametrize('max_seq_len_source, max_seq_len_target, bucket_width, bucket_scaling, length_ratio,expected_buckets', define_parallel_bucket_tests)
def test_define_parallel_buckets(
    max_seq_len_source: int,
    max_seq_len_target: int,
    bucket_width: int,
    bucket_scaling: bool,
    length_ratio: float,
    expected_buckets: List[Tuple[int, int]],
) -> None:
    buckets = data_io.define_parallel_buckets(
        max_seq_len_source, max_seq_len_target, bucket_width=bucket_width, bucket_scaling=bucket_scaling, length_ratio=length_ratio
    )
    assert buckets == expected_buckets

get_bucket_tests: List[Tuple[List[int], int, Optional[int]]] = [
    ([10, 20, 30, 40, 50], 50, 50),
    ([10, 20, 30, 40, 50], 11, 20),
    ([10, 20, 30, 40, 50], 9, 10),
    ([10, 20, 30, 40, 50], 51, None),
    ([10, 20, 30, 40, 50], 1, 10),
    ([10, 20, 30, 40, 50], 0, 10),
    ([], 50, None),
]

@pytest.mark.parametrize('buckets, length, expected_bucket', get_bucket_tests)
def test_get_bucket(buckets: List[int], length: int, expected_bucket: Optional[int]) -> None:
    bucket = data_io.get_bucket(length, buckets)
    assert bucket == expected_bucket

tokens2ids_tests: List[Tuple[List[str], Dict[str, int], List[int]]] = [
    (
        ['a', 'b', 'c'],
        {'a': 1, 'b': 0, 'c': 300, C.UNK_SYMBOL: 12},
        [1, 0, 300],
    ),
    (
        ['a', 'x', 'c'],
        {'a': 1, 'b': 0, 'c': 300, C.UNK_SYMBOL: 12},
        [1, 12, 300],
    ),
]

@pytest.mark.parametrize('tokens, vocab, expected_ids', tokens2ids_tests)
def test_tokens2ids(tokens: List[str], vocab: Dict[str, int], expected_ids: List[int]) -> None:
    ids = data_io.tokens2ids(tokens, vocab)
    assert ids == expected_ids

@pytest.mark.parametrize('tokens, expected_ids', [
    (['1', '2', '3', '0'], [1, 2, 3, 0]),
    ([], []),
])
def test_strids2ids(tokens: List[str], expected_ids: List[int]) -> None:
    ids = data_io.strids2ids(tokens)
    assert ids == expected_ids

sequence_reader_tests: List[Tuple[List[str], bool, bool, bool]] = [
    (['1 2 3', '2', '', '2 2 2'], False, False, False),
    (['a b c', 'c'], True, False, False),
    (['a b c', ''], True, False, False),
    (['a b c', 'c'], True, True, True),
]

@pytest.mark.parametrize('sequences, use_vocab, add_bos, add_eos', sequence_reader_tests)
def test_sequence_reader(
    sequences: List[str],
    use_vocab: bool,
    add_bos: bool,
    add_eos: bool
) -> None:
    with TemporaryDirectory() as work_dir:
        path = os.path.join(work_dir, 'input')
        with open(path, 'w') as f:
            for sequence in sequences:
                print(sequence, file=f)
        vocabulary: Optional[Dict[str, int]] = vocab.build_vocab(sequences) if use_vocab else None
        reader = data_io.SequenceReader(path, vocabulary=vocabulary, add_bos=add_bos, add_eos=add_eos)
        read_sequences: List[Optional[List[int]]] = [s for s in reader]
        assert len(read_sequences) == len(sequences)
        if vocabulary is None:
            with pytest.raises(SockeyeError) as e:
                data_io.SequenceReader(path, vocabulary=vocabulary, add_bos=True)
            assert str(e.value) == 'Adding a BOS or EOS symbol requires a vocabulary'
            expected_sequences: List[Optional[List[int]]] = [
                data_io.strids2ids(get_tokens(s)) if s else None for s in sequences
            ]
            assert read_sequences == expected_sequences
        else:
            expected_sequences: List[Optional[List[int]]] = [
                data_io.tokens2ids(get_tokens(s), vocabulary) if s else None for s in sequences
            ]
            if add_bos:
                expected_sequences = [
                    [vocabulary[C.BOS_SYMBOL]] + s if s else None for s in expected_sequences
                ]
            if add_eos:
                expected_sequences = [
                    s + [vocabulary[C.EOS_SYMBOL]] if s else None for s in expected_sequences
                ]
            assert read_sequences == expected_sequences

@pytest.mark.parametrize('source_iterables, target_iterables', [
    (
        [
            [[0], [1, 1], [2], [3, 3, 3]],
            [[0], [1, 1], [2], [3, 3, 3]]
        ],
        [[[0], [1]]]
    ),
    (
        [
            [[0], [1, 1]],
            [[0], [1, 1]]
        ],
        [[[0], [1, 1], [2], [3, 3, 3]]]
    ),
    (
        [[[0], [1, 1]]],
        [[[0], [1, 1], [2], [3, 3, 3]]]
    ),
])
def test_nonparallel_iter(
    source_iterables: List[List[List[int]]],
    target_iterables: List[List[List[int]]]
) -> None:
    with pytest.raises(SockeyeError) as e:
        list(data_io.parallel_iter(source_iterables, target_iterables))
    assert str(e.value) == 'Different number of lines in source(s) and target(s) iterables.'

@pytest.mark.parametrize('source_iterables, target_iterables', [
    ([[[0], [1, 1]], [[0], [1]]], [[[0], [1]]]),
])
def test_not_source_token_parallel_iter(
    source_iterables: List[List[List[int]]],
    target_iterables: List[List[List[int]]]
) -> None:
    with pytest.raises(SockeyeError) as e:
        list(data_io.parallel_iter(source_iterables, target_iterables))
    assert str(e.value).startswith('Source sequences are not token-parallel')

@pytest.mark.parametrize('source_iterables, target_iterables', [
    ([[[0], [1]]], [[[0], [1, 1]], [[0], [1]]]),
])
def test_not_target_token_parallel_iter(
    source_iterables: List[List[List[int]]],
    target_iterables: List[List[List[int]]]
) -> None:
    with pytest.raises(SockeyeError) as e:
        list(data_io.parallel_iter(source_iterables, target_iterables))
    assert str(e.value).startswith('Target sequences are not token-parallel')

@pytest.mark.parametrize('source_iterables, target_iterables, expected', [
    (
        [
            [[0], [1, 1]],
            [[0], [1, 1]]
        ],
        [[[0], [1]]],
        [([[0], [0]], [[0]]), ([[1, 1], [1, 1]], [[1]])]
    ),
    (
        [
            [[0], None],
            [[0], None]
        ],
        [[[0], [1]]],
        [([[0], [0]], [[0]])]
    ),
    (
        [
            [[0], [1, 1]],
            [[0], [1, 1]]
        ],
        [[[0], None]],
        [([[0], [0]], [[0]])]
    ),
    (
        [
            [None, [1, 1]],
            [None, [1, 1]]
        ],
        [[None, [1]]],
        [([[1, 1], [1, 1]], [[1]])]
    ),
    (
        [
            [None, [1]]
        ],
        [[None, [1, 1]], [None, [1, 1]]],
        [([[1]], [[1, 1], [1, 1]])]
    ),
    (
        [
            [None, [1, 1]],
            [None, [1, 1]]
        ],
        [[None, None]],
        []
    ),
])
def test_parallel_iter(
    source_iterables: List[List[Optional[List[int]]]],
    target_iterables: List[List[Optional[List[int]]]],
    expected: List[Tuple[List[Optional[List[int]]], List[Optional[List[int]]]]]
) -> None:
    assert list(data_io.parallel_iter(source_iterables, target_iterables)) == expected

def test_sample_based_define_bucket_batch_sizes() -> None:
    batch_type: str = C.BATCH_TYPE_SENTENCE
    batch_size: int = 32
    max_seq_len: int = 100
    buckets: List[Tuple[int, int]] = data_io.define_parallel_buckets(100, 100, 10, True, 1.0)
    bucket_batch_sizes: List[data_io.BucketBatchSize] = data_io.define_bucket_batch_sizes(
        buckets=buckets,
        batch_size=batch_size,
        batch_type=batch_type,
        data_target_average_len=[None] * len(buckets)
    )
    for bbs in bucket_batch_sizes:
        assert bbs.batch_size == batch_size
        assert bbs.average_target_words_per_batch == bbs.bucket[1] * batch_size

@pytest.mark.parametrize('length_ratio,batch_sentences_multiple_of,expected_batch_sizes', [
    (0.5, 1, [200, 100, 67, 50, 40, 33, 29, 25, 22, 20]),
    (1.5, 1, [100, 50, 33, 25, 20, 20, 20, 20]),
    (1.5, 8, [96, 48, 32, 24, 16, 16, 16, 16]),
])
def test_word_based_define_bucket_batch_sizes(
    length_ratio: float,
    batch_sentences_multiple_of: int,
    expected_batch_sizes: List[int]
) -> None:
    batch_type: str = C.BATCH_TYPE_WORD
    batch_size: int = 1000
    max_seq_len: int = 50
    buckets: List[Tuple[int, int]] = data_io.define_parallel_buckets(50, 50, 10, True, length_ratio)
    bucket_batch_sizes: List[data_io.BucketBatchSize] = data_io.define_bucket_batch_sizes(
        buckets=buckets,
        batch_size=batch_size,
        batch_type=batch_type,
        data_target_average_len=[None] * len(buckets),
        batch_sentences_multiple_of=batch_sentences_multiple_of
    )
    for bbs, expected_batch_size in zip(bucket_batch_sizes, expected_batch_sizes):
        assert bbs.batch_size == expected_batch_size
        expected_average_target_words_per_batch: float = expected_batch_size * bbs.bucket[1]
        assert bbs.average_target_words_per_batch == expected_average_target_words_per_batch

@pytest.mark.parametrize('length_ratio,batch_sentences_multiple_of,expected_batch_sizes', [
    (0.5, 1, [200, 100, 66, 50, 40, 33, 28, 25, 22, 20]),
    (1.5, 1, [100, 50, 33, 25, 20, 20, 20, 20]),
    (1.5, 8, [96, 48, 32, 24, 16, 16, 16, 16]),
])
def test_max_word_based_define_bucket_batch_sizes(
    length_ratio: float,
    batch_sentences_multiple_of: int,
    expected_batch_sizes: List[int]
) -> None:
    batch_type: str = C.BATCH_TYPE_MAX_WORD
    batch_size: int = 1000
    max_seq_len: int = 50
    buckets: List[Tuple[int, int]] = data_io.define_parallel_buckets(50, 50, 10, True, length_ratio)
    bucket_batch_sizes: List[data_io.BucketBatchSize] = data_io.define_bucket_batch_sizes(
        buckets=buckets,
        batch_size=batch_size,
        batch_type=batch_type,
        data_target_average_len=[None] * len(buckets),
        batch_sentences_multiple_of=batch_sentences_multiple_of
    )
    for bbs, expected_batch_size in zip(bucket_batch_sizes, expected_batch_sizes):
        assert bbs.batch_size == expected_batch_size
        expected_average_target_words_per_batch: float = expected_batch_size * bbs.bucket[1]
        assert bbs.average_target_words_per_batch == expected_average_target_words_per_batch

def _get_random_bucketed_data(
    buckets: List[Tuple[int, int]],
    min_count: int,
    max_count: int,
    bucket_counts: Optional[List[Optional[int]]] = None,
    include_prepended_source_length: bool = False
) -> Tuple[List[torch.Tensor], List[torch.Tensor], Optional[List[torch.Tensor]]]:
    """
    Get random bucket data.

    :param buckets: The list of buckets.
    :param min_count: The minimum number of samples that will be sampled if no exact count is given.
    :param max_count: The maximum number of samples that will be sampled if no exact count is given.
    :param bucket_counts: For each bucket an optional exact example count can be given. If it is not given it will be
                         sampled.
    :param include_prepended_source_length: Generate random length of prepended source tokens (otherwise return None).
    :return: The random source, target and optional prepended_source_length tensors.
    """
    if bucket_counts is None:
        bucket_counts = [None for _ in buckets]
    bucket_counts = [
        random.randint(min_count, max_count) if given_count is None else given_count
        for given_count in bucket_counts
    ]
    source: List[torch.Tensor] = [
        torch.randint(0, 10, (count, random.randint(1, bucket[0]), 1))
        for count, bucket in zip(bucket_counts, buckets)
    ]
    target: List[torch.Tensor] = [
        torch.randint(0, 10, (count, random.randint(2, bucket[1]), 1))
        for count, bucket in zip(bucket_counts, buckets)
    ]
    prepended_source_length: Optional[List[torch.Tensor]] = (
        [torch.randint(0, 10, (count, 1)) for count in bucket_counts]
        if include_prepended_source_length
        else None
    )
    return (source, target, prepended_source_length)

@pytest.mark.parametrize('include_prepended_source_length', [False, True])
def test_parallel_data_set(include_prepended_source_length: bool) -> None:
    buckets: List[Tuple[int, int]] = data_io.define_parallel_buckets(100, 100, 10, True, 1.0)
    source, target, prepended_source_length = _get_random_bucketed_data(
        buckets, min_count=0, max_count=5, include_prepended_source_length=include_prepended_source_length
    )

    def check_equal(tensors1: List[torch.Tensor], tensors2: List[torch.Tensor]) -> None:
        assert len(tensors1) == len(tensors2)
        for a1, a2 in zip(tensors1, tensors2):
            assert torch.equal(a1, a2)

    with TemporaryDirectory() as work_dir:
        dataset: data_io.ParallelDataSet = data_io.ParallelDataSet(source, target, prepended_source_length)
        fname: str = os.path.join(work_dir, 'dataset')
        dataset.save(fname)
        dataset_loaded: data_io.ParallelDataSet = data_io.ParallelDataSet.load(fname)
        check_equal(dataset.source, dataset_loaded.source)
        check_equal(dataset.target, dataset_loaded.target)
        if include_prepended_source_length:
            check_equal(dataset.prepended_source_length, dataset_loaded.prepended_source_length)
        else:
            dataset.save(fname, use_legacy_format=True)
            dataset_loaded = data_io.ParallelDataSet.load(fname)
            check_equal(dataset.source, dataset_loaded.source)
            check_equal(dataset.target, dataset_loaded.target)

@pytest.mark.parametrize('include_prepended_source_length', [False, True])
def test_parallel_data_set_fill_up(include_prepended_source_length: bool) -> None:
    batch_size: int = 32
    buckets: List[Tuple[int, int]] = data_io.define_parallel_buckets(100, 100, 10, True, 1.0)
    bucket_batch_sizes: List[data_io.BucketBatchSize] = data_io.define_bucket_batch_sizes(
        buckets, batch_size, batch_type=C.BATCH_TYPE_SENTENCE, data_target_average_len=[None] * len(buckets)
    )
    dataset: data_io.ParallelDataSet = data_io.ParallelDataSet(*_get_random_bucketed_data(
        buckets, min_count=1, max_count=5, include_prepended_source_length=include_prepended_source_length
    ))
    dataset_filled_up: data_io.ParallelDataSet = dataset.fill_up(bucket_batch_sizes)
    assert len(dataset_filled_up.source) == len(dataset.source)
    assert len(dataset_filled_up.target) == len(dataset.target)
    if include_prepended_source_length:
        assert len(dataset_filled_up.prepended_source_length) == len(dataset.prepended_source_length)
    for bidx in range(len(dataset)):
        bucket_batch_size: int = bucket_batch_sizes[bidx].batch_size
        assert dataset_filled_up.source[bidx].shape[0] == bucket_batch_size
        assert dataset_filled_up.target[bidx].shape[0] == bucket_batch_size
        if include_prepended_source_length:
            assert dataset_filled_up.prepended_source_length[bidx].shape[0] == bucket_batch_size

@pytest.mark.parametrize('buckets, source_length, target_length, expected_bucket_index, expected_bucket', [
    (
        [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)],
        50, 50, 4, (50, 50)
    ),
    (
        [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)],
        50, 10, 4, (50, 50)
    ),
    (
        [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)],
        20, 10, 1, (20, 20)
    ),
    (
        [(10, 10)],
        20, 10, None, None
    ),
    (
        [],
        20, 10, None, None
    ),
    (
        [(10, 11)],
        11, 10, None, None
    ),
    (
        [(11, 10)],
        11, 10, 0, (11, 10)
    ),
])
@pytest.mark.parametrize('buckets, source_length, target_length, expected_bucket_index, expected_bucket', [
    (
        [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)],
        50, 50, 4, (50, 50)
    ),
    (
        [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)],
        50, 10, 4, (50, 50)
    ),
    (
        [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)],
        20, 10, 1, (20, 20)
    ),
    (
        [(10, 10)],
        20, 10, None, None
    ),
    (
        [],
        20, 10, None, None
    ),
    (
        [(10, 11)],
        11, 10, None, None
    ),
    (
        [(11, 10)],
        11, 10, 0, (11, 10)
    ),
])
def test_get_parallel_bucket(
    buckets: List[Tuple[int, int]],
    source_length: int,
    target_length: int,
    expected_bucket_index: Optional[int],
    expected_bucket: Optional[Tuple[int, int]]
) -> None:
    bucket_index, bucket = data_io.get_parallel_bucket(buckets, source_length, target_length)
    assert bucket_index == expected_bucket_index
    assert bucket == expected_bucket

@pytest.mark.parametrize('sources, targets, expected_num_sents, expected_mean, expected_std', [
    (
        [[[1, 1, 1], [2, 2, 2], [3, 3, 3]]],
        [[[1, 1, 1], [2, 2, 2], [3, 3, 3]]],
        3,
        1.0,
        0.0
    ),
    (
        [[[1, 1], [2, 2], [3, 3]]],
        [[[1, 1, 1], [2, 2, 2], [3, 3, 3]]],
        3,
        1.5,
        0.0
    ),
    (
        [[[1, 1, 1], [2, 2], [3, 3, 3, 3, 3, 3, 3]]],
        [[[1, 1, 1], [2], [3, 3, 3]]],
        2,
        0.75,
        0.25
    ),
])
def test_calculate_length_statistics(
    sources: List[List[List[int]]],
    targets: List[List[List[int]]],
    expected_num_sents: int,
    expected_mean: float,
    expected_std: float
) -> None:
    length_statistics: data_io.LengthStatistics = data_io.calculate_length_statistics(sources, targets, 5, 5)
    assert len(sources[0]) == len(targets[0])
    assert length_statistics.num_sents == expected_num_sents
    assert np.isclose(length_statistics.length_ratio_mean, expected_mean)
    assert np.isclose(length_statistics.length_ratio_std, expected_std)

@pytest.mark.parametrize('sources, targets', [
    (
        [
            [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
            [[1, 1, 1], [2, 2], [3, 3, 3]]
        ],
        [[[1, 1, 1], [2, 2, 2], [3, 3, 3]]]
    ),
])
def test_non_parallel_calculate_length_statistics(
    sources: List[List[List[int]]],
    targets: List[List[List[int]]]
) -> None:
    with pytest.raises(SockeyeError):
        data_io.calculate_length_statistics(sources, targets, 5, 5)

@pytest.mark.parametrize('end_of_prepending_tag', [None, '<EOP>'])
def test_get_training_data_iters(end_of_prepending_tag: Optional[str]) -> None:
    if end_of_prepending_tag:
        source_text_prefix_token: str = end_of_prepending_tag
        expected_mean: float = 0.9208152692746332
        expected_std: float = 0.0698611911724421
    else:
        source_text_prefix_token = ''
        expected_mean = 1.0
        expected_std = 0.0
    train_line_count: int = 100
    train_line_count_empty: int = 0
    train_max_length: int = 30
    dev_line_count: int = 20
    dev_max_length: int = 30
    test_line_count: int = 20
    test_line_count_empty: int = 0
    test_max_length: int = 30
    batch_size: int = 5
    num_source_factors: int = num_target_factors = 1
    with tmp_digits_dataset(
        'tmp_corpus',
        train_line_count,
        train_line_count_empty,
        train_max_length - C.SPACE_FOR_XOS,
        dev_line_count,
        dev_max_length - C.SPACE_FOR_XOS,
        test_line_count,
        test_line_count_empty,
        test_max_length - C.SPACE_FOR_XOS,
        source_text_prefix_token=source_text_prefix_token
    ) as data:
        vcb: vocab.Vocab = vocab.build_from_paths([data['train_source'], data['train_target']])
        train_iter, val_iter, config_data, data_info = data_io.get_training_data_iters(
            sources=[data['train_source']],
            targets=[data['train_target']],
            validation_sources=[data['dev_source']],
            validation_targets=[data['dev_target']],
            source_vocabs=[vcb],
            target_vocabs=[vcb],
            source_vocab_paths=[None],
            target_vocab_paths=[None],
            shared_vocab=True,
            batch_size=batch_size,
            batch_type=C.BATCH_TYPE_SENTENCE,
            max_seq_len_source=train_max_length,
            max_seq_len_target=train_max_length,
            bucketing=True,
            bucket_width=10,
            end_of_prepending_tag=end_of_prepending_tag
        )
        assert isinstance(train_iter, data_io.ParallelSampleIter)
        assert isinstance(val_iter, data_io.ParallelSampleIter)
        assert isinstance(config_data, data_io.DataConfig)
        assert data_info.sources == [data['train_source']]
        assert data_info.targets == [data['train_target']]
        assert data_info.source_vocabs == [None]
        assert data_info.target_vocabs == [None]
        assert config_data.data_statistics.max_observed_len_source == train_max_length
        if end_of_prepending_tag:
            assert config_data.data_statistics.max_observed_len_target == train_max_length - 1
        else:
            assert config_data.data_statistics.max_observed_len_target == train_max_length
        assert np.isclose(config_data.data_statistics.length_ratio_mean, expected_mean)
        assert np.isclose(config_data.data_statistics.length_ratio_std, expected_std)
        assert train_iter.batch_size == batch_size
        assert val_iter.batch_size == batch_size
        bos_id: int = vcb[C.BOS_SYMBOL]
        eos_id: int = vcb[C.EOS_SYMBOL]
        expected_first_target_symbols: torch.Tensor = torch.full((batch_size, 1), bos_id, dtype=torch.int32)
        for epoch in range(2):
            while train_iter.iter_next():
                batch: data_io.Batch = train_iter.next()
                assert isinstance(batch, data_io.Batch)
                source: torch.Tensor = batch.source
                target: torch.Tensor = batch.target
                label: torch.Tensor = batch.labels[C.TARGET_LABEL_NAME]
                length_ratio_label: torch.Tensor = batch.labels[C.LENRATIO_LABEL_NAME]
                assert source.shape[0] == target.shape[0] == label.shape[0] == batch_size
                assert source.shape[2] == target.shape[2] == num_source_factors == num_target_factors
                assert torch.sum(source == eos_id) == batch_size
                assert torch.equal(target[:, 0], expected_first_target_symbols)
                assert torch.equal(label[:, 0], target[:, 1, 0])
                assert torch.sum(label == eos_id) == batch_size
            train_iter.reset()

def _data_batches_equal(db1: data_io.Batch, db2: data_io.Batch) -> bool:
    equal: bool = True
    equal = equal and torch.allclose(db1.source, db2.source)
    equal = equal and torch.allclose(db1.source_length, db2.source_length)
    equal = equal and torch.allclose(db1.target, db2.target)
    equal = equal and torch.allclose(db1.target_length, db2.target_length)
    equal = equal and db1.labels.keys() == db2.labels.keys()
    equal = equal and db1.samples == db2.samples
    equal = equal and db1.tokens == db2.tokens
    return equal

def test_parallel_sample_iter() -> None:
    batch_size: int = 2
    buckets: List[Tuple[int, int]] = data_io.define_parallel_buckets(100, 100, 10, True, 1.0)
    bucket_counts: List[Optional[int]] = [0] + [None] * (len(buckets) - 1)
    bucket_batch_sizes: List[data_io.BucketBatchSize] = data_io.define_bucket_batch_sizes(
        buckets, batch_size, batch_type=C.BATCH_TYPE_SENTENCE, data_target_average_len=[None] * len(buckets)
    )
    dataset: data_io.ParallelDataSet = data_io.ParallelDataSet(*_get_random_bucketed_data(
        buckets=buckets,
        min_count=0,
        max_count=5,
        bucket_counts=bucket_counts
    ))
    it: data_io.ParallelSampleIter = data_io.ParallelSampleIter(dataset, buckets, batch_size, bucket_batch_sizes)
    with TemporaryDirectory() as work_dir:
        it.next()
        expected_batch: data_io.Batch = it.next()
        fname: str = os.path.join(work_dir, 'saved_iter')
        it.save_state(fname)
        it_loaded: data_io.ParallelSampleIter = data_io.ParallelSampleIter(dataset, buckets, batch_size, bucket_batch_sizes)
        it_loaded.reset()
        it_loaded.load_state(fname)
        loaded_batch: data_io.Batch = it_loaded.next()
        assert _data_batches_equal(expected_batch, loaded_batch)
        it.reset()
        expected_batch = it.next()
        it.save_state(fname)
        it_loaded = data_io.ParallelSampleIter(dataset, buckets, batch_size, bucket_batch_sizes)
        it_loaded.reset()
        it_loaded.load_state(fname)
        loaded_batch = it_loaded.next()
        assert _data_batches_equal(expected_batch, loaded_batch)
        it.reset()
        expected_batch = it.next()
        it.save_state(fname)
        it_loaded = data_io.ParallelSampleIter(dataset, buckets, batch_size, bucket_batch_sizes)
        it_loaded.reset()
        it_loaded.load_state(fname)
        loaded_batch = it_loaded.next()
        assert _data_batches_equal(expected_batch, loaded_batch)
        while it.iter_next():
            it.next()
            it_loaded.next()
        assert not it_loaded.iter_next()

def test_sharded_parallel_sample_iter() -> None:
    batch_size: int = 2
    buckets: List[Tuple[int, int]] = data_io.define_parallel_buckets(100, 100, 10, True, 1.0)
    bucket_counts: List[Optional[int]] = [0] + [None] * (len(buckets) - 1)
    bucket_batch_sizes: List[data_io.BucketBatchSize] = data_io.define_bucket_batch_sizes(
        buckets, batch_size, batch_type=C.BATCH_TYPE_SENTENCE, data_target_average_len=[None] * len(buckets)
    )
    dataset1: data_io.ParallelDataSet = data_io.ParallelDataSet(*_get_random_bucketed_data(
        buckets=buckets,
        min_count=0,
        max_count=5,
        bucket_counts=bucket_counts
    ))
    dataset2: data_io.ParallelDataSet = data_io.ParallelDataSet(*_get_random_bucketed_data(
        buckets=buckets,
        min_count=0,
        max_count=5,
        bucket_counts=bucket_counts
    ))
    with TemporaryDirectory() as work_dir:
        shard1_fname: str = os.path.join(work_dir, 'shard1')
        shard2_fname: str = os.path.join(work_dir, 'shard2')
        dataset1.save(shard1_fname)
        dataset2.save(shard2_fname)
        shard_fnames: List[str] = [shard1_fname, shard2_fname]
        it: data_io.ShardedParallelSampleIter = data_io.ShardedParallelSampleIter(
            shard_fnames, buckets, batch_size, bucket_batch_sizes
        )
        it.next()
        expected_batch: data_io.Batch = it.next()
        fname: str = os.path.join(work_dir, 'saved_iter')
        it.save_state(fname)
        it_loaded: data_io.ShardedParallelSampleIter = data_io.ShardedParallelSampleIter(
            shard_fnames, buckets, batch_size, bucket_batch_sizes
        )
        it_loaded.reset()
        it_loaded.load_state(fname)
        loaded_batch: data_io.Batch = it_loaded.next()
        assert _data_batches_equal(expected_batch, loaded_batch)
        it.reset()
        expected_batch = it.next()
        it.save_state(fname)
        it_loaded = data_io.ShardedParallelSampleIter(
            shard_fnames, buckets, batch_size, bucket_batch_sizes
        )
        it_loaded.reset()
        it_loaded.load_state(fname)
        loaded_batch = it_loaded.next()
        assert _data_batches_equal(expected_batch, loaded_batch)
        it.reset()
        expected_batch = it.next()
        it.save_state(fname)
        it_loaded = data_io.ShardedParallelSampleIter(
            shard_fnames, buckets, batch_size, bucket_batch_sizes
        )
        it_loaded.reset()
        it_loaded.load_state(fname)
        loaded_batch = it_loaded.next()
        assert _data_batches_equal(expected_batch, loaded_batch)
        while it.iter_next():
            it.next()
            it_loaded.next()
        assert not it_loaded.iter_next()

def test_sharded_parallel_sample_iter_num_batches() -> None:
    num_shards: int = 2
    batch_size: int = 2
    num_batches_per_bucket: int = 10
    buckets: List[Tuple[int, int]] = data_io.define_parallel_buckets(100, 100, 10, True, 1.0)
    bucket_counts: List[int] = [batch_size * num_batches_per_bucket for _ in buckets]
    num_batches_per_shard: int = num_batches_per_bucket * len(buckets)
    num_batches: int = num_shards * num_batches_per_shard
    bucket_batch_sizes: List[data_io.BucketBatchSize] = data_io.define_bucket_batch_sizes(
        buckets, batch_size, batch_type=C.BATCH_TYPE_SENTENCE, data_target_average_len=[None] * len(buckets)
    )
    dataset1: data_io.ParallelDataSet = data_io.ParallelDataSet(*_get_random_bucketed_data(
        buckets=buckets,
        min_count=0,
        max_count=5,
        bucket_counts=bucket_counts
    ))
    dataset2: data_io.ParallelDataSet = data_io.ParallelDataSet(*_get_random_bucketed_data(
        buckets=buckets,
        min_count=0,
        max_count=5,
        bucket_counts=bucket_counts
    ))
    with TemporaryDirectory() as work_dir:
        shard1_fname: str = os.path.join(work_dir, 'shard1')
        shard2_fname: str = os.path.join(work_dir, 'shard2')
        dataset1.save(shard1_fname)
        dataset2.save(shard2_fname)
        shard_fnames: List[str] = [shard1_fname, shard2_fname]
        it: data_io.ShardedParallelSampleIter = data_io.ShardedParallelSampleIter(
            shard_fnames, buckets, batch_size, bucket_batch_sizes
        )
        num_batches_seen: int = 0
        while it.iter_next():
            it.next()
            num_batches_seen += 1
        assert num_batches_seen == num_batches

def test_sharded_and_parallel_iter_same_num_batches() -> None:
    """ Tests that a sharded data iterator with just a single shard produces as many shards as an iterator directly
    using the same dataset. """
    batch_size: int = 2
    num_batches_per_bucket: int = 10
    buckets: List[Tuple[int, int]] = data_io.define_parallel_buckets(100, 100, 10, True, 1.0)
    bucket_counts: List[int] = [batch_size * num_batches_per_bucket for _ in buckets]
    num_batches: int = num_batches_per_bucket * len(buckets)
    bucket_batch_sizes: List[data_io.BucketBatchSize] = data_io.define_bucket_batch_sizes(
        buckets, batch_size, batch_type=C.BATCH_TYPE_SENTENCE, data_target_average_len=[None] * len(buckets)
    )
    dataset: data_io.ParallelDataSet = data_io.ParallelDataSet(*_get_random_bucketed_data(
        buckets=buckets,
        min_count=0,
        max_count=5,
        bucket_counts=bucket_counts
    ))
    with TemporaryDirectory() as work_dir:
        shard_fname: str = os.path.join(work_dir, 'shard1')
        dataset.save(shard_fname)
        shard_fnames: List[str] = [shard_fname]
        it_sharded: data_io.ShardedParallelSampleIter = data_io.ShardedParallelSampleIter(
            shard_fnames, buckets, batch_size, bucket_batch_sizes
        )
        it_parallel: data_io.ParallelSampleIter = data_io.ParallelSampleIter(
            dataset, buckets, batch_size, bucket_batch_sizes
        )
        num_batches_seen: int = 0
        while it_parallel.iter_next():
            assert it_sharded.iter_next()
            it_parallel.next()
            it_sharded.next()
            num_batches_seen += 1
        assert num_batches_seen == num_batches
        print('Resetting...')
        it_sharded.reset()
        it_parallel.reset()
        num_batches_seen = 0
        while it_parallel.iter_next():
            assert it_sharded.iter_next()
            it_parallel.next()
            it_sharded.next()
            num_batches_seen += 1
        assert num_batches_seen == num_batches

def test_create_target_and_shifted_label_sequences() -> None:
    target_and_label: torch.Tensor = torch.tensor([
        [C.BOS_ID, 4, 17, 35, 12, C.EOS_ID, C.PAD_ID, C.PAD_ID],
        [C.BOS_ID, 15, 23, 23, 77, 55, 22, C.EOS_ID],
        [C.BOS_ID, 4, C.EOS_ID, C.PAD_ID, C.PAD_ID, C.PAD_ID, C.PAD_ID, C.PAD_ID]
    ])
    expected_label: torch.Tensor = torch.tensor([
        [4, 17, 35, 12, C.EOS_ID, C.PAD_ID, C.PAD_ID],
        [15, 23, 23, 77, 55, 22, C.EOS_ID],
        [4, C.EOS_ID, C.PAD_ID, C.PAD_ID, C.PAD_ID, C.PAD_ID, C.PAD_ID]
    ]).unsqueeze(2)
    expected_target: torch.Tensor = torch.tensor([
        [C.BOS_ID, 4, 17, 35, 12, C.PAD_ID, C.PAD_ID],
        [C.BOS_ID, 15, 23, 23, 77, 55, 22],
        [C.BOS_ID, 4, C.PAD_ID, C.PAD_ID, C.PAD_ID, C.PAD_ID, C.PAD_ID]
    ]).unsqueeze(2)
    target_and_label = torch.unsqueeze(target_and_label, dim=2)
    expected_lengths: torch.Tensor = torch.tensor([5, 7, 2])
    target, label = data_io.create_target_and_shifted_label_sequences(target_and_label)
    assert target.shape[0] == label.shape[0] == target_and_label.shape[0]
    assert target.shape[1] == label.shape[1] == target_and_label.shape[1] - 1
    assert torch.allclose(target, expected_target)
    assert torch.allclose(label, expected_label)
    lengths: torch.Tensor = (target != C.PAD_ID).sum(dim=1).squeeze()
    assert torch.allclose(lengths, expected_lengths)
