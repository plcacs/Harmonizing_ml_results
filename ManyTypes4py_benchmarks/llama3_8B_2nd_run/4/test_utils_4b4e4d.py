import gzip
import math
import os
import re
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import pytest
import torch as pt
from sockeye import __version__
from sockeye import constants as C
from sockeye import utils

@pytest.mark.parametrize('some_list, expected', [([1, 2, 3, 4, 5, 6, 7, 8], [[1, 2, 3], [4, 5, 6], [7, 8]]), ([1, 2], [[1, 2]]), ([1, 2, 3], [[1, 2, 3]]), ([1, 2, 3, 4], [[1, 2, 3], [4]])])
def test_chunks(some_list: list[int], expected: list[list[int]]) -> None:
    chunk_size: int = 3
    chunked_list: list[list[int]] = list(utils.chunks(some_list, chunk_size))
    assert chunked_list == expected

def test_check_condition_true() -> None:
    utils.check_condition(1 == 1, 'Nice')

def test_check_condition_false() -> None:
    with pytest.raises(utils.SockeyeError) as e:
        utils.check_condition(1 == 2, 'Wrong')
    assert 'Wrong' == str(e.value)

@pytest.mark.parametrize('version_string,expected_version', [('1.0.3', ('1', '0', '3')), ('1.0.2.3', ('1', '0', '2.3'))])
def test_parse_version(version_string: str, expected_version: tuple[str, str, str]) -> None:
    assert expected_version == utils.parse_version(version_string)

def test_check_version_disregards_minor() -> None:
    release: str
    major: str
    minor: str
    release, major, minor = utils.parse_version(__version__)
    other_minor_version: str = '%s.%s.%d' % (release, major, int(minor) + 1)
    utils.check_version(other_minor_version)

def _get_later_major_version() -> str:
    release: str
    major: str
    minor: str
    release, major, minor = utils.parse_version(__version__)
    return '%s.%d.%s' % (release, int(major) + 1, minor)

def test_check_version_checks_major() -> None:
    version: str = _get_later_major_version()
    with pytest.raises(utils.SockeyeError) as e:
        utils.check_version(version)
    assert 'Given major version (%s) does not match major code version (%s)' % (version, __version__) == str(e.value)

def test_version_matches_changelog() -> None:
    """
    Tests whether the last version mentioned in CHANGELOG.md matches the sockeye version (sockeye/__init__.py).
    """
    pattern: re.Pattern = re.compile('## \\[([0-9.]+)\\]')
    changelog: str = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'CHANGELOG.md')).read()
    last_changelog_version: str = pattern.findall(changelog)[0]
    assert __version__ == last_changelog_version

@pytest.mark.parametrize('samples, expected_mean, expected_variance', [([1, 2], 1.5, 0.25), ([4.0, 100.0, 12.0, -3, 1000, 1.0, -200], 130.57142857142858, 132975.38775510204)])
def test_online_mean_and_variance(samples: list[float], expected_mean: float, expected_variance: float) -> None:
    mean_and_variance: utils.OnlineMeanAndVariance = utils.OnlineMeanAndVariance()
    for sample in samples:
        mean_and_variance.update(sample)
    assert np.isclose(mean_and_variance.mean, expected_mean)
    assert np.isclose(mean_and_variance.variance, expected_variance)

@pytest.mark.parametrize('samples, expected_mean', [([], 0.0), ([5.0], 5.0)])
def test_online_mean_and_variance_nan(samples: list[float], expected_mean: float) -> None:
    mean_and_variance: utils.OnlineMeanAndVariance = utils.OnlineMeanAndVariance()
    for sample in samples:
        mean_and_variance.update(sample)
    assert np.isclose(mean_and_variance.mean, expected_mean)
    assert math.isnan(mean_and_variance.variance)

get_tokens_tests = [('this is a line  \n', ['this', 'is', 'a', 'line']), (' a  \tb \r \n', ['a', 'b'])]

@pytest.mark.parametrize('line, expected_tokens', get_tokens_tests)
def test_get_tokens(line: str, expected_tokens: list[str]) -> None:
    tokens: list[str] = list(utils.get_tokens(line))
    assert tokens == expected_tokens

@pytest.mark.parametrize('samples, sample_means, expected_mean', [([[1.23, 0.474, 9.516], [10.219, 5.31, 9, 21.9, 98]], [3.74, 28.8858], 19.456125), ([[-10, 10, 4.3, -4.3], [102], [0, 1]], [0.0, 102.0, 0.5], 14.714285714285714), ([[], [-1], [0, 1]], [None, -1.0, 0.5], 0.0), ([[], [1.99], [], [], [0]], [None, 1.99, None, None, 0.0], 0.995), ([[2.45, -5.21, -20, 81.92, 41, 1, 0.1123, 1.2], []], [12.8090375, None], 12.8090375)])
def test_combine_means(samples: list[list[float]], sample_means: list[float], expected_mean: float) -> None:
    num_sents: list[int] = [len(l) for l in samples]
    combined_mean: float = utils.combine_means(sample_means, num_sents)
    assert np.isclose(expected_mean, combined_mean)

def test_average_tensors() -> None:
    n: int = 4
    shape: tuple[int, int] = (12, 14)
    arrays: list[pt.Tensor] = [pt.rand(*shape) for _ in range(n)]
    expected_average: pt.Tensor = pt.zeros(*shape)
    for array in arrays:
        expected_average += array
    expected_average /= 4
    pt.testing.assert_close(utils.average_tensors(arrays), expected_average)

@pytest.mark.parametrize('new, old, metric, result', [(0, 0, C.PERPLEXITY, False), (1.0, 1.0, C.PERPLEXITY, False), (1.0, 0.9, C.PERPLEXITY, False), (0.99, 1.0, C.PERPLEXITY, True), (C.LARGE_POSITIVE_VALUE, np.inf, C.PERPLEXITY, True), (0, 0, C.BLEU, False), (1.0, 1.0, C.BLEU, False), (1.0, 0.9, C.BLEU, True), (0.99, 1.0, C.BLEU, False), (C.LARGE_POSITIVE_VALUE, np.inf, C.BLEU, False)])
def test_metric_value_is_better(new: float, old: float, metric: str, result: bool) -> None:
    assert utils.metric_value_is_better(new, old, metric) == result

def test_write_read_metric_file() -> None:
    expected_metrics: list[dict[str, float]] = [{'float_metric': 3.45, 'bool_metric': True}, {'float_metric': 1.0, 'bool_metric': False}]
    with TemporaryDirectory(prefix='metric_file') as work_dir:
        metric_path: str = os.path.join(work_dir, 'metrics')
        utils.write_metrics_file(expected_metrics, metric_path)
        read_metrics: list[dict[str, float]] = utils.read_metrics_file(metric_path)
    assert len(read_metrics) == len(expected_metrics)
    assert expected_metrics == read_metrics

def test_adjust_first_step_masking() -> None:
    first_step_mask: pt.Tensor = pt.tensor([[0.0], [np.inf], [np.inf], [np.inf], [0.0], [np.inf], [np.inf], [np.inf]])
    target_prefix: pt.Tensor = pt.tensor([[1, 2], [1, 0]])
    adjust_first_step_mask: pt.Tensor = pt.tensor([[0.0, 0.0, 0.0], [np.inf, np.inf, 0.0], [np.inf, np.inf, 0.0], [np.inf, np.inf, 0.0], [0.0, 0.0, 0.0], [np.inf, np.inf, 0.0], [np.inf, np.inf, 0.0], [np.inf, np.inf, 0.0]])
    assert pt.equal(adjust_first_step_mask, utils.adjust_first_step_masking(target_prefix, first_step_mask)) == True

def test_count_seq_len() -> None:
    sample: str = '▁Bonob os , ▁like ▁humans , ▁love ▁to ▁play ▁throughout ▁their ▁entire ▁lives . Parliament Does Not Support Amendment Fre@@ eing Ty@@ mo@@ sh@@ en@@ ko'
    count_type: str = C.SEQ_LEN_IN_CHARACTERS
    replace_tokens: str = C.TOKEN_SEGMENTATION_MARKERS
    expected_seq_len: int = 106
    assert utils.count_seq_len(sample, count_type, replace_tokens) == expected_seq_len

def test_rerank_hypotheses_isometric() -> None:
    hypothesis: str = 'No Liber@@ ation for Ty@@ mo@@ sh@@ en@@ ko by Parliament'
    hypothesis_score: float = 0.377
    source: str = 'El Parlamento no lib@@ era a Ty@@ mo@@ sh@@ en@@ ko'
    metric: str = 'isometric-ratio'
    alpha: float = 0.7
    expected_score: float = 0.4322176470588236
    assert utils.compute_isometric_score(hypothesis, hypothesis_score, source, metric, alpha) == expected_score

def test_update_dict_with_prefix_kv() -> None:
    dest: dict[str, dict[str, float]] = {}
    prefix_kv: dict[str, float] = {'a': 1, 'b.c': 2, 'b.d': 3, 'e.f.g': 4, 'e.z': 5}
    expected: dict[str, dict[str, float]] = {'a': 1, 'b': {'c': 2, 'd': 3}, 'e': {'f': {'g': 4}, 'z': 5}}
    utils.update_dict_with_prefix_kv(dest, prefix_kv)
    assert dest == expected

@unittest.mock.patch('time.sleep')
def test_fault_tolerant_symlink(mock_sleep) -> None:
    with TemporaryDirectory() as temp:
        src_fname: str = os