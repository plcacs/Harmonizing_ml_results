import gzip
import math
import os
import re
from tempfile import TemporaryDirectory
import unittest
from typing import List, Tuple, Dict, Any
import numpy as np
import pytest
import torch as pt
from sockeye import __version__
from sockeye import constants as C
from sockeye import utils

def test_chunks(some_list: List[int], expected: List[List[int]]) -> None:
    chunk_size: int = 3
    chunked_list: List[List[int]] = list(utils.chunks(some_list, chunk_size))
    assert chunked_list == expected

def test_check_condition_true() -> None:
    utils.check_condition(1 == 1, 'Nice')

def test_check_condition_false() -> None:
    with pytest.raises(utils.SockeyeError) as e:
        utils.check_condition(1 == 2, 'Wrong')
    assert 'Wrong' == str(e.value)

def test_parse_version(version_string: str, expected_version: Tuple[str, str, str]) -> None:
    assert expected_version == utils.parse_version(version_string)

def test_check_version_disregards_minor() -> None:
    release, major, minor = utils.parse_version(__version__)
    other_minor_version = '%s.%s.%d' % (release, major, int(minor) + 1)
    utils.check_version(other_minor_version)

def _get_later_major_version() -> str:
    release, major, minor = utils.parse_version(__version__)
    return '%s.%d.%s' % (release, int(major) + 1, minor)

def test_check_version_checks_major() -> None:
    version = _get_later_major_version()
    with pytest.raises(utils.SockeyeError) as e:
        utils.check_version(version)
    assert 'Given major version (%s) does not match major code version (%s)' % (version, __version__) == str(e.value)

def test_version_matches_changelog() -> None:
    pattern = re.compile('## \\[([0-9.]+)\\]')
    changelog = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'CHANGELOG.md')).read()
    last_changelog_version = pattern.findall(changelog)[0]
    assert __version__ == last_changelog_version

def test_online_mean_and_variance(samples: List[float], expected_mean: float, expected_variance: float) -> None:
    mean_and_variance = utils.OnlineMeanAndVariance()
    for sample in samples:
        mean_and_variance.update(sample)
    assert np.isclose(mean_and_variance.mean, expected_mean)
    assert np.isclose(mean_and_variance.variance, expected_variance)

def test_online_mean_and_variance_nan(samples: List[float], expected_mean: float) -> None:
    mean_and_variance = utils.OnlineMeanAndVariance()
    for sample in samples:
        mean_and_variance.update(sample)
    assert np.isclose(mean_and_variance.mean, expected_mean)
    assert math.isnan(mean_and_variance.variance)

get_tokens_tests: List[Tuple[str, List[str]]] = [('this is a line  \n', ['this', 'is', 'a', 'line']), (' a  \tb \r \n', ['a', 'b'])]

def test_get_tokens(line: str, expected_tokens: List[str]) -> None:
    tokens = list(utils.get_tokens(line))
    assert tokens == expected_tokens

def test_combine_means(samples: List[List[float]], sample_means: List[float], expected_mean: float) -> None:
    num_sents: List[int] = [len(l) for l in samples]
    combined_mean = utils.combine_means(sample_means, num_sents)
    assert np.isclose(expected_mean, combined_mean)

def test_combine_stds(samples: List[List[float]], sample_means: List[float], sample_stds: List[float], expected_std: float) -> None:
    num_sents: List[int] = [len(l) for l in samples]
    combined_std = utils.combine_stds(sample_stds, sample_means, num_sents)
    assert np.isclose(expected_std, combined_std)

def test_average_tensors() -> None:
    n: int = 4
    shape: Tuple[int, int] = (12, 14)
    arrays: List[pt.Tensor] = [pt.rand(12, 14) for _ in range(n)]
    expected_average = pt.zeros(*shape)
    for array in arrays:
        expected_average += array
    expected_average /= 4
    pt.testing.assert_close(utils.average_tensors(arrays), expected_average)
    with pytest.raises(utils.SockeyeError) as e:
        other_shape: Tuple[int, int] = (12, 13)
        utils.average_tensors(arrays + [pt.zeros(*other_shape)])
    assert 'tensor shapes do not match' == str(e.value)

def test_metric_value_is_better(new: float, old: float, metric: str, result: bool) -> None:
    assert utils.metric_value_is_better(new, old, metric) == result

def _touch_file(fname: str, compressed: bool, empty: bool) -> str:
    if compressed:
        open_func = gzip.open
    else:
        open_func = open
    with open_func(fname, encoding='utf8', mode='wt') as f:
        if not empty:
            for i in range(10):
                print(str(i), file=f)
    return fname

def test_is_gzip_file() -> None:
    with TemporaryDirectory() as temp:
        fname = os.path.join(temp, 'test')
        assert utils.is_gzip_file(_touch_file(fname, compressed=True, empty=True))
        assert utils.is_gzip_file(_touch_file(fname, compressed=True, empty=False))
        assert not utils.is_gzip_file(_touch_file(fname, compressed=False, empty=True))
        assert not utils.is_gzip_file(_touch_file(fname, compressed=False, empty=False))

def test_smart_open_without_suffix() -> None:
    with TemporaryDirectory() as temp:
        fname = os.path.join(temp, 'test')
        _touch_file(fname, compressed=True, empty=False)
        with utils.smart_open(fname) as fin:
            assert len(fin.readlines()) == 10
        _touch_file(fname, compressed=False, empty=False)
        with utils.smart_open(fname) as fin:
            assert len(fin.readlines()) == 10

def test_parse_metrics_line(line_num: int, line: str, expected_metrics: Dict[str, Any]) -> None:
    if line_num == int(line.split('\t')[0]):
        parsed_metrics = utils.parse_metrics_line(line_num, line)
        for k, v in parsed_metrics.items():
            assert isinstance(v, type(expected_metrics[k]))
            assert v == expected_metrics[k]
    else:
        with pytest.raises(utils.SockeyeError) as e:
            utils.parse_metrics_line(line_num, line)

def test_write_read_metric_file() -> None:
    expected_metrics = [{'float_metric': 3.45, 'bool_metric': True}, {'float_metric': 1.0, 'bool_metric': False}]
    with TemporaryDirectory(prefix='metric_file') as work_dir:
        metric_path = os.path.join(work_dir, 'metrics')
        utils.write_metrics_file(expected_metrics, metric_path)
        read_metrics = utils.read_metrics_file(metric_path)
    assert len(read_metrics) == len(expected_metrics)
    assert expected_metrics == read_metrics

def test_adjust_first_step_masking() -> None:
    first_step_mask = pt.tensor([[0.0], [np.inf], [np.inf], [np.inf], [0.0], [np.inf], [np.inf], [np.inf]])
    target_prefix = pt.tensor([[1, 2], [1, 0]])
    adjust_first_step_mask = pt.tensor([[0.0, 0.0, 0.0], [np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf], [0.0, 0.0, 0.0], [np.inf, np.inf, 0.0], [np.inf, np.inf, 0.0], [np.inf, np.inf, 0.0]])
    assert pt.equal(adjust_first_step_mask, utils.adjust_first_step_masking(target_prefix, first_step_mask)) == True
    target_prefix = pt.tensor([[1, 0], [2, 3]])
    adjust_first_step_mask = pt.tensor([[0.0, 0.0, 0.0], [np.inf, np.inf, 0.0], [np.inf, np.inf, 0.0], [np.inf, np.inf, 0.0], [0.0, 0.0, 0.0], [np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf]])
    assert pt.equal(adjust_first_step_mask, utils.adjust_first_step_masking(target_prefix, first_step_mask)) == True
    target_prefix = pt.tensor([[1, 0, 0], [2, 3, 4]])
    adjust_first_step_mask = pt.tensor([[0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, 0.0, 0.0], [np.inf, np.inf, 0.0, 0.0], [np.inf, np.inf, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf, np.inf]])
    assert pt.equal(adjust_first_step_mask, utils.adjust_first_step_masking(target_prefix, first_step_mask)) == True
    target_prefix = pt.tensor([[1, 0, 0, 0], [2, 3, 4, 5]])
    adjust_first_step_mask = pt.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, 0.0, 0.0, 0.0], [np.inf, np.inf, 0.0, 0.0, 0.0], [np.inf, np.inf, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf]])
    assert pt.equal(adjust_first_step_mask, utils.adjust_first_step_masking(target_prefix, first_step_mask)) == True
    first_step_mask = pt.tensor([[0.0], [np.inf], [np.inf], [np.inf], [0.0], [np.inf], [np.inf], [np.inf], [0.0], [np.inf], [np.inf], [np.inf]])
    target_prefix = pt.tensor([[1, 0, 0, 0], [1, 0, 0, 0], [2, 3, 4, 5]])
    adjust_first_step_mask = pt.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, 0.0, 0.0, 0.0], [np.inf, np.inf, 0.0, 0.0, 0.0], [np.inf, np.inf, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, 0.0, 0.0, 0.0], [np.inf, np.inf, 0.0, 0.0, 0.0], [np.inf, np.inf, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf]])
    assert pt.equal(adjust_first_step_mask, utils.adjust_first_step_masking(target_prefix, first_step_mask)) == True
    target_prefix = pt.tensor([[1, 0, 0, 0], [1, 3, 0, 0], [2, 3, 4, 5]])
    adjust_first_step_mask = pt.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, 0.0, 0.0, 0.0], [np.inf, np.inf, 0.0, 0.0, 0.0], [np.inf, np.inf, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, np.inf, 0.0, 0.0], [np.inf, np.inf, np.inf, 0.0, 0.0], [np.inf, np.inf, np.inf, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf]])
    assert pt.equal(adjust_first_step_mask, utils.adjust_first_step_masking(target_prefix, first_step_mask)) == True
    target_prefix = pt.tensor([[0, 0, 0, 0], [1, 3, 0, 0], [2, 3, 4, 5]])
    adjust_first_step_mask = pt.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [np.inf, 0.0, 0.0, 0.0, 0.0], [np.inf, 0.0, 0.0, 0.0, 0.0], [np.inf, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, np.inf, 0.0, 0.0], [np.inf, np.inf, np.inf, 0.0, 0.0], [np.inf, np.inf, np.inf, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf]])
    assert pt.equal(adjust_first_step_mask, utils.adjust_first_step_masking(target_prefix, first_step_mask)) == True
    target_prefix = pt.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [2, 3, 4, 5]])
    adjust_first_step_mask = pt.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [np.inf, 0.0, 0.0, 0.0, 0.0], [np.inf, 0.0, 0.0, 0.0, 0.0], [np.inf, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [np.inf, 0.0, 0.0, 0.0, 0.0], [np.inf, 0.0, 0.0, 0.0, 0.0], [np.inf, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf]])
    assert pt.equal(adjust_first_step_mask, utils.adjust_first_step_masking(target_prefix, first_step_mask)) == True

def test_count_seq_len(sample: str, count_type: str, replace_tokens: str, expected_seq_len: int) -> None:
    assert utils.count_seq_len(sample, count_type, replace_tokens) == expected_seq_len

def test_rerank_hypotheses_isometric(hypothesis: str, hypothesis_score: float, source: str, metric: str, alpha: float, expected_score: float) -> None:
    assert utils.compute_isometric_score(hypothesis, hypothesis_score, source, metric, alpha) == expected_score

def test_update_dict_with_prefix_kv(dest: Dict[str, Any], prefix_kv: Dict[str, Any], expected: Dict[str, Any]) -> None:
    utils.update_dict_with_prefix_kv(dest, prefix_kv)
    assert dest == expected

@unittest.mock.patch('time.sleep')
def test_fault_tolerant_symlink(mock_sleep) -> None:
    with TemporaryDirectory() as temp:
        src_fname = os.path.join(temp, 'src')
        dst_fname = os.path.join(temp, 'dst')
        _touch_file(src_fname, compressed=False, empty=False)
        utils.fault_tolerant_symlink(src_fname, dst_fname)
        with pytest.raises(OSError):
            utils.fault_tolerant_symlink(src_fname, dst_fname)
        assert mock_sleep.called
        with utils.smart_open(src_fname) as src_in, utils.smart_open(dst_fname) as dst_in:
            assert src_in.readlines() == dst_in.readlines()
