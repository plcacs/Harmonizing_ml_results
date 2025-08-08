from typing import List, Tuple, Dict, Any
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

def test_chunks(some_list: List[int], expected: List[List[int]]) -> None:
def test_check_condition_true() -> None:
def test_check_condition_false() -> None:
def test_parse_version(version_string: str, expected_version: Tuple[str, str, str]) -> None:
def test_check_version_disregards_minor() -> None:
def _get_later_major_version() -> str:
def test_check_version_checks_major() -> None:
def test_version_matches_changelog() -> None:
def test_online_mean_and_variance(samples: List[float], expected_mean: float, expected_variance: float) -> None:
def test_online_mean_and_variance_nan(samples: List[float], expected_mean: float) -> None:
def test_get_tokens(line: str, expected_tokens: List[str]) -> None:
def test_combine_means(samples: List[List[float]], sample_means: List[float], expected_mean: float) -> None:
def test_combine_stds(samples: List[List[float]], sample_means: List[float], sample_stds: List[float], expected_std: float) -> None:
def test_average_tensors() -> None:
def test_metric_value_is_better(new: float, old: float, metric: str, result: bool) -> None:
def _touch_file(fname: str, compressed: bool, empty: bool) -> str:
def test_is_gzip_file() -> None:
def test_smart_open_without_suffix() -> None:
def test_parse_metrics_line(line_num: int, line: str, expected_metrics: Dict[str, Any]) -> None:
def test_write_read_metric_file() -> None:
def test_adjust_first_step_masking() -> None:
def test_count_seq_len(sample: str, count_type: str, replace_tokens: str, expected_seq_len: int) -> None:
def test_rerank_hypotheses_isometric(hypothesis: str, hypothesis_score: float, source: str, metric: str, alpha: float, expected_score: float) -> None:
def test_update_dict_with_prefix_kv(dest: Dict[str, Any], prefix_kv: Dict[str, Any], expected: Dict[str, Any]) -> None:
def test_fault_tolerant_symlink() -> None:
