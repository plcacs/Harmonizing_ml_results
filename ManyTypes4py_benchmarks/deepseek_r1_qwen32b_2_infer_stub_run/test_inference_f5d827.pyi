import itertools
import json
import numpy as np
import pytest
import torch as pt
import sockeye.beam_search
import sockeye.constants as C
import sockeye.data_io
import sockeye.inference
import sockeye.lexicon
import sockeye.model
import sockeye.utils

_BOS: int = ...
_EOS: int = ...

def mock_translator(batch_size: int = 1, beam_size: int = 5, nbest_size: int = 1, num_source_factors: int = 1, dtype: pt.dtype = pt.float32) -> sockeye.inference.Translator:
    ...

def test_concat_translations(lp_alpha: float, lp_beta: float, bp_weight: float) -> None:
    ...

def test_translator_input(sentence_id: int, sentence: str, factors: list[list[str]] | None, chunk_size: int) -> None:
    ...

def test_translator_input_with_source_prefix(sentence_id: int, sentence: str, factors: list[list[str]] | None, chunk_size: int, source_prefix: str, source_prefix_factors: list[list[str]] | None) -> None:
    ...

def test_get_max_input_output_length(supported_max_seq_len_source: int, supported_max_seq_len_target: int, forced_max_input_len: int | None, forced_max_output_len: int | None, length_ratio_mean: float, length_ratio_std: float, expected_max_input_len: int, expected_max_output_len: int) -> None:
    ...

def test_make_input_from_factored_string(sentence: str, num_expected_factors: int, delimiter: str, expected_tokens: list[str], expected_factors: list[list[str]] | None) -> None:
    ...

def test_factor_parsing(sentence: str, num_expected_factors: int, delimiter: str) -> None:
    ...

def test_make_input_whitespace_delimiter(delimiter: str) -> None:
    ...

def test_make_input_from_valid_json_string(text: str, factors: list[str] | None) -> None:
    ...

def test_make_input_from_valid_json_string_restrict_lexicon() -> None:
    ...

def test_failed_make_input_from_valid_json_string(text: str, text_key: str, factors: list[str] | None, factors_key: str) -> None:
    ...

def test_make_input_from_valid_dict(text: str, factors: list[str] | None) -> None:
    ...

def test_failed_make_input_from_valid_dict(text: str, text_key: str, factors: list[str] | None, factors_key: str) -> None:
    ...

def test_make_input_from_multiple_strings(strings: list[str]) -> None:
    ...

def test_get_best_word_indices_for_kth_hypotheses() -> None:
    ...

def test_get_best_translations(expected_best_ids: np.ndarray, expected_best_indices: np.ndarray) -> None:
    ...

def test_unshift_target_factors(sequence: np.ndarray, fill_with: int, expected_sequence: list[int] | list[list[int]]) -> None:
    ...