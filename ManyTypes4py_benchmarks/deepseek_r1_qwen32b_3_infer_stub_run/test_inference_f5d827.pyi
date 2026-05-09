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

_BOS: int = 0
_EOS: int = -1

def mock_translator(batch_size: int = 1, beam_size: int = 5, nbest_size: int = 1, num_source_factors: int = 1, dtype: pt.dtype = pt.float32) -> sockeye.inference.Translator:
    ...

@pytest.mark.parametrize('lp_alpha, lp_beta, bp_weight', [(1.0, 0.0, 0.0), (1.0, 2.0, 0.0), (1.0, 2.0, 4.0), (1.0, 0.0, 5.0)])
def test_concat_translations(lp_alpha: float, lp_beta: float, bp_weight: float) -> None:
    ...

@pytest.mark.parametrize('sentence_id, sentence, factors, chunk_size', [(1, 'a test', None, 4), (1, 'a test', None, 2), (1, 'a test', None, 1), (0, '', None, 1), (1, 'a test', [['h', 'l']], 4), (1, 'a test', [['h', 'h'], ['x', 'y']], 1)])
def test_translator_input(sentence_id: int, sentence: str, factors: list[list[str]] | None, chunk_size: int) -> None:
    ...

@pytest.mark.parametrize('sentence_id, sentence, factors, chunk_size, source_prefix, source_prefix_factors', [(1, 'a test', None, 4, 'prefix test', None), (1, 'a test', None, 2, 'prefix test', None), (1, 'a test', None, 1, 'prefix test', None), (0, '', None, 1, '', None), (1, 'a test', [['h', 'l']], 4, 'prefix test', [['h', 'l']]), (1, 'a test', [['h', 'h'], ['x', 'y']], 1, 'prefix test', [['h', 'h'], ['x', 'y']])])
def test_translator_input_with_source_prefix(sentence_id: int, sentence: str, factors: list[list[str]] | None, chunk_size: int, source_prefix: str, source_prefix_factors: list[list[str]] | None) -> None:
    ...

@pytest.mark.parametrize('supported_max_seq_len_source, supported_max_seq_len_target, forced_max_input_len, forced_max_output_len, length_ratio_mean, length_ratio_std, expected_max_input_len, expected_max_output_len', [(99 + 1, 99 + 1, None, None, 1.0, 0.0, 100, 100), (99 + 1, 99 + 1, None, None, 0.9, 0.2, 100, 111), (99 + 1, 99 + 1, None, None, 1.1, 0.2, 100, 130), (99 + 1, 99 + 1, 50, None, 1.1, 0.2, 51, 67), (99 + 1, 99 + 1, 50, None, 1.1, 0.2, 51, 67), (99 + 1, 99 + 1, 50, 80, 1.1, 0.2, 51, 81)])
def test_get_max_input_output_length(supported_max_seq_len_source: int, supported_max_seq_len_target: int, forced_max_input_len: int | None, forced_max_output_len: int | None, length_ratio_mean: float, length_ratio_std: float, expected_max_input_len: int, expected_max_output_len: int) -> None:
    ...

@pytest.mark.parametrize('sentence, num_expected_factors, delimiter, expected_tokens, expected_factors', [('this is a test', 1, '|', ['this', 'is', 'a', 'test'], None), ('this|X is| a|X test|', 1, '|', ['this|X', 'is|', 'a|X', 'test|'], None), ('space   space', 1, '|', ['space', 'space'], None), ('', 1, '|', [], None), ('', 2, '|', [], [[]]), ('a|l b|l C|u', 2, '|', ['a', 'b', 'C'], [['l', 'l', 'u']]), ('a-X-Y b-Y-X', 3, '-', ['a', 'b'], [['X', 'Y'], ['Y', 'X']]), ('a-X-Y ', 3, '-', ['a'], [['X'], ['Y']])])
def test_make_input_from_factored_string(sentence: str, num_expected_factors: int, delimiter: str, expected_tokens: list[str], expected_factors: list[list[str]] | None) -> None:
    ...

@pytest.mark.parametrize('sentence, num_expected_factors, delimiter', [('this is a test', 2, '|'), ('this|X is a test', 2, '|'), ('this|X is|X a|X test', 2, '|'), ('this| is|X a|X test|', 2, '|'), ('this|X is|X a|X test|', 2, '|'), ('w1||w2||f22', 2, '|'), ('this', 2, '|'), ('this|', 2, '|'), ('this||', 3, '|'), ('this|| another', 2, '|'), ('this|||', 2, '|'), ('|this', 2, '|'), ('|this|that', 3, '|'), ('|this|that|', 4, '|')])
def test_factor_parsing(sentence: str, num_expected_factors: int, delimiter: str) -> None:
    ...

@pytest.mark.parametrize('delimiter', ['\t', '\t \t', '\t\t', '\n', '\r', '\r\n', ' ', '\n\n', '  ', ' \t', '\x0c', '\x0b', '\xa0', '\u1680', '\u2000', None, '', '\u200a', '\u205f', '\u3000'])
def test_make_input_whitespace_delimiter(delimiter: str) -> None:
    ...

@pytest.mark.parametrize('text, factors', [('this is a test without factors', None), ('', None), ('test', ['X', 'X']), ('a b c', ['x y z']), ('a', [])])
def test_make_input_from_valid_json_string(text: str, factors: list[str] | None) -> None:
    ...

def test_make_input_from_valid_json_string_restrict_lexicon() -> None:
    ...

@pytest.mark.parametrize('text, text_key, factors, factors_key', [('a', 'blub', None, '')])
def test_failed_make_input_from_valid_json_string(text: str, text_key: str, factors: list[str] | None, factors_key: str) -> None:
    ...

@pytest.mark.parametrize('text, factors', [('this is a test without factors', None), ('', None), ('test', ['X', 'X']), ('a b c', ['x y z']), ('a', [])])
def test_make_input_from_valid_dict(text: str, factors: list[str] | None) -> None:
    ...

@pytest.mark.parametrize('text, text_key, factors, factors_key', [('a', 'blub', None, '')])
def test_failed_make_input_from_valid_dict(text: str, text_key: str, factors: list[str] | None, factors_key: str) -> None:
    ...

@pytest.mark.parametrize('strings', [['a b c'], ['a b c', 'f1 f2 f3', 'f3 f3 f3']])
def test_make_input_from_multiple_strings(strings: list[str]) -> None:
    ...

def test_get_best_word_indices_for_kth_hypotheses() -> None:
    ...

@pytest.mark.parametrize('expected_best_ids, expected_best_indices', [(np.array([0, 2], dtype='int32'), np.array([[1, 1, 1], [3, 3, 3]], dtype='int32'))])
def test_get_best_translations(expected_best_ids: np.ndarray, expected_best_indices: np.ndarray) -> None:
    ...

@pytest.mark.parametrize('sequence, fill_with, expected_sequence', [(np.array([1, 2, 3]), C.EOS_ID, [1, 2, 3]), (np.array([[1], [2], [3]]), C.EOS_ID, [[1], [2], [3]]), (np.array([[1, 0], [2, 1], [3, 2]]), C.EOS_ID, [(1, 1), (2, 2), (3, C.EOS_ID)]), (np.array([[1, 0], [2, 1], [3, 2]]), C.PAD_ID, [(1, 1), (2, 2), (3, C.PAD_ID)])])
def test_unshift_target_factors(sequence: np.ndarray, fill_with: int, expected_sequence: list[int] | list[list[int]] | list[tuple[int, int]]) -> None:
    ...