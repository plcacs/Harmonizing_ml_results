#!/usr/bin/env python3
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import itertools
import json
from math import ceil
from unittest.mock import patch, Mock
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

def mock_translator(batch_size: int = 1, beam_size: int = 5, nbest_size: int = 1, 
                    num_source_factors: int = 1, dtype: pt.dtype = pt.float32
                   ) -> sockeye.inference.Translator:
    """
    Creates a fake translator object but with real values for things that we need.
    This lets us avoid a messy call to the constructor.
    """
    with patch.object(sockeye.inference.Translator, '__init__', lambda self, **kwargs: None):
        translator: sockeye.inference.Translator = sockeye.inference.Translator(
            device=None, batch_size=None, beam_size=None, ensemble_mode=None, scorer=None, beam_search_stop=None, 
            nbest_size=None, models=None, source_vocabs=None, target_vocabs=None, restrict_lexicon=None, 
            strip_unknown_words=None)

        def mock_model() -> Any:
            t_mock: Any = Mock(sockeye.model.SockeyeModel)
            t_mock.num_source_factors = num_source_factors
            t_mock.dtype = dtype
            return t_mock

        translator.batch_size = batch_size
        translator.beam_size = beam_size
        translator.nbest_size = nbest_size
        translator.models = [mock_model()]
        translator.dtype = translator.models[0].dtype
        translator.zeros_array = pt.zeros(beam_size, dtype=pt.int)
        translator.inf_array = pt.full((batch_size * beam_size,), fill_value=np.inf, dtype=pt.float32)
        translator.inf_array = translator.inf_array[:beam_size]
        translator.restrict_lexicon = None
        return translator

@pytest.mark.parametrize('lp_alpha, lp_beta, bp_weight', [(1.0, 0.0, 0.0), (1.0, 2.0, 0.0), (1.0, 2.0, 4.0), (1.0, 0.0, 5.0)])
def test_concat_translations(lp_alpha: float, lp_beta: float, bp_weight: float) -> None:
    expected_target_ids: List[List[int]] = [[0], [1], [2], [0], [8], [9], [0], [3], [4], [5], [-1]]
    scorer: Any = sockeye.beam_search.CandidateScorer(lp_alpha, lp_beta, bp_weight)
    raw_score: float = 1 + 2 + 3
    length: int = len(expected_target_ids)
    reference_length: int = 10 + 11 + 12
    expected_score: List[float] = [scorer(raw_score, length, reference_length)]
    translations: List[sockeye.inference.Translation] = [
        sockeye.inference.Translation([[0], [1], [2], [-1]], [scorer(1.0, 4, 10)], None, 10),
        sockeye.inference.Translation([[0], [8], [9]], [scorer(2.0, 3, 11)], None, 11),
        sockeye.inference.Translation([[0], [3], [4], [5], [-1]], [scorer(3.0, 5, 12)], None, 12)
    ]
    combined: sockeye.inference.Translation = sockeye.inference._concat_translations(
        translations, stop_ids={_EOS}, scorer=scorer)
    assert combined.target_ids == expected_target_ids
    assert np.isclose(combined.scores, expected_score)

@pytest.mark.parametrize('sentence_id, sentence, factors, chunk_size', [
    (1, 'a test', None, 4),
    (1, 'a test', None, 2),
    (1, 'a test', None, 1),
    (0, '', None, 1),
    (1, 'a test', [['h', 'l']], 4),
    (1, 'a test', [['h', 'h'], ['x', 'y']], 1)
])
def test_translator_input(sentence_id: int, sentence: str, 
                          factors: Optional[List[List[str]]], chunk_size: int) -> None:
    tokens: List[str] = sentence.split()
    trans_input: sockeye.inference.TranslatorInput = sockeye.inference.TranslatorInput(
        sentence_id=sentence_id, tokens=tokens, factors=factors)
    assert trans_input.sentence_id == sentence_id
    assert trans_input.tokens == tokens
    assert len(trans_input) == len(tokens)
    assert trans_input.factors == factors
    if factors is not None:
        for factor in trans_input.factors:
            assert len(factor) == len(tokens)
    chunked_inputs: List[sockeye.inference.TranslatorInput] = list(trans_input.chunks(chunk_size))
    assert len(chunked_inputs) == ceil(len(tokens) / chunk_size)
    for chunk_id, chunk_input in enumerate(chunked_inputs):
        assert chunk_input.sentence_id == sentence_id
        assert chunk_input.tokens == trans_input.tokens[chunk_id * chunk_size:(chunk_id + 1) * chunk_size]
        if factors:
            assert len(chunk_input.factors) == len(factors)
            for factor, expected_factor in zip(chunk_input.factors, factors):
                assert len(factor) == len(chunk_input.tokens)
                assert factor == expected_factor[chunk_id * chunk_size:(chunk_id + 1) * chunk_size]

@pytest.mark.parametrize('sentence_id, sentence, factors, chunk_size, source_prefix, source_prefix_factors', [
    (1, 'a test', None, 4, 'prefix test', None),
    (1, 'a test', None, 2, 'prefix test', None),
    (1, 'a test', None, 1, 'prefix test', None),
    (0, '', None, 1, '', None),
    (1, 'a test', [['h', 'l']], 4, 'prefix test', [['h', 'l']]),
    (1, 'a test', [['h', 'h'], ['x', 'y']], 1, 'prefix test', [['h', 'h'], ['x', 'y']])
])
def test_translator_input_with_source_prefix(sentence_id: int, sentence: str, 
                                             factors: Optional[List[List[str]]], chunk_size: int, 
                                             source_prefix: str, 
                                             source_prefix_factors: Optional[List[List[str]]]) -> None:
    tokens: List[str] = sentence.split()
    source_prefix_tokens: List[str] = source_prefix.split()
    trans_input: sockeye.inference.TranslatorInput = sockeye.inference.TranslatorInput(
        sentence_id=sentence_id, tokens=tokens, factors=factors, 
        source_prefix_tokens=source_prefix_tokens, source_prefix_factors=source_prefix_factors)
    assert trans_input.sentence_id == sentence_id
    assert trans_input.tokens == tokens
    # The total length adds the source prefix length
    assert len(trans_input) == len(tokens) + len(source_prefix_tokens)
    assert trans_input.factors == factors
    assert trans_input.source_prefix_tokens == source_prefix_tokens
    assert trans_input.source_prefix_factors == source_prefix_factors
    if factors is not None:
        for factor in trans_input.factors:
            assert len(factor) == len(tokens)
        if trans_input.source_prefix_factors is not None:
            assert len(factors) == len(trans_input.source_prefix_factors)
    chunked_inputs: List[sockeye.inference.TranslatorInput] = list(trans_input.chunks(chunk_size))
    assert len(chunked_inputs) == ceil(len(tokens) / chunk_size)
    for chunk_id, chunk_input in enumerate(chunked_inputs):
        assert chunk_input.sentence_id == sentence_id
        assert chunk_input.tokens == trans_input.tokens[chunk_id * chunk_size:(chunk_id + 1) * chunk_size]
        assert chunk_input.source_prefix_tokens == trans_input.source_prefix_tokens
        assert chunk_input.num_source_prefix_tokens == trans_input.num_source_prefix_tokens
        if source_prefix_factors is not None:
            assert len(chunk_input.source_prefix_factors) == len(source_prefix_factors)
            for chunk_input_source_prefix_factor, source_prefix_factor in zip(chunk_input.source_prefix_factors, trans_input.source_prefix_factors):
                assert len(chunk_input_source_prefix_factor) == len(source_prefix_factor)

@pytest.mark.parametrize('supported_max_seq_len_source, supported_max_seq_len_target, forced_max_input_len, forced_max_output_len, length_ratio_mean, length_ratio_std, expected_max_input_len, expected_max_output_len', [
    (99 + 1, 99 + 1, None, None, 1.0, 0.0, 100, 100),
    (99 + 1, 99 + 1, None, None, 0.9, 0.2, 100, 111),
    (99 + 1, 99 + 1, None, None, 1.1, 0.2, 100, 130),
    (99 + 1, 99 + 1, 50, None, 1.1, 0.2, 51, 67),
    (99 + 1, 99 + 1, 50, None, 1.1, 0.2, 51, 67),
    (99 + 1, 99 + 1, 50, 80, 1.1, 0.2, 51, 81)
])
def test_get_max_input_output_length(supported_max_seq_len_source: int, supported_max_seq_len_target: int, 
                                     forced_max_input_len: Optional[int], forced_max_output_len: Optional[int], 
                                     length_ratio_mean: float, length_ratio_std: float, 
                                     expected_max_input_len: int, expected_max_output_len: int) -> None:
    max_input_len, get_max_output_len = sockeye.inference.get_max_input_output_length(
        supported_max_seq_len_source=supported_max_seq_len_source,
        supported_max_seq_len_target=supported_max_seq_len_target,
        forced_max_input_len=forced_max_input_len,
        forced_max_output_len=forced_max_output_len,
        length_ratio_mean=length_ratio_mean,
        length_ratio_std=length_ratio_std,
        num_stds=1)
    max_output_len: int = get_max_output_len(max_input_len)
    assert max_input_len <= supported_max_seq_len_source
    assert max_input_len == expected_max_input_len
    assert max_output_len == expected_max_output_len

@pytest.mark.parametrize('sentence, num_expected_factors, delimiter, expected_tokens, expected_factors', [
    ('this is a test', 1, '|', ['this', 'is', 'a', 'test'], None),
    ('this|X is| a|X test|', 1, '|', ['this|X', 'is|', 'a|X', 'test|'], None),
    ('space   space', 1, '|', ['space', 'space'], None),
    ('', 1, '|', [], None),
    ('', 2, '|', [], [[]]),
    ('a|l b|l C|u', 2, '|', ['a', 'b', 'C'], [['l', 'l', 'u']]),
    ('a-X-Y b-Y-X', 3, '-', ['a', 'b'], [['X', 'Y'], ['Y', 'X']]),
    ('a-X-Y ', 3, '-', ['a'], [['X'], ['Y']])
])
def test_make_input_from_factored_string(sentence: str, num_expected_factors: int, delimiter: str, 
                                           expected_tokens: List[str], expected_factors: Optional[List[List[str]]]) -> None:
    sentence_id: int = 1
    translator: sockeye.inference.Translator = mock_translator(num_source_factors=num_expected_factors)
    inp: sockeye.inference.TranslatorInput = sockeye.inference.make_input_from_factored_string(
        sentence_id=sentence_id, factored_string=sentence, translator=translator, delimiter=delimiter)
    assert isinstance(inp, sockeye.inference.TranslatorInput)
    assert inp.sentence_id == sentence_id
    assert inp.tokens == expected_tokens
    assert inp.factors == expected_factors
    if num_expected_factors > 1:
        assert len(inp.factors) == num_expected_factors - 1

@pytest.mark.parametrize('sentence, num_expected_factors, delimiter', [
    ('this is a test', 2, '|'),
    ('this|X is a test', 2, '|'),
    ('this|X is|X a|X test', 2, '|'),
    ('this| is|X a|X test|', 2, '|'),
    ('this|X is|X a|X test|', 2, '|'),
    ('w1||w2||f22', 2, '|'),
    ('this', 2, '|'),
    ('this|', 2, '|'),
    ('this||', 3, '|'),
    ('this|| another', 2, '|'),
    ('this|||', 2, '|'),
    ('|this', 2, '|'),
    ('|this|that', 3, '|'),
    ('|this|that|', 4, '|')
])
def test_factor_parsing(sentence: str, num_expected_factors: int, delimiter: str) -> None:
    """
    Test to ensure we fail on parses with invalid factors.
    """
    sentence_id: int = 1
    translator: sockeye.inference.Translator = mock_translator(num_source_factors=num_expected_factors)
    inp: Union[sockeye.inference.TranslatorInput, sockeye.inference.BadTranslatorInput] = \
        sockeye.inference.make_input_from_factored_string(sentence_id=sentence_id, factored_string=sentence, translator=translator, delimiter=delimiter)
    assert isinstance(inp, sockeye.inference.BadTranslatorInput)

@pytest.mark.parametrize('delimiter', [
    '\t', '\t \t', '\t\t', '\n', '\r', '\r\n', ' ', '\n\n', '  ', ' \t', '\x0c', '\x0b', '\xa0',
    '\u1680', '\u2000', None, '', '\u200a', '\u205f', '\u3000'
])
def test_make_input_whitespace_delimiter(delimiter: Optional[str]) -> None:
    """
    Test to ensure we disallow a variety of whitespace strings as factor delimiters.
    """
    sentence_id: int = 1
    translator: sockeye.inference.Translator = mock_translator(num_source_factors=2)
    sentence: str = 'foo'
    with pytest.raises(sockeye.utils.SockeyeError) as e:
        sockeye.inference.make_input_from_factored_string(sentence_id=sentence_id, factored_string=sentence, translator=translator, delimiter=delimiter)  # type: ignore
    assert str(e.value) == 'Factor delimiter can not be whitespace or empty.'

@pytest.mark.parametrize('text, factors', [
    ('this is a test without factors', None),
    ('', None),
    ('test', ['X', 'X']),
    ('a b c', ['x y z']),
    ('a', [])
])
def test_make_input_from_valid_json_string(text: str, factors: Optional[List[str]]) -> None:
    sentence_id: int = 1
    translator: sockeye.inference.Translator = mock_translator()
    expected_tokens: List[str] = list(sockeye.data_io.get_tokens(text))
    inp: sockeye.inference.TranslatorInput = sockeye.inference.make_input_from_json_string(
        sentence_id, json.dumps({C.JSON_TEXT_KEY: text, C.JSON_FACTORS_KEY: factors}), translator)
    assert len(inp) == len(expected_tokens)
    assert inp.tokens == expected_tokens
    if factors is not None:
        assert len(inp.factors) == len(factors)
    else:
        assert inp.factors is None

def test_make_input_from_valid_json_string_restrict_lexicon() -> None:
    sentence_id: int = 1
    text: str = 'this is a test'
    translator: sockeye.inference.Translator = mock_translator()
    lexicon1: Any = Mock(sockeye.lexicon.RestrictLexicon)
    lexicon2: Any = Mock(sockeye.lexicon.RestrictLexicon)
    translator.restrict_lexicon = {'lexicon1': lexicon1, 'lexicon2': lexicon2}
    assert translator.restrict_lexicon['lexicon1'] is not translator.restrict_lexicon['lexicon2']
    restrict_lexicon1: str = 'lexicon1'
    inp1: sockeye.inference.TranslatorInput = sockeye.inference.make_input_from_json_string(
        sentence_id, json.dumps({C.JSON_TEXT_KEY: text, C.JSON_RESTRICT_LEXICON_KEY: restrict_lexicon1}), translator)
    assert inp1.restrict_lexicon is lexicon1
    restrict_lexicon2: str = 'lexicon2'
    inp2: sockeye.inference.TranslatorInput = sockeye.inference.make_input_from_json_string(
        sentence_id, json.dumps({C.JSON_TEXT_KEY: text, C.JSON_RESTRICT_LEXICON_KEY: restrict_lexicon2}), translator)
    assert inp2.restrict_lexicon is lexicon2
    assert inp1.restrict_lexicon is not inp2.restrict_lexicon

@pytest.mark.parametrize('text, text_key, factors, factors_key', [
    ('a', 'blub', None, '')
])
def test_failed_make_input_from_valid_json_string(text: str, text_key: str, 
                                                   factors: Optional[List[str]], factors_key: str) -> None:
    sentence_id: int = 1
    translator: sockeye.inference.Translator = mock_translator()
    inp: Union[sockeye.inference.TranslatorInput, sockeye.inference.BadTranslatorInput] = \
        sockeye.inference.make_input_from_json_string(sentence_id, json.dumps({text_key: text, factors_key: factors}), translator)
    assert isinstance(inp, sockeye.inference.BadTranslatorInput)

@pytest.mark.parametrize('text, factors', [
    ('this is a test without factors', None),
    ('', None),
    ('test', ['X', 'X']),
    ('a b c', ['x y z']),
    ('a', [])
])
def test_make_input_from_valid_dict(text: str, factors: Optional[List[str]]) -> None:
    sentence_id: int = 1
    translator: sockeye.inference.Translator = mock_translator()
    expected_tokens: List[str] = list(sockeye.data_io.get_tokens(text))
    inp: sockeye.inference.TranslatorInput = sockeye.inference.make_input_from_dict(
        sentence_id, {C.JSON_TEXT_KEY: text, C.JSON_FACTORS_KEY: factors}, translator)
    assert len(inp) == len(expected_tokens)
    assert inp.tokens == expected_tokens
    if factors is not None:
        assert len(inp.factors) == len(factors)
    else:
        assert inp.factors is None

@pytest.mark.parametrize('text, text_key, factors, factors_key', [
    ('a', 'blub', None, '')
])
def test_failed_make_input_from_valid_dict(text: str, text_key: str, 
                                           factors: Optional[List[str]], factors_key: str) -> None:
    sentence_id: int = 1
    translator: sockeye.inference.Translator = mock_translator()
    inp: Union[sockeye.inference.TranslatorInput, sockeye.inference.BadTranslatorInput] = \
        sockeye.inference.make_input_from_dict(sentence_id, {text_key: text, factors_key: factors}, translator)
    assert isinstance(inp, sockeye.inference.BadTranslatorInput)

@pytest.mark.parametrize('strings', [
    ['a b c'],
    ['a b c', 'f1 f2 f3', 'f3 f3 f3']
])
def test_make_input_from_multiple_strings(strings: List[str]) -> None:
    inp: sockeye.inference.TranslatorInput = sockeye.inference.make_input_from_multiple_strings(1, strings)
    expected_tokens: List[str] = list(sockeye.data_io.get_tokens(strings[0]))
    expected_factors: List[List[str]] = [list(sockeye.data_io.get_tokens(f)) for f in strings[1:]]
    assert len(inp) == len(expected_tokens)
    assert inp.tokens == expected_tokens
    assert inp.factors == expected_factors

def test_get_best_word_indices_for_kth_hypotheses() -> None:
    all_hyp_indices: np.ndarray = np.array(
        [[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 2, 0, 0, 4, 3],
         [0, 2, 2, 0, 1, 0, 0, 2, 1, 1, 3, 1, 1, 0, 1, 4, 0, 4],
         [0, 1, 0, 1, 2, 1, 4, 3, 2, 3, 0, 4, 3, 1, 2, 1, 1, 0],
         [0, 1, 0, 0, 3, 2, 2, 1, 3, 4, 4, 2, 2, 3, 3, 2, 2, 1],
         [0, 2, 4, 1, 4, 2, 3, 4, 4, 2, 0, 3, 4, 4, 4, 3, 3, 2]],
        dtype='int32')
    ks: List[np.ndarray] = [np.array([0]), np.array([1]), np.array([2]), np.array([3]), np.array([4])]
    expected_indices: List[np.ndarray] = [
        np.array([[2, 1, 0, 0, 0, 0, 1, 3, 3, 2, 0, 0, 0, 1, 1, 2, 3]], dtype='int32'),
        np.array([[1, 2, 1, 2, 2, 3, 4, 4, 4, 3, 1, 1, 1, 2, 2, 3, 4]], dtype='int32'),
        np.array([[2, 1, 0, 0, 0, 1, 0, 0, 0, 0, 4, 2, 3, 3, 3, 4, 0]], dtype='int32'),
        np.array([[2, 1, 0, 0, 0, 1, 0, 0, 0, 0, 2, 3, 2, 0, 0, 0, 1]], dtype='int32'),
        np.array([[2, 1, 0, 1, 1, 2, 3, 2, 2, 4, 3, 4, 4, 4, 4, 1, 2]], dtype='int32')
    ]
    for k, expected_result in zip(ks, expected_indices):
        result: np.ndarray = sockeye.inference.Translator._get_best_word_indices_for_kth_hypotheses(k, all_hyp_indices)
        assert result.shape == expected_result.shape
        assert (result == expected_result).all()
    ks_combined: np.ndarray = np.concatenate(ks, axis=0)
    expected_indices_combined: np.ndarray = np.concatenate(expected_indices, axis=0)
    result: np.ndarray = sockeye.inference.Translator._get_best_word_indices_for_kth_hypotheses(ks_combined, all_hyp_indices)
    assert result.shape == expected_indices_combined.shape
    assert (result == expected_indices_combined).all()

@pytest.mark.parametrize('expected_best_ids, expected_best_indices', [
    (np.array([0, 2], dtype='int32'), np.array([[1, 1, 1], [3, 3, 3]], dtype='int32'))
])
def test_get_best_translations(expected_best_ids: np.ndarray, expected_best_indices: np.ndarray) -> None:
    best_hyp_indices: pt.Tensor = pt.tensor([[0, 1, 0, 1], [0, 1, 1, 0], [2, 3, 2, 3], [2, 3, 3, 2]], dtype=pt.int32)
    best_word_indices: pt.Tensor = pt.tensor([[[3, 3, 0]], [[4, 4, 3]], [[3, 3, 0]], [[4, 5, 3]]], dtype=pt.int32)
    seq_scores: pt.Tensor = pt.tensor([[3.8197377], [5.081118], [3.8068485], [5.0746527]], dtype=pt.float32)
    lengths: pt.Tensor = pt.tensor([[3], [2], [3], [2]], dtype=pt.int32)
    translator: sockeye.inference.Translator = mock_translator(beam_size=2, batch_size=2)
    expected_result: List[sockeye.inference.Translation] = [
        sockeye.inference.Translator._assemble_translation(
            *x) for x in zip(best_word_indices[expected_best_indices, :, np.arange(expected_best_indices.shape[1])],
                              lengths[expected_best_ids], seq_scores[expected_best_ids],
                              itertools.repeat(None))
    ]
    search_result: sockeye.beam_search.SearchResult = sockeye.beam_search.SearchResult(
        best_hyp_indices=best_hyp_indices, best_word_indices=best_word_indices, accumulated_scores=seq_scores,
        lengths=lengths, estimated_reference_lengths=None)
    actual_result: List[sockeye.inference.Translation] = sockeye.inference.Translator._get_best_translations(translator, search_result)
    for expected_translation, actual_translation in zip(expected_result, actual_result):
        assert expected_translation.target_ids == actual_translation.target_ids
        assert expected_translation.scores == actual_translation.scores

@pytest.mark.parametrize('sequence, fill_with, expected_sequence', [
    (np.array([1, 2, 3]), C.EOS_ID, [1, 2, 3]),
    (np.array([[1], [2], [3]]), C.EOS_ID, [[1], [2], [3]]),
    (np.array([[1, 0], [2, 1], [3, 2]]), C.EOS_ID, [(1, 1), (2, 2), (3, C.EOS_ID)]),
    (np.array([[1, 0], [2, 1], [3, 2]]), C.PAD_ID, [(1, 1), (2, 2), (3, C.PAD_ID)])
])
def test_unshift_target_factors(sequence: np.ndarray, fill_with: int, expected_sequence: Union[List[Any], List[List[Any]]]) -> None:
    result: Any = sockeye.inference._unshift_target_factors(sequence, fill_last_with=fill_with)
    assert result == expected_sequence
