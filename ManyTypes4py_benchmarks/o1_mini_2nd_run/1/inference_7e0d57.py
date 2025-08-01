"""
Code for inference/translation
"""
import copy
import itertools
import json
import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union
import numpy as np
import torch as pt
from . import constants as C
from . import lexicon
from . import utils
from . import vocab
from .beam_search import CandidateScorer, get_search_algorithm, GreedySearch, SearchResult
from .data_io import tokens2ids, get_prepended_token_length
from .model import SockeyeModel

logger = logging.getLogger(__name__)

def models_max_input_output_length(
    models: List[SockeyeModel], 
    num_stds: float, 
    forced_max_input_length: Optional[int] = None, 
    forced_max_output_length: Optional[int] = None
) -> Tuple[int, Callable[[int], int]]:
    """
    Returns a function to compute maximum output length given a fixed number of standard deviations as a
    safety margin, and the current input length.
    Mean and std are taken from the model with the largest values to allow proper ensembling of models
    trained on different data sets.

    :param models: List of models.
    :param num_stds: Number of standard deviations to add as a safety margin. If -1, returned maximum output lengths
                     will always be 2 * input_length.
    :param forced_max_input_length: An optional overwrite of the maximum input length. Does not include eos.
    :param forced_max_output_length: An optional overwrite of the maximum output length. Does not include bos.
    :return: The maximum input length and a function to get the output length given the input length.
    """
    max_mean = max((model.length_ratio_mean for model in models))
    max_std = max((model.length_ratio_std for model in models))
    supported_max_seq_len_source = min((model.max_supported_len_source for model in models))
    supported_max_seq_len_target = min((model.max_supported_len_target for model in models))
    return get_max_input_output_length(
        supported_max_seq_len_source, 
        supported_max_seq_len_target, 
        length_ratio_mean=max_mean, 
        length_ratio_std=max_std, 
        num_stds=num_stds, 
        forced_max_input_len=forced_max_input_length, 
        forced_max_output_len=forced_max_output_length
    )

def get_max_input_output_length(
    supported_max_seq_len_source: int, 
    supported_max_seq_len_target: int, 
    length_ratio_mean: float, 
    length_ratio_std: float, 
    num_stds: float, 
    forced_max_input_len: Optional[int] = None, 
    forced_max_output_len: Optional[int] = None
) -> Tuple[int, Callable[[int], int]]:
    """
    Returns a function to compute maximum output length given a fixed number of standard deviations as a
    safety margin, and the current input length. It takes into account optional maximum source and target lengths.

    :param supported_max_seq_len_source: The maximum source length supported by the models (includes eos).
    :param supported_max_seq_len_target: The maximum target length supported by the models (includes bos).
    :param length_ratio_mean: Length ratio mean computed on the training data (including bos/eos).
    :param length_ratio_std: The standard deviation of the length ratio.
    :param num_stds: The number of standard deviations the target length may exceed the mean target length (as long as
           the supported maximum length allows for this).
    :param forced_max_input_len: An optional overwrite of the maximum input length. Does not include eos.
    :param forced_max_output_len: An optional overwrite of the maximum output length. Does not include bos.
    :return: The maximum input length and a function to get the output length given the input length.
    """
    if num_stds < 0:
        factor = C.TARGET_MAX_LENGTH_FACTOR
    else:
        factor = length_ratio_mean + length_ratio_std * num_stds
    if forced_max_input_len is not None:
        max_input_len = min(supported_max_seq_len_source, forced_max_input_len + C.SPACE_FOR_XOS)
    else:
        max_input_len = supported_max_seq_len_source

    def get_max_output_length(input_length: int) -> int:
        """
        Returns the maximum output length (including bos/eos) for inference given an input length that includes <eos>.
        """
        if forced_max_output_len is not None:
            return forced_max_output_len + C.SPACE_FOR_XOS
        return int(np.ceil(factor * input_length))
    
    return (max_input_len, get_max_output_length)

Tokens = List[str]
TokenIds = List[List[int]]
SentenceId = Union[int, str]

@dataclass
class TranslatorInput:
    """
    Object required by Translator.translate().
    If not None, `pass_through_dict` is an arbitrary dictionary instantiated from a JSON object
    via `make_input_from_dict()`, and it contains extra fields found in an input JSON object.
    If `--output-type json` is selected, all such fields that are not fields used or changed by
    Sockeye will be included in the output JSON object. This provides a mechanism for passing
    fields through the call to Sockeye.
    """
    sentence_id: SentenceId
    tokens: Tokens
    factors: Optional[List[List[str]]] = None
    source_prefix_tokens: Optional[Tokens] = None
    source_prefix_factors: Optional[List[List[str]]] = None
    target_prefix_tokens: Optional[Tokens] = None
    target_prefix_factors: Optional[List[List[str]]] = None
    use_target_prefix_all_chunks: bool = True
    keep_target_prefix_key: bool = True
    restrict_lexicon: Optional[Any] = None
    constraints: Optional[List[Tokens]] = None
    avoid_list: Optional[List[Tokens]] = None
    pass_through_dict: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        return (
            f'TranslatorInput({self.sentence_id}, {self.tokens}, factors={self.factors}, '
            f'source_prefix_tokens={self.source_prefix_tokens}, '
            f'source_prefix_factors={self.source_prefix_factors}, '
            f'target_prefix_tokens={self.target_prefix_tokens}, '
            f'target_prefix_factors={self.target_prefix_factors}, '
            f'use_target_prefix_all_chunks={self.use_target_prefix_all_chunks}, '
            f'keep_target_prefix_key={self.keep_target_prefix_key}, '
            f'constraints={self.constraints}, avoid={self.avoid_list})'
        )

    def __len__(self) -> int:
        return len(self.tokens) + self.num_source_prefix_tokens

    @property
    def num_factors(self) -> int:
        """
        Returns the number of factors of this instance.
        """
        return 1 + (0 if not self.factors else len(self.factors))

    def get_source_prefix_tokens(self) -> Tokens:
        """
        Returns the source prefix tokens of this instance.
        """
        return self.source_prefix_tokens if self.source_prefix_tokens is not None else []

    @property
    def num_source_prefix_tokens(self) -> int:
        """
        Returns the number of source prefix tokens of this instance.
        """
        return len(self.get_source_prefix_tokens())

    def get_target_prefix_tokens(self) -> Tokens:
        """
        Returns the target prefix tokens of this instance.
        """
        return self.target_prefix_tokens if self.target_prefix_tokens is not None else []

    @property
    def num_target_prefix_tokens(self) -> int:
        """
        Returns the number of target prefix tokens of this instance.
        """
        return len(self.get_target_prefix_tokens())

    def get_target_prefix_factors(self) -> List[List[str]]:
        """
        Returns the target prefix factors of this instance.
        """
        return self.target_prefix_factors if self.target_prefix_factors is not None else [[]]

    @property
    def num_target_prefix_factors(self) -> int:
        """
        Returns the number of target prefix factors of this instance.
        """
        return len(self.get_target_prefix_factors()[0])

    def chunks(self, chunk_size: int) -> Generator['TranslatorInput', None, None]:
        """
        Takes a TranslatorInput (itself) and yields TranslatorInputs for chunks of size chunk_size.

        :param chunk_size: The maximum size of a chunk.
        :return: A generator of TranslatorInputs, one for each chunk created.
        """
        if len(self.tokens) > chunk_size and self.constraints is not None:
            logger.warning(
                'Input %s has length (%d) that exceeds max input length (%d), triggering internal splitting. '
                'Placing all target-side constraints with the first chunk, which is probably wrong.',
                self.sentence_id, len(self.tokens), chunk_size
            )
        for chunk_id, i in enumerate(range(0, len(self) - self.num_source_prefix_tokens, chunk_size)):
            factors = (
                [factor[i:i + chunk_size] for factor in self.factors] 
                if self.factors is not None else None
            )
            constraints = self.constraints if chunk_id == 0 else None
            target_prefix_tokens = (
                self.target_prefix_tokens if chunk_id == 0 or self.use_target_prefix_all_chunks else None
            )
            target_prefix_factors = (
                self.target_prefix_factors if chunk_id == 0 or self.use_target_prefix_all_chunks else None
            )
            pass_through_dict = (
                copy.deepcopy(self.pass_through_dict) 
                if chunk_id == 0 and self.pass_through_dict is not None else None
            )
            yield TranslatorInput(
                sentence_id=self.sentence_id, 
                tokens=self.tokens[i:i + chunk_size], 
                factors=factors, 
                source_prefix_tokens=self.source_prefix_tokens, 
                source_prefix_factors=self.source_prefix_factors, 
                target_prefix_tokens=target_prefix_tokens, 
                target_prefix_factors=target_prefix_factors, 
                use_target_prefix_all_chunks=self.use_target_prefix_all_chunks, 
                keep_target_prefix_key=self.keep_target_prefix_key, 
                restrict_lexicon=self.restrict_lexicon, 
                constraints=constraints, 
                avoid_list=self.avoid_list, 
                pass_through_dict=pass_through_dict
            )

    def with_eos(self) -> 'TranslatorInput':
        """
        :return: A new translator input with EOS appended to the tokens and factors.
        """
        return TranslatorInput(
            sentence_id=self.sentence_id, 
            tokens=self.tokens + [C.EOS_SYMBOL], 
            factors=[
                factor + [C.EOS_SYMBOL] for factor in self.factors
            ] if self.factors is not None else None, 
            source_prefix_tokens=self.source_prefix_tokens, 
            source_prefix_factors=self.source_prefix_factors, 
            target_prefix_tokens=self.target_prefix_tokens, 
            target_prefix_factors=self.target_prefix_factors, 
            use_target_prefix_all_chunks=self.use_target_prefix_all_chunks, 
            keep_target_prefix_key=self.keep_target_prefix_key, 
            restrict_lexicon=self.restrict_lexicon, 
            constraints=self.constraints, 
            avoid_list=self.avoid_list, 
            pass_through_dict=self.pass_through_dict
        )

class BadTranslatorInput(TranslatorInput):

    def __init__(self, sentence_id: SentenceId, tokens: Tokens) -> None:
        super().__init__(
            sentence_id=sentence_id, 
            tokens=tokens, 
            factors=None
        )

def _bad_input(sentence_id: SentenceId, reason: str = '') -> TranslatorInput:
    logger.warning(
        "Bad input (%s): '%s'. Will return empty output.", 
        sentence_id, 
        reason.strip()
    )
    return BadTranslatorInput(sentence_id=sentence_id, tokens=[])

def make_input_from_plain_string(sentence_id: SentenceId, string: str) -> TranslatorInput:
    """
    Returns a TranslatorInput object from a plain string.

    :param sentence_id: Sentence id.
    :param string: An input string.
    :return: A TranslatorInput.
    """
    return TranslatorInput(
        sentence_id=sentence_id, 
        tokens=list(utils.get_tokens(string)), 
        factors=None
    )

def make_input_from_json_string(
    sentence_id: SentenceId, 
    json_string: str, 
    translator: 'Translator'
) -> TranslatorInput:
    """
    Returns a TranslatorInput object from a JSON object, serialized as a string.

    :param sentence_id: Sentence id.
    :param json_string: A JSON object serialized as a string that must contain a key "text", mapping to the input text,
           and optionally a key "factors" that maps to a list of strings, each of which representing a factor sequence
           for the input text. Constraints and an avoid list can also be added through the "constraints" and "avoid"
           keys.
    :param translator: A translator object.
    :return: A TranslatorInput.
    """
    try:
        jobj = json.loads(json_string)
        return make_input_from_dict(sentence_id, jobj, translator)
    except Exception as e:
        logger.exception(e, exc_info=True)
        return _bad_input(sentence_id, reason=json_string)

def make_input_from_dict(
    sentence_id: SentenceId, 
    input_dict: Dict[str, Any], 
    translator: 'Translator'
) -> TranslatorInput:
    """
    Returns a TranslatorInput object from a JSON object, serialized as a string.

    :param sentence_id: Sentence id.
    :param input_dict: A dict that must contain a key "text", mapping to the input text, and optionally a key "factors"
           that maps to a list of strings, each of which representing a factor sequence for the input text.
           Constraints and an avoid list can also be added through the "constraints" and "avoid" keys.
    :param translator: A translator object.
    :return: A TranslatorInput.
    """
    try:
        tokens: Tokens = input_dict[C.JSON_TEXT_KEY]
        tokens = list(utils.get_tokens(tokens))
        factors: Optional[List[str]] = input_dict.get(C.JSON_FACTORS_KEY)
        source_prefix_tokens: Optional[Tokens] = input_dict.get(C.JSON_SOURCE_PREFIX_KEY)
        source_prefix_tokens = list(utils.get_tokens(source_prefix_tokens)) if source_prefix_tokens is not None else None
        if source_prefix_tokens is not None and (not source_prefix_tokens):
            logger.warning(
                f"Empty string is specified as a source prefix for input '{input_dict[C.JSON_SOURCE_PREFIX_KEY]}'."
            )
        source_prefix_factors: Optional[List[str]] = input_dict.get(C.JSON_SOURCE_PREFIX_FACTORS_KEY)
        if source_prefix_factors is not None and (not source_prefix_tokens):
            logger.error(
                'Source prefix factors cannot be specified when source prefix is not specified'
            )
            return _bad_input(sentence_id, reason=str(input_dict))
        if source_prefix_factors is not None and (not factors):
            logger.error(
                'Source prefix factors cannot be specified when source factors are not specified'
            )
            return _bad_input(sentence_id, reason=str(input_dict))
        if source_prefix_tokens is not None and (factors is not None and (not source_prefix_factors)):
            logger.error(
                'Source prefix factors need to be also specified together with source factors'
            )
            return _bad_input(sentence_id, reason=str(input_dict))
        if isinstance(factors, list):
            factors = [list(utils.get_tokens(factor)) for factor in factors]
            lengths = [len(f) for f in factors]
            if not all((length == len(tokens) for length in lengths)):
                logger.error(
                    'Factors have different length than input text: %d vs. %s', 
                    len(tokens), 
                    str(lengths)
                )
                return _bad_input(sentence_id, reason=str(input_dict))
        if isinstance(source_prefix_factors, list):
            source_prefix_factors = [list(utils.get_tokens(spf)) for spf in source_prefix_factors]
            for source_prefix_factor in source_prefix_factors:
                if not source_prefix_factor:
                    logger.warning(
                        f"Empty list is specified as source prefix factors for input '%s'.", 
                        input_dict[C.JSON_TEXT_KEY]
                    )
            lengths = [len(source_prefix_factor) for source_prefix_factor in source_prefix_factors]
            if not all((len(source_prefix_tokens) == length for length in lengths)):
                logger.error(
                    'Source prefix has %d tokens but there are %s prefix factors', 
                    len(source_prefix_tokens), 
                    str(lengths)
                )
                return _bad_input(sentence_id, reason=str(input_dict))
            if len(source_prefix_factors) != len(factors):
                logger.error(
                    'There is mismatch in source factors %d and prefix factors %d', 
                    len(factors), 
                    len(source_prefix_factors)
                )
                return _bad_input(sentence_id, reason=str(input_dict))
        target_prefix_tokens: Optional[Tokens] = input_dict.get(C.JSON_TARGET_PREFIX_KEY)
        target_prefix_tokens = list(utils.get_tokens(target_prefix_tokens)) if target_prefix_tokens is not None else None
        if target_prefix_tokens is not None and (not target_prefix_tokens):
            logger.warning(
                f"Empty string is specified as a target prefix for input '{input_dict[C.JSON_TEXT_KEY]}'."
            )
        target_prefix_factors: Optional[List[str]] = input_dict.get(C.JSON_TARGET_PREFIX_FACTORS_KEY)
        if isinstance(target_prefix_factors, list):
            target_prefix_factors = [list(utils.get_tokens(tpf)) for tpf in target_prefix_factors]
            if len(target_prefix_factors) != translator.num_target_factors - 1:
                logger.error(
                    'Must provide target prefix for each target factor. Given: %s required: %s', 
                    len(target_prefix_factors), 
                    translator.num_target_factors - 1
                )
                return _bad_input(sentence_id, reason=str(input_dict))
        use_target_prefix_all_chunks: bool = input_dict.get(C.JSON_USE_TARGET_PREFIX_ALL_CHUNKS_KEY, True)
        keep_target_prefix_key: bool = input_dict.get(C.JSON_KEEP_TARGET_PREFIX_KEY, True)
        restrict_lexicon: Optional[Any] = None
        restrict_lexicon_name: Optional[str] = input_dict.get(C.JSON_RESTRICT_LEXICON_KEY, None)
        if isinstance(translator.restrict_lexicon, dict) and restrict_lexicon_name is not None:
            restrict_lexicon = translator.restrict_lexicon.get(restrict_lexicon_name, None)
            if restrict_lexicon is None:
                logger.error(
                    "Unknown restrict_lexicon '%s'. Choices: %s", 
                    restrict_lexicon_name, 
                    ' '.join(sorted(translator.restrict_lexicon))
                )
                return _bad_input(sentence_id, reason=str(input_dict))
        avoid_list: Optional[List[str]] = input_dict.get(C.JSON_AVOID_KEY)
        constraints: Optional[List[str]] = input_dict.get(C.JSON_CONSTRAINTS_KEY)
        if constraints is not None and avoid_list is not None:
            avoid_set: Set[str] = set(avoid_list)
            overlap: Set[str] = set(constraints).intersection(avoid_set)
            if len(overlap) > 0:
                logger.warning(
                    'Overlap between constraints and avoid set, dropping the overlapping avoids'
                )
                avoid_list = list(avoid_set.difference(overlap))
        if isinstance(avoid_list, list):
            avoid_list = [list(utils.get_tokens(phrase)) for phrase in avoid_list]
        if isinstance(constraints, list):
            constraints = [list(utils.get_tokens(constraint)) for constraint in constraints]
        return TranslatorInput(
            sentence_id=sentence_id, 
            tokens=tokens, 
            factors=factors, 
            source_prefix_tokens=source_prefix_tokens, 
            source_prefix_factors=source_prefix_factors, 
            target_prefix_tokens=target_prefix_tokens, 
            target_prefix_factors=target_prefix_factors, 
            use_target_prefix_all_chunks=use_target_prefix_all_chunks, 
            keep_target_prefix_key=keep_target_prefix_key, 
            restrict_lexicon=restrict_lexicon, 
            constraints=constraints, 
            avoid_list=avoid_list, 
            pass_through_dict=input_dict
        )
    except Exception as e:
        logger.exception(e, exc_info=True)
        return _bad_input(sentence_id, reason=str(input_dict))

def make_input_from_factored_string(
    sentence_id: SentenceId, 
    factored_string: str, 
    translator: 'Translator', 
    delimiter: str = C.DEFAULT_FACTOR_DELIMITER
) -> TranslatorInput:
    """
    Returns a TranslatorInput object from a string with factor annotations on a token level, separated by delimiter.
    If translator does not require any source factors, the string is parsed as a plain token string.

    :param sentence_id: Sentence id.
    :param factored_string: An input string with additional factors per token, separated by delimiter.
    :param translator: A translator object.
    :param delimiter: A factor delimiter. Default: '|'.
    :return: A TranslatorInput.
    """
    utils.check_condition(
        bool(delimiter) and (not delimiter.isspace()), 
        'Factor delimiter can not be whitespace or empty.'
    )
    model_num_source_factors: int = translator.num_source_factors
    if model_num_source_factors == 1:
        return make_input_from_plain_string(
            sentence_id=sentence_id, 
            string=factored_string
        )
    tokens: Tokens = []
    factors: List[List[str]] = [[] for _ in range(model_num_source_factors - 1)]
    for token_id, token in enumerate(utils.get_tokens(factored_string)):
        pieces: List[str] = token.split(delimiter)
        if not all(pieces) or len(pieces) != model_num_source_factors:
            logger.error(
                "Failed to parse %d factors at position %d ('%s') in '%s'", 
                model_num_source_factors, 
                token_id, 
                token, 
                factored_string.strip()
            )
            return _bad_input(sentence_id, reason=factored_string)
        tokens.append(pieces[0])
        for i, factor in enumerate(factors):
            factors[i].append(pieces[i + 1])
    return TranslatorInput(
        sentence_id=sentence_id, 
        tokens=tokens, 
        factors=factors
    )

def make_input_from_multiple_strings(
    sentence_id: SentenceId, 
    strings: List[str]
) -> TranslatorInput:
    """
    Returns a TranslatorInput object from multiple strings, where the first element corresponds to the surface tokens
    and the remaining elements to additional factors. All strings must parse into token sequences of the same length.

    :param sentence_id: Sentence id.
    :param strings: A list of strings representing a factored input sequence.
    :return: A TranslatorInput.
    """
    if not bool(strings):
        return TranslatorInput(
            sentence_id=sentence_id, 
            tokens=[], 
            factors=None
        )
    tokens: Tokens = list(utils.get_tokens(strings[0]))
    factors: List[List[str]] = [list(utils.get_tokens(factor)) for factor in strings[1:]]
    if not all((len(factor) == len(tokens) for factor in factors)):
        logger.error("Length of string sequences do not match: '%s'", strings)
        return _bad_input(sentence_id, reason=str(strings))
    return TranslatorInput(
        sentence_id=sentence_id, 
        tokens=tokens, 
        factors=factors
    )

@dataclass
class TranslatorOutput:
    """
    Output structure from Translator.

    sentence_id: Sentence id.
    translation: Translation string without sentence boundary tokens.
    tokens: List of translated tokens.
    score: Negative log probability of generated translation.
    pass_through_dict: Dictionary of key/value pairs to pass through when working with JSON.
    nbest_translations: List of nbest translations as strings.
    nbest_tokens: List of nbest translations as lists of tokens.
    nbest_scores: List of nbest scores, one for each nbest translation.
    factor_translations: List of factor outputs.
    factor_tokens: List of list of secondary factor tokens.
    factor_scores: List of secondary factor scores.
    """
    sentence_id: SentenceId
    translation: str
    tokens: Tokens
    score: float
    pass_through_dict: Optional[Dict[str, Any]] = None
    nbest_translations: Optional[List[str]] = None
    nbest_tokens: Optional[List[Tokens]] = None
    nbest_scores: Optional[List[float]] = None
    factor_translations: Optional[List[str]] = None
    factor_tokens: Optional[List[Tokens]] = None
    factor_scores: Optional[List[float]] = None
    nbest_factor_translations: Optional[List[List[str]]] = None
    nbest_factor_tokens: Optional[List[List[Tokens]]] = None

    def json(self) -> Dict[str, Any]:
        """
        Returns a dictionary suitable for json.dumps() representing all
        the information in the class. It is initialized with any keys
        present in the corresponding `TranslatorInput` object's pass_through_dict.
        Keys from here that are not overwritten by Sockeye will thus be passed
        through to the output.

        :return: A dictionary.
        """
        _d: Dict[str, Any] = copy.deepcopy(self.pass_through_dict) if self.pass_through_dict is not None else {}
        _d['sentence_id'] = self.sentence_id
        _d['translation'] = self.translation
        _d['score'] = self.score
        if self.nbest_translations is not None and len(self.nbest_translations) > 1:
            _d['translations'] = self.nbest_translations
            _d['scores'] = self.nbest_scores
        if self.factor_translations is not None:
            for i, factor in enumerate(self.factor_translations, 1):
                _d[f'factor{i}'] = factor
        if self.factor_scores is not None:
            for i, score in enumerate(self.factor_scores, 1):
                _d[f'factor{i}_score'] = score
        if self.nbest_factor_translations is not None and len(self.nbest_factor_translations) > 1:
            _d['translations_factors'] = []
            for factor_translations in self.nbest_factor_translations:
                _d['translations_factors'].append({
                    f'factor{i}': factor_translation 
                    for i, factor_translation in enumerate(factor_translations, 1)
                })
        return _d

@dataclass
class NBestTranslations:
    target_ids_list: List[List[int]] = None
    scores: List[float] = None

@dataclass
class Translation:
    target_ids: List[int]
    scores: List[float]
    nbest_translations: Optional[NBestTranslations] = None
    estimated_reference_length: Optional[float] = None

def empty_translation(add_nbest: bool = False) -> Translation:
    """
    Return an empty translation.

    :param add_nbest: Include (empty) nbest_translations in the translation object.
    """
    return Translation(
        target_ids=[], 
        scores=[-np.inf], 
        nbest_translations=NBestTranslations([], []) if add_nbest else None,
        estimated_reference_length=None
    )

@dataclass
class IndexedTranslatorInput:
    """
    Translation of a chunk of a sentence.

    input_idx: Internal index of translation requests to keep track of the correct order of translations.
    chunk_idx: The index of the chunk. Used when TranslatorInputs get split across multiple chunks.
    translator_input: The translator input.
    """
    input_idx: int
    chunk_idx: int
    translator_input: TranslatorInput

@dataclass(order=True)
@dataclass
class IndexedTranslation:
    """
    Translation of a chunk of a sentence.

    input_idx: Internal index of translation requests to keep track of the correct order of translations.
    chunk_idx: The index of the chunk. Used when TranslatorInputs get split across multiple chunks.
    translation: The translation of the input chunk.
    """
    input_idx: int
    chunk_idx: int
    translation: Translation

def _concat_nbest_translations(
    translations: List[Translation], 
    stop_ids: Set[int], 
    scorer: CandidateScorer
) -> Translation:
    """
    Combines nbest translations through concatenation.

    :param translations: A list of translations (sequence starting with BOS symbol), score and length.
    :param stop_ids: The EOS symbols.
    :param scorer: Candidate scorer for recomputing score of concatenated translations.
    :return: A concatenation of the translations with a score.
    """
    expanded_translations = (_expand_nbest_translation(translation) for translation in translations)
    concatenated_translations: List[Translation] = []
    for translations_to_concat in zip(*expanded_translations):
        concatenated_translations.append(
            _concat_translations(
                translations=list(translations_to_concat), 
                stop_ids=stop_ids, 
                scorer=scorer
            )
        )
    return _reduce_nbest_translations(concatenated_translations)

def _reduce_nbest_translations(
    nbest_translations_list: List[Translation]
) -> Translation:
    """
    Combines Translation objects that are nbest translations of the same sentence.

    :param nbest_translations_list: A list of Translation objects, all of them translations of
        the same source sentence.
    :return: A single Translation object where nbest lists are collapsed.
    """
    best_translation = nbest_translations_list[0]
    sequences = [translation.target_ids for translation in nbest_translations_list]
    scores = [translation.scores for translation in nbest_translations_list]
    nbest_translations = NBestTranslations(sequences, scores)
    return Translation(
        target_ids=best_translation.target_ids, 
        scores=best_translation.scores, 
        nbest_translations=nbest_translations, 
        estimated_reference_length=best_translation.estimated_reference_length
    )

def _expand_nbest_translation(translation: Translation) -> List[Translation]:
    """
    Expand nbest translations in a single Translation object to one Translation
        object per nbest translation.

    :param translation: A Translation object.
    :return: A list of Translation objects.
    """
    nbest_list: List[Translation] = []
    if translation.nbest_translations is not None:
        for target_ids, score in zip(translation.nbest_translations.target_ids_list, translation.nbest_translations.scores):
            nbest_list.append(
                Translation(
                    target_ids=target_ids, 
                    scores=[score], 
                    estimated_reference_length=translation.estimated_reference_length
                )
            )
    return nbest_list

def _remove_target_prefix_tokens(
    target_ids: List[int], 
    num_target_prefix_tokens: int
) -> List[int]:
    """
    Remove target prefix tokens from target token Ids

    :param target_ids: target token Ids of translation of an input
    :param num_target_prefix_tokens: number of target prefix tokens included in the translation
    :return: new target_ids
    """
    starting_idx = min(len(target_ids), num_target_prefix_tokens)
    return target_ids[starting_idx:]

def _concat_translations(
    translations: List[Translation], 
    stop_ids: Set[int], 
    scorer: CandidateScorer
) -> Translation:
    """
    Combines translations through concatenation.

    :param translations: A list of translations (sequence starting with BOS symbol), score and length.
    :param stop_ids: The EOS symbols.
    :param scorer: Candidate scorer for recomputing score of concatenated translations.
    :return: A concatenation of the translations with a score.
    """
    if len(translations) == 1:
        return translations[0]
    target_ids: List[int] = []
    estimated_reference_length: Optional[float] = None
    scores: np.ndarray = np.zeros_like(translations[0].scores)
    for idx, translation in enumerate(translations):
        if idx == len(translations) - 1:
            target_ids.extend(translation.target_ids)
        elif translation.target_ids[-1] in stop_ids:
            target_ids.extend(translation.target_ids[:-1])
        else:
            target_ids.extend(translation.target_ids)
        if translation.estimated_reference_length is not None:
            if estimated_reference_length is None:
                estimated_reference_length = translation.estimated_reference_length
            else:
                estimated_reference_length += translation.estimated_reference_length
        score: float = translation.scores[0]
        raw_score: float = scorer.unnormalize(score, len(translation.target_ids), translation.estimated_reference_length)
        scores = np.add(scores, [raw_score, *translation.scores[1:]])
    scores[0] = scorer(scores[0], len(target_ids), estimated_reference_length)
    return Translation(
        target_ids=target_ids, 
        scores=scores.tolist(), 
        estimated_reference_length=estimated_reference_length
    )

class Translator:
    """
    Translator uses one or several models to translate input.
    The translator holds a reference to vocabularies to convert between word ids and text tokens for input and
    translation strings.

    :param device: Pytorch device to bind modules to.
    :param ensemble_mode: Ensemble mode: linear or log_linear combination.
    :param scorer: Hypothesis/Candidate scoring instance
    :param beam_search_stop: The stopping criterion.
    :param models: List of models.
    :param source_vocabs: Source vocabularies.
    :param target_vocabs: Target vocabularies.
    :param nbest_size: Size of nbest list of translations.
    :param restrict_lexicon: Lexicon to use for target vocabulary selection. Can be a dict of named lexicons. When
           it is a single lexicon it will be applied to all inputs. If is a Dict the lexicon with the given name will
           be used or no lexicon be used if the name is None.
    :param strip_unknown_words: If True, removes any <unk> symbols from outputs.
    :param sample: If True, sample from softmax multinomial instead of using topk.
    :param output_scores: Whether the scores will be needed as outputs. If True, scores will be normalized, negative
           log probabilities. If False, scores will be negative, raw logit activations if decoding with beam size 1
           and a single model.
    :param constant_length_ratio: If > 0, will override models' prediction of the length ratio (if any).
    :param max_output_length_num_stds: Number of standard deviations to add as a safety margin when computing the
           maximum output length. If -1, returned maximum output lengths will always be 2 * input_length.
    :param max_input_length: Maximum input length this Translator should allow. If None, value will be taken from the
           model(s). Inputs larger than this value will be chunked and translated in sequence.
           If model(s) do not support given input length it will fall back to what the model(s) support.
    :param max_output_length: Maximum output length this Translator is allowed to decode. If None, value will be taken
           from the model(s). Decodings that do not finish within this limit, will be force-stopped.
           If model(s) do not support given input length it will fall back to what the model(s) support.
    :param skip_nvs: Manually turn off Neural Vocabulary Selection (NVS) to do a softmax over the full target vocabulary.
    :param nvs_thresh: The probability threshold for a word to be added to the set of target words. Default: 0.5.
    """

    def __init__(
        self, 
        device: pt.device, 
        ensemble_mode: Optional[str], 
        scorer: CandidateScorer, 
        batch_size: int, 
        beam_search_stop: str, 
        models: List[SockeyeModel], 
        source_vocabs: List[vocab.Vocab], 
        target_vocabs: List[vocab.Vocab], 
        beam_size: int = 5, 
        nbest_size: int = 1, 
        restrict_lexicon: Optional[Union[Dict[str, Any], Any]] = None, 
        strip_unknown_words: bool = False, 
        sample: Optional[bool] = None, 
        output_scores: bool = False, 
        constant_length_ratio: float = 0.0, 
        knn_lambda: float = C.DEFAULT_KNN_LAMBDA, 
        max_output_length_num_stds: float = C.DEFAULT_NUM_STD_MAX_OUTPUT_LENGTH, 
        max_input_length: Optional[int] = None, 
        max_output_length: Optional[int] = None, 
        prevent_unk: bool = False, 
        greedy: bool = False, 
        skip_nvs: bool = False, 
        nvs_thresh: float = 0.5
    ) -> None:
        self.device: pt.device = device
        self.dtype: torch.dtype = models[0].dtype
        self._scorer: CandidateScorer = scorer
        self.batch_size: int = batch_size
        self.beam_size: int = beam_size
        self.beam_search_stop: str = beam_search_stop
        self.source_vocabs: List[vocab.Vocab] = source_vocabs
        self.vocab_targets: List[vocab.Vocab] = target_vocabs
        self.vocab_targets_inv: List[Dict[int, str]] = [vocab.reverse_vocab(v) for v in self.vocab_targets]
        self.restrict_lexicon: Optional[Union[Dict[str, Any], Any]] = restrict_lexicon
        assert C.PAD_ID == 0, 'pad id should be 0'
        self.stop_ids: Set[int] = {C.EOS_ID, C.PAD_ID}
        self.strip_ids: Set[int] = self.stop_ids.copy()
        self.unk_id: int = C.UNK_ID
        if strip_unknown_words:
            self.strip_ids.add(self.unk_id)
        self.models: List[SockeyeModel] = models
        for model in self.models:
            model.eval()
        self._max_input_length: int
        self._get_max_output_length: Callable[[int], int]
        self._max_input_length, self._get_max_output_length = models_max_input_output_length(
            models, 
            max_output_length_num_stds, 
            forced_max_input_length=max_input_length, 
            forced_max_output_length=max_output_length
        )
        self.nbest_size: int = nbest_size
        utils.check_condition(
            self.beam_size >= self.nbest_size, 
            'nbest_size must be smaller or equal to beam_size.'
        )
        if self.nbest_size > 1:
            utils.check_condition(
                self.beam_search_stop == C.BEAM_SEARCH_STOP_ALL, 
                "nbest_size > 1 requires beam_search_stop to be set to 'all'"
            )
        self._search: Union[GreedySearch, Any] = get_search_algorithm(
            models=self.models, 
            beam_size=self.beam_size, 
            device=self.device, 
            output_scores=output_scores, 
            sample=sample, 
            ensemble_mode=ensemble_mode, 
            beam_search_stop=beam_search_stop, 
            scorer=self._scorer, 
            constant_length_ratio=constant_length_ratio, 
            knn_lambda=knn_lambda, 
            prevent_unk=prevent_unk, 
            greedy=greedy, 
            skip_nvs=skip_nvs, 
            nvs_thresh=nvs_thresh
        )
        self._concat_translations: Callable[[List[Translation]], Translation] = partial(
            _concat_nbest_translations if self.nbest_size > 1 else _concat_translations, 
            stop_ids=self.stop_ids, 
            scorer=self._scorer
        )
        logger.info(
            'Translator (%d model(s) beam_size=%d algorithm=%s, beam_search_stop=%s max_input_length=%s '
            'nbest_size=%s ensemble_mode=%s max_batch_size=%d dtype=%s skip_nvs=%s nvs_thresh=%s)', 
            len(self.models), 
            self.beam_size, 
            'GreedySearch' if isinstance(self._search, GreedySearch) else 'BeamSearch', 
            self.beam_search_stop, 
            self.max_input_length, 
            self.nbest_size, 
            'None' if len(self.models) == 1 else ensemble_mode, 
            self.max_batch_size, 
            self.dtype, 
            skip_nvs, 
            nvs_thresh
        )

    @property
    def max_input_length(self) -> int:
        """
        Returns maximum input length for TranslatorInput objects passed to translate()
        """
        return self._max_input_length - C.SPACE_FOR_XOS

    @property
    def max_batch_size(self) -> int:
        """
        Returns the maximum batch size allowed for this Translator.
        """
        return self.batch_size

    @property
    def num_source_factors(self) -> int:
        return self.models[0].num_source_factors

    @property
    def num_target_factors(self) -> int:
        return self.models[0].num_target_factors

    @property
    def eop_id(self) -> int:
        return self.models[0].eop_id

    def translate(
        self, 
        trans_inputs: List[TranslatorInput], 
        fill_up_batches: bool = True
    ) -> List[TranslatorOutput]:
        """
        Batch-translates a list of TranslatorInputs, returns a list of TranslatorOutputs.
        Empty or bad inputs are skipped.
        Splits inputs longer than Translator.max_input_length into segments of size max_input_length,
        and then groups segments into batches of at most Translator.max_batch_size.
        Too-long segments that were split are reassembled into a single output after translation.
        If fill_up_batches is set to True, underfilled batches are padded to Translator.max_batch_size, otherwise
        dynamic batch sizing is used, which comes at increased memory usage.

        :param trans_inputs: List of TranslatorInputs as returned by make_input().
        :param fill_up_batches: If True, underfilled batches are padded to Translator.max_batch_size.
        :return: List of translation results.
        """
        num_inputs: int = len(trans_inputs)
        translated_chunks: List[IndexedTranslation] = []
        input_chunks: List[IndexedTranslatorInput] = []
        for trans_input_idx, trans_input in enumerate(trans_inputs):
            if isinstance(trans_input, BadTranslatorInput):
                translated_chunks.append(
                    IndexedTranslation(
                        input_idx=trans_input_idx, 
                        chunk_idx=0, 
                        translation=empty_translation(add_nbest=self.nbest_size > 1)
                    )
                )
            elif len(trans_input.tokens) == 0:
                translated_chunks.append(
                    IndexedTranslation(
                        input_idx=trans_input_idx, 
                        chunk_idx=0, 
                        translation=empty_translation(add_nbest=self.nbest_size > 1)
                    )
                )
            else:
                max_input_length_for_chunking: int = self.max_input_length - trans_input.num_source_prefix_tokens
                if max_input_length_for_chunking <= 0:
                    logger.warning(
                        'Input %s has a source prefix with length (%d) that already equals or exceeds max input length (%d). '
                        'Return an empty translation instead.', 
                        trans_input.sentence_id, 
                        trans_input.num_source_prefix_tokens, 
                        self.max_input_length
                    )
                    translated_chunks.append(
                        IndexedTranslation(
                            input_idx=trans_input_idx, 
                            chunk_idx=0, 
                            translation=empty_translation(add_nbest=self.nbest_size > 1)
                        )
                    )
                elif len(trans_input.tokens) > max_input_length_for_chunking:
                    logger.debug(
                        'Input %s has length (%d) that exceeds max input length (%d). Splitting into chunks of size %d.', 
                        trans_input.sentence_id, 
                        len(trans_input.tokens), 
                        max_input_length_for_chunking, 
                        max_input_length_for_chunking
                    )
                    chunks = [
                        trans_input_chunk.with_eos() 
                        for trans_input_chunk in trans_input.chunks(max_input_length_for_chunking)
                    ]
                    input_chunks.extend([
                        IndexedTranslatorInput(
                            input_idx=trans_input_idx, 
                            chunk_idx=chunk_idx, 
                            translator_input=chunk_input
                        ) 
                        for chunk_idx, chunk_input in enumerate(chunks)
                    ])
                else:
                    input_chunks.append(
                        IndexedTranslatorInput(
                            input_idx=trans_input_idx, 
                            chunk_idx=0, 
                            translator_input=trans_input.with_eos()
                        )
                    )
            if trans_input.constraints is not None:
                logger.info(
                    'Input %s has %d %s: %s', 
                    trans_input.sentence_id, 
                    len(trans_input.constraints), 
                    'constraint' if len(trans_input.constraints) == 1 else 'constraints', 
                    ', '.join((' '.join(x) for x in trans_input.constraints))
                )
        num_bad_empty: int = len(translated_chunks)
        input_chunks.sort(
            key=lambda chunk: len(chunk.translator_input.tokens), 
            reverse=True
        )
        batch_size: int = self.max_batch_size if fill_up_batches else min(len(input_chunks), self.max_batch_size)
        num_batches: int = 0
        for batch_id, batch in enumerate(utils.grouper(input_chunks, batch_size)):
            logger.debug('Translating batch %d', batch_id)
            rest: int = batch_size - len(batch)
            if fill_up_batches and rest > 0:
                logger.debug(
                    'Padding batch of size %d to full batch size (%d)', 
                    len(batch), 
                    batch_size
                )
                batch = batch + [batch[0]] * rest
            translator_inputs: List[TranslatorInput] = [
                indexed_translator_input.translator_input for indexed_translator_input in batch
            ]
            batch_translations: List[Translation] = self._translate_batch(translator_inputs)
            if fill_up_batches and rest > 0:
                batch_translations = batch_translations[:-rest]
            for chunk, translation in zip(batch, batch_translations):
                translated_chunks.append(
                    IndexedTranslation(
                        input_idx=chunk.input_idx, 
                        chunk_idx=chunk.chunk_idx, 
                        translation=translation
                    )
                )
            num_batches += 1
        translated_chunks.sort()
        num_chunks: int = len(translated_chunks)
        results: List[TranslatorOutput] = []
        chunks_by_input_idx: Iterator[Tuple[int, Iterator[IndexedTranslation]]] = itertools.groupby(
            translated_chunks, 
            key=lambda translation: translation.input_idx
        )
        for trans_input, (input_idx, translations_for_input_idx) in zip(
            trans_inputs, 
            chunks_by_input_idx
        ):
            translations_for_input_idx: List[IndexedTranslation] = list(translations_for_input_idx)
            num_target_prefix_tokens: int = trans_input.num_target_prefix_tokens
            if len(translations_for_input_idx) == 1:
                translation: Translation = translations_for_input_idx[0].translation
                if num_target_prefix_tokens > 0 and (not trans_input.keep_target_prefix_key):
                    translation.target_ids = _remove_target_prefix_tokens(
                        translation.target_ids, 
                        num_target_prefix_tokens
                    )
            else:
                translations_to_concat: List[Translation] = [
                    translated_chunk.translation for translated_chunk in translations_for_input_idx
                ]
                if num_target_prefix_tokens > 0 and (not trans_input.keep_target_prefix_key):
                    for i in range(len(translations_to_concat)):
                        if i == 0 or trans_input.use_target_prefix_all_chunks:
                            translations_to_concat[i].target_ids = _remove_target_prefix_tokens(
                                translations_to_concat[i].target_ids, 
                                num_target_prefix_tokens
                            )
                translation: Translation = self._concat_translations(translations_to_concat)
            results.append(self._make_result(trans_input, translation))
        num_outputs: int = len(results)
        logger.debug(
            'Translated %d inputs (%d chunks) in %d batches to %d outputs. %d empty/bad inputs.', 
            num_inputs, 
            num_chunks, 
            num_batches, 
            num_outputs, 
            num_bad_empty
        )
        self._search.log_search_stats()
        return results

    def _translate_batch(self, translator_inputs: List[TranslatorInput]) -> List[Translation]:
        """
        Translate a batch of inputs.

        :param translator_inputs: List of TranslatorInputs.
        :return: List of Translation.
        """
        with pt.inference_mode():
            return self._translate_np(*self._get_inference_input(translator_inputs))

    def _get_inference_input(
        self, 
        trans_inputs: List[TranslatorInput]
    ) -> Tuple[pt.Tensor, pt.Tensor, Optional[Any], pt.Tensor, Optional[pt.Tensor], Optional[pt.Tensor]]:
        """
        Assembles the numerical data for the batch. This comprises a tensor for the source sentences,
        the bucket key (padded source length), a tensor of maximum output lengths for each sentence in the batch.

        :param trans_inputs: List of TranslatorInputs.
        :return tensor of source ids (shape=(batch_size, bucket_key, num_factors)),
                tensor of valid source lengths, lexicon for vocabulary restriction, tensor of maximum output lengths,
                optional target prefix, and optional target prefix factors.
        """
        batch_size: int = len(trans_inputs)
        max_target_prefix_length: int = max((inp.num_target_prefix_tokens for inp in trans_inputs))
        max_target_prefix_factors_length: int = max((inp.num_target_prefix_factors for inp in trans_inputs))
        max_length: int = max((len(inp) for inp in trans_inputs))
        source_np: np.ndarray = np.zeros(
            (batch_size, max_length, self.num_source_factors), 
            dtype='int32'
        )
        length_np: np.ndarray = np.zeros((batch_size, 2), dtype='int32')
        target_prefix_np: Optional[np.ndarray] = (
            np.zeros((batch_size, max_target_prefix_length), dtype='int32') 
            if max_target_prefix_length > 0 else None
        )
        target_prefix_factors_np: Optional[np.ndarray] = (
            np.zeros(
                (batch_size, max_target_prefix_factors_length, self.num_target_factors - 1), 
                dtype='int32'
            ) 
            if self.num_target_factors > 1 and max_target_prefix_factors_length > 0 else None
        )
        restrict_lexicon: Optional[Any] = None
        max_output_lengths: List[int] = []
        for j, trans_input in enumerate(trans_inputs):
            num_tokens: int = len(trans_input)
            primary_source_ids: List[int] = tokens2ids(
                itertools.chain(trans_input.get_source_prefix_tokens(), trans_input.tokens), 
                self.source_vocabs[0]
            )
            source_np[j, :num_tokens, 0] = primary_source_ids
            length_np[j, 0] = num_tokens
            length_np[j, 1] = get_prepended_token_length(primary_source_ids, self.eop_id)
            max_output_lengths.append(self._get_max_output_length(length_np[j, 0] - length_np[j, 1]))
            if target_prefix_np is not None and trans_input.num_target_prefix_tokens > 0:
                target_prefix_np[j, :trans_input.num_target_prefix_tokens] = tokens2ids(
                    trans_input.get_target_prefix_tokens(), 
                    self.vocab_targets[0]
                )
            if target_prefix_factors_np is not None and self.num_target_factors > 1 and (trans_input.num_target_prefix_factors > 0):
                for i in range(1, self.num_target_factors):
                    target_prefix_factors_np[j, :trans_input.num_target_prefix_factors, i - 1] = tokens2ids(
                        trans_input.get_target_prefix_factors()[i - 1], 
                        self.vocab_targets[i]
                    )
            factors: Optional[List[List[str]]] = trans_input.factors
            num_factors: int = 1 + (len(factors) if factors else 0)
            if num_factors != self.num_source_factors:
                logger.warning(
                    'Input %d factors, but model(s) expect %d', 
                    num_factors, 
                    self.num_source_factors
                )
            if not trans_input.source_prefix_factors:
                for i, factor in enumerate(factors[:self.num_source_factors - 1], start=1):
                    source_np[j, :num_tokens, i] = tokens2ids(
                        factor, 
                        self.source_vocabs[i]
                    )[:num_tokens]
            else:
                for i, (factor, source_prefix_factor) in enumerate(
                    zip(trans_input.factors[:self.num_source_factors - 1], trans_input.source_prefix_factors[:self.num_source_factors - 1]), 
                    start=1
                ):
                    source_np[j, :num_tokens, i] = tokens2ids(
                        itertools.chain(source_prefix_factor, factor), 
                        self.source_vocabs[i]
                    )[:num_tokens]
            if trans_input.restrict_lexicon is not None:
                if restrict_lexicon is not None and restrict_lexicon is not trans_input.restrict_lexicon:
                    logger.warning(
                        'Sentence %s: different restrict_lexicon specified, will overrule previous. All inputs in batch must use same lexicon.', 
                        trans_input.sentence_id
                    )
                restrict_lexicon = trans_input.restrict_lexicon
            elif self.restrict_lexicon is not None:
                if isinstance(self.restrict_lexicon, dict):
                    restrict_lexicon = None
                else:
                    restrict_lexicon = self.restrict_lexicon
        if restrict_lexicon is None and isinstance(self.restrict_lexicon, dict):
            logger.info(
                'No restrict_lexicon specified for input when using multiple lexicons, will default to not using a restrict lexicon.'
            )
        source: pt.Tensor = pt.tensor(source_np, device=self.device, dtype=pt.int32)
        source_length: pt.Tensor = pt.tensor(length_np, device=self.device, dtype=pt.int32)
        max_out_lengths: pt.Tensor = pt.tensor(max_output_lengths, device=self.device, dtype=pt.int32)
        target_prefix: Optional[pt.Tensor] = (
            pt.tensor(target_prefix_np, device=self.device, dtype=pt.int32) 
            if target_prefix_np is not None else None
        )
        target_prefix_factors: Optional[pt.Tensor] = (
            pt.tensor(target_prefix_factors_np, device=self.device, dtype=pt.int32) 
            if target_prefix_factors_np is not None else None
        )
        target_prefix_factors = (
            utils.shift_prefix_factors(target_prefix_factors) 
            if target_prefix_factors is not None and C.TARGET_FACTOR_SHIFT else target_prefix_factors
        )
        return (source, source_length, restrict_lexicon, max_out_lengths, target_prefix, target_prefix_factors)

    def _get_translation_tokens_and_factors(
        self, 
        target_ids: List[List[int]]
    ) -> Tuple[Tokens, str, List[Tokens], List[str]]:
        """
        Separates surface translation from factors. Input is a nested list of target ids.
        Creates tokens and output string for surface translation and for each factor, using the inverted target-side
        vocabularies. Ensures that factor strings are of the same length as the translation string.

        :param target_ids: Nested list of target ids.
        """
        all_target_tokens: List[Tokens] = []
        all_target_strings: List[str] = []
        pruned_target_ids: Generator[List[int], None, None] = (
            tokens for tokens in target_ids if not tokens[0] in self.strip_ids
        )
        for factor_index, factor_sequence in enumerate(zip(*pruned_target_ids)):
            vocab_target_inv: Dict[int, str] = self.vocab_targets_inv[factor_index]
            target_tokens: Tokens = [vocab_target_inv.get(target_id, C.UNK_SYMBOL) for target_id in factor_sequence]
            target_string: str = C.TOKEN_SEPARATOR.join(target_tokens)
            all_target_tokens.append(target_tokens)
            all_target_strings.append(target_string)
        if not all_target_strings:
            all_target_tokens = [[] for _ in range(len(self.vocab_targets_inv))]
            all_target_strings = ['' for _ in range(len(self.vocab_targets_inv))]
        tokens, *factor_tokens = all_target_tokens
        translation, *factor_translations = all_target_strings
        return (tokens, translation, factor_tokens, factor_translations)

    def _make_result(
        self, 
        trans_input: TranslatorInput, 
        translation: Translation
    ) -> TranslatorOutput:
        """
        Returns a translator result from generated target-side word ids and scores.
        Strips stop ids from translation string.

        :param trans_input: Translator input.
        :param translation: The translation and score.
        :return: TranslatorOutput.
        """
        primary_tokens, primary_translation, factor_tokens, factor_translations = self._get_translation_tokens_and_factors(
            translation.target_ids
        )
        if translation.nbest_translations is None:
            nbest_translations: Optional[List[str]] = None
            nbest_tokens: Optional[List[Tokens]] = None
            nbest_scores: Optional[List[float]] = None
            nbest_factor_translations: Optional[List[List[str]]] = None
            nbest_factor_tokens: Optional[List[List[Tokens]]] = None
        else:
            nbest_tokens: List[Tokens] = []
            nbest_translations: List[str] = []
            nbest_factor_translations: List[List[str]] = []
            nbest_factor_tokens: List[List[Tokens]] = []
            for nbest_target_ids in translation.nbest_translations.target_ids_list:
                ith_target_tokens, ith_primary_translation, ith_nbest_factor_tokens, ith_nbest_factor_translations = self._get_translation_tokens_and_factors(
                    nbest_target_ids
                )
                nbest_tokens.append(ith_target_tokens)
                nbest_translations.append(ith_primary_translation)
                nbest_factor_tokens.append(ith_nbest_factor_tokens)
                nbest_factor_translations.append(ith_nbest_factor_translations)
            nbest_scores = translation.nbest_translations.scores
        return TranslatorOutput(
            sentence_id=trans_input.sentence_id, 
            translation=primary_translation, 
            tokens=primary_tokens, 
            score=translation.scores[0], 
            pass_through_dict=trans_input.pass_through_dict, 
            nbest_translations=nbest_translations, 
            nbest_tokens=nbest_tokens, 
            nbest_scores=nbest_scores, 
            factor_translations=factor_translations, 
            factor_tokens=factor_tokens, 
            factor_scores=translation.scores[1:], 
            nbest_factor_translations=nbest_factor_translations, 
            nbest_factor_tokens=nbest_factor_tokens
        )

    def _translate_np(
        self, 
        source: pt.Tensor, 
        source_length: pt.Tensor, 
        restrict_lexicon: Optional[Any], 
        max_output_lengths: pt.Tensor, 
        target_prefix: Optional[pt.Tensor], 
        target_prefix_factors: Optional[pt.Tensor]
    ) -> List[Translation]:
        """
        Translates source of source_length and returns list of Translations.

        :param source: Source ids. Shape: (batch_size, bucket_key, num_factors).
        :param source_length: Valid source lengths.
        :param restrict_lexicon: Lexicon to use for vocabulary restriction.
        :param max_output_lengths: Tensor of maximum output lengths per input in source.
                 Shape: (batch_size,). Dtype: int32.
        :param target_prefix: Target prefix ids.
        :param target_prefix_factors: Target prefix factors ids.

        :return: List of translations.
        """
        return self._get_best_translations(
            self._search(
                source, 
                source_length, 
                restrict_lexicon, 
                max_output_lengths, 
                target_prefix, 
                target_prefix_factors
            )
        )

    def _get_best_translations(
        self, 
        result: SearchResult
    ) -> List[Translation]:
        """
        Return the nbest (aka n top) entries from the n-best list.

        :param result: SearchResult from Beam or Greedy search.
        :return: List of Translation objects containing all relevant information.
        """
        best_hyp_indices: np.ndarray = result.best_hyp_indices.cpu().numpy()
        best_word_indices: np.ndarray = result.best_word_indices.cpu().numpy()
        result_accumulated_scores_cpu: pt.Tensor = result.accumulated_scores.cpu()
        if self.dtype == pt.bfloat16:
            result_accumulated_scores_cpu = result_accumulated_scores_cpu.to(dtype=pt.float32)
        accumulated_scores: np.ndarray = result_accumulated_scores_cpu.numpy()
        lengths: np.ndarray = result.lengths.cpu().numpy()
        estimated_reference_lengths: Optional[np.ndarray] = (
            result.estimated_reference_lengths.cpu().numpy() 
            if result.estimated_reference_lengths is not None else None
        )
        batch_size: int = best_hyp_indices.shape[0] // self.beam_size
        nbest_translations: List[List[Translation]] = [[] for _ in range(batch_size)]
        reference_lengths: np.ndarray = (
            estimated_reference_lengths if estimated_reference_lengths is not None 
            else np.zeros((batch_size * self.beam_size, 1))
        )
        for n in range(self.nbest_size):
            best_ids: np.ndarray = np.arange(
                n, 
                batch_size * self.beam_size, 
                self.beam_size, 
                dtype='int32'
            )
            indices: np.ndarray = self._get_best_word_indices_for_kth_hypotheses(
                ks=best_ids, 
                all_hyp_indices=best_hyp_indices
            )
            indices_shape_1: int = indices.shape[1]
            for i in range(batch_size):
                sequence = best_word_indices[indices[i], :, np.arange(indices_shape_1)]
                length = lengths[best_ids[i]]
                score = accumulated_scores[best_ids[i]]
                ref_length = reference_lengths[best_ids[i]][0]
                translation = self._assemble_translation(
                    sequence=sequence, 
                    length=length, 
                    seq_scores=[score], 
                    estimated_reference_length=ref_length, 
                    unshift_target_factors=C.TARGET_FACTOR_SHIFT
                )
                nbest_translations[i].append(translation)
        reduced_translations: List[Translation] = [
            _reduce_nbest_translations(grouped_nbest) 
            for grouped_nbest in zip(*nbest_translations)
        ]
        return reduced_translations

    @staticmethod
    def _get_best_word_indices_for_kth_hypotheses(
        ks: np.ndarray, 
        all_hyp_indices: np.ndarray
    ) -> np.ndarray:
        """
        Traverses the matrix of best hypotheses indices collected during beam search in reversed order by
        using the kth hypotheses index as a backpointer.
        Returns an array containing the indices into the best_word_indices collected during beam search to extract
        the kth hypothesis.

        :param ks: The kth-best hypotheses to extract. Supports multiple for batch_size > 1. Shape: (batch,).
        :param all_hyp_indices: All best hypotheses indices list collected in beam search. Shape: (batch * beam, steps).
        :return: Array of indices into the best_word_indices collected during beam search
            that extract the kth-best hypothesis. Shape: (batch,).
        """
        batch_size: int = ks.shape[0]
        num_steps: int = all_hyp_indices.shape[1]
        result: np.ndarray = np.zeros((batch_size, num_steps - 1), dtype=all_hyp_indices.dtype)
        pointer: np.ndarray = all_hyp_indices[ks, -1]
        for step in range(num_steps - 2, -1, -1):
            result[:, step] = pointer
            pointer = all_hyp_indices[pointer, step]
        return result

    @staticmethod
    def _assemble_translation(
        sequence: np.ndarray, 
        length: int, 
        seq_scores: List[float], 
        estimated_reference_length: float, 
        unshift_target_factors: bool = False
    ) -> Translation:
        """
        Takes a set of data pertaining to a single translated item, performs slightly different
        processing on each, and merges it into a Translation object.
        :param sequence: Array of word ids. Shape: (bucketed_length, num_target_factors).
        :param length: The length of the translated segment.
        :param seq_scores: Array of length-normalized negative log-probs, one for each factor.
        :param estimated_reference_length: Estimated reference length (if any).
        :return: A Translation object.
        """
        if unshift_target_factors:
            sequence = _unshift_target_factors(sequence, fill_last_with=C.EOS_ID)
        else:
            sequence = sequence.tolist()
        sequence = sequence[:length]
        scores = seq_scores
        estimated_reference_length = float(estimated_reference_length) if estimated_reference_length else None
        return Translation(
            target_ids=sequence, 
            scores=scores, 
            nbest_translations=None, 
            estimated_reference_length=estimated_reference_length
        )

def _unshift_target_factors(
    sequence: Union[List[int], np.ndarray], 
    fill_last_with: int = C.EOS_ID
) -> List[Union[int, Tuple[int, ...]]]:
    """
    Shifts back target factors so that they re-align with the words.

    :param sequence: Array of word ids. Shape: (bucketed_length, num_target_factors).
    """
    if isinstance(sequence, list) or (isinstance(sequence, np.ndarray) and len(sequence.shape) == 1):
        return sequence
    if sequence.shape[1] == 1:
        return sequence[:, 0].tolist()
    num_factors_to_shift: int = sequence.shape[1] - 1
    _fillvalue: List[int] = num_factors_to_shift * [fill_last_with]
    _words: List[int] = sequence[:, 0].tolist()
    _next_factors: List[List[int]] = sequence[1:, 1:].tolist()
    sequence_list: List[Union[int, Tuple[int, ...]]] = [
        (w, *fs) for w, fs in itertools.zip_longest(_words, _next_factors, fillvalue=_fillvalue)
    ]
    return sequence_list

class Translator:
    # Previous definitions...

    def _translate_np(
        self, 
        source: pt.Tensor, 
        source_length: pt.Tensor, 
        restrict_lexicon: Optional[Any], 
        max_output_lengths: pt.Tensor, 
        target_prefix: Optional[pt.Tensor], 
        target_prefix_factors: Optional[pt.Tensor]
    ) -> List[Translation]:
        # Already included above

    # The rest of the Translator class continues as previously defined
