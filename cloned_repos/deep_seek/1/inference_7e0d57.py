"""
Code for inference/translation
"""
import copy
import itertools
import json
import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union, Iterable
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

def models_max_input_output_length(models: List[SockeyeModel], 
                                  num_stds: int, 
                                  forced_max_input_length: Optional[int] = None, 
                                  forced_max_output_length: Optional[int] = None) -> Tuple[int, Callable[[int], int]]:
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
    return get_max_input_output_length(supported_max_seq_len_source, supported_max_seq_len_target, length_ratio_mean=max_mean, length_ratio_std=max_std, num_stds=num_stds, forced_max_input_len=forced_max_input_length, forced_max_output_len=forced_max_output_length)

def get_max_input_output_length(supported_max_seq_len_source: int, 
                               supported_max_seq_len_target: int, 
                               length_ratio_mean: float, 
                               length_ratio_std: float, 
                               num_stds: int, 
                               forced_max_input_len: Optional[int] = None, 
                               forced_max_output_len: Optional[int] = None) -> Tuple[int, Callable[[int], int]]:
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
    factors: Optional[List[Tokens]] = None
    source_prefix_tokens: Optional[Tokens] = None
    source_prefix_factors: Optional[List[Tokens]] = None
    target_prefix_tokens: Optional[Tokens] = None
    target_prefix_factors: Optional[List[Tokens]] = None
    use_target_prefix_all_chunks: bool = True
    keep_target_prefix_key: bool = True
    restrict_lexicon: Optional[Any] = None
    constraints: Optional[List[Tokens]] = None
    avoid_list: Optional[List[Tokens]] = None
    pass_through_dict: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        return f'TranslatorInput({self.sentence_id}, {self.tokens}, factors={self.factors}, source_prefix_tokens={self.source_prefix_tokens}, source_prefix_factors={self.source_prefix_factors}, target_prefix_tokens={self.target_prefix_tokens}, target_prefix_factors={self.target_prefix_factors}, use_target_prefix_all_chunks={self.use_target_prefix_all_chunks}, keep_target_prefix_key={self.keep_target_prefix_key}, constraints={self.constraints}, avoid={self.avoid_list})'

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

    def get_target_prefix_factors(self) -> List[Tokens]:
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
            logger.warning('Input %s has length (%d) that exceeds max input length (%d), triggering internal splitting. Placing all target-side constraints with the first chunk, which is probably wrong.', self.sentence_id, len(self.tokens), chunk_size)
        for chunk_id, i in enumerate(range(0, len(self) - self.num_source_prefix_tokens, chunk_size)):
            factors = [factor[i:i + chunk_size] for factor in self.factors] if self.factors is not None else None
            constraints = self.constraints if chunk_id == 0 else None
            target_prefix_tokens = self.target_prefix_tokens if chunk_id == 0 or self.use_target_prefix_all_chunks else None
            target_prefix_factors = self.target_prefix_factors if chunk_id == 0 or self.use_target_prefix_all_chunks else None
            pass_through_dict = copy.deepcopy(self.pass_through_dict) if chunk_id == 0 and self.pass_through_dict is not None else None
            yield TranslatorInput(sentence_id=self.sentence_id, tokens=self.tokens[i:i + chunk_size], factors=factors, source_prefix_tokens=self.source_prefix_tokens, source_prefix_factors=self.source_prefix_factors, target_prefix_tokens=target_prefix_tokens, target_prefix_factors=self.target_prefix_factors, use_target_prefix_all_chunks=self.use_target_prefix_all_chunks, keep_target_prefix_key=self.keep_target_prefix_key, restrict_lexicon=self.restrict_lexicon, constraints=constraints, avoid_list=self.avoid_list, pass_through_dict=pass_through_dict)

    def with_eos(self) -> 'TranslatorInput':
        """
        :return: A new translator input with EOS appended to the tokens and factors.
        """
        return TranslatorInput(sentence_id=self.sentence_id, tokens=self.tokens + [C.EOS_SYMBOL], factors=[factor + [C.EOS_SYMBOL] for factor in self.factors] if self.factors is not None else None, source_prefix_tokens=self.source_prefix_tokens, source_prefix_factors=self.source_prefix_factors, target_prefix_tokens=self.target_prefix_tokens, target_prefix_factors=self.target_prefix_factors, use_target_prefix_all_chunks=self.use_target_prefix_all_chunks, keep_target_prefix_key=self.keep_target_prefix_key, restrict_lexicon=self.restrict_lexicon, constraints=self.constraints, avoid_list=self.avoid_list, pass_through_dict=self.pass_through_dict)

class BadTranslatorInput(TranslatorInput):

    def __init__(self, sentence_id: SentenceId, tokens: Tokens):
        super().__init__(sentence_id=sentence_id, tokens=tokens, factors=None)

def _bad_input(sentence_id: SentenceId, reason: str = '') -> BadTranslatorInput:
    logger.warning("Bad input (%s): '%s'. Will return empty output.", sentence_id, reason.strip())
    return BadTranslatorInput(sentence_id=sentence_id, tokens=[])

def make_input_from_plain_string(sentence_id: SentenceId, string: str) -> TranslatorInput:
    """
    Returns a TranslatorInput object from a plain string.

    :param sentence_id: Sentence id.
    :param string: An input string.
    :return: A TranslatorInput.
    """
    return TranslatorInput(sentence_id, tokens=list(utils.get_tokens(string)), factors=None)

def make_input_from_json_string(sentence_id: SentenceId, json_string: str, translator: 'Translator') -> TranslatorInput:
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

def make_input_from_dict(sentence_id: SentenceId, input_dict: Dict[str, Any], translator: 'Translator') -> TranslatorInput:
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
        tokens = input_dict[C.JSON_TEXT_KEY]
        tokens = list(utils.get_tokens(tokens))
        factors = input_dict.get(C.JSON_FACTORS_KEY)
        source_prefix_tokens = input_dict.get(C.JSON_SOURCE_PREFIX_KEY)
        source_prefix_tokens = list(utils.get_tokens(source_prefix_tokens)) if source_prefix_tokens is not None else None
        if source_prefix_tokens is not None and (not source_prefix_tokens):
            logger.warning(f"Empty string is specified as a source prefix for input '{input_dict[C.JSON_SOURCE_PREFIX_KEY]}'.")
        source_prefix_factors = input_dict.get(C.JSON_SOURCE_PREFIX_FACTORS_KEY)
        if source_prefix_factors is not None and (not source_prefix_tokens):
            logger.error('Source prefix factors cannot be specified when source prefix is not specified')
            return _bad_input(sentence_id, reason=str(input_dict))
        if source_prefix_factors is not None and (not factors):
            logger.error('Source prefix factors cannot be specified when source factors are not specified')
            return _bad_input(sentence_id, reason=str(input_dict))
        if source_prefix_tokens is not None and (factors is not None and (not source_prefix_factors)):
            logger.error('Source prefix factors need to be also specified together with source factors')
            return _bad_input(sentence_id, reason=str(input_dict))
        if isinstance(factors, list):
            factors = [list(utils.get_tokens(factor)) for factor in factors]
            lengths = [len(f) for f in factors]
            if not all((length == len(tokens) for length in lengths)):
                logger.error('Factors have different length than input text: %d vs. %s', len(tokens), str(lengths))
                return _bad_input(sentence_id, reason=str(input_dict))
        if isinstance(source_prefix_factors, list):
            source_prefix_factors = [list(utils.get_tokens(spf)) for spf in source_prefix_factors]
            for source_prefix_factor in source_prefix_factors:
                if not source_prefix_factor:
                    logger.warning(f"Empty list is specified as source prefix factors for input '%s'.", input_dict[C.JSON_TEXT_KEY])
            lengths = [len(source_prefix_factor) for source_prefix_factor in source_prefix_factors]
            if not all((len(source_prefix_tokens) == length for length in lengths)):
                logger.error('Source prefix has %d tokens but there are %s prefix factors', len(source_prefix_tokens), str(lengths))
                return _bad_input(sentence_id, reason=str(input_dict))
            if len(source_prefix_factors) != len(factors):
                logger.error('There is mismatch in source factors %d and prefix factors %d', len(factors), len(source_prefix_factors))
                return _bad_input(sentence_id, reason=str(input_dict))
        target_prefix_tokens = input_dict.get(C.JSON_TARGET_PREFIX_KEY)
        target_prefix_tokens = list(utils.get_tokens(target_prefix_tokens)) if target_prefix_tokens is not None else None
        if target_prefix_tokens is not None and (not target_prefix_tokens):
            logger.warning(f"Empty string is specified as a target prefix for input '{input_dict[C.JSON_TEXT_KEY]}'.")
        target_prefix_factors = input_dict.get(C.JSON_TARGET_PREFIX_FACTORS_KEY)
        if isinstance(target_prefix_factors, list):
            target_prefix_factors = [list(utils.get_tokens(tpf)) for tpf in target_prefix_factors]
            if len(target_prefix_factors) != translator.num_target_factors - 1:
                logger.error('Must provide target prefix for each target factor. Given: %s required: %s', len(target_prefix_factors), translator.num_target_factors - 1)
                return _bad_input(sentence_id, reason=str(input_dict))
        use_target_prefix_all_chunks = input_dict.get(C.JSON_USE_TARGET_PREFIX_ALL_CHUNKS_KEY, True)
        keep_target_prefix_key = input_dict.get(C.JSON_KEEP_TARGET_PREFIX_KEY, True)
        restrict_lexicon = None
        restrict_lexicon_name = input_dict.get(C.JSON_RESTRICT_LEXICON_KEY, None)
        if isinstance(translator.restrict_lexicon, dict) and restrict_lexicon_name is not None:
            restrict_lexicon = translator.restrict_lexicon.get(restrict_lexicon_name, None)
            if restrict_lexicon is None:
                logger.error("Unknown restrict_lexicon '%s'. Choices: %s" % (restrict_lexicon_name, ' '.join(sorted(translator.restrict_lexicon))))
                return _bad_input(sentence_id, reason=str(input_dict))
        avoid_list = input_dict.get(C.JSON_AVOID_KEY)
        constraints = input_dict.get(C.JSON_CONSTRAINTS_KEY)
        if constraints is not None and avoid_list is not None:
            avoid_set = set(avoid_list)
            overlap = set(constraints).intersection(avoid_set)
            if len(overlap) > 0:
                logger.warning('Overlap between constraints and avoid set, dropping the overlapping avoids')
                avoid_list = list(avoid_set.difference(overlap))
        if isinstance(avoid_list, list):
            avoid_list = [list(utils.get_tokens(phrase)) for phrase in avoid_list]
       