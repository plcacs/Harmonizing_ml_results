from typing import List, Optional, Tuple, Dict, Set, Union, Callable, Generator, Any
import numpy as np
import torch as pt
from dataclasses import dataclass
from functools import partial
C = type('C', (), {'TARGET_MAX_LENGTH_FACTOR': 2.0, 'SPACE_FOR_XOS': 1, 'EOS_SYMBOL': '</s>', 'JSON_TEXT_KEY': 'text', 'JSON_FACTORS_KEY': 'factors', 'JSON_SOURCE_PREFIX_KEY': 'source_prefix', 'JSON_SOURCE_PREFIX_FACTORS_KEY': 'source_prefix_factors', 'JSON_TARGET_PREFIX_KEY': 'target_prefix', 'JSON_TARGET_PREFIX_FACTORS_KEY': 'target_prefix_factors', 'JSON_USE_TARGET_PREFIX_ALL_CHUNKS_KEY': 'use_target_prefix_all_chunks', 'JSON_KEEP_TARGET_PREFIX_KEY': 'keep_target_prefix_key', 'JSON_RESTRICT_LEXICON_KEY': 'restrict_lexicon', 'JSON_AVOID_KEY': 'avoid', 'JSON_CONSTRAINTS_KEY': 'constraints', 'DEFAULT_FACTOR_DELIMITER': '|', 'PAD_ID': 0, 'EOS_ID': 1, 'UNK_ID': 2, 'BEAM_SEARCH_STOP_ALL': 'all', 'DEFAULT_KNN_LAMBDA': 0.5, 'DEFAULT_NUM_STD_MAX_OUTPUT_LENGTH': 2, 'TARGET_FACTOR_SHIFT': True, 'TOKEN_SEPARATOR': ' '})

class SockeyeModel:
    length_ratio_mean: float
    length_ratio_std: float
    max_supported_len_source: int
    max_supported_len_target: int
    num_source_factors: int
    num_target_factors: int
    eop_id: int
    dtype: Any

class CandidateScorer:

    def unnormalize(self, score, length, estimated_reference_length):
        pass

    def __call__(self, score, length, estimated_reference_length):
        pass

class SearchResult:
    best_hyp_indices: pt.Tensor
    best_word_indices: pt.Tensor
    accumulated_scores: pt.Tensor
    lengths: pt.Tensor
    estimated_reference_lengths: Optional[pt.Tensor]

class GreedySearch:
    pass

class lexicon:

    class RestrictLexicon:
        pass

class vocab:

    class Vocab:
        pass

    @staticmethod
    def reverse_vocab(v):
        pass

class utils:

    @staticmethod
    def check_condition(condition, message):
        pass

    @staticmethod
    def get_tokens(string):
        pass

    @staticmethod
    def grouper(iterable, n):
        pass

    @staticmethod
    def shift_prefix_factors(target_prefix_factors):
        pass
Tokens = List[str]
TokenIds = List[List[int]]
SentenceId = Union[int, str]

@dataclass
class TranslatorInput:
    sentence_id: SentenceId
    tokens: Tokens
    factors: Optional[List[Tokens]] = None
    source_prefix_tokens: Optional[Tokens] = None
    source_prefix_factors: Optional[List[Tokens]] = None
    target_prefix_tokens: Optional[Tokens] = None
    target_prefix_factors: Optional[List[Tokens]] = None
    use_target_prefix_all_chunks: Optional[bool] = True
    keep_target_prefix_key: Optional[bool] = True
    restrict_lexicon: Optional[lexicon.RestrictLexicon] = None
    constraints: Optional[List[Tokens]] = None
    avoid_list: Optional[List[Tokens]] = None
    pass_through_dict: Optional[Dict[str, Any]] = None

    def __str__(self):
        return f'TranslatorInput({self.sentence_id}, {self.tokens}, factors={self.factors}, source_prefix_tokens={self.source_prefix_tokens}, source_prefix_factors={self.source_prefix_factors}, target_prefix_tokens={self.target_prefix_tokens}, target_prefix_factors={self.target_prefix_factors}, use_target_prefix_all_chunks={self.use_target_prefix_all_chunks}, keep_target_prefix_key={self.keep_target_prefix_key}, constraints={self.constraints}, avoid={self.avoid_list})'

    def __len__(self):
        return len(self.tokens) + self.num_source_prefix_tokens

    @property
    def num_factors(self):
        return 1 + (0 if not self.factors else len(self.factors))

    def get_source_prefix_tokens(self):
        return self.source_prefix_tokens if self.source_prefix_tokens is not None else []

    @property
    def num_source_prefix_tokens(self):
        return len(self.get_source_prefix_tokens())

    def get_target_prefix_tokens(self):
        return self.target_prefix_tokens if self.target_prefix_tokens is not None else []

    @property
    def num_target_prefix_tokens(self):
        return len(self.get_target_prefix_tokens())

    def get_target_prefix_factors(self):
        return self.target_prefix_factors if self.target_prefix_factors is not None else [[]]

    @property
    def num_target_prefix_factors(self):
        return len(self.get_target_prefix_factors()[0])

    def chunks(self, chunk_size):
        if len(self.tokens) > chunk_size and self.constraints is not None:
            logger.warning('Input %s has length (%d) that exceeds max input length (%d), triggering internal splitting. Placing all target-side constraints with the first chunk, which is probably wrong.', self.sentence_id, len(self.tokens), chunk_size)
        for chunk_id, i in enumerate(range(0, len(self) - self.num_source_prefix_tokens, chunk_size)):
            factors = [factor[i:i + chunk_size] for factor in self.factors] if self.factors is not None else None
            constraints = self.constraints if chunk_id == 0 else None
            target_prefix_tokens = self.target_prefix_tokens if chunk_id == 0 or self.use_target_prefix_all_chunks else None
            target_prefix_factors = self.target_prefix_factors if chunk_id == 0 or self.use_target_prefix_all_chunks else None
            pass_through_dict = copy.deepcopy(self.pass_through_dict) if chunk_id == 0 and self.pass_through_dict is not None else None
            yield TranslatorInput(sentence_id=self.sentence_id, tokens=self.tokens[i:i + chunk_size], factors=factors, source_prefix_tokens=self.source_prefix_tokens, source_prefix_factors=self.source_prefix_factors, target_prefix_tokens=target_prefix_tokens, target_prefix_factors=self.target_prefix_factors, use_target_prefix_all_chunks=self.use_target_prefix_all_chunks, keep_target_prefix_key=self.keep_target_prefix_key, restrict_lexicon=self.restrict_lexicon, constraints=constraints, avoid_list=self.avoid_list, pass_through_dict=pass_through_dict)

    def with_eos(self):
        return TranslatorInput(sentence_id=self.sentence_id, tokens=self.tokens + [C.EOS_SYMBOL], factors=[factor + [C.EOS_SYMBOL] for factor in self.factors] if self.factors is not None else None, source_prefix_tokens=self.source_prefix_tokens, source_prefix_factors=self.source_prefix_factors, target_prefix_tokens=self.target_prefix_tokens, target_prefix_factors=self.target_prefix_factors, use_target_prefix_all_chunks=self.use_target_prefix_all_chunks, keep_target_prefix_key=self.keep_target_prefix_key, restrict_lexicon=self.restrict_lexicon, constraints=self.constraints, avoid_list=self.avoid_list, pass_through_dict=self.pass_through_dict)

class BadTranslatorInput(TranslatorInput):

    def __init__(self, sentence_id, tokens):
        super().__init__(sentence_id=sentence_id, tokens=tokens, factors=None)

def _bad_input(sentence_id, reason=''):
    logger.warning("Bad input (%s): '%s'. Will return empty output.", sentence_id, reason.strip())
    return BadTranslatorInput(sentence_id=sentence_id, tokens=[])

def make_input_from_plain_string(sentence_id, string):
    return TranslatorInput(sentence_id, tokens=list(utils.get_tokens(string)), factors=None)

def make_input_from_json_string(sentence_id, json_string, translator):
    try:
        jobj = json.loads(json_string)
        return make_input_from_dict(sentence_id, jobj, translator)
    except Exception as e:
        logger.exception(e, exc_info=True)
        return _bad_input(sentence_id, reason=json_string)

def make_input_from_dict(sentence_id, input_dict, translator):
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
        if isinstance(constraints, list):
            constraints = [list(utils.get_tokens(constraint)) for constraint in constraints]
        return TranslatorInput(sentence_id=sentence_id, tokens=tokens, factors=factors, source_prefix_tokens=source_prefix_tokens, source_prefix_factors=source_prefix_factors, target_prefix_tokens=target_prefix_tokens, target_prefix_factors=target_prefix_factors, use_target_prefix_all_chunks=use_target_prefix_all_chunks, keep_target_prefix_key=keep_target_prefix_key, restrict_lexicon=restrict_lexicon, constraints=constraints, avoid_list=avoid_list, pass_through_dict=input_dict)
    except Exception as e:
        logger.exception(e, exc_info=True)
        return _bad_input(sentence_id, reason=str(input_dict))

def make_input_from_factored_string(sentence_id, factored_string, translator, delimiter=C.DEFAULT_FACTOR_DELIMITER):
    utils.check_condition(bool(delimiter) and (not delimiter.isspace()), 'Factor delimiter can not be whitespace or empty.')
    model_num_source_factors = translator.num_source_factors
    if model_num_source_factors == 1:
        return make_input_from_plain_string(sentence_id=sentence_id, string=factored_string)
    tokens = []
    factors = [[] for _ in range(model_num_source_factors - 1)]
    for token_id, token in enumerate(utils.get_tokens(factored_string)):
        pieces = token.split(delimiter)
        if not all(pieces) or len(pieces) != model_num_source_factors:
            logger.error("Failed to parse %d factors at position %d ('%s') in '%s'" % (model_num_source_factors, token_id, token, factored_string.strip()))
            return _bad_input(sentence_id, reason=factored_string)
        tokens.append(pieces[0])
        for i, factor in enumerate(factors):
            factors[i].append(pieces[i + 1])
    return TranslatorInput(sentence_id=sentence_id, tokens=tokens, factors=factors)

def make_input_from_multiple_strings(sentence_id, strings):
    if not bool(strings):
        return TranslatorInput(sentence_id=sentence_id, tokens=[], factors=None)
    tokens = list(utils.get_tokens(strings[0]))
    factors = [list(utils.get_tokens(factor)) for factor in strings[1:]]
    if not all((len(factor) == len(tokens) for factor in factors)):
        logger.error("Length of string sequences do not match: '%s'", strings)
        return _bad_input(sentence_id, reason=str(strings))
    return TranslatorInput(sentence_id=sentence_id, tokens=tokens, factors=factors)

@dataclass
class TranslatorOutput:
    sentence_id: SentenceId
    translation: str
    tokens: Tokens
    score: float
    pass_through_dict: Optional[Dict[str, Any]] = None
    nbest_translations: Optional[List[str]] = None
    nbest_tokens: Optional[List[Tokens]] = None
    nbest_scores: Optional[List[List[float]]] = None
    factor_translations: Optional[List[str]] = None
    factor_tokens: Optional[List[Tokens]] = None
    factor_scores: Optional[List[float]] = None
    nbest_factor_translations: Optional[List[List[str]]] = None
    nbest_factor_tokens: Optional[List[List[Tokens]]] = None

    def json(self):
        _d = copy.deepcopy(self.pass_through_dict) if self.pass_through_dict is not None else {}
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
                _d['translations_factors'].append({f'factor{i}': factor_translation for i, factor_translation in enumerate(factor_translations, 1)})
        return _d

@dataclass
class NBestTranslations:
    target_ids_list: List[TokenIds]
    scores: List[List[float]]

@dataclass
class Translation:
    target_ids: TokenIds
    scores: List[float]
    nbest_translations: Optional[NBestTranslations] = None
    estimated_reference_length: Optional[float] = None

def empty_translation(add_nbest=False):
    return Translation(target_ids=[], scores=[-np.inf], nbest_translations=NBestTranslations([], []) if add_nbest else None)

@dataclass
class IndexedTranslatorInput:
    input_idx: int
    chunk_idx: int
    translator_input: TranslatorInput

@dataclass(order=True)
class IndexedTranslation:
    input_idx: int
    chunk_idx: int
    translation: Translation

def _concat_nbest_translations(translations, stop_ids, scorer):
    expanded_translations = (_expand_nbest_translation(translation) for translation in translations)
    concatenated_translations = []
    for translations_to_concat in zip(*expanded_translations):
        concatenated_translations.append(_concat_translations(translations=list(translations_to_concat), stop_ids=stop_ids, scorer=scorer))
    return _reduce_nbest_translations(concatenated_translations)

def _reduce_nbest_translations(nbest_translations_list):
    best_translation = nbest_translations_list[0]
    sequences = [translation.target_ids for translation in nbest_translations_list]
    scores = [translation.scores for translation in nbest_translations_list]
    nbest_translations = NBestTranslations(sequences, scores)
    return Translation(best_translation.target_ids, best_translation.scores, nbest_translations, best_translation.estimated_reference_length)

def _expand_nbest_translation(translation):
    nbest_list = []
    for target_ids, score in zip(translation.nbest_translations.target_ids_list, translation.nbest_translations.scores):
        nbest_list.append(Translation(target_ids, score, estimated_reference_length=translation.estimated_reference_length))
    return nbest_list

def _remove_target_prefix_tokens(target_ids, num_target_prefix_tokens):
    starting_idx = min(len(target_ids), num_target_prefix_tokens)
    return target_ids[starting_idx:]

def _concat_translations(translations, stop_ids, scorer):
    if len(translations) == 1:
        return translations[0]
    target_ids = []
    estimated_reference_length = None
    scores = np.zeros_like(translations[0].scores)
    for idx, translation in enumerate(translations):
        if idx == len(translations) - 1:
            target_ids.extend(translation.target_ids)
        elif translation.target_ids[-1][0] in stop_ids:
            target_ids.extend(translation.target_ids[:-1])
        else:
            target_ids.extend(translation.target_ids)
        if translation.estimated_reference_length is not None:
            if estimated_reference_length is None:
                estimated_reference_length = translation.estimated_reference_length
            else:
                estimated_reference_length += translation.estimated_reference_length
        score, *factor_scores = translation.scores
        raw_score = scorer.unnormalize(score, len(translation.target_ids), translation.estimated_reference_length)
        scores = np.add(scores, [raw_score, *factor_scores])
    scores[0] = scorer(scores[0], len(target_ids), estimated_reference_length)
    return Translation(target_ids, scores.tolist(), estimated_reference_length=estimated_reference_length)

class Translator:

    def __init__(self, device, ensemble_mode, scorer, batch_size, beam_search_stop, models, source_vocabs, target_vocabs, beam_size=5, nbest_size=1, restrict_lexicon=None, strip_unknown_words=False, sample=None, output_scores=False, constant_length_ratio=0.0, knn_lambda=C.DEFAULT_KNN_LAMBDA, max_output_length_num_stds=C.DEFAULT_NUM_STD_MAX_OUTPUT_LENGTH, max_input_length=None, max_output_length=None, prevent_unk=False, greedy=False, skip_nvs=False, nvs_thresh=0.5):
        self.device = device
        self.dtype = models[0].dtype
        self._scorer = scorer
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.beam_search_stop = beam_search_stop
        self.source_vocabs = source_vocabs
        self.vocab_targets = target_vocabs
        self.vocab_targets_inv = [vocab.reverse_vocab(v) for v in self.vocab_targets]
        self.restrict_lexicon = restrict_lexicon
        assert C.PAD_ID == 0, 'pad id should be 0'
        self.stop_ids = {C.EOS_ID, C.PAD_ID}
        self.strip_ids = self.stop_ids.copy()
        self.unk_id = C.UNK_ID
        if strip_unknown_words:
            self.strip_ids.add(self.unk_id)
        self.models = models
        for model in self.models:
            model.eval()
        self._max_input_length, self._get_max_output_length = models_max_input_output_length(models, max_output_length_num_stds, forced_max_input_length=max_input_length, forced_max_output_length=max_output_length)
        self.nbest_size = nbest_size
        utils.check_condition(self.beam_size >= nbest_size, 'nbest_size must be smaller or equal to beam_size.')
        if self.nbest_size > 1:
            utils.check_condition(self.beam_search_stop == C.BEAM_SEARCH_STOP_ALL, "nbest_size > 1 requires beam_search_stop to be set to 'all'")
        self._search = get_search_algorithm(models=self.models, beam_size=self.beam_size, device=self.device, output_scores=output_scores, sample=sample, ensemble_mode=ensemble_mode, beam_search_stop=beam_search_stop, scorer=self._scorer, constant_length_ratio=constant_length_ratio, knn_lambda=knn_lambda, prevent_unk=prevent_unk, greedy=greedy, skip_nvs=skip_nvs, nvs_thresh=nvs_thresh)
        self._concat_translations = partial(_concat_nbest_translations if self.nbest_size > 1 else _concat_translations, stop_ids=self.stop_ids, scorer=self._scorer)
        logger.info('Translator (%d model(s) beam_size=%d algorithm=%s, beam_search_stop=%s max_input_length=%s nbest_size=%s ensemble_mode=%s max_batch_size=%d dtype=%s skip_nvs=%s nvs_thresh=%s)', len(self.models), self.beam_size, 'GreedySearch' if isinstance(self._search, GreedySearch) else 'BeamSearch', self.beam_search_stop, self.max_input_length, self.nbest_size, 'None' if len(self.models) == 1 else ensemble_mode, self.max_batch_size, self.dtype, skip_nvs, nvs_thresh)

    @property
    def max_input_length(self):
        return self._max_input_length - C.SPACE_FOR_XOS

    @property
    def max_batch_size(self):
        return self.batch_size

    @property
    def num_source_factors(self):
        return self.models[0].num_source_factors

    @property
    def num_target_factors(self):
        return self.models[0].num_target_factors

    @property
    def eop_id(self):
        return self.models[0].eop_id

    def translate(self, trans_inputs, fill_up_batches=True):
        num_inputs = len(trans_inputs)
        translated_chunks = []
        input_chunks = []
        for trans_input_idx, trans_input in enumerate(trans_inputs):
            if isinstance(trans_input, BadTranslatorInput):
                translated_chunks.append(IndexedTranslation(input_idx=trans_input_idx, chunk_idx=0, translation=empty_translation(add_nbest=self.nbest_size > 1)))
            elif len(trans_input.tokens) == 0:
                translated_chunks.append(IndexedTranslation(input_idx=trans_input_idx, chunk_idx=0, translation=empty_translation(add_nbest=self.nbest_size > 1)))
            else:
                max_input_length_for_chunking = self.max_input_length - trans_input.num_source_prefix_tokens
                if max_input_length_for_chunking <= 0:
                    logger.warning('Input %s has a source prefix with length (%d) that already equals or exceeds max input length (%d). Return an empty translation instead.', trans_input.sentence_id, trans_input.num_source_prefix_tokens, self.max_input_length)
                    translated_chunks.append(IndexedTranslation(input_idx=trans_input_idx, chunk_idx=0, translation=empty_translation(add_nbest=self.nbest_size > 1)))
                elif len(trans_input.tokens) > max_input_length_for_chunking:
                    logger.debug('Input %s has length (%d) that exceeds max input length (%d). Splitting into chunks of size %d.', trans_input.sentence_id, len(trans_input.tokens), max_input_length_for_chunking, max_input_length_for_chunking)
                    chunks = [trans_input_chunk.with_eos() for trans_input_chunk in trans_input.chunks(max_input_length_for_chunking)]
                    input_chunks.extend([IndexedTranslatorInput(trans_input_idx, chunk_idx, chunk_input) for chunk_idx, chunk_input in enumerate(chunks)])
                else:
                    input_chunks.append(IndexedTranslatorInput(trans_input_idx, chunk_idx=0, translator_input=trans_input.with_eos()))
            if trans_input.constraints is not None:
                logger.info('Input %s has %d %s: %s', trans_input.sentence_id, len(trans_input.constraints), 'constraint' if len(trans_input.constraints) == 1 else 'constraints', ', '.join((' '.join(x) for x in trans_input.constraints)))
        num_bad_empty = len(translated_chunks)
        input_chunks = sorted(input_chunks, key=lambda chunk: len(chunk.translator_input.tokens), reverse=True)
        batch_size = self.max_batch_size if fill_up_batches else min(len(input_chunks), self.max_batch_size)
        num_batches = 0
        for batch_id, batch in enumerate(utils.grouper(input_chunks, batch_size)):
            logger.debug('Translating batch %d', batch_id)
            rest = batch_size - len(batch)
            if fill_up_batches and rest > 0:
                logger.debug('Padding batch of size %d to full batch size (%d)', len(batch), batch_size)
                batch = batch + [batch[0]] * rest
            translator_inputs = [indexed_translator_input.translator_input for indexed_translator_input in batch]
            batch_translations = self._translate_batch(translator_inputs)
            if fill_up_batches and rest > 0:
                batch_translations = batch_translations[:-rest]
            for chunk, translation in zip(batch, batch_translations):
                translated_chunks.append(IndexedTranslation(chunk.input_idx, chunk.chunk_idx, translation))
            num_batches += 1
        translated_chunks = sorted(translated_chunks)
        num_chunks = len(translated_chunks)
        results = []
        chunks_by_input_idx = itertools.groupby(translated_chunks, key=lambda translation: translation.input_idx)
        for trans_input, (input_idx, translations_for_input_idx) in zip(trans_inputs, chunks_by_input_idx):
            translations_for_input_idx = list(translations_for_input_idx)
            num_target_prefix_tokens = trans_input.num_target_prefix_tokens
            if len(translations_for_input_idx) == 1:
                translation = translations_for_input_idx[0].translation
                if num_target_prefix_tokens > 0 and (not trans_input.keep_target_prefix_key):
                    translation.target_ids = _remove_target_prefix_tokens(translation.target_ids, num_target_prefix_tokens)
            else:
                translations_to_concat = [translated_chunk.translation for translated_chunk in translations_for_input_idx]
                if num_target_prefix_tokens > 0 and (not trans_input.keep_target_prefix_key):
                    for i in range(len(translations_to_concat)):
                        if i == 0 or trans_input.use_target_prefix_all_chunks:
                            translations_to_concat[i].target_ids = _remove_target_prefix_tokens(translations_to_concat[i].target_ids, num_target_prefix_tokens)
                translation = self._concat_translations(translations_to_concat)
            results.append(self._make_result(trans_input, translation))
        num_outputs = len(results)
        logger.debug('Translated %d inputs (%d chunks) in %d batches to %d outputs. %d empty/bad inputs.', num_inputs, num_chunks, num_batches, num_outputs, num_bad_empty)
        self._search.log_search_stats()
        return results

    def _translate_batch(self, translator_inputs):
        with pt.inference_mode():
            return self._translate_np(*self._get_inference_input(translator_inputs))

    def _get_inference_input(self, trans_inputs):
        batch_size = len(trans_inputs)
        max_target_prefix_length = max((inp.num_target_prefix_tokens for inp in trans_inputs))
        max_target_prefix_factors_length = max((inp.num_target_prefix_factors for inp in trans_inputs))
        max_length = max((len(inp) for inp in trans_inputs))
        source_np = np.zeros((batch_size, max_length, self.num_source_factors), dtype='int32')
        length_np = np.zeros((batch_size, 2), dtype='int32')
        target_prefix_np = np.zeros((batch_size, max_target_prefix_length), dtype='int32')