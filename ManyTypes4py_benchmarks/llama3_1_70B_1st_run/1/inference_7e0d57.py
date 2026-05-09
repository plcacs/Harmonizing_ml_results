from typing import List, Optional, Tuple, Dict, Any, Union

def models_max_input_output_length(models: List[SockeyeModel], num_stds: int, forced_max_input_length: Optional[int] = None, forced_max_output_length: Optional[int] = None) -> Tuple[int, Callable[[int], int]]:
    ...

def get_max_input_output_length(supported_max_seq_len_source: int, supported_max_seq_len_target: int, length_ratio_mean: float, length_ratio_std: float, num_stds: int, forced_max_input_len: Optional[int] = None, forced_max_output_len: Optional[int] = None) -> Tuple[int, Callable[[int], int]]:
    ...

Tokens = List[str]
TokenIds = List[List[int]]
SentenceId = Union[int, str]

@dataclass
class TranslatorInput:
    sentence_id: SentenceId
    tokens: Tokens
    factors: Optional[List[List[str]]]
    source_prefix_tokens: Optional[Tokens]
    source_prefix_factors: Optional[List[List[str]]]
    target_prefix_tokens: Optional[Tokens]
    target_prefix_factors: Optional[List[List[str]]]
    use_target_prefix_all_chunks: bool
    keep_target_prefix_key: bool
    restrict_lexicon: Optional[Any]
    constraints: Optional[List[Tokens]]
    avoid_list: Optional[List[Tokens]]
    pass_through_dict: Optional[Dict[str, Any]]

    ...

@dataclass
class TranslatorOutput:
    sentence_id: SentenceId
    translation: str
    tokens: Tokens
    score: float
    pass_through_dict: Optional[Dict[str, Any]]
    nbest_translations: Optional[List[str]]
    nbest_tokens: Optional[List[Tokens]]
    nbest_scores: Optional[List[float]]
    factor_translations: Optional[List[str]]
    factor_tokens: Optional[List[List[str]]]
    factor_scores: Optional[List[float]]
    nbest_factor_translations: Optional[List[List[str]]]
    nbest_factor_tokens: Optional[List[List[List[str]]]]

    ...

@dataclass
class NBestTranslations:
    target_ids_list: List[List[List[int]]]
    scores: List[float]

    ...

@dataclass
class Translation:
    target_ids: List[List[int]]
    scores: List[float]
    nbest_translations: Optional[NBestTranslations]
    estimated_reference_length: Optional[float]

    ...

@dataclass
class IndexedTranslatorInput:
    input_idx: int
    chunk_idx: int
    translator_input: TranslatorInput

    ...

@dataclass(order=True)
class IndexedTranslation:
    input_idx: int
    chunk_idx: int
    translation: Translation

    ...

class Translator:
    def __init__(self, device: str, ensemble_mode: str, scorer: CandidateScorer, batch_size: int, beam_search_stop: str, models: List[SockeyeModel], source_vocabs: List[vocab.Vocab], target_vocabs: List[vocab.Vocab], beam_size: int = 5, nbest_size: int = 1, restrict_lexicon: Optional[Any] = None, strip_unknown_words: bool = False, sample: Optional[bool] = None, output_scores: bool = False, constant_length_ratio: float = 0.0, knn_lambda: float = 0.0, max_output_length_num_stds: int = 0, max_input_length: Optional[int] = None, max_output_length: Optional[int] = None, prevent_unk: bool = False, greedy: bool = False, skip_nvs: bool = False, nvs_thresh: float = 0.5):
        ...

    def translate(self, trans_inputs: List[TranslatorInput], fill_up_batches: bool = True) -> List[TranslatorOutput]:
        ...

    def _translate_batch(self, translator_inputs: List[TranslatorInput]) -> List[Translation]:
        ...

    def _get_inference_input(self, trans_inputs: List[TranslatorInput]) -> Tuple[pt.Tensor, pt.Tensor, Any, pt.Tensor, Optional[pt.Tensor], Optional[pt.Tensor]]:
        ...

    def _get_translation_tokens_and_factors(self, target_ids: List[List[int]]) -> Tuple[Tokens, str, List[List[str]], List[str]]:
        ...

    def _make_result(self, trans_input: TranslatorInput, translation: Translation) -> TranslatorOutput:
        ...

    def _translate_np(self, source: pt.Tensor, source_length: pt.Tensor, restrict_lexicon: Any, max_output_lengths: pt.Tensor, target_prefix: Optional[pt.Tensor] = None, target_prefix_factors: Optional[pt.Tensor] = None) -> List[Translation]:
        ...

    def _get_best_translations(self, result: SearchResult) -> List[Translation]:
        ...

    @staticmethod
    def _get_best_word_indices_for_kth_hypotheses(ks: np.ndarray, all_hyp_indices: np.ndarray) -> np.ndarray:
        ...

    @staticmethod
    def _assemble_translation(sequence: np.ndarray, length: int, seq_scores: np.ndarray, estimated_reference_length: Optional[float], unshift_target_factors: bool) -> Translation:
        ...

def _unshift_target_factors(sequence: np.ndarray, fill_last_with: int = 2) -> List[List[int]]:
    ...

def _concat_nbest_translations(translations: List[Translation], stop_ids: Set[int], scorer: CandidateScorer) -> List[Translation]:
    ...

def _reduce_nbest_translations(nbest_translations_list: List[Translation]) -> Translation:
    ...

def _expand_nbest_translation(translation: Translation) -> List[Translation]:
    ...

def _remove_target_prefix_tokens(target_ids: List[List[int]], num_target_prefix_tokens: int) -> List[List[int]]:
    ...

def _concat_translations(translations: List[Translation], stop_ids: Set[int], scorer: CandidateScorer) -> Translation:
    ...
