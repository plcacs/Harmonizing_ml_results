from snorkel.preprocess.nlp import SpacyPreprocessor
from snorkel.types import HashingFunction

class SpacyPreprocessorParameters(NamedTuple):
    text_field: str
    doc_field: str
    language: str
    disable: Optional[List[str]]
    pre: List[Callable[[Any], Any]]
    memoize: bool
    memoize_key: Optional[HashingFunction]
    gpu: bool

class SpacyPreprocessorConfig(NamedTuple):
    nlp: SpacyPreprocessor
    parameters: SpacyPreprocessorParameters

class BaseNLPLabelingFunction(LabelingFunction):
    @classmethod
    def _create_preprocessor(cls, parameters: SpacyPreprocessorParameters) -> SpacyPreprocessor:
    @classmethod
    def _create_or_check_preprocessor(cls, text_field: str, doc_field: str, language: str, disable: Optional[List[str]], pre: List[Callable[[Any], Any]], memoize: bool, memoize_key: Optional[HashingFunction], gpu: bool) -> None:

    def __init__(self, name: str, f: Callable[[Any], int], resources: Optional[Mapping[str, Any]] = None, pre: Optional[List[Callable[[Any], Any]]] = None, text_field: str = 'text', doc_field: str = 'doc', language: str = EN_CORE_WEB_SM, disable: Optional[List[str]] = None, memoize: bool = True, memoize_key: Optional[HashingFunction] = None, gpu: bool = False) -> None:

class NLPLabelingFunction(BaseNLPLabelingFunction):
    @classmethod
    def _create_preprocessor(cls, parameters: SpacyPreprocessorParameters) -> SpacyPreprocessor:

class base_nlp_labeling_function(labeling_function):
    def __init__(self, name: Optional[str] = None, resources: Optional[Mapping[str, Any]] = None, pre: Optional[List[Callable[[Any], Any]]] = None, text_field: str = 'text', doc_field: str = 'doc', language: str = EN_CORE_WEB_SM, disable: Optional[List[str]] = None, memoize: bool = True, memoize_key: Optional[HashingFunction] = None, gpu: bool = False) -> None:
    def __call__(self, f: Callable[[Any], int]) -> BaseNLPLabelingFunction:

class nlp_labeling_function(base_nlp_labeling_function):
