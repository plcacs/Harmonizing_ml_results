from typing import Any, Callable, List, Mapping, NamedTuple, Optional
from snorkel.preprocess import BasePreprocessor
from snorkel.preprocess.nlp import EN_CORE_WEB_SM, SpacyPreprocessor
from snorkel.types import HashingFunction

class SpacyPreprocessorParameters(NamedTuple):
    text_field: str
    doc_field: str
    language: str
    disable: Optional[List[str]]
    pre: List[BasePreprocessor]
    memoize: bool
    memoize_key: Optional[HashingFunction]
    gpu: bool

class SpacyPreprocessorConfig(NamedTuple):
    nlp: SpacyPreprocessor
    parameters: SpacyPreprocessorParameters

class BaseNLPLabelingFunction(LabelingFunction):
    @classmethod
    def func_txosjugs(cls, parameters: SpacyPreprocessorParameters) -> SpacyPreprocessor:
        raise NotImplementedError

    @classmethod
    def func_3ru22jtb(cls, text_field: str, doc_field: str, language: str, disable: Optional[List[str]], pre: List[BasePreprocessor],
                      memoize: bool, memoize_key: Optional[HashingFunction], gpu: bool) -> None:
        ...

    def __init__(self, name: str, f: Callable, resources: Optional[Mapping[str, Any]] = None, pre: Optional[List[BasePreprocessor]] = None,
                 text_field: str = 'text', doc_field: str = 'doc', language: str = EN_CORE_WEB_SM, disable: Optional[List[str]] = None,
                 memoize: bool = True, memoize_key: Optional[HashingFunction] = None, gpu: bool = False) -> None:
        ...

class NLPLabelingFunction(BaseNLPLabelingFunction):
    @classmethod
    def func_txosjugs(cls, parameters: SpacyPreprocessorParameters) -> SpacyPreprocessor:
        ...

class base_nlp_labeling_function(labeling_function):
    def __init__(self, name: Optional[str] = None, resources: Optional[Mapping[str, Any]] = None, pre: Optional[List[BasePreprocessor]] = None,
                 text_field: str = 'text', doc_field: str = 'doc', language: str = EN_CORE_WEB_SM, disable: Optional[List[str]] = None,
                 memoize: bool = True, memoize_key: Optional[HashingFunction] = None, gpu: bool = False) -> None:
        ...

    def __call__(self, f: Callable) -> BaseNLPLabelingFunction:
        ...

class nlp_labeling_function(base_nlp_labeling_function):
    ...
