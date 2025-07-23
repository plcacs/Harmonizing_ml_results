from typing import Any, Callable, List, Mapping, NamedTuple, Optional, Type
from snorkel.preprocess import BasePreprocessor
from snorkel.preprocess.nlp import EN_CORE_WEB_SM, SpacyPreprocessor
from snorkel.types import HashingFunction
from .core import LabelingFunction, labeling_function

class SpacyPreprocessorParameters(NamedTuple):
    """Parameters needed to construct a SpacyPreprocessor.

    See ``snorkel.preprocess.nlp.SpacyPreprocessor``.
    """
    text_field: str
    doc_field: str
    language: str
    disable: Optional[List[str]]
    pre: List[BasePreprocessor]
    memoize: bool
    memoize_key: Optional[HashingFunction]
    gpu: bool

class SpacyPreprocessorConfig(NamedTuple):
    """Tuple of SpacyPreprocessor and the parameters used to construct it."""
    nlp: SpacyPreprocessor
    parameters: SpacyPreprocessorParameters

class BaseNLPLabelingFunction(LabelingFunction):
    """Base class for spaCy-based LFs."""

    @classmethod
    def _create_preprocessor(cls, parameters: SpacyPreprocessorParameters) -> SpacyPreprocessor:
        raise NotImplementedError

    @classmethod
    def _create_or_check_preprocessor(
        cls,
        text_field: str,
        doc_field: str,
        language: str,
        disable: Optional[List[str]],
        pre: List[BasePreprocessor],
        memoize: bool,
        memoize_key: Optional[HashingFunction],
        gpu: bool
    ) -> None:
        parameters = SpacyPreprocessorParameters(
            text_field=text_field,
            doc_field=doc_field,
            language=language,
            disable=disable,
            pre=pre,
            memoize=memoize,
            memoize_key=memoize_key,
            gpu=gpu
        )
        if not hasattr(cls, '_nlp_config'):
            nlp = cls._create_preprocessor(parameters)
            cls._nlp_config = SpacyPreprocessorConfig(nlp=nlp, parameters=parameters)
        elif parameters != cls._nlp_config.parameters:
            raise ValueError(f'{cls.__name__} already configured with different parameters: {cls._nlp_config.parameters}')

    def __init__(
        self,
        name: str,
        f: Callable[..., int],
        resources: Optional[Mapping[str, Any]] = None,
        pre: Optional[List[BasePreprocessor]] = None,
        text_field: str = 'text',
        doc_field: str = 'doc',
        language: str = EN_CORE_WEB_SM,
        disable: Optional[List[str]] = None,
        memoize: bool = True,
        memoize_key: Optional[HashingFunction] = None,
        gpu: bool = False
    ) -> None:
        self._create_or_check_preprocessor(
            text_field, doc_field, language, disable, pre or [], memoize, memoize_key, gpu
        )
        super().__init__(name, f, resources=resources, pre=[self._nlp_config.nlp])

class NLPLabelingFunction(BaseNLPLabelingFunction):
    """Special labeling function type for spaCy-based LFs."""

    @classmethod
    def _create_preprocessor(cls, parameters: SpacyPreprocessorParameters) -> SpacyPreprocessor:
        return SpacyPreprocessor(**parameters._asdict())

class base_nlp_labeling_function(labeling_function):
    """Decorator to define a BaseNLPLabelingFunction child object from a function."""
    _lf_cls: Optional[Type[BaseNLPLabelingFunction]] = None

    def __init__(
        self,
        name: Optional[str] = None,
        resources: Optional[Mapping[str, Any]] = None,
        pre: Optional[List[BasePreprocessor]] = None,
        text_field: str = 'text',
        doc_field: str = 'doc',
        language: str = EN_CORE_WEB_SM,
        disable: Optional[List[str]] = None,
        memoize: bool = True,
        memoize_key: Optional[HashingFunction] = None,
        gpu: bool = False
    ) -> None:
        super().__init__(name, resources, pre)
        self.text_field = text_field
        self.doc_field = doc_field
        self.language = language
        self.disable = disable
        self.memoize = memoize
        self.memoize_key = memoize_key
        self.gpu = gpu

    def __call__(self, f: Callable[..., int]) -> BaseNLPLabelingFunction:
        """Wrap a function to create an ``BaseNLPLabelingFunction``."""
        if self._lf_cls is None:
            raise NotImplementedError('_lf_cls must be defined')
        name = self.name or f.__name__
        return self._lf_cls(
            name=name,
            f=f,
            resources=self.resources,
            pre=self.pre,
            text_field=self.text_field,
            doc_field=self.doc_field,
            language=self.language,
            disable=self.disable,
            memoize=self.memoize,
            memoize_key=self.memoize_key,
            gpu=self.gpu
        )

class nlp_labeling_function(base_nlp_labeling_function):
    """Decorator to define an NLPLabelingFunction object from a function."""
    _lf_cls: Type[NLPLabelingFunction] = NLPLabelingFunction
