from typing import Optional, Tuple, Iterable, Callable, Union, List, Any, Dict, Sequence
import itertools
import numpy as np
from checklist.test_suite import TestSuite
from checklist.test_types import MFT, INV, DIR, Expect
from checklist.perturb import Perturb
from allennlp.confidence_checks.task_checklists.task_suite import TaskSuite
from allennlp.confidence_checks.task_checklists import utils
from allennlp.predictors import Predictor

class _TemplateData:
    def __init__(self, data: Any) -> None: ...
    data: Any
    def __add__(self, other: _TemplateData) -> _TemplateData: ...

@TaskSuite.register('textual-entailment')
class TextualEntailmentSuite(TaskSuite):
    _entails: int
    _contradicts: int
    _neutral: int
    _premise: str
    _hypothesis: str
    _probs_key: str
    editor: Any

    def __init__(
        self,
        suite: Optional[TestSuite] = None,
        entails: int = 0,
        contradicts: int = 1,
        neutral: int = 2,
        premise: str = 'premise',
        hypothesis: str = 'hypothesis',
        probs_key: str = 'probs',
        **kwargs: Any,
    ) -> None: ...

    def _prediction_and_confidence_scores(
        self, predictor: Predictor
    ) -> Callable[[List[Tuple[str, str]]], Tuple[np.ndarray, np.ndarray]]: ...

    def _format_failing_examples(
        self,
        inputs: Tuple[str, str],
        pred: Any,
        conf: np.ndarray,
        label: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> str: ...

    @classmethod
    def contractions(cls) -> Callable[[Tuple[str, str]], List[Tuple[str, str]]]: ...

    @classmethod
    def typos(cls) -> Callable[[Tuple[str, str]], List[Tuple[str, str]]]: ...

    @classmethod
    def punctuation(cls) -> Callable[[Tuple[str, str]], List[Tuple[str, str]]]: ...

    def _setup_editor(self) -> None: ...

    def _default_tests(
        self, data: Optional[List[Tuple[str, str]]], num_test_cases: int = 100
    ) -> None: ...

    def _default_vocabulary_tests(
        self, data: Optional[List[Tuple[str, str]]], num_test_cases: int = 100
    ) -> None: ...

    def _default_taxonomy_tests(
        self, data: Optional[List[Tuple[str, str]]], num_test_cases: int = 100
    ) -> None: ...

    def _default_coreference_tests(
        self, data: Optional[List[Tuple[str, str]]], num_test_cases: int = 100
    ) -> None: ...

    def _default_robustness_tests(
        self, data: Optional[List[Tuple[str, str]]], num_test_cases: int = 100
    ) -> None: ...

    def _default_logic_tests(
        self, data: Optional[List[Tuple[str, str]]], num_test_cases: int = 100
    ) -> None: ...

    def _default_negation_tests(
        self, data: Optional[List[Tuple[str, str]]], num_test_cases: int = 100
    ) -> None: ...

    def _default_ner_tests(
        self, data: Optional[List[Tuple[str, str]]], num_test_cases: int = 100
    ) -> None: ...

    def _default_temporal_tests(
        self, data: Optional[List[Tuple[str, str]]], num_test_cases: int = 100
    ) -> None: ...

    def _default_fairness_tests(
        self, data: Optional[List[Tuple[str, str]]], num_test_cases: int = 100
    ) -> None: ...