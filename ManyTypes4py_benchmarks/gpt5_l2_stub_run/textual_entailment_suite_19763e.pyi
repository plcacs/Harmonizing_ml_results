from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np
from checklist.test_suite import TestSuite
from allennlp.confidence_checks.task_checklists.task_suite import TaskSuite
from allennlp.predictors import Predictor

def _wrap_apply_to_each(
    perturb_fn: Callable[..., Union[str, Iterable[str]]],
    both: bool = False,
    *args: Any,
    **kwargs: Any
) -> Callable[..., List[Tuple[str, str]]]: ...

@TaskSuite.register('textual-entailment')
class TextualEntailmentSuite(TaskSuite):
    def __init__(
        self,
        suite: Optional[TestSuite] = None,
        entails: int = 0,
        contradicts: int = 1,
        neutral: int = 2,
        premise: str = 'premise',
        hypothesis: str = 'hypothesis',
        probs_key: str = 'probs',
        **kwargs: Any
    ) -> None: ...
    def _prediction_and_confidence_scores(
        self, predictor: Predictor
    ) -> Callable[[Iterable[Tuple[str, str]]], Tuple[np.ndarray, np.ndarray]]: ...
    def _format_failing_examples(
        self,
        inputs: Tuple[str, str],
        pred: Any,
        conf: Sequence[float],
        label: Optional[int] = ...,
        *args: Any,
        **kwargs: Any
    ) -> str: ...
    @classmethod
    def contractions(cls) -> Callable[..., List[Tuple[str, str]]]: ...
    @classmethod
    def typos(cls) -> Callable[..., List[Tuple[str, str]]]: ...
    @classmethod
    def punctuation(cls) -> Callable[..., List[Tuple[str, str]]]: ...
    def _setup_editor(self) -> None: ...
    def _default_tests(
        self, data: Optional[Iterable[Tuple[str, str]]], num_test_cases: int = 100
    ) -> None: ...
    def _default_vocabulary_tests(
        self, data: Optional[Iterable[Tuple[str, str]]], num_test_cases: int = 100
    ) -> None: ...
    def _default_taxonomy_tests(
        self, data: Optional[Iterable[Tuple[str, str]]], num_test_cases: int = 100
    ) -> None: ...
    def _default_coreference_tests(
        self, data: Optional[Iterable[Tuple[str, str]]], num_test_cases: int = 100
    ) -> None: ...
    def _default_robustness_tests(
        self, data: Optional[Iterable[Tuple[str, str]]], num_test_cases: int = 100
    ) -> None: ...
    def _default_logic_tests(
        self, data: Optional[Iterable[Tuple[str, str]]], num_test_cases: int = 100
    ) -> None: ...
    def _default_negation_tests(
        self, data: Optional[Iterable[Tuple[str, str]]], num_test_cases: int = 100
    ) -> None: ...
    def _default_ner_tests(
        self, data: Optional[Iterable[Tuple[str, str]]], num_test_cases: int = 100
    ) -> None: ...
    def _default_temporal_tests(
        self, data: Optional[Iterable[Tuple[str, str]]], num_test_cases: int = 100
    ) -> None: ...
    def _default_fairness_tests(
        self, data: Optional[Iterable[Tuple[str, str]]], num_test_cases: int = 100
    ) -> None: ...