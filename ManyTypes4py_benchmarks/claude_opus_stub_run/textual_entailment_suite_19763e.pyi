from typing import Optional, Tuple, Iterable, Callable, Union, Any, List
import numpy as np
from checklist.test_suite import TestSuite
from allennlp.confidence_checks.task_checklists.task_suite import TaskSuite
from allennlp.predictors import Predictor


def _wrap_apply_to_each(
    perturb_fn: Callable[..., Any],
    both: bool = False,
    *args: Any,
    **kwargs: Any,
) -> Callable[[Tuple[str, str]], List[Tuple[str, str]]]: ...


class TextualEntailmentSuite(TaskSuite):
    _entails: int
    _contradicts: int
    _neutral: int
    _premise: str
    _hypothesis: str
    _probs_key: str

    def __init__(
        self,
        suite: Optional[TestSuite] = ...,
        entails: int = ...,
        contradicts: int = ...,
        neutral: int = ...,
        premise: str = ...,
        hypothesis: str = ...,
        probs_key: str = ...,
        **kwargs: Any,
    ) -> None: ...

    def _prediction_and_confidence_scores(
        self, predictor: Predictor
    ) -> Callable[[List[Tuple[str, str]]], Tuple[np.ndarray, np.ndarray]]: ...

    def _format_failing_examples(
        self,
        inputs: Tuple[str, str],
        pred: int,
        conf: Any,
        label: Optional[int] = ...,
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

    def _default_tests(self, data: Any, num_test_cases: int = ...) -> None: ...

    def _default_vocabulary_tests(self, data: Any, num_test_cases: int = ...) -> None: ...

    def _default_taxonomy_tests(self, data: Any, num_test_cases: int = ...) -> None: ...

    def _default_coreference_tests(self, data: Any, num_test_cases: int = ...) -> None: ...

    def _default_robustness_tests(self, data: Any, num_test_cases: int = ...) -> None: ...

    def _default_logic_tests(self, data: Any, num_test_cases: int = ...) -> None: ...

    def _default_negation_tests(self, data: Any, num_test_cases: int = ...) -> None: ...

    def _default_ner_tests(self, data: Any, num_test_cases: int = ...) -> None: ...

    def _default_temporal_tests(self, data: Any, num_test_cases: int = ...) -> None: ...

    def _default_fairness_tests(self, data: Any, num_test_cases: int = ...) -> None: ...