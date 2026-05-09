from typing import Optional, Iterable, List, Union, Tuple, Any
import numpy as np
from checklist.test_suite import TestSuite
from checklist.test_types import MFT, INV, DIR, Expect
from checklist.perturb import Perturb
from allennlp.confidence_checks.task_checklists.task_suite import TaskSuite
from allennlp.data.instance import Instance
from allennlp.predictors import Predictor

def _add_phrase_function(phrases: List[str], num_samples: int = 10) -> callable[[str], List[str]]:
    ...

@TaskSuite.register('sentiment-analysis')
class SentimentAnalysisSuite(TaskSuite):
    def __init__(self, suite: Optional[TestSuite] = None, positive: int = 0, negative: int = 1, **kwargs: Any) -> None:
        ...

    def _prediction_and_confidence_scores(self, predictor: Predictor) -> callable[[Iterable], Tuple[np.ndarray, np.ndarray]]:
        ...

    def _format_failing_examples(self, inputs: str, pred: np.ndarray, conf: np.ndarray, label: Optional[int] = None, *args: Any, **kwargs: Any) -> str:
        ...

    def _default_tests(self, data: Iterable, num_test_cases: int = 100) -> None:
        ...

    def _setup_editor(self) -> None:
        ...

    def _default_vocabulary_tests(self, data: Iterable, num_test_cases: int = 100) -> None:
        ...

    def _default_robustness_tests(self, data: Iterable, num_test_cases: int = 100) -> None:
        ...

    def _default_ner_tests(self, data: Iterable, num_test_cases: int = 100) -> None:
        ...

    def _default_temporal_tests(self, data: Iterable, num_test_cases: int = 100) -> None:
        ...

    def _default_fairness_tests(self, data: Iterable, num_test_cases: int = 100) -> None:
        ...

    def _default_negation_tests(self, data: Iterable, num_test_cases: int = 100) -> None:
        ...

    def _positive_change(self, orig_conf: np.ndarray, conf: np.ndarray) -> float:
        ...

    def _diff_up(self, orig_pred: Any, pred: Any, orig_conf: np.ndarray, conf: np.ndarray, labels: Optional[Any] = None, meta: Optional[Any] = None) -> Union[bool, float]:
        ...

    def _diff_down(self, orig_pred: Any, pred: Any, orig_conf: np.ndarray, conf: np.ndarray, labels: Optional[Any] = None, meta: Optional[Any] = None) -> Union[bool, float]:
        ...