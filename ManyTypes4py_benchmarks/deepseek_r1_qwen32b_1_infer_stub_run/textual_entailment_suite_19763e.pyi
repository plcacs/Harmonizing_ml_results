from typing import List, Tuple, Callable, Optional, Union
import numpy as np
from allennlp.confidence_checks.task_checklists.task_suite import TaskSuite

def _wrap_apply_to_each(perturb_fn: Callable, both: bool, *args, **kwargs) -> Callable[[Tuple[str, str]], List[Tuple[str, str]]]:
    ...

@TaskSuite.register('textual-entailment')
class TextualEntailmentSuite(TaskSuite):
    def __init__(self, suite: Optional[TestSuite] = None, entails: int = 0, contradicts: int = 1, neutral: int = 2, premise: str = 'premise', hypothesis: str = 'hypothesis', probs_key: str = 'probs', **kwargs) -> None:
        ...

    def _prediction_and_confidence_scores(self, predictor: Predictor) -> Callable[[List[Tuple[str, str]]], Tuple[np.ndarray, np.ndarray]]:
        ...

    def _format_failing_examples(self, inputs: Tuple[str, str], pred: int, conf: np.ndarray, label: Optional[int] = None, *args, **kwargs) -> str:
        ...

    @classmethod
    def contractions(cls) -> Callable[[Tuple[str, str]], List[Tuple[str, str]]]:
        ...

    @classmethod
    def typos(cls) -> Callable[[Tuple[str, str]], List[Tuple[str, str]]]:
        ...

    @classmethod
    def punctuation(cls) -> Callable[[Tuple[str, str]], List[Tuple[str, str]]]:
        ...

    def _setup_editor(self) -> None:
        ...

    def _default_tests(self, data: List[Tuple[str, str]], num_test_cases: int = 100) -> None:
        ...

    def _default_vocabulary_tests(self, data: List[Tuple[str, str]], num_test_cases: int = 100) -> None:
        ...

    def _default_taxonomy_tests(self, data: List[Tuple[str, str]], num_test_cases: int = 100) -> None:
        ...

    def _default_coreference_tests(self, data: List[Tuple[str, str]], num_test_cases: int = 100) -> None:
        ...

    def _default_robustness_tests(self, data: List[Tuple[str, str]], num_test_cases: int = 100) -> None:
        ...

    def _default_logic_tests(self, data: List[Tuple[str, str]], num_test_cases: int = 100) -> None:
        ...

    def _default_negation_tests(self, data: List[Tuple[str, str]], num_test_cases: int = 100) -> None:
        ...

    def _default_ner_tests(self, data: List[Tuple[str, str]], num_test_cases: int = 100) -> None:
        ...

    def _default_temporal_tests(self, data: List[Tuple[str, str]], num_test_cases: int = 100) -> None:
        ...

    def _default_fairness_tests(self, data: List[Tuple[str, str]], num_test_cases: int = 100) -> None:
        ...