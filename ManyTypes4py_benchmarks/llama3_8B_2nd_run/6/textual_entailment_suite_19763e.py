from typing import Optional, Tuple, Iterable, Callable, Union

def _wrap_apply_to_each(perturb_fn: Callable[[str, ...], str], both: bool, *args, **kwargs) -> Callable[[Tuple[str, str], ...], Iterable[Tuple[str, str]]]:
    ...

@TaskSuite.register('textual-entailment')
class TextualEntailmentSuite(TaskSuite):
    def __init__(self, 
                 suite: Optional[TaskSuite] = None, 
                 entails: int = 0, 
                 contradicts: int = 1, 
                 neutral: int = 2, 
                 premise: str = 'premise', 
                 hypothesis: str = 'hypothesis', 
                 probs_key: str = 'probs', 
                 **kwargs: Union[int, str]) -> None:
        ...

    def _prediction_and_confidence_scores(self, 
                                           predictor: Predictor) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def _format_failing_examples(self, 
                                  inputs: Tuple[str, str], 
                                  pred: np.ndarray, 
                                  conf: np.ndarray, 
                                  label: Optional[int] = None, 
                                  *args: str, 
                                  **kwargs: str) -> str:
        ...

    @classmethod
    def contractions(cls) -> Iterable[Tuple[str, str]]:
        ...

    @classmethod
    def typos(cls) -> Iterable[Tuple[str, str]]:
        ...

    @classmethod
    def punctuation(cls) -> Iterable[Tuple[str, str]]:
        ...

    def _setup_editor(self) -> None:
        ...

    def _default_tests(self, 
                       data: Iterable[Tuple[str, str]], 
                       num_test_cases: int = 100) -> None:
        ...

    def _default_vocabulary_tests(self, 
                                  data: Iterable[Tuple[str, str]], 
                                  num_test_cases: int = 100) -> None:
        ...

    def _default_taxonomy_tests(self, 
                                data: Iterable[Tuple[str, str]], 
                                num_test_cases: int = 100) -> None:
        ...

    def _default_coreference_tests(self, 
                                   data: Iterable[Tuple[str, str]], 
                                   num_test_cases: int = 100) -> None:
        ...

    def _default_robustness_tests(self, 
                                  data: Iterable[Tuple[str, str]], 
                                  num_test_cases: int = 100) -> None:
        ...

    def _default_logic_tests(self, 
                             data: Iterable[Tuple[str, str]], 
                             num_test_cases: int = 100) -> None:
        ...

    def _default_negation_tests(self, 
                                data: Iterable[Tuple[str, str]], 
                                num_test_cases: int = 100) -> None:
        ...

    def _default_ner_tests(self, 
                           data: Iterable[Tuple[str, str]], 
                           num_test_cases: int = 100) -> None:
        ...

    def _default_temporal_tests(self, 
                                data: Iterable[Tuple[str, str]], 
                                num_test_cases: int = 100) -> None:
        ...

    def _default_fairness_tests(self, 
                                data: Iterable[Tuple[str, str]], 
                                num_test_cases: int = 100) -> None:
        ...
