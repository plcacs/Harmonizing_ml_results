```python
import sys
from typing import Any, Callable, Dict, List, Optional, TextIO, Tuple, Union, Iterable
import numpy as np
from checklist.test_suite import TestSuite
from checklist.editor import Editor
from checklist.test_types import MFT, INV, DIR
from checklist.perturb import Perturb
from allennlp.common.registrable import Registrable
from allennlp.predictors.predictor import Predictor

class TaskSuite(Registrable):
    _capabilities: List[str] = ...
    
    def __init__(
        self,
        suite: Optional[TestSuite] = ...,
        add_default_tests: bool = ...,
        data: Optional[List[Any]] = ...,
        num_test_cases: int = ...,
        **kwargs: Any
    ) -> None: ...
    
    def _prediction_and_confidence_scores(self, predictor: Predictor) -> Any: ...
    
    def describe(self) -> None: ...
    
    def summary(
        self,
        capabilities: Optional[List[str]] = ...,
        file: TextIO = ...,
        **kwargs: Any
    ) -> None: ...
    
    def _summary(
        self,
        overview_only: bool = ...,
        capabilities: Optional[List[str]] = ...,
        **kwargs: Any
    ) -> None: ...
    
    def _format_failing_examples(
        self,
        inputs: Any,
        pred: Any,
        conf: Any,
        *args: Any,
        **kwargs: Any
    ) -> str: ...
    
    def run(
        self,
        predictor: Predictor,
        capabilities: Optional[List[str]] = ...,
        max_examples: Optional[int] = ...
    ) -> None: ...
    
    @classmethod
    def constructor(
        cls,
        name: Optional[str] = ...,
        suite_file: Optional[str] = ...,
        extra_args: Optional[Dict[str, Any]] = ...
    ) -> 'TaskSuite': ...
    
    def save_suite(self, suite_file: str) -> None: ...
    
    def _default_tests(self, data: Optional[List[Any]], num_test_cases: int = ...) -> None: ...
    
    @classmethod
    def contractions(cls) -> Callable[..., Any]: ...
    
    @classmethod
    def typos(cls) -> Callable[..., Any]: ...
    
    @classmethod
    def punctuation(cls) -> Callable[..., Any]: ...
    
    def _punctuation_test(self, data: List[Any], num_test_cases: int) -> None: ...
    
    def _typo_test(self, data: List[Any], num_test_cases: int) -> None: ...
    
    def _contraction_test(self, data: List[Any], num_test_cases: int) -> None: ...
    
    def _setup_editor(self) -> None: ...
    
    def add_test(self, test: Any) -> None: ...
```