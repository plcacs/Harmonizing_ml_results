import sys
import logging
from typing import Type, Optional, Dict, Any, Callable, List, Iterable, Union, TextIO, Tuple
import numpy as np
from checklist.test_suite import TestSuite
from checklist.editor import Editor
from checklist.test_types import MFT, INV, DIR
from checklist.perturb import Perturb
from allennlp.common.registrable import Registrable
from allennlp.common.file_utils import cached_path
from allennlp.predictors.predictor import Predictor
from allennlp.confidence_checks.task_checklists import utils
logger = logging.getLogger(__name__)

class TaskSuite(Registrable):
    _capabilities: List[str] = ['Vocabulary', 'Taxonomy', 'Robustness', 'NER', 'Fairness', 'Temporal', 'Negation', 'Coref', 'SRL', 'Logic']

    def __init__(self, suite: Optional[TestSuite] = None, add_default_tests: bool = True, data: Optional[List[Any]] = None, num_test_cases: int = 100, **kwargs: Any) -> None:
    
    def _prediction_and_confidence_scores(self, predictor: Predictor) -> Callable:
    
    def describe(self) -> None:
    
    def summary(self, capabilities: Optional[List[str]] = None, file: TextIO = sys.stdout, **kwargs: Any) -> None:
    
    def _summary(self, overview_only: bool = False, capabilities: Optional[List[str]] = None, **kwargs: Any) -> None:
    
    def _format_failing_examples(self, inputs: Any, pred: Any, conf: np.ndarray, *args: Any, **kwargs: Any) -> str:
    
    def run(self, predictor: Predictor, capabilities: Optional[List[str]] = None, max_examples: Optional[int] = None) -> None:
    
    @classmethod
    def constructor(cls, name: Optional[str] = None, suite_file: Optional[str] = None, extra_args: Optional[Dict[str, Any]] = None) -> 'TaskSuite':
    
    def save_suite(self, suite_file: str) -> None:
    
    def _default_tests(self, data: List[Any], num_test_cases: int = 100) -> None:
    
    @classmethod
    def contractions(cls) -> Callable:
    
    @classmethod
    def typos(cls) -> Callable:
    
    @classmethod
    def punctuation(cls) -> Callable:
    
    def _punctuation_test(self, data: List[Any], num_test_cases: int) -> None:
    
    def _typo_test(self, data: List[Any], num_test_cases: int) -> None:
    
    def _contraction_test(self, data: List[Any], num_test_cases: int) -> None:
    
    def _setup_editor(self) -> None:
    
    def add_test(self, test: Union[MFT, INV, DIR]) -> None:
