from typing import Optional, Iterable, Tuple, Union
import itertools
import numpy as np
from checklist.editor import MunchWithAdd as CheckListTemplate
from checklist.test_suite import TestSuite
from checklist.test_types import MFT
from checklist.perturb import Perturb
from allennlp.confidence_checks.task_checklists.task_suite import TaskSuite
from allennlp.confidence_checks.task_checklists import utils
from allennlp.predictors import Predictor

def _crossproduct(template: CheckListTemplate) -> CheckListTemplate:
    """
    Takes the output of editor.template and does the cross product of contexts and qas
    """
    ret: list = []
    ret_labels: list = []
    for instance in template.data:
        cs: Iterable[str] = instance['contexts']
        qas: Iterable[Tuple[str, str]] = instance['qas']
        d = list(itertools.product(cs, qas))
        ret.append([(x[0], x[1][0]) for x in d])
        ret_labels.append([x[1][1] for x in d])
    template.data = ret
    template.labels = ret_labels
    return template

@TaskSuite.register('question-answering')
class QuestionAnsweringSuite(TaskSuite):
    def __init__(self, suite: Optional[TestSuite] = None, context_key: str = 'context', question_key: str = 'question', answer_key: str = 'best_span_str', **kwargs: Union[str, int]) -> None:
        self._context_key: str = context_key
        self._question_key: str = question_key
        self._answer_key: str = answer_key
        super().__init__(suite, **kwargs)

    # ... rest of the code ...
