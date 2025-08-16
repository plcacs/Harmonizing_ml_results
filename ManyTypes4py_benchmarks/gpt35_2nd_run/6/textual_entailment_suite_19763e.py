from typing import Optional, Tuple, Iterable, Callable, Union, List
import itertools
import numpy as np
from checklist.test_suite import TestSuite
from checklist.test_types import MFT, INV, DIR, Expect
from checklist.perturb import Perturb
from allennlp.confidence_checks.task_checklists.task_suite import TaskSuite
from allennlp.confidence_checks.task_checklists import utils
from allennlp.predictors import Predictor

def _wrap_apply_to_each(perturb_fn: Callable, both: bool = False, *args, **kwargs) -> Callable:
    def new_fn(pair: Tuple[str, str], *args, **kwargs) -> List[Tuple[str, str]]:
        premise, hypothesis = (pair[0], pair[1])
        ret = []
        fn_premise = perturb_fn(premise, *args, **kwargs)
        fn_hypothesis = perturb_fn(hypothesis, *args, **kwargs)
        if type(fn_premise) != list:
            fn_premise = [fn_premise]
        if type(fn_hypothesis) != list:
            fn_hypothesis = [fn_hypothesis]
        ret.extend([(x, str(hypothesis)) for x in fn_premise])
        ret.extend([(str(premise), x) for x in fn_hypothesis])
        if both:
            ret.extend([(x, x2) for x, x2 in itertools.product(fn_premise, fn_hypothesis)])
        return [x for x in ret if x[0] and x[1]]
    return new_fn

@TaskSuite.register('textual-entailment')
class TextualEntailmentSuite(TaskSuite):

    def __init__(self, suite: Optional[TestSuite] = None, entails: int = 0, contradicts: int = 1, neutral: int = 2, premise: str = 'premise', hypothesis: str = 'hypothesis', probs_key: str = 'probs', **kwargs):
        self._entails = entails
        self._contradicts = contradicts
        self._neutral = neutral
        self._premise = premise
        self._hypothesis = hypothesis
        self._probs_key = probs_key
        super().__init__(suite, **kwargs)

    def _prediction_and_confidence_scores(self, predictor: Predictor) -> Callable:
        def preds_and_confs_fn(data: List[Tuple[str, str]]) -> Tuple[np.ndarray, np.ndarray]:
            labels = []
            confs = []
            data = [{self._premise: pair[0], self._hypothesis: pair[1]} for pair in data]
            predictions = predictor.predict_batch_json(data)
            for pred in predictions:
                label = np.argmax(pred[self._probs_key])
                labels.append(label)
                confs.append(pred[self._probs_key])
            return (np.array(labels), np.array(confs))
        return preds_and_confs_fn

    def _format_failing_examples(self, inputs: Tuple[str, str], pred: np.ndarray, conf: np.ndarray, label: Optional[int] = None, *args, **kwargs) -> str:
        labels = {self._entails: 'Entails', self._contradicts: 'Contradicts', self._neutral: 'Neutral'}
        ret = 'Premise: %s\nHypothesis: %s' % (inputs[0], inputs[1])
        if label is not None:
            ret += '\nOriginal: %s' % labels[label]
        ret += '\nPrediction: Entails (%.1f), Contradicts (%.1f), Neutral (%.1f)' % (conf[self._entails], conf[self._contradicts], conf[self._neutral])
        return ret

    @classmethod
    def contractions(cls) -> Callable:
        return _wrap_apply_to_each(Perturb.contractions, both=True)

    @classmethod
    def typos(cls) -> Callable:
        return _wrap_apply_to_each(Perturb.add_typos, both=False)

    @classmethod
    def punctuation(cls) -> Callable:
        return _wrap_apply_to_each(utils.toggle_punctuation, both=False)
