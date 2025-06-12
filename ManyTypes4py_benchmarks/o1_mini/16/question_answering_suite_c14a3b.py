from typing import Optional, Iterable, Tuple, Union, List, Callable, Any
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
    ret: List[List[Tuple[str, str]]] = []
    ret_labels: List[List[str]] = []
    for instance in template.data:
        cs: List[str] = instance['contexts']
        qas: List[Tuple[str, str]] = instance['qas']
        d = list(itertools.product(cs, qas))
        ret.append([(x[0], x[1][0]) for x in d])
        ret_labels.append([x[1][1] for x in d])
    template.data = ret
    template.labels = ret_labels
    return template


@TaskSuite.register('question-answering')
class QuestionAnsweringSuite(TaskSuite):

    def __init__(
        self,
        suite: Optional[Any] = None,
        context_key: str = 'context',
        question_key: str = 'question',
        answer_key: str = 'best_span_str',
        **kwargs: Any
    ) -> None:
        self._context_key = context_key
        self._question_key = question_key
        self._answer_key = answer_key
        super().__init__(suite, **kwargs)

    def _prediction_and_confidence_scores(
        self, predictor: Predictor
    ) -> Callable[[List[Tuple[str, str]]], Tuple[List[str], np.ndarray]]:

        def preds_and_confs_fn(data: List[Tuple[str, str]]) -> Tuple[List[str], np.ndarray]:
            formatted_data = [{self._context_key: pair[0], self._question_key: pair[1]} for pair in data]
            predictions = predictor.predict_batch_json(formatted_data)
            labels = [pred[self._answer_key] for pred in predictions]
            return (labels, np.ones(len(labels)))
        return preds_and_confs_fn

    def _format_failing_examples(
        self,
        inputs: Tuple[str, str],
        pred: str,
        conf: float,
        label: Optional[str] = None,
        *args: Any,
        **kwargs: Any
    ) -> str:
        """
        Formatting function for printing failed test examples.
        """
        context, question = inputs
        ret = f'Context: {context}\nQuestion: {question}\n'
        if label is not None:
            ret += f'Original answer: {label}\n'
        ret += f'Predicted answer: {pred}\n'
        return ret

    @classmethod
    def contractions(cls) -> Callable[[Tuple[str, str]], List[Tuple[str, str]]]:

        def _contractions(x: Tuple[str, str]) -> List[Tuple[str, str]]:
            conts = Perturb.contractions(x[1])
            return [(x[0], a) for a in conts]
        return _contractions

    @classmethod
    def typos(cls) -> Callable[[Tuple[str, str], Any], Tuple[str, str]]:

        def question_typo(x: Tuple[str, str], **kwargs: Any) -> Tuple[str, str]:
            return (x[0], Perturb.add_typos(x[1], **kwargs))
        return question_typo

    @classmethod
    def punctuation(cls) -> Callable[[Tuple[str, str]], Tuple[str, str]]:

        def context_punctuation(x: Tuple[str, str]) -> Tuple[str, str]:
            return (utils.strip_punctuation(x[0]), x[1])
        return context_punctuation

    def _setup_editor(self) -> None:
        super()._setup_editor()
        adj: List[str] = [
            'old', 'smart', 'tall', 'young', 'strong', 'short', 'tough',
            'cool', 'fast', 'nice', 'small', 'dark', 'wise', 'rich',
            'great', 'weak', 'high', 'slow', 'strange', 'clean'
        ]
        adj = [(x.rstrip('e'), x) for x in adj]
        self.editor.add_lexicon('adjectives_to_compare', adj, overwrite=True)
        comp_pairs: List[Tuple[str, str]] = [
            ('better', 'worse'), ('older', 'younger'), ('smarter', 'dumber'),
            ('taller', 'shorter'), ('bigger', 'smaller'), ('stronger', 'weaker'),
            ('faster', 'slower'), ('darker', 'lighter'), ('richer', 'poorer'),
            ('happier', 'sadder'), ('louder', 'quieter'), ('warmer', 'colder')
        ]
        self.editor.add_lexicon('comp_pairs', comp_pairs, overwrite=True)

    def _default_tests(self, data: Any, num_test_cases: int = 100) -> None:
        super()._default_tests(data, num_test_cases)
        self._setup_editor()
        self._default_vocabulary_tests(data, num_test_cases)
        self._default_taxonomy_tests(data, num_test_cases)

    def _default_vocabulary_tests(self, data: Any, num_test_cases: int = 100) -> None:
        template = self.editor.template(
            [
                (
                    '{first_name} is {adjectives_to_compare[0]}er than {first_name1}.',
                    'Who is less {adjectives_to_compare[1]}?'
                ),
                (
                    '{first_name} is {adjectives_to_compare[0]}er than {first_name1}.',
                    'Who is {adjectives_to_compare[0]}er?'
                )
            ],
            labels=['{first_name1}', '{first_name}'],
            remove_duplicates=True,
            nsamples=num_test_cases,
            save=True
        )
        test = MFT(
            **template,
            name='A is COMP than B. Who is more / less COMP?',
            description='Eg. Context: "A is taller than B" Q: "Who is taller?" A: "A", Q: "Who is less tall?" A: "B"',
            capability='Vocabulary'
        )
        self.add_test(test)

    def _default_taxonomy_tests(self, data: Any, num_test_cases: int = 100) -> None:
        template = _crossproduct(
            self.editor.template(
                {
                    'contexts': [
                        '{first_name} is {comp_pairs[0]} than {first_name1}.',
                        '{first_name1} is {comp_pairs[1]} than {first_name}.'
                    ],
                    'qas': [
                        ('Who is {comp_pairs[1]}?', '{first_name1}'),
                        ('Who is {comp_pairs[0]}?', '{first_name}')
                    ]
                },
                remove_duplicates=True,
                nsamples=num_test_cases,
                save=True
            )
        )
        test = MFT(
            **template,
            name='A is COMP than B. Who is antonym(COMP)? B',
            description='Eg. Context: "A is taller than B", Q: "Who is shorter?", A: "B"',
            capability='Taxonomy'
        )
        self.add_test(test)
