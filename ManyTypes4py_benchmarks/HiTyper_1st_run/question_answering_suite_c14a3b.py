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

def _crossproduct(template: Union[str, dict]) -> Union[str, dict]:
    """
    Takes the output of editor.template and does the cross product of contexts and qas
    """
    ret = []
    ret_labels = []
    for instance in template.data:
        cs = instance['contexts']
        qas = instance['qas']
        d = list(itertools.product(cs, qas))
        ret.append([(x[0], x[1][0]) for x in d])
        ret_labels.append([x[1][1] for x in d])
    template.data = ret
    template.labels = ret_labels
    return template

@TaskSuite.register('question-answering')
class QuestionAnsweringSuite(TaskSuite):

    def __init__(self, suite: Union[None, typing.Type, bytes, str]=None, context_key: typing.Text='context', question_key: typing.Text='question', answer_key: typing.Text='best_span_str', **kwargs) -> None:
        self._context_key = context_key
        self._question_key = question_key
        self._answer_key = answer_key
        super().__init__(suite, **kwargs)

    def _prediction_and_confidence_scores(self, predictor: Union[bool, dict[str, str]]):

        def preds_and_confs_fn(data: Any) -> tuple[list]:
            data = [{self._context_key: pair[0], self._question_key: pair[1]} for pair in data]
            predictions = predictor.predict_batch_json(data)
            labels = [pred[self._answer_key] for pred in predictions]
            return (labels, np.ones(len(labels)))
        return preds_and_confs_fn

    def _format_failing_examples(self, inputs: Union[str, list[list[str]], dict], pred: Union[allennlp.data.vocabulary.Vocabulary, list[str], str], conf: Union[str, list[allennlp.data.instance.Instance], typing.Sequence[typing.Union[str,int]]], label: Union[None, str, list[str], dict]=None, *args, **kwargs) -> typing.Text:
        """
        Formatting function for printing failed test examples.
        """
        context, question = inputs
        ret = 'Context: %s\nQuestion: %s\n' % (context, question)
        if label is not None:
            ret += 'Original answer: %s\n' % label
        ret += 'Predicted answer: %s\n' % pred
        return ret

    @classmethod
    def contractions(cls: Union[typing.Type, bool]):

        def _contractions(x: Any) -> list[tuple]:
            conts = Perturb.contractions(x[1])
            return [(x[0], a) for a in conts]
        return _contractions

    @classmethod
    def typos(cls: Union[str, typing.Callable[typing.Any, T]]):

        def question_typo(x: Any, **kwargs) -> tuple:
            return (x[0], Perturb.add_typos(x[1], **kwargs))
        return question_typo

    @classmethod
    def punctuation(cls: Union[str, typing.Type]):

        def context_punctuation(x: Any) -> tuple:
            return (utils.strip_punctuation(x[0]), x[1])
        return context_punctuation

    def _setup_editor(self) -> None:
        super()._setup_editor()
        adj = ['old', 'smart', 'tall', 'young', 'strong', 'short', 'tough', 'cool', 'fast', 'nice', 'small', 'dark', 'wise', 'rich', 'great', 'weak', 'high', 'slow', 'strange', 'clean']
        adj = [(x.rstrip('e'), x) for x in adj]
        self.editor.add_lexicon('adjectives_to_compare', adj, overwrite=True)
        comp_pairs = [('better', 'worse'), ('older', 'younger'), ('smarter', 'dumber'), ('taller', 'shorter'), ('bigger', 'smaller'), ('stronger', 'weaker'), ('faster', 'slower'), ('darker', 'lighter'), ('richer', 'poorer'), ('happier', 'sadder'), ('louder', 'quieter'), ('warmer', 'colder')]
        self.editor.add_lexicon('comp_pairs', comp_pairs, overwrite=True)

    def _default_tests(self, data: Union[int, list[dict[str, typing.Any]], dict[str, int]], num_test_cases: int=100) -> None:
        super()._default_tests(data, num_test_cases)
        self._setup_editor()
        self._default_vocabulary_tests(data, num_test_cases)
        self._default_taxonomy_tests(data, num_test_cases)

    def _default_vocabulary_tests(self, data: Union[int, tuple[float], list[str]], num_test_cases: int=100) -> None:
        template = self.editor.template([('{first_name} is {adjectives_to_compare[0]}er than {first_name1}.', 'Who is less {adjectives_to_compare[1]}?'), ('{first_name} is {adjectives_to_compare[0]}er than {first_name1}.', 'Who is {adjectives_to_compare[0]}er?')], labels=['{first_name1}', '{first_name}'], remove_duplicates=True, nsamples=num_test_cases, save=True)
        test = MFT(**template, name='A is COMP than B. Who is more / less COMP?', description='Eg. Context: "A is taller than B" Q: "Who is taller?" A: "A", Q: "Who is less tall?" A: "B"', capability='Vocabulary')
        self.add_test(test)

    def _default_taxonomy_tests(self, data: Union[int, tuple[float]], num_test_cases: int=100) -> None:
        template = _crossproduct(self.editor.template({'contexts': ['{first_name} is {comp_pairs[0]} than {first_name1}.', '{first_name1} is {comp_pairs[1]} than {first_name}.'], 'qas': [('Who is {comp_pairs[1]}?', '{first_name1}'), ('Who is {comp_pairs[0]}?', '{first_name}')]}, remove_duplicates=True, nsamples=num_test_cases, save=True))
        test = MFT(**template, name='A is COMP than B. Who is antonym(COMP)? B', description='Eg. Context: "A is taller than B", Q: "Who is shorter?", A: "B"', capability='Taxonomy')
        self.add_test(test)