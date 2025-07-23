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
        self.suite: TestSuite = suite or TestSuite()
        if add_default_tests:
            self._default_tests(data, num_test_cases)

    def _prediction_and_confidence_scores(self, predictor: Predictor) -> Callable:
        return NotImplementedError

    def describe(self) -> None:
        self._summary(overview_only=True)

    def summary(self, capabilities: Optional[List[str]] = None, file: TextIO = sys.stdout, **kwargs: Any) -> None:
        old_stdout = sys.stdout
        try:
            sys.stdout = file
            self._summary(capabilities=capabilities, **kwargs)
        finally:
            sys.stdout = old_stdout

    def _summary(self, overview_only: bool = False, capabilities: Optional[List[str]] = None, **kwargs: Any) -> None:
        def cap_order(x: str) -> int:
            return self._capabilities.index(x) if x in self._capabilities else 100
        capabilities = capabilities or sorted(set([x['capability'] for x in self.suite.info.values()]), key=cap_order)
        print('\n\nThis suite contains {} tests across {} capabilities.'.format(len(self.suite.tests), len(capabilities)))
        print()
        for capability in capabilities:
            tests = [name for name, test in self.suite.info.items() if test['capability'] == capability]
            num_tests = len(tests)
            if num_tests > 0:
                print(f'\nCapability: "{capability}" ({num_tests} tests)\n')
                for test in tests:
                    description = self.suite.info[test]['description']
                    num_test_cases = len(self.suite.tests[test].data)
                    about_test = f'* Name: {test} ({num_test_cases} test cases)'
                    if description:
                        about_test += f'\n{description}'
                    print(about_test)
                    if not overview_only:
                        if 'format_example_fn' not in kwargs:
                            kwargs['format_example_fn'] = self.suite.info[test].get('format_example_fn', self._format_failing_examples)
                        if 'print_fn' not in kwargs:
                            kwargs['print_fn'] = self.suite.info[test].get('print_fn', self.suite.print_fn)
                        print()
                        self.suite.tests[test].summary(**kwargs)
                        print()

    def _format_failing_examples(self, inputs: Any, pred: Any, conf: np.ndarray, *args: Any, **kwargs: Any) -> str:
        if conf.shape[0] <= 4:
            confs = ' '.join(['%.1f' % c for c in conf])
            ret = '%s %s' % (confs, str(inputs))
        else:
            conf = conf[pred]
            ret = '%s (%.1f) %s' % (pred, conf, str(inputs))
        return ret

    def run(self, predictor: Predictor, capabilities: Optional[List[str]] = None, max_examples: Optional[int] = None) -> None:
        preds_and_confs_fn = self._prediction_and_confidence_scores(predictor)
        if preds_and_confs_fn is NotImplementedError:
            raise NotImplementedError('The `_prediction_and_confidence_scores` function needs to be implemented for the class `{}`'.format(self.__class__))
        if not capabilities:
            self.suite.run(preds_and_confs_fn, overwrite=True, n=max_examples)
        else:
            for _, test in self.suite.tests.items():
                if test.capability in capabilities:
                    test.run(preds_and_confs_fn, verbose=True, overwrite=True, n=max_examples)

    @classmethod
    def constructor(cls: Type['TaskSuite'], name: Optional[str] = None, suite_file: Optional[str] = None, extra_args: Optional[Dict[str, Any]] = None) -> 'TaskSuite':
        suite_class = TaskSuite.by_name(name) if name is not None else cls
        if extra_args is None:
            extra_args = {}
        if suite_file is not None:
            return suite_class(TestSuite.from_file(cached_path(suite_file)), **extra_args)
        return suite_class(**extra_args)

    def save_suite(self, suite_file: str) -> None:
        self.suite.save(suite_file)

    def _default_tests(self, data: Optional[List[Any]], num_test_cases: int = 100) -> None:
        if data:
            self._punctuation_test(data, num_test_cases)
            self._typo_test(data, num_test_cases)
            self._contraction_test(data, num_test_cases)

    @classmethod
    def contractions(cls) -> Callable:
        return Perturb.contractions

    @classmethod
    def typos(cls) -> Callable:
        return Perturb.add_typos

    @classmethod
    def punctuation(cls) -> Callable:
        return utils.toggle_punctuation

    def _punctuation_test(self, data: List[Any], num_test_cases: int) -> None:
        template = Perturb.perturb(data, self.punctuation(), nsamples=num_test_cases)
        test = INV(template.data, name='Punctuation', description="Strip punctuation and / or add '.'", capability='Robustness')
        self.add_test(test)

    def _typo_test(self, data: List[Any], num_test_cases: int) -> None:
        template = Perturb.perturb(data, self.typos(), nsamples=num_test_cases, typos=1)
        test = INV(template.data, name='Typos', capability='Robustness', description='Add one typo to input by swapping two adjacent characters')
        self.add_test(test)
        template = Perturb.perturb(data, self.typos(), nsamples=num_test_cases, typos=2)
        test = INV(template.data, name='2 Typos', capability='Robustness', description='Add two typos to input by swapping two adjacent characters twice')
        self.add_test(test)

    def _contraction_test(self, data: List[Any], num_test_cases: int) -> None:
        template = Perturb.perturb(data, self.contractions(), nsamples=num_test_cases)
        test = INV(template.data, name='Contractions', capability='Robustness', description="Contract or expand contractions, e.g. What is <-> What's")
        self.add_test(test)

    def _setup_editor(self) -> None:
        if not hasattr(self, 'editor'):
            self.editor = Editor()
            utils.add_common_lexicons(self.editor)

    def add_test(self, test: Union[MFT, INV, DIR]) -> None:
        if test.data:
            self.suite.add(test)
        else:
            logger.warning("'{}' was not added, as it contains no examples.".format(test.name))
