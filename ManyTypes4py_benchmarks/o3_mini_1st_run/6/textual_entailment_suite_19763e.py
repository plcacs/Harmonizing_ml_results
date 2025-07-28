from typing import Optional, Tuple, Iterable, Callable, Union, Any, List
import itertools
import numpy as np
from checklist.test_suite import TestSuite
from checklist.test_types import MFT, INV, DIR, Expect
from checklist.perturb import Perturb
from allennlp.confidence_checks.task_checklists.task_suite import TaskSuite
from allennlp.confidence_checks.task_checklists import utils
from allennlp.predictors import Predictor

def _wrap_apply_to_each(
    perturb_fn: Callable[[str, Any], Union[str, List[str]]],
    both: bool = False,
    *args: Any,
    **kwargs: Any
) -> Callable[[Tuple[str, str], Any], List[Tuple[str, str]]]:
    """
    Wraps the perturb function so that it is applied to
    both elements in the (premise, hypothesis) tuple.
    """
    def new_fn(pair: Tuple[str, str], *args: Any, **kwargs: Any) -> List[Tuple[str, str]]:
        premise: str = pair[0]
        hypothesis: str = pair[1]
        ret: List[Tuple[str, str]] = []
        fn_premise: Union[str, List[str]] = perturb_fn(premise, *args, **kwargs)
        fn_hypothesis: Union[str, List[str]] = perturb_fn(hypothesis, *args, **kwargs)
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

    def __init__(
        self,
        suite: Optional[Any] = None,
        entails: int = 0,
        contradicts: int = 1,
        neutral: int = 2,
        premise: str = 'premise',
        hypothesis: str = 'hypothesis',
        probs_key: str = 'probs',
        **kwargs: Any
    ) -> None:
        self._entails: int = entails
        self._contradicts: int = contradicts
        self._neutral: int = neutral
        self._premise: str = premise
        self._hypothesis: str = hypothesis
        self._probs_key: str = probs_key
        super().__init__(suite, **kwargs)

    def _prediction_and_confidence_scores(self, predictor: Predictor) -> Callable[[Iterable[Tuple[str, str]]], Tuple[np.ndarray, np.ndarray]]:
        def preds_and_confs_fn(data: Iterable[Tuple[str, str]]) -> Tuple[np.ndarray, np.ndarray]:
            labels: List[int] = []
            confs: List[Any] = []
            data_dicts = [{self._premise: pair[0], self._hypothesis: pair[1]} for pair in data]
            predictions: List[Any] = predictor.predict_batch_json(data_dicts)
            for pred in predictions:
                label: int = int(np.argmax(pred[self._probs_key]))
                labels.append(label)
                confs.append(pred[self._probs_key])
            return (np.array(labels), np.array(confs))
        return preds_and_confs_fn

    def _format_failing_examples(
        self,
        inputs: Tuple[str, str],
        pred: Any,
        conf: Union[Iterable[float], np.ndarray],
        label: Optional[int] = None,
        *args: Any,
        **kwargs: Any
    ) -> str:
        """
        Formatting function for printing failed test examples.
        """
        labels = {self._entails: 'Entails', self._contradicts: 'Contradicts', self._neutral: 'Neutral'}
        ret: str = 'Premise: %s\nHypothesis: %s' % (inputs[0], inputs[1])
        if label is not None:
            ret += '\nOriginal: %s' % labels[label]
        ret += '\nPrediction: Entails (%.1f), Contradicts (%.1f), Neutral (%.1f)' % (conf[self._entails], conf[self._contradicts], conf[self._neutral])
        return ret

    @classmethod
    def contractions(cls) -> Callable[..., List[Tuple[str, str]]]:
        return _wrap_apply_to_each(Perturb.contractions, both=True)

    @classmethod
    def typos(cls) -> Callable[..., List[Tuple[str, str]]]:
        return _wrap_apply_to_each(Perturb.add_typos, both=False)

    @classmethod
    def punctuation(cls) -> Callable[..., List[Tuple[str, str]]]:
        return _wrap_apply_to_each(utils.toggle_punctuation, both=False)

    def _setup_editor(self) -> None:
        super()._setup_editor()
        antonyms: List[Tuple[str, str]] = [
            ('progressive', 'conservative'),
            ('positive', 'negative'),
            ('defensive', 'offensive'),
            ('rude', 'polite'),
            ('optimistic', 'pessimistic'),
            ('stupid', 'smart'),
            ('negative', 'positive'),
            ('unhappy', 'happy'),
            ('active', 'passive'),
            ('impatient', 'patient'),
            ('powerless', 'powerful'),
            ('visible', 'invisible'),
            ('fat', 'thin'),
            ('bad', 'good'),
            ('cautious', 'brave'),
            ('hopeful', 'hopeless'),
            ('insecure', 'secure'),
            ('humble', 'proud'),
            ('passive', 'active'),
            ('dependent', 'independent'),
            ('pessimistic', 'optimistic'),
            ('irresponsible', 'responsible'),
            ('courageous', 'fearful')
        ]
        self.editor.add_lexicon('antonyms', antonyms, overwrite=True)
        synonyms: List[Tuple[str, str]] = [
            ('smart', 'intelligent'),
            ('optimistic', 'hopeful'),
            ('brave', 'courageous'),
            ('adorable', 'cute'),
            ('huge', 'enormous'),
            ('intelligent', 'clever'),
            ('lazy', 'indolent'),
            ('rude', 'impolite'),
            ('thin', 'lean'),
            ('sad', 'unhappy'),
            ('little', 'small')
        ]
        self.editor.add_lexicon('synonyms', synonyms, overwrite=True)
        comp: List[str] = ['smarter', 'better', 'worse', 'brighter', 'bigger', 'louder', 'longer', 'larger', 'smaller', 'warmer', 'colder', 'thicker', 'lighter', 'heavier']
        self.editor.add_lexicon('compare', comp, overwrite=True)
        nouns: List[str] = ['humans', 'cats', 'dogs', 'people', 'mice', 'pigs', 'birds', 'sheep', 'cows', 'rats', 'chickens', 'fish', 'bears', 'elephants', 'rabbits', 'lions', 'monkeys', 'snakes', 'bees', 'spiders', 'bats', 'puppies', 'dolphins', 'babies', 'kittens', 'children', 'frogs', 'ants', 'butterflies', 'insects', 'turtles', 'trees', 'ducks', 'whales', 'robots', 'animals', 'bugs', 'kids', 'crabs', 'carrots', 'dragons', 'mosquitoes', 'cars', 'sharks', 'dinosaurs', 'horses', 'tigers']
        self.editor.add_lexicon('nouns', nouns, overwrite=True)
        adjectives: List[str] = ['good', 'great', 'excellent', 'amazing', 'extraordinary', 'beautiful', 'fantastic', 'nice', 'awful', 'bad', 'horrible', 'weird', 'rough']
        self.editor.add_lexicon('adjectives', adjectives, overwrite=True)
        intens_adj: List[str] = ['very', 'really', 'absolutely', 'truly', 'extremely', 'quite', 'incredibly', 'amazingly', 'especially', 'exceptionally', 'unbelievably', 'utterly', 'exceedingly', 'rather', 'totally', 'particularly']
        intens_verb: List[str] = ['really', 'absolutely', 'truly', 'extremely', 'especially', 'utterly', 'totally', 'particularly', 'highly', 'definitely', 'certainly', 'genuinely', 'honestly', 'strongly', 'sure', 'sincerely']
        self.editor.add_lexicon('intens_adj', intens_adj, overwrite=True)
        self.editor.add_lexicon('intens_verb', intens_verb, overwrite=True)
        reducer_adj: List[str] = ['somewhat', 'kinda', 'mostly', 'probably', 'generally', 'reasonably', 'a little', 'a bit', 'slightly']
        self.editor.add_lexicon('reducer_adj', reducer_adj, overwrite=True)
        subclasses: List[Tuple[str, str]] = [
            ('vehicles', subtype) for subtype in ['cars', 'trucks', 'jeeps', 'bikes', 'motorcycles', 'tractors', 'vans', 'SUVs', 'minivans', 'bicycles']
        ] + [
            ('animals', subtype) for subtype in ['dogs', 'cats', 'turtles', 'lizards', 'snakes', 'fish', 'hamsters', 'rabbits', 'guinea pigs', 'ducks']
        ] + [
            ('clothes', subtype) for subtype in ['jackets', 'pants', 'shirts', 'skirts', 't-shirts', 'raincoats', 'sweaters', 'jeans', 'sweatpants']
        ]
        self.editor.add_lexicon('subclasses', subclasses, overwrite=True)

    def _default_tests(self, data: Any, num_test_cases: int = 100) -> None:
        super()._default_tests(data, num_test_cases)
        self._setup_editor()
        self._default_vocabulary_tests(data, num_test_cases)
        self._default_ner_tests(data, num_test_cases)
        self._default_temporal_tests(data, num_test_cases)
        self._default_logic_tests(data, num_test_cases)
        self._default_negation_tests(data, num_test_cases)
        self._default_taxonomy_tests(data, num_test_cases)
        self._default_coreference_tests(data, num_test_cases)
        self._default_fairness_tests(data, num_test_cases)

    def _default_vocabulary_tests(self, data: Any, num_test_cases: int = 100) -> None:
        template = self.editor.template(('{first_name1} is more {antonyms[0]} than {first_name2}', '{first_name2} is more {antonyms[1]} than {first_name1}'),
                                        remove_duplicates=True, nsamples=num_test_cases)
        test = MFT(**template, labels=self._entails, name='"A is more COMP than B" entails "B is more antonym(COMP) than A"',
                   capability='Vocabulary', description='Eg. A is more active than B implies that B is more passive than A')
        self.add_test(test)
        template = self.editor.template([('All {nouns} are {synonyms[0]}.', 'Some {nouns} are {synonyms[0]}.'),
                                         ('All {nouns} are {synonyms[0]}.', 'Some {nouns} are {synonyms[1]}.')],
                                        remove_duplicates=True, nsamples=num_test_cases)
        _num_entails = len(template.data)
        template += self.editor.template([('All {nouns} are {synonyms[0]}.', 'Some {nouns} are not {synonyms[0]}.'),
                                          ('All {nouns} are {synonyms[0]}.', 'Some {nouns} are not {synonyms[1]}.')],
                                         remove_duplicates=True, nsamples=num_test_cases)
        _num_contradicts = len(template.data) - _num_entails
        test = INV(template.data, labels=[self._entails for _ in range(_num_entails)] + [self._contradicts for _ in range(_num_contradicts)],
                   name='Changing X to a synonym(X) should not change the label', capability='Vocabulary',
                   description='"Eg. All tigers are huge -> All tigers are enormous" should not change the label')
        self.add_test(test)

    def _default_taxonomy_tests(self, data: Any, num_test_cases: int = 100) -> None:
        template = self.editor.template(('{first_name1} owns {subclasses[1]}.', '{first_name1} owns {subclasses[0]}.'),
                                        remove_duplicates=True, nsamples=num_test_cases)
        test = MFT(**template, labels=self._entails,
                   name='"A owns SUBTYPE" entails "A owns SUPERTYPE"',
                   capability='Taxonomy', description='Eg. A owns rabbits implies that A owns animals.')
        self.add_test(test)

    def _default_coreference_tests(self, data: Any, num_test_cases: int = 100) -> None:
        _quarter: int = num_test_cases // 4
        template = self.editor.template(('{first_name1} and {first_name2} are friends. The former is {a:profession}.', '{first_name1} is {a:profession}.'),
                                        remove_duplicates=True, nsamples=_quarter)
        template += self.editor.template(('{first_name1} and {first_name2} are friends. The latter is {a:profession}.', '{first_name2} is {a:profession}.'),
                                         remove_duplicates=True, nsamples=_quarter)
        _num_entails: int = len(template.data)
        template += self.editor.template(('{first_name1} and {first_name2} are friends. The former is {a:profession}.', '{first_name2} is {a:profession}.'),
                                         remove_duplicates=True, nsamples=_quarter)
        template += self.editor.template(('{first_name1} and {first_name2} are friends. The latter is {a:profession}.', '{first_name1} is {a:profession}.'),
                                         remove_duplicates=True, nsamples=_quarter)
        _num_neutral: int = len(template.data) - _num_entails
        test = MFT(**template, labels=[self._entails for _ in range(_num_entails)] + [self._neutral for _ in range(_num_neutral)],
                   name='Former / Latter',
                   capability='Coreference',
                   description='Eg. "A and B are friends. The former is a teacher." entails "A is a teacher." (while "B is a teacher" is neutral).')
        self.add_test(test)

    def _default_robustness_tests(self, data: Any, num_test_cases: int = 100) -> None:
        template = self.editor.template(('{nouns1} and {nouns2} are {adjectives}.', '{nouns2} and {nouns1} are {adjectives}.'),
                                        remove_duplicates=True, nsamples=num_test_cases)
        test = MFT(**template, labels=self._entails,
                   name='"A and B are X" entails "B and A are X"',
                   capability='Vocabulary', description='Eg. "tigers and lions are huge" entails that "lions and tigers are huge"')
        self.add_test(test)
        if data:
            template = Perturb.perturb(data, _wrap_apply_to_each(utils.add_random_strings), nsamples=num_test_cases)
            test = INV(template.data, name='Add random urls and handles', capability='Robustness',
                       description='Add randomly generated urls and handles to the start or end of sentence')
            self.add_test(test)

    def _default_logic_tests(self, data: Any, num_test_cases: int = 100) -> None:
        template = self.editor.template(('{nouns1} are {compare} than {nouns2}', '{nouns2} are {compare} than {nouns1}'),
                                        nsamples=num_test_cases, remove_duplicates=True)
        test = MFT(**template, labels=self._contradicts,
                   name='"A is COMP than B" contradicts "B is COMP than A"',
                   capability='Logic', description='Eg. "A is better than B" contradicts "B is better than A"')
        self.add_test(test)
        if data:
            template = Perturb.perturb(data, lambda x: (x[0], x[0]), nsamples=num_test_cases, keep_original=False)
            template += Perturb.perturb(data, lambda x: (x[1], x[1]), nsamples=num_test_cases, keep_original=False)
            test = MFT(**template, labels=self._entails,
                       name='A entails A (premise == hypothesis)',
                       capability='Logic', description='If premise and hypothesis are the same, then premise entails the hypothesis')
            self.add_test(test)

    def _default_negation_tests(self, data: Any, num_test_cases: int = 100) -> None:
        template = self.editor.template(('{first_name1} is {compare} than {first_name2}', '{first_name1} is not {compare} than {first_name2}'),
                                        nsamples=num_test_cases, remove_duplicates=True)
        test = MFT(**template, labels=self._contradicts,
                   name='"A is COMP than B" contradicts "A is not COMP than B"',
                   capability='Negation', description='Eg. A is better than B contradicts A is not better than C')
        self.add_test(test)

    def _default_ner_tests(self, data: Any, num_test_cases: int = 100) -> None:
        template = self.editor.template(('{first_name1} is {compare} than {first_name2}', '{first_name1} is {compare} than {first_name3}'),
                                        nsamples=num_test_cases, remove_duplicates=True)
        test = MFT(**template, labels=self._neutral,
                   name='"A is COMP than B" gives no information about "A is COMP than C"',
                   capability='NER', description='Eg. "A is better than B" gives no information about "A is better than C"')
        self.add_test(test)

    def _default_temporal_tests(self, data: Any, num_test_cases: int = 100) -> None:
        template = self.editor.template(('{first_name} works as {a:profession}', '{first_name} used to work as a {profession}'),
                                        nsamples=num_test_cases, remove_duplicates=True)
        template += self.editor.template(('{first_name} {last_name} is {a:profession}', '{first_name} {last_name} was {a:profession}'),
                                         nsamples=num_test_cases, remove_duplicates=True)
        test = MFT(**template, labels=self._neutral,
                   name='"A works as P" gives no information about "A used to work as P"',
                   capability='Temporal', description='Eg. "A is a writer" gives no information about "A was a writer"')
        self.add_test(test)
        template = self.editor.template(('{first_name} was {a:profession1} before they were {a:profession2}', '{first_name} was {a:profession1} after they were {a:profession2}'),
                                        nsamples=num_test_cases, remove_duplicates=True)
        test = MFT(**template, labels=self._contradicts,
                   name='Before != After', capability='Temporal',
                   description='Eg. "A was a writer before they were a journalist" contradicts "A was a writer after they were a journalist"')
        self.add_test(test)

    def _default_fairness_tests(self, data: Any, num_test_cases: int = 100) -> None:
        male_stereotypes: List[Tuple[str, str]] = [
            ('maestro', 'maestros'),
            ('skipper', 'skippers'),
            ('philosopher', 'philosophers'),
            ('architect', 'architects'),
            ('financier', 'financiers'),
            ('warrior', 'warriors'),
            ('broadcaster', 'broadcasters'),
            ('doctor', 'doctors'),
            ('captain', 'captains'),
            ('protege', 'proteges'),
            ('boss', 'bosses'),
            ('magician', 'magicians'),
            ('fighter pilot', 'fighter pilots')
        ]
        female_stereotypes: List[Tuple[str, str]] = [
            ('nurse', 'nurses'),
            ('homemaker', 'homemakers'),
            ('receptionist', 'receptionists'),
            ('stylist', 'stylists'),
            ('housekeeper', 'housekeepers'),
            ('socialite', 'socialites'),
            ('librarian', 'librarians'),
            ('hairdresser', 'hairdressers'),
            ('nanny', 'nannies'),
            ('interior designer', 'interior designers'),
            ('guidance counselor', 'guidance counselors')
        ]
        template = self.editor.template(
            [('All people in the park are {male_stereotypes[1]}. {male} is in the park.', 'He is a {male_stereotypes[0]}.'),
             ('All people in the park are {male_stereotypes[1]}. {female} is in the park.', 'She is a {male_stereotypes[0]}.')],
            nsamples=num_test_cases // 2,
            remove_duplicates=True,
            male_stereotypes=male_stereotypes
        )
        test = DIR(template.data, Expect.monotonic(label=self._entails, increasing=True, tolerance=0.1),
                   name='Male stereotypes',
                   capability='Fairness',
                   description="Eg. 'All people are X' should entail 'He is X' and 'She is X' equally.")
        self.add_test(test)
        template = self.editor.template(
            [('All people in the park are {female_stereotypes[1]}. She is in the park.', '{female} is a {female_stereotypes[0]}.'),
             ('All people in the park are {female_stereotypes[1]}. He is in the park.', '{male} is a {female_stereotypes[0]}.')],
            nsamples=num_test_cases // 2,
            remove_duplicates=True,
            female_stereotypes=female_stereotypes
        )
        test = DIR(template.data, Expect.monotonic(label=self._entails, increasing=True, tolerance=0.1),
                   name='Female stereotypes',
                   capability='Fairness',
                   description="Eg. 'All people are X' should entail 'He is X' and 'She is X' equally.")
        self.add_test(test)