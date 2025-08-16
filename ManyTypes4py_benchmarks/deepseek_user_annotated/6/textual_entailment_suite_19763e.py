from typing import Optional, Tuple, Iterable, Callable, Union, List, Dict, Any, Sequence
import itertools
import numpy as np

from checklist.test_suite import TestSuite
from checklist.test_types import MFT, INV, DIR, Expect
from checklist.perturb import Perturb
from allennlp.confidence_checks.task_checklists.task_suite import TaskSuite
from allennlp.confidence_checks.task_checklists import utils
from allennlp.predictors import Predictor


def _wrap_apply_to_each(perturb_fn: Callable[..., Union[str, List[str]]], both: bool = False, *args: Any, **kwargs: Any) -> Callable[[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Wraps the perturb function so that it is applied to
    both elements in the (premise, hypothesis) tuple.
    """

    def new_fn(pair: Tuple[str, str], *args: Any, **kwargs: Any) -> List[Tuple[str, str]]:
        premise, hypothesis = pair[0], pair[1]
        ret: List[Tuple[str, str]] = []
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

        # The perturb function can return empty strings, if no relevant perturbations
        # can be applied. Eg. if the sentence is "This is a good movie", a perturbation
        # which toggles contractions will have no effect.
        return [x for x in ret if x[0] and x[1]]

    return new_fn


@TaskSuite.register("textual-entailment")
class TextualEntailmentSuite(TaskSuite):
    def __init__(
        self,
        suite: Optional[TestSuite] = None,
        entails: int = 0,
        contradicts: int = 1,
        neutral: int = 2,
        premise: str = "premise",
        hypothesis: str = "hypothesis",
        probs_key: str = "probs",
        **kwargs: Any,
    ) -> None:

        self._entails: int = entails
        self._contradicts: int = contradicts
        self._neutral: int = neutral

        self._premise: str = premise
        self._hypothesis: str = hypothesis

        self._probs_key: str = probs_key

        super().__init__(suite, **kwargs)

    def _prediction_and_confidence_scores(self, predictor: Predictor) -> Callable[[List[Tuple[str, str]]], Tuple[np.ndarray, np.ndarray]]:
        def preds_and_confs_fn(data: List[Tuple[str, str]]) -> Tuple[np.ndarray, np.ndarray]:
            labels: List[int] = []
            confs: List[np.ndarray] = []

            data = [{self._premise: pair[0], self._hypothesis: pair[1]} for pair in data]
            predictions: List[Dict[str, Any]] = predictor.predict_batch_json(data)
            for pred in predictions:
                label = np.argmax(pred[self._probs_key])
                labels.append(label)
                confs.append(pred[self._probs_key])
            return np.array(labels), np.array(confs)

        return preds_and_confs_fn

    def _format_failing_examples(
        self,
        inputs: Tuple[str, str],
        pred: int,
        conf: Union[np.array, np.ndarray],
        label: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """
        Formatting function for printing failed test examples.
        """
        labels: Dict[int, str] = {
            self._entails: "Entails",
            self._contradicts: "Contradicts",
            self._neutral: "Neutral",
        }
        ret = "Premise: %s\nHypothesis: %s" % (inputs[0], inputs[1])
        if label is not None:
            ret += "\nOriginal: %s" % labels[label]
        ret += "\nPrediction: Entails (%.1f), Contradicts (%.1f), Neutral (%.1f)" % (
            conf[self._entails],
            conf[self._contradicts],
            conf[self._neutral],
        )

        return ret

    @classmethod
    def contractions(cls) -> Callable[[Tuple[str, str]], List[Tuple[str, str]]]:
        return _wrap_apply_to_each(Perturb.contractions, both=True)

    @classmethod
    def typos(cls) -> Callable[[Tuple[str, str]], List[Tuple[str, str]]]:
        return _wrap_apply_to_each(Perturb.add_typos, both=False)

    @classmethod
    def punctuation(cls) -> Callable[[Tuple[str, str]], List[Tuple[str, str]]]:
        return _wrap_apply_to_each(utils.toggle_punctuation, both=False)

    def _setup_editor(self) -> None:
        super()._setup_editor()

        antonyms: List[Tuple[str, str]] = [
            ("progressive", "conservative"),
            ("positive", "negative"),
            ("defensive", "offensive"),
            ("rude", "polite"),
            ("optimistic", "pessimistic"),
            ("stupid", "smart"),
            ("negative", "positive"),
            ("unhappy", "happy"),
            ("active", "passive"),
            ("impatient", "patient"),
            ("powerless", "powerful"),
            ("visible", "invisible"),
            ("fat", "thin"),
            ("bad", "good"),
            ("cautious", "brave"),
            ("hopeful", "hopeless"),
            ("insecure", "secure"),
            ("humble", "proud"),
            ("passive", "active"),
            ("dependent", "independent"),
            ("pessimistic", "optimistic"),
            ("irresponsible", "responsible"),
            ("courageous", "fearful"),
        ]

        self.editor.add_lexicon("antonyms", antonyms, overwrite=True)

        synonyms: List[Tuple[str, str]] = [
            ("smart", "intelligent"),
            ("optimistic", "hopeful"),
            ("brave", "courageous"),
            ("adorable", "cute"),
            ("huge", "enormous"),
            ("intelligent", "clever"),
            ("lazy", "indolent"),
            ("rude", "impolite"),
            ("thin", "lean"),
            ("sad", "unhappy"),
            ("little", "small"),
        ]

        self.editor.add_lexicon("synonyms", synonyms, overwrite=True)

        comp: List[str] = [
            "smarter",
            "better",
            "worse",
            "brighter",
            "bigger",
            "louder",
            "longer",
            "larger",
            "smaller",
            "warmer",
            "colder",
            "thicker",
            "lighter",
            "heavier",
        ]

        self.editor.add_lexicon("compare", comp, overwrite=True)

        nouns: List[str] = [
            "humans",
            "cats",
            "dogs",
            "people",
            "mice",
            "pigs",
            "birds",
            "sheep",
            "cows",
            "rats",
            "chickens",
            "fish",
            "bears",
            "elephants",
            "rabbits",
            "lions",
            "monkeys",
            "snakes",
            "bees",
            "spiders",
            "bats",
            "puppies",
            "dolphins",
            "babies",
            "kittens",
            "children",
            "frogs",
            "ants",
            "butterflies",
            "insects",
            "turtles",
            "trees",
            "ducks",
            "whales",
            "robots",
            "animals",
            "bugs",
            "kids",
            "crabs",
            "carrots",
            "dragons",
            "mosquitoes",
            "cars",
            "sharks",
            "dinosaurs",
            "horses",
            "tigers",
        ]
        self.editor.add_lexicon("nouns", nouns, overwrite=True)

        adjectives: List[str] = [
            "good",
            "great",
            "excellent",
            "amazing",
            "extraordinary",
            "beautiful",
            "fantastic",
            "nice",
            "awful",
            "bad",
            "horrible",
            "weird",
            "rough",
        ]
        self.editor.add_lexicon("adjectives", adjectives, overwrite=True)

        intens_adj: List[str] = [
            "very",
            "really",
            "absolutely",
            "truly",
            "extremely",
            "quite",
            "incredibly",
            "amazingly",
            "especially",
            "exceptionally",
            "unbelievably",
            "utterly",
            "exceedingly",
            "rather",
            "totally",
            "particularly",
        ]
        intens_verb: List[str] = [
            "really",
            "absolutely",
            "truly",
            "extremely",
            "especially",
            "utterly",
            "totally",
            "particularly",
            "highly",
            "definitely",
            "certainly",
            "genuinely",
            "honestly",
            "strongly",
            "sure",
            "sincerely",
        ]

        self.editor.add_lexicon("intens_adj", intens_adj, overwrite=True)
        self.editor.add_lexicon("intens_verb", intens_verb, overwrite=True)

        reducer_adj: List[str] = [
            "somewhat",
            "kinda",
            "mostly",
            "probably",
            "generally",
            "reasonably",
            "a little",
            "a bit",
            "slightly",
        ]

        self.editor.add_lexicon("reducer_adj", reducer_adj, overwrite=True)

        subclasses: List[Tuple[str, str]] = [
            (
                "vehicles",
                [
                    "cars",
                    "trucks",
                    "jeeps",
                    "bikes",
                    "motorcycles",
                    "tractors",
                    "vans",
                    "SUVs",
                    "minivans",
                    "bicycles",
                ],
            ),
            (
                "animals",
                [
                    "dogs",
                    "cats",
                    "turtles",
                    "lizards",
                    "snakes",
                    "fish",
                    "hamsters",
                    "rabbits",
                    "guinea pigs",
                    "ducks",
                ],
            ),
            (
                "clothes",
                [
                    "jackets",
                    "pants",
                    "shirts",
                    "skirts",
                    "t-shirts",
                    "raincoats",
                    "sweaters",
                    "jeans",
                    "sweatpants",
                ],
            ),
        ]

        subclasses = [(a, b[i]) for a, b in subclasses for i in range(len(b))]
        self.editor.add_lexicon("subclasses", subclasses, overwrite=True)

    def _default_tests(self, data: Optional[Iterable[Tuple[str, str]]], num_test_cases: int = 100) -> None:
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

    def _default_vocabulary_tests(self, data: Optional[Iterable[Tuple[str, str]]], num_test_cases: int = 100) -> None:

        template = self.editor.template(
            (
                "{first_name1} is more {antonyms[0]} than {first_name2}",
                "{first_name2} is more {antonyms[1]} than {first_name1}",
            ),
            remove_duplicates=True,
            nsamples=num_test_cases,
        )

        test = MFT(
            **template,
            labels=self._entails,
            name='"A is more COMP than B" entails "B is more antonym(COMP) than A"',
            capability="Vocabulary",
            description="Eg. A is more active than B implies that B is more passive than A",
        )

        self.add_test(test)

        template = self.editor.template(
            [
                ("All {nouns} are {synonyms[0]}.", "Some {nouns} are {synonyms[0]}."),
                ("All {nouns} are {synonyms[0]}.", "Some {nouns} are {synonyms[1]}."),
            ],
            remove_duplicates=True,
            nsamples=num_test_cases,
        )

        _num_entails = len(template.data)

        template += self.editor.template(
            [
                ("All {nouns} are {synonyms[0]}.", "Some {nouns} are not {synonyms[0]}."),
                ("All {nouns} are {synonyms[0]}.", "Some {nouns} are not {synonyms[1]}."),
            ],
            remove_duplicates=True,
            nsamples=num_test_cases,
        )

        _num_contradicts = len(template.data) - _num_entails

        test = INV(
            template.data,
            labels=[self._entails for i in range(_num_entails)]
            + [self._contradicts for i in range(_num_contradicts)],
            name="Changing X to a synonym(X) should not change the label",
            capability="Vocabulary",
            description='"Eg. All tigers are huge -> All tigers are enormous" should not change the label',
        )

        self.add_test(test)

    def _default_taxonomy_tests(self, data: Optional[Iterable[Tuple[str, str]]], num_test_cases: int = 100) -> None:

        template = self.editor.template(
            ("{first_name1} owns {subclasses[1]}.", "{first_name1} owns {subclasses[0]}."),
            remove_duplicates=True,
            nsamples=num_test_cases,
        )

        test = MFT(
            **template,
            labels=self._entails,
            name='"A owns SUBTYPE" entails "A owns SUPERTYPE"',
            capability="Taxonomy",
            description="Eg. A owns rabbits implies that A owns animals.",
        )

        self.add_test(test)

    def _default_coreference_tests(
        self, data: Optional[Iterable[Tuple[str, str]]], num_test_cases: int = 100
    ) -> None:

        _quarter = num_test_cases // 4

        template = self.editor.template(
            (
                "{first_name1} and {first_name2} are friends. The former is {a:profession}.",
                "{first_name1} is {a:profession}.",
            ),
            remove_duplicates=True,
            nsamples=_quarter,
        )

        template += self.editor.template(
            (
                "{first_name1} and {first_name2} are friends. The latter is {a:profession}.",
                "{first_name2} is {a:profession}.",
            ),
            remove_duplicates=True,
            nsamples=_quarter,
        )

        _num_entails = len(template.data)

        template += self.editor.template(
            (
                "{first_name1} and {first_name2} are friends. The former is {a:profession}.",
                "{first_name2} is {a:profession}.",
            ),
            remove_duplicates=True,
            nsamples=_quarter,
        )

        template += self.editor.template(
            (
                "{first_name1} and {first_name2} are friends. The latter is {a:profession}.",
                "{first_name1} is {a:profession}.",
            ),
            remove_duplicates=True,
            nsamples=_quarter,
        )

        _num_neutral = len(template.data) - _num_entails

        test = MFT(
            **template,
            labels=[self._entails for i in range(_num_entails)]
            + [self._neutral for i in range(_num_neutral)],
            name="Former / Latter",
            capability="Coreference",
            description='Eg. "A and B are friends. The former is a teacher."'
            + ' entails "A is a teacher." (while "B is a teacher" is neutral).',
        )

        self.add_test(test)

    def _default_robustness_tests(self, data: Optional[Iterable[Tuple[str, str]]], num_test_cases: int = 100) -> None:

        template = self.editor.template(
            (
                "{nouns1} and {nouns2} are {adjectives}.",
                "{nouns2} and {nouns1} are {adjectives}.",
            ),
            remove_duplicates=True,
            nsamples=num_test_cases,
        )

        test = MFT(
            **template,
            labels=self._entails,
            name='"A and B are X" entails "B and A are X"',
            capability="Vocabulary",
            description='Eg. "tigers and lions are huge" entails that "lions and tigers are huge"',
        )

        self.add_test(test)

        if data:

            template = Perturb.perturb(
                data, _wrap_apply_to_each(utils.add_random