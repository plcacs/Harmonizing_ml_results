from typing import Optional, Iterable, List, Union, Tuple, Dict, Any, Callable
import numpy as np

from checklist.test_suite import TestSuite
from checklist.test_types import MFT, INV, DIR, Expect
from checklist.perturb import Perturb
from allennlp.confidence_checks.task_checklists.task_suite import TaskSuite
from allennlp.confidence_checks.task_checklists import utils
from allennlp.data.instance import Instance
from allennlp.predictors import Predictor


def _add_phrase_function(phrases: List[str], num_samples: int = 10) -> Callable[[str], List[str]]:
    """
    Returns a function which adds each str in `phrases`
    at the end of the input string and returns that list.
    """

    def perturb_fn(inp: str) -> List[str]:
        input_str = utils.strip_punctuation(inp)
        total = len(phrases)
        idx = np.random.choice(total, min(num_samples, total), replace=False)
        ret = [input_str + ". " + phrases[i] for i in idx]
        return ret

    return perturb_fn


@TaskSuite.register("sentiment-analysis")
class SentimentAnalysisSuite(TaskSuite):
    """
    This suite was built using the checklist process with the self.editor
    suggestions. Users are encouraged to add/modify as they see fit.

    Note: `editor.suggest(...)` can be slow as it runs a language model.
    """

    def __init__(
        self,
        suite: Optional[TestSuite] = None,
        positive: Optional[int] = 0,
        negative: Optional[int] = 1,
        **kwargs: Any,
    ) -> None:

        self._positive = positive
        self._negative = negative
        super().__init__(suite, **kwargs)

    def _prediction_and_confidence_scores(self, predictor: Predictor) -> Callable[[List[Union[str, Instance]]], Tuple[np.ndarray, np.ndarray]]:
        def preds_and_confs_fn(data: List[Union[str, Instance]]) -> Tuple[np.ndarray, np.ndarray]:
            labels = []
            confs = []
            if isinstance(data[0], Instance):
                predictions = predictor.predict_batch_instance(data)
            else:
                data = [{"sentence": sentence} for sentence in data]
                predictions = predictor.predict_batch_json(data)
            for pred in predictions:
                label = pred["probs"].index(max(pred["probs"]))
                labels.append(label)
                confs.append(pred["probs"])
            return np.array(labels), np.array(confs)

        return preds_and_confs_fn

    def _format_failing_examples(
        self,
        inputs: Tuple,
        pred: int,
        conf: Union[np.array, np.ndarray],
        label: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """
        Formatting function for printing failed test examples.
        """
        labels = {self._positive: "Positive", self._negative: "Negative"}
        ret = str(inputs)
        if label is not None:
            ret += " (Original: %s)" % labels[label]
        ret += "\nPrediction: %s (Confidence: %.1f)" % (labels[pred], conf[pred])

        return ret

    def _default_tests(self, data: Optional[Iterable[str]], num_test_cases: int = 100) -> None:
        super()._default_tests(data, num_test_cases)
        self._setup_editor()
        self._default_vocabulary_tests(data, num_test_cases)
        self._default_ner_tests(data, num_test_cases)
        self._default_temporal_tests(data, num_test_cases)
        self._default_fairness_tests(data, num_test_cases)
        self._default_negation_tests(data, num_test_cases)

    def _setup_editor(self) -> None:
        super()._setup_editor()

        pos_adj = [
            "good",
            "great",
            "excellent",
            "amazing",
            "extraordinary",
            "beautiful",
            "fantastic",
            "nice",
            "incredible",
            "exceptional",
            "awesome",
            "perfect",
            "fun",
            "adorable",
            "brilliant",
            "exciting",
            "sweet",
            "wonderful",
        ]
        neg_adj = [
            "awful",
            "bad",
            "horrible",
            "weird",
            "rough",
            "lousy",
            "average",
            "difficult",
            "poor",
            "sad",
            "frustrating",
            "lame",
            "nasty",
            "annoying",
            "boring",
            "creepy",
            "dreadful",
            "ridiculous",
            "terrible",
            "ugly",
            "unpleasant",
        ]
        self.editor.add_lexicon("pos_adj", pos_adj, overwrite=True)
        self.editor.add_lexicon("neg_adj", neg_adj, overwrite=True)

        pos_verb_present = [
            "like",
            "enjoy",
            "appreciate",
            "love",
            "recommend",
            "admire",
            "value",
            "welcome",
        ]
        neg_verb_present = ["hate", "dislike", "regret", "abhor", "dread", "despise"]
        pos_verb_past = [
            "liked",
            "enjoyed",
            "appreciated",
            "loved",
            "admired",
            "valued",
            "welcomed",
        ]
        neg_verb_past = ["hated", "disliked", "regretted", "abhorred", "dreaded", "despised"]
        self.editor.add_lexicon("pos_verb_present", pos_verb_present, overwrite=True)
        self.editor.add_lexicon("neg_verb_present", neg_verb_present, overwrite=True)
        self.editor.add_lexicon("pos_verb_past", pos_verb_past, overwrite=True)
        self.editor.add_lexicon("neg_verb_past", neg_verb_past, overwrite=True)
        self.editor.add_lexicon("pos_verb", pos_verb_present + pos_verb_past, overwrite=True)
        self.editor.add_lexicon("neg_verb", neg_verb_present + neg_verb_past, overwrite=True)

        noun = [
            "airline",
            "movie",
            "product",
            "customer service",
            "restaurant",
            "hotel",
            "food",
            "staff",
            "company",
            "crew",
            "service",
        ]
        self.editor.add_lexicon("noun", noun, overwrite=True)

        intens_adj = [
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
        intens_verb = [
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

        reducer_adj = [
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

        self.monotonic_label = Expect.monotonic(increasing=True, tolerance=0.1)
        self.monotonic_label_down = Expect.monotonic(increasing=False, tolerance=0.1)

    def _default_vocabulary_tests(self, data: Optional[Iterable[str]], num_test_cases: int = 100) -> None:

        positive_words = (
            self.editor.lexicons["pos_adj"]
            + self.editor.lexicons["pos_verb_present"]
            + self.editor.lexicons["pos_verb_past"]
        )

        test = MFT(
            positive_words,
            labels=self._positive,
            name="Single Positive Words",
            capability="Vocabulary",
            description="Correctly recognizes positive words",
        )

        self.add_test(test)

        negative_words = (
            self.editor.lexicons["neg_adj"]
            + self.editor.lexicons["neg_verb_present"]
            + self.editor.lexicons["neg_verb_past"]
        )

        test = MFT(
            negative_words,
            labels=self._negative,
            name="Single Negative Words",
            capability="Vocabulary",
            description="Correctly recognizes negative words",
        )

        self.add_test(test)

        template = self.editor.template(
            "{it} {noun} {be} {pos_adj}.",
            it=["The", "This", "That"],
            be=["is", "was"],
            labels=self._positive,
            save=True,
        )
        template += self.editor.template(
            "{it} {be} {a:pos_adj} {noun}.",
            it=["It", "This", "That"],
            be=["is", "was"],
            labels=self._positive,
            save=True,
        )
        template += self.editor.template(
            "{i} {pos_verb} {the} {noun}.",
            i=["I", "We"],
            the=["this", "that", "the"],
            labels=self._positive,
            save=True,
        )
        template += self.editor.template(
            "{it} {noun} {be} {neg_adj}.",
            it=["That", "This", "The"],
            be=["is", "was"],
            labels=self._negative,
            save=True,
        )
        template += self.editor.template(
            "{it} {be} {a:neg_adj} {noun}.",
            it=["It", "This", "That"],
            be=["is", "was"],
            labels=self._negative,
            save=True,
        )
        template += self.editor.template(
            "{i} {neg_verb} {the} {noun}.",
            i=["I", "We"],
            the=["this", "that", "the"],
            labels=self._negative,
            save=True,
        )

        test = MFT(
            **template,
            name="Sentiment-laden words in context",
            capability="Vocabulary",
            description="Use positive and negative verbs and adjectives "
            "with nouns such as product, movie, airline, etc. "
            'E.g. "This was a bad movie"',
        )

        self.add_test(test)

        template = self.editor.template(
            ["{it} {be} {a:pos_adj} {noun}.", "{it} {be} {a:intens_adj} {pos_adj} {noun}."],
            it=["It", "This", "That"],
            be=["is", "was"],
            nsamples=num_test_cases,
            save=True,
        )
        template += self.editor.template(
            ["{i} {pos_verb} {the} {noun}.", "{i} {intens_verb} {pos_verb} {the} {noun}."],
            i=["I", "We"],
            the=["this", "that", "the"],
            nsamples=num_test_cases,
            save=True,
        )
        template += self.editor.template(
            ["{it} {be} {a:neg_adj} {noun}.", "{it} {be} {a:intens_adj} {neg_adj} {noun}."],
            it=["It", "This", "That"],
            be=["is", "was"],
            nsamples=num_test_cases,
            save=True,
        )
        template += self.editor.template(
            ["{i} {neg_verb} {the} {noun}.", "{i} {intens_verb} {neg_verb} {the} {noun}."],
            i=["I", "We"],
            the=["this", "that", "the"],
            nsamples=num_test_cases,
            save=True,
        )

        test = DIR(
            template.data,
            self.monotonic_label,
            templates=template.templates,
            name="Intensifiers",
            capability="Vocabulary",
            description="Test is composed of pairs of sentences (x1, x2), where we add an intensifier "
            "such as 'really',or 'very' to x2 and expect the confidence to NOT go down "
            "(with tolerance=0.1). e.g.:"
            "x1 = 'That was a good movie'"
            "x2 = 'That was a very good movie'",
        )

        self.add_test(test)

        template = self.editor.template(
            ["{it} {noun} {be} {pos_adj}.", "{it} {noun} {be} {reducer_adj} {pos_adj}."],
            it=["The", "This", "That"],
            be=["is", "was"],
            nsamples=num_test_cases,
            save=True,
        )
        template += self.editor.template(
            ["{it} {noun} {be} {neg_adj}.", "{it} {noun} {be} {reducer_adj} {neg_adj}."],
            it=["The", "This", "That"],
            be=["is", "was"],
            nsamples=num_test_cases,
            save=True,
        )
        test = DIR(
            template.data,
            self.monotonic_label_down,
            templates=template.templates,
            name="Reducers",
            capability="Vocabulary",
            description="Test is composed of pairs of sentences (x1, x2), where we add a reducer "
            "such as 'somewhat', or 'kinda' to x2 and expect the confidence to NOT go up "
            " (with tolerance=0.1). e.g.:"
            "x1 = 'The staff was good.'"
            "x2 = 'The staff was somewhat good.'",
        )

        self.add_test(test)

        if data:

            positive = self.editor.template("I {pos_verb_present} you.").data
            positive += self.editor.template("You are {pos_adj}.").data

            negative = self.editor.template("I {neg_verb_present} you.").data
            negative += self.editor.template("You are {neg_adj}.").data

            template = Perturb.perturb(
                data, _add_phrase_function(positive), nsamples=num_test_cases
            )
            test = DIR(
                template.data,
                Expect.pairwise(self._diff_up),
                name="Add positive phrases",
                capability="Vocabulary",
                description="Add very positive phrases (e.g. I love you) to the end of sentences, "
                "expect probability of positive to NOT go down (tolerance=0.1)",
            )

            self.add_test(test)

            template = Perturb.perturb(
                data, _add_phrase_function(negative), nsamples=num_test_cases
            )
            test = DIR(
                template.data,
                Expect.pairwise(self._diff_down),
                name="Add negative phrases",
                capability="Vocabulary",
                description="Add very negative phrases (e.g. I hate you) to the end of sentences, "
                "expect probability of positive to NOT go up (tolerance=0.1)",
            )

            self.add_test(test)

    def _default_robustness_tests(self, data: Optional[Iterable[str]], num_test_cases: int = 100) -> None:

        template = Perturb.perturb(data, utils.add_random_strings, nsamples=num_test_cases)
        test = INV(
            template.data,
            name="Add random urls and handles",
            capability="Robustness",
            description="Add randomly generated urls and handles to the start or end of sentence",
        )

        self.add_test(test)

    def _default_ner_tests(self, data: Optional[Iterable[str]], num_test_cases: int = 100) -> None:
        if data:
            template = Perturb.perturb(
                data, utils.spacy_wrap(Perturb.change_names, ner=True), nsamples=num_test_cases
            )
            test = INV(
                template.data,
                name="Change names",
                capability="NER",
                description="Replace names with other common names",
            )
            self.add_test(test)

            template = Perturb.perturb(
                data, utils.spacy_wrap(Perturb.change_location, ner=True), nsamples=num_test_cases
            )
            test = INV(
                template.data,
                name="Change locations",
                capability="NER",
                description="Replace city or country names with other cities or countries",
            )
            self.add_test(test)

            template = Perturb.perturb(
                data, utils.spacy_wrap(Perturb.change_number, ner=True), nsamples=num_test_cases
            )
            test = INV(
                template.data,
                name="Change numbers",
                capability="NER",
                description="Replace integers with random integers within a 20% radius of the original",
            )
            self.add_test(test)

    def _default_temporal_tests(self, data: Optional[Iterable[str]], num_test_cases: int = 100) -> None:
        self._setup_editor()

        change = ["but", "even though", "although", ""]
        template = self.editor.template(
            [
                "I used to think this {noun} was {neg_adj}, {change} now I think it is {pos_adj}.",
                "I think this {noun} is {pos_adj}, {change} I used to think it was {neg_adj}.",
                "In the past I thought this {noun} was {neg_adj}, {change} now I think it is {pos_adj}.",
                "I think this {