import logging
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union, overload, Literal
from os import PathLike
import numpy as np
from numpy.typing import NDArray
from snorkel.analysis import Scorer
from snorkel.utils import probs_to_preds

TieBreakPolicy = Literal["abstain", "true-random", "random"]


class BaseLabeler(ABC):
    """Abstract baseline label voter class."""

    def __init__(self, cardinality: int = 2, **kwargs: Any) -> None:
        self.cardinality: int = cardinality

    @abstractmethod
    def predict_proba(self, L: NDArray[np.int_]) -> NDArray[np.float_]:
        """Abstract method for predicting probabilistic labels given a label matrix.

        Parameters
        ----------
        L
            An [n,m] matrix with values in {-1,0,1,...,k-1}f

        Returns
        -------
        np.ndarray
            An [n,k] array of probabilistic labels
        """
        pass

    @overload
    def predict(
        self,
        L: NDArray[np.int_],
        return_probs: Literal[True],
        tie_break_policy: TieBreakPolicy = "abstain",
    ) -> Tuple[NDArray[np.int_], NDArray[np.float_]]: ...
    @overload
    def predict(
        self,
        L: NDArray[np.int_],
        return_probs: Literal[False] = False,
        tie_break_policy: TieBreakPolicy = "abstain",
    ) -> NDArray[np.int_]: ...
    def predict(
        self,
        L: NDArray[np.int_],
        return_probs: bool = False,
        tie_break_policy: TieBreakPolicy = "abstain",
    ) -> Union[NDArray[np.int_], Tuple[NDArray[np.int_], NDArray[np.float_]]]:
        """Return predicted labels, with ties broken according to policy.

        Policies to break ties include:
        "abstain": return an abstain vote (-1)
        "true-random": randomly choose among the tied options
        "random": randomly choose among tied option using deterministic hash

        NOTE: if tie_break_policy="true-random", repeated runs may have slightly different
        results due to difference in broken ties


        Parameters
        ----------
        L
            An [n,m] matrix with values in {-1,0,1,...,k-1}
        return_probs
            Whether to return probs along with preds
        tie_break_policy
            Policy to break ties when converting probabilistic labels to predictions

        Returns
        -------
        np.ndarray
            An [n,1] array of integer labels

        (np.ndarray, np.ndarray)
            An [n,1] array of integer labels and an [n,k] array of probabilistic labels
        """
        Y_probs: NDArray[np.float_] = self.predict_proba(L)
        Y_p: NDArray[np.int_] = probs_to_preds(Y_probs, tie_break_policy)
        if return_probs:
            return (Y_p, Y_probs)
        return Y_p

    def score(
        self,
        L: NDArray[np.int_],
        Y: NDArray[np.int_],
        metrics: List[str] = ["accuracy"],
        tie_break_policy: TieBreakPolicy = "abstain",
    ) -> Dict[str, float]:
        """Calculate one or more scores from user-specified and/or user-defined metrics.

        Parameters
        ----------
        L
            An [n,m] matrix with values in {-1,0,1,...,k-1}
        Y
            Gold labels associated with data points in L
        metrics
            A list of metric names
        tie_break_policy
            Policy to break ties when converting probabilistic labels to predictions


        Returns
        -------
        Dict[str, float]
            A dictionary mapping metric names to metric scores
        """
        if tie_break_policy == "abstain":
            logging.warning("Metrics calculated over data points with non-abstain labels only")
        Y_pred, Y_prob = self.predict(L, return_probs=True, tie_break_policy=tie_break_policy)
        scorer = Scorer(metrics=metrics)
        results: Dict[str, float] = scorer.score(Y, Y_pred, Y_prob)
        return results

    def save(self, destination: Union[str, PathLike[str]]) -> None:
        """Save label model.

        Parameters
        ----------
        destination
            Filename for saving model

        Example
        -------
        >>> label_model.save('./saved_label_model.pkl')  # doctest: +SKIP
        """
        f = open(destination, "wb")
        pickle.dump(self.__dict__, f)
        f.close()

    def load(self, source: Union[str, PathLike[str]]) -> None:
        """Load existing label model.

        Parameters
        ----------
        source
            Filename to load model from

        Example
        -------
        Load parameters saved in ``saved_label_model``

        >>> label_model.load('./saved_label_model.pkl')  # doctest: +SKIP
        """
        f = open(source, "rb")
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)