import logging
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union
import numpy as np
from snorkel.analysis import Scorer
from snorkel.utils import probs_to_preds


class BaseLabeler(ABC):
    """Abstract baseline label voter class."""

    def __init__(self, cardinality: int = 2, **kwargs: Any) -> None:
        self.cardinality = cardinality

    @abstractmethod
    def predict_proba(self, L: np.ndarray) -> np.ndarray:
        """Abstract method for predicting probabilistic labels given a label matrix.

        Parameters
        ----------
        L : np.ndarray
            An [n,m] matrix with values in {-1,0,1,...,k-1}

        Returns
        -------
        np.ndarray
            An [n,k] array of probabilistic labels
        """
        pass

    def predict(
        self, L: np.ndarray, return_probs: bool = False, tie_break_policy: str = 'abstain'
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Return predicted labels, with ties broken according to policy.

        Policies to break ties include:
        "abstain": return an abstain vote (-1)
        "true-random": randomly choose among the tied options
        "random": randomly choose among tied option using deterministic hash

        NOTE: if tie_break_policy="true-random", repeated runs may have slightly different
        results due to difference in broken ties

        Parameters
        ----------
        L : np.ndarray
            An [n,m] matrix with values in {-1,0,1,...,k-1}
        return_probs : bool
            Whether to return probs along with preds
        tie_break_policy : str
            Policy to break ties when converting probabilistic labels to predictions

        Returns
        -------
        np.ndarray
            An [n,1] array of integer labels

        Tuple[np.ndarray, np.ndarray]
            An [n,1] array of integer labels and an [n,k] array of probabilistic labels
        """
        Y_probs: np.ndarray = self.predict_proba(L)
        Y_p: np.ndarray = probs_to_preds(Y_probs, tie_break_policy)
        if return_probs:
            return Y_p, Y_probs
        return Y_p

    def score(
        self, L: np.ndarray, Y: np.ndarray, metrics: List[str] = ['accuracy'], tie_break_policy: str = 'abstain'
    ) -> Dict[str, float]:
        """Calculate one or more scores from user-specified and/or user-defined metrics.

        Parameters
        ----------
        L : np.ndarray
            An [n,m] matrix with values in {-1,0,1,...,k-1}
        Y : np.ndarray
            Gold labels associated with data points in L
        metrics : List[str]
            A list of metric names
        tie_break_policy : str
            Policy to break ties when converting probabilistic labels to predictions

        Returns
        -------
        Dict[str, float]
            A dictionary mapping metric names to metric scores
        """
        if tie_break_policy == 'abstain':
            logging.warning('Metrics calculated over data points with non-abstain labels only')
        Y_pred, Y_prob = self.predict(L, return_probs=True, tie_break_policy=tie_break_policy)
        scorer: Scorer = Scorer(metrics=metrics)
        results: Dict[str, float] = scorer.score(Y, Y_pred, Y_prob)
        return results

    def save(self, destination: str) -> None:
        """Save label model.

        Parameters
        ----------
        destination : str
            Filename for saving model

        Example
        -------
        >>> label_model.save('./saved_label_model.pkl')  # doctest: +SKIP
        """
        with open(destination, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, source: str) -> None:
        """Load existing label model.

        Parameters
        ----------
        source : str
            Filename to load model from

        Example
        -------
        Load parameters saved in ``saved_label_model``

        >>> label_model.load('./saved_label_model.pkl')  # doctest: +SKIP
        """
        with open(source, 'rb') as f:
            tmp_dict: Dict[str, Any] = pickle.load(f)
        self.__dict__.update(tmp_dict)