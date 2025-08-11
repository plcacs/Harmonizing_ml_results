import logging
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from snorkel.analysis import Scorer
from snorkel.utils import probs_to_preds

class BaseLabeler(ABC):
    """Abstract baseline label voter class."""

    def __init__(self, cardinality: int=2, **kwargs) -> None:
        self.cardinality = cardinality

    @abstractmethod
    def predict_proba(self, L: Union[list[str], list, list[list[int]]]) -> None:
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

    def predict(self, L: Union[str, numpy.ndarray, bool, None], return_probs: bool=False, tie_break_policy: typing.Text='abstain') -> Union[tuple[typing.Union[float,list[list[int]],list[float],tuple[typing.Union[float,int]]]], float, list[list[int]]]:
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
        Y_probs = self.predict_proba(L)
        Y_p = probs_to_preds(Y_probs, tie_break_policy)
        if return_probs:
            return (Y_p, Y_probs)
        return Y_p

    def score(self, L: Union[numpy.ndarray, str, list[str], None], Y: Union[list, numpy.ndarray, str], metrics: list[typing.Text]=['accuracy'], tie_break_policy: typing.Text='abstain') -> Union[list, dict, dict[str, typing.Any]]:
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
        if tie_break_policy == 'abstain':
            logging.warning('Metrics calculated over data points with non-abstain labels only')
        Y_pred, Y_prob = self.predict(L, return_probs=True, tie_break_policy=tie_break_policy)
        scorer = Scorer(metrics=metrics)
        results = scorer.score(Y, Y_pred, Y_prob)
        return results

    def save(self, destination: str) -> None:
        """Save label model.

        Parameters
        ----------
        destination
            Filename for saving model

        Example
        -------
        >>> label_model.save('./saved_label_model.pkl')  # doctest: +SKIP
        """
        f = open(destination, 'wb')
        pickle.dump(self.__dict__, f)
        f.close()

    def load(self, source: Union[str, bytes]) -> None:
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
        f = open(source, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)