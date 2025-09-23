import logging
import random
from collections import Counter, defaultdict
from itertools import chain
from typing import Any, DefaultDict, Dict, List, NamedTuple, Optional, Set, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from munkres import Munkres
from tqdm import trange
from snorkel.labeling.analysis import LFAnalysis
from snorkel.labeling.model.base_labeler import BaseLabeler
from snorkel.labeling.model.graph_utils import get_clique_tree
from snorkel.labeling.model.logger import Logger
from snorkel.types import Config
from snorkel.utils.config_utils import merge_config
from snorkel.utils.lr_schedulers import LRSchedulerConfig
from snorkel.utils.optimizers import OptimizerConfig
Metrics = Dict[str, float]

class TrainConfig(Config):
    """Settings for the fit() method of LabelModel.

    Parameters
    ----------
    n_epochs
        The number of epochs to train (where each epoch is a single optimization step)
    lr
        Base learning rate (will also be affected by lr_scheduler choice and settings)
    l2
        Centered L2 regularization strength
    optimizer
        Which optimizer to use (one of ["sgd", "adam", "adamax"])
    optimizer_config
        Settings for the optimizer
    lr_scheduler
        Which lr_scheduler to use (one of ["constant", "linear", "exponential", "step"])
    lr_scheduler_config
        Settings for the LRScheduler
    prec_init
        LF precision initializations / priors
    seed
        A random seed to initialize the random number generator with
    log_freq
        Report loss every this many epochs (steps)
    mu_eps
        Restrict the learned conditional probabilities to [mu_eps, 1-mu_eps]
    """
    n_epochs: int = 100
    lr: float = 0.01
    l2: float = 0.0
    optimizer: str = 'sgd'
    optimizer_config: OptimizerConfig = OptimizerConfig()
    lr_scheduler: str = 'constant'
    lr_scheduler_config: LRSchedulerConfig = LRSchedulerConfig()
    prec_init: float = 0.7
    seed: int = np.random.randint(1000000)
    log_freq: int = 10
    mu_eps: Optional[float] = None

class LabelModelConfig(Config):
    """Settings for the LabelModel initialization.

    Parameters
    ----------
    verbose
        Whether to include print statements
    device
        What device to place the model on ('cpu' or 'cuda:0', for example)
    """
    verbose: bool = True
    device: str = 'cpu'

class _CliqueData(NamedTuple):
    start_index: int
    end_index: int
    max_cliques: Set[int]

class LabelModel(nn.Module, BaseLabeler):
    """A model for learning the LF accuracies and combining their output labels.

    This class learns a model of the labeling functions' conditional probabilities
    of outputting the true (unobserved) label `Y`, `P(\\lf | Y)`, and uses this learned
    model to re-weight and combine their output labels.

    This class is based on the approach in [Training Complex Models with Multi-Task
    Weak Supervision](https://arxiv.org/abs/1810.02840), published in AAAI'19. In this
    approach, we compute the inverse generalized covariance matrix of the junction tree
    of a given LF dependency graph, and perform a matrix completion-style approach with
    respect to these empirical statistics. The result is an estimate of the conditional
    LF probabilities, `P(\\lf | Y)`, which are then set as the parameters of the label
    model used to re-weight and combine the labels output by the LFs.

    Currently this class uses a conditionally independent label model, in which the LFs
    are assumed to be conditionally independent given `Y`.

    Examples
    --------
    >>> label_model = LabelModel()
    >>> label_model = LabelModel(cardinality=3)
    >>> label_model = LabelModel(cardinality=3, device='cpu')
    >>> label_model = LabelModel(cardinality=3)

    Parameters
    ----------
    cardinality
        Number of classes, by default 2
    **kwargs
        Arguments for changing config defaults

    Raises
    ------
    ValueError
        If config device set to cuda but only cpu is available

    Attributes
    ----------
    cardinality
        Number of classes, by default 2
    config
        Training configuration
    seed
        Random seed
    """

    def __init__(self, cardinality: int = 2, **kwargs: Any) -> None:
        super().__init__()
        self.config: LabelModelConfig = LabelModelConfig(**kwargs)
        self.cardinality: int = cardinality
        if self.config.device != 'cpu' and (not torch.cuda.is_available()):
            raise ValueError('device=cuda but CUDA not available.')
        self.eval()

    def _create_L_ind(self, L: np.ndarray) -> np.ndarray:
        """Convert a label matrix with labels in 0...k to a one-hot format.

        Parameters
        ----------
        L
            An [n,m] label matrix with values in {0,1,...,k}

        Returns
        -------
        np.ndarray
            An [n,m*k] dense np.ndarray with values in {0,1}
        """
        L_ind: np.ndarray = np.zeros((self.n, self.m * self.cardinality))
        for y in range(1, self.cardinality + 1):
            L_ind[:, y - 1::self.cardinality] = np.where(L == y, 1, 0)
        return L_ind

    def _get_augmented_label_matrix(self, L: np.ndarray, higher_order: bool = False) -> np.ndarray:
        """Create augmented version of label matrix.

        In augmented version, each column is an indicator
        for whether a certain source or clique of sources voted in a certain
        pattern.

        Parameters
        ----------
        L
            An [n,m] label matrix with values in {0,1,...,k}
        higher_order
            Whether to include higher-order correlations (e.g. LF pairs) in matrix

        Returns
        -------
        np.ndarray
            An [n,m*k] dense matrix with values in {0,1}
        """
        self.c_data: Dict[int, _CliqueData] = {}
        for i in range(self.m):
            self.c_data[i] = _CliqueData(start_index=i * self.cardinality, end_index=(i + 1) * self.cardinality, max_cliques=set([j for j in self.c_tree.nodes() if i in self.c_tree.nodes[j]['members']]))
        L_ind: np.ndarray = self._create_L_ind(L)
        if higher_order:
            L_aug: np.ndarray = np.copy(L_ind)
            for item in chain(self.c_tree.nodes(), self.c_tree.edges()):
                if isinstance(item, int):
                    C = self.c_tree.nodes[item]
                elif isinstance(item, tuple):
                    C = self.c_tree[item[0]][item[1]]
                else:
                    raise ValueError(item)
                members: List[int] = list(C['members'])
                C['start_index'] = members[0] * self.cardinality
                C['end_index'] = (members[0] + 1) * self.cardinality
            return L_aug
        else:
            return L_ind

    def _build_mask(self) -> None:
        """Build mask applied to O^{-1}, O for the matrix approx constraint."""
        self.mask: torch.Tensor = torch.ones(self.d, self.d).bool()
        for ci in self.c_data.values():
            si: int = ci.start_index
            ei: int = ci.end_index
            for cj in self.c_data.values():
                sj: int = cj.start_index
                ej: int = cj.end_index
                if len(ci.max_cliques.intersection(cj.max_cliques)) > 0:
                    self.mask[si:ei, sj:ej] = 0
                    self.mask[sj:ej, si:ei] = 0

    def _generate_O(self, L: np.ndarray, higher_order: bool = False) -> None:
        """Generate overlaps and conflicts matrix from label matrix.

        Parameters
        ----------
        L
            An [n,m] label matrix with values in {0,1,...,k}
        higher_order
            Whether to include higher-order correlations (e.g. LF pairs) in matrix
        """
        L_aug: np.ndarray = self._get_augmented_label_matrix(L, higher_order=higher_order)
        self.d: int = L_aug.shape[1]
        self.O: torch.Tensor = torch.from_numpy(L_aug.T @ L_aug / self.n).float().to(self.config.device)

    def _init_params(self) -> None:
        """Initialize the learned params.

        - \\mu is the primary learned parameter, where each row corresponds to
        the probability of a clique C emitting a specific combination of labels,
        conditioned on different values of Y (for each column); that is:

            self.mu[i*self.cardinality + j, y] = P(\\lambda_i = j | Y = y)

        and similarly for higher-order cliques.

        Raises
        ------
        ValueError
            If prec_init shape does not match number of LFs
        """
        if isinstance(self.train_config.prec_init, (int, float)):
            self._prec_init: torch.Tensor = self.train_config.prec_init * torch.ones(self.m)
        elif isinstance(self.train_config.prec_init, np.ndarray):
            self._prec_init = torch.Tensor(self.train_config.prec_init)
        elif isinstance(self.train_config.prec_init, list):
            self._prec_init = torch.Tensor(self.train_config.prec_init)
        elif not isinstance(self.train_config.prec_init, torch.Tensor):
            raise TypeError(f'prec_init is of type {type(self.train_config.prec_init)} which is not supported currently.')
        if self._prec_init.shape[0] != self.m:
            raise ValueError(f'prec_init must have shape {self.m}.')
        lps: np.ndarray = torch.diag(self.O).cpu().detach().numpy()
        self.mu_init: torch.Tensor = torch.zeros(self.d, self.cardinality)
        for i in range(self.m):
            for y in range(self.cardinality):
                idx: int = i * self.cardinality + y
                mu_init: torch.Tensor = torch.clamp(lps[idx] * self._prec_init[i] / self.p[y], 0, 1)
                self.mu_init[idx, y] += mu_init
        self.mu: nn.Parameter = nn.Parameter(self.mu_init.clone() * np.random.random()).float()
        self._build_mask()

    def _get_conditional_probs(self, mu: np.ndarray) -> np.ndarray:
        """Return the estimated conditional probabilities table given parameters mu.

        Given a parameter vector mu, return the estimated conditional probabilites
        table cprobs, where cprobs is an (m, k+1, k)-dim np.ndarray with:

            cprobs[i, j, k] = P(\\lf_i = j-1 | Y = k)

        where m is the number of LFs, k is the cardinality, and cprobs includes the
        conditional abstain probabilities P(\\lf_i = -1 | Y = y).

        Parameters
        ----------
        mu
            An [m * k, k] np.ndarray with entries in [0, 1]

        Returns
        -------
        np.ndarray
            An [m, k + 1, k] np.ndarray conditional probabilities table.
        """
        cprobs: np.ndarray = np.zeros((self.m, self.cardinality + 1, self.cardinality))
        for i in range(self.m):
            mu_i: np.ndarray = mu[i * self.cardinality:(i + 1) * self.cardinality, :]
            cprobs[i, 1:, :] = mu_i
            cprobs[i, 0, :] = 1 - mu_i.sum(axis=0)
        return cprobs

    def get_conditional_probs(self) -> np.ndarray:
        """Return the estimated conditional probabilities table.

        Return the estimated conditional probabilites table cprobs, where cprobs is an
        (m, k+1, k)-dim np.ndarray with:

            cprobs[i, j, k] = P(\\lf_i = j-1 | Y = k)

        where m is the number of LFs, k is the cardinality, and cprobs includes the
        conditional abstain probabilities P(\\lf_i = -1 | Y = y).

        Returns
        -------
        np.ndarray
            An [m, k + 1, k] np.ndarray conditional probabilities table.
        """
        return self._get_conditional_probs(self.mu.cpu().detach().numpy())

    def get_weights(self) -> np.ndarray:
        """Return the vector of learned LF weights for combining LFs.

        Returns
        -------
        np.ndarray
            [m,1] vector of learned LF weights for combining LFs.

        Example
        -------
        >>> L = np.array([[1, 1, 1], [1, 1, -1], [-1, 0, 0], [0, 0, 0]])
        >>> label_model = LabelModel(verbose=False)
        >>> label_model.fit(L, seed=123)
        >>> np.around(label_model.get_weights(), 2)  # doctest: +SKIP
        array([0.99, 0.99, 0.99])
        """
        accs: np.ndarray = np.zeros(self.m)
        cprobs: np.ndarray = self.get_conditional_probs()
        for i in range(self.m):
            accs[i] = np.diag(cprobs[i, 1:, :] @ self.P.cpu().detach().numpy()).sum()
        return np.clip(accs / self.coverage, 1e-06, 1.0)

    def predict_proba(self, L: np.ndarray) -> np.ndarray:
        """Return label probabilities P(Y | \\lambda).

        Parameters
        ----------
        L
            An [n,m] matrix with values in {-1,0,1,...,k-1}f

        Returns
        -------
        np.ndarray
            An [n,k] array of probabilistic labels

        Example
        -------
        >>> L = np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1]])
        >>> label_model = LabelModel(verbose=False)
        >>> label_model.fit(L, seed=123)
        >>> np.around(label_model.predict_proba(L), 1)  # doctest: +SKIP
        array([[1., 0.],
               [0., 1.],
               [0., 1.]])
        """
        L_shift: np.ndarray = L + 1
        self._set_constants(L_shift)
        L_aug: np.ndarray = self._get_augmented_label_matrix(L_shift)
        mu: np.ndarray = self.mu.cpu().detach().numpy()
        jtm: np.ndarray = np.ones(L_aug.shape[1])
        X: np.ndarray = np.exp(L_aug @ np.diag(jtm) @ np.log(mu) + np.log(self.p))
        Z: np.ndarray = np.tile(X.sum(axis=1).reshape(-1, 1), self.cardinality)
        return X / Z

    def predict(self, L: np.ndarray, return_probs: Optional[bool] = False, tie_break_policy: str = 'abstain') -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Return predicted labels, with ties broken according to policy.

        Policies to break ties include:

        - "abstain": return an abstain vote (-1)
        - "true-random": randomly choose among the tied options
        - "random": randomly choose among tied option using deterministic hash

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


        Example
        -------
        >>> L = np.array([[0, 0, -1], [1, 1, -1], [0, 0, -1]])
        >>> label_model = LabelModel(verbose=False)
        >>> label_model.fit(L)
        >>> label_model.predict(L)
        array([0, 1, 0])
        """
        return super(LabelModel, self).predict(L, return_probs, tie_break_policy)

    def score(self, L: np.ndarray, Y: np.ndarray, metrics: Optional[List[str]] = None, tie_break_policy: str = 'abstain') -> Dict[str, float]:
        """Calculate one or more scores from user-specified and/or user-defined metrics.

        Parameters
        ----------
        L
            An [n,m] matrix with values in {-1,0,1,...,k-1}
        Y
            Gold labels associated with data points in L
        metrics
            A list of metric names. Possbile metrics are - `accuracy`, `coverage`,
            `precision`, `recall`, `f1`, `f1_micro`, `f1_macro`, `fbeta`,
            `matthews_corrcoef`, `roc_auc`. See `sklearn.metrics
            <https://scikit-learn.org/stable/mod