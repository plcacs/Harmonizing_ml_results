import logging
import random
from collections import Counter, defaultdict
from itertools import chain
from typing import Any, DefaultDict, Dict, List, NamedTuple, Optional, Set, Tuple, Union, Callable
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
    prec_init: Union[float, np.ndarray, List[float], torch.Tensor] = 0.7
    seed: int = np.random.randint(1000000.0)
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
        L_ind = np.zeros((self.n, self.m * self.cardinality))
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
        L_ind = self._create_L_ind(L)
        if higher_order:
            L_aug = np.copy(L_ind)
            for item in chain(self.c_tree.nodes(), self.c_tree.edges()):
                if isinstance(item, int):
                    C = self.c_tree.nodes[item]
                elif isinstance(item, tuple):
                    C = self.c_tree[item[0]][item[1]]
                else:
                    raise ValueError(item)
                members = list(C['members'])
                C['start_index'] = members[0] * self.cardinality
                C['end_index'] = (members[0] + 1) * self.cardinality
            return L_aug
        else:
            return L_ind

    def _build_mask(self) -> None:
        """Build mask applied to O^{-1}, O for the matrix approx constraint."""
        self.mask: torch.Tensor = torch.ones(self.d, self.d).bool()
        for ci in self.c_data.values():
            si = ci.start_index
            ei = ci.end_index
            for cj in self.c_data.values():
                sj, ej = (cj.start_index, cj.end_index)
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
        L_aug = self._get_augmented_label_matrix(L, higher_order=higher_order)
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
        lps = torch.diag(self.O).cpu().detach().numpy()
        self.mu_init: torch.Tensor = torch.zeros(self.d, self.cardinality)
        for i in range(self.m):
            for y in range(self.cardinality):
                idx = i * self.cardinality + y
                mu_init = torch.clamp(lps[idx] * self._prec_init[i] / self.p[y], 0, 1)
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
        cprobs = np.zeros((self.m, self.cardinality + 1, self.cardinality))
        for i in range(self.m):
            mu_i = mu[i * self.cardinality:(i + 1) * self.cardinality, :]
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
        accs = np.zeros(self.m)
        cprobs = self.get_conditional_probs()
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
        L_shift = L + 1
        self._set_constants(L_shift)
        L_aug = self._get_augmented_label_matrix(L_shift)
        mu = self.mu.cpu().detach().numpy()
        jtm = np.ones(L_aug.shape[1])
        X = np.exp(L_aug @ np.diag(jtm) @ np.log(mu) + np.log(self.p))
        Z = np.tile(X.sum(axis=1).reshape(-1, 1), self.cardinality)
        return X / Z

    def predict(self, L: np.ndarray, return_probs: bool = False, tie_break_policy: str = 'abstain') -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
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

    def score(self, L: np.ndarray, Y: np.ndarray, metrics: List[str] = ['accuracy'], tie_break_policy: str = 'abstain') -> Dict[str, float]:
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
            <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics>`_
            for details on the metrics.
        tie_break_policy
            Policy to break ties when converting probabilistic labels to predictions.
            Same as :func:`.predict` method above.


        Returns
        -------
        Dict[str, float]
            A dictionary mapping metric names to metric scores

        Example
        -------
        >>> L = np.array([[1, 1, -1], [0, 0, -1], [1, 1, -1]])
        >>> label_model = LabelModel(verbose=False)
        >>> label_model.fit(L)
        >>> label_model.score(L, Y=np.array([1, 1, 1]))
        {'accuracy': 0.6666666666666666}
        >>> label_model.score(L, Y=np.array([1, 1, 1]), metrics=["f1"])
        {'f1': 0.8}
        """
        return super(LabelModel, self).score(L, Y, metrics, tie_break_policy)

    def _loss_l2(self, l2: Union[float, np.ndarray] = 0) -> torch.Tensor:
        """L2 loss centered around mu_init, scaled optionally per-source.

        In other words, diagonal Tikhonov regularization,
            ||D(\\mu-\\mu_{init})||_2^2
        where D is diagonal.

        Parameters
        ----------
        l2
            A float or np.array representing the per-source regularization
            strengths to use, by default 0

        Returns
        -------
        torch.Tensor
            L2 loss between learned mu and initial mu
        """
        if isinstance(l2, (int, float)):
            D = l2 * torch.eye(self.d)
        else:
            D = torch.diag(torch.from_numpy(l2)).type(torch.float32)
        D = D.to(self.config.device)
        return torch.norm(D @ (self.mu - self.mu_init)) ** 2

    def _loss_mu(self, l2: Union[float, np.ndarray] = 0) -> torch.Tensor:
        """Overall mu loss.

        Parameters
        ----------
        l2
            A float or np.array representing the per-source regularization
                strengths to use, by default 0

        Returns
        -------
        torch.Tensor
            Overall mu loss between learned mu and initial mu
        """
        loss_1 = torch.norm((self.O - self.mu @ self.P @ self.mu.t())[self.mask]) ** 2
        loss_2 = torch.norm(torch.sum(self.mu @ self.P, 1) - torch.diag(self.O)) ** 2
        return loss_1 + loss_2 + self._loss_l2(l2=l2)

    def _set_class_balance(self, class_balance: Optional[List[float]], Y_dev: Optional[np.ndarray] = None) -> None:
        """Set a prior for the class balance.

        In order of preference:
        1) Use user-provided class_balance
        2) Estimate balance from Y_dev
        3) Assume uniform class distribution
        """
        if class_balance is not None:
            self.p: np.ndarray = np.array(class_balance)
            if len(self.p) != self.cardinality:
                raise ValueError(f'class_balance has {len(self.p)} entries. Does not match LabelModel cardinality {self.cardinality}.')
        elif Y_dev is not None:
            class_counts = Counter(Y_dev)
            sorted_counts = np.array([v for k, v in sorted(class_counts.items())])
            self.p = sorted_counts / sum(sorted_counts)
            if len(self.p) != self.cardinality:
                raise ValueError(f'Y_dev has {len(self.p)} class(es). Does not match LabelModel cardinality {self.cardinality}.')
        else:
            self.p = 1 / self.cardinality * np.ones(self.cardinality)
        if np.any(self.p == 0):
            raise ValueError(f'Class balance prior is 0 for class(es) {np.where(self.p)[0]}.')
        self.P: torch.Tensor = torch.diag(torch.from_numpy(self.p)).float().to(self.config.device)

    def _set_constants(self, L: np.ndarray) -> None:
        self.n: int
        self.m: int
        self.n, self.m = L.shape
        if self.m < 3:
            raise ValueError('L_train should have at least 3 labeling functions')
        self.t: int = 1

    def _create_tree(self) -> None:
        nodes = range(self.m)
        self.c_tree = get_clique_tree(nodes, [])

    def _execute_logging(self, loss: torch.Tensor) -> Dict[str, float]:
        self.eval()
        self.running_loss += loss.item()
        self.running_examples += 1
        metrics_dict: Dict[str, float] = {'train/loss': self.running_loss / self.running_examples}
        if self.logger.check():
            if self.config.verbose:
                self.logger.log(metrics_dict)
            self.running_loss = 0.0
            self.running_examples = 0
        self.train()
        return metrics_dict

    def _set_logger(self) -> None:
        self.logger: Logger = Logger(self.train_config.log_freq)
        if self.config.verbose:
            logging.basicConfig(level=logging.INFO)

    def _set_optimizer(self) -> None:
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        optimizer_config = self.train_config.optimizer_config
        optimizer_name = self.train_config.optimizer
        if optimizer_name == 'sgd':
            optimizer = optim.SGD(parameters, lr=self.train_config.lr, weight_decay=self.train_config.l2, **optimizer_config.sgd_config._asdict())
        elif optimizer_name == 'adam':
            optimizer = optim.Adam(parameters, lr=self.train_config.lr, weight_decay=self.train_config.l2, **optimizer_config.adam_config._asdict())
        elif optimizer_name == 'adamax':
            optimizer = optim.Adamax(parameters, lr=self.train_config.lr, weight_decay=self.train_config.l2, **optimizer_config.adamax_config._asdict())
        else:
            raise ValueError(f"Unrecognized optimizer option '{optimizer_name}'")
        self.optimizer: optim.Optimizer = optimizer

    def _set_lr_scheduler(self) -> None:
        self._set_warmup_scheduler()
        lr_scheduler_name = self.train_config.lr_scheduler
        lr_scheduler_config = self.train_config.lr_scheduler_config
        if lr_scheduler_name == 'constant':
            lr_scheduler = None
        elif lr_scheduler_name == 'linear':
            total_steps = self.train_config.n_epochs
            linear_decay_func: Callable[[int], float] = lambda x: (total_steps - self.warmup_steps - x) / (total_steps - self.warmup_steps)
            lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, linear_decay_func)
        elif lr_scheduler_name == 'exponential':
            lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, **lr_scheduler_config.exponential_config._asdict())
        elif lr_scheduler_name == 'step':
            lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, **lr_scheduler_config.step_config._asdict())
        else:
            raise ValueError(f"Unrecognized lr scheduler option '{lr_scheduler_name}'")
        self.lr_scheduler: Optional[optim.lr_scheduler._LRScheduler] = lr_scheduler

    def _set_warmup_scheduler(self) -> None:
        if self.train_config.lr_scheduler_config.warmup_steps:
            warmup_steps = self.train_config.lr_scheduler_config.warmup_steps
            if warmup_steps < 0:
                raise ValueError('warmup_steps much greater or equal than 0.')
            warmup_unit = self.train_config.lr_scheduler_config.warmup_unit
            if warmup_unit == 'epochs':
                self.warmup_steps: int = int(warmup_steps)
            else:
                raise ValueError("LabelModel does not support any warmup_unit other than 'epochs'.")
            linear_warmup_func: Callable[[int], float] = lambda x: x / self.warmup_steps
            warmup_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, linear_warmup_func)
            if self.config.verbose:
                logging.info(f'Warmup {self.warmup_steps} steps.')
        elif self.train_config.lr_scheduler_config.warmup_percentage:
            warmup_percentage = self.train_config.lr_scheduler_config.warmup_percentage
            self.warmup_steps = int(warmup_percentage * self.train_config.n_epochs)
            linear_warmup_func = lambda x: x / self.warmup_steps
            warmup_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, linear_warmup_func)
            if self.config.verbose:
                logging.info(f'Warmup {self.warmup_steps} steps.')
        else:
            warmup_scheduler = None
            self.warmup_steps = 0
        self.warmup_scheduler: Optional[optim.lr_scheduler._LRScheduler] = warmup_scheduler

    def _update_lr_scheduler(self, step: int) -> None:
        if self.warmup_scheduler and step < self.warmup_steps:
            self.warmup_scheduler.step()
        elif self.lr_scheduler is not None:
            self.lr_scheduler.step()
            min_lr = self.train_config.lr_scheduler_config.min_lr
            if min_lr and self.optimizer.param_groups[0]['lr'] < min_lr:
                self.optimizer.param_groups[0]['lr'] = min_lr

    def _clamp_params(self) -> None:
        """Clamp the values of the learned parameter vector.

        Clamp the entries of self.mu to be in [mu_eps, 1 - mu_eps], where mu_eps is
        either set by the user, or defaults to 1 / 10 ** np.ceil(np.log10(self.n)).

        Note that if mu_eps is set too high, e.g. in sparse settings where LFs
        mostly abstain, this will result in learning conditional probabilities all
        equal to mu_eps (and/or 1 - mu_eps)!  See issue #1422.

        Note: Use user-provided value of mu_eps in train_config, else default to
            mu_eps = 1 / 10 ** np.ceil(np.log10(self.n))
        this rounding is done to make it more obvious when the parameters have been
        clamped.
        """
        if self.train_config.mu_eps is not None:
            mu_eps = self.train_config.mu_eps
        else:
            mu_eps = min(0.01, 1 / 10 ** np.ceil(np.log10(self.n)))
        self.mu.data = self.mu.clamp(mu_eps, 1 - mu_eps)

    def _break_col_permutation_symmetry(self) -> None:
        """Heuristically choose amongst (possibly) several valid mu values.

        If there are several values of mu that equivalently satisfy the optimization
        objective, as there often are due to column permutation symmetries, then pick
        the solution that trusts the user-written LFs most.

        In more detail, suppose that mu satisfies (minimizes) the two loss objectives:
            1. O = mu @ P @ mu.T
            2. diag(O) = sum(mu @ P, axis=1)
        Then any column permutation matrix Z that commutes with P will also equivalently
        satisfy these objectives, and thus is an equally valid (symmetric) solution.
        Therefore, we select the solution that maximizes the summed probability of the
        LFs being accurate when not abstaining.

            \\sum_lf \\sum_{y=1}^{cardinality} P(\\lf = y, Y = y)
        """
        mu = self.mu.cpu().detach().numpy()
        P = self.P.cpu().detach().numpy()
        d, k = mu.shape
        probs_sum = sum([mu[i:i + k] for i in range(0, self.m * k, k)]) @ P
        munkres_solver = Munkres()
        Z = np.zeros([k, k])
        groups: DefaultDict[float, List[int]] = defaultdict(list)
        for i, f in enumerate(P.diagonal()):
            groups[np.around(f, 3)].append(i)
        for group in groups.values():
            if len(group) == 1:
                Z[group[0], group[0]] = 1.0
                continue
            probs_proj = probs_sum[[[g] for g in group], group]
            permutation_pairs = munkres_solver.compute(-probs_proj.T)
            for i, j in permutation_pairs:
                Z[group[i], group[j]] = 1.0
        self.mu = nn.Parameter(torch.Tensor(mu @ Z).to(self.config.device))

    def fit(self, L_train: np.ndarray, Y_dev: Optional[np.ndarray] = None, class_balance: Optional[List[float]] = None, progress_bar: bool = True, **kwargs: Any) -> None:
        """Train label model.

        Train label model to estimate mu, the parameters used to combine LFs.

        Parameters
        ----------
        L_train
            An [n,m] matrix with values in {-1,0,1,...,k-1}
        Y_dev
            Gold labels for dev set for estimating class_balance, by default None
        class_balance
            Each class's percentage of the population, by default None
        progress_bar
            To display a progress bar, by default True
        **kwargs
            Arguments for changing train config defaults.

            n_epochs
                The number of epochs to train (where each epoch is a single
                optimization step), default is 100
            lr
                Base learning rate (will also be affected by lr_scheduler choice
                and settings), default is 0.01
            l2
                Centered L2 regularization strength, default is 0.0
            optimizer
                Which optimizer to use (one of ["sgd", "adam", "adamax"]),
                default is "sgd"
            optimizer_config
                Settings for the optimizer
            lr_scheduler
                Which lr_scheduler to use (one of ["constant", "linear",
                "exponential", "step"]), default is "constant"
            lr_scheduler_config
                Settings for the LRScheduler
            prec_init
                LF precision initializations / priors, default is 0.7
            seed
                A random seed to initialize the random number generator with
            log_freq
                Report loss every this many epochs (steps), default is 10
            mu_eps
                Restrict the learned conditional probabilities to
                [mu_eps, 1-mu_eps], default is None

        Raises
        ------
        Exception
            If loss in NaN

        Examples
        --------
        >>> L = np.array([[0, 0, -1], [-1, 0, 1], [1, -1, 0]])
        >>> Y_dev = [0, 1, 0]
        >>> label_model = LabelModel(verbose=False)
        >>> label_model.fit(L)
        >>> label_model.fit(L, Y_dev=Y_dev, seed=2020, lr=0.05)
        >>> label_model.fit(L, class_balance=[0.7, 0.3], n_epochs=200, l2=0.4)
        """
        self.train_config: TrainConfig = merge_config(TrainConfig(), kwargs)
        random.seed(self.train_config.seed)
        np.random.seed(self.train_config.seed)
        torch.manual_seed(self.train_config.seed)
        self._set_logger()
        L_shift = L_train + 1
        if L_shift.max() > self.cardinality:
            raise ValueError(f'L_train has cardinality {L_shift.max()}, cardinality={self.cardinality} passed in.')
        self._set_constants(L_shift)
        self._set_class_balance(class_balance, Y_dev)
        self._create_tree()
        lf_analysis = LFAnalysis(L_train)
        self.coverage: np.ndarray = lf_analysis.lf_coverages()
        if self.config.verbose:
            logging.info('Computing O...')
        self._generate_O(L_shift)
        self._init_params()
        if self.config.verbose:
            logging.info('Estimating \\mu...')
        self.train()
        self.mu_init = self.mu_init.to(self.config.device)
        if self.config.verbose and self.config.device != 'cpu':
            logging.info('Using GPU...')
        self.to(self.config.device)
        self._set_optimizer()
        self._set_lr_scheduler()
        start_iteration = 0
        metrics_hist: Dict[str, float] = {}
        self.running_loss: float = 0.0
        self.running_examples: int = 0
        if progress_bar:
            epochs = trange(start_iteration, self.train_config.n_epochs, unit='epoch')
        else:
            epochs = range(start_iteration, self.train_config.n_epochs)
        for epoch in epochs:
            self.running_loss = 0.0
            self.running_examples = 0
            self.optimizer.zero_grad()
            loss = self._loss_mu(l2=self.train_config.l2)
            if torch.isnan(loss):
                msg = 'Loss is NaN. Consider reducing learning rate.'
                raise Exception(msg)
            loss.backward()
            self.optimizer.step()
            metrics_dict = self._execute_logging(loss)
            metrics_hist.update(metrics_dict)
            self._update_lr_scheduler(epoch)
        if progress_bar:
            epochs.close()
        self._clamp_params()
        self._break_col_permutation_symmetry()
        self.eval()
        if self.config.verbose:
            logging.info('Finished Training')
