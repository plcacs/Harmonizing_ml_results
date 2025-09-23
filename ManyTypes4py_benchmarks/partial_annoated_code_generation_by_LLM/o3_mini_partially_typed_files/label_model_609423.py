#!/usr/bin/env python3
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
    prec_init: Union[float, np.ndarray, List[float]] = 0.7
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
    """

    def __init__(self, cardinality: int = 2, **kwargs: Any) -> None:
        super().__init__()
        self.config: LabelModelConfig = LabelModelConfig(**kwargs)
        self.cardinality: int = cardinality
        if self.config.device != 'cpu' and (not torch.cuda.is_available()):
            raise ValueError('device=cuda but CUDA not available.')
        self.eval()

    def _create_L_ind(self, L: np.ndarray) -> np.ndarray:
        """Convert a label matrix with labels in 0...k to a one-hot format."""
        self.n = L.shape[0]
        # self.m will be set later in _set_constants if needed
        L_ind: np.ndarray = np.zeros((self.n, self.m * self.cardinality))
        for y in range(1, self.cardinality + 1):
            L_ind[:, y - 1::self.cardinality] = np.where(L == y, 1, 0)
        return L_ind

    def _get_augmented_label_matrix(self, L: np.ndarray, higher_order: bool = False) -> np.ndarray:
        """Create augmented version of label matrix.

        In augmented version, each column is an indicator for whether a certain source or clique of sources
        voted in a certain pattern.
        """
        self.c_data: Dict[int, _CliqueData] = {}
        for i in range(self.m):
            self.c_data[i] = _CliqueData(
                start_index=i * self.cardinality,
                end_index=(i + 1) * self.cardinality,
                max_cliques=set([j for j in self.c_tree.nodes() if i in self.c_tree.nodes[j]['members']])
            )
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
        self.mask: torch.Tensor = torch.ones(self.d, self.d, dtype=torch.bool)
        for ci in self.c_data.values():
            si = ci.start_index
            ei = ci.end_index
            for cj in self.c_data.values():
                (sj, ej) = (cj.start_index, cj.end_index)
                if len(ci.max_cliques.intersection(cj.max_cliques)) > 0:
                    self.mask[si:ei, sj:ej] = 0
                    self.mask[sj:ej, si:ei] = 0

    def _generate_O(self, L: np.ndarray, higher_order: bool = False) -> None:
        """Generate overlaps and conflicts matrix from label matrix."""
        L_aug: np.ndarray = self._get_augmented_label_matrix(L, higher_order=higher_order)
        self.d: int = L_aug.shape[1]
        self.O: torch.Tensor = torch.from_numpy(L_aug.T @ L_aug / self.n).float().to(self.config.device)

    def _init_params(self) -> None:
        """Initialize the learned params."""
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
                mu_init_val: torch.Tensor = torch.clamp(lps[idx] * self._prec_init[i] / self.p[y], 0, 1)
                self.mu_init[idx, y] += mu_init_val
        self.mu: nn.Parameter = nn.Parameter(self.mu_init.clone() * np.random.random())
        self.mu = self.mu.float()
        self._build_mask()

    def _get_conditional_probs(self, mu: np.ndarray) -> np.ndarray:
        """Return the estimated conditional probabilities table given parameters mu."""
        cprobs: np.ndarray = np.zeros((self.m, self.cardinality + 1, self.cardinality))
        for i in range(self.m):
            mu_i: np.ndarray = mu[i * self.cardinality:(i + 1) * self.cardinality, :]
            cprobs[i, 1:, :] = mu_i
            cprobs[i, 0, :] = 1 - mu_i.sum(axis=0)
        return cprobs

    def get_conditional_probs(self) -> np.ndarray:
        """Return the estimated conditional probabilities table."""
        return self._get_conditional_probs(self.mu.cpu().detach().numpy())

    def get_weights(self) -> np.ndarray:
        """Return the vector of learned LF weights for combining LFs."""
        accs: np.ndarray = np.zeros(self.m)
        cprobs: np.ndarray = self.get_conditional_probs()
        for i in range(self.m):
            accs[i] = np.diag(cprobs[i, 1:, :] @ self.P.cpu().detach().numpy()).sum()
        return np.clip(accs / self.coverage, 1e-06, 1.0)

    def predict_proba(self, L: np.ndarray) -> np.ndarray:
        """Return label probabilities P(Y | \\lambda)."""
        L_shift: np.ndarray = L + 1
        self._set_constants(L_shift)
        L_aug: np.ndarray = self._get_augmented_label_matrix(L_shift)
        mu: np.ndarray = self.mu.cpu().detach().numpy()
        jtm: np.ndarray = np.ones(L_aug.shape[1])
        X: np.ndarray = np.exp(L_aug @ np.diag(jtm) @ np.log(mu) + np.log(self.p))
        Z: np.ndarray = np.tile(X.sum(axis=1).reshape(-1, 1), self.cardinality)
        return X / Z

    def predict(self, 
                L: np.ndarray, 
                return_probs: Optional[bool] = False, 
                tie_break_policy: str = 'abstain'
               ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Return predicted labels, with ties broken according to policy."""
        return super(LabelModel, self).predict(L, return_probs, tie_break_policy)

    def score(self, 
              L: np.ndarray, 
              Y: np.ndarray, 
              metrics: Optional[List[str]] = ['accuracy'], 
              tie_break_policy: str = 'abstain'
             ) -> Dict[str, float]:
        """Calculate one or more scores from user-specified and/or user-defined metrics."""
        return super(LabelModel, self).score(L, Y, metrics, tie_break_policy)

    def _loss_l2(self, l2: float = 0) -> torch.Tensor:
        """L2 loss centered around mu_init, scaled optionally per-source."""
        if isinstance(l2, (int, float)):
            D: torch.Tensor = l2 * torch.eye(self.d)
        else:
            D = torch.diag(torch.from_numpy(l2)).type(torch.float32)
        D = D.to(self.config.device)
        return torch.norm(D @ (self.mu - self.mu_init)) ** 2

    def _loss_mu(self, l2: float = 0) -> torch.Tensor:
        """Overall mu loss."""
        loss_1: torch.Tensor = torch.norm((self.O - self.mu @ self.P @ self.mu.t())[self.mask]) ** 2
        loss_2: torch.Tensor = torch.norm(torch.sum(self.mu @ self.P, 1) - torch.diag(self.O)) ** 2
        return loss_1 + loss_2 + self._loss_l2(l2=l2)

    def _set_class_balance(self, 
                           class_balance: Optional[List[float]], 
                           Y_dev: Optional[np.ndarray] = None
                          ) -> None:
        """Set a prior for the class balance."""
        if class_balance is not None:
            self.p: np.ndarray = np.array(class_balance)
            if len(self.p) != self.cardinality:
                raise ValueError(f'class_balance has {len(self.p)} entries. Does not match LabelModel cardinality {self.cardinality}.')
        elif Y_dev is not None:
            class_counts: Counter = Counter(Y_dev)
            sorted_counts: np.ndarray = np.array([v for (k, v) in sorted(class_counts.items())])
            self.p = sorted_counts / sum(sorted_counts)
            if len(self.p) != self.cardinality:
                raise ValueError(f'Y_dev has {len(self.p)} class(es). Does not match LabelModel cardinality {self.cardinality}.')
        else:
            self.p = 1 / self.cardinality * np.ones(self.cardinality)
        if np.any(self.p == 0):
            raise ValueError(f'Class balance prior is 0 for class(es) {np.where(self.p)[0]}.')
        self.P: torch.Tensor = torch.diag(torch.from_numpy(self.p)).float().to(self.config.device)

    def _set_constants(self, L: np.ndarray) -> None:
        """Set basic constants from input label matrix."""
        self.n, self.m = L.shape  # type: int, int
        if self.m < 3:
            raise ValueError('L_train should have at least 3 labeling functions')
        self.t: int = 1

    def _create_tree(self) -> None:
        """Create clique tree from LF indices."""
        nodes: List[int] = list(range(self.m))
        self.c_tree: Any = get_clique_tree(nodes, [])

    def _execute_logging(self, loss: torch.Tensor) -> Metrics:
        """Execute logging if necessary."""
        self.eval()
        # Initialize running loss and examples if not already set
        if not hasattr(self, 'running_loss'):
            self.running_loss: float = 0.0
        if not hasattr(self, 'running_examples'):
            self.running_examples: int = 0
        self.running_loss += loss.item()
        self.running_examples += 1
        metrics_dict: Metrics = {'train/loss': self.running_loss / self.running_examples}
        if self.logger.check():
            if self.config.verbose:
                self.logger.log(metrics_dict)
            self.running_loss = 0.0
            self.running_examples = 0
        self.train()
        return metrics_dict

    def _set_logger(self) -> None:
        """Set up the logger."""
        self.logger: Logger = Logger(self.train_config.log_freq)
        if self.config.verbose:
            logging.basicConfig(level=logging.INFO)

    def _set_optimizer(self) -> None:
        """Initialize the optimizer."""
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        optimizer_config: OptimizerConfig = self.train_config.optimizer_config
        optimizer_name: str = self.train_config.optimizer
        if optimizer_name == 'sgd':
            self.optimizer: optim.Optimizer = optim.SGD(parameters, lr=self.train_config.lr, weight_decay=self.train_config.l2, **optimizer_config.sgd_config._asdict())
        elif optimizer_name == 'adam':
            self.optimizer = optim.Adam(parameters, lr=self.train_config.lr, weight_decay=self.train_config.l2, **optimizer_config.adam_config._asdict())
        elif optimizer_name == 'adamax':
            self.optimizer = optim.Adamax(parameters, lr=self.train_config.lr, weight_decay=self.train_config.l2, **optimizer_config.adamax_config._asdict())
        else:
            raise ValueError(f"Unrecognized optimizer option '{optimizer_name}'")

    def _set_lr_scheduler(self) -> None:
        """Set up the learning rate scheduler."""
        self._set_warmup_scheduler()
        lr_scheduler_name: str = self.train_config.lr_scheduler
        lr_scheduler_config: LRSchedulerConfig = self.train_config.lr_scheduler_config
        if lr_scheduler_name == 'constant':
            self.lr_scheduler = None
        elif lr_scheduler_name == 'linear':
            total_steps: int = self.train_config.n_epochs
            linear_decay_func = lambda x: (total_steps - self.warmup_steps - x) / (total_steps - self.warmup_steps)
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, linear_decay_func)
        elif lr_scheduler_name == 'exponential':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, **lr_scheduler_config.exponential_config._asdict())
        elif lr_scheduler_name == 'step':
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, **lr_scheduler_config.step_config._asdict())
        else:
            raise ValueError(f"Unrecognized lr scheduler option '{lr_scheduler_name}'")

    def _set_warmup_scheduler(self) -> None:
        """Initialize warmup scheduler if applicable."""
        if self.train_config.lr_scheduler_config.warmup_steps:
            warmup_steps: int = self.train_config.lr_scheduler_config.warmup_steps
            if warmup_steps < 0:
                raise ValueError('warmup_steps must be greater or equal than 0.')
            warmup_unit: str = self.train_config.lr_scheduler_config.warmup_unit
            if warmup_unit == 'epochs':
                self.warmup_steps = int(warmup_steps)
            else:
                raise ValueError("LabelModel does not support any warmup_unit other than 'epochs'.")
            linear_warmup_func = lambda x: x / self.warmup_steps
            self.warmup_scheduler: optim.lr_scheduler.LambdaLR = optim.lr_scheduler.LambdaLR(self.optimizer, linear_warmup_func)
            if self.config.verbose:
                logging.info(f'Warmup {self.warmup_steps} steps.')
        elif self.train_config.lr_scheduler_config.warmup_percentage:
            warmup_percentage: float = self.train_config.lr_scheduler_config.warmup_percentage
            self.warmup_steps = int(warmup_percentage * self.train_config.n_epochs)
            linear_warmup_func = lambda x: x / self.warmup_steps
            self.warmup_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, linear_warmup_func)
            if self.config.verbose:
                logging.info(f'Warmup {self.warmup_steps} steps.')
        else:
            self.warmup_scheduler = None
            self.warmup_steps = 0

    def _update_lr_scheduler(self, step: int) -> None:
        """Update the lr scheduler based on current step."""
        if self.warmup_scheduler and step < self.warmup_steps:
            self.warmup_scheduler.step()
        elif self.lr_scheduler is not None:
            self.lr_scheduler.step()
            min_lr: Optional[float] = self.train_config.lr_scheduler_config.min_lr
            if min_lr and self.optimizer.param_groups[0]['lr'] < min_lr:
                self.optimizer.param_groups[0]['lr'] = min_lr

    def _clamp_params(self) -> None:
        """Clamp the values of the learned parameter vector."""
        if self.train_config.mu_eps is not None:
            mu_eps: float = self.train_config.mu_eps
        else:
            mu_eps = min(0.01, 1 / 10 ** np.ceil(np.log10(self.n)))
        self.mu.data = self.mu.clamp(mu_eps, 1 - mu_eps)

    def _break_col_permutation_symmetry(self) -> None:
        """Heuristically break column permutation symmetry."""
        mu: np.ndarray = self.mu.cpu().detach().numpy()
        P: np.ndarray = self.P.cpu().detach().numpy()
        (d, k) = mu.shape
        probs_sum: np.ndarray = sum([mu[i:i + k] for i in range(0, self.m * k, k)]) @ P
        munkres_solver: Munkres = Munkres()
        Z: np.ndarray = np.zeros([k, k])
        groups: DefaultDict[float, List[int]] = defaultdict(list)
        for (i, f) in enumerate(P.diagonal()):
            groups[np.around(f, 3)].append(i)
        for group in groups.values():
            if len(group) == 1:
                Z[group[0], group[0]] = 1.0
                continue
            probs_proj: np.ndarray = probs_sum[[[g] for g in group], group]
            permutation_pairs: List[Tuple[int, int]] = munkres_solver.compute(-probs_proj.T)
            for (i, j) in permutation_pairs:
                Z[group[i], group[j]] = 1.0
        self.mu = nn.Parameter(torch.Tensor(mu @ Z).to(self.config.device))

    def fit(self, 
            L_train: np.ndarray, 
            Y_dev: Optional[np.ndarray] = None, 
            class_balance: Optional[List[float]] = None, 
            progress_bar: bool = True, 
            **kwargs: Any
           ) -> None:
        """Train label model to estimate mu, the parameters used to combine LFs."""
        self.train_config: TrainConfig = merge_config(TrainConfig(), kwargs)
        random.seed(self.train_config.seed)
        np.random.seed(self.train_config.seed)
        torch.manual_seed(self.train_config.seed)
        self._set_logger()
        L_shift: np.ndarray = L_train + 1
        if L_shift.max() > self.cardinality:
            raise ValueError(f'L_train has cardinality {L_shift.max()}, cardinality={self.cardinality} passed in.')
        self._set_constants(L_shift)
        self._set_class_balance(class_balance, Y_dev)
        self._create_tree()
        lf_analysis: LFAnalysis = LFAnalysis(L_train)
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
        start_iteration: int = 0
        metrics_hist: Dict[str, float] = {}
        if progress_bar:
            epochs = trange(start_iteration, self.train_config.n_epochs, unit='epoch')
        else:
            epochs = range(start_iteration, self.train_config.n_epochs)
        for epoch in epochs:
            self.running_loss = 0.0
            self.running_examples = 0
            self.optimizer.zero_grad()
            loss: torch.Tensor = self._loss_mu(l2=self.train_config.l2)
            if torch.isnan(loss):
                msg: str = 'Loss is NaN. Consider reducing learning rate.'
                raise Exception(msg)
            loss.backward()
            self.optimizer.step()
            metrics_dict: Metrics = self._execute_logging(loss)
            metrics_hist.update(metrics_dict)
            self._update_lr_scheduler(epoch)
        if progress_bar:
            epochs.close()  # type: ignore
        self._clamp_params()
        self._break_col_permutation_symmetry()
        self.eval()
        if self.config.verbose:
            logging.info('Finished Training')
