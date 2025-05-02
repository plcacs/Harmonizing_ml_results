import time
import pickle
import warnings
from pathlib import Path
from numbers import Real
from collections import deque
from typing import (
    Any, Callable, Dict, Deque, List, Optional, Set, Tuple, Type, TypeVar, Union,
    cast, Generic, overload
)
import numpy as np
import nevergrad.common.typing as tp
from nevergrad.parametrization import parameter as p
from nevergrad.common import tools as ngtools
from nevergrad.common import errors as errors
from nevergrad.common.decorators import Registry
from . import utils
from . import multiobjective as mobj

OptCls = tp.Union['ConfiguredOptimizer', Type['Optimizer']]
registry = Registry()
_OptimCallBack = tp.Union[
    tp.Callable[['Optimizer', 'p.Parameter', float], None],
    tp.Callable[['Optimizer'], None]
]
X = TypeVar('X', bound='Optimizer')
Y = TypeVar('Y')
IntOrParameter = tp.Union[int, p.Parameter]
_PruningCallable = tp.Callable[[utils.Archive[utils.MultiValue]], utils.Archive[utils.MultiValue]]

def _loss(param: p.Parameter) -> float:
    """Returns the loss if available, or inf otherwise.
    Used to simplify handling of losses
    """
    return param.loss if param.loss is not None else float('inf')

def load(cls: Type['Optimizer'], filepath: Union[str, Path]) -> 'Optimizer':
    """Loads a pickle file and checks that it contains an optimizer.
    The optimizer class is not always fully reliable though (e.g.: optimizer families) so the user is responsible for it.
    """
    filepath = Path(filepath)
    with filepath.open('rb') as f:
        opt = pickle.load(f)
    assert isinstance(opt, cls), f'You should only load {cls} with this method (found {type(opt)})'
    return opt

class Optimizer:
    """Algorithm framework with 3 main functions:

    - :code:`ask()` which provides a candidate on which to evaluate the function to optimize.
    - :code:`tell(candidate, loss)` which lets you provide the loss associated to points.
    - :code:`provide_recommendation()` which provides the best final candidate.

    Typically, one would call :code:`ask()` num_workers times, evaluate the
    function on these num_workers points in parallel, update with the fitness value when the
    evaluations is finished, and iterate until the budget is over. At the very end,
    one would call provide_recommendation for the estimated optimum.

    This class is abstract, it provides internal equivalents for the 3 main functions,
    among which at least :code:`_internal_ask_candidate` has to be overridden.

    Each optimizer instance should be used only once, with the initial provided budget
    """
    recast: bool = False
    one_shot: bool = False
    no_parallelization: bool = False

    def __init__(
        self,
        parametrization: Union[int, np.int_, p.Parameter],
        budget: Optional[int] = None,
        num_workers: int = 1
    ) -> None:
        if self.no_parallelization and num_workers > 1:
            raise ValueError(f'{self.__class__.__name__} does not support parallelization')
        self.num_workers: int = int(num_workers)
        self.budget: Optional[int] = budget
        self.optim_curve: List[Tuple[int, float]] = []
        self.skip_constraints: bool = False
        self._constraints_manager: utils.ConstraintManager = utils.ConstraintManager()
        self._penalize_cheap_violations: bool = False
        self.parametrization: p.Parameter = parametrization if not isinstance(parametrization, (int, np.int_)) else p.Array(shape=(parametrization,))
        self.parametrization.freeze()
        if not self.dimension:
            raise ValueError('No variable to optimize in this parametrization.')
        self.name: str = self.__class__.__name__
        self.archive: utils.Archive = utils.Archive()
        self.current_bests: Dict[str, utils.MultiValue] = {
            x: utils.MultiValue(self.parametrization, np.inf, reference=self.parametrization)
            for x in ['optimistic', 'pessimistic', 'average', 'minimum']
        }
        self.pruning: Optional[_PruningCallable] = utils.Pruning.sensible_default(
            num_workers=num_workers,
            dimension=self.parametrization.dimension
        )
        self._MULTIOBJECTIVE_AUTO_BOUND: float = mobj.AUTO_BOUND
        self._hypervolume_pareto: Optional[mobj.HypervolumePareto] = None
        self._asked: Set[str] = set()
        self._num_objectives: int = 0
        self._suggestions: Deque[p.Parameter] = deque()
        self._num_ask: int = 0
        self._num_tell: int = 0
        self._num_tell_not_asked: int = 0
        self._callbacks: Dict[str, List[Callable[..., None]] = {}
        self._running_jobs: List[Tuple[p.Parameter, Any]] = []
        self._finished_jobs: Deque[Tuple[p.Parameter, Any]] = deque()
        self._sent_warnings: Set[Type[Warning]] = set()
        self._no_hypervolume: bool = False

    def _warn(self, msg: str, e: Type[Warning]) -> None:
        """Warns only once per warning type"""
        if e not in self._sent_warnings:
            warnings.warn(msg, e)
            self._sent_warnings.add(e)

    @property
    def _rng(self) -> np.random.RandomState:
        """np.random.RandomState: parametrization random state the optimizer must pull from.
        It can be seeded or updated directly on the parametrization instance (`optimizer.parametrization.random_state`)
        """
        return self.parametrization.random_state

    @property
    def dimension(self) -> int:
        """int: Dimension of the optimization space."""
        return self.parametrization.dimension

    @property
    def num_objectives(self) -> int:
        """Provides 0 if the number is not known yet, else the number of objectives
        to optimize upon.
        """
        if self._hypervolume_pareto is not None and self._num_objectives != self._hypervolume_pareto.num_objectives:
            raise RuntimeError('Number of objectives is incorrectly set. Please create a nevergrad issue')
        return self._num_objectives

    @num_objectives.setter
    def num_objectives(self, num: int) -> None:
        num = int(num)
        if num <= 0:
            raise ValueError('Number of objectives must be strictly positive')
        if not self._num_objectives:
            self._num_objectives = num
            self._num_objectives_set_callback()
        elif num != self._num_objectives:
            raise ValueError(f'Expected {self._num_objectives} loss(es), but received {num}.')

    def _num_objectives_set_callback(self) -> None:
        """Callback for when num objectives is first known"""

    @property
    def num_ask(self) -> int:
        """int: Number of time the `ask` method was called."""
        return self._num_ask

    @property
    def num_tell(self) -> int:
        """int: Number of time the `tell` method was called."""
        return self._num_tell

    @property
    def num_tell_not_asked(self) -> int:
        """int: Number of time the :code:`tell` method was called on candidates that were not asked for by the optimizer
        (or were suggested).
        """
        return self._num_tell_not_asked

    def pareto_front(
        self,
        size: Optional[int] = None,
        subset: str = 'random',
        subset_tentatives: int = 12
    ) -> List[p.Parameter]:
        """Pareto front, as a list of Parameter. The losses can be accessed through
        parameter.losses"""
        pareto = [] if self._hypervolume_pareto is None else self._hypervolume_pareto.pareto_front(
            size=size,
            subset=subset,
            subset_tentatives=subset_tentatives
        )
        return pareto if pareto else [self.provide_recommendation()]

    def dump(self, filepath: Union[str, Path]) -> None:
        """Pickles the optimizer into a file."""
        filepath = Path(filepath)
        with filepath.open('wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls: Type[X], filepath: Union[str, Path]) -> X:
        """Loads a pickle and checks that the class is correct."""
        return load(cls, filepath)

    def __repr__(self) -> str:
        inststr = self.parametrization.name
        return f'Instance of {self.name}(parametrization={inststr}, budget={self.budget}, num_workers={self.num_workers})'

    def register_callback(self, name: str, callback: _OptimCallBack) -> None:
        """Add a callback method called either when `tell` or `ask` are called, with the same
        arguments (including the optimizer / self). This can be useful for custom logging."""
        assert name in ['ask', 'tell'], f'Only "ask" and "tell" methods can have callbacks (not {name})'
        self._callbacks.setdefault(name, []).append(callback)

    def remove_all_callbacks(self) -> None:
        """Removes all registered callables"""
        self._callbacks = {}

    def suggest(self, *args: Any, **kwargs: Any) -> None:
        """Suggests a new point to ask."""
        if isinstance(self.parametrization, p.Instrumentation):
            new_value = (args, kwargs)
        else:
            assert len(args) == 1 and (not kwargs)
            new_value = args[0]
        self._suggestions.append(self.parametrization.spawn_child(new_value=new_value))

    def tell(
        self,
        candidate: p.Parameter,
        loss: Union[float, List[float], np.ndarray],
        constraint_violation: Optional[Union[float, List[float], np.ndarray]] = None,
        penalty_style: Optional[List[float]] = None
    ) -> None:
        """Provides the optimizer with the evaluation of a fitness value for a candidate."""
        if isinstance(loss, (Real, float)) or (isinstance(loss, np.ndarray) and (not loss.shape)):
            loss = float(loss)
            if not loss < 5e+20:
                self._warn(f'Clipping very high value {loss} in tell (rescale the cost function?).', errors.LossTooLargeWarning)
                loss = 5e+20
        elif isinstance(loss, (tuple, list, np.ndarray)):
            loss = np.asarray(loss, dtype=float).ravel() if len(loss) != 1 else loss[0]
        elif not isinstance(loss, np.ndarray):
            raise TypeError(f'"tell" method only supports float values but the passed loss was: {loss} (type: {type(loss)}.')
        if isinstance(loss, float) and (len(self.optim_curve) == 0 or self.num_tell > self.optim_curve[-1][0] * 1.1):
            self.optim_curve += [(self.num_tell, loss)]
        if not isinstance(candidate, p.Parameter):
            raise TypeError("'tell' must be provided with the candidate.\nUse optimizer.parametrization.spawn_child(new_value)) if you want to create a candidate that as not been asked for, or optimizer.suggest(*args, **kwargs) to suggest a point that should be used for the next ask")
        self.num_objectives = 1 if isinstance(loss, float) else loss.size
        candidate.freeze()
        if isinstance(candidate, p.MultiobjectiveReference):
            if self._hypervolume_pareto is not None:
                raise RuntimeError('MultiobjectiveReference can only be provided before the first tell.')
            if not isinstance(loss, np.ndarray):
                raise RuntimeError('MultiobjectiveReference must only be used for multiobjective losses')
            self._hypervolume_pareto = mobj.HypervolumePareto(upper_bounds=loss, seed=self._rng, no_hypervolume=self._no_hypervolume)
            if candidate.value is None:
                return
            candidate = candidate.value
        if isinstance(loss, np.ndarray):
            candidate._losses = loss
        if not isinstance(loss, float):
            loss = self._preprocess_multiobjective(candidate)
        candidate.loss = loss
        assert isinstance(loss, float)
        for callback in self._callbacks.get('tell', []):
            callback(self, candidate, loss)
        no_update = False
        if not candidate.satisfies_constraints(self.parametrization) and self.budget is not None:
            penalty = self._constraints_manager.penalty(candidate, self.num_ask, self.budget)
            no_update = True
            loss = loss + penalty
        if constraint_violation is not None:
            if penalty_style is not None:
                a, b, c, d, e, f = penalty_style
            else:
                a, b, c, d, e, f = (100000.0, 1.0, 0.5, 1.0, 0.5, 1.0)
            ratio = 1 if self.budget is not None and self._num_tell > self.budget / 2.0 else 0.0
            iviolation = np.sum(np.maximum(constraint_violation, 0.0))
            if iviolation > 0.0:
                no_update = True
            violation = float((a * ratio + np.sum(np.maximum(loss, 0.0))) * (f + self._num_tell) ** e * (b * np.sum(np.maximum(constraint_violation, 0.0) ** c) ** d)
            loss += violation
        if isinstance(loss, float) and (self.num_objectives == 1 or (self.num_objectives > 1 and (not self._no_hypervolume))) and (not no_update):
            self._update_archive_and_bests(candidate, loss)
        if candidate.uid in self._asked:
            self._internal_tell_candidate(candidate, loss)
            self._asked.remove(candidate.uid)
        else:
            self._internal_tell_not_asked(candidate, loss)
            self._num_tell_not_asked += 1
        self._num_tell += 1

    def _preprocess_multiobjective(self, candidate: p.Parameter) -> float:
        if self._hypervolume_pareto is None:
            self._hypervolume_pareto = mobj.HypervolumePareto(auto_bound=self._MULTIOBJECTIVE_AUTO_BOUND, no_hypervolume=self._no_hypervolume)
        return self._hypervolume_pareto.add(candidate)

    def _update_archive_and_bests(self, candidate: p.Parameter, loss: float) -> None:
        x = candidate.get_standardized_data(reference=self.parametrization)
        if not isinstance(loss, (Real, float)):
            raise TypeError(f'"tell" method only supports float values but the passed loss was: {loss} (type: {type(loss)}.')
        if np.isnan(loss) or loss == np.inf:
            self._warn(f'Updating fitness with {loss} value', errors.BadLossWarning)
        mvalue = None
        if x not in self.archive:
            self.archive[x] = utils.MultiValue(candidate, loss, reference=self.parametrization)
        else:
            mvalue = self.archive[x]
            mvalue.add_evaluation(loss)
            if mvalue.parameter.loss > candidate.loss:
                mvalue.parameter = candidate
        for name in self.current_bests:
            if mvalue is self.current_bests[name]:
                best = min(self.archive.values(), key=lambda mv, n=name: mv.get_estimation(n))
                self.current_bests[name] = best
            elif self.archive[x].get_estimation(name) <= self.current_bests[name].get_estimation(name):
                self.current_bests[name] = self.archive[x]
        if self.pruning is not None:
            self.archive = self.pruning(self.archive)

    def ask(self) -> p.Parameter:
        """Provides a point to explore."""
        for callback in self._callbacks.get('ask', []):
            callback(self)
        current_num_ask = self.num_ask
        max_trials = max(1, self._constraints_manager.max_trials // 2)
        self.parametrization.tabu_fails = 0
        if self.skip_constraints and (not self._suggestions):
            candidate = self._internal_ask_candidate()
            is_suggestion = False
        else:
            for _ in range(max_trials):
                is_suggestion = False
                if self._suggestions:
                    is_suggestion = True
                    candidate = self._suggestions.pop()
                else:
                    try:
                        candidate = self._internal_ask_candidate()
                    except AssertionError as e:
                        assert self.parametrization._constraint_checkers, f'Error: {e}'
                        candidate = self.parametrization.spawn_child()
                if candidate.satisfies_constraints(self.parametrization):
                    if self._num_ask % 10 == 0:
                        if candidate.can_skip_constraints(self.parametrization):
                            self.skip_constraints = True
                    break
                if self._penalize_cheap_violations or self.no_parallelization:
                    self._internal_tell_candidate(candidate, float('Inf'))
                self._num_ask += 1
        satisfies = candidate.satisfies_constraints(self.parametrization)
        if not satisfies and self.parametrization.tabu_length == 