import time
import pickle
import warnings
from pathlib import Path
from numbers import Real
from collections import deque
import numpy as np
import nevergrad.common.typing as tp
from nevergrad.parametrization import parameter as p
from nevergrad.common import tools as ngtools
from nevergrad.common import errors as errors
from nevergrad.common.decorators import Registry
from . import utils
from . import multiobjective as mobj

OptCls = tp.Union['ConfiguredOptimizer', tp.Type['Optimizer']]
registry = Registry()
_OptimCallBack = tp.Union[tp.Callable[['Optimizer', 'p.Parameter', float], None], tp.Callable[['Optimizer'], None]]
X = tp.TypeVar('X', bound='Optimizer')
Y = tp.TypeVar('Y')
IntOrParameter = tp.Union[int, p.Parameter]
_PruningCallable = tp.Callable[[utils.Archive[utils.MultiValue]], utils.Archive[utils.MultiValue]]

def _loss(param: p.Parameter) -> float:
    """Returns the loss if available, or inf otherwise.
    Used to simplify handling of losses
    """
    return param.loss if param.loss is not None else float('inf')

def load(cls: tp.Type[X], filepath: str) -> X:
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

    Parameters
    ----------
    parametrization: int or Parameter
        either the dimension of the optimization space, or its parametrization
    budget: int/None
        number of allowed evaluations
    num_workers: int
        number of evaluations which will be run in parallel at once
    """
    recast: bool = False
    one_shot: bool = False
    no_parallelization: bool = False

    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None:
        if self.no_parallelization and num_workers > 1:
            raise ValueError(f'{self.__class__.__name__} does not support parallelization')
        self.num_workers: int = int(num_workers)
        self.budget: tp.Optional[int] = budget
        self.optim_curve: list = []
        self.skip_constraints: bool = False
        self._constraints_manager: utils.ConstraintManager = utils.ConstraintManager()
        self._penalize_cheap_violations: bool = False
        self.parametrization: p.Parameter = parametrization if not isinstance(parametrization, (int, np.int_)) else p.Array(shape=(parametrization,))
        self.parametrization.freeze()
        if not self.dimension:
            raise ValueError('No variable to optimize in this parametrization.')
        self.name: str = self.__class__.__name__
        self.archive: utils.Archive = utils.Archive()
        self.current_bests: dict = {x: utils.MultiValue(self.parametrization, np.inf, reference=self.parametrization) for x in ['optimistic', 'pessimistic', 'average', 'minimum']}
        self.pruning: utils.Pruning = utils.Pruning.sensible_default(num_workers=num_workers, dimension=self.parametrization.dimension)
        self._MULTIOBJECTIVE_AUTO_BOUND: bool = mobj.AUTO_BOUND
        self._hypervolume_pareto: tp.Optional[mobj.HypervolumePareto] = None
        self._asked: set = set()
        self._num_objectives: int = 0
        self._suggestions: deque = deque()
        self._num_ask: int = 0
        self._num_tell: int = 0
        self._num_tell_not_asked: int = 0
        self._callbacks: dict = {}
        self._running_jobs: list = []
        self._finished_jobs: deque = deque()
        self._sent_warnings: set = set()
        self._no_hypervolume: bool = False

    def _warn(self, msg: str, e: Exception) -> None:
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

    def pareto_front(self, size: tp.Optional[int] = None, subset: str = 'random', subset_tentatives: int = 12) -> list:
        """Pareto front, as a list of Parameter. The losses can be accessed through
        parameter.losses

        Parameters
        ------------
        size:  int (optional)
            if provided, selects a subset of the full pareto front with the given maximum size
        subset: str
            method for selecting the subset ("random, "loss-covering", "domain-covering", "hypervolume")
        subset_tentatives: int
            number of random tentatives for finding a better subset

        Returns
        --------
        list
            the list of Parameter of the pareto front

        Note
        ----
        During non-multiobjective optimization, this returns the current pessimistic best
        """
        pareto = [] if self._hypervolume_pareto is None else self._hypervolume_pareto.pareto_front(size=size, subset=subset, subset_tentatives=subset_tentatives)
        return pareto if pareto else [self.provide_recommendation()]

    def dump(self, filepath: str) -> None:
        """Pickles the optimizer into a file."""
        filepath = Path(filepath)
        with filepath.open('wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> 'Optimizer':
        """Loads a pickle and checks that the class is correct."""
        return load(cls, filepath)

    def __repr__(self) -> str:
        inststr = self.parametrization.name
        return f'Instance of {self.name}(parametrization={inststr}, budget={self.budget}, num_workers={self.num_workers})'

    def register_callback(self, name: str, callback: tp.Callable) -> None:
        """Add a callback method called either when `tell` or `ask` are called, with the same
        arguments (including the optimizer / self). This can be useful for custom logging.

        Parameters
        ----------
        name: str
            name of the method to register the callback for (either :code:`ask` or :code:`tell`)
        callback: callable
            a callable taking the same parameters as the method it is registered upon (including self)
        """
        assert name in ['ask', 'tell'], f'Only "ask" and "tell" methods can have callbacks (not {name})'
        self._callbacks.setdefault(name, []).append(callback)

    def remove_all_callbacks(self) -> None:
        """Removes all registered callables"""
        self._callbacks = {}

    def suggest(self, *args, **kwargs) -> None:
        """Suggests a new point to ask.
        It will be asked at the next call (last in first out).

        Parameters
        ----------
        *args: Any
            positional arguments matching the parametrization pattern.
        *kwargs: Any
            keyword arguments matching the parametrization pattern.

        Note
        ----
        - This relies on optmizers implementing a way to deal with unasked candidate.
          Some optimizers may not support it and will raise a :code:`TellNotAskedNotSupportedError`
          at :code:`tell` time.
        - LIFO is used so as to be able to suggest and ask straightaway, as an alternative to
          creating a new candidate with :code:`optimizer.parametrization.spawn_child(new_value)`
        """
        if isinstance(self.parametrization, p.Instrumentation):
            new_value = (args, kwargs)
        else:
            assert len(args) == 1 and (not kwargs)
            new_value = args[0]
        self._suggestions.append(self.parametrization.spawn_child(new_value=new_value))

    def tell(self, candidate: p.Parameter, loss: tp.Union[float, np.ndarray], constraint_violation: tp.Optional[tp.Union[float, np.ndarray]] = None, penalty_style: tp.Optional[tp.ArrayLike] = None) -> None:
        """Provides the optimizer with the evaluation of a fitness value for a candidate.

        Parameters
        ----------
        x: np.ndarray
            point where the function was evaluated
        loss: float/list/np.ndarray
            loss of the function (or multi-objective function
        constraint_violation: float/list/np.ndarray/None
            constraint violation (> 0 means that this is not correct)
        penalty_style: ArrayLike/None
            to be read as [a,b,c,d,e,f]
            with cv the constraint violation vector (above):
            penalty = (a + sum(|loss|)) * (f+num_tell)**e * (b * sum(cv**c)) ** d
            default: [1e5, 1., .5, 1., .5, 1.]

        Note
        ----
        The candidate should generally be one provided by :code:`ask()`, but can be also
        a non-asked candidate. To create a p.Parameter instance from args and kwargs,
        you can use :code:`candidate = optimizer.parametrization.spawn_child(new_value=your_value)`:

        - for an :code:`Array(shape(2,))`: :code:`optimizer.parametrization.spawn_child(new_value=[12, 12])`

        - for an :code:`Instrumentation`: :code:`optimizer.parametrization.spawn_child(new_value=(args, kwargs))`

        Alternatively, you can provide a suggestion with :code:`optimizer.suggest(*args, **kwargs)`, the next :code:`ask`
        will use this suggestion.
        """
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
            violation = float((a * ratio + np.sum(np.maximum(loss, 0.0))) * (f + self._num_tell) ** e * (b * np.sum(np.maximum(constraint_violation, 0.0) ** c) ** d))
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
        """Provides a point to explore.
        This function can be called multiple times to explore several points in parallel

        Returns
        -------
        p.Parameter:
            The candidate to try on