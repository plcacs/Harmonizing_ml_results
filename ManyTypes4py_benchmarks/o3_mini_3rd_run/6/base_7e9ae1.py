#!/usr/bin/env python3
from __future__ import annotations
import time
import pickle
import warnings
from pathlib import Path
from numbers import Real
from collections import deque
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple
import nevergrad.common.typing as tp
from nevergrad.parametrization import parameter as p
from nevergrad.common import tools as ngtools
from nevergrad.common import errors as errors
from nevergrad.common.decorators import Registry
from . import utils
from . import multiobjective as mobj

OptCls = tp.Union[ConfiguredOptimizer, Type[Optimizer]]
_OptimCallBack = tp.Union[
    Callable[[Optimizer, p.Parameter, float], None],
    Callable[[Optimizer], None]
]
X = tp.TypeVar("X", bound="Optimizer")
Y = tp.TypeVar("Y")
IntOrParameter = tp.Union[int, p.Parameter]
_PruningCallable = Callable[[utils.Archive[utils.MultiValue]], utils.Archive[utils.MultiValue]]

def _loss(param: p.Parameter) -> float:
    """Returns the loss if available, or inf otherwise.
    Used to simplify handling of losses
    """
    return param.loss if param.loss is not None else float("inf")

def load(cls: Type[Optimizer], filepath: Union[str, Path]) -> Optimizer:
    """Loads a pickle file and checks that it contains an optimizer.
    The optimizer class is not always fully reliable though (e.g.: optimizer families) so the user is responsible for it.
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)
    with filepath.open("rb") as f:
        opt = pickle.load(f)
    assert isinstance(opt, cls), f"You should only load {cls} with this method (found {type(opt)})"
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
    recast = False
    one_shot = False
    no_parallelization = False

    def __init__(self, parametrization: Union[int, p.Parameter], budget: Optional[int] = None, num_workers: int = 1) -> None:
        if self.no_parallelization and num_workers > 1:
            raise ValueError(f"{self.__class__.__name__} does not support parallelization")
        self.num_workers: int = int(num_workers)
        self.budget: Optional[int] = budget
        self.optim_curve: List[Tuple[int, float]] = []
        self.skip_constraints: bool = False
        self._constraints_manager: utils.ConstraintManager = utils.ConstraintManager()
        self._penalize_cheap_violations: bool = False
        self.parametrization: Union[p.Parameter, p.Array] = parametrization if not isinstance(parametrization, (int, np.int_)) else p.Array(shape=(parametrization,))
        self.parametrization.freeze()
        if not self.dimension:
            raise ValueError("No variable to optimize in this parametrization.")
        self.name: str = self.__class__.__name__
        self.archive: utils.Archive[utils.MultiValue] = utils.Archive()
        self.current_bests: Dict[str, utils.MultiValue] = {
            x: utils.MultiValue(self.parametrization, np.inf, reference=self.parametrization)
            for x in ["optimistic", "pessimistic", "average", "minimum"]
        }
        self.pruning: Optional[_PruningCallable] = utils.Pruning.sensible_default(num_workers=num_workers, dimension=self.parametrization.dimension)
        self._MULTIOBJECTIVE_AUTO_BOUND: Any = mobj.AUTO_BOUND
        self._hypervolume_pareto: Optional[mobj.HypervolumePareto] = None
        self._asked: set = set()
        self._num_objectives: int = 0
        self._suggestions: deque[p.Parameter] = deque()
        self._num_ask: int = 0
        self._num_tell: int = 0
        self._num_tell_not_asked: int = 0
        self._callbacks: Dict[str, List[Callable[..., None]]] = {}
        self._running_jobs: List[Tuple[p.Parameter, Any]] = []
        self._finished_jobs: deque[Tuple[p.Parameter, Any]] = deque()
        self._sent_warnings: set = set()
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
            raise RuntimeError("Number of objectives is incorrectly set. Please create a nevergrad issue")
        return self._num_objectives

    @num_objectives.setter
    def num_objectives(self, num: int) -> None:
        num = int(num)
        if num <= 0:
            raise ValueError("Number of objectives must be strictly positive")
        if not self._num_objectives:
            self._num_objectives = num
            self._num_objectives_set_callback()
        elif num != self._num_objectives:
            raise ValueError(f"Expected {self._num_objectives} loss(es), but received {num}.")

    def _num_objectives_set_callback(self) -> None:
        """Callback for when num objectives is first known"""
        pass

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

    def pareto_front(self, size: Optional[int] = None, subset: str = "random", subset_tentatives: int = 12) -> List[p.Parameter]:
        """Pareto front, as a list of Parameter. The losses can be accessed through
        parameter.losses

        Parameters
        ------------
        size:  int (optional)
            if provided, selects a subset of the full pareto front with the given maximum size
        subset: str
            method for selecting the subset ("random", "loss-covering", "domain-covering", "hypervolume")
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
        pareto: List[p.Parameter] = [] if self._hypervolume_pareto is None else self._hypervolume_pareto.pareto_front(size=size, subset=subset, subset_tentatives=subset_tentatives)
        return pareto if pareto else [self.provide_recommendation()]

    def dump(self, filepath: Union[str, Path]) -> None:
        """Pickles the optimizer into a file."""
        if isinstance(filepath, str):
            filepath = Path(filepath)
        with filepath.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls: Type[Optimizer], filepath: Union[str, Path]) -> Optimizer:
        """Loads a pickle and checks that the class is correct."""
        return load(cls, filepath)

    def __repr__(self) -> str:
        inststr: str = self.parametrization.name
        return f"Instance of {self.name}(parametrization={inststr}, budget={self.budget}, num_workers={self.num_workers})"

    def register_callback(self, name: str, callback: Callable[..., None]) -> None:
        """Add a callback method called either when `tell` or `ask` are called, with the same
        arguments (including the optimizer / self). This can be useful for custom logging.

        Parameters
        ----------
        name: str
            name of the method to register the callback for (either :code:`ask` or :code:`tell`)
        callback: callable
            a callable taking the same parameters as the method it is registered upon (including self)
        """
        assert name in ["ask", "tell"], f'Only "ask" and "tell" methods can have callbacks (not {name})'
        self._callbacks.setdefault(name, []).append(callback)

    def remove_all_callbacks(self) -> None:
        """Removes all registered callables"""
        self._callbacks = {}

    def suggest(self, *args: Any, **kwargs: Any) -> None:
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
        - This relies on optimizers implementing a way to deal with unasked candidate.
          Some optimizers may not support it and will raise a :code:`TellNotAskedNotSupportedError`
          at :code:`tell` time.
        - LIFO is used so as to be able to suggest and ask straightaway, as an alternative to
          creating a new candidate with :code:`optimizer.parametrization.spawn_child(new_value)`
        """
        if isinstance(self.parametrization, p.Instrumentation):
            new_value: Tuple[Any, Any] = (args, kwargs)
        else:
            assert len(args) == 1 and (not kwargs)
            new_value = args[0]
        self._suggestions.append(self.parametrization.spawn_child(new_value=new_value))

    def tell(
        self,
        candidate: p.Parameter,
        loss: Union[float, List[float], np.ndarray],
        constraint_violation: Optional[Union[float, List[float], np.ndarray]] = None,
        penalty_style: Optional[tp.ArrayLike] = None
    ) -> None:
        """Provides the optimizer with the evaluation of a fitness value for a candidate.

        Parameters
        ----------
        candidate: p.Parameter
            the candidate produced by ask (or a suggested candidate)
        loss: float/list/np.ndarray
            loss of the function (or multi-objective function)
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
        """
        if isinstance(loss, (Real, float)) or (isinstance(loss, np.ndarray) and (not loss.shape)):
            loss = float(loss)  # type: ignore
            if not loss < 5e+20:
                self._warn(f"Clipping very high value {loss} in tell (rescale the cost function?).", errors.LossTooLargeWarning)
                loss = 5e+20
        elif isinstance(loss, (tuple, list, np.ndarray)):
            loss = np.asarray(loss, dtype=float).ravel() if len(loss) != 1 else loss[0]
        elif not isinstance(loss, np.ndarray):
            raise TypeError(f'"tell" method only supports float values but the passed loss was: {loss} (type: {type(loss)}).')
        if isinstance(loss, float) and (len(self.optim_curve) == 0 or self.num_tell > self.optim_curve[-1][0] * 1.1):
            self.optim_curve += [(self.num_tell, loss)]
        if not isinstance(candidate, p.Parameter):
            raise TypeError("'tell' must be provided with the candidate.\nUse optimizer.parametrization.spawn_child(new_value)) if you want to create a candidate that as not been asked for, or optimizer.suggest(*args, **kwargs) to suggest a point that should be used for the next ask")
        self.num_objectives = 1 if isinstance(loss, float) else loss.size  # type: ignore
        candidate.freeze()
        if isinstance(candidate, p.MultiobjectiveReference):
            if self._hypervolume_pareto is not None:
                raise RuntimeError("MultiobjectiveReference can only be provided before the first tell.")
            if not isinstance(loss, np.ndarray):
                raise RuntimeError("MultiobjectiveReference must only be used for multiobjective losses")
            self._hypervolume_pareto = mobj.HypervolumePareto(upper_bounds=loss, seed=self._rng, no_hypervolume=self._no_hypervolume)
            if candidate.value is None:
                return
            candidate = candidate.value
        if isinstance(loss, np.ndarray):
            candidate._losses = loss  # type: ignore
        if not isinstance(loss, float):
            loss = self._preprocess_multiobjective(candidate)
        candidate.loss = loss  # type: ignore
        assert isinstance(loss, float)
        for callback in self._callbacks.get("tell", []):
            callback(self, candidate, loss)
        no_update: bool = False
        if not candidate.satisfies_constraints(self.parametrization) and self.budget is not None:
            penalty: float = self._constraints_manager.penalty(candidate, self.num_ask, self.budget)
            no_update = True
            loss = loss + penalty
        if constraint_violation is not None:
            if penalty_style is not None:
                a, b, c, d, e, f = penalty_style  # type: ignore
            else:
                a, b, c, d, e, f = (100000.0, 1.0, 0.5, 1.0, 0.5, 1.0)
            ratio: float = 1 if (self.budget is not None and self._num_tell > self.budget / 2.0) else 0.0
            iviolation: float = np.sum(np.maximum(constraint_violation, 0.0))  # type: ignore
            if iviolation > 0.0:
                no_update = True
            violation: float = float((a * ratio + np.sum(np.maximum(loss, 0.0))) * (f + self._num_tell) ** e * (b * np.sum(np.maximum(constraint_violation, 0.0) ** c) ** d))  # type: ignore
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
        x: Any = candidate.get_standardized_data(reference=self.parametrization)
        if not isinstance(loss, (Real, float)):
            raise TypeError(f'"tell" method only supports float values but the passed loss was: {loss} (type: {type(loss)}).')
        if np.isnan(loss) or loss == np.inf:
            self._warn(f"Updating fitness with {loss} value", errors.BadLossWarning)
        mvalue: Optional[utils.MultiValue] = None
        if x not in self.archive:
            self.archive[x] = utils.MultiValue(candidate, loss, reference=self.parametrization)
        else:
            mvalue = self.archive[x]
            mvalue.add_evaluation(loss)
            if mvalue.parameter.loss > candidate.loss:  # type: ignore
                mvalue.parameter = candidate
        for name in self.current_bests:
            if mvalue is self.current_bests[name]:
                best: utils.MultiValue = min(self.archive.values(), key=lambda mv, n=name: mv.get_estimation(n))
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
            The candidate to try on the objective function. :code:`p.Parameter` have field :code:`args` and :code:`kwargs`
            which can be directly used on the function (:code:`objective_function(*candidate.args, **candidate.kwargs)`).
        """
        for callback in self._callbacks.get("ask", []):
            callback(self)
        current_num_ask: int = self.num_ask
        max_trials: int = max(1, self._constraints_manager.max_trials // 2)
        self.parametrization.tabu_fails = 0
        is_suggestion: bool = False
        if self.skip_constraints and (not self._suggestions):
            candidate: p.Parameter = self._internal_ask_candidate()
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
                        assert self.parametrization._constraint_checkers, f"Error: {e}"
                        candidate = self.parametrization.spawn_child()
                if candidate.satisfies_constraints(self.parametrization):
                    if self._num_ask % 10 == 0:
                        if candidate.can_skip_constraints(self.parametrization):
                            self.skip_constraints = True
                    break
                if self._penalize_cheap_violations or self.no_parallelization:
                    self._internal_tell_candidate(candidate, float("Inf"))
                self._num_ask += 1
        satisfies: bool = candidate.satisfies_constraints(self.parametrization)
        if not satisfies and self.parametrization.tabu_length == 0:
            candidate = _constraint_solver(candidate, budget=max_trials)
        if not (satisfies or candidate.satisfies_constraints(self.parametrization, no_tabu=True)):
            self._warn(f"Could not bypass the constraint after {max_trials} tentatives, sending candidate anyway.", errors.FailedConstraintWarning)
        if not is_suggestion:
            if candidate.uid in self._asked:
                raise RuntimeError("Cannot submit the same candidate twice: please recreate a new candidate from data.\nThis is to make sure that stochastic parametrizations are resampled.")
            self._asked.add(candidate.uid)
        self._num_ask = current_num_ask + 1
        assert candidate is not None, f"{self.__class__.__name__}._internal_ask method returned None instead of a point."
        candidate.value  # Ensure value is computed.
        candidate.freeze()
        return candidate

    def provide_recommendation(self) -> p.Parameter:
        """Provides the best point to use as a minimum, given the budget that was used

        Returns
        -------
        p.Parameter
            The candidate with minimal value. p.Parameters have field :code:`args` and :code:`kwargs` which can be directly used
            on the function (:code:`objective_function(*candidate.args, **candidate.kwargs)`).
        """
        return self.recommend()

    def recommend(self) -> p.Parameter:
        """Provides the best candidate to use as a minimum, given the budget that was used.

        Returns
        -------
        p.Parameter
            The candidate with minimal loss. :code:`p.Parameters` have field :code:`args` and :code:`kwargs` which can be directly used
            on the function (:code:`objective_function(*candidate.args, **candidate.kwargs)`).
        """
        if self.num_objectives > 1:
            raise RuntimeError("No best candidate in MOO. Use optimizer.pareto_front() instead to get the set of all non-dominated candidates.")
        recom_data: Optional[np.ndarray] = self._internal_provide_recommendation()
        if recom_data is None or any(np.isnan(recom_data)):
            name: str = "minimum" if self.parametrization.function.deterministic else "pessimistic"
            return self.current_bests[name].parameter
        out: p.Parameter = self.parametrization.spawn_child()
        with p.helpers.deterministic_sampling(out):
            out.set_standardized_data(recom_data)
        return out

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: float) -> None:
        """Called whenever calling :code:`tell` on a candidate that was not "asked".
        Defaults to the standard tell pipeline.
        """
        self._internal_tell_candidate(candidate, loss)

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        """Called whenever calling :code:`tell` on a candidate that was "asked"."""
        data: Any = candidate.get_standardized_data(reference=self.parametrization)
        self._internal_tell(data, loss)

    def _internal_ask_candidate(self) -> p.Parameter:
        return self.parametrization.spawn_child().set_standardized_data(self._internal_ask())

    def _internal_tell(self, x: Any, loss: float) -> None:
        pass

    def _internal_ask(self) -> Any:
        raise RuntimeError("Not implemented, should not be called.")

    def _internal_provide_recommendation(self) -> Optional[np.ndarray]:
        """Override to provide a recommendation in standardized space"""
        return None

    def enable_pickling(self) -> None:
        """
        Some optimizers are only optionally picklable, because picklability
        requires saving the whole history which would be a waste of memory
        in general. To tell an optimizer to be picklable, call this function
        before any asks.

        In this base class, the function is a no-op, but it is overridden
        in some optimizers.
        """
        pass

    def minimize(
        self,
        objective_function: Callable[..., float],
        executor: Optional[Any] = None,
        batch_mode: bool = False,
        verbosity: int = 0,
        constraint_violation: Optional[List[Callable[[Any], float]]] = None,
        max_time: Optional[float] = None
    ) -> p.Parameter:
        """Optimization (minimization) procedure

        Parameters
        ----------
        objective_function: callable
            A callable to optimize (minimize)
        executor: Executor
            An executor object, with method :code:`submit(callable, *args, **kwargs)` and returning a Future-like object
            with methods :code:`done() -> bool` and :code:`result() -> float`. The executor role is to dispatch the execution of
            the jobs locally/on a cluster/with multithreading depending on the implementation.
            Eg: :code:`concurrent.futures.ProcessPoolExecutor`
        batch_mode: bool
            when :code:`num_workers = n > 1`, whether jobs are executed by batch (:code:`n` function evaluations are launched,
            we wait for all results and relaunch n evals) or not (whenever an evaluation is finished, we launch
            another one)
        verbosity: int
            print information about the optimization (0: None, 1: fitness values, 2: fitness values and recommendation)
        constraint_violation: list of functions or None
            each function in the list returns >0 for a violated constraint.
        max_time: float/None
            maximum allowed time for the minimization

        Returns
        -------
        p.Parameter
            The candidate with minimal value. In multiobjective optimization, a constant None is returned.
        """
        if self.budget is None:
            raise ValueError("Budget must be specified")
        if executor is None:
            executor = utils.SequentialExecutor()
            if self.num_workers > 1:
                self._warn(f"num_workers = {self.num_workers} > 1 is suboptimal when run sequentially", errors.InefficientSettingsWarning)
        assert executor is not None
        tmp_runnings: List[Tuple[p.Parameter, Any]] = []
        tmp_finished: deque[Tuple[p.Parameter, Any]] = deque()
        sleeper: ngtools.Sleeper = ngtools.Sleeper()
        remaining_budget: int = self.budget - self.num_ask
        first_iteration: bool = True
        t0: float = time.time()
        while (remaining_budget or self._running_jobs or self._finished_jobs) and (max_time is None or time.time() < t0 + max_time):
            if self._finished_jobs:
                if (remaining_budget or sleeper._start is not None) and (not first_iteration):
                    sleeper.stop_timer()
                while self._finished_jobs:
                    x, job = self._finished_jobs[0]
                    result: float = job.result()
                    if constraint_violation is not None:
                        self.tell(x, result, [f(x.value) for f in constraint_violation])
                    else:
                        self.tell(x, result)
                    self._finished_jobs.popleft()
                    if verbosity:
                        print(f"Updating fitness with value {job.result()}")
                if verbosity:
                    print(f"{remaining_budget} remaining budget and {len(self._running_jobs)} running jobs")
                    if verbosity > 1:
                        print("Current pessimistic best is: {}".format(self.current_bests["pessimistic"]))
            elif not first_iteration:
                sleeper.sleep()
            if not batch_mode or not self._running_jobs:
                new_sugg: int = max(0, min(remaining_budget, self.num_workers - len(self._running_jobs)))
                if verbosity and new_sugg:
                    print(f"Launching {new_sugg} jobs with new suggestions")
                for _ in range(new_sugg):
                    try:
                        args: p.Parameter = self.ask()
                    except errors.NevergradEarlyStopping:
                        remaining_budget = 0
                        break
                    self._running_jobs.append((args, executor.submit(objective_function, *args.args, **args.kwargs)))
                if new_sugg:
                    sleeper.start_timer()
            if remaining_budget > 0:
                remaining_budget = self.budget - self.num_ask
            tmp_runnings, tmp_finished = ([], deque())
            for x_job in self._running_jobs:
                if x_job[1].done():
                    tmp_finished.append(x_job)
                else:
                    tmp_runnings.append(x_job)
            self._running_jobs, self._finished_jobs = (tmp_runnings, tmp_finished)
            first_iteration = False
        return self.provide_recommendation() if self.num_objectives == 1 else p.Constant(None)

    def _info(self) -> Dict[str, Any]:
        """Easy access to debug/benchmark info"""
        return {}

def addCompare(optimizer: Optimizer) -> None:
    def compare(self: Optimizer, winners: List[p.Parameter], losers: List[p.Parameter]) -> None:
        ref: p.Parameter = self.parametrization
        best_fitness_value: float = 0.0
        for candidate in losers:
            data: Any = candidate.get_standardized_data(reference=self.parametrization)
            if data in self.archive:
                best_fitness_value = min(best_fitness_value, self.archive[data].get_estimation("average"))
        for i, candidate in enumerate(winners):
            self.tell(candidate, best_fitness_value - len(winners) + i)
            data = candidate.get_standardized_data(reference=self.parametrization)
            self.archive[data] = utils.MultiValue(candidate, best_fitness_value - len(winners) + i, reference=ref)
    setattr(optimizer.__class__, "compare", compare)

class ConfiguredOptimizer:
    """Creates optimizer-like instances with configuration.

    Parameters
    ----------
    OptimizerClass: type
        class of the optimizer to configure, or another ConfiguredOptimizer (config will then be ignored
        except for the optimizer name/representation)
    config: dict
        dictionnary of all the configurations
    as_config: bool
        whether to provide all config as kwargs to the optimizer instantiation (default, see ConfiguredCMA for an example),
        or through a config kwarg referencing self. (if True, see EvolutionStrategy for an example)

    Note
    ----
    This provides a default repr which can be bypassed through set_name
    """
    recast = False
    one_shot = False
    no_parallelization = False

    def __init__(self, OptimizerClass: Union[Type[Optimizer], ConfiguredOptimizer], config: Dict[str, Any], as_config: bool = False) -> None:
        self._OptimizerClass: Union[Type[Optimizer], ConfiguredOptimizer] = OptimizerClass
        config.pop("self", None)
        config.pop("__class__", None)
        self._as_config: bool = as_config
        self._config: Dict[str, Any] = config
        diff: Dict[str, Any] = ngtools.different_from_defaults(instance=self, instance_dict=config, check_mismatches=True)
        params: str = ", ".join((f"{x}={y!r}" for x, y in sorted(diff.items())))
        self.name: str = f"{self.__class__.__name__}({params})"
        if not as_config:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=errors.InefficientSettingsWarning)
                self(parametrization=4, budget=100)

    def config(self) -> Dict[str, Any]:
        return dict(self._config)

    def __call__(self, parametrization: Union[int, p.Parameter], budget: Optional[int] = None, num_workers: int = 1) -> Optimizer:
        """Creates an optimizer from the parametrization

        Parameters
        ----------
        parametrization: int or Parameter
            either the dimension of the optimization space, or its instrumentation
        budget: int/None
            number of allowed evaluations
        num_workers: int
            number of evaluations which will be run in parallel at once
        """
        config: Dict[str, Any] = dict(config=self) if self._as_config else self.config()
        if isinstance(self._OptimizerClass, ConfiguredOptimizer):
            config = {}
        run: Optimizer = self._OptimizerClass(parametrization=parametrization, budget=budget, num_workers=num_workers, **config)  # type: ignore
        run.name = self.name
        run._configured_optimizer = self
        return run

    def __repr__(self) -> str:
        return self.name

    def set_name(self, name: str, register: bool = False) -> ConfiguredOptimizer:
        """Set a new representation for the instance"""
        self.name = name
        if register:
            registry.register_name(name, self)
        return self

    def load(self, filepath: Union[str, Path]) -> Optimizer:
        """Loads a pickle and checks that it is an Optimizer."""
        return self._OptimizerClass.load(filepath)  # type: ignore

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ConfiguredOptimizer) and self.__class__ == other.__class__:
            if self._config == other._config:
                return True
        return False

def _constraint_solver(parameter: p.Parameter, budget: int) -> p.Parameter:
    """Runs a suboptimization to solve the parameter constraints"""
    parameter_without_constraint: p.Parameter = parameter.copy()
    parameter_without_constraint._constraint_checkers.clear()
    parameter_without_constraint.tabu_length = 0
    opt: Optimizer = registry["OnePlusOne"](parameter_without_constraint, num_workers=1, budget=budget)
    for _ in range(budget):
        cand: p.Parameter = opt.ask()
        penalty: float = sum((utils._float_penalty(func(cand.value)) for func in parameter._constraint_checkers))
        distance: float = np.tanh(np.sum(cand.get_standardized_data(reference=parameter) ** 2))
        loss: float = distance if penalty <= 0 else penalty + distance + 1.0
        opt.tell(cand, loss)
        if penalty <= 0:
            break
    data: np.ndarray = opt.recommend().get_standardized_data(reference=parameter_without_constraint)
    return parameter.spawn_child().set_standardized_data(data)
