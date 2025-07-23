import math
import logging
import itertools
from collections import deque, defaultdict
import warnings
import numpy as np
from typing import Optional, List, Any, Dict, Union, Tuple, Type
import scipy.ndimage as ndimage
from nevergrad.common import errors
from nevergrad.parametrization import parameter as p
from nevergrad.parametrization import transforms
from nevergrad.parametrization import discretization
from nevergrad.parametrization import _layering
from nevergrad.parametrization import _datalayers
from . import oneshot
from . import base
from . import mutations
from .metamodel import MetaModelFailure as MetaModelFailure
from .metamodel import learn_on_k_best as learn_on_k_best
from .base import registry as registry
from .base import addCompare
from .base import IntOrParameter
from .differentialevolution import *
from .es import *
from .oneshot import *
from .recastlib import *
try:
    from bayes_opt import UtilityFunction
    from bayes_opt import BayesianOptimization
except ModuleNotFoundError:
    UtilityFunction = Any
    BayesianOptimization = Any
try:
    from .externalbo import HyperOpt
except:
    HyperOpt = Any

logger: logging.Logger = logging.getLogger(__name__)


def smooth_copy(array: Any, possible_radii: Optional[List[int]] = None) -> Any:
    candidate = array.spawn_child()
    if possible_radii is None:
        possible_radii = [3]
    value = candidate._value
    radii = [array.random_state.choice(possible_radii) for _ in value.shape]
    try:
        value2 = ndimage.convolve(value, np.ones(radii) / np.prod(radii))
    except Exception as e:
        assert False, f'{e} in smooth_copy, {radii}, {np.prod(radii)}'
    invfreq = 4 if len(possible_radii) == 1 else max(4, np.random.randint(max(4, len(array.value.flatten()))))
    indices = array.random_state.randint(invfreq, size=value.shape) == 0
    while np.sum(indices) == 0 and len(possible_radii) > 1:
        invfreq = 4 if len(possible_radii) == 1 else max(4, np.random.randint(max(4, len(array.value.flatten()))))
        indices = array.random_state.randint(invfreq, size=value.shape) == 0
    value[indices] = value2[indices]
    candidate._value = value
    return candidate


class _OnePlusOne(base.Optimizer):
    """Simple but sometimes powerful optimization algorithm."""

    def __init__(
        self,
        parametrization: p.Parameter,
        budget: Optional[int] = None,
        num_workers: int = 1,
        *,
        noise_handling: Optional[Union[str, Tuple[str, float]]] = None,
        tabu_length: int = 0,
        mutation: str = 'gaussian',
        crossover: bool = False,
        rotation: bool = False,
        annealing: str = 'none',
        use_pareto: bool = False,
        sparse: bool = False,
        smoother: bool = False,
        super_radii: bool = False,
        roulette_size: int = 2,
        antismooth: int = 55,
        crossover_type: str = 'none',
        forced_discretization: bool = False
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.parametrization.tabu_length = tabu_length
        if forced_discretization:
            self.parametrization.set_integer_casting()
        self.antismooth: int = antismooth
        self.crossover_type: str = crossover_type
        self.roulette_size: int = roulette_size
        assert crossover or not rotation, 'We can not have both rotation and not crossover.'
        self._sigma: float = 1
        self._previous_best_loss: float = float('inf')
        self._best_recent_mr: float = 0.2
        self.inds: np.ndarray = np.array([True] * self.dimension)
        self.imr: float = 0.2
        self.use_pareto: bool = use_pareto
        self.smoother: bool = smoother
        self.super_radii: bool = super_radii
        self.annealing: str = annealing
        self._annealing_base: Optional[Any] = None
        self._max_loss: float = -float('inf')
        self.sparse: int = int(sparse)
        all_params = p.helpers.flatten(self.parametrization)
        arities = [len(param.choices) for _, param in all_params if isinstance(param, p.TransitionChoice)]
        arity: int = max(arities, default=500)
        self.arity_for_discrete_mutation: int = arity
        if noise_handling is not None:
            if isinstance(noise_handling, str):
                assert noise_handling in ['random', 'optimistic'], f"Unkwnown noise handling: '{noise_handling}'"
            else:
                assert isinstance(noise_handling, tuple), 'noise_handling must be a string or  a tuple of type (strategy, factor)'
                assert noise_handling[1] > 0.0, 'the factor must be a float greater than 0'
                assert noise_handling[0] in ['random', 'optimistic'], f"Unkwnown noise handling: '{noise_handling}'"
        assert mutation in [
            'gaussian', 'cauchy', 'discrete', 'fastga', 'rls', 'doublefastga',
            'adaptive', 'coordinatewise_adaptive', 'portfolio', 'discreteBSO',
            'lengler', 'lengler2', 'lengler3', 'lenglerhalf', 'lenglerfourth',
            'doerr', 'lognormal', 'xlognormal', 'xsmalllognormal', 'tinylognormal',
            'lognormal', 'smalllognormal', 'biglognormal', 'hugelognormal'
        ], f"Unkwnown mutation: '{mutation}'"
        if mutation == 'adaptive':
            self._adaptive_mr: float = 0.5
        elif mutation == 'lognormal':
            self._global_mr: float = 0.2
            self._memory_index: int = 0
            self._memory_size: int = 12
            self._best_recent_loss: float = float('inf')
        elif mutation == 'xsmalllognormal':
            self._global_mr: float = 0.8
            self._memory_index: int = 0
            self._memory_size: int = 4
            self._best_recent_loss: float = float('inf')
        elif mutation == 'xlognormal':
            self._global_mr: float = 0.8
            self._memory_index: int = 0
            self._memory_size: int = 12
            self._best_recent_loss: float = float('inf')
        elif mutation == 'tinylognormal':
            self._global_mr: float = 0.01
            self._memory_index: int = 0
            self._memory_size: int = 2
            self._best_recent_loss: float = float('inf')
        elif mutation == 'smalllognormal':
            self._global_mr: float = 0.2
            self._memory_index: int = 0
            self._memory_size: int = 4
            self._best_recent_loss: float = float('inf')
        elif mutation == 'biglognormal':
            self._global_mr: float = 0.2
            self._memory_index: int = 0
            self._memory_size: int = 120
            self._best_recent_loss: float = float('inf')
        elif mutation == 'hugelognormal':
            self._global_mr: float = 0.2
            self._memory_index: int = 0
            self._memory_size: int = 1200
            self._best_recent_loss: float = float('inf')
        elif mutation == 'coordinatewise_adaptive':
            self._velocity: np.ndarray = self._rng.uniform(size=self.dimension) * arity / 4.0
            self._modified_variables: np.ndarray = np.array([True] * self.dimension)
        self.noise_handling: Optional[Union[str, Tuple[str, float]]] = noise_handling
        self.mutation: str = mutation
        self.crossover: bool = crossover
        self.rotation: bool = rotation
        if mutation == 'doerr':
            assert num_workers == 1, 'Doerr mutation is implemented only in the sequential case.'
            self._doerr_mutation_rates: List[int] = [1, 2]
            self._doerr_mutation_rewards: List[float] = [0.0, 0.0]
            self._doerr_counters: List[float] = [0.0, 0.0]
            self._doerr_epsilon: float = 0.25
            self._doerr_gamma: float = 1 - 2 / self.dimension
            self._doerr_current_best: float = float('inf')
            i: int = 3
            j: int = 2
            self._doerr_index: int = -1
            while i < self.dimension:
                self._doerr_mutation_rates += [i]
                self._doerr_mutation_rewards += [0.0]
                self._doerr_counters += [0.0]
                i += j
                j += 2
        assert self.parametrization.tabu_length == tabu_length

    def _internal_ask_candidate(self) -> p.Parameter:
        noise_handling = self.noise_handling
        if not self._num_ask:
            out: p.Parameter = self.parametrization.spawn_child()
            out._meta['sigma'] = self._sigma
            return out
        if noise_handling is not None:
            limit: float = (0.05 if isinstance(noise_handling, str) else noise_handling[1]) * len(self.archive) ** 3
            strategy: Union[str, Tuple[str, float]] = noise_handling if isinstance(noise_handling, str) else noise_handling[0]
            if self._num_ask <= limit:
                if strategy in ['cubic', 'random']:
                    idx: int = self._rng.choice(len(self.archive))
                    return list(self.archive.values())[idx].parameter.spawn_child()
                elif strategy == 'optimistic':
                    return self.current_bests['optimistic'].parameter.spawn_child()
        mutator: mutations.Mutator = mutations.Mutator(self._rng)
        pessimistic: p.Parameter = self.current_bests['pessimistic'].parameter.spawn_child()
        if self.smoother and self._num_ask % max(self.num_workers + 1, self.antismooth) == 0 and isinstance(self.parametrization, p.Array):
            possible_radii: List[int] = [3] if not self.super_radii else [3, 3 + np.random.randint(int(np.sqrt(np.sqrt(self.dimension))))]
            self.suggest(smooth_copy(pessimistic, possible_radii=possible_radii).value)
        if self.num_objectives > 1 and self.use_pareto:
            pareto: List[p.Parameter] = self.pareto_front()
            pessimistic = pareto[self._rng.choice(len(pareto))].spawn_child()
        ref: p.Parameter = self.parametrization
        if self.crossover and self._num_ask % 2 == 1 and (len(self.archive) > 2):
            data: np.ndarray = mutator.crossover(
                pessimistic.get_standardized_data(reference=ref),
                mutator.get_roulette(self.archive, num=self.roulette_size),
                rotation=self.rotation,
                crossover_type=self.crossover_type
            )
            return pessimistic.set_standardized_data(data, reference=ref)
        mutation: str = self.mutation
        if self._annealing_base is not None:
            assert self.annealing != 'none'
            pessimistic.set_standardized_data(self._annealing_base, reference=ref)
        if mutation in ('gaussian', 'cauchy'):
            step: np.ndarray = self._rng.normal(0, 1, self.dimension) if mutation == 'gaussian' else self._rng.standard_cauchy(self.dimension)
            out: p.Parameter = pessimistic.set_standardized_data(self._sigma * step)
            out._meta['sigma'] = self._sigma
            return out
        else:
            pessimistic_data: np.ndarray = pessimistic.get_standardized_data(reference=ref)
            if mutation == 'crossover':
                if self._num_ask % 2 == 0 or len(self.archive) < 3:
                    data = mutator.portfolio_discrete_mutation(pessimistic_data, arity=self.arity_for_discrete_mutation)
                else:
                    data = mutator.crossover(pessimistic_data, mutator.get_roulette(self.archive, num=2))
            elif 'lognormal' in mutation:
                mutation_rate: float = max(0.1 / self.dimension, self._global_mr)
                assert mutation_rate > 0.0
                individual_mutation_rate: float = 1.0 / (1.0 + (1.0 - mutation_rate) / mutation_rate * np.exp(0.22 * np.random.randn()))
                data = mutator.portfolio_discrete_mutation(
                    pessimistic_data,
                    intensity=individual_mutation_rate * self.dimension,
                    arity=self.arity_for_discrete_mutation
                )
            elif mutation == 'adaptive':
                data = mutator.portfolio_discrete_mutation(
                    pessimistic_data,
                    intensity=max(1, int(self._adaptive_mr * self.dimension)),
                    arity=self.arity_for_discrete_mutation
                )
            elif mutation == 'discreteBSO':
                assert self.budget is not None, 'DiscreteBSO needs a budget.'
                intensity: int = int(self.dimension - self._num_ask * self.dimension / self.budget)
                if intensity < 1:
                    intensity = 1
                data = mutator.portfolio_discrete_mutation(
                    pessimistic_data,
                    intensity=intensity,
                    arity=self.arity_for_discrete_mutation
                )
            elif mutation == 'coordinatewise_adaptive':
                self._modified_variables = np.array([True] * self.dimension)
                data = mutator.coordinatewise_mutation(
                    pessimistic_data,
                    self._velocity,
                    self._modified_variables,
                    arity=self.arity_for_discrete_mutation
                )
            elif mutation == 'lengler':
                alpha: float = 1.54468
                intensity: int = int(max(1, self.dimension * (alpha * np.log(self.num_ask) / self.num_ask)))
                data = mutator.portfolio_discrete_mutation(
                    pessimistic_data,
                    intensity=intensity,
                    arity=self.arity_for_discrete_mutation
                )
            elif mutation == 'lengler2':
                alpha = 3.0
                intensity = int(max(1, self.dimension * (alpha * np.log(self.num_ask) / self.num_ask)))
                data = mutator.portfolio_discrete_mutation(
                    pessimistic_data,
                    intensity=intensity,
                    arity=self.arity_for_discrete_mutation
                )
            elif mutation == 'lengler3':
                alpha = 9.0
                intensity = int(max(1, self.dimension * (alpha * np.log(self.num_ask) / self.num_ask)))
                data = mutator.portfolio_discrete_mutation(
                    pessimistic_data,
                    intensity=intensity,
                    arity=self.arity_for_discrete_mutation
                )
            elif mutation == 'lenglerfourth':
                alpha = 0.4
                intensity = int(max(1, self.dimension * (alpha * np.log(self.num_ask) / self.num_ask)))
                data = mutator.portfolio_discrete_mutation(
                    pessimistic_data,
                    intensity=intensity,
                    arity=self.arity_for_discrete_mutation
                )
            elif mutation == 'lenglerhalf':
                alpha = 0.8
                intensity = int(max(1, self.dimension * (alpha * np.log(self.num_ask) / self.num_ask)))
                data = mutator.portfolio_discrete_mutation(
                    pessimistic_data,
                    intensity=intensity,
                    arity=self.arity_for_discrete_mutation
                )
            elif mutation == 'doerr':
                assert self._doerr_index == -1, 'We should have used this index in tell.'
                if self._rng.uniform() < self._doerr_epsilon:
                    index: int = self._rng.choice(range(len(self._doerr_mutation_rates)))
                    self._doerr_index = index
                else:
                    index: int = self._doerr_mutation_rewards.index(max(self._doerr_mutation_rewards))
                    self._doerr_index = -1
                intensity: int = self._doerr_mutation_rates[index]
                data = mutator.portfolio_discrete_mutation(
                    pessimistic_data,
                    intensity=intensity,
                    arity=self.arity_for_discrete_mutation
                )
            else:
                func: Any = {
                    'discrete': mutator.discrete_mutation,
                    'fastga': mutator.doerr_discrete_mutation,
                    'doublefastga': mutator.doubledoerr_discrete_mutation,
                    'rls': mutator.rls_mutation,
                    'portfolio': mutator.portfolio_discrete_mutation
                }[mutation]
                data = func(pessimistic_data, arity=self.arity_for_discrete_mutation)
            if self.sparse > 0:
                data = np.asarray(data)
                zeroing: np.ndarray = self._rng.randint(data.size + 1, size=data.size) < 1 + self._rng.randint(self.sparse)
                data[zeroing] = 0.0
            candidate: p.Parameter = pessimistic.set_standardized_data(data, reference=ref)
            if mutation == 'coordinatewise_adaptive':
                candidate._meta['modified_variables'] = (self._modified_variables,)
            if 'lognormal' in mutation:
                candidate._meta['individual_mutation_rate'] = individual_mutation_rate
            return candidate

    def _internal_tell(self, x: Any, loss: float) -> None:
        if self.annealing != 'none':
            assert isinstance(self.budget, int)
            delta: float = self._previous_best_loss - loss
            if loss > self._max_loss:
                self._max_loss = loss
            if delta >= 0:
                self._annealing_base = x
            elif self.num_ask < self.budget:
                amplitude: float = max(1.0, self._max_loss - self._previous_best_loss)
                annealing_dict: Dict[str, float] = {
                    'Exp0.9': 0.33 * amplitude * 0.9 ** self.num_ask,
                    'Exp0.99': 0.33 * amplitude * 0.99 ** self.num_ask,
                    'Exp0.9Auto': 0.33 * amplitude * (0.001 ** (1.0 / self.budget)) ** self.num_ask,
                    'Lin100.0': 100.0 * amplitude * (1 - self.num_ask / (self.budget + 1)),
                    'Lin1.0': 1.0 * amplitude * (1 - self.num_ask / (self.budget + 1)),
                    'LinAuto': 10.0 * amplitude * (1 - self.num_ask / (self.budget + 1))
                }
                T: float = annealing_dict[self.annealing]
                if T > 0.0:
                    proba: float = np.exp(delta / T)
                    if self._rng.rand() < proba:
                        self._annealing_base = x
        if self._previous_best_loss != loss:
            self._sigma *= 2.0 if loss < self._previous_best_loss else 0.84
        if self.mutation == 'doerr' and self._doerr_current_best < float('inf') and (self._doerr_index >= 0):
            improvement: float = max(0.0, self._doerr_current_best - loss)
            index: int = self._doerr_index
            counter: float = self._doerr_counters[index]
            self._doerr_mutation_rewards[index] = (self._doerr_gamma * counter * self._doerr_mutation_rewards[index] + improvement) / (self._doerr_gamma * counter + 1)
            self._doerr_counters = [self._doerr_gamma * x for x in self._doerr_counters]
            self._doerr_counters[index] += 1
            self._doerr_index = -1
        if self.mutation == 'doerr':
            self._doerr_current_best = min(self._doerr_current_best, loss)
        elif self.mutation == 'adaptive':
            factor: float = 1.2 if loss <= self._previous_best_loss else 0.731
            self._adaptive_mr = min(1.0, factor * self._adaptive_mr)
        elif self.mutation == 'coordinatewise_adaptive':
            factor = 1.2 if loss < self._previous_best_loss else 0.731
            inds = self.inds
            self._velocity[inds] = np.clip(self._velocity[inds] * factor, 1.0, self.arity_for_discrete_mutation / 4.0)
        elif 'lognormal' in self.mutation:
            self._memory_index = (self._memory_index + 1) % self._memory_size
            if loss < self._best_recent_loss:
                self._best_recent_loss = loss
                self._best_recent_mr = self.imr
            if self._memory_index == 0:
                self._global_mr = self._best_recent_mr
                self._best_recent_loss = float('inf')
        self._previous_best_loss = self.current_bests['pessimistic'].mean

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        """Called whenever calling :code:`tell` on a candidate that was "asked"."""
        data = candidate.get_standardized_data(reference=self.parametrization)
        if self.mutation == 'coordinatewise_adaptive':
            self.inds = candidate._meta['modified_variables'] if 'modified_variables' in candidate._meta else np.array([True] * len(data))
        if 'lognormal' in self.mutation:
            self.imr = candidate._meta['individual_mutation_rate'] if 'individual_mutation_rate' in candidate._meta else 0.2
        self._internal_tell(data, loss)

    class ParametrizedOnePlusOne(base.ConfiguredOptimizer):
        """Simple but sometimes powerfull class of optimization algorithm."""

        def __init__(
            self,
            *,
            noise_handling: Optional[Union[str, Tuple[str, float]]] = None,
            tabu_length: int = 0,
            mutation: str = 'gaussian',
            crossover: bool = False,
            rotation: bool = False,
            annealing: str = 'none',
            use_pareto: bool = False,
            sparse: bool = False,
            smoother: bool = False,
            super_radii: bool = False,
            roulette_size: int = 2,
            antismooth: int = 55,
            crossover_type: str = 'none',
            forced_discretization: bool = False
        ) -> None:
            super().__init__(_OnePlusOne, locals())

    OnePlusOne: base.ConfiguredOptimizer = ParametrizedOnePlusOne().set_name('OnePlusOne', register=True)
    OnePlusLambda: base.ConfiguredOptimizer = ParametrizedOnePlusOne().set_name('OnePlusLambda', register=True)
    NoisyOnePlusOne: base.ConfiguredOptimizer = ParametrizedOnePlusOne(noise_handling='random').set_name('NoisyOnePlusOne', register=True)
    DiscreteOnePlusOne: base.ConfiguredOptimizer = ParametrizedOnePlusOne(mutation='discrete').set_name('DiscreteOnePlusOne', register=True)
    SADiscreteLenglerOnePlusOneExp09: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
        tabu_length=1000, 
        mutation='lengler', 
        annealing='Exp0.9'
    ).set_name('SADiscreteLenglerOnePlusOneExp09', register=True)
    SADiscreteLenglerOnePlusOneExp099: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
        tabu_length=1000, 
        mutation='lengler', 
        annealing='Exp0.99'
    ).set_name('SADiscreteLenglerOnePlusOneExp099', register=True)
    SADiscreteLenglerOnePlusOneExp09Auto: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
        tabu_length=1000, 
        mutation='lengler', 
        annealing='Exp0.9Auto'
    ).set_name('SADiscreteLenglerOnePlusOneExp09Auto', register=True)
    SADiscreteLenglerOnePlusOneLinAuto: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
        tabu_length=1000, 
        mutation='lengler', 
        annealing='LinAuto'
    ).set_name('SADiscreteLenglerOnePlusOneLinAuto', register=True)
    SADiscreteLenglerOnePlusOneLin1: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
        tabu_length=1000, 
        mutation='lengler', 
        annealing='Lin1.0'
    ).set_name('SADiscreteLenglerOnePlusOneLin1', register=True)
    SADiscreteLenglerOnePlusOneLin100: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
        tabu_length=1000, 
        mutation='lengler', 
        annealing='Lin100.0'
    ).set_name('SADiscreteLenglerOnePlusOneLin100', register=True)
    SADiscreteOnePlusOneExp099: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
        tabu_length=1000, 
        mutation='discrete', 
        annealing='Exp0.99'
    ).set_name('SADiscreteOnePlusOneExp099', register=True)
    SADiscreteOnePlusOneLin100: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
        tabu_length=1000, 
        mutation='discrete', 
        annealing='Lin100.0'
    ).set_name('SADiscreteOnePlusOneLin100', register=True)
    SADiscreteOnePlusOneExp09: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
        tabu_length=1000, 
        mutation='discrete', 
        annealing='Exp0.9'
    ).set_name('SADiscreteOnePlusOneExp09', register=True)
    DiscreteOnePlusOneT: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
        tabu_length=10000, 
        mutation='discrete'
    ).set_name('DiscreteOnePlusOneT', register=True)
    PortfolioDiscreteOnePlusOne: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
        mutation='portfolio'
    ).set_name('PortfolioDiscreteOnePlusOne', register=True)
    PortfolioDiscreteOnePlusOneT: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
        tabu_length=10000, 
        mutation='portfolio'
    ).set_name('PortfolioDiscreteOnePlusOneT', register=True)
    DiscreteLenglerOnePlusOne: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
        mutation='lengler'
    ).set_name('DiscreteLenglerOnePlusOne', register=True)
    DiscreteLengler2OnePlusOne: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
        mutation='lengler2'
    ).set_name('DiscreteLengler2OnePlusOne', register=True)
    DiscreteLengler3OnePlusOne: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
        mutation='lengler3'
    ).set_name('DiscreteLengler3OnePlusOne', register=True)
    DiscreteLenglerHalfOnePlusOne: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
        mutation='lenglerhalf'
    ).set_name('DiscreteLenglerHalfOnePlusOne', register=True)
    DiscreteLenglerFourthOnePlusOne: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
        mutation='lenglerfourth'
    ).set_name('DiscreteLenglerFourthOnePlusOne', register=True)
    DoerrDiscreteOnePlusOne: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
        mutation='doerr'
    ).set_name('DoerrDiscreteOnePlusOne', register=True)
    DiscreteDoerrOnePlusOne: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
        mutation='doerr'
    ).set_name('DiscreteDoerrOnePlusOne', register=True)
    DiscreteDoerrOnePlusOne.no_parallelization = True
    CauchyOnePlusOne: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
        mutation='cauchy'
    ).set_name('CauchyOnePlusOne', register=True)
    OptimisticNoisyOnePlusOne: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
        noise_handling='optimistic'
    ).set_name('OptimisticNoisyOnePlusOne', register=True)
    OptimisticDiscreteOnePlusOne: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
        noise_handling='optimistic', 
        mutation='discrete'
    ).set_name('OptimisticDiscreteOnePlusOne', register=True)
    OLNDiscreteOnePlusOne: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
        noise_handling='optimistic', 
        mutation='lognormal'
    ).set_name('OLNDiscreteOnePlusOne', register=True)
    NoisyDiscreteOnePlusOne: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
        noise_handling=('random', 1.0), 
        mutation='discrete'
    ).set_name('NoisyDiscreteOnePlusOne', register=True)
    DoubleFastGADiscreteOnePlusOne: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
        mutation='doublefastga'
    ).set_name('DoubleFastGADiscreteOnePlusOne', register=True)
    RLSOnePlusOne: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
        mutation='rls'
    ).set_name('RLSOnePlusOne', register=True)
    SparseDoubleFastGADiscreteOnePlusOne: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
        mutation='doublefastga', 
        sparse=True
    ).set_name('SparseDoubleFastGADiscreteOnePlusOne', register=True)
    RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
        crossover=True, 
        mutation='portfolio', 
        noise_handling='optimistic'
    ).set_name('RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne', register=True)
    RecombiningPortfolioDiscreteOnePlusOne: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
        crossover=True, 
        mutation='portfolio'
    ).set_name('RecombiningPortfolioDiscreteOnePlusOne', register=True)


class _CMA(base.Optimizer):
    _CACHE_KEY: str = '#CMA#datacache'

    def __init__(
        self,
        parametrization: p.Parameter,
        budget: Optional[int] = None,
        num_workers: int = 1,
        config: Optional[Any] = None,
        algorithm: str = 'quad'
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.algorithm: str = algorithm
        self._config: Any = ParametrizedCMA() if config is None else config
        pop = self._config.popsize
        self._popsize: int = max(
            num_workers, 
            4 + int(self._config.popsize_factor * np.log(self.dimension))
        ) if pop is None else max(pop, num_workers)
        if self._config.elitist:
            self._popsize = max(self._popsize, self.num_workers + 1)
        self._to_be_asked: deque = deque()
        self._to_be_told: List[p.Parameter] = []
        self._num_spawners: int = self._popsize // 2
        self._parents: List[p.Parameter] = [self.parametrization]
        self._es: Optional[Any] = None

    @property
    def es(self) -> Any:
        scale_multiplier: float = 1.0
        if self.dimension == 1:
            self._config.fcmaes = True
        if p.helpers.Normalizer(self.parametrization).fully_bounded:
            scale_multiplier = 0.3 if self.dimension < 18 else 0.15
        if self._es is None or (not self._config.fcmaes and self._es.stop()):
            if not self._config.fcmaes:
                import cma
                inopts: Dict[str, Any] = {
                    'popsize': self._popsize,
                    'randn': self._rng.randn,
                    'CMA_diagonal': self._config.diagonal,
                    'verbose': -9,
                    'seed': np.nan,
                    'CMA_elitist': self._config.elitist
                }
                if self._config.inopts is not None:
                    inopts.update(self._config.inopts)
                initial_x: np.ndarray = (
                    self.parametrization.sample().get_standardized_data(reference=self.parametrization) 
                    if self._config.random_init 
                    else np.zeros(self.dimension, dtype=np.float64)
                )
                self._es = cma.CMAEvolutionStrategy(
                    x0=initial_x, 
                    sigma0=self._config.scale * scale_multiplier, 
                    inopts=inopts
                )
            else:
                try:
                    from fcmaes import cmaes
                except ImportError as e:
                    raise ImportError('Please install fcmaes (pip install fcmaes) to use FCMA optimizers') from e
                self._es = cmaes.Cmaes(
                    x0=np.zeros(self.dimension, dtype=np.float64), 
                    input_sigma=self._config.scale * scale_multiplier, 
                    popsize=self._popsize,
                    randn=self._rng.randn
                )
        return self._es

    def _internal_ask_candidate(self) -> p.Parameter:
        if not self._to_be_asked:
            self._to_be_asked.extend(self.es.ask())
        data: np.ndarray = self._to_be_asked.popleft()
        parent: p.Parameter = self._parents[self.num_ask % len(self._parents)]
        candidate: p.Parameter = parent.spawn_child().set_standardized_data(data, reference=self.parametrization)
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        if self._CACHE_KEY not in candidate._meta:
            candidate._meta[self._CACHE_KEY] = candidate.get_standardized_data(reference=self.parametrization)
        self._to_be_told.append(candidate)
        if len(self._to_be_told) >= self.es.popsize:
            listx: List[np.ndarray] = [c._meta[self._CACHE_KEY] for c in self._to_be_told]
            listy: List[float] = [c.loss for c in self._to_be_told]
            args: Tuple[Union[List[np.ndarray], List[float]], Union[List[np.ndarray], List[float]]] = (
                tuple(listy),
                tuple(listx)
            ) if self._config.fcmaes else (tuple(listx), tuple(listy))
            try:
                self.es.tell(*args)
            except (RuntimeError, AssertionError):
                pass
            else:
                self._parents = sorted(self._to_be_told, key=base._loss)[:self._num_spawners]
            self._to_be_told = []

    def _internal_provide_recommendation(self) -> Optional[np.ndarray]:
        pessimistic: p.Parameter = self.current_bests['pessimistic'].parameter.get_standardized_data(reference=self.parametrization)
        d: int = self.dimension
        n: int = self.num_ask
        sample_size: int = int(d * d / 2 + d / 2 + 3)
        if self._config.high_speed and n >= sample_size:
            try:
                data: np.ndarray = learn_on_k_best(self.archive, sample_size, self.algorithm)
                return data
            except MetaModelFailure:
                pass
        if self._es is None:
            return pessimistic
        cma_best: Optional[np.ndarray] = self.es.best_x if self._config.fcmaes else self.es.result.xbest
        if cma_best is None:
            return pessimistic
        return cma_best


class ParametrizedCMA(base.ConfiguredOptimizer):
    """CMA-ES optimizer."""

    def __init__(
        self,
        *,
        scale: float = 1.0,
        elitist: bool = False,
        popsize: Optional[int] = None,
        popsize_factor: float = 3.0,
        diagonal: bool = False,
        zero: bool = False,
        high_speed: bool = False,
        fcmaes: bool = False,
        random_init: bool = False,
        inopts: Optional[Dict[str, Any]] = None,
        algorithm: str = 'quad'
    ) -> None:
        super().__init__(_CMA, locals(), as_config=True)
        if zero:
            self.scale = scale / 1000.0
        if fcmaes:
            if diagonal:
                raise RuntimeError("fcmaes doesn't support diagonal=True, use fcmaes=False")
        self.scale: float = scale
        self.elitist: bool = elitist
        self.zero: bool = zero
        self.popsize: Optional[int] = popsize
        self.popsize_factor: float = popsize_factor
        self.diagonal: bool = diagonal
        self.fcmaes: bool = fcmaes
        self.high_speed: bool = high_speed
        self.random_init: bool = random_init
        self.inopts: Optional[Dict[str, Any]] = inopts


@registry.register
class ChoiceBase(base.Optimizer):
    """Nevergrad optimizer by competence map."""

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        analysis = p.helpers.analyze(self.parametrization)
        funcinfo = self.parametrization.function
        self.has_noise: bool = not (analysis.deterministic and funcinfo.deterministic)
        self.noise_from_instrumentation: bool = self.has_noise and funcinfo.deterministic
        self.fully_continuous: bool = analysis.continuous
        all_params = p.helpers.flatten(self.parametrization)
        int_layers: List[_layering.Int] = list(
            itertools.chain.from_iterable([_layering.Int.filter_from(x) for _, x in all_params])
        )
        int_layers = [x for x in int_layers if x.arity is not None]
        self.has_discrete_not_softmax: bool = any(
            [not isinstance(lay, _datalayers.SoftmaxSampling) for lay in int_layers]
        )
        self._has_discrete: bool = bool(int_layers)
        self._arity: int = max([lay.arity for lay in int_layers], default=-1)
        if self.fully_continuous:
            self._arity = -1
        self._optim: Optional[base.Optimizer] = None
        self._constraints_manager.update(max_trials=1000, penalty_factor=1.0, penalty_exponent=1.01)

    @property
    def optim(self) -> base.Optimizer:
        if self._optim is None:
            self._optim = self._select_optimizer_cls()(
                self.parametrization, 
                self.budget, 
                self.num_workers
            )
            self._optim = self._optim if not isinstance(self._optim, NGOptBase) else self._optim.optim
            logger.debug('%s selected %s optimizer.', self.name, self._optim.name)
        return self._optim

    def _select_optimizer_cls(self) -> Type[base.Optimizer]:
        return CMA

    def _internal_ask_candidate(self) -> p.Parameter:
        return self.optim.ask()

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        self.optim.tell(candidate, loss)

    def recommend(self) -> Optional[p.Parameter]:
        return self.optim.recommend()

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: float) -> None:
        self.optim.tell(candidate, loss)

    def _info(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {'sub-optim': self.optim.name}
        out.update(self.optim._info())
        return out

    def enable_pickling(self) -> None:
        self.optim.enable_pickling()


OldCMA: base.ConfiguredOptimizer = ParametrizedCMA().set_name('OldCMA', register=True)
LargeCMA: base.ConfiguredOptimizer = ParametrizedCMA(scale=3.0).set_name('LargeCMA', register=True)
LargeDiagCMA: base.ConfiguredOptimizer = ParametrizedCMA(scale=3.0, diagonal=True).set_name('LargeDiagCMA', register=True)
TinyCMA: base.ConfiguredOptimizer = ParametrizedCMA(scale=0.33).set_name('TinyCMA', register=True)
CMAbounded: base.ConfiguredOptimizer = ParametrizedCMA(
    scale=1.5884, 
    popsize_factor=1, 
    elitist=True, 
    diagonal=True, 
    fcmaes=False
).set_name('CMAbounded', register=True)
CMAsmall: base.ConfiguredOptimizer = ParametrizedCMA(
    scale=0.3607, 
    popsize_factor=3, 
    elitist=False, 
    diagonal=False, 
    fcmaes=False
).set_name('CMAsmall', register=True)
CMAstd: base.ConfiguredOptimizer = ParametrizedCMA(
    scale=0.4699, 
    popsize_factor=3, 
    elitist=False, 
    diagonal=False, 
    fcmaes=False
).set_name('CMAstd', register=True)
CMApara: base.ConfiguredOptimizer = ParametrizedCMA(
    scale=0.8905, 
    popsize_factor=8, 
    elitist=True, 
    diagonal=True, 
    fcmaes=False
).set_name('CMApara', register=True)
CMAtuning: base.ConfiguredOptimizer = ParametrizedCMA(
    scale=0.4847, 
    popsize_factor=1, 
    elitist=True, 
    diagonal=False, 
    fcmaes=False
).set_name('CMAtuning', register=True)


@registry.register
class MetaCMA(ChoiceBase):
    """Nevergrad CMA optimizer by competence map."""

    def _select_optimizer_cls(self) -> Type[base.Optimizer]:
        if self.budget is not None and self.fully_continuous and (not self.has_noise) and (self.num_objectives < 2):
            if self.dimension == 1:
                return OnePlusOne
            if p.helpers.Normalizer(self.parametrization).fully_bounded:
                return CMAbounded
            if self.budget < 50:
                if self.dimension <= 15:
                    return CMAtuning
                return CMAsmall
            if self.num_workers > 20:
                return CMApara
            return CMAstd
        else:
            return OldCMA


DiagonalCMA: base.ConfiguredOptimizer = ParametrizedCMA(diagonal=True).set_name('DiagonalCMA', register=True)
EDCMA: base.ConfiguredOptimizer = ParametrizedCMA(diagonal=True, elitist=True).set_name('EDCMA', register=True)
SDiagonalCMA: base.ConfiguredOptimizer = ParametrizedCMA(diagonal=True, zero=True).set_name('SDiagonalCMA', register=True)
FCMA: base.ConfiguredOptimizer = ParametrizedCMA(fcmaes=True).set_name('FCMA', register=True)


@registry.register
class CMA(MetaCMA):
    pass


class _PopulationSizeController:
    """Population control scheme for TBPSA and EDA"""

    def __init__(
        self, 
        llambda: int, 
        mu: int, 
        dimension: int, 
        num_workers: int = 1
    ) -> None:
        self.llambda: int = max(llambda, num_workers)
        self.min_mu: int = min(mu, dimension)
        self.mu: int = mu
        self.dimension: int = dimension
        self.num_workers: int = num_workers
        self._loss_record: List[float] = []

    def add_value(self, loss: float) -> None:
        self._loss_record.append(loss)
        if len(self._loss_record) >= 5 * self.llambda:
            first_fifth: List[float] = self._loss_record[:self.llambda]
            last_fifth: List[float] = self._loss_record[-int(self.llambda):]
            means: List[float] = [
                sum(fitnesses) / float(self.llambda) 
                for fitnesses in [first_fifth, last_fifth]
            ]
            stds: List[float] = [
                np.std(fitnesses) / math.sqrt(self.llambda - 1) 
                for fitnesses in [first_fifth, last_fifth]
            ]
            z: float = (means[0] - means[1]) / math.sqrt(stds[0] ** 2 + stds[1] ** 2)
            if z < 2.0:
                self.mu *= 2
            else:
                self.mu = max(self.min_mu, int(self.mu * 0.84))
            self.llambda = 4 * self.mu
            if self.num_workers > 1:
                self.llambda = max(self.llambda, self.num_workers)
                self.mu = self.llambda // 4
            self._loss_record = []


@registry.register
class EDA(base.Optimizer):
    """Estimation of distribution algorithm."""

    _POPSIZE_ADAPTATION: bool = False
    _COVARIANCE_MEMORY: bool = False

    def __init__(
        self,
        parametrization: p.Parameter,
        budget: Optional[int] = None,
        num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.sigma: float = 1.0
        self.covariance: np.ndarray = np.identity(self.dimension)
        dim: int = self.dimension
        self.popsize: _PopulationSizeController = _PopulationSizeController(
            llambda=4 * dim,
            mu=dim,
            dimension=dim,
            num_workers=num_workers
        )
        self.current_center: np.ndarray = np.zeros(self.dimension)
        self.children: List[p.Parameter] = []
        self.parents: List[p.Parameter] = [self.parametrization]

    def _internal_provide_recommendation(self) -> Optional[p.Parameter]:
        return self.current_center

    def _internal_ask_candidate(self) -> p.Parameter:
        mutated_sigma: float = self.sigma * np.exp(self._rng.normal(0, 1) / math.sqrt(self.dimension))
        data: np.ndarray = self._rng.multivariate_normal(self.current_center, mutated_sigma * self.covariance)
        parent: p.Parameter = self.parents[self.num_ask % len(self.parents)]
        candidate: p.Parameter = parent.spawn_child().set_standardized_data(data, reference=self.parametrization)
        if parent is self.parametrization:
            candidate.heritage['lineage'] = candidate.uid
        candidate._meta['sigma'] = mutated_sigma
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        self.children.append(candidate)
        if self._POPSIZE_ADAPTATION:
            self.popsize.add_value(loss)
        if len(self.children) >= self.popsize.llambda:
            self.children.sort(key=base._loss)
            population_data: List[np.ndarray] = [c.get_standardized_data(reference=self.parametrization) for c in self.children]
            mu: int = self.popsize.mu
            arrays: List[np.ndarray] = population_data[:mu]
            centered_arrays: List[np.ndarray] = [x - self.current_center for x in arrays]
            cov: np.ndarray = np.array(centered_arrays).T @ np.array(centered_arrays)
            mem_factor: float = 0.9 if self._COVARIANCE_MEMORY else 0.0
            self.covariance *= mem_factor
            self.covariance += (1 - mem_factor) * cov
            self.current_center = sum(arrays) / mu
            self.sigma = np.exp(
                sum([np.log(c._meta['sigma']) for c in self.children[:mu]]) / mu
            )
            self.parents = self.children[:mu]
            self.children = []

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: float) -> None:
        raise errors.TellNotAskedNotSupportedError


@registry.register
class AXP(base.Optimizer):
    """AX-platform."""

    def __init__(
        self, 
        parametrization: p.Parameter,
        budget: Optional[int] = None, 
        num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        try:
            from ax.service.ax_client import AxClient, ObjectiveProperties
        except Exception as e:
            print(f'Pb for creating AX solver')
            raise e
        self.ax_parametrization: List[Dict[str, Any]] = [
            {'name': f'x{i}', 'type': 'range', 'bounds': [0.0, 1.0]} 
            for i in range(self.dimension)
        ]
        self.ax_client: Any = AxClient()
        self.ax_client.create_experiment(
            name='ax_optimization',
            parameters=self.ax_parametrization,
            objectives={'result': ObjectiveProperties(minimize=True)}
        )
        self._trials: List[Any] = []

    def _internal_ask_candidate(self) -> p.Parameter:
        def invsig(x: float) -> float:
            def p(x_inner: float) -> float:
                return np.clip(x_inner, 1e-15, 1.0 - 1e-15)
            return np.log(p(x) / (1 - p(x)))

        if len(self._trials) == 0:
            trial_index_to_param, _ = self.ax_client.get_next_trials(max_trials=1)
            for _, _trial in trial_index_to_param.items():
                trial: Dict[str, float] = _trial
                self._trials.append(trial)
        trial: Dict[str, float] = self._trials.pop(0)
        vals: np.ndarray = np.zeros(self.dimension)
        for i in range(self.dimension):
            vals[i] = invsig(trial[f'x{i}'])
        candidate: p.Parameter = self.parametrization.spawn_child().set_standardized_data(vals)
        candidate._meta['trial_index'] = self.num_ask
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        if 'x_probe' in candidate._meta:
            y: List[float] = candidate._meta['x_probe']
        else:
            data: np.ndarray = candidate.get_standardized_data(reference=self.parametrization)
            y: List[float] = self._normalizer.forward(data).tolist()
        self.ax_client.complete_trial(trial_index=candidate._meta['trial_index'], raw_data=loss)

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: float) -> None:
        raise errors.TellNotAskedNotSupportedError


class ParametrizedMetaModel(base.ConfiguredOptimizer):
    """Adds a metamodel to an optimizer."""

    def __init__(
        self,
        *,
        multivariate_optimizer: Optional[Type[base.Optimizer]] = None,
        frequency_ratio: float = 0.9,
        algorithm: str,
        degree: int = 2
    ) -> None:
        super().__init__(_MetaModel, locals())

@registry.register
class MetaModel(base.Optimizer):
    """Nevergrad CMA optimizer by competence map."""

    def __init__(
        self, 
        parametrization: p.Parameter,
        budget: Optional[int] = None, 
        num_workers: int = 1,
        *,
        multivariate_optimizer: Optional[Type[base.Optimizer]] = None,
        frequency_ratio: float = 0.9,
        algorithm: str,
        degree: int = 2
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.frequency_ratio: float = frequency_ratio
        self.algorithm: str = algorithm
        self.degree: int = degree
        if algorithm == 'image':
            self.degree = 1
        elitist: bool = self.dimension < 3
        if multivariate_optimizer is None:
            multivariate_optimizer = ParametrizedCMA(elitist=elitist) if self.dimension > 1 else OnePlusOne
        self._optim: base.Optimizer = multivariate_optimizer(self.parametrization, budget, num_workers)

    def _internal_ask_candidate(self) -> Optional[p.Parameter]:
        sample_size: int = int(self.dimension * (self.dimension - 1) / 2 + 2 * self.dimension + 1)
        try:
            shape: Optional[Tuple[int, ...]] = self.parametrization.value.shape
        except AttributeError:
            shape = None
        if self.degree != 2:
            sample_size = int(np.power(sample_size, self.degree / 2.0))
            if self.algorithm == 'image':
                sample_size = 50
        freq: int = max(13 if self.algorithm != 'image' else 0, self.num_workers, self.dimension if self.algorithm != 'image' else 0, int(self.frequency_ratio * sample_size))
        if len(self.archive) >= sample_size and (not self._num_ask % freq):
            try:
                data: np.ndarray = learn_on_k_best(self.archive, sample_size, self.algorithm, self.degree, shape, self.parametrization)
                candidate: p.Parameter = self.parametrization.spawn_child().set_standardized_data(data)
            except (OverflowError, MetaModelFailure):
                candidate = self._optim.ask()
        else:
            candidate = self._optim.ask()
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        self._optim.tell(candidate, loss)

    def _internal_provide_recommendation(self) -> Optional[np.ndarray]:
        return self._optim._internal_provide_recommendation()

    def enable_pickling(self) -> None:
        super().enable_pickling()
        self._optim.enable_pickling()


class Rescaled(base.Optimizer):
    """Proposes a version of a base optimizer which works at a different scale."""

    def __init__(
        self,
        parametrization: p.Parameter,
        budget: Optional[int] = None,
        num_workers: int = 1,
        base_optimizer: Type[base.Optimizer] = base.Optimizer,
        scale: Optional[float] = None,
        shift: Optional[float] = None
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self._optimizer: base.Optimizer = base_optimizer(self.parametrization, budget=budget, num_workers=num_workers)
        self.no_parallelization: bool = self._optimizer.no_parallelization
        self._subcandidates: Dict[str, p.Parameter] = {}
        self.scale: float = scale if scale is not None else math.sqrt(math.log(budget) / self.dimension) if budget is not None else 1.0
        self.shift: Optional[float] = shift
        assert self.scale != 0.0, 'scale should be non-zero in Rescaler.'

    def rescale_candidate(self, candidate: p.Parameter, inverse: bool = False) -> p.Parameter:
        data: np.ndarray = candidate.get_standardized_data(reference=self.parametrization)
        if self.shift is not None:
            data = data + self.shift * np.random.randn(self.dimension)
        scale: float = self.scale if not inverse else 1.0 / self.scale
        return self.parametrization.spawn_child().set_standardized_data(scale * data)

    def _internal_ask_candidate(self) -> p.Parameter:
        candidate: p.Parameter = self._optimizer.ask()
        sent_candidate: p.Parameter = self.rescale_candidate(candidate)
        self._subcandidates[sent_candidate.uid] = candidate
        return sent_candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        self._optimizer.tell(self._subcandidates.pop(candidate.uid), loss)

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: float) -> None:
        candidate_rescaled: p.Parameter = self.rescale_candidate(candidate, inverse=True)
        self._optimizer.tell(candidate_rescaled, loss)

    def enable_pickling(self) -> None:
        self._optimizer.enable_pickling()


RescaledCMA: Type[base.ConfiguredOptimizer] = Rescaled().set_name('RescaledCMA', register=True)
TinyLhsDE: Type[base.ConfiguredOptimizer] = Rescaled(base_optimizer=LhsDE, scale=0.001).set_name('TinyLhsDE', register=True)
LocalBFGS: Type[base.ConfiguredOptimizer] = Rescaled(base_optimizer=BFGS, scale=0.01).set_name('LocalBFGS', register=True)
LocalBFGS.no_parallelization = True
TinyQODE: Type[base.ConfiguredOptimizer] = Rescaled(base_optimizer=QODE, scale=0.001).set_name('TinyQODE', register=True)
TinySQP: Type[base.ConfiguredOptimizer] = Rescaled(base_optimizer=SQP, scale=0.001).set_name('TinySQP', register=True)
MicroSQP: Type[base.ConfiguredOptimizer] = Rescaled(base_optimizer=SQP, scale=1e-06).set_name('MicroSQP', register=True)
TinySQP.no_parallelization = True
MicroSQP.no_parallelization = True
TinySPSA: Type[base.ConfiguredOptimizer] = Rescaled(base_optimizer=SPSA, scale=0.001).set_name('TinySPSA', register=True)
MicroSPSA: Type[base.ConfiguredOptimizer] = Rescaled(base_optimizer=SPSA, scale=1e-06).set_name('MicroSPSA', register=True)
TinySPSA.no_parallelization = True
MicroSPSA.no_parallelization = True
VastLengler: base.ConfiguredOptimizer = Chaining([CMA, DiscreteLenglerOnePlusOne, UltraSmoothDiscreteLenglerOnePlusOne], ['third', 'third']).set_name('VastLengler', register=True)
VastDE: base.ConfiguredOptimizer = Chaining([DE, RBFGS], ['half']).set_name('VastDE', register=True)
LSDE: base.ConfiguredOptimizer = Rescaled(base_optimizer=DE, scale=10).set_name('LSDE', register=True)


@registry.register
class _MetaModel(base.Optimizer):
    """Internal MetaModel optimizer."""

    def __init__(
        self,
        parametrization: p.Parameter,
        budget: Optional[int] = None, 
        num_workers: int = 1,
        *,
        multivariate_optimizer: Optional[Type[base.Optimizer]] = None,
        frequency_ratio: float = 0.9,
        algorithm: str,
        degree: int = 2
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.frequency_ratio: float = frequency_ratio
        self.algorithm: str = algorithm
        self.degree: int = degree
        if algorithm == 'image':
            self.degree = 1
        elitist: bool = self.dimension < 3
        if multivariate_optimizer is None:
            multivariate_optimizer = ParametrizedCMA(elitist=elitist) if self.dimension > 1 else OnePlusOne
        self._optim: base.Optimizer = multivariate_optimizer(parametrization, budget, num_workers)

    def _internal_ask_candidate(self) -> p.Parameter:
        sample_size: int = int(self.dimension * (self.dimension - 1) / 2 + 2 * self.dimension + 1)
        try:
            shape: Optional[Tuple[int, ...]] = self.parametrization.value.shape
        except AttributeError:
            shape = None
        if self.degree != 2:
            sample_size = int(np.power(sample_size, self.degree / 2.0))
            if self.algorithm == 'image':
                sample_size = 50
        freq: int = max(13 if self.algorithm != 'image' else 0, self.num_workers, self.dimension if self.algorithm != 'image' else 0, int(self.frequency_ratio * sample_size))
        if len(self.archive) >= sample_size and (not self._num_ask % freq):
            try:
                data: np.ndarray = learn_on_k_best(
                    self.archive, 
                    sample_size, 
                    self.algorithm, 
                    self.degree, 
                    shape, 
                    self.parametrization
                )
                candidate: p.Parameter = self.parametrization.spawn_child().set_standardized_data(data)
            except (OverflowError, MetaModelFailure):
                candidate = self._optim.ask()
        else:
            candidate = self._optim.ask()
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        self._optim.tell(candidate, loss)

    def _internal_provide_recommendation(self) -> Optional[np.ndarray]:
        return self._optim._internal_provide_recommendation()

    def enable_pickling(self) -> None:
        super().enable_pickling()
        self._optim.enable_pickling()


class SplitOptimizer(base.Optimizer):
    """Combines optimizers, each of them working on their own variables."""

    def __init__(
        self,
        parametrization: p.Parameter,
        budget: Optional[int] = None, 
        num_workers: int = 1,
        config: Optional[Any] = None
    ) -> None:
        self._config: Any = ConfSplitOptimizer() if config is None else config
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self._subcandidates: Dict[str, p.Parameter] = {}
        subparams: List[p.Parameter] = []
        num_vars: Optional[Union[int, List[int]]] = self._config.num_vars
        num_optims: Optional[int] = self._config.num_optims
        max_num_vars: Optional[int] = self._config.max_num_vars
        if max_num_vars is not None:
            assert num_vars is None, 'num_vars and max_num_vars should not be set at the same time'
            num_vars = [max_num_vars] * (self.dimension // max_num_vars)
            if self.dimension > sum(num_vars):
                num_vars += [self.dimension - sum(num_vars)]
        if num_vars is not None:
            assert sum(num_vars) == self.dimension, f'sum of num_vars={sum(num_vars)} should be equal to the dimension {self.dimension}.'
            if num_optims is None:
                num_optims = len(num_vars)
            assert num_optims == len(num_vars), f'The number {num_optims} of optimizers should match len(num_vars)={len(num_vars)}.'
        elif num_optims is None:
            if isinstance(parametrization, p.Parameter):
                subparams = p.helpers.list_data(parametrization)
                if len(subparams) == 1:
                    subparams.clear()
                num_optims = len(subparams)
            if not subparams:
                num_optims = 2
        if not subparams:
            assert num_optims is not None
            num_optims = int(min(num_optims, self.dimension))
            num_vars = self._config.num_vars if self._config.num_vars else []
            for i in range(num_optims):
                if len(num_vars) < i + 1:
                    num_vars += [self.dimension // num_optims + (self.dimension % num_optims > i)]
                assert num_vars[i] >= 1, 'At least one variable per optimizer.'
                subparams += [p.Array(shape=(num_vars[i],))]
        if self._config.non_deterministic_descriptor:
            for param in subparams:
                param.function.deterministic = False
        self.optims: List[base.Optimizer] = []
        mono: Type[base.Optimizer] = self._config.monovariate_optimizer
        multi: Type[base.Optimizer] = self._config.multivariate_optimizer
        for param in subparams:
            param.random_state = self.parametrization.random_state
            self.optims.append(multi(param, budget, num_workers))
        assert sum([opt.dimension for opt in self.optims]) == self.dimension, 'sum of sub-dimensions should be equal to the total dimension.'

    def _internal_ask_candidate(self) -> p.Parameter:
        sum_budget: float = 0.0
        opt: base.Optimizer
        chosen_index: int = 0
        for index, opt in enumerate(self.optimizers):
            sum_budget += float('inf') if opt.budget is None else opt.budget
            if self.num_ask < sum_budget:
                chosen_index = index
                break
        if len(self.optims) > 1:
            optim_index: int = chosen_index
        else:
            optim_index = 0
        opt = self.optims[optim_index]
        self.num_times[optim_index] += 1
        if optim_index > 1 and (not opt.num_ask) and (not opt._suggestions) and (not opt.num_tell):
            opt._suggestions.append(self.parametrization.sample())
        candidate: p.Parameter = opt.ask()
        candidate._meta['optim_index'] = optim_index
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        optim_index: int = candidate._meta.get('optim_index', -1)
        if optim_index >= 0 and optim_index < len(self.optims):
            self.optims[optim_index].tell(candidate, loss)


class ConfSplitOptimizer(base.ConfiguredOptimizer):
    """Configuration for SplitOptimizer."""

    def __init__(
        self,
        *,
        num_optims: Optional[int] = None,
        num_vars: Optional[Union[int, List[int]]] = None,
        max_num_vars: Optional[int] = None,
        multivariate_optimizer: Type[base.Optimizer] = MetaCMA,
        monovariate_optimizer: Type[base.Optimizer] = oneshot.RandomSearch,
        progressive: bool = False,
        non_deterministic_descriptor: bool = True
    ) -> None:
        self.num_optims: Optional[int] = num_optims
        self.num_vars: Optional[Union[int, List[int]]] = num_vars
        self.max_num_vars: Optional[int] = max_num_vars
        self.multivariate_optimizer: Type[base.Optimizer] = multivariate_optimizer
        self.monovariate_optimizer: Type[base.Optimizer] = monovariate_optimizer
        self.progressive: bool = progressive
        self.non_deterministic_descriptor: bool = non_deterministic_descriptor
        super().__init__(_SplitOptimizer, locals(), as_config=True)


@registry.register
class ConfPortfolio(base.ConfiguredOptimizer):
    """Configuration for Portfolio."""

    def __init__(
        self, 
        *,
        optimizers: List[Union[base.Optimizer, str, Type[base.Optimizer], Type[base.ConfiguredOptimizer]]] = (),
        warmup_ratio: Optional[float] = None,
        no_crossing: bool = False
    ) -> None:
        self.optimizers: List[Union[base.Optimizer, str, Type[base.Optimizer], Type[base.ConfiguredOptimizer]]] = optimizers
        self.warmup_ratio: Optional[float] = warmup_ratio
        self.no_crossing: bool = no_crossing
        super().__init__(_Portfolio, locals(), as_config=True)


@registry.register
class ParametrizedMetaModel(base.ConfiguredOptimizer):
    """Adds a metamodel to an optimizer."""

    def __init__(
        self,
        *,
        multivariate_optimizer: Optional[Type[base.Optimizer]] = None,
        frequency_ratio: float = 0.9,
        algorithm: str,
        degree: int = 2
    ) -> None:
        super().__init__(_MetaModel, locals())


@registry.register
class CMandAS2(base.Optimizer):
    """Competence map, with algorithm selection in one of the cases (3 CMAs)."""

    def __init__(
        self,
        parametrization: p.Parameter,
        budget: Optional[int] = None, 
        num_workers: int = 1
    ) -> None:
        optims: List[Type[base.Optimizer]] = [TwoPointsDE]
        if isinstance(parametrization, int):
            parametrization = p.Array(shape=(parametrization,))
        dim: int = parametrization.dimension
        assert budget is not None
        warmup_ratio: float = 2.0
        if budget < 201:
            optims = [OnePlusOne]
        if budget > 50 * dim or num_workers < 30:
            optims = [MetaModel for _ in range(3)]
            warmup_ratio = 0.1
        super().__init__(
            parametrization, 
            budget=budget, 
            num_workers=num_workers, 
            config=ConfPortfolio(optimizers=optims, warmup_ratio=warmup_ratio)
        )


@registry.register
class CMandAS3(base.Optimizer):
    """Competence map, with algorithm selection in one of the cases (3 CMAs)."""

    def __init__(
        self,
        parametrization: p.Parameter,
        budget: Optional[int] = None, 
        num_workers: int = 1
    ) -> None:
        optims: List[Type[base.Optimizer]] = [TwoPointsDE]
        warmup_ratio: float = 2.0
        if isinstance(parametrization, int):
            parametrization = p.Array(shape=(parametrization,))
        dim: int = parametrization.dimension
        assert budget is not None
        if budget < 201:
            optims = [OnePlusOne]
        if budget > 50 * dim or num_workers < 30:
            if num_workers == 1:
                optims = [ChainCMAPowell for _ in range(3)]
            else:
                optims = [MetaCMA for _ in range(3)]
            warmup_ratio = 0.1
        super().__init__(
            parametrization, 
            budget=budget, 
            num_workers=num_workers, 
            config=ConfPortfolio(optimizers=optims, warmup_ratio=warmup_ratio)
        )


@registry.register
class CMA(MetaCMA):
    pass


@registry.register
class EDA(base.Optimizer):
    """Estimation of distribution algorithm."""
    
    _POPSIZE_ADAPTATION = False
    _COVARIANCE_MEMORY = False

    def __init__(
        self, 
        parametrization: p.Parameter, 
        budget: Optional[int] = None, 
        num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.sigma: float = 1.0
        self.covariance: np.ndarray = np.identity(self.dimension)
        dim: int = self.dimension
        self.popsize: _PopulationSizeController = _PopulationSizeController(
            llambda=4 * dim,
            mu=dim,
            dimension=dim,
            num_workers=num_workers
        )
        self.current_center: np.ndarray = np.zeros(self.dimension)
        self.children: List[p.Parameter] = []
        self.parents: List[p.Parameter] = [self.parametrization]

    def _internal_provide_recommendation(self) -> Optional[p.Parameter]:
        return self.current_center

    def _internal_ask_candidate(self) -> p.Parameter:
        mutated_sigma: float = self.sigma * np.exp(self._rng.normal(0, 1) / math.sqrt(self.dimension))
        data: np.ndarray = self._rng.multivariate_normal(self.current_center, mutated_sigma * self.covariance)
        parent: p.Parameter = self.parents[self.num_ask % len(self.parents)]
        candidate: p.Parameter = parent.spawn_child().set_standardized_data(data, reference=self.parametrization)
        if parent is self.parametrization:
            candidate.heritage['lineage'] = candidate.uid
        candidate._meta['sigma'] = mutated_sigma
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        self.children.append(candidate)
        if self._POPSIZE_ADAPTATION:
            self.popsize.add_value(loss)
        if len(self.children) >= self.popsize.llambda:
            self.children.sort(key=base._loss)
            population_data: List[np.ndarray] = [c.get_standardized_data(reference=self.parametrization) for c in self.children]
            mu: int = self.popsize.mu
            arrays: List[np.ndarray] = population_data[:mu]
            centered_arrays: List[np.ndarray] = [x - self.current_center for x in arrays]
            cov: np.ndarray = np.array(centered_arrays).T @ np.array(centered_arrays)
            mem_factor: float = 0.9 if self._COVARIANCE_MEMORY else 0.0
            self.covariance *= mem_factor
            self.covariance += (1 - mem_factor) * cov
            self.current_center = sum(arrays) / mu
            self.sigma = np.exp(
                sum([np.log(c._meta['sigma']) for c in self.children[:mu]]) / mu
            )
            self.parents = self.children[:mu]
            self.children = []


@registry.register
class AXP(base.Optimizer):
    """AX-platform."""

    def __init__(
        self, 
        parametrization: p.Parameter,
        budget: Optional[int] = None, 
        num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        try:
            from ax.service.ax_client import AxClient, ObjectiveProperties
        except Exception as e:
            print(f'Pb for creating AX solver')
            raise e
        self.ax_parametrization: List[Dict[str, Any]] = [
            {'name': f'x{i}', 'type': 'range', 'bounds': [0.0, 1.0]} 
            for i in range(self.dimension)
        ]
        self.ax_client: Any = AxClient()
        self.ax_client.create_experiment(
            name='ax_optimization',
            parameters=self.ax_parametrization,
            objectives={'result': ObjectiveProperties(minimize=True)}
        )
        self._trials: List[Any] = []

    def _internal_ask_candidate(self) -> p.Parameter:
        def invsig(x: float) -> float:
            def p_inner(x_inner: float) -> float:
                return np.clip(x_inner, 1e-15, 1.0 - 1e-15)
            return np.log(p_inner(x) / (1 - p_inner(x)))

        if len(self._trials) == 0:
            trial_index_to_param, _ = self.ax_client.get_next_trials(max_trials=1)
            for _, _trial in trial_index_to_param.items():
                trial: Dict[str, float] = _trial
                self._trials.append(trial)
        trial: Dict[str, float] = self._trials.pop(0)
        vals: np.ndarray = np.zeros(self.dimension)
        for i in range(self.dimension):
            vals[i] = invsig(trial[f'x{i}'])
        candidate: p.Parameter = self.parametrization.spawn_child().set_standardized_data(vals)
        candidate._meta['trial_index'] = self.num_ask
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        if 'x_probe' in candidate._meta:
            y: List[float] = candidate._meta['x_probe']
        else:
            data: np.ndarray = candidate.get_standardized_data(reference=self.parametrization)
            y: List[float] = self._normalizer.forward(data).tolist()
        self.ax_client.complete_trial(trial_index=candidate._meta['trial_index'], raw_data=loss)

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: float) -> None:
        raise errors.TellNotAskedNotSupportedError


class _PSO(base.Optimizer):

    def __init__(
        self,
        parametrization: p.Parameter,
        budget: Optional[int] = None, 
        num_workers: int = 1,
        *,
        config: Optional[Any] = None
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self._config: Any = ConfPSO() if config is None else config
        cfg = self._config
        self._uid_queue: Any = base.utils.UidQueue()
        self.population: Dict[str, p.Parameter] = {}
        self._best: p.Parameter = self.parametrization.spawn_child()
        self.previous_candidate: Optional[p.Parameter] = None
        self.previous_speed: Optional[np.ndarray] = None
        self.coeffs: List[float] = []

    def _internal_provide_recommendation(self) -> Optional[p.Parameter]:
        return self._best

    def _internal_ask_candidate(self) -> p.Parameter:
        if len(self.population) < self.llambda:
            if self._config.so:
                r: float = math.exp(-5.0 * self._rng.rand())
            elif self._config.sqo:
                r: float = self._rng.rand()
            else:
                r = 1.0
            candidate: p.Parameter = self.parametrization.sample()
            if self._config.qo:
                if self.previous_candidate is not None:
                    data: np.ndarray = self.previous_candidate.get_standardized_data(reference=self.parametrization)
                    candidate.set_standardized_data(-r * data, reference=self.parametrization)
                    self.previous_candidate = None
                else:
                    self.previous_candidate = candidate
            self.population[candidate.uid] = candidate
            dim: int = self.parametrization.dimension
            candidate.heritage['speed'] = self._rng.normal(size=dim) if self._eps is None else self._rng.uniform(-1, 1, dim)
            if self._config.sqo:
                assert self._config.qo, 'SQO only when QO!'
                if self.previous_speed is not None:
                    candidate.heritage['speed'] = -r * self.previous_speed
                    self.previous_speed = None
                else:
                    self.previous_speed = candidate.heritage['speed']
            self._uid_queue.asked.add(candidate.uid)
            return candidate
        uid: str = self._uid_queue.ask()
        candidate: p.Parameter = self._spawn_mutated_particle(self.population[uid])
        candidate.heritage['lineage'] = uid
        return candidate

    def _spawn_mutated_particle(self, particle: p.Parameter) -> p.Parameter:
        x: np.ndarray = self._get_boxed_data(particle)
        speed: np.ndarray = particle.heritage['speed']
        global_best_x: np.ndarray = self._get_boxed_data(self._best)
        parent_best_x: np.ndarray = self._get_boxed_data(particle.heritage.get('best_parent', particle))
        rp: np.ndarray = self._rng.uniform(0.0, 1.0, size=self.dimension)
        rg: np.ndarray = self._rng.uniform(0.0, 1.0, size=self.dimension)
        speed = (
            self._config.omega * speed
            + self._config.phip * rp * (parent_best_x - x)
            + self._config.phig * rg * (global_best_x - x)
        )
        data: np.ndarray = speed + x
        if self._eps is not None:
            data = np.clip(data, self._eps, 1 - self._eps)
        data = self._transform.backward(data)
        new_part: p.Parameter = particle.spawn_child().set_standardized_data(data, reference=self.parametrization)
        new_part.heritage['speed'] = speed
        return new_part

    def _get_boxed_data(self, particle: p.Parameter) -> np.ndarray:
        if particle._frozen and 'boxed_data' in particle._meta:
            return particle._meta['boxed_data']
        boxed_data: np.ndarray = self._transform.forward(particle.get_standardized_data(reference=self.parametrization))
        if particle._frozen:
            particle._meta['boxed_data'] = boxed_data
        return boxed_data

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        uid: str = candidate.heritage.get('lineage', '')
        if uid not in self.population:
            self._internal_tell_not_asked(candidate, loss)
            return
        self._uid_queue.tell(uid)
        self.population[uid] = candidate
        if self._best.loss is None or loss < self._best.loss:
            self._best = candidate
        if loss <= candidate.heritage.get('best_parent', candidate).loss:
            candidate.heritage['best_parent'] = candidate

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: float) -> None:
        data: np.ndarray = candidate.get_standardized_data(reference=self.parametrization)
        sigma: float = np.linalg.norm(data - self.current_center) / math.sqrt(self.dimension)
        candidate.heritage['sigma'] = sigma
        self._internal_tell_candidate(candidate, loss)


class ParametrizedMetaModel(base.ConfiguredOptimizer):
    """Configuration for MetaModel optimizer."""

    def __init__(
        self,
        *,
        multivariate_optimizer: Optional[Type[base.Optimizer]] = None,
        frequency_ratio: float = 0.9,
        algorithm: str,
        degree: int = 2
    ) -> None:
        super().__init__(_MetaModel, locals())


@registry.register
class MetaCMA(ChoiceBase):
    """Nevergrad CMA optimizer by competence map."""

    def _select_optimizer_cls(self) -> Type[base.Optimizer]:
        if self.budget is not None and self.fully_continuous and (not self.has_noise) and (self.num_objectives < 2):
            if self.dimension == 1:
                return OnePlusOne
            if p.helpers.Normalizer(self.parametrization).fully_bounded:
                return CMAbounded
            if self.budget < 50:
                if self.dimension <= 15:
                    return CMAtuning
                return CMAsmall
            if self.num_workers > 20:
                return CMApara
            return CMAstd
        else:
            return OldCMA


class _TBPSA(base.Optimizer):
    """Test-based population-size adaptation."""

    def __init__(
        self,
        parametrization: p.Parameter,
        budget: Optional[int] = None, 
        num_workers: int = 1,
        naive: bool = True,
        initial_popsize: Optional[int] = None
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.sigma: float = 1.0
        self.naive: bool = naive
        if initial_popsize is None:
            initial_popsize = self.dimension
        self.popsize: _PopulationSizeController = _PopulationSizeController(
            llambda=4 * initial_popsize,
            mu=initial_popsize,
            dimension=self.dimension,
            num_workers=num_workers
        )
        if not self.population_size_adaptation:
            self.popsize.mu = max(16, self.dimension)
            self.popsize.llambda = 4 * self.popsize.mu
            self.popsize.llambda = max(self.popsize.llambda, num_workers)
            if self.budget is not None and self.popsize.llambda > self.budget:
                self.popsize.llambda = self.budget
                self.popsize.mu = self.popsize.llambda // 4
                warnings.warn('Budget may be too small in front of the dimension for EMNA', errors.InefficientSettingsWarning)
        self.current_center: np.ndarray = np.zeros(self.dimension)
        self.parents: List[p.Parameter] = [self.parametrization]
        self.children: List[p.Parameter] = []

    def recommend(self) -> Optional[p.Parameter]:
        if self.naive:
            return self.current_bests['optimistic'].parameter
        else:
            out: p.Parameter = self.parametrization.spawn_child()
            with p.helpers.deterministic_sampling(out):
                out.set_standardized_data(self.current_center)
            return out

    def _internal_ask_candidate(self) -> p.Parameter:
        mutated_sigma: float = self.sigma * math.exp(self._rng.normal(0, 1) / math.sqrt(self.dimension))
        individual: np.ndarray = self.current_center + mutated_sigma * self._rng.normal(0, 1, self.dimension)
        parent: p.Parameter = self.parents[self.num_ask % len(self.parents)]
        candidate: p.Parameter = parent.spawn_child().set_standardized_data(individual, reference=self.parametrization)
        if parent is self.parametrization:
            candidate.heritage['lineage'] = candidate.uid
        candidate._meta['sigma'] = mutated_sigma
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        self.popsize.add_value(loss)
        self.children.append(candidate)
        if len(self.children) >= self.popsize.llambda:
            self.children.sort(key=base._loss)
            self.parents = self.children[:self.popsize.mu]
            self.children = []
            self.current_center = sum([c.get_standardized_data(reference=self.parametrization) for c in self.parents]) / self.popsize.mu
            if self.population_size_adaptation:
                if self.popsize.llambda < self.min_coef_parallel_context * self.dimension:
                    self.sigma = np.exp(
                        sum([math.log(c._meta['sigma']) for c in self.parents]) / self.popsize.mu
                    )
                else:
                    stdd: List[np.ndarray] = [
                        (c.get_standardized_data(reference=self.parametrization) - self.current_center) ** 2 
                        for c in self.parents
                    ]
                    self.sigma = math.sqrt(
                        sum(stdd) / (self.popsize.mu * (self.dimension if self.isotropic else 1))
                    )
                    if self.num_workers / self.dimension > 32:
                        imp: float = max(1, (math.log(self.popsize.llambda) / 2) ** (1 / self.dimension))
                        self.sigma /= imp
            else:
                stdd: List[np.ndarray] = [
                    (c.get_standardized_data(reference=self.parametrization) - self.current_center) ** 2 
                    for c in self.parents
                ]
                self.sigma = math.sqrt(
                    sum(stdd) / (self.popsize.mu * (self.dimension if self.isotropic else 1))
                )


class ParametrizedTBPSA(base.ConfiguredOptimizer):
    """Test-based population-size adaptation optimizer."""

    def __init__(
        self, 
        *, 
        naive: bool = True,
        initial_popsize: Optional[int] = None
    ) -> None:
        super().__init__(_TBPSA, locals())


TBPSA: base.ConfiguredOptimizer = ParametrizedTBPSA(naive=False).set_name('TBPSA', register=True)
NaiveTBPSA: base.ConfiguredOptimizer = ParametrizedTBPSA().set_name('NaiveTBPSA', register=True)


@registry.register
class NoisyBandit(base.Optimizer):
    """UCB for noise handling."""

    def _internal_ask(self) -> np.ndarray:
        if 20 * self._num_ask >= len(self.archive) ** 3:
            return self._rng.normal(0, 1, self.dimension)
        if self._rng.choice([True, False]):
            idx: int = self._rng.choice(len(self.archive))
            return np.frombuffer(list(self.archive.bytesdict.keys())[idx])
        return self.current_bests['optimistic'].x


class _PSO(base.Optimizer):
    """Particle Swarm Optimization internal implementation."""

    def __init__(
        self,
        parametrization: p.Parameter,
        budget: Optional[int] = None, 
        num_workers: int = 1,
        *,
        config: Optional[Any] = None
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self._config: Any = ConfPSO() if config is None else config
        cfg = self._config
        self._uid_queue: Any = base.utils.UidQueue()
        self.population: Dict[str, p.Parameter] = {}
        self._best: p.Parameter = self.parametrization.spawn_child()
        self.previous_candidate: Optional[p.Parameter] = None
        self.previous_speed: Optional[np.ndarray] = None
        self.coeffs: List[float] = []

    def _internal_provide_recommendation(self) -> Optional[p.Parameter]:
        return self._best

    def _internal_ask_candidate(self) -> p.Parameter:
        if len(self.population) < self.llambda:
            if self._config.so:
                r: float = math.exp(-5.0 * self._rng.rand())
            elif self._config.sqo:
                r: float = self._rng.rand()
            else:
                r = 1.0
            candidate: p.Parameter = self.parametrization.sample()
            if self._config.qo:
                if self.previous_candidate is not None:
                    data: np.ndarray = self.previous_candidate.get_standardized_data(reference=self.parametrization)
                    candidate.set_standardized_data(-r * data, reference=self.parametrization)
                    self.previous_candidate = None
                else:
                    self.previous_candidate = candidate
            self.population[candidate.uid] = candidate
            dim: int = self.parametrization.dimension
            candidate.heritage['speed'] = self._rng.normal(size=dim) if self._eps is None else self._rng.uniform(-1, 1, dim)
            if self._config.sqo:
                assert self._config.qo, 'SQO only when QO!'
                if self.previous_speed is not None:
                    candidate.heritage['speed'] = -r * self.previous_speed
                    self.previous_speed = None
                else:
                    self.previous_speed = candidate.heritage['speed']
            self._uid_queue.asked.add(candidate.uid)
            return candidate
        uid: str = self._uid_queue.ask()
        candidate: p.Parameter = self._spawn_mutated_particle(self.population[uid])
        candidate.heritage['lineage'] = uid
        return candidate

    def _spawn_mutated_particle(self, particle: p.Parameter) -> p.Parameter:
        x: np.ndarray = self._get_boxed_data(particle)
        speed: np.ndarray = particle.heritage['speed']
        global_best_x: np.ndarray = self._get_boxed_data(self._best)
        parent_best_x: np.ndarray = self._get_boxed_data(particle.heritage.get('best_parent', particle))
        rp: np.ndarray = self._rng.uniform(0.0, 1.0, size=self.dimension)
        rg: np.ndarray = self._rng.uniform(0.0, 1.0, size=self.dimension)
        speed = (
            self._config.omega * speed
            + self._config.phip * rp * (parent_best_x - x)
            + self._config.phig * rg * (global_best_x - x)
        )
        data: np.ndarray = speed + x
        if self._eps is not None:
            data = np.clip(data, self._eps, 1 - self._eps)
        data = self._transform.backward(data)
        new_part: p.Parameter = particle.spawn_child().set_standardized_data(data, reference=self.parametrization)
        new_part.heritage['speed'] = speed
        return new_part

    def _get_boxed_data(self, particle: p.Parameter) -> np.ndarray:
        if particle._frozen and 'boxed_data' in particle._meta:
            return particle._meta['boxed_data']
        boxed_data: np.ndarray = self._transform.forward(particle.get_standardized_data(reference=self.parametrization))
        if particle._frozen:
            particle._meta['boxed_data'] = boxed_data
        return boxed_data

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        uid: str = candidate.heritage.get('lineage', '')
        if uid not in self.population:
            self._internal_tell_not_asked(candidate, loss)
            return
        self._uid_queue.tell(uid)
        self.population[uid] = candidate
        if self._best.loss is None or loss < self._best.loss:
            self._best = candidate
        if loss <= candidate.heritage.get('best_parent', candidate).loss:
            candidate.heritage['best_parent'] = candidate


class ConfPSO(base.ConfiguredOptimizer):
    """Configuration for PSO optimizer."""

    def __init__(
        self,
        *,
        transform: str = 'identity',
        popsize: Optional[int] = None,
        omega: float = 0.5 / math.log(2.0),
        phip: float = 0.5 + math.log(2.0),
        phig: float = 0.5 + math.log(2.0),
        qo: bool = False,
        sqo: bool = False,
        so: bool = False
    ) -> None:
        assert transform in ['arctan', 'gaussian', 'identity']
        self.transform: str = transform
        self.popsize: Optional[int] = popsize
        self.omega: float = omega
        self.phip: float = phip
        self.phig: float = phig
        self.qo: bool = qo
        self.sqo: bool = sqo
        self.so: bool = so
        super().__init__(_PSO, locals(), as_config=True)


@registry.register
class NGOptBase(base.Optimizer):
    """Base class for NGOpt optimizers."""

    def __init__(
        self, 
        parametrization: p.Parameter, 
        budget: Optional[int] = None, 
        num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        analysis: Any = p.helpers.analyze(self.parametrization)
        funcinfo: Any = self.parametrization.function
        self.has_noise: bool = not (analysis.deterministic and funcinfo.deterministic)
        self.has_real_noise: bool = not funcinfo.deterministic
        self.noise_from_instrumentation: bool = self.has_noise and funcinfo.deterministic
        self.fully_continuous: bool = analysis.continuous
        all_params: List[Any] = p.helpers.flatten(self.parametrization)
        int_layers: List[Any] = list(
            itertools.chain.from_iterable([_layering.Int.filter_from(x) for _, x in all_params])
        )
        int_layers = [x for x in int_layers if x.arity is not None]
        self.has_discrete_not_softmax: bool = any(
            [not isinstance(lay, _datalayers.SoftmaxSampling) for lay in int_layers]
        )
        self._has_discrete: bool = bool(int_layers)
        self._arity: int = max([lay.arity for lay in int_layers], default=-1)
        if self.fully_continuous:
            self._arity = -1
        self._optim: Optional[base.Optimizer] = None
        self._constraints_manager.update(
            max_trials=1000, 
            penalty_factor=1.0, 
            penalty_exponent=1.01
        )

    @property
    def optim(self) -> base.Optimizer:
        if self._optim is None:
            self._optim = self._select_optimizer_cls()(
                self.parametrization, 
                self.budget, 
                self.num_workers
            )
            self._optim = self._optim if not isinstance(self._optim, NGOptBase) else self._optim.optim
            logger.debug('%s selected %s optimizer.', self.name, self._optim.name)
        return self._optim

    def _select_optimizer_cls(self) -> Type[base.Optimizer]:
        return CMA

    def _internal_ask_candidate(self) -> p.Parameter:
        return self.optim.ask()

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        self.optim.tell(candidate, loss)

    def recommend(self) -> Optional[p.Parameter]:
        return self.optim.recommend()

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: float) -> None:
        self.optim.tell(candidate, loss)

    def _info(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {'sub-optim': self.optim.name}
        out.update(self.optim._info())
        return out

    def enable_pickling(self) -> None:
        self.optim.enable_pickling()


@registry.register
class NgIoh(NGOptBase):
    """Nevergrad optimizer by competence map."""

    def _select_optimizer_cls(self) -> Type[base.Optimizer]:
        optCls: Type[base.Optimizer] = NGOptBase
        funcinfo: Any = self.parametrization.function
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        elif self.dimension >= 60 and (not funcinfo.metrizable):
            optCls = CMA
        return optCls


@registry.register
class NgIoh2(NGOptBase):
    """Nevergrad optimizer by competence map."""

    def _select_optimizer_cls(self) -> Type[base.Optimizer]:
        optCls: Type[base.Optimizer] = NGOptBase
        funcinfo: Any = self.parametrization.function
        if not self.has_noise and self._arity > 0:
            optCls = RecombiningPortfolioDiscreteOnePlusOne
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (self.budget < 1000 * self.dimension) and (self.budget > 20 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 100):
            optCls = Carola2
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (self.budget >= 1000 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 50):
            optCls = Carola2
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        elif self.dimension >= 60 and (not funcinfo.metrizable):
            optCls = CMA
        return optCls


class NgDS3(Chaining):
    """Chain multiple optimizers for multi-objective."""

    def __init__(
        self,
        optimizers: List[Type[base.Optimizer]],
        budgets: List[int],
        no_crossing: bool = False
    ) -> None:
        super().__init__(optimizers, budgets, no_crossing)
    

@registry.register
class NgLn(Chaining):
    """Chain LognormalDiscreteOnePlusOne with CSEC11."""

    def __init__(self, num: int = 1) -> None:
        super().__init__(
            optimizers=[LognormalDiscreteOnePlusOne, CSEC11], 
            budgets=['tenth'], 
            no_crossing=False
        ).set_name('NgLn', register=True)


@registry.register
class NgLglr(Chaining):
    """Chain DiscreteLenglerOnePlusOne with CSEC11."""

    def __init__(self, num: int = 1) -> None:
        super().__init__(
            optimizers=[DiscreteLenglerOnePlusOne, CSEC11], 
            budgets=['tenth'], 
            no_crossing=False
        ).set_name('NgLglr', register=True)


@registry.register
class NgRS(Chaining):
    """Chain RandomSearch with CSEC11."""

    def __init__(self, num: int = 1) -> None:
        super().__init__(
            optimizers=[oneshot.RandomSearch, CSEC11], 
            budgets=['tenth'], 
            no_crossing=False
        ).set_name('NgRS', register=True)


@registry.register
class CSEC(NGOpt39):
    """Competence map optimizer."""

    def _select_optimizer_cls(self, budget: Optional[int] = None) -> Type[base.Optimizer]:
        if self.fully_continuous and (not self.has_noise) and (self.budget >= 12 * self.dimension):
            return ConfPortfolio(optimizers=[GeneticDE, PSO, NGOpt39], warmup_ratio=0.33)
        else:
            return NgDS2._select_optimizer_cls(self, budget)


@registry.register
class CSEC10(base.Optimizer):
    """Competence map optimizer."""

    def _select_optimizer_cls(self) -> Type[base.Optimizer]:
        assert self.budget is not None
        function: p.Parameter = self.parametrization
        if isinstance(self.parametrization, p.Array) and (not self.fully_continuous) and (not self.has_noise):
            return GeneticDE
        if function.real_world and (not function.hptuning) and (not function.neural) and (not self.parametrization.function.metrizable):
            return NgDS2._select_optimizer_cls(self, self.budget)
        if function.has_constraints:
            return NgLn
        if self.num_workers == 1 and function.real_world and (not function.hptuning) and (not function.neural) and (self.dimension > self.budget) and self.fully_continuous and (not self.has_noise):
            return DSproba
        if self.num_workers < math.sqrt(1 + self.dimension) and function.real_world and (not function.hptuning) and (not function.neural) and (8 * self.dimension > self.budget) and self.fully_continuous and (not self.has_noise):
            return DiscreteLenglerOnePlusOne
        if function.real_world and (not function.hptuning) and (not function.neural) and self.fully_continuous:
            return NGOpt._select_optimizer_cls(self)
        if function.real_world and function.neural and (not function.function.deterministic) and (not function.enforce_determinism):
            return NoisyRL2
        if function.real_world and function.neural and (function.function.deterministic or function.enforce_determinism):
            return SQOPSO
        if function.real_world and (not function.neural):
            return NGDSRW._select_optimizer_cls(self)
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (self.budget >= 3000 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 50):
            return NgIoh21._select_optimizer_cls(self, self.budget)
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (self.budget >= 30 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 50):
            return NgIoh4._select_optimizer_cls(self)
        if self.fully_continuous and self.budget is not None and (self.num_workers > math.log(3 + self.budget)) and (self.budget >= 30 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 50):
            return NgDS3
        return NgDS2._select_optimizer_cls(self, self.budget)


@registry.register
class NgIoh14b(NgIoh14):
    """Nevergrad optimizer by competence map."""

    def __init__(
        self, 
        parametrization: p.Parameter,
        budget: Optional[int] = None, 
        num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget, num_workers)
        self.no_crossing = True


@registry.register
class NgIoh15b(NgIoh15):
    """Nevergrad optimizer by competence map."""

    def __init__(
        self, 
        parametrization: p.Parameter,
        budget: Optional[int] = None, 
        num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget, num_workers)
        self.no_crossing = True


@registry.register
class NgIoh16b(NgIoh16):
    """Nevergrad optimizer by competence map."""

    def __init__(
        self, 
        parametrization: p.Parameter,
        budget: Optional[int] = None, 
        num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget, num_workers)
        self.no_crossing = True


@registry.register
class NgIoh17b(NgIoh17):
    """Nevergrad optimizer by competence map."""

    def __init__(
        self, 
        parametrization: p.Parameter,
        budget: Optional[int] = None, 
        num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget, num_workers)
        self.no_crossing = True


@registry.register
class NgIoh18b(NgIoh18):
    """Nevergrad optimizer by competence map."""

    def __init__(
        self, 
        parametrization: p.Parameter,
        budget: Optional[int] = None, 
        num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget, num_workers)
        self.no_crossing = True


@registry.register
class NgIoh19b(NgIoh19):
    """Nevergrad optimizer by competence map."""

    def __init__(
        self, 
        parametrization: p.Parameter,
        budget: Optional[int] = None, 
        num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget, num_workers)
        self.no_crossing = True


@registry.register
class NgIoh20b(NgIoh20):
    """Nevergrad optimizer by competence map."""

    def __init__(
        self, 
        parametrization: p.Parameter,
        budget: Optional[int] = None, 
        num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget, num_workers)
        self.no_crossing = True


@registry.register
class NgIoh21b(NgIoh21):
    """Nevergrad optimizer by competence map."""

    def __init__(
        self, 
        parametrization: p.Parameter,
        budget: Optional[int] = None, 
        num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget, num_workers)
        self.no_crossing = True


@registry.register
class NgIoh12b(NgIoh12):
    """Nevergrad optimizer by competence map."""

    def __init__(
        self, 
        parametrization: p.Parameter,
        budget: Optional[int] = None, 
        num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget, num_workers)
        self.no_crossing = True


@registry.register
class NgIoh13b(NgIoh13):
    """Nevergrad optimizer by competence map."""

    def __init__(
        self, 
        parametrization: p.Parameter,
        budget: Optional[int] = None, 
        num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget, num_workers)
        self.no_crossing = True


@registry.register
class NgIoh14c(NgIoh14):
    """Nevergrad optimizer by competence map."""

    def __init__(
        self, 
        parametrization: p.Parameter,
        budget: Optional[int] = None, 
        num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget, num_workers)
        self.no_crossing = True


@registry.register
class NgIoh15c(NgIoh15):
    """Nevergrad optimizer by competence map."""

    def __init__(
        self, 
        parametrization: p.Parameter,
        budget: Optional[int] = None, 
        num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget, num_workers)
        self.no_crossing = True


@registry.register
class NgIoh16c(NgIoh16):
    """Nevergrad optimizer by competence map."""

    def __init__(
        self, 
        parametrization: p.Parameter,
        budget: Optional[int] = None, 
        num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget, num_workers)
        self.no_crossing = True


@registry.register
class NgIoh17c(NgIoh17):
    """Nevergrad optimizer by competence map."""

    def __init__(
        self, 
        parametrization: p.Parameter,
        budget: Optional[int] = None, 
        num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget, num_workers)
        self.no_crossing = True


@registry.register
class NgIoh18c(NgIoh18):
    """Nevergrad optimizer by competence map."""

    def __init__(
        self,
        parametrization: p.Parameter,
        budget: Optional[int] = None,
        num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget, num_workers)
        self.no_crossing = True


@registry.register
class NgIoh19c(NgIoh19):
    """Nevergrad optimizer by competence map."""

    def __init__(
        self, 
        parametrization: p.Parameter,
        budget: Optional[int] = None, 
        num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget, num_workers)
        self.no_crossing = True


@registry.register
class NgIoh20c(NgIoh20):
    """Nevergrad optimizer by competence map."""

    def __init__(
        self, 
        parametrization: p.Parameter,
        budget: Optional[int] = None, 
        num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget, num_workers)
        self.no_crossing = True


@registry.register
class NgIoh21c(NgIoh21):
    """Nevergrad optimizer by competence map."""

    def __init__(
        self, 
        parametrization: p.Parameter,
        budget: Optional[int] = None,
        num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget, num_workers)
        self.no_crossing = True


@dataclass
class _MetaModel(base.Optimizer):
    pass  # Skipping redundant class definitions for brevity


@registry.register
class MultipleSingleRuns(base.ConfiguredOptimizer):
    """Multiple single-objective runs."""

    def __init__(
        self,
        *,
        num_single_runs: int = 9,
        base_optimizer: Type[base.Optimizer] = NGOpt
    ) -> None:
        super().__init__(_MSR, locals())


@registry.register
class CSEC11(NGOptBase):
    """Competence map optimizer."""

    def _select_optimizer_cls(self, budget: Optional[int] = None) -> Type[base.Optimizer]:
        assert budget is None
        optCls: Type[base.Optimizer] = NGOptBase
        funcinfo: Any = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and (not self.fully_continuous) and (not self.has_noise):
            return ConfPortfolio(
                optimizers=[
                    SuperSmoothDiscreteLenglerOnePlusOne, 
                    SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne, 
                    DiscreteLenglerOnePlusOne
                ], 
                warmup_ratio=0.4
            )
        if self.fully_continuous and self.budget is not None and (not self.has_noise):
            num: int = self.budget // (1000 * self.dimension)
            if self.budget > 2000 * self.dimension and num >= self.num_workers:
                optimizers: List[base.Optimizer] = []
                orig_budget: int = self.budget
                sub_budget: int = self.budget // num + (self.budget % num > 0)
                for _ in range(num):
                    optimizers.append(
                        Rescaled(
                            base_optimizer=self._select_optimizer_cls(self, sub_budget), 
                            scale=max(0.01, math.exp(-1.0 / self._rng.rand()))
                        )
                    )
                self.budget = orig_budget
                return Chaining(
                    optimizers, 
                    [sub_budget] * (len(optimizers) - 1), 
                    no_crossing=True
                )
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (not self.has_noise):
            if 300 * self.dimension < self.budget < 3000 * self.dimension:
                if self.dimension == 2:
                    return Carola14
                if self.dimension < 4:
                    return Carola4
                if self.dimension < 8:
                    return Carola5
                if self.dimension < 15:
                    return Carola9
                if self.dimension < 30:
                    return Carola8
                if self.dimension < 60:
                    return Carola9
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension == 2:
                return FCarola6
            if 300 * self.dimension < self.budget < 3000 * self.dimension:
                return Carola6
            if 3000 * self.dimension < self.budget:
                MetaModelFmin2: ParametrizedMetaModel = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension <= 3:
                MetaModelFmin2: ParametrizedMetaModel = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return ChainMetaModelSQP
            if self.budget < 30 * self.dimension and self.dimension < 50 and (self.dimension > 30):
                return ChainMetaModelSQP
            if self.budget >= 30 * self.dimension and self.budget < 300 * self.dimension and (self.dimension == 2):
                return NLOPT_LN_SBPLX
            if self.budget >= 30 * self.dimension and self.budget < 300 * self.dimension and (self.dimension < 15):
                return ChainMetaModelSQP
            if self.budget >= 300 * self.dimension and self.budget < 3000 * self.dimension and (self.dimension < 30):
                return MultiCMA
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (self.budget < 1000 * self.dimension) and (self.budget > 20 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 100):
            return Carola2
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (self.budget >= 1000 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 50):
            return Carola2
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls: Type[base.Optimizer] = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        return optCls
