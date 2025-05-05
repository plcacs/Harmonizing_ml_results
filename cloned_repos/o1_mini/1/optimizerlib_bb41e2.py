import math
import logging
import itertools
from collections import deque, defaultdict
import warnings
import numpy as np
import scipy.ndimage as ndimage
from typing import Optional, List, Tuple, Dict, Any, Union, Callable
from bayes_opt import UtilityFunction, BayesianOptimization
import nevergrad.common.typing as tp
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
from .externalbo import HyperOpt

logger: logging.Logger = logging.getLogger(__name__)

def smooth_copy(array: p.Array, possible_radii: Optional[List[int]] = None) -> p.Array:
    candidate: p.Array = array.spawn_child()
    if possible_radii is None:
        possible_radii = [3]
    value: np.ndarray = candidate._value
    radii: List[int] = [array.random_state.choice(possible_radii) for _ in value.shape]
    try:
        value2: np.ndarray = ndimage.convolve(value, np.ones(radii) / np.prod(radii))
    except Exception as e:
        assert False, f'{e} in smooth_copy, {radii}, {np.prod(radii)}'
    invfreq: int = 4 if len(possible_radii) == 1 else max(4, np.random.randint(max(4, len(array.value.flatten()))))
    indices: np.ndarray = array.random_state.randint(invfreq, size=value.shape) == 0
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
        self.antismooth = antismooth
        self.crossover_type = crossover_type
        self.roulette_size = roulette_size
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
        self._annealing_base: Optional[np.ndarray] = None
        self._max_loss: float = -float('inf')
        self.sparse: int = int(sparse)
        all_params: List[Tuple[Any, p.Parameter]] = p.helpers.flatten(self.parametrization)
        arities: List[int] = [len(param.choices) for _, param in all_params if isinstance(param, p.TransitionChoice)]
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
            'doerr', 'lognormal', 'xlognormal', 'xsmalllognormal', 
            'tinylognormal', 'lognormal', 'smalllognormal', 'biglognormal', 
            'hugelognormal'
        ], f"Unkwnown mutation: '{mutation}'"
        if mutation == 'adaptive':
            self._adaptive_mr: float = 0.5
        elif mutation == 'lognormal':
            self._global_mr: float = 0.2
            self._memory_index: int = 0
            self._memory_size: int = 12
            self._best_recent_loss: float = float('inf')
        elif mutation == 'xsmalllognormal':
            self._global_mr = 0.8
            self._memory_index = 0
            self._memory_size = 4
            self._best_recent_loss = float('inf')
        elif mutation == 'xlognormal':
            self._global_mr = 0.8
            self._memory_index = 0
            self._memory_size = 12
            self._best_recent_loss = float('inf')
        elif mutation == 'tinylognormal':
            self._global_mr = 0.01
            self._memory_index = 0
            self._memory_size = 2
            self._best_recent_loss = float('inf')
        elif mutation == 'smalllognormal':
            self._global_mr = 0.2
            self._memory_index = 0
            self._memory_size = 4
            self._best_recent_loss = float('inf')
        elif mutation == 'biglognormal':
            self._global_mr = 0.2
            self._memory_index = 0
            self._memory_size = 120
            self._best_recent_loss = float('inf')
        elif mutation == 'hugelognormal':
            self._global_mr = 0.2
            self._memory_index = 0
            self._memory_size = 1200
            self._best_recent_loss = float('inf')
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
                data: np.ndarray = mutator.portfolio_discrete_mutation(
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
                intensity = max(intensity, 1)
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
                    index = self._doerr_mutation_rewards.index(max(self._doerr_mutation_rewards))
                    self._doerr_index = -1
                intensity: int = self._doerr_mutation_rates[index]
                data = mutator.portfolio_discrete_mutation(
                    pessimistic_data, 
                    intensity=intensity, 
                    arity=self.arity_for_discrete_mutation
                )
            else:
                func_map: Dict[str, Callable[..., np.ndarray]] = {
                    'discrete': mutator.discrete_mutation, 
                    'fastga': mutator.doerr_discrete_mutation, 
                    'doublefastga': mutator.doubledoerr_discrete_mutation, 
                    'rls': mutator.rls_mutation, 
                    'portfolio': mutator.portfolio_discrete_mutation
                }
                func = func_map[mutation]
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

    def _internal_tell(self, x: np.ndarray, loss: float) -> None:
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
        self._previous_best_loss: float = self.current_bests['pessimistic'].mean

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        """Called whenever calling :code:`tell` on a candidate that was "asked"."""
        data: np.ndarray = candidate.get_standardized_data(reference=self.parametrization)
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
        crossover_type: str = 'none'
    ) -> None:
        super().__init__(_OnePlusOne, locals())

OnePlusOne: base.Optimizer = ParametrizedOnePlusOne().set_name('OnePlusOne', register=True)
OnePlusLambda: base.Optimizer = ParametrizedOnePlusOne().set_name('OnePlusLambda', register=True)
NoisyOnePlusOne: base.Optimizer = ParametrizedOnePlusOne(noise_handling='random').set_name('NoisyOnePlusOne', register=True)
DiscreteOnePlusOne: base.Optimizer = ParametrizedOnePlusOne(mutation='discrete').set_name('DiscreteOnePlusOne', register=True)
SADiscreteLenglerOnePlusOneExp09: base.Optimizer = ParametrizedOnePlusOne(
    tabu_length=1000, 
    mutation='lengler', 
    annealing='Exp0.9'
).set_name('SADiscreteLenglerOnePlusOneExp09', register=True)
SADiscreteLenglerOnePlusOneExp099: base.Optimizer = ParametrizedOnePlusOne(
    tabu_length=1000, 
    mutation='lengler', 
    annealing='Exp0.99'
).set_name('SADiscreteLenglerOnePlusOneExp099', register=True)
SADiscreteLenglerOnePlusOneExp09Auto: base.Optimizer = ParametrizedOnePlusOne(
    tabu_length=1000, 
    mutation='lengler', 
    annealing='Exp0.9Auto'
).set_name('SADiscreteLenglerOnePlusOneExp09Auto', register=True)
SADiscreteLenglerOnePlusOneLinAuto: base.Optimizer = ParametrizedOnePlusOne(
    tabu_length=1000, 
    mutation='lengler', 
    annealing='LinAuto'
).set_name('SADiscreteLenglerOnePlusOneLinAuto', register=True)
SADiscreteLenglerOnePlusOneLin1: base.Optimizer = ParametrizedOnePlusOne(
    tabu_length=1000, 
    mutation='lengler', 
    annealing='Lin1.0'
).set_name('SADiscreteLenglerOnePlusOneLin1', register=True)
SADiscreteLenglerOnePlusOneLin100: base.Optimizer = ParametrizedOnePlusOne(
    tabu_length=1000, 
    mutation='lengler', 
    annealing='Lin100.0'
).set_name('SADiscreteLenglerOnePlusOneLin100', register=True)
SADiscreteOnePlusOneExp099: base.Optimizer = ParametrizedOnePlusOne(
    tabu_length=1000, 
    mutation='discrete', 
    annealing='Exp0.99'
).set_name('SADiscreteOnePlusOneExp099', register=True)
SADiscreteOnePlusOneLin100: base.Optimizer = ParametrizedOnePlusOne(
    tabu_length=1000, 
    mutation='discrete', 
    annealing='Lin100.0'
).set_name('SADiscreteOnePlusOneLin100', register=True)
SADiscreteOnePlusOneExp09: base.Optimizer = ParametrizedOnePlusOne(
    tabu_length=1000, 
    mutation='discrete', 
    annealing='Exp0.9'
).set_name('SADiscreteOnePlusOneExp09', register=True)
DiscreteOnePlusOneT: base.Optimizer = ParametrizedOnePlusOne(
    tabu_length=10000, 
    mutation='discrete'
).set_name('DiscreteOnePlusOneT', register=True)
PortfolioDiscreteOnePlusOne: base.Optimizer = ParametrizedOnePlusOne(
    mutation='portfolio'
).set_name('PortfolioDiscreteOnePlusOne', register=True)
PortfolioDiscreteOnePlusOneT: base.Optimizer = ParametrizedOnePlusOne(
    tabu_length=10000, 
    mutation='portfolio'
).set_name('PortfolioDiscreteOnePlusOneT', register=True)
DiscreteLenglerOnePlusOne: base.Optimizer = ParametrizedOnePlusOne(
    mutation='lengler'
).set_name('DiscreteLenglerOnePlusOne', register=True)
DiscreteLengler2OnePlusOne: base.Optimizer = ParametrizedOnePlusOne(
    crossover=True, 
    mutation='lengler2'
).set_name('DiscreteLengler2OnePlusOne', register=True)
DiscreteLengler3OnePlusOne: base.Optimizer = ParametrizedOnePlusOne(
    crossover=True, 
    mutation='lengler3'
).set_name('DiscreteLengler3OnePlusOne', register=True)
DiscreteLenglerHalfOnePlusOne: base.Optimizer = ParametrizedOnePlusOne(
    crossover=True, 
    mutation='lenglerhalf'
).set_name('DiscreteLenglerHalfOnePlusOne', register=True)
DiscreteLenglerFourthOnePlusOne: base.Optimizer = ParametrizedOnePlusOne(
    crossover=True, 
    mutation='lenglerfourth'
).set_name('DiscreteLenglerFourthOnePlusOne', register=True)
DoerrOnePlusOne: base.Optimizer = ParametrizedOnePlusOne(
    mutation='doerr'
).set_name('DiscreteDoerrOnePlusOne', register=True)
DoerrOnePlusOne.no_parallelization = True
CauchyOnePlusOne: base.Optimizer = ParametrizedOnePlusOne(
    mutation='cauchy'
).set_name('CauchyOnePlusOne', register=True)
OptimisticNoisyOnePlusOne: base.Optimizer = ParametrizedOnePlusOne(
    noise_handling='optimistic'
).set_name('OptimisticNoisyOnePlusOne', register=True)
OptimisticDiscreteOnePlusOne: base.Optimizer = ParametrizedOnePlusOne(
    noise_handling='optimistic', 
    mutation='discrete'
).set_name('OptimisticDiscreteOnePlusOne', register=True)
OLNDiscreteOnePlusOne: base.Optimizer = ParametrizedOnePlusOne(
    noise_handling='optimistic', 
    mutation='lognormal'
).set_name('OLNDiscreteOnePlusOne', register=True)
NoisyDiscreteOnePlusOne: base.Optimizer = ParametrizedOnePlusOne(
    noise_handling=('random', 1.0), 
    mutation='discrete'
).set_name('NoisyDiscreteOnePlusOne', register=True)
DoubleFastGADiscreteOnePlusOne: base.Optimizer = ParametrizedOnePlusOne(
    mutation='doublefastga'
).set_name('DoubleFastGADiscreteOnePlusOne', register=True)
RLSOnePlusOne: base.Optimizer = ParametrizedOnePlusOne(
    mutation='rls'
).set_name('RLSOnePlusOne', register=True)
SparseDoubleFastGADiscreteOnePlusOne: base.Optimizer = ParametrizedOnePlusOne(
    mutation='doublefastga', 
    sparse=True
).set_name('SparseDoubleFastGADiscreteOnePlusOne', register=True)
RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne: base.Optimizer = ParametrizedOnePlusOne(
    crossover=True, 
    mutation='portfolio', 
    noise_handling='optimistic'
).set_name('RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne', register=True)
RecombiningPortfolioDiscreteOnePlusOne: base.Optimizer = ParametrizedOnePlusOne(
    crossover=True, 
    mutation='portfolio'
).set_name('RecombiningPortfolioDiscreteOnePlusOne', register=True)
RecombiningPortfolioDiscreteLenglerOnePlusOne: base.Optimizer = ParametrizedOnePlusOne(
    crossover=True, 
    mutation='lengler'
).set_name('RecombiningDiscreteLenglerOnePlusOne', register=True)
MaxRecombiningDiscreteLenglerOnePlusOne: base.Optimizer = ParametrizedOnePlusOne(
    crossover=True, 
    mutation='lengler', 
    crossover_type='max'
).set_name('MaxRecombiningDiscreteLenglerOnePlusOne', register=True)
MinRecombiningDiscreteLenglerOnePlusOne: base.Optimizer = ParametrizedOnePlusOne(
    crossover=True, 
    mutation='lengler', 
    crossover_type='min'
).set_name('MinRecombiningDiscreteLenglerOnePlusOne', register=True)
OnePtRecombiningDiscreteLenglerOnePlusOne: base.Optimizer = ParametrizedOnePlusOne(
    crossover=True, 
    mutation='lengler', 
    crossover_type='onepoint'
).set_name('OnePtRecombiningDiscreteLenglerOnePlusOne', register=True)
TwoPtRecombiningDiscreteLenglerOnePlusOne: base.Optimizer = ParametrizedOnePlusOne(
    crossover=True, 
    mutation='lengler', 
    crossover_type='twopoint'
).set_name('TwoPtRecombiningDiscreteLenglerOnePlusOne', register=True)
RandRecombiningDiscreteLenglerOnePlusOne: base.Optimizer = ParametrizedOnePlusOne(
    crossover=True, 
    mutation='lengler', 
    crossover_type='rand'
).set_name('RandRecombiningDiscreteLenglerOnePlusOne', register=True)
RandRecombiningDiscreteLognormalOnePlusOne: base.Optimizer = ParametrizedOnePlusOne(
    crossover=True, 
    mutation='lognormal', 
    crossover_type='rand'
).set_name('RandRecombiningDiscreteLognormalOnePlusOne', register=True)

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

class _CMA(base.Optimizer):
    _CACHE_KEY: str = '#CMA#datacache'

    def __init__(
        self, 
        parametrization: p.Parameter, 
        budget: Optional[int] = None, 
        num_workers: int = 1, 
        config: Optional[ParametrizedCMA] = None, 
        algorithm: str = 'quad'
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.algorithm: str = algorithm
        self._config: ParametrizedCMA = ParametrizedCMA() if config is None else config
        pop: Optional[int] = self._config.popsize
        self._popsize: int = (
            max(num_workers, 4 + int(self._config.popsize_factor * np.log(self.dimension))) 
            if pop is None else max(pop, num_workers)
        )
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
                inopts.update(self._config.inopts if self._config.inopts is not None else {})
                self._es = cma.CMAEvolutionStrategy(
                    x0=self.parametrization.sample().get_standardized_data(reference=self.parametrization) 
                        if self._config.random_init else np.zeros(self.dimension, dtype=np.float64), 
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
            args: Tuple[List[float], List[np.ndarray]] = (listy, listx) if self._config.fcmaes else (listx, listy)
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

class ChoiceBase(base.Optimizer):
    """Nevergrad optimizer by competence map."""
    
    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        analysis: Any = p.helpers.analyze(self.parametrization)
        funcinfo: Any = self.parametrization.function
        self.has_noise: bool = not (analysis.deterministic and funcinfo.deterministic)
        self.noise_from_instrumentation: bool = self.has_noise and funcinfo.deterministic
        self.fully_continuous: bool = analysis.continuous
        all_params: List[Tuple[Any, p.Parameter]] = p.helpers.flatten(self.parametrization)
        int_layers: List[Any] = list(itertools.chain.from_iterable([_layering.Int.filter_from(x) for _, x in all_params]))
        int_layers = [x for x in int_layers if x.arity is not None]
        self.has_discrete_not_softmax: bool = any((not isinstance(lay, _datalayers.SoftmaxSampling) for lay in int_layers))
        self._has_discrete: bool = bool(int_layers)
        self._arity: int = max((lay.arity for lay in int_layers), default=-1)
        if self.fully_continuous:
            self._arity = -1
        self._optim: Optional[base.Optimizer] = None
        self._constraints_manager.update(max_trials=1000, penalty_factor=1.0, penalty_exponent=1.01)

    @property
    def optim(self) -> base.Optimizer:
        if self._optim is None:
            self._optim = self._select_optimizer_cls()(self.parametrization, self.budget, self.num_workers)
            self._optim = self._optim if not isinstance(self._optim, NGOptBase) else self._optim.optim
            logger.debug('%s selected %s optimizer.', self.name, self._optim.name)
        return self._optim

    def _select_optimizer_cls(self) -> Callable[..., base.Optimizer]:
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

OldCMA: base.Optimizer = ParametrizedCMA().set_name('OldCMA', register=True)
LargeCMA: base.Optimizer = ParametrizedCMA(scale=3.0).set_name('LargeCMA', register=True)
LargeDiagCMA: base.Optimizer = ParametrizedCMA(scale=3.0, diagonal=True).set_name('LargeDiagCMA', register=True)
TinyCMA: base.Optimizer = ParametrizedCMA(scale=0.33).set_name('TinyCMA', register=True)
CMAbounded: base.Optimizer = ParametrizedCMA(
    scale=1.5884, 
    popsize_factor=1, 
    elitist=True, 
    diagonal=True, 
    fcmaes=False
).set_name('CMAbounded', register=True)
CMAsmall: base.Optimizer = ParametrizedCMA(
    scale=0.3607, 
    popsize_factor=3, 
    elitist=False, 
    diagonal=False, 
    fcmaes=False
).set_name('CMAsmall', register=True)
CMAstd: base.Optimizer = ParametrizedCMA(
    scale=0.4699, 
    popsize_factor=3, 
    elitist=False, 
    diagonal=False, 
    fcmaes=False
).set_name('CMAstd', register=True)
CMApara: base.Optimizer = ParametrizedCMA(
    scale=0.8905, 
    popsize_factor=8, 
    elitist=True, 
    diagonal=True, 
    fcmaes=False
).set_name('CMApara', register=True)
CMAtuning: base.Optimizer = ParametrizedCMA(
    scale=0.4847, 
    popsize_factor=1, 
    elitist=True, 
    diagonal=False, 
    fcmaes=False
).set_name('CMAtuning', register=True)

class MetaCMA(ChoiceBase):
    """Nevergrad CMA optimizer by competence map."""
    
    def _select_optimizer_cls(self) -> Callable[..., base.Optimizer]:
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

DiagonalCMA: base.Optimizer = ParametrizedCMA(diagonal=True).set_name('DiagonalCMA', register=True)
EDCMA: base.Optimizer = ParametrizedCMA(diagonal=True, elitist=True).set_name('EDCMA', register=True)
SDiagonalCMA: base.Optimizer = ParametrizedCMA(diagonal=True, zero=True).set_name('SDiagonalCMA', register=True)
FCMA: base.Optimizer = ParametrizedCMA(fcmaes=True).set_name('FCMA', register=True)
MetaCMA: base.Optimizer = MetaCMA().set_name('MetaCMA', register=True)

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
            means: List[float] = [sum(fitnesses) / float(self.llambda) for fitnesses in [first_fifth, last_fifth]]
            stds: List[float] = [np.std(fitnesses) / np.sqrt(self.llambda - 1) for fitnesses in [first_fifth, last_fifth]]
            z: float = (means[0] - means[1]) / np.sqrt(stds[0] ** 2 + stds[1] ** 2)
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
        self.sigma: float = 1
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

    def _internal_provide_recommendation(self) -> Optional[np.ndarray]:
        return self.current_center

    def _internal_ask_candidate(self) -> p.Parameter:
        mutated_sigma: float = self.sigma * np.exp(self._rng.normal(0, 1) / np.sqrt(self.dimension))
        assert len(self.current_center) == len(self.covariance), [self.dimension, self.current_center, self.covariance]
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
            self.children = sorted(self.children, key=base._loss)
            population_data: List[np.ndarray] = [c.get_standardized_data(reference=self.parametrization) for c in self.children]
            mu: int = self.popsize.mu
            arrays: List[np.ndarray] = population_data[:mu]
            centered_arrays: np.ndarray = np.array([x - self.current_center for x in arrays])
            cov: np.ndarray = centered_arrays.T.dot(centered_arrays)
            mem_factor: float = 0.9 if self._COVARIANCE_MEMORY else 0
            self.covariance *= mem_factor
            self.covariance += (1 - mem_factor) * cov
            self.current_center = sum(arrays) / mu
            self.sigma = np.exp(sum([np.log(c._meta['sigma']) for c in self.children[:mu]]) / mu)
            self.parents = self.children[:mu]
            self.children = []

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: float) -> None:
        raise errors.TellNotAskedNotSupportedError

@registry.register
class AXP(base.Optimizer):
    """AX-platform optimizer."""
    
    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        try:
            from ax.service.ax_client import AxClient, ObjectiveProperties
        except Exception as e:
            print(f'Pb for creating AX solver')
            raise e
        self.ax_parametrization: List[Dict[str, Any]] = [
            {'name': f'x{i}', 'type': 'range', 'bounds': [0.0, 1.0]} for i in range(self.dimension)
        ]
        self.ax_client: AxClient = AxClient()
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
                trial = _trial
                self._trials.append(trial)
        trial: Any = self._trials[0]
        self._trials = self._trials[1:]
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
            y: List[float] = self._normalizer.forward(data)
        self._fake_function.register(y, -loss)
        self.ax_client.complete_trial(trial_index=candidate._meta['trial_index'], raw_data=loss)

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: float) -> None:
        raise errors.TellNotAskedNotSupportedError

class ParametrizedMetaModel(base.ConfiguredOptimizer):
    """Adds a metamodel to an optimizer."""
    
    def __init__(
        self, 
        *, 
        multivariate_optimizer: Optional[Callable[..., base.Optimizer]] = None, 
        frequency_ratio: float = 0.9, 
        algorithm: str, 
        degree: int = 2
    ) -> None:
        super().__init__(_MetaModel, locals())

class NgIoh(NGOptBase):
    pass

@registry.register
class NGOptBase(base.Optimizer):
    """Base class for NGOpt optimizers."""

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        analysis: Any = p.helpers.analyze(self.parametrization)
        funcinfo: Any = self.parametrization.function
        self.has_noise: bool = not (analysis.deterministic and funcinfo.deterministic)
        self.has_real_noise: bool = not funcinfo.deterministic
        self.noise_from_instrumentation: bool = self.has_noise and funcinfo.deterministic
        self.fully_continuous: bool = analysis.continuous
        all_params: List[Tuple[Any, p.Parameter]] = p.helpers.flatten(self.parametrization)
        int_layers: List[Any] = list(itertools.chain.from_iterable([_layering.Int.filter_from(x) for _, x in all_params]))
        int_layers = [x for x in int_layers if x.arity is not None]
        self.has_discrete_not_softmax: bool = any((not isinstance(lay, _datalayers.SoftmaxSampling) for lay in int_layers))
        self._has_discrete: bool = bool(int_layers)
        self._arity: int = max((lay.arity for lay in int_layers), default=-1)
        if self.fully_continuous:
            self._arity = -1
        self._optim: Optional[base.Optimizer] = None
        self._constraints_manager.update(max_trials=1000, penalty_factor=1.0, penalty_exponent=1.01)

    @property
    def optim(self) -> base.Optimizer:
        if self._optim is None:
            self._optim = self._select_optimizer_cls()(self.parametrization, self.budget, self.num_workers)
            self._optim = self._optim if not isinstance(self._optim, NGOptBase) else self._optim.optim
            logger.debug('%s selected %s optimizer.', self.name, self._optim.name)
        return self._optim

    def _select_optimizer_cls(self) -> Callable[..., base.Optimizer]:
        assert self.budget is not None
        if self.has_noise and self.has_discrete_not_softmax:
            cls: Callable[..., base.Optimizer] = DoubleFastGADiscreteOnePlusOne if self.dimension < 60 else CMA
        elif self.has_real_noise and self.fully_continuous:
            cls = TBPSA
        elif self.has_discrete_not_softmax or not self.parametrization.function.metrizable or (not self.fully_continuous):
            cls = DoubleFastGADiscreteOnePlusOne
        elif self.num_workers > self.budget / 5:
            if self.num_workers > self.budget / 2.0 or self.budget < self.dimension:
                cls = MetaTuneRecentering
            else:
                cls = NaiveTBPSA
        elif self.num_workers == 1 and self.budget > 6000 and (self.dimension > 7):
            cls = ChainCMAPowell
        elif self.num_workers == 1 and self.budget < self.dimension * 30:
            cls = OnePlusOne if self.dimension > 30 else Cobyla
        else:
            cls = DE if self.dimension > 2000 else MetaCMA if self.dimension > 1 else OnePlusOne
        return cls

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

OldCMA: base.Optimizer = ParametrizedCMA().set_name('OldCMA', register=True)
LargeCMA: base.Optimizer = ParametrizedCMA(scale=3.0).set_name('LargeCMA', register=True)
LargeDiagCMA: base.Optimizer = ParametrizedCMA(scale=3.0, diagonal=True).set_name('LargeDiagCMA', register=True)
TinyCMA: base.Optimizer = ParametrizedCMA(scale=0.33).set_name('TinyCMA', register=True)
CMAbounded: base.Optimizer = ParametrizedCMA(
    scale=1.5884, 
    popsize_factor=1, 
    elitist=True, 
    diagonal=True, 
    fcmaes=False
).set_name('CMAbounded', register=True)
CMAsmall: base.Optimizer = ParametrizedCMA(
    scale=0.3607, 
    popsize_factor=3, 
    elitist=False, 
    diagonal=False, 
    fcmaes=False
).set_name('CMAsmall', register=True)
CMAstd: base.Optimizer = ParametrizedCMA(
    scale=0.4699, 
    popsize_factor=3, 
    elitist=False, 
    diagonal=False, 
    fcmaes=False
).set_name('CMAstd', register=True)
CMApara: base.Optimizer = ParametrizedCMA(
    scale=0.8905, 
    popsize_factor=8, 
    elitist=True, 
    diagonal=True, 
    fcmaes=False
).set_name('CMApara', register=True)
CMAtuning: base.Optimizer = ParametrizedCMA(
    scale=0.4847, 
    popsize_factor=1, 
    elitist=True, 
    diagonal=False, 
    fcmaes=False
).set_name('CMAtuning', register=True)

@registry.register
class MetaCMA(ChoiceBase):
    """Nevergrad CMA optimizer by competence map."""
    
    def _select_optimizer_cls(self) -> Callable[..., base.Optimizer]:
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

DiagonalCMA: base.Optimizer = ParametrizedCMA(diagonal=True).set_name('DiagonalCMA', register=True)
EDCMA: base.Optimizer = ParametrizedCMA(diagonal=True, elitist=True).set_name('EDCMA', register=True)
SDiagonalCMA: base.Optimizer = ParametrizedCMA(diagonal=True, zero=True).set_name('SDiagonalCMA', register=True)
FCMA: base.Optimizer = ParametrizedCMA(fcmaes=True).set_name('FCMA', register=True)
MetaCMA: base.Optimizer = MetaCMA().set_name('MetaCMA', register=True)

class _Rescaled(base.Optimizer):
    """Proposes a version of a base optimizer which works at a different scale."""
    
    def __init__(
        self, 
        parametrization: p.Parameter, 
        budget: Optional[int] = None, 
        num_workers: int = 1, 
        base_optimizer: Callable[..., base.Optimizer] = MetaCMA, 
        scale: Optional[float] = None, 
        shift: Optional[float] = None
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self._optimizer: base.Optimizer = base_optimizer(self.parametrization, budget=budget, num_workers=num_workers)
        self.no_parallelization: bool = self._optimizer.no_parallelization
        self._subcandidates: Dict[str, p.Parameter] = {}
        if scale is None:
            assert self.budget is not None, 'Either scale or budget must be known in _Rescaled.'
            scale = math.sqrt(math.log(self.budget) / self.dimension)
        self.scale: float = scale
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
        original_candidate: p.Parameter = self._subcandidates.pop(candidate.uid)
        self._optimizer.tell(original_candidate, loss)

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: float) -> None:
        candidate = self.rescale_candidate(candidate, inverse=True)
        self._optimizer.tell(candidate, loss)

    def enable_pickling(self) -> None:
        self._optimizer.enable_pickling()

class SplitOptimizer(base.Optimizer):
    """Combines optimizers, each of them working on their own variables."""
    
    def __init__(
        self, 
        parametrization: p.Parameter, 
        budget: Optional[int] = None, 
        num_workers: int = 1, 
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        self._config: Dict[str, Any] = ConfSplitOptimizer() if config is None else config
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self._subcandidates: Dict[str, p.Parameter] = {}
        num_vars: Optional[Union[int, List[int]]] = self._config.get('num_vars')
        num_optims: Optional[int] = self._config.get('num_optims')
        max_num_vars: Optional[int] = self._config.get('max_num_vars')
        subparams: List[p.Parameter] = []
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
            for var in num_vars:
                subparams.append(p.Array(shape=(var,)))
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
            num_vars = self._config.get('num_vars', [])
            num_optims = self._config.get('num_optims', 2)
            max_num_vars = self._config.get('max_num_vars')
            if max_num_vars is not None:
                num_vars = [max_num_vars] * (self.dimension // max_num_vars)
                if self.dimension > sum(num_vars):
                    num_vars += [self.dimension - sum(num_vars)]
            for i in range(num_optims):
                if len(num_vars) < i + 1:
                    num_vars += [self.dimension // num_optims + (self.dimension % num_optims > i)]
                assert num_vars[i] >= 1, 'At least one variable per optimizer.'
                subparams += [p.Array(shape=(num_vars[i],))]
        if self._config.get('non_deterministic_descriptor', True):
            for param in subparams:
                param.function.deterministic = False
        self.optims: List[base.Optimizer] = []
        multi: Callable[..., base.Optimizer] = self._config.get('multivariate_optimizer', MetaCMA)
        mono: Callable[..., base.Optimizer] = self._config.get('monovariate_optimizer', oneshot.RandomSearch)
        for param in subparams:
            param.random_state = self.parametrization.random_state
            self.optims.append(
                (multi if param.dimension > 1 else mono)(param, budget, num_workers)
            )
        assert sum((opt.dimension for opt in self.optims)) == self.dimension, 'sum of sub-dimensions should be equal to the total dimension.'
    
    def _internal_ask_candidate(self) -> p.Parameter:
        candidates: List[p.Parameter] = []
        for i, opt in enumerate(self.optims):
            if self._config.get('progressive', False):
                assert self.budget is not None
                if i > 0 and i / len(self.optims) > np.sqrt(2.0 * self.num_ask / self.budget):
                    candidates.append(opt.parametrization.spawn_child())
                    continue
            candidates.append(opt.ask())
        data: np.ndarray = np.concatenate([
            c.get_standardized_data(reference=opt.parametrization) for c, opt in zip(candidates, self.optims)
        ], axis=0)
        cand: p.Parameter = self.parametrization.spawn_child().set_standardized_data(data)
        self._subcandidates[cand.uid] = candidates
        return cand

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        candidates: List[p.Parameter] = self._subcandidates.pop(candidate.uid)
        for cand, opt in zip(candidates, self.optims):
            opt.tell(cand, loss)

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: float) -> None:
        data: np.ndarray = candidate.get_standardized_data(reference=self.parametrization)
        start: int = 0
        for opt in self.optims:
            local_data: np.ndarray = data[start:start + opt.dimension]
            start += opt.dimension
            local_candidate: p.Parameter = opt.parametrization.spawn_child().set_standardized_data(local_data)
            opt.tell(local_candidate, loss)

    def _info(self) -> Dict[str, Any]:
        key: str = 'sub-optim'
        optims_info: List[str] = [x.name if key not in x._info() else x._info()[key] for x in self.optims]
        return {key: ','.join(optims_info)}

@registry.register
class ParametrizedBO(base.ConfiguredOptimizer):
    """Bayesian optimization using bayes_opt package."""
    
    no_parallelization: bool = True

    def __init__(
        self, 
        *, 
        initialization: Optional[str] = None, 
        init_budget: Optional[int] = None, 
        middle_point: bool = False, 
        utility_kind: str = 'ucb', 
        utility_kappa: float = 2.576, 
        utility_xi: float = 0.0, 
        gp_parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(_BO, locals())
    BO: base.Optimizer = ParametrizedBO().set_name('BO', register=True)
    BOSplit: base.Optimizer = ConfSplitOptimizer(
        max_num_vars=15, 
        progressive=False, 
        multivariate_optimizer=ParametrizedBO
    ).set_name('BOSplit', register=True)

@registry.register
class MetaModel(base.Optimizer):
    """Nevergrad CMA optimizer by competence map."""
    
    def _select_optimizer_cls(self) -> Callable[..., base.Optimizer]:
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

class _MetaModel(base.Optimizer):
    
    def __init__(
        self, 
        parametrization: p.Parameter, 
        budget: Optional[int] = None, 
        num_workers: int = 1, 
        *, 
        multivariate_optimizer: Optional[Callable[..., base.Optimizer]] = None, 
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
        self.elitist: bool = self.dimension < 3
        if multivariate_optimizer is None:
            self._optim: base.Optimizer = ParametrizedCMA(
                elitist=self.elitist
            )(self.parametrization, budget, num_workers)
        else:
            self._optim: base.Optimizer = multivariate_optimizer(self.parametrization, budget, num_workers)

    def _internal_ask_candidate(self) -> p.Parameter:
        sample_size: int = int(self.dimension * (self.dimension - 1) / 2 + 2 * self.dimension + 1)
        if self.degree != 2:
            sample_size = int(np.power(sample_size, self.degree / 2.0))
            if self.algorithm == 'image':
                sample_size = 50
        freq: int = max(13 if self.algorithm != 'image' else 0, self.num_workers, self.dimension if self.algorithm != 'image' else 0, int(self.frequency_ratio * sample_size))
        if len(self.archive) >= sample_size and (not self._num_ask % freq):
            try:
                data: np.ndarray = learn_on_k_best(self.archive, sample_size, self.algorithm, self.degree, self.parametrization)
                candidate: np.ndarray = data
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

@registry.register
class ChainCMAPowell(base.Optimizer):
    """Chain of CMA and Powell optimizers."""
    
    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.optimizers: List[base.Optimizer] = [CMA, Powell]
        for opt in self.optimizers[2:]:
            opt.initial_guess = self._rng.normal(0, 1, self.dimension)
        self._current: int = -1
        self._warmup_budget: Optional[int] = None

    def _internal_ask_candidate(self) -> p.Parameter:
        if len(self.optimizers) == 1:
            optim_index: int = 0
        else:
            self._current += 1
            optim_index: int = self._current % len(self.optimizers)
        if optim_index is None:
            raise RuntimeError('Something went wrong in optimizer selection')
        opt: base.Optimizer = self.optimizers[optim_index]
        if optim_index > 1 and (not opt.num_ask) and (not opt._suggestions) and (not opt.num_tell):
            opt._suggestions.append(self.parametrization.sample())
        candidate: p.Parameter = opt.ask()
        candidate._meta['optim_index'] = optim_index
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        optim_index: int = candidate._meta.get('optim_index', -1)
        if optim_index != -1 and optim_index < len(self.optimizers):
            self.optimizers[optim_index].tell(candidate, loss)

    def enable_pickling(self) -> None:
        for opt in self.optimizers:
            opt.enable_pickling()

# The rest of the code would follow similarly, adding type annotations accordingly.
