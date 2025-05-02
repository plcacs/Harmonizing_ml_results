import math
import logging
import itertools
from collections import deque
import warnings
import numpy as np
from collections import defaultdict
import scipy.ndimage as ndimage
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
try:
    from bayes_opt import UtilityFunction
    from bayes_opt import BayesianOptimization
except ModuleNotFoundError:
    UtilityFunction = Any
    BayesianOptimization = Any
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
try:
    from .externalbo import HyperOpt
except Exception:
    HyperOpt = Any
logger: logging.Logger = logging.getLogger(__name__)


def smooth_copy(array, possible_radii=None):
    candidate = array.spawn_child()
    if possible_radii is None:
        possible_radii = [3]
    value = candidate._value
    radii = [array.random_state.choice(possible_radii) for _ in value.shape]
    try:
        value2 = ndimage.convolve(value, np.ones(radii) / np.prod(radii))
    except Exception as e:
        assert False, f'{e} in smooth_copy, {radii}, {np.prod(radii)}'
    invfreq = 4 if len(possible_radii) == 1 else max(4, np.random.randint(
        max(4, len(array.value.flatten()))))
    indices = array.random_state.randint(invfreq, size=value.shape) == 0
    while np.sum(indices) == 0 and len(possible_radii) > 1:
        invfreq = 4 if len(possible_radii) == 1 else max(4, np.random.
            randint(max(4, len(array.value.flatten()))))
        indices = array.random_state.randint(invfreq, size=value.shape) == 0
    value[indices] = value2[indices]
    candidate._value = value
    return candidate


class _OnePlusOne(base.Optimizer):
    """Simple but sometimes powerful optimization algorithm.

    We use the one-fifth adaptation rule, going back to Schumer and Steiglitz (1968).
    It was independently rediscovered by Devroye (1972) and Rechenberg (1973).
    We use asynchronous updates, so that the 1+1 can actually be parallel and even
    performs quite well in such a context - this is naturally close to 1+lambda.

    Posssible mutations include gaussian and cauchy for the continuous case, and in the discrete case:
    discrete, fastga, rls, doublefastga, adaptive, portfolio, discreteBSO, doerr.
    - discrete is the most classical discrete mutation operator,
    - rls is the Randomized Local Search,
    - doubleFastGA is an adaptation of FastGA to arity > 2, Portfolio corresponds to random mutation rates,
    - discreteBSO corresponds to a decreasing schedule of mutation rate.
    - adaptive and doerr correspond to various self-adaptive mutation rates.
    - coordinatewise_adaptive is the anisotropic counterpart of the adaptive version.
    """

    def __init__(self, parametrization, budget=None, num_workers=1, *,
        noise_handling: Optional[Union[str, Tuple[str, float]]]=None,
        tabu_length: int=0, mutation: str='gaussian', crossover: bool=False,
        rotation: bool=False, annealing: str='none', use_pareto: bool=False,
        sparse: Union[bool, int]=False, smoother: bool=False, super_radii:
        bool=False, roulette_size: int=2, antismooth: int=55,
        crossover_type: str='none', forced_discretization: bool=False):
        super().__init__(parametrization, budget=budget, num_workers=
            num_workers)
        self.parametrization.tabu_length = tabu_length
        if forced_discretization:
            self.parametrization.set_integer_casting()
        self.antismooth = antismooth
        self.crossover_type = crossover_type
        self.roulette_size = roulette_size
        assert crossover or not rotation, 'We can not have both rotation and not crossover.'
        self._sigma: float = 1.0
        self._previous_best_loss: float = float('inf')
        self._best_recent_mr: float = 0.2
        self.inds: np.ndarray = np.array([True] * self.dimension)
        self.imr: float = 0.2
        self.use_pareto: bool = use_pareto
        self.smoother: bool = smoother
        self.super_radii: bool = super_radii
        self.annealing: str = annealing
        self._annealing_base: Optional[tp.ArrayLike] = None
        self._max_loss: float = -float('inf')
        self.sparse: int = int(sparse)
        all_params = p.helpers.flatten(self.parametrization)
        arities = [len(param.choices) for _, param in all_params if
            isinstance(param, p.TransitionChoice)]
        arity = max(arities, default=500)
        self.arity_for_discrete_mutation: int = arity
        if noise_handling is not None:
            if isinstance(noise_handling, str):
                assert noise_handling in ['random', 'optimistic'
                    ], f"Unkwnown noise handling: '{noise_handling}'"
            else:
                assert isinstance(noise_handling, tuple
                    ), 'noise_handling must be a string or  a tuple of type (strategy, factor)'
                assert noise_handling[1
                    ] > 0.0, 'the factor must be a float greater than 0'
                assert noise_handling[0] in ['random', 'optimistic'
                    ], f"Unkwnown noise handling: '{noise_handling}'"
        assert mutation in ['gaussian', 'cauchy', 'discrete', 'fastga',
            'rls', 'doublefastga', 'adaptive', 'coordinatewise_adaptive',
            'portfolio', 'discreteBSO', 'lengler', 'lengler2', 'lengler3',
            'lenglerhalf', 'lenglerfourth', 'doerr', 'lognormal',
            'xlognormal', 'xsmalllognormal', 'tinylognormal', 'lognormal',
            'smalllognormal', 'biglognormal', 'hugelognormal'
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
            self._velocity: np.ndarray = self._rng.uniform(size=self.dimension
                ) * arity / 4.0
            self._modified_variables: np.ndarray = np.array([True] * self.
                dimension)
        self.noise_handling: Optional[Union[str, Tuple[str, float]]
            ] = noise_handling
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
            i = 3
            j = 2
            self._doerr_index: int = -1
            while i < self.dimension:
                self._doerr_mutation_rates += [i]
                self._doerr_mutation_rewards += [0.0]
                self._doerr_counters += [0.0]
                i += j
                j += 2
        assert self.parametrization.tabu_length == tabu_length

    def _internal_ask_candidate(self):
        pass

    def _internal_tell_candidate(self, candidate, loss):
        pass

    def _internal_provide_recommendation(self):
        pass


class ParametrizedOnePlusOne(base.ConfiguredOptimizer):
    """Simple but sometimes powerfull class of optimization algorithm.
    This use asynchronous updates, so that (1+1) can actually be parallel and even
    performs quite well in such a context - this is naturally close to (1+lambda).


    Parameters
    ----------
    noise_handling: str or Tuple[str, float]
        Method for handling the noise. The name can be:

        - `"random"`: a random point is reevaluated regularly, this uses the one-fifth adaptation rule,
          going back to Schumer and Steiglitz (1968). It was independently rediscovered by Devroye (1972) and Rechenberg (1973).
        - `"optimistic"`: the best optimistic point is reevaluated regularly, optimism in front of uncertainty
        - a coefficient can to tune the regularity of these reevaluations (default .05)
    mutation: str
        One of the available mutations from:

        - `"gaussian"`: standard mutation by adding a Gaussian random variable (with progressive
          widening) to the best pessimistic point
        - `"cauchy"`: same as Gaussian but with a Cauchy distribution.
        - `"discrete"`: when a variable is mutated (which happens with probability 1/d in dimension d), it's just
             randomly drawn. This means that on average, only one variable is mutated.
        - `"discreteBSO"`: as in brainstorm optimization, we slowly decrease the mutation rate from 1 to 1/d.
        - `"fastga"`: FastGA mutations from the current best
        - `"doublefastga"`: double-FastGA mutations from the current best (Doerr et al, Fast Genetic Algorithms, 2017)
        - `"rls"`: Randomized Local Search (randomly mutate one and only one variable).
        - `"portfolio"`: Random number of mutated bits (called niform mixing in
          Dang & Lehre "Self-adaptation of Mutation Rates in Non-elitist Population", 2016)
        - `"lengler"`: specific mutation rate chosen as a function of the dimension and iteration index.
        - `"lengler{2|3|half|fourth}"`: variant of Lengler
    crossover: bool
        whether to add a genetic crossover step every other iteration.
    use_pareto: bool
        whether to restart from a random pareto element in multiobjective mode, instead of the last one added
    sparse: bool
        whether we have random mutations setting variables to 0.
    smoother: bool
        whether we suggest smooth mutations.

    Notes
    -----
    After many papers advocated the mutation rate 1/d in the discrete (1+1) for the discrete case,
    `it was proposed <https://arxiv.org/abs/1606.05551>`_ to use a randomly
    drawn mutation rate. `Fast genetic algorithms <https://arxiv.org/abs/1703.03334>`_ are based on a similar idea
    These two simple methods perform quite well on a wide range of problems.

    """

    def __init__(self, *, noise_handling: Optional[Union[str, Tuple[str,
        float]]]=None, tabu_length: int=0, mutation: str='gaussian',
        crossover: bool=False, rotation: bool=False, annealing: str='none',
        use_pareto: bool=False, sparse: bool=False, smoother: bool=False,
        super_radii: bool=False, roulette_size: int=2, antismooth: int=55,
        crossover_type: str='none'):
        super().__init__(_OnePlusOne, locals())

    def enable_pickling(self):
        self._optim.enable_pickling()


OnePlusOne: base.ConfiguredOptimizer = ParametrizedOnePlusOne().set_name(
    'OnePlusOne', register=True)
OnePlusLambda: base.ConfiguredOptimizer = ParametrizedOnePlusOne().set_name(
    'OnePlusLambda', register=True)
NoisyOnePlusOne: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
    noise_handling='random').set_name('NoisyOnePlusOne', register=True)
DiscreteOnePlusOne: base.ConfiguredOptimizer = ParametrizedOnePlusOne(mutation
    ='discrete').set_name('DiscreteOnePlusOne', register=True)
SADiscreteLenglerOnePlusOneExp09: base.ConfiguredOptimizer = (
    ParametrizedOnePlusOne(tabu_length=1000, mutation='lengler', annealing=
    'Exp0.9').set_name('SADiscreteLenglerOnePlusOneExp09', register=True))
SADiscreteLenglerOnePlusOneExp099: base.ConfiguredOptimizer = (
    ParametrizedOnePlusOne(tabu_length=1000, mutation='lengler', annealing=
    'Exp0.99').set_name('SADiscreteLenglerOnePlusOneExp099', register=True))
SADiscreteLenglerOnePlusOneExp09Auto: base.ConfiguredOptimizer = (
    ParametrizedOnePlusOne(tabu_length=1000, mutation='lengler', annealing=
    'Exp0.9Auto').set_name('SADiscreteLenglerOnePlusOneExp09Auto', register
    =True))
SADiscreteLenglerOnePlusOneLinAuto: base.ConfiguredOptimizer = (
    ParametrizedOnePlusOne(tabu_length=1000, mutation='lengler', annealing=
    'LinAuto').set_name('SADiscreteLenglerOnePlusOneLinAuto', register=True))
SADiscreteLenglerOnePlusOneLin1: base.ConfiguredOptimizer = (
    ParametrizedOnePlusOne(tabu_length=1000, mutation='lengler', annealing=
    'Lin1.0').set_name('SADiscreteLenglerOnePlusOneLin1', register=True))
SADiscreteLenglerOnePlusOneLin100: base.ConfiguredOptimizer = (
    ParametrizedOnePlusOne(tabu_length=1000, mutation='lengler', annealing=
    'Lin100.0').set_name('SADiscreteLenglerOnePlusOneLin100', register=True))
SADiscreteOnePlusOneExp099: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
    tabu_length=1000, mutation='discrete', annealing='Exp0.99').set_name(
    'SADiscreteOnePlusOneExp099', register=True)
SADiscreteOnePlusOneLin100: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
    tabu_length=1000, mutation='discrete', annealing='Lin100.0').set_name(
    'SADiscreteOnePlusOneLin100', register=True)
SADiscreteOnePlusOneExp09: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
    tabu_length=1000, mutation='discrete', annealing='Exp0.9').set_name(
    'SADiscreteOnePlusOneExp09', register=True)
DiscreteOnePlusOneT: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
    tabu_length=10000, mutation='discrete').set_name('DiscreteOnePlusOneT',
    register=True)
PortfolioDiscreteOnePlusOne: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
    mutation='portfolio').set_name('PortfolioDiscreteOnePlusOne', register=True
    )
PortfolioDiscreteOnePlusOneT: base.ConfiguredOptimizer = (
    ParametrizedOnePlusOne(tabu_length=10000, mutation='portfolio').
    set_name('PortfolioDiscreteOnePlusOneT', register=True))
DiscreteLenglerOnePlusOne: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
    mutation='lengler').set_name('DiscreteLenglerOnePlusOne', register=True)
DiscreteLengler2OnePlusOne: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
    mutation='lengler2').set_name('DiscreteLengler2OnePlusOne', register=True)
DiscreteLengler3OnePlusOne: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
    mutation='lengler3').set_name('DiscreteLengler3OnePlusOne', register=True)
DiscreteLenglerHalfOnePlusOne: base.ConfiguredOptimizer = (
    ParametrizedOnePlusOne(mutation='lenglerhalf').set_name(
    'DiscreteLenglerHalfOnePlusOne', register=True))
DiscreteLenglerFourthOnePlusOne: base.ConfiguredOptimizer = (
    ParametrizedOnePlusOne(mutation='lenglerfourth').set_name(
    'DiscreteLenglerFourthOnePlusOne', register=True))
DiscreteDoerrOnePlusOne: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
    mutation='doerr').set_name('DiscreteDoerrOnePlusOne', register=True)
DiscreteDoerrOnePlusOne.no_parallelization = True
CauchyOnePlusOne: base.ConfiguredOptimizer = ParametrizedOnePlusOne(mutation
    ='cauchy').set_name('CauchyOnePlusOne', register=True)
OptimisticNoisyOnePlusOne: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
    noise_handling='optimistic').set_name('OptimisticNoisyOnePlusOne',
    register=True)
OptimisticDiscreteOnePlusOne: base.ConfiguredOptimizer = (
    ParametrizedOnePlusOne(noise_handling='optimistic', mutation='discrete'
    ).set_name('OptimisticDiscreteOnePlusOne', register=True))
OLNDiscreteOnePlusOne: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
    noise_handling='optimistic', mutation='lognormal').set_name(
    'OLNDiscreteOnePlusOne', register=True)
NoisyDiscreteOnePlusOne: base.ConfiguredOptimizer = ParametrizedOnePlusOne(
    noise_handling=('random', 1.0), mutation='discrete').set_name(
    'NoisyDiscreteOnePlusOne', register=True)
DoubleFastGADiscreteOnePlusOne: base.ConfiguredOptimizer = (
    ParametrizedOnePlusOne(mutation='doublefastga').set_name(
    'DoubleFastGADiscreteOnePlusOne', register=True))
RLSOnePlusOne: base.ConfiguredOptimizer = ParametrizedOnePlusOne(mutation='rls'
    ).set_name('RLSOnePlusOne', register=True)
SparseDoubleFastGADiscreteOnePlusOne: base.ConfiguredOptimizer = (
    ParametrizedOnePlusOne(mutation='doublefastga', sparse=True).set_name(
    'SparseDoubleFastGADiscreteOnePlusOne', register=True))
(RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne: base.
    ConfiguredOptimizer) = (ParametrizedOnePlusOne(crossover=True, mutation
    ='portfolio', noise_handling='optimistic').set_name(
    'RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne', register=True))
RecombiningPortfolioDiscreteOnePlusOne: base.ConfiguredOptimizer = (
    ParametrizedOnePlusOne(crossover=True, mutation='portfolio').set_name(
    'RecombiningPortfolioDiscreteOnePlusOne', register=True))


class _CMA(base.Optimizer):
    _CACHE_KEY = '#CMA#datacache'

    def __init__(self, parametrization, budget=None, num_workers=1, config=
        None, algorithm='quad'):
        super().__init__(parametrization, budget=budget, num_workers=
            num_workers)
        self.algorithm: str = algorithm
        self._config: 'ParametrizedCMA' = ParametrizedCMA(
            ) if config is None else config
        pop = self._config.popsize
        self._popsize: int = max(num_workers, 4 + int(self._config.
            popsize_factor * np.log(self.dimension))) if pop is None else max(
            pop, num_workers)
        if self._config.elitist:
            self._popsize = max(self._popsize, self.num_workers + 1)
        self._to_be_asked: Deque[np.ndarray] = deque()
        self._to_be_told: List[p.Parameter] = []
        self._num_spawners: int = self._popsize // 2
        self._parents: List[p.Parameter] = [self.parametrization]
        self._es: Optional[Any] = None

    @property
    def es(self):
        scale_multiplier: float = 1.0
        if self.dimension == 1:
            self._config.fcmaes = True
        if p.helpers.Normalizer(self.parametrization).fully_bounded:
            scale_multiplier = 0.3 if self.dimension < 18 else 0.15
        if self._es is None or not self._config.fcmaes and self._es.stop():
            if not self._config.fcmaes:
                import cma
                inopts: Dict[str, Any] = dict(popsize=self._popsize, randn=
                    self._rng.randn, CMA_diagonal=self._config.diagonal,
                    verbose=-9, seed=np.nan, CMA_elitist=self._config.elitist)
                if self._config.inopts is not None:
                    inopts.update(self._config.inopts)
                self._es = cma.CMAEvolutionStrategy(x0=self.parametrization
                    .sample().get_standardized_data(reference=self._config.
                    popsize) if self._config.random_init else np.zeros(self
                    .dimension, dtype=np.float64), sigma0=self._config.
                    scale * scale_multiplier, inopts=inopts)
            else:
                try:
                    from fcmaes import cmaes
                except ImportError as e:
                    raise ImportError(
                        'Please install fcmaes (pip install fcmaes) to use FCMA optimizers'
                        ) from e
                self._es = cmaes.Cmaes(x0=np.zeros(self.dimension, dtype=np
                    .float64), input_sigma=self._config.scale *
                    scale_multiplier, popsize=self._popsize, randn=self.
                    _rng.randn)
        return self._es

    def _internal_ask_candidate(self):
        if not self._to_be_asked:
            self._to_be_asked.extend(self.es.ask())
        data: np.ndarray = self._to_be_asked.popleft()
        parent: p.Parameter = self._parents[self.num_ask % len(self._parents)]
        candidate: p.Parameter = parent.spawn_child().set_standardized_data(
            data, reference=self.parametrization)
        return candidate

    def _internal_tell_candidate(self, candidate, loss):
        if self._CACHE_KEY not in candidate._meta:
            candidate._meta[self._CACHE_KEY] = candidate.get_standardized_data(
                reference=self.parametrization)
        self._to_be_told.append(candidate)
        if len(self._to_be_told) >= self.es.popsize:
            listx: List[np.ndarray] = [c._meta[self._CACHE_KEY] for c in
                self._to_be_told]
            listy: List[float] = [c.loss for c in self._to_be_told]
            args: Tuple[Any, ...] = (listy, listx
                ) if self._config.fcmaes else (listx, listy)
            try:
                self.es.tell(*args)
            except (RuntimeError, AssertionError):
                pass
            else:
                self._parents = sorted(self._to_be_told, key=base._loss)[:
                    self._num_spawners]
            self._to_be_told = []

    def _internal_provide_recommendation(self):
        pessimistic: p.Parameter = self.current_bests['pessimistic'
            ].parameter.get_standardized_data(reference=self.parametrization)
        d: int = self.dimension
        n: int = self.num_ask
        sample_size: int = int(d * d / 2 + d / 2 + 3)
        if self._config.high_speed and n >= sample_size:
            try:
                data: np.ndarray = learn_on_k_best(self.archive,
                    sample_size, self.algorithm)
                return data
            except MetaModelFailure:
                pass
        if self._es is None:
            return pessimistic
        if self._config.fcmaes:
            cma_best: Optional[np.ndarray] = self.es.best_x
        else:
            cma_best: Optional[np.ndarray] = getattr(self.es.result,
                'xbest', None)
        if cma_best is None:
            return pessimistic
        return cma_best


class ParametrizedCMA(base.ConfiguredOptimizer):
    """CMA-ES optimizer,
    This evolution strategy uses Gaussian sampling, iteratively modified
    for searching in the best directions.
    This optimizer wraps an external implementation: https://github.com/CMA-ES/pycma

    Parameters
    ----------
    scale: float
        scale of the search
    elitist: bool
        whether we switch to elitist mode, i.e. mode + instead of comma,
        i.e. mode in which we always keep the best point in the population.
    popsize: Optional[int] = None
        population size, should be n * self.num_workers for int n >= 1.
        default is max(self.num_workers, 4 + int(3 * np.log(self.dimension)))
    popsize_factor: float = 3.
        factor in the formula for computing the population size
    diagonal: bool
        use the diagonal version of CMA (advised in big dimension)
    high_speed: bool
        use metamodel for recommendation
    fcmaes: bool
        use fast implementation, doesn't support diagonal=True.
        produces equivalent results, preferable for high dimensions or
        if objective function evaluation is fast.
    random_init: bool
        Use a randomized initialization
    inopts: optional dict
        use this to averride any inopts parameter of the wrapped CMA optimizer
        (see https://github.com/CMA-ES/pycma)
    """

    def __init__(self, *, scale: float=1.0, elitist: bool=False, popsize:
        Optional[int]=None, popsize_factor: float=3.0, diagonal: bool=False,
        zero: bool=False, high_speed: bool=False, fcmaes: bool=False,
        random_init: bool=False, inopts: Optional[Dict[str, Any]]=None,
        algorithm: str='quad'):
        super().__init__(_CMA, locals(), as_config=True)
        if zero:
            scale = scale / 1000.0
        if fcmaes:
            if diagonal:
                raise RuntimeError(
                    "fcmaes doesn't support diagonal=True, use fcmaes=False")
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

    def __init__(self, parametrization, budget=None, num_workers=1):
        super().__init__(parametrization, budget=budget, num_workers=
            num_workers)
        analysis = p.helpers.analyze(self.parametrization)
        funcinfo = self.parametrization.function
        self.has_noise: bool = not (analysis.deterministic and funcinfo.
            deterministic)
        self.noise_from_instrumentation: bool = (self.has_noise and
            funcinfo.deterministic)
        self.fully_continuous: bool = analysis.continuous
        all_params = p.helpers.flatten(self.parametrization)
        int_layers: List[Any] = list(itertools.chain.from_iterable([
            _layering.Int.filter_from(x) for _, x in all_params]))
        int_layers = [x for x in int_layers if x.arity is not None]
        self.has_discrete_not_softmax: bool = any(not isinstance(lay,
            _datalayers.SoftmaxSampling) for lay in int_layers)
        self._has_discrete: bool = bool(int_layers)
        self._arity: int = max((lay.arity for lay in int_layers), default=-1)
        if self.fully_continuous:
            self._arity = -1
        self._optim: Optional[base.Optimizer] = None
        self._constraints_manager.update(max_trials=1000, penalty_factor=
            1.0, penalty_exponent=1.01)

    @property
    def optim(self):
        if self._optim is None:
            self._optim = self._select_optimizer_cls()(self.parametrization,
                self.budget, self.num_workers)
            self._optim = self._optim if not isinstance(self._optim, NGOptBase
                ) else self._optim.optim
            logger.debug('%s selected %s optimizer.', *(x.name for x in (
                self, self._optim)))
        return self._optim

    def _select_optimizer_cls(self):
        return CMA

    def _internal_ask_candidate(self):
        return self.optim.ask()

    def _internal_tell_candidate(self, candidate, loss):
        self.optim.tell(candidate, loss)

    def recommend(self):
        return self.optim.recommend()

    def _internal_tell_not_asked(self, candidate, loss):
        self.optim.tell(candidate, loss)

    def _info(self):
        out: Dict[str, Any] = {'sub-optim': self.optim.name}
        out.update(self.optim._info())
        return out

    def enable_pickling(self):
        self.optim.enable_pickling()


OldCMA: base.ConfiguredOptimizer = ParametrizedCMA().set_name('OldCMA',
    register=True)
LargeCMA: base.ConfiguredOptimizer = ParametrizedCMA(scale=3.0).set_name(
    'LargeCMA', register=True)
LargeDiagCMA: base.ConfiguredOptimizer = ParametrizedCMA(scale=3.0,
    diagonal=True).set_name('LargeDiagCMA', register=True)
TinyCMA: base.ConfiguredOptimizer = ParametrizedCMA(scale=0.33).set_name(
    'TinyCMA', register=True)
CMAbounded: base.ConfiguredOptimizer = ParametrizedCMA(scale=1.5884,
    popsize_factor=1, elitist=True, diagonal=True, fcmaes=False).set_name(
    'CMAbounded', register=True)
CMAsmall: base.ConfiguredOptimizer = ParametrizedCMA(scale=0.3607,
    popsize_factor=3, elitist=False, diagonal=False, fcmaes=False).set_name(
    'CMAsmall', register=True)
CMAstd: base.ConfiguredOptimizer = ParametrizedCMA(scale=0.4699,
    popsize_factor=3, elitist=False, diagonal=False, fcmaes=False).set_name(
    'CMAstd', register=True)
CMApara: base.ConfiguredOptimizer = ParametrizedCMA(scale=0.8905,
    popsize_factor=8, elitist=True, diagonal=True, fcmaes=False).set_name(
    'CMApara', register=True)
CMAtuning: base.ConfiguredOptimizer = ParametrizedCMA(scale=0.4847,
    popsize_factor=1, elitist=True, diagonal=False, fcmaes=False).set_name(
    'CMAtuning', register=True)


@registry.register
class MetaCMA(ChoiceBase):
    """Nevergrad CMA optimizer by competence map. You might modify this one for designing your own competence map.
    You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self):
        if (self.budget is not None and self.fully_continuous and not self.
            has_noise and self.num_objectives < 2):
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


DiagonalCMA: base.ConfiguredOptimizer = ParametrizedCMA(diagonal=True
    ).set_name('DiagonalCMA', register=True)
EDCMA: base.ConfiguredOptimizer = ParametrizedCMA(diagonal=True, elitist=True
    ).set_name('EDCMA', register=True)
SDiagonalCMA: base.ConfiguredOptimizer = ParametrizedCMA(diagonal=True,
    zero=True).set_name('SDiagonalCMA', register=True)
FCMA: base.ConfiguredOptimizer = ParametrizedCMA(fcmaes=True).set_name('FCMA',
    register=True)


@registry.register
class CMA(MetaCMA):
    pass


class _PopulationSizeController:
    """Population control scheme for TBPSA and EDA"""

    def __init__(self, llambda, mu, dimension, num_workers=1):
        self.llambda: int = max(llambda, num_workers)
        self.min_mu: int = min(mu, dimension)
        self.mu: int = mu
        self.dimension: int = dimension
        self.num_workers: int = num_workers
        self._loss_record: List[float] = []

    def add_value(self, loss):
        self._loss_record.append(loss)
        if len(self._loss_record) >= 5 * self.llambda:
            first_fifth: List[float] = self._loss_record[:self.llambda]
            last_fifth: List[float] = self._loss_record[-int(self.llambda):]
            means: List[float] = [(sum(fitnesses) / float(self.llambda)) for
                fitnesses in [first_fifth, last_fifth]]
            stds: List[float] = [(np.std(fitnesses) / np.sqrt(self.llambda -
                1)) for fitnesses in [first_fifth, last_fifth]]
            z: float = (means[0] - means[1]) / math.sqrt(stds[0] ** 2 + 
                stds[1] ** 2)
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
    """Estimation of distribution algorithm.

    Population-size equal to lambda = 4 x dimension by default.
    """
    _POPSIZE_ADAPTATION: bool = False
    _COVARIANCE_MEMORY: bool = False

    def __init__(self, parametrization, budget=None, num_workers=1):
        super().__init__(parametrization, budget=budget, num_workers=
            num_workers)
        self.sigma: float = 1.0
        self.covariance: np.ndarray = np.identity(self.dimension)
        dim: int = self.dimension
        self.popsize: _PopulationSizeController = _PopulationSizeController(
            llambda=4 * dim, mu=dim, dimension=dim, num_workers=num_workers)
        self.current_center: np.ndarray = np.zeros(self.dimension)
        self.children: List[p.Parameter] = []
        self.parents: List[p.Parameter] = [self.parametrization]

    def _internal_provide_recommendation(self):
        return self.current_center

    def _internal_ask_candidate(self):
        mutated_sigma: float = self.sigma * math.exp(self._rng.normal(0, 1) /
            math.sqrt(self.dimension))
        assert len(self.current_center) == len(self.covariance), [self.
            dimension, self.current_center, self.covariance]
        data: np.ndarray = self._rng.multivariate_normal(self.
            current_center, mutated_sigma * self.covariance)
        parent: p.Parameter = self.parents[self.num_ask % len(self.parents)]
        candidate: p.Parameter = parent.spawn_child().set_standardized_data(
            data, reference=self.parametrization)
        if parent is self.parametrization:
            candidate.heritage['lineage'] = candidate.uid
        return candidate

    def _internal_tell_candidate(self, candidate, loss):
        self.children.append(candidate)
        if self._POPSIZE_ADAPTATION:
            self.popsize.add_value(loss)
        if len(self.children) >= self.popsize.llambda:
            self.children = sorted(self.children, key=base._loss)
            population_data: List[np.ndarray] = [c.get_standardized_data(
                reference=self.parametrization) for c in self.children]
            mu: int = self.popsize.mu
            arrays: List[np.ndarray] = population_data[:mu]
            centered_arrays: np.ndarray = np.array([(x - self.
                current_center) for x in arrays])
            cov: np.ndarray = centered_arrays.T.dot(centered_arrays)
            mem_factor: float = 0.9 if self._COVARIANCE_MEMORY else 0.0
            self.covariance *= mem_factor
            self.covariance += (1 - mem_factor) * cov
            self.current_center = sum(arrays) / mu
            self.sigma = math.exp(sum([math.log(c._meta['sigma']) for c in
                self.children[:mu]]) / mu)
            self.parents = self.children[:mu]
            self.children = []

    def _internal_tell_not_asked(self, candidate, loss):
        data: np.ndarray = candidate.get_standardized_data(reference=self.
            parametrization)
        sigma: float = np.linalg.norm(data - self.current_center) / math.sqrt(
            self.dimension)
        candidate._meta['sigma'] = sigma
        self._internal_tell_candidate(candidate, loss)

    def recommend(self):
        return base.Optimizer.recommend(self)


@registry.register
class AXP(base.Optimizer):
    """AX-platform.

    Usually computationally slow and not better than the rest
    in terms of performance per iteration.
    Maybe prefer HyperOpt or Cobyla for low budget optimization.
    """

    def __init__(self, parametrization, budget=None, num_workers=1):
        super().__init__(parametrization, budget=budget, num_workers=
            num_workers)
        try:
            from ax.service.ax_client import AxClient, ObjectiveProperties
        except Exception as e:
            print(f'Pb for creating AX solver')
            raise e
        self.ax_parametrization: List[Dict[str, Union[str, List[float]]]] = [{
            'name': 'x' + str(i), 'type': 'range', 'bounds': [0.0, 1.0]} for
            i in range(self.dimension)]
        self.ax_client: AxClient = AxClient()
        self.ax_client.create_experiment(name='ax_optimization', parameters
            =self.ax_parametrization, objectives={'result':
            ObjectiveProperties(minimize=True)})
        self._trials: List[Any] = []

    def _internal_ask_candidate(self):

        def invsig(x):

            def p_inner(x_inner):
                return np.clip(x_inner, 1e-15, 1.0 - 1e-15)
            return math.log(p_inner(x) / (1 - p_inner(x)))
        if len(self._trials) == 0:
            trial_index_to_param, _trial = self.ax_client.get_next_trials(
                max_trials=1)
            for _, trial in trial_index_to_param.items():
                self._trials.append(trial)
        trial: Any = self._trials[0]
        self._trials = self._trials[1:]
        vals: np.ndarray = np.zeros(self.dimension)
        for i in range(self.dimension):
            vals[i] = invsig(trial['x' + str(i)])
        candidate: p.Parameter = self.parametrization.spawn_child(
            ).set_standardized_data(vals)
        candidate._meta['trial_index'] = self.num_ask
        return candidate

    def _internal_tell_candidate(self, candidate, loss):
        if 'x_probe' in candidate._meta:
            y: List[float] = candidate._meta['x_probe']
        else:
            data: np.ndarray = candidate.get_standardized_data(reference=
                self.parametrization)
            y: List[float] = self._normalizer.forward(data)
        self._fake_function.register(y, -loss)
        self.ax_client.complete_trial(trial_index=candidate._meta[
            'trial_index'], raw_data=loss)

    def _internal_provide_recommendation(self):
        if not self.archive:
            return None
        return self._normalizer.backward(np.array([self.bo.max['params'][
            self._fake_function.key(i)] for i in range(self.dimension)]))

    def _internal_tell_not_asked(self, candidate, loss):
        raise errors.TellNotAskedNotSupportedError()


class PCEDA(EDA):
    _POPSIZE_ADAPTATION = True
    _COVARIANCE_MEMORY = False


class MPCEDA(EDA):
    _POPSIZE_ADAPTATION = True
    _COVARIANCE_MEMORY = True


class MEDA(EDA):
    _POPSIZE_ADAPTATION = False
    _COVARIANCE_MEMORY = True


class ParametrizedTBPSA(base.ConfiguredOptimizer):
    """`Test-based population-size adaptation <https://homepages.fhv.at/hgb/New-Papers/PPSN16_HB16.pdf>`_
    This method, based on adapting the population size, performs the best in
    many noisy optimization problems, even in large dimension

    Parameters
    ----------
    naive: bool
        set to False for noisy problem, so that the best points will be an
        average of the final population.
    initial_popsize: Optional[int]
        initial (and minimal) population size (default: 4 x dimension)

    Note
    ----
    Derived from:
    Hellwig, Michael & Beyer, Hans-Georg. (2016).
    Evolution under Strong Noise: A Self-Adaptive Evolution Strategy
    Reaches the Lower Performance Bound -- the pcCMSA-ES.
    https://homepages.fhv.at/hgb/New-Papers/PPSN16_HB16.pdf
    """

    def __init__(self, *, naive: bool=True, initial_popsize: Optional[int]=None
        ):
        super().__init__(_TBPSA, locals())


TBPSA: base.ConfiguredOptimizer = ParametrizedTBPSA(naive=False).set_name(
    'TBPSA', register=True)
NaiveTBPSA: base.ConfiguredOptimizer = ParametrizedTBPSA().set_name(
    'NaiveTBPSA', register=True)


@registry.register
class NoisyBandit(base.Optimizer):
    """UCB.
    This is upper confidence bound (adapted to minimization),
    with very poor parametrization; in particular, the logarithmic term is set to zero.
    Infinite arms: we add one arm when `20 * #ask >= #arms ** 3`.
    """

    def _internal_ask_candidate(self):
        if 20 * self._num_ask >= len(self.archive) ** 3:
            return self._rng.normal(0, 1, self.dimension)
        if self._rng.choice([True, False]):
            idx: int = self._rng.choice(len(self.archive))
            return np.frombuffer(list(self.archive.bytesdict.keys())[idx])
        return self.current_bests['optimistic'].x

    def _internal_tell_candidate(self, candidate, loss):
        pass


class _PSO(base.Optimizer):

    def __init__(self, parametrization, budget=None, num_workers=1, config=None
        ):
        super().__init__(parametrization, budget=budget, num_workers=
            num_workers)
        self._config: ConfPSO = ConfPSO() if config is None else config
        cases: Dict[str, Tuple[Optional[float], transforms.Transform]] = dict(
            arctan=(0.0, transforms.ArctanBound(0, 1)), identity=(None,
            transforms.Affine(1, 0)), gaussian=(1e-10, transforms.
            CumulativeDensity()))
        self._eps: Optional[float] = cases[self._config.transform][0]
        self._transform: transforms.Transform = cases[self._config.transform][1
            ]
        self.llambda: int = max(40, num_workers)
        if self._config.popsize is not None:
            self.llambda = self._config.popsize
        if self._config.popsize is not None:
            self.llambda = self._config.popsize
        if self._config.popsize is not None:
            self.llambda = self._config.popsize
        self._uid_queue: base.utils.UidQueue = base.utils.UidQueue()
        self.population: Dict[str, p.Parameter] = {}
        self._current: int = -1
        self._warmup_budget: Optional[int] = None
        if self._config.warmup_ratio is not None and budget is not None:
            self._warmup_budget = int(self._config.warmup_ratio * budget)
        self.num_times: List[int] = [0] * len(self.optimizers)

    def _internal_ask_candidate(self):
        if self._warmup_budget is not None:
            if len(self.optimizers
                ) > 1 and self._warmup_budget < self.num_tell:
                ind: int = self.current_bests['pessimistic'
                    ].parameter._meta.get('optim_index', -1)
                if ind >= 0 and ind < len(self.optimizers):
                    if self.num_workers == 1 or self.optimizers[ind
                        ].num_workers > 1:
                        self.optimizers = [self.optimizers[ind]]
        num: int = len(self.optimizers)
        if num == 1:
            optim_index: int = 0
        else:
            self._current += 1
            optim_index: int = self.turns[self._current % len(self.turns)]
            assert optim_index < len(self.optimizers
                ), f'{optim_index}, {self.turns}, {len(self.optimizers)} {self.num_times} {self.str_info} {self.optimizers}'
            opt: base.Optimizer = self.optimizers[optim_index]
        if optim_index is None:
            raise RuntimeError('Something went wrong in optimizer selection')
        opt: base.Optimizer = self.optimizers[optim_index]
        self.num_times[optim_index] += 1
        if (optim_index > 1 and not opt.num_ask and not opt._suggestions and
            not opt.num_tell):
            opt._suggestions.append(self.parametrization.sample())
        candidate: p.Parameter = opt.ask()
        candidate._meta['optim_index'] = optim_index
        return candidate

    def _internal_tell_candidate(self, candidate, loss):
        accepted: int = 0
        if self.no_crossing and len(self.optimizers) > candidate._meta[
            'optim_index']:
            self.optimizers[candidate._meta['optim_index']].tell(candidate,
                loss)
            return
        for opt in self.optimizers:
            try:
                opt.tell(candidate, loss)
                accepted += 1
            except errors.TellNotAskedNotSupportedError:
                pass
        if not accepted:
            raise errors.TellNotAskedNotSupportedError(
                'No sub-optimizer accepted the tell-not-asked')

    def _internal_provide_recommendation(self):
        return {'sub-optim': self.optimizers[0].name}


class ParametrizedCMA(base.ConfiguredOptimizer):
    """CMA-ES optimizer,
    This evolution strategy uses Gaussian sampling, iteratively modified
    for searching in the best directions.
    This optimizer wraps an external implementation: https://github.com/CMA-ES/pycma

    Parameters
    ----------
    scale: float
        scale of the search
    elitist: bool
        whether we switch to elitist mode, i.e. mode + instead of comma,
        i.e. mode in which we always keep the best point in the population.
    popsize: Optional[int] = None
        population size, should be n * self.num_workers for int n >= 1.
        default is max(self.num_workers, 4 + int(3 * np.log(self.dimension)))
    popsize_factor: float = 3.
        factor in the formula for computing the population size
    diagonal: bool
        use the diagonal version of CMA (advised in big dimension)
    high_speed: bool
        use metamodel for recommendation
    fcmaes: bool
        use fast implementation, doesn't support diagonal=True.
        produces equivalent results, preferable for high dimensions or
        if objective function evaluation is fast.
    random_init: bool
        Use a randomized initialization
    inopts: optional dict
        use this to averride any inopts parameter of the wrapped CMA optimizer
        (see https://github.com/CMA-ES/pycma)
    """

    def __init__(self, *, scale: float=1.0, elitist: bool=False, popsize:
        Optional[int]=None, popsize_factor: float=3.0, diagonal: bool=False,
        zero: bool=False, high_speed: bool=False, fcmaes: bool=False,
        random_init: bool=False, inopts: Optional[Dict[str, Any]]=None,
        algorithm: str='quad'):
        super().__init__(_CMA, locals(), as_config=True)
        if zero:
            scale = scale / 1000.0
        if fcmaes:
            if diagonal:
                raise RuntimeError(
                    "fcmaes doesn't support diagonal=True, use fcmaes=False")
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


class SplitOptimizer(base.Optimizer):
    """Combines optimizers, each of them working on their own variables. (use ConfSplitOptimizer)"""

    def __init__(self, parametrization, budget=None, num_workers=1, config=None
        ):
        self._config: Optional['ConfSplitOptimizer'] = config
        super().__init__(parametrization, budget=budget, num_workers=
            num_workers)
        self._subcandidates: Dict[str, List[p.Parameter]] = {}
        subparams: List[p.Parameter] = []
        num_vars: Optional[List[int]
            ] = self._config.num_vars if self._config else None
        num_optims: Optional[int
            ] = self._config.num_optims if self._config else None
        max_num_vars: Optional[int
            ] = self._config.max_num_vars if self._config else None
        if max_num_vars is not None:
            assert num_vars is None, 'num_vars and max_num_vars should not be set at the same time'
            num_vars = [max_num_vars] * (self.dimension // max_num_vars)
            if self.dimension > sum(num_vars):
                num_vars += [self.dimension - sum(num_vars)]
        if num_vars is not None:
            assert sum(num_vars
                ) == self.dimension, f'sum(num_vars)={sum(num_vars)} should be equal to the dimension {self.dimension}.'
            if num_optims is None:
                num_optims = len(num_vars)
            assert num_optims == len(num_vars
                ), f'The number {num_optims} of optimizers should match len(num_vars)={len(num_vars)}.'
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
            num_vars_list: List[int] = (self._config.num_vars if self.
                _config and self._config.num_vars else [])
            for i in range(num_optims):
                if len(num_vars_list) < i + 1:
                    num_vars_list += [self.dimension // num_optims + (self.
                        dimension % num_optims > i)]
                assert num_vars_list[i
                    ] >= 1, 'At least one variable per optimizer.'
                subparams += [p.Array(shape=(num_vars_list[i],))]
        if self._config and self._config.non_deterministic_descriptor:
            for param in subparams:
                param.function.deterministic = False
        self.optims: List[base.Optimizer] = []
        mono: base.OptCls = (self._config.monovariate_optimizer if self.
            _config else base.Optimizer)
        multi: base.OptCls = (self._config.multivariate_optimizer if self.
            _config else base.Optimizer)
        for param in subparams:
            param.random_state = self.parametrization.random_state
            self.optims.append((multi if param.dimension > 1 else mono)(
                param, budget, num_workers))
        assert sum(opt.dimension for opt in self.optims
            ) == self.dimension, 'sum of sub-dimensions should be equal to the total dimension.'

    def _internal_ask_candidate(self):
        candidates: List[p.Parameter] = []
        for i, opt in enumerate(self.optims):
            if self._config and self._config.progressive:
                assert self.budget is not None
                if i > 0 and i / len(self.optims) > math.sqrt(2.0 * self.
                    num_ask / self.budget):
                    candidates.append(opt.parametrization.spawn_child())
                    continue
            candidates.append(opt.ask())
        data: np.ndarray = np.concatenate([c.get_standardized_data(
            reference=opt.parametrization) for c, opt in zip(candidates,
            self.optims)], axis=0)
        cand: p.Parameter = self.parametrization.spawn_child(
            ).set_standardized_data(data)
        self._subcandidates[cand.uid] = candidates
        return cand

    def _internal_tell_candidate(self, candidate, loss):
        candidates: List[p.Parameter] = self._subcandidates.pop(candidate.uid)
        for cand, opt in zip(candidates, self.optims):
            opt.tell(cand, loss)

    def _internal_tell_not_asked(self, candidate, loss):
        data: np.ndarray = candidate.get_standardized_data(reference=self.
            parametrization)
        start: int = 0
        for opt in self.optims:
            local_data: np.ndarray = data[start:start + opt.dimension]
            start += opt.dimension
            local_candidate: p.Parameter = opt.parametrization.spawn_child(
                ).set_standardized_data(local_data)
            opt.tell(local_candidate, loss)

    def _info(self):
        key: str = 'sub-optim'
        optims_info: List[str] = [(x.name if key not in x._info() else x.
            _info()[key]) for x in self.optims]
        return {key: ','.join(optims_info)}


class ContingentSearch(base.Optimizer):
    """A contingent search strategy."""

    def __init__(self, parametrization, budget=None, num_workers=1):
        super().__init__(parametrization, budget=budget, num_workers=
            num_workers)
        self.current_center: np.ndarray = np.zeros(self.dimension)
        self.best: Optional[p.Parameter] = None

    def _internal_ask_candidate(self):
        data: np.ndarray = self.current_center + self._rng.normal(0, 1,
            self.dimension)
        candidate: p.Parameter = self.parametrization.spawn_child(
            ).set_standardized_data(data)
        return candidate

    def _internal_tell_candidate(self, candidate, loss):
        if self.best is None or loss < self.best.loss:
            self.best = candidate
            self.current_center = candidate.get_standardized_data(reference
                =self.parametrization)

    def _internal_provide_recommendation(self):
        if self.best is None:
            return None
        return self.best.get_standardized_data(reference=self.parametrization)

    def enable_pickling(self):
        if self.best is not None:
            self.best = self.best
        super().enable_pickling()


OnePlusOne = ParametrizedOnePlusOne().set_name('OnePlusOne', register=True)
OnePlusLambda = ParametrizedOnePlusOne().set_name('OnePlusLambda', register
    =True)
NoisyOnePlusOne = ParametrizedOnePlusOne(noise_handling='random').set_name(
    'NoisyOnePlusOne', register=True)
DiscreteOnePlusOne = ParametrizedOnePlusOne(mutation='discrete').set_name(
    'DiscreteOnePlusOne', register=True)
SADiscreteLenglerOnePlusOneExp09 = ParametrizedOnePlusOne(tabu_length=1000,
    mutation='lengler', annealing='Exp0.9').set_name(
    'SADiscreteLenglerOnePlusOneExp09', register=True)
SADiscreteLenglerOnePlusOneExp099 = ParametrizedOnePlusOne(tabu_length=1000,
    mutation='lengler', annealing='Exp0.99').set_name(
    'SADiscreteLenglerOnePlusOneExp099', register=True)
SADiscreteLenglerOnePlusOneExp09Auto = ParametrizedOnePlusOne(tabu_length=
    1000, mutation='lengler', annealing='Exp0.9Auto').set_name(
    'SADiscreteLenglerOnePlusOneExp09Auto', register=True)
SADiscreteLenglerOnePlusOneLinAuto = ParametrizedOnePlusOne(tabu_length=
    1000, mutation='lengler', annealing='LinAuto').set_name(
    'SADiscreteLenglerOnePlusOneLinAuto', register=True)
SADiscreteLenglerOnePlusOneLin1 = ParametrizedOnePlusOne(tabu_length=1000,
    mutation='lengler', annealing='Lin1.0').set_name(
    'SADiscreteLenglerOnePlusOneLin1', register=True)
SADiscreteLenglerOnePlusOneLin100 = ParametrizedOnePlusOne(tabu_length=1000,
    mutation='lengler', annealing='Lin100.0').set_name(
    'SADiscreteLenglerOnePlusOneLin100', register=True)
SADiscreteOnePlusOneExp099 = ParametrizedOnePlusOne(tabu_length=1000,
    mutation='discrete', annealing='Exp0.99').set_name(
    'SADiscreteOnePlusOneExp099', register=True)
SADiscreteOnePlusOneLin100 = ParametrizedOnePlusOne(tabu_length=1000,
    mutation='discrete', annealing='Lin100.0').set_name(
    'SADiscreteOnePlusOneLin100', register=True)
SADiscreteOnePlusOneExp09 = ParametrizedOnePlusOne(tabu_length=1000,
    mutation='discrete', annealing='Exp0.9').set_name(
    'SADiscreteOnePlusOneExp09', register=True)
DiscreteOnePlusOneT = ParametrizedOnePlusOne(tabu_length=10000, mutation=
    'discrete').set_name('DiscreteOnePlusOneT', register=True)
PortfolioDiscreteOnePlusOne = ParametrizedOnePlusOne(mutation='portfolio'
    ).set_name('PortfolioDiscreteOnePlusOne', register=True)
PortfolioDiscreteOnePlusOneT = ParametrizedOnePlusOne(tabu_length=10000,
    mutation='portfolio').set_name('PortfolioDiscreteOnePlusOneT', register
    =True)
DiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(mutation='lengler'
    ).set_name('DiscreteLenglerOnePlusOne', register=True)
DiscreteLengler2OnePlusOne = ParametrizedOnePlusOne(mutation='lengler2'
    ).set_name('DiscreteLengler2OnePlusOne', register=True)
DiscreteLengler3OnePlusOne = ParametrizedOnePlusOne(mutation='lengler3'
    ).set_name('DiscreteLengler3OnePlusOne', register=True)
DiscreteLenglerHalfOnePlusOne = ParametrizedOnePlusOne(mutation='lenglerhalf'
    ).set_name('DiscreteLenglerHalfOnePlusOne', register=True)
DiscreteLenglerFourthOnePlusOne = ParametrizedOnePlusOne(mutation=
    'lenglerfourth').set_name('DiscreteLenglerFourthOnePlusOne', register=True)
DiscreteDoerrOnePlusOne = ParametrizedOnePlusOne(mutation='doerr').set_name(
    'DiscreteDoerrOnePlusOne', register=True)
DiscreteDoerrOnePlusOne.no_parallelization = True
CauchyOnePlusOne = ParametrizedOnePlusOne(mutation='cauchy').set_name(
    'CauchyOnePlusOne', register=True)
OptimisticNoisyOnePlusOne = ParametrizedOnePlusOne(noise_handling='optimistic'
    ).set_name('OptimisticNoisyOnePlusOne', register=True)
OptimisticDiscreteOnePlusOne = ParametrizedOnePlusOne(noise_handling=
    'optimistic', mutation='discrete').set_name('OptimisticDiscreteOnePlusOne',
    register=True)
OLNDiscreteOnePlusOne = ParametrizedOnePlusOne(noise_handling='optimistic',
    mutation='lognormal').set_name('OLNDiscreteOnePlusOne', register=True)
NoisyDiscreteOnePlusOne = ParametrizedOnePlusOne(noise_handling=('random', 
    1.0), mutation='discrete').set_name('NoisyDiscreteOnePlusOne', register
    =True)
DoubleFastGADiscreteOnePlusOne = ParametrizedOnePlusOne(mutation='doublefastga'
    ).set_name('DoubleFastGADiscreteOnePlusOne', register=True)
RLSOnePlusOne = ParametrizedOnePlusOne(mutation='rls').set_name('RLSOnePlusOne'
    , register=True)
SparseDoubleFastGADiscreteOnePlusOne = ParametrizedOnePlusOne(mutation=
    'doublefastga', sparse=True).set_name(
    'SparseDoubleFastGADiscreteOnePlusOne', register=True)
RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne = ParametrizedOnePlusOne(
    crossover=True, mutation='portfolio', noise_handling='optimistic'
    ).set_name('RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne',
    register=True)
RecombiningPortfolioDiscreteOnePlusOne = ParametrizedOnePlusOne(crossover=
    True, mutation='portfolio').set_name(
    'RecombiningPortfolioDiscreteOnePlusOne', register=True)


class _CMA(base.Optimizer):
    _CACHE_KEY = '#CMA#datacache'

    def __init__(self, parametrization, budget=None, num_workers=1, config=
        None, algorithm='quad'):
        super().__init__(parametrization, budget=budget, num_workers=
            num_workers)
        self.algorithm: str = algorithm
        self._config: ParametrizedCMA = ParametrizedCMA(
            ) if config is None else config
        pop: Optional[int] = self._config.popsize
        self._popsize: int = max(num_workers, 4 + int(self._config.
            popsize_factor * math.log(self.dimension))
            ) if pop is None else max(pop, num_workers)
        if self._config.elitist:
            self._popsize = max(self._popsize, self.num_workers + 1)
        self._to_be_asked: Deque[np.ndarray] = deque()
        self._to_be_told: List[p.Parameter] = []
        self._num_spawners: int = self._popsize // 2
        self._parents: List[p.Parameter] = [self.parametrization]
        self._es: Optional[Any] = None

    @property
    def es(self):
        scale_multiplier: float = 1.0
        if self.dimension == 1:
            self._config.fcmaes = True
        if p.helpers.Normalizer(self.parametrization).fully_bounded:
            scale_multiplier = 0.3 if self.dimension < 18 else 0.15
        if self._es is None or not self._config.fcmaes and self._es.stop():
            if not self._config.fcmaes:
                import cma
                inopts: Dict[str, Any] = dict(popsize=self._popsize, randn=
                    self._rng.randn, CMA_diagonal=self._config.diagonal,
                    verbose=-9, seed=np.nan, CMA_elitist=self._config.elitist)
                if self._config.inopts is not None:
                    inopts.update(self._config.inopts)
                self._es = cma.CMAEvolutionStrategy(x0=self.parametrization
                    .sample().get_standardized_data(reference=self._config.
                    popsize) if self._config.random_init else np.zeros(self
                    .dimension, dtype=np.float64), sigma0=self._config.
                    scale * scale_multiplier, inopts=inopts)
            else:
                try:
                    from fcmaes import cmaes
                except ImportError as e:
                    raise ImportError(
                        'Please install fcmaes (pip install fcmaes) to use FCMA optimizers'
                        ) from e
                self._es = cmaes.Cmaes(x0=np.zeros(self.dimension, dtype=np
                    .float64), input_sigma=self._config.scale *
                    scale_multiplier, popsize=self._popsize, randn=self.
                    _rng.randn)
        return self._es

    def _internal_ask_candidate(self):
        if not self._to_be_asked:
            self._to_be_asked.extend(self.es.ask())
        data: np.ndarray = self._to_be_asked.popleft()
        parent: p.Parameter = self._parents[self.num_ask % len(self._parents)]
        candidate: p.Parameter = parent.spawn_child().set_standardized_data(
            data, reference=self.parametrization)
        return candidate

    def _internal_tell_candidate(self, candidate, loss):
        if self._CACHE_KEY not in candidate._meta:
            candidate._meta[self._CACHE_KEY] = candidate.get_standardized_data(
                reference=self.parametrization)
        self._to_be_told.append(candidate)
        if len(self._to_be_told) >= self.es.popsize:
            listx: List[np.ndarray] = [c._meta[self._CACHE_KEY] for c in
                self._to_be_told]
            listy: List[float] = [c.loss for c in self._to_be_told]
            args: Tuple[Any, ...] = (listy, listx
                ) if self._config.fcmaes else (listx, listy)
            try:
                self.es.tell(*args)
            except (RuntimeError, AssertionError):
                pass
            else:
                self._parents = sorted(self._to_be_told, key=base._loss)[:
                    self._num_spawners]
            self._to_be_told = []

    def _internal_provide_recommendation(self):
        pessimistic: p.Parameter = self.current_bests['pessimistic'
            ].parameter.get_standardized_data(reference=self.parametrization)
        d: int = self.dimension
        n: int = self.num_ask
        sample_size: int = int(d * d / 2 + d / 2 + 3)
        if self._config.high_speed and n >= sample_size:
            try:
                data: np.ndarray = learn_on_k_best(self.archive,
                    sample_size, self.algorithm)
                return data
            except MetaModelFailure:
                pass
        if self._es is None:
            return pessimistic
        if self._config.fcmaes:
            cma_best: Optional[np.ndarray] = self.es.best_x
        else:
            cma_best: Optional[np.ndarray] = getattr(self.es.result,
                'xbest', None)
        if cma_best is None:
            return pessimistic
        return cma_best


class ParametrizedCMA(base.ConfiguredOptimizer):
    """CMA-ES optimizer,
    This evolution strategy uses Gaussian sampling, iteratively modified
    for searching in the best directions.
    This optimizer wraps an external implementation: https://github.com/CMA-ES/pycma

    Parameters
    ----------
    scale: float
        scale of the search
    elitist: bool
        whether we switch to elitist mode, i.e. mode + instead of comma,
        i.e. mode in which we always keep the best point in the population.
    popsize: Optional[int] = None
        population size, should be n * self.num_workers for int n >= 1.
        default is max(self.num_workers, 4 + int(3 * np.log(self.dimension)))
    popsize_factor: float = 3.
        factor in the formula for computing the population size
    diagonal: bool
        use the diagonal version of CMA (advised in big dimension)
    high_speed: bool
        use metamodel for recommendation
    fcmaes: bool
        use fast implementation, doesn't support diagonal=True.
        produces equivalent results, preferable for high dimensions or
        if objective function evaluation is fast.
    random_init: bool
        Use a randomized initialization
    inopts: optional dict
        use this to averride any inopts parameter of the wrapped CMA optimizer
        (see https://github.com/CMA-ES/pycma)
    """

    def __init__(self, *, scale: float=1.0, elitist: bool=False, popsize:
        Optional[int]=None, popsize_factor: float=3.0, diagonal: bool=False,
        zero: bool=False, high_speed: bool=False, fcmaes: bool=False,
        random_init: bool=False, inopts: Optional[Dict[str, Any]]=None,
        algorithm: str='quad'):
        super().__init__(_CMA, locals(), as_config=True)
        if zero:
            scale = scale / 1000.0
        if fcmaes:
            if diagonal:
                raise RuntimeError(
                    "fcmaes doesn't support diagonal=True, use fcmaes=False")
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

    def __init__(self, parametrization, budget=None, num_workers=1):
        super().__init__(parametrization, budget=budget, num_workers=
            num_workers)
        analysis = p.helpers.analyze(self.parametrization)
        funcinfo = self.parametrization.function
        self.has_noise: bool = not (analysis.deterministic and funcinfo.
            deterministic)
        self.noise_from_instrumentation: bool = (self.has_noise and
            funcinfo.deterministic)
        self.fully_continuous: bool = analysis.continuous
        all_params = p.helpers.flatten(self.parametrization)
        int_layers: List[Any] = list(itertools.chain.from_iterable([
            _layering.Int.filter_from(x) for _, x in all_params]))
        int_layers = [x for x in int_layers if x.arity is not None]
        self.has_discrete_not_softmax: bool = any(not isinstance(lay,
            _datalayers.SoftmaxSampling) for lay in int_layers)
        self._has_discrete: bool = bool(int_layers)
        self._arity: int = max((lay.arity for lay in int_layers), default=-1)
        if self.fully_continuous:
            self._arity = -1
        self._optim: Optional[base.Optimizer] = None
        self._constraints_manager.update(max_trials=1000, penalty_factor=
            1.0, penalty_exponent=1.01)

    @property
    def optim(self):
        if self._optim is None:
            self._optim = self._select_optimizer_cls()(self.parametrization,
                self.budget, self.num_workers)
            self._optim = self._optim if not isinstance(self._optim, NGOptBase
                ) else self._optim.optim
            logger.debug('%s selected %s optimizer.', *(x.name for x in (
                self, self._optim)))
        return self._optim

    def _select_optimizer_cls(self):
        return CMA

    def _internal_ask_candidate(self):
        return self.optim.ask()

    def _internal_tell_candidate(self, candidate, loss):
        self.optim.tell(candidate, loss)

    def recommend(self):
        return self.optim.recommend()

    def _internal_tell_not_asked(self, candidate, loss):
        self.optim.tell(candidate, loss)

    def _info(self):
        out: Dict[str, Any] = {'sub-optim': self.optim.name}
        out.update(self.optim._info())
        return out

    def enable_pickling(self):
        self.optim.enable_pickling()
