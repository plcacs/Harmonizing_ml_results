import numpy as np
from typing import Optional

class _OnePlusOne(base.Optimizer):
    """Simple but sometimes powerful optimization algorithm.

    We use the one-fifth adaptation rule, going back to Schumer and Steiglitz (1968).
    It was independently rediscovered by Devroye (1972) and Rechenberg (1973).
    We use asynchronous updates, so that the 1+1 can actually be parallel and even
    performs quite well in such a context - this is naturally close to 1+lambda.

    Parameters
    ----------
    parametrization: :class:`p.Parameter`
        the parameterization of the problem
    budget: Optional[int]
        the budget for the optimization (default is None)
    num_workers: int
        the number of workers (default is 1)
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

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1, *, noise_handling: Optional[str] = None, mutation: str = "gaussian", crossover: bool = False, rotation: bool = False, annealing: str = "none", use_pareto: bool = False, sparse: bool = False, smoother: bool = False, super_radii: bool = False, roulette_size: int = 2, antismooth: int = 55, crossover_type: str = "none"):
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.parametrization.tabu_length = tabu_length
        if forced_discretization:
            self.parametrization.set_integer_casting()
        self.antismooth = antismooth
        self.crossover_type = crossover_type
        self.roulette_size = roulette_size
        assert crossover or not rotation, 'We can not have both rotation and not crossover.'
        self._sigma = 1
        self._previous_best_loss = float('inf')
        self._best_recent_mr = 0.2
        self.inds = np.array([True] * self.dimension)
        self.imr = 0.2
        self.use_pareto = use_pareto
        self.smoother = smoother
        self.super_radii = super_radii
        self.annealing = annealing
        self._annealing_base = None
        self._max_loss = -float('inf')
        self.sparse = int(sparse)
        all_params = p.helpers.flatten(self.parametrization)
        arities = [len(param.choices) for _, param in all_params if isinstance(param, p.TransitionChoice)]
        arity = max(arities, default=500)
        self.arity_for_discrete_mutation = arity
        if noise_handling is not None:
            if isinstance(noise_handling, str):
                assert noise_handling in ['random', 'optimistic'], f"Unkwnown noise handling: '{noise_handling}'"
            else:
                assert isinstance(noise_handling, tuple), 'noise_handling must be a string or  a tuple of type (strategy, factor)'
                assert noise_handling[1] > 0.0, 'the factor must be a float greater than 0'
                assert noise_handling[0] in ['random', 'optimistic'], f"Unkwnown noise handling: '{noise_handling}'"
        assert mutation in ['gaussian', 'cauchy', 'discrete', 'fastga', 'rls', 'doublefastga', 'adaptive', 'coordinatewise_adaptive', 'portfolio', 'discreteBSO', 'lengler', 'lengler2', 'lengler3', 'lenglerfourth', 'lenglerhalf', 'doerr'], f"Unkwnown mutation: '{mutation}'"
        if mutation == 'adaptive':
            self._adaptive_mr = 0.5
        elif mutation == 'lognormal':
            self._global_mr = 0.2
            self._memory_index = 0
            self._memory_size = 12
            self._best_recent_loss = float('inf')
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
            self._velocity = self._rng.uniform(size=self.dimension) * arity / 4.0
            self._modified_variables = np.array([True] * self.dimension)
        self.noise_handling = noise_handling
        self.mutation = mutation
        self.crossover = crossover
        self.rotation = rotation
        if mutation == 'doerr':
            assert num_workers == 1, 'Doerr mutation is implemented only in the sequential case.'
            self._doerr_mutation_rates = [1, 2]
            self._doerr_mutation_rewards = [0.0, 0.0]
            self._doerr_counters = [0.0, 0.0]
            self._doerr_epsilon = 0.25
            self._doerr_gamma = 1 - 2 / self.dimension
            self._doerr_current_best = float('inf')
            i = 3
            j = 2
            self._doerr_index = -1
            while i < self.dimension:
                self._doerr_mutation_rates += [i]
                self._doerr_mutation_rewards += [0.0]
                self._doerr_counters += [0.0]
                i += j
                j += 2
        assert self.parametrization.tabu_length == tabu_length

    def _internal_ask_candidate(self):
        noise_handling = self.noise_handling
        if not self._num_ask:
            out = self.parametrization.spawn_child()
            out._meta['sigma'] = self._sigma
            return out
        if noise_handling is not None:
            limit = (0.05 if isinstance(noise_handling, str) else noise_handling[1]) * len(self.archive) ** 3
            strategy = noise_handling if isinstance(noise_handling, str) else noise_handling[0]
            if self._num_ask <= limit:
                if strategy in ['cubic', 'random']:
                    idx = self._rng.choice(len(self.archive))
                    return list(self.archive.values())[idx].parameter.spawn_child()
                elif strategy == 'optimistic':
                    return self.current_bests['optimistic'].parameter.spawn_child()
        mutator = mutations.Mutator(self._rng)
        pessimistic = self.current_bests['pessimistic'].parameter
        if self.smoother and self._num_ask % max(self.num_workers + 1, self.antismooth) == 0 and isinstance(self.parametrization, p.Array):
            possible_radii = [3] if not self.super_radii else [3, 3 + np.random.randint(int(np.sqrt(np.sqrt(self.dimension))))]
            self.suggest(smooth_copy(pessimistic, possible_radii=possible_radii).value)
        if self.num_objectives > 1 and self.use_pareto:
            pareto = self.pareto_front()
            pessimistic = pareto[self._rng.choice(len(pareto))].spawn_child()
        ref = self.parametrization
        if self.crossover and self._num_ask % 2 == 1 and (len(self.archive) > 2):
            data = mutator.crossover(pessimistic.get_standardized_data(reference=ref), mutator.get_roulette(self.archive, num=self.roulette_size), rotation=self.rotation, crossover_type=self.crossover_type)
            return pessimistic.set_standardized_data(data, reference=ref)
        mutation = self.mutation
        if self._annealing_base is not None:
            assert self.annealing != 'none'
            pessimistic.set_standardized_data(self._annealing_base, reference=ref)
        if mutation in ('gaussian', 'cauchy'):
            step = self._rng.normal(0, 1, self.dimension) if mutation == 'gaussian' else self._rng.standard_cauchy(self.dimension)
            out = pessimistic.set_standardized_data(self._sigma * step)
            out._meta['sigma'] = self._sigma
            return out
        else:
            pessimistic_data = pessimistic.get_standardized_data(reference=ref)
            if mutation == 'crossover':
                if self._num_ask % 2 == 0 or len(self.archive) < 3:
                    data = mutator.portfolio_discrete_mutation(pessimistic_data, arity=self.arity_for_discrete_mutation)
                else:
                    data = mutator.crossover(pessimistic_data, mutator.get_roulette(self.archive, num=2))
            elif 'lognormal' in mutation:
                mutation_rate = max(0.1 / self.dimension, self._global_mr)
                assert mutation_rate > 0.0
                individual_mutation_rate = 1.0 / (1.0 + (1.0 - mutation_rate) / mutation_rate * np.exp(0.22 * np.random.randn()))
                data = mutator.portfolio_discrete_mutation(pessimistic_data, intensity=individual_mutation_rate * self.dimension, arity=self.arity_for_discrete_mutation)
            elif mutation == 'adaptive':
                data = mutator.portfolio_discrete_mutation(pessimistic_data, intensity=max(1, int(self._adaptive_mr * self.dimension)), arity=self.arity_for_discrete_mutation)
            elif mutation == 'discreteBSO':
                assert self.budget is not None, 'DiscreteBSO needs a budget.'
                intensity = int(self.dimension - self._num_ask * self.dimension / self.budget)
                if intensity < 1:
                    intensity = 1
                data = mutator.portfolio_discrete_mutation(pessimistic_data, intensity=intensity, arity=self.arity_for_discrete_mutation)
            elif mutation == 'coordinatewise_adaptive':
                self._modified_variables = np.array([True] * self.dimension)
                data = mutator.coordinatewise_mutation(pessimistic_data, self._velocity, self._modified_variables, arity=self.arity_for_discrete_mutation)
            elif mutation == 'lengler':
                alpha = 1.54468
                intensity = int(max(1, self.dimension * (alpha * np.log(self.num_ask) / self.num_ask)))
                data = mutator.portfolio_discrete_mutation(pessimistic_data, intensity=intensity, arity=self.arity_for_discrete_mutation)
            elif mutation == 'lengler2':
                alpha = 3.0
                intensity = int(max(1, self.dimension * (alpha * np.log(self.num_ask) / self.num_ask)))
                data = mutator.portfolio_discrete_mutation(pessimistic_data, intensity=intensity, arity=self.arity_for_discrete_mutation)
            elif mutation == 'lengler3':
                alpha = 9.0
                intensity = int(max(1, self.dimension * (alpha * np.log(self.num_ask) / self.num_ask)))
                data = mutator.portfolio_discrete_mutation(pessimistic_data, intensity=intensity, arity=self.arity_for_discrete_mutation)
            elif mutation == 'lenglerfourth':
                alpha = 0.4
                intensity = int(max(1, self.dimension * (alpha * np.log(self.num_ask) / self.num_ask)))
                data = mutator.portfolio_discrete_mutation(pessimistic_data, intensity=intensity, arity=self.arity_for_discrete_mutation)
            elif mutation == 'lenglerhalf':
                alpha = 0.8
                intensity = int(max(1, self.dimension * (alpha * np.log(self.num_ask) / self.num_ask)))
                data = mutator.portfolio_discrete_mutation(pessimistic_data, intensity=intensity, arity=self.arity_for_discrete_mutation)
            elif mutation == 'doerr':
                assert self._doerr_index == -1, 'We should have used this index in tell.'
                if self._rng.uniform() < self._doerr_epsilon:
                    index = self._rng.choice(range(len(self._doerr_mutation_rates)))
                    self._doerr_index = index
                else:
                    index = self._doerr_mutation_rewards.index(max(self._doerr_mutation_rewards))
                    self._doerr_index = -1
                intensity = self._doerr_mutation_rates[index]
                data = mutator.portfolio_discrete_mutation(pessimistic_data, intensity=intensity, arity=self.arity_for_discrete_mutation)
            else:
                func = {'discrete': mutator.discrete_mutation, 'fastga': mutator.doerr_discrete_mutation, 'doublefastga': mutator.doubledoerr_discrete_mutation, 'rls': mutator.rls_mutation, 'portfolio': mutator.portfolio_discrete_mutation}[mutation]
                data = func(pessimistic_data, arity=self.arity_for_discrete_mutation)
            if self.sparse > 0:
                data = np.asarray(data)
                zeroing = self._rng.randint(data.size + 1, size=data.size).reshape(data.shape) < 1 + self._rng.randint(self.sparse)
                data[zeroing] = 0.0
            candidate = pessimistic.set_standardized_data(data, reference=ref)
            if mutation == 'coordinatewise_adaptive':
                candidate._meta['modified_variables'] = (self._modified_variables,)
            if 'lognormal' in mutation:
                candidate._meta['individual_mutation_rate'] = individual_mutation_rate
            return candidate

    def _internal_tell_candidate(self, candidate, loss):
        if self.annealing != 'none':
            assert isinstance(self.budget, int)
            delta = self._previous_best_loss - loss
            if loss > self._max_loss:
                self._max_loss = loss
            if delta >= 0:
                self._annealing_base = candidate
            elif self.num_ask < self.budget:
                amplitude = max(1.0, self._max_loss - self._previous_best_loss)
                annealing_dict = {'Exp0.9': 0.33 * amplitude * 0.9 ** self.num_ask, 'Exp0.99': 0.33 * amplitude * 0.99 ** self.num_ask, 'Exp0.9Auto': 0.33 * amplitude * (0.001 ** (1.0 / self.budget)) ** self.num_ask, 'Lin100.0': 100.0 * amplitude * (1 - self.num_ask / (self.budget + 1)), 'Lin1.0': 1.0 * amplitude * (1 - self.num_ask / (self.budget + 1)), 'LinAuto': 10.0 * amplitude * (1 - self.num_ask / (self.budget + 1))}
                T = annealing_dict[self.annealing]
                if T > 0.0:
                    proba = np.exp(delta / T)
                    if self._rng.rand() < proba:
                        self._annealing_base = candidate
        if self._previous_best_loss != loss:
            self._sigma *= 2.0 if loss < self._previous_best_loss else 0.84
        if self.mutation == 'doerr' and self._doerr_current_best < float('inf') and (self._doerr_index >= 0):
            improvement = max(0.0, self._doerr_current_best - loss)
            index = self._doerr_index
            counter = self._doerr_counters[index]
            self._doerr_mutation_rewards[index] = (self._doerr_gamma * counter * self._doerr_mutation_rewards[index] + improvement) / (self._doerr_gamma * counter + 1)
            self._doerr_counters = [self._doerr_gamma * x for x in self._doerr_counters]
            self._doerr_counters[index] += 1
            self._doerr_index = -1
        if self.mutation == 'doerr':
            self._doerr_current_best = min(self._doerr_current_best, loss)
        elif self.mutation == 'adaptive':
            factor = 1.2 if loss <= self._previous_best_loss else 0.731
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

    def __init__(self, *, noise_handling=None, tabu_length=0, mutation='gaussian', crossover=False, rotation=False, annealing='none', use_pareto=False, sparse=False, smoother=False, super_radii=False, roulette_size=2, antismooth=55, crossover_type='none'):
        super().__init__(_OnePlusOne, locals())
OnePlusOne = ParametrizedOnePlusOne().set_name('OnePlusOne', register=True)
OnePlusLambda = ParametrizedOnePlusOne().set_name('OnePlusLambda', register=True)
NoisyOnePlusOne = ParametrizedOnePlusOne(noise_handling='random').set_name('NoisyOnePlusOne', register=True)
DiscreteOnePlusOne = ParametrizedOnePlusOne(mutation='discrete').set_name('DiscreteOnePlusOne', register=True)
SADiscreteLenglerOnePlusOneExp09 = ParametrizedOnePlusOne(tabu_length=1000, mutation='lengler', annealing='Exp0.9').set_name('SADiscreteLenglerOnePlusOneExp09', register=True)
SADiscreteLenglerOnePlusOneExp099 = ParametrizedOnePlusOne(tabu_length=1000, mutation='lengler', annealing='Exp0.99').set_name('SADiscreteLenglerOnePlusOneExp099', register=True)
SADiscreteLenglerOnePlusOneExp09Auto = ParametrizedOnePlusOne(tabu_length=1000, mutation='lengler', annealing='Exp0.9Auto').set_name('SADiscreteLenglerOnePlusOneExp09Auto', register=True)
SADiscreteLenglerOnePlusOneLinAuto = ParametrizedOnePlusOne(tabu_length=1000, mutation='lengler', annealing='LinAuto').set_name('SADiscreteLenglerOnePlusOneLinAuto', register=True)
SADiscreteLenglerOnePlusOneLin1 = ParametrizedOnePlusOne(tabu_length=1000, mutation='lengler', annealing='Lin1.0').set_name('SADiscreteLenglerOnePlusOneLin1', register=True)
SADiscreteLenglerOnePlusOneLin100 = ParametrizedOnePlusOne(tabu_length=1000, mutation='lengler', annealing='Lin100.0').set_name('SADiscreteLenglerOnePlusOneLin100', register=True)
SADiscreteOnePlusOneExp099 = ParametrizedOnePlusOne(tabu_length=1000, mutation='discrete', annealing='Exp0.99').set_name('SADiscreteOnePlusOneExp099', register=True)
SADiscreteOnePlusOneLin100 = ParametrizedOnePlusOne(tabu_length=1000, mutation='discrete', annealing='Lin100.0').set_name('SADiscreteOnePlusOneLin100', register=True)
SADiscreteOnePlusOneExp09 = ParametrizedOnePlusOne(tabu_length=1000, mutation='discrete', annealing='Exp0.9').set_name('SADiscreteOnePlusOneExp09', register=True)
DiscreteOnePlusOneT = ParametrizedOnePlusOne(tabu_length=10000, mutation='discrete').set_name('DiscreteOnePlusOneT', register=True)
PortfolioDiscreteOnePlusOne = ParametrizedOnePlusOne(mutation='portfolio').set_name('PortfolioDiscreteOnePlusOne', register=True)
PortfolioDiscreteOnePlusOneT = ParametrizedOnePlusOne(tabu_length=10000, mutation='portfolio').set_name('PortfolioDiscreteOnePlusOneT', register=True)
DiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(mutation='lengler').set_name('DiscreteLenglerOnePlusOne', register=True)
DiscreteLengler2OnePlusOne = ParametrizedOnePlusOne(mutation='lengler2').set_name('DiscreteLengler2OnePlusOne', register=True)
DiscreteLengler3OnePlusOne = ParametrizedOnePlusOne(mutation='lengler3').set_name('DiscreteLengler3OnePlusOne', register=True)
DiscreteLenglerHalfOnePlusOne = ParametrizedOnePlusOne(mutation='lenglerhalf').set_name('DiscreteLenglerHalfOnePlusOne', register=True)
DiscreteLenglerFourthOnePlusOne = ParametrizedOnePlusOne(mutation='lenglerfourth').set_name('DiscreteLenglerFourthOnePlusOne', register=True)
DiscreteLenglerOnePlusOneT = ParametrizedOnePlusOne(tabu_length=10000, mutation='lengler').set_name('DiscreteLenglerOnePlusOneT', register=True)
AdaptiveDiscreteOnePlusOne = ParametrizedOnePlusOne(mutation='adaptive').set_name('AdaptiveDiscreteOnePlusOne', register=True)
LognormalDiscreteOnePlusOne = ParametrizedOnePlusOne(mutation='lognormal').set_name('LognormalDiscreteOnePlusOne', register=True)
XLognormalDiscreteOnePlusOne = ParametrizedOnePlusOne(mutation='xlognormal').set_name('XLognormalDiscreteOnePlusOne', register=True)
XSmallLognormalDiscreteOnePlusOne = ParametrizedOnePlusOne(mutation='xsmalllognormal').set_name('XSmallLognormalDiscreteOnePlusOne', register=True)
BigLognormalDiscreteOnePlusOne = ParametrizedOnePlusOne(mutation='biglognormal').set_name('BigLognormalDiscreteOnePlusOne', register=True)
SmallLognormalDiscreteOnePlusOne = ParametrizedOnePlusOne(mutation='smalllognormal').set_name('SmallLognormalDiscreteOnePlusOne', register=True)
TinyLognormalDiscreteOnePlusOne = ParametrizedOnePlusOne(mutation='tinylognormal').set_name('TinyLognormalDiscreteOnePlusOne', register=True)
HugeLognormalDiscreteOnePlusOne = ParametrizedOnePlusOne(mutation='hugelognormal').set_name('HugeLognormalDiscreteOnePlusOne', register=True)
AnisotropicAdaptiveDiscreteOnePlusOne = ParametrizedOnePlusOne(mutation='coordinatewise_adaptive').set_name('AnisotropicAdaptiveDiscreteOnePlusOne', register=True)
DiscreteBSOOnePlusOne = ParametrizedOnePlusOne(mutation='discreteBSO').set_name('DiscreteBSOOnePlusOne', register=True)
DiscreteDoerrOnePlusOne = ParametrizedOnePlusOne(mutation='doerr').set_name('DiscreteDoerrOnePlusOne', register=True)
DiscreteDoerrOnePlusOne.no_parallelization = True
CauchyOnePlusOne = ParametrizedOnePlusOne(mutation='cauchy').set_name('CauchyOnePlusOne', register=True)
OptimisticNoisyOnePlusOne = ParametrizedOnePlusOne(noise_handling='optimistic').set_name('OptimisticNoisyOnePlusOne', register=True)
OptimisticDiscreteOnePlusOne = ParametrizedOnePlusOne(noise_handling='optimistic', mutation='discrete').set_name('OptimisticDiscreteOnePlusOne', register=True)
OLNDiscreteOnePlusOne = ParametrizedOnePlusOne(noise_handling=('optimistic', 1.0), mutation='lognormal').set_name('OLNDiscreteOnePlusOne', register=True)
NoisyDiscreteOnePlusOne = ParametrizedOnePlusOne(noise_handling=('random', 1.0), mutation='discrete').set_name('NoisyDiscreteOnePlusOne', register=True)
DoubleFastGADiscreteOnePlusOne = ParametrizedOnePlusOne(mutation='doublefastga').set_name('DoubleFastGADiscreteOnePlusOne', register=True)
RLSOnePlusOne = ParametrizedOnePlusOne(mutation='rls').set_name('RLSOnePlusOne', register=True)
SparseDoubleFastGADiscreteOnePlusOne = ParametrizedOnePlusOne(mutation='doublefastga', sparse=True).set_name('SparseDoubleFastGADiscreteOnePlusOne', register=True)
RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne = ParametrizedOnePlusOne(crossover=True, mutation='portfolio', noise_handling='optimistic').set_name('RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne', register=True)
RecombiningPortfolioDiscreteOnePlusOne = ParametrizedOnePlusOne(crossover=True, mutation='portfolio').set_name('RecombiningPortfolioDiscreteOnePlusOne', register=True)

class _CMA(base.Optimizer):
    """CMA-ES optimizer,
    This evolution strategy uses Gaussian sampling, iteratively modified
    for searching in the best directions.
    This optimizer wraps an external implementation: https://github.com/CMA-ES/pycma

    Parameters
    ----------
    parametrization: :class:`p.Parameter`
        the parameterization of the problem
    budget: Optional[int]
        the budget for the optimization (default is None)
    num_workers: int
        the number of workers (default is 1)
    config: Optional[ParametrizedCMA]
        configuration of the optimizer (default is None)
    algorithm: str
        algorithm used to optimize the mean of the distribution (default is "quad")
    """

    _CACHE_KEY = '#CMA#datacache'

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1, config: Optional[ParametrizedCMA] = None, algorithm: str = 'quad'):
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.algorithm = algorithm
        self._config = ParametrizedCMA() if config is None else config
        pop = self._config.popsize
        self._popsize = max(num_workers, 4 + int(self._config.popsize_factor * np.log(self.dimension))) if pop is None else max(pop, num_workers)
        if self._config.elitist:
            self._popsize = max(self._popsize, self.num_workers + 1)
        self._to_be_asked = deque()
        self._to_be_told = []
        self._num_spawners = self._popsize // 2
        self._parents = [self.parametrization]
        self._es = None

    @property
    def es(self):
        scale_multiplier = 1.0
        if self.dimension == 1:
            self._config.fcmaes = True
        if p.helpers.Normalizer(self.parametrization).fully_bounded:
            scale_multiplier = 0.3 if self.dimension < 18 else 0.15
        if self._es is None or (not self._config.fcmaes and self._es.stop()):
            if not self._config.fcmaes:
                import cma
                inopts = dict(popsize=self._popsize, randn=self._rng.randn, CMA_diagonal=self._config.diagonal, verbose=-9, seed=np.nan, CMA_elitist=self._config.elitist)
                inopts.update(self._config.inopts if self._config.inopts is not None else {})
                self._es = cma.CMAEvolutionStrategy(x0=self.parametrization.sample().get_standardized_data(reference=self.parametrization) if self._config.random_init else np.zeros(self.dimension, dtype=np.float64), sigma0=self._config.scale * scale_multiplier, inopts=inopts)
            else:
                try:
                    from fcmaes import cmaes
                except ImportError as e:
                    raise ImportError('Please install fcmaes (pip install fcmaes) to use FCMA optimizers') from e
                self._es = cmaes.Cmaes(x0=np.zeros(self.dimension, dtype=np.float64), input_sigma=self._config.scale * scale_multiplier, popsize=self._popsize, randn=self._rng.randn)
        return self._es

    def _internal_ask_candidate(self):
        if not self._to_be_asked:
            self._to_be_asked.extend(self.es.ask())
        data = self._to_be_asked.popleft()
        parent = self._parents[self.num_ask % len(self._parents)]
        candidate = parent.spawn_child().set_standardized_data(data)
        return candidate

    def _internal_tell_candidate(self, candidate, loss):
        if self._CACHE_KEY not in candidate._meta:
            candidate._meta[self._CACHE_KEY] = candidate.get_standardized_data(reference=self.parametrization)
        self._to_be_told.append(candidate)
        if len(self._to_be_told) >= self.es.popsize:
            listx = [c._meta[self._CACHE_KEY] for c in self._to_be_told]
            listy = [c.loss for c in self._to_be_told]
            args = (listy, listx) if self._config.fcmaes else (listx, listy)
            try:
                self.es.tell(*args)
            except (RuntimeError, AssertionError):
                pass
            else:
                self._parents = sorted(self._to_be_told, key=base._loss)[:self._num_spawners]
            self._to_be_told = []

    def _internal_provide_recommendation(self):
        return self.current_bests['optimistic'].parameter

    def enable_pickling(self):
        self._es.enable_pickling()

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
    popsize: Optional[int]
        population size, should be n * self.num_workers for int n >= 1.
        default is max(self.num_workers, 4 + int(3 * np.log(self.dimension)))
    popsize_factor: float
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

    def __init__(self, *, scale=1.0, elitist=False, popsize=None, popsize_factor=3.0, diagonal=False, zero=False, high_speed=False, fcmaes=False, random_init=False, inopts=None, algorithm='quad'):
        super().__init__(_CMA, locals(), as_config=True)
        if zero:
            scale = scale / 1000.0
        if fcmaes:
            if diagonal:
                raise RuntimeError("fcmaes doesn't support diagonal=True, use fcmaes=False")
        self.scale = scale
        self.elitist = elitist
        self.zero = zero
        self.popsize = popsize
        self.popsize_factor = popsize_factor
        self.diagonal = diagonal
        self.fcmaes = fcmaes
        self.high_speed = high_speed
        self.random_init = random_init
        self.inopts = inopts

@registry.register
class ChoiceBase(base.Optimizer):
    """Nevergrad optimizer by competence map."""

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1):
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        analysis = p.helpers.analyze(self.parametrization)
        funcinfo = self.parametrization.function
        self.has_noise = not (analysis.deterministic and funcinfo.deterministic)
        self.noise_from_instrumentation = self.has_noise and funcinfo.deterministic
        self.fully_continuous = analysis.continuous
        all_params = p.helpers.flatten(self.parametrization)
        int_layers = list(itertools.chain.from_iterable([_layering.Int.filter_from(x) for _, x in all_params]))
        int_layers = [x for x in int_layers if x.arity is not None]
        self.has_discrete_not_softmax = any((not isinstance(lay, _datalayers.SoftmaxSampling) for lay in int_layers))
        self._has_discrete = bool(int_layers)
        self._arity = max((lay.arity for lay in int_layers), default=-1)
        if self.fully_continuous:
            self._arity = -1
        self._optim = None
        self._constraints_manager.update(max_trials=1000, penalty_factor=1.0, penalty_exponent=1.01)

    @property
    def optim(self):
        if self._optim is None:
            self._optim = self._select_optimizer_cls()(self.parametrization, self.budget, self.num_workers)
            self._optim = self._optim if not isinstance(self._optim, NGOptBase) else self._optim.optim
            logger.debug('%s selected %s optimizer.', *(x.name for x in (self, self._optim)))
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
        out = {'sub-optim': self.optim.name}
        out.update(self.optim._info())
        return out

    def enable_pickling(self):
        self.optim.enable_pickling()

class MetaCMA(ChoiceBase):
    """Nevergrad CMA optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self):
        optCls = ChoiceBase
        funcinfo = self.parametrization.function
        if self.budget is not None and self.fully_continuous and (self.num_workers <= self._popsize) and (self.dimension < 100) and (self.budget < self.dimension * 50) and (self.budget > 50):
            return MetaModel
        elif self.budget is not None and self.fully_continuous and (self.num_workers <= self._popsize) and (self.dimension < 100) and (self.budget < self.dimension * 5) and (self.budget > 50):
            return MetaModel
        elif self.budget is not None and self.fully_continuous and (self.num_workers <= self._popsize) and (self.dimension < 100) and (self.budget < self.dimension * 50) and (self.num_workers > self.budget / 5):
            return MetaCMA
        elif self.budget is not None and self.fully_continuous and (self.num_workers <= self._popsize) and (self.dimension < 100) and (self.budget < self.dimension * 5) and (self.num_workers > self.budget / 5):
            return MetaCMA
        elif self.dimension >= 60 and (not funcinfo.metrizable):
            return CMA
        elif self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            return DoubleFastGADiscreteOnePlusOne
        elif self._arity > 0:
            if self._arity == 2:
                return DiscreteOnePlusOne
            else:
                return AdaptiveDiscreteOnePlusOne if self._arity < 5 else CMandAS2
        elif self.has_noise and self.fully_continuous and (self.dimension > 100):
            return ConfSplitOptimizer(num_optims=13, progressive=True, multivariate_optimizer=OptimisticDiscreteOnePlusOne)
        elif self.has_noise and self.fully_continuous:
            if self.budget > 100:
                return OnePlusOne if self.noise_from_instrumentation or self.num_workers > 1 else SQP
            else:
                return OnePlusOne
        elif self.has_discrete_not_softmax or not funcinfo.metrizable or (not self.fully_continuous):
            return DoubleFastGADiscreteOnePlusOne
        elif self.num_workers > self.budget / 5:
            if self.num_workers > self.budget / 2.0 or self.budget < self.dimension:
                return MetaTuneRecentering
            else:
                return NaiveTBPSA
        elif self.num_workers == 1 and self.budget > 6000 and (self.dimension > 7):
            return ChainNaiveTBPSACMAPowell
        elif self.num_workers == 1 and self.budget < self.dimension * 30:
            if self.dimension > 30:
                return OnePlusOne
            elif self.dimension < 5:
                return MetaModel
            else:
                return Cobyla
        elif self.dimension > 2000:
            return DE if self.dimension > 2000 else MetaCMA if self.dimension > 1 else OnePlusOne
        elif self.dimension < 10 and self.budget < 500:
            return MetaModel
        elif 3 * self.num_workers > self.dimension ** 2 and self.budget > self.dimension ** 2:
            return MetaModel
        else:
            return CMA

class _EMNA(base.Optimizer):
    """Simple Estimation of Multivariate Normal Algorithm (EMNA)."""

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1, isotropic: bool = True, naive: bool = True, population_size_adaptation: bool = False, initial_popsize: Optional[int] = None):
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.isotropic = isotropic
        self.naive = naive
        self.population_size_adaptation = population_size_adaptation
        self.min_coef_parallel_context = 8
        if initial_popsize is None:
            initial_popsize = self.dimension
        if self.isotropic:
            self.sigma = 1.0
        else:
            self.sigma = np.ones(self.dimension)
        self.popsize = _PopulationSizeController(llambda=4 * initial_popsize, mu=initial_popsize, dimension=self.dimension, num_workers=num_workers)
        if not self.population_size_adaptation:
            self.popsize.mu = max(16, self.dimension)
            self.popsize.llambda = 4 * self.popsize.mu
            self.popsize.llambda = max(self.popsize.llambda, num_workers)
            self.popsize.llambda = min(self.popsize.llambda, self.min_coef_parallel_context * self.dimension)
            warnings.warn('Budget may be too small in front of the dimension for EMNA', errors.InefficientSettingsWarning)
        self.current_center = np.zeros(self.dimension)
        self.parents = [self.parametrization]
        self.children = []

    def recommend(self):
        if self.naive:
            return self.current_bests['optimistic'].parameter
        else:
            out = self.parametrization.spawn_child()
            with p.helpers.deterministic_sampling(out):
                out.set_standardized_data(self.current_center)
            return out

    def _internal_ask_candidate(self):
        sigma_tmp = self.sigma
        if self.population_size_adaptation and self.popsize.llambda < self.min_coef_parallel_context * self.dimension:
            sigma_tmp = self.sigma * np.exp(self._rng.normal(0, 1) / np.sqrt(self.dimension))
        individual = self.current_center + sigma_tmp * self._rng.normal(0, 1, self.dimension)
        parent = self.parents[self.num_ask % len(self.parents)]
        candidate = parent.spawn_child().set_standardized_data(individual, reference=self.parametrization)
        if parent is self.parametrization:
            candidate.heritage['lineage'] = candidate.uid
        candidate._meta['sigma'] = sigma_tmp
        return candidate

    def _internal_tell_candidate(self, candidate, loss):
        if self.population_size_adaptation:
            self.popsize.add_value(loss)
        self.children.append(candidate)
        if len(self.children) >= self.popsize.llambda:
            self.children.sort(key=base._loss)
            self.parents = self.children[:self.popsize.mu]
            self.children = []
            self.current_center = sum((c.get_standardized_data(reference=self.parametrization) for c in self.parents)) / self.popsize.mu
            if self.population_size_adaptation:
                if self.popsize.llambda < self.min_coef_parallel_context * self.dimension:
                    self.sigma = np.exp(np.sum(np.log([c._meta['sigma'] for c in self.parents]), axis=0 if self.isotropic else None) / self.popsize.mu)
                else:
                    stdd = [(self.parents[i].get_standardized_data(reference=self.parametrization) - self.current_center) ** 2 for i in range(self.popsize.mu)]
                    self.sigma = np.sqrt(np.sum(stdd) / (self.popsize.mu * (self.dimension if self.isotropic else 1)))
            else:
                stdd = [(self.parents[i].get_standardized_data(reference=self.parametrization) - self.current_center) ** 2 for i in range(self.popsize.mu)]
                self.sigma = np.sqrt(np.sum(stdd, axis=0 if self.isotropic else None) / (self.popsize.mu * (self.dimension if self.isotropic else 1)))
            if self.num_workers / self.dimension > 32:
                imp = max(1, (np.log(self.popsize.llambda) / 2) ** (1 / self.dimension))
                self.sigma /= imp

    def _internal_tell_not_asked(self, candidate, loss):
        raise errors.TellNotAskedNotSupportedError

class EMNA(base.ConfiguredOptimizer):
    """Estimation of Multivariate Normal Algorithm
    This algorithm is quite efficient in a parallel context, i.e. when
    the population size is large.

    Parameters
    ----------
    isotropic: bool
        isotropic version on EMNA if True, i.e. we have an
        identity matrix for the Gaussian, else  we here consider the separable
        version, meaning we have a diagonal matrix for the Gaussian (anisotropic)
    naive: bool
        set to False for noisy problem, so that the best points will be an
        average of the final population.
    population_size_adaptation: bool
        population size automatically adapts to the landscape
    initial_popsize: Optional[int]
        initial (and minimal) population size (default: 4 x dimension)
    """

    def __init__(self, *, isotropic: bool = True, naive: bool = True, population_size_adaptation: bool = False, initial_popsize: Optional[int] = None):
        super().__init__(_EMNA, locals())
NaiveIsoEMNA = EMNA().set_name('NaiveIsoEMNA', register=True)

@registry.register
class NGOptBase(base.Optimizer):
    """Nevergrad optimizer by competence map."""

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1):
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        analysis = p.helpers.analyze(self.parametrization)
        funcinfo = self.parametrization.function
        self.has_noise = not (analysis.deterministic and funcinfo.deterministic)
        self.has_real_noise = not funcinfo.deterministic
        self.noise_from_instrumentation = self.has_noise and funcinfo.deterministic
        self.fully_continuous = analysis.continuous
        all_params = p.helpers.flatten(self.parametrization)
        int_layers = list(itertools.chain.from_iterable([_layering.Int.filter_from(x) for _, x in all_params]))
        int_layers = [x for x in int_layers if x.arity is not None]
        self.has_discrete_not_softmax = any((not isinstance(lay, _datalayers.SoftmaxSampling) for lay in int_layers))
        self._has_discrete = bool(int_layers)
        self._arity = max((lay.arity for lay in int_layers), default=-1)
        if self.fully_continuous:
            self._arity = -1
        self._optim = None
        self._constraints_manager.update(max_trials=1000, penalty_factor=1.0, penalty_exponent=1.01)

    @property
    def optim(self):
        if self._optim is None:
            self._optim = self._select_optimizer_cls()(self.parametrization, self.budget, self.num_workers)
            self._optim = self._optim if not isinstance(self._optim, NGOptBase) else self._optim.optim
            logger.debug('%s selected %s optimizer.', *(x.name for x in (self, self._optim)))
        return self._optim

    def _select_optimizer_cls(self):
        assert self.budget is not None
        if self.has_noise and (self.has_discrete_not_softmax or not self.parametrization.function.metrizable):
            return DoubleFastGADiscreteOnePlusOne if self.dimension < 60 else CMA
        elif self.has_real_noise and self.fully_continuous:
            return TBPSA
        elif self.has_discrete_not_softmax or not self.parametrization.function.metrizable or (not self.fully_continuous):
            return DoubleFastGADiscreteOnePlusOne
        elif self.num_workers > self.budget / 5:
            if self.num_workers > self.budget / 2.0 or self.budget < self.dimension:
                return MetaTuneRecentering
            else:
                return NaiveTBPSA
        elif self.num_workers == 1 and self.budget > 6000 and (self.dimension > 7):
            return ChainNaiveTBPSACMAPowell
        elif self.num_workers == 1 and self.budget < self.dimension * 30:
            if self.dimension > 30:
                return OnePlusOne
            elif self.dimension < 5:
                return MetaModel
            else:
                return Cobyla
        elif self.dimension > 2000:
            return DE if self.dimension > 2000 else MetaCMA if self.dimension > 1 else OnePlusOne
        elif self.dimension < 10 and self.budget < 500:
            return MetaModel
        elif 3 * self.num_workers > self.dimension ** 2 and self.budget > self.dimension ** 2:
            return MetaModel
        else:
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
        out = {'sub-optim': self.optim.name}
        out.update(self.optim._info())
        return out

    def enable_pickling(self):
        self.optim.enable_pickling()

@registry.register
class NGOptDSBase(NGOptBase):
    """Nevergrad optimizer by competence map."""

    def _select_optimizer_cls(self):
        assert self.budget is not None
        if self.has_noise and (self.has_discrete_not_softmax or not self.parametrization.function.metrizable):
            return DoubleFastGADiscreteOnePlusOne if self.dimension < 60 else CMA
        elif self.has_noise and self.fully_continuous:
            if self.budget > 100:
                return ConfPortfolio(optimizers=[DiagonalCMA, PSO, GeneticDE], warmup_ratio=0.33)
            return Chaining([DiagonalCMA, NoisyOnePlusOne], ['half'])
        elif self.has_discrete_not_softmax or not self.parametrization.function.metrizable:
            return DoubleFastGADiscreteOnePlusOne
        elif self.num_workers > self.budget / 5:
            if self.num_workers > self.budget / 2.0 or self.budget < self.dimension:
                return MetaTuneRecentering
            else:
                return NaiveTBPSA
        elif self.num_workers == 1 and self.budget > 6000 and (self.dimension > 7):
            return ChainNaiveTBPSACMAPowell
        elif self.num_workers == 1 and self.budget < self.dimension * 30:
            if self.dimension > 30:
                return OnePlusOne
            elif self.dimension < 5:
                return MetaModel
            else:
                return Cobyla
        elif self.dimension > 2000:
            return DE if self.dimension > 2000 else MetaCMA if self.dimension > 1 else OnePlusOne
        elif self.dimension < 10 and self.budget < 500:
            return MetaModel
        elif 3 * self.num_workers > self.dimension ** 2 and self.budget > self.dimension ** 2:
            return MetaModel
        else:
            return CMA

@registry.register
class NGOpt4(NGOptBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self):
        assert self.budget is not None
        funcinfo = self.parametrization.function
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            mutation = 'portfolio' if self.budget > 1000 else 'discrete'
            optimClass = ParametrizedOnePlusOne(crossover=True, mutation=mutation, noise_handling='optimistic')
        elif self._arity > 0:
            if self.budget < 1000 and self.num_workers == 1:
                optimClass = DiscreteBSOOnePlusOne
            elif self.num_workers > 2:
                optimClass = CMandAS2
            else:
                optimClass = super()._select_optimizer_cls()
        elif self.has_noise and self.fully_continuous and (self.dimension > 100):
            optimClass = ConfSplitOptimizer(num_optims=13, progressive=True, multivariate_optimizer=OptimisticDiscreteOnePlusOne)
        elif self.has_noise and self.fully_continuous:
            if self.budget > 100:
                optimClass = OnePlusOne if self.noise_from_instrumentation or self.num_workers > 1 else SQP
            else:
                optimClass = OnePlusOne
        elif self.has_discrete_not_softmax or not funcinfo.metrizable or (not self.fully_continuous):
            optimClass = DoubleFastGADiscreteOnePlusOne
        elif self.num_workers > self.budget / 5:
            if self.num_workers > self.budget / 2.0 or self.budget < self.dimension:
                optimClass = MetaTuneRecentering
            elif self.dimension < 5 and self.budget < 100:
                optimClass = DiagonalCMA
            elif self.dimension < 5 and self.budget < 500:
                optimClass = Chaining([DiagonalCMA, MetaModel], [100])
            else:
                optimClass = NaiveTBPSA
        elif self.num_workers == 1 and self.budget > 6000 and (self.dimension > 7):
            optimClass = ChainNaiveTBPSACMAPowell
        elif self.num_workers == 1 and self.budget < self.dimension * 30:
            if self.dimension > 30:
                optimClass = OnePlusOne
            elif self.dimension < 5:
                optimClass = MetaModel
            else:
                optimClass = Cobyla
        elif self.dimension > 2000:
            optimClass = DE
        elif self.dimension < 10 and self.budget < 500:
            optimClass = MetaModel
        elif 3 * self.num_workers > self.dimension ** 2 and self.budget > self.dimension ** 2:
            optimClass = MetaModel
        else:
            optimClass = CMA

@registry.register
class NGOpt10(NGOpt4):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self):
        cma_vars = max(1, 4 + int(3 * np.log(self.dimension)))
        if not self.has_noise and self.fully_continuous and (self.num_workers <= cma_vars) and (self.dimension < 100) and (self.budget is not None) and (self.budget < self.dimension * 50) and (self.budget > 50):
            return MetaModel
        elif not self.has_noise and self.fully_continuous and (self.num_workers <= cma_vars) and (self.dimension < 100) and (self.budget is not None) and (self.budget < self.dimension * 5) and (self.budget > 50):
            return MetaModel
        else:
            return super()._select_optimizer_cls()

@registry.register
class NGOpt13(NGOpt4):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self):
        cma_vars = max(1, 4 + int(3 * np.log(self.dimension)))
        if self.budget is not None and self.budget > 2000 * self.dimension and (self.num_workers >= self.budget // (1000 * self.dimension)):
            optimizers = []
            optimizers += [Rescaled(base_optimizer=NGOpt4._select_optimizer_cls(self), scale=max(0.01, np.exp(-1.0 / np.random.rand())))]
            while len(optimizers) < self.num_workers:
                optimizers += [Rescaled(base_optimizer=NGOpt4._select_optimizer_cls(self), scale=max(0.01, np.exp(-1.0 / np.random.rand())))]
            return ConfPortfolio(optimizers=optimizers, warmup_ratio=1.0, no_crossing=True)
        else:
            return super()._select_optimizer_cls()

@registry.register
class NGOpt15(NGOpt4):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self):
        cma_vars = max(1, 4 + int(3 * np.log(self.dimension)))
        if self.budget is not None and self.budget > 2000 * self.dimension and (self.num_workers >= self.budget // (1000 * self.dimension)):
            optimizers = []
            optimizers += [Rescaled(base_optimizer=NGOpt4._select_optimizer_cls(self), scale=max(0.01, np.exp(-1.0 / np.random.rand())))]
            while len(optimizers) < self.num_workers:
                optimizers += [Rescaled(base_optimizer=NGOpt4._select_optimizer_cls(self), scale=max(0.01, np.exp(-1.0 / np.random.rand())))]
            return ConfPortfolio(optimizers=optimizers, warmup_ratio=0.7, no_crossing=True)
        else:
            return super()._select_optimizer_cls()

@registry.register
class NGOpt21(NGOpt4):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self):
        cma_vars = max(1, 4 + int(3 * np.log(self.dimension)))
        if self.budget is not None and self.budget > 2000 * self.dimension and (self.num_workers >= self.budget // (1000 * self.dimension)):
            optimizers = []
            optimizers += [Rescaled(base_optimizer=NGOpt4._select_optimizer_cls(self), scale=max(0.01, np.exp(-1.0 / np.random.rand())))]
            while len(optimizers) < self.num_workers:
                optimizers += [Rescaled(base_optimizer=NGOpt4._select_optimizer_cls(self), scale=max(0.01, np.exp(-1.0 / np.random.rand())))]
            return ConfPortfolio(optimizers=optimizers, warmup_ratio=1.0, no_crossing=True)
        else:
            return super()._select_optimizer_cls()

@registry.register
class NGOpt36(NGOpt4):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self):
        cma_vars = max(1, 4 + int(3 * np.log(self.dimension)))
        if self.budget is not None and self.budget > 2000 * self.dimension and (self.num_workers >= self.budget // (1000 * self.dimension)):
            optimizers = []
            optimizers += [Rescaled(base_optimizer=NGOpt4._select_optimizer_cls(self), scale=max(0.01, np.exp(-1.0 / np.random.rand())))]
            while len(optimizers) < self.num_workers:
                optimizers += [Rescaled(base_optimizer=NGOpt4._select_optimizer_cls(self), scale=max(0.01, np.exp(-1.0 / np.random.rand())))]
            return ConfPortfolio(optimizers=optimizers, warmup_ratio=0.7, no_crossing=True)
        else:
            return super()._select_optimizer_cls()

@registry.register
class NGOpt38(NGOpt4):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self):
        cma_vars = max(1, 4 + int(3 * np.log(self.dimension)))
        if self.budget is not None and self.budget > 5000 * self.dimension:
            num = self.budget // (1000 * self.dimension)
            if self.num_workers >= num:
                optimizers = []
                optimizers += [Rescaled(base_optimizer=NGOpt4._select_optimizer_cls(self), scale=max(0.01, np.exp(-1.0 / np.random.rand())))]
                while len(optimizers) < num:
                    optimizers += [Rescaled(base_optimizer=NGOpt4._select_optimizer_cls(self), scale=max(0.01, np.exp(-1.0 / np.random.rand())))]
                return ConfPortfolio(optimizers=optimizers, warmup_ratio=0.7, no_crossing=True)
        else:
            return super()._select_optimizer_cls()

@registry.register
class NGOptF(NGOpt4):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self):
        best = defaultdict(lambda: defaultdict(list))

        def recommend_method(d, nod):
            best[12][833.333] += ['MemeticDE']
            best[12][83.3333] += ['MemeticDE']
            best[12][8.33333] += ['RealSpacePSO']
            best[12][0.833333] += ['CauchyOnePlusOne']
            best[12][833.333] += ['VLPCMA']
            best[12][83.3333] += ['NLOPT_LN_SBPLX']
            best[12][8.33333] += ['NLOPT_LN_SBPLX']
            best[12][0.833333] += ['NLOPT_LN_SBPLX']
            best[12][833.333] += ['VLPCMA']
            best[12][83.3333] += ['MemeticDE']
            best[12][8.33333] += ['SMAC3']
            best[12][0.833333] += ['Cobyla']
            best[24][416.667] += ['VLPCMA']
            best[24][41.6667] += ['Wiz']
            best[24][4.16667] += ['NLOPT_LN_SBPLX']
            best[24][0.416667] += ['Cobyla']
            best[24][416.667] += ['NLOPT_LN_SBPLX']
            best[24][41.6667] += ['Wiz']
            best[24][4.16667] += ['NLOPT_LN_SBPLX']
            best[24][0.416667] += ['NLOPT_LN_SBPLX']
            best[24][416.667] += ['ChainDiagonalCMAPowell']
            best[24][41.6667] += ['NLOPT_LN_SBPLX']
            best[24][4.16667] += ['QORealSpacePSO']
            best[24][0.416667] += ['Cobyla']
            best[2][5000] += ['NGOpt16']
            best[2][500] += ['LhsDE']
            best[2][50] += ['SODE']
            best[2][5] += ['Carola2']
            best[2][5000] += ['MetaModelQODE']
            best[2][500] += ['MetaModelQODE']
            best[2][50] += ['PCABO']
            best[2][5] += ['HammersleySearchPlusMiddlePoint']
            best[2][5000] += ['QORealSpacePSO']
            best[2][500] += ['ChainDiagonalCMAPowell']
            best[2][50] += ['MetaModelQODE']
            best[2][5] += ['Cobyla']
            best[5][2000] += ['MemeticDE']
            best[5][200] += ['MemeticDE']
            best[5][20] += ['LhsDE']
            best[5][2] += ['MultiSQP']
            best[5][2000] += ['MemeticDE']
            best[5][200] += ['LhsDE']
            best[5][20] += ['LhsDE']
            best[5][2] += ['NLOPT_LN_SBPLX']
            best[5][2000] += ['LhsDE']
            best[5][200] += ['VLPCMA']
            best[5][20] += ['BayesOptimBO']
            best[5][2] += ['NLOPT_LN_SBPLX']
            best[96][0.104167] += ['PCABO']
            bestdist = float('inf')
            for d1 in best:
                for nod2 in best[d1]:
                    dist = (d - d1) ** 2 + (nod - nod2) ** 2
                    if dist < bestdist:
                        bestdist = dist
                        bestalg = best[d1][nod2]
            return bestalg
        if self.fully_continuous and (not self.has_noise):
            algs = recommend_method(self.dimension, self.budget / self.dimension)
            if self.num_workers > 1:
                algs = [a for a in algs if not registry[a].no_parallelization]
                if len(algs) == 0:
                    return SQPCMA
            return registry[most_frequent(algs)]
        if self.fully_continuous and (not self.has_noise) and (self.budget >= 12 * self.dimension):
            return ConfPortfolio(optimizers=[GeneticDE, PSO, NGOpt4], warmup_ratio=0.33)
        else:
            return super()._select_optimizer_cls()

@registry.register
class NGOptF2(NGOpt4):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self):
        best = defaultdict(lambda: defaultdict(list))

        def recommend_method(d, nod):
            best[12][833.333] += ['MemeticDE']
            best[12][83.3333] += ['NGOptRW']
            best[12][8.33333] += ['RealSpacePSO']
            best[12][0.833333] += ['ASCMADEthird']
            best[12][8333.33] += ['GeneticDE']
            best[12][833.333] += ['TripleCMA']
            best[12][83.3333] += ['NLOPT_LN_SBPLX']
            best[12][8.33333] += ['NLOPT_LN_SBPLX']
            best[12][0.833333] += ['NLOPT_LN_SBPLX']
            best[12][8333.33] += ['GeneticDE']
            best[12][833.333] += ['VLPCMA']
            best[12][83.3333] += ['MemeticDE']
            best[12][8.33333] += ['SMAC3']
            best[12][0.833333] += ['Cobyla']
            best[24][416.667] += ['VLPCMA']
            best[24][41.6667] += ['Wiz']
            best[24][4.16667] += ['NLOPT_LN_SBPLX']
            best[24][0.416667] += ['Cobyla']
            best[24][4166.67] += ['NGOptF']
            best[24][416.667] += ['NLOPT_LN_SBPLX']
            best[24][41.6667] += ['ChainNaiveTBPSAPowell']
            best[24][4.16667] += ['NLOPT_LN_SBPLX']
            best[24][0.416667] += ['NLOPT_LN_NELDERMEAD']
            best[24][4166.67] += ['Carola1']
            best[24][416.667] += ['ChainDiagonalCMAPowell']
            best[24][41.6667] += ['NLOPT_LN_SBPLX']
            best[24][4.16667] += ['NLOPT_GN_CRS2_LM']
            best[24][0.416667] += ['Cobyla']
            best[2][500000] += ['BAR']
            best[2][50000] += ['ASCMADEthird']
            best[2][5000] += ['ChainMetaModelPowell']
            best[2][500] += ['DiscreteDE']
            best[2][50] += ['RFMetaModelOnePlusOne']
            best[2][5] += ['Carola2']
            best[2][500000] += ['BAR3']
            best[2][50000] += ['ChainCMAPowell']
            best[2][5000] += ['ASCMADEthird']
            best[2][500] += ['NeuralMetaModelTwoPointsDE']
            best[2][50] += ['CMandAS2']
            best[2][5] += ['NaiveTBPSA']
            best[2][500000] += ['BAR']
            best[2][50000] += ['ASCMADEthird']
            best[2][5000] += ['CM']
            best[2][500] += ['ChainDiagonalCMAPowell']
            best[2][50] += ['Powell']
            best[2][5] += ['Cobyla']
            best[48][20.8333] += ['RLSOnePlusOne']
            best[48][2.08333] += ['NGOptF']
            best[48][0.208333] += ['NGOptF2']
            best[48][208.333] += ['MetaModelQODE']
            best[48][20.8333] += ['DiscreteLengler2OnePlusOne']
            best[48][2.08333] += ['NGOptF']
            best[48][0.208333] += ['NGOptF2']
            best[48][2083.33] += ['NGOptF2']
            best[48][208.333] += ['ChainNaiveTBPSACMAPowell']
            best[48][20.8333] += ['ChainNaiveTBPSACMAPowell']
            best[48][2.08333] += ['NLOPT_LN_NELDERMEAD']
            best[48][0.208333] += ['BOBYQA']
            best[5][200000] += ['RescaledCMA']
            best[5][20000] += ['RFMetaModelDE']
            best[5][2000] += ['DiscreteDE']
            best[5][200] += ['TwoPointsDE']
            best[5][20] += ['NGOpt39']
            best[5][2] += ['OnePlusLambda']
            best[5][200000] += ['BAR']
            best[5][20000] += ['ChainNaiveTBPSACMAPowell']
            best[5][2000] += ['CmaFmin2']
            best[5][200] += ['RotatedTwoPointsDE']
            best[5][20] += ['pysot']
            best[5][2] += ['NLOPT_LN_SBPLX']
            best[5][200000] += ['ASCMADEthird']
            best[5][20000] += ['ASCMADEthird']
            best[5][2000] += ['QOTPDE']
            best[5][200] += ['NGOpt10']
            best[5][20] += ['LQODE']
            best[5][2] += ['NLOPT_LN_SBPLX']
            best[96][104.167] += ['NGOptF']
            best[96][10.4167] += ['NGOpt4']
            best[96][1.04167] += ['RPowell']
            best[96][0.104167] += ['ASCMADEthird']
            best[96][1041.67] += ['NLOPT_LN_NELDERMEAD']
            best[96][104.167] += ['RPowell']
            best[96][10.4167] += ['NGOpt8']
            best[96][1.04167] += ['RPowell']
            best[96][0.104167] += ['NLOPT_LN_NELDERMEAD']
            best[96][1041.67] += ['CMandAS3']
            best[96][104.167] += ['Powell']
            best[96][10.4167] += ['NGOptBase']
            best[96][1.04167] += ['Powell']
            best[96][0.104167] += ['NLOPT_LN_NELDERMEAD']
            bestdist = float('inf')
            for d1 in best:
                for nod2 in best[d1]:
                    dist = (d - d1) ** 2 + (nod - nod2) ** 2
                    if dist < bestdist:
                        bestdist = dist
                        bestalg = best[d1][nod2]
            return bestalg
        if self.fully_continuous and (not self.has_noise):
            algs = recommend_method(self.dimension, self.budget / self.dimension)
            if self.num_workers > 1:
                algs = [a for a in algs if not registry[a].no_parallelization]
                if len(algs) == 0:
                    return SQPCMA

            def most_frequent(List):
                return max(set(List), key=List.count)
            return registry[most_frequent(algs)]
        if self.fully_continuous and (not self.has_noise) and (self.budget >= 12 * self.dimension):
            return ConfPortfolio(optimizers=[GeneticDE, PSO, NGOpt4], warmup_ratio=0.33)
        else:
            return super()._select_optimizer_cls()

@registry.register
class NGOptF3(NGOpt4):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self):
        best = defaultdict(lambda: defaultdict(list))

        def recommend_method(d, nod):
            best[12][8333.33] += ['DiscreteDE']
            best[12][833.333] += ['DiscreteDE']
            best[12][83.3333] += ['MetaModelDE']
            best[12][8.33333] += ['NGOpt']
            best[12][0.833333] += ['NLOPT_GN_DIRECT_L']
            best[12][83333.3] += ['RPowell']
            best[12][8333.33] += ['DiscreteDE']
            best[12][833.333] += ['NLOPT_GN_ISRES']
            best[12][83.3333] += ['SMAC3']
            best[12][8.33333] += ['NGOpt']
            best[12][0.833333] += ['NLOPT_LN_SBPLX']
            best[12][83333.3] += ['NGOptBase']
            best[12][8333.33] += ['RPowell']
            best[12][833.333] += ['ChainMetaModelPowell']
            best[12][83.3333] += ['RotatedTwoPointsDE']
            best[12][8.33333] += ['NGOpt39']
            best[12][0.833333] += ['NLOPT_LN_NEWUOA_BOUND']
            best[24][4166.67] += ['CmaFmin2']
            best[24][416.667] += ['CmaFmin2']
            best[24][41.6667] += ['CmaFmin2']
            best[24][4.16667] += ['NLOPT_LN_SBPLX']
            best[24][0.416667] += ['Cobyla']
            best[24][41666.7] += ['MultiCMA']
            best[24][4166.67] += ['CmaFmin2']
            best[24][416.667] += ['RotatedTwoPointsDE']
            best[24][41.6667] += ['CmaFmin2']
            best[24][4.16667] += ['pysot']
            best[24][0.416667] += ['NLOPT_LN_NELDERMEAD']
            best[24][41666.7] += ['MultiCMA']
            best[24][4166.67] += ['Shiwa']
            best[24][416.667] += ['CmaFmin2']
            best[24][41.6667] += ['NLOPT_GN_CRS2_LM']
            best[24][4.16667] += ['NLOPT_GN_CRS2_LM']
            best[24][0.416667] += ['Cobyla']
            best[2][500000] += ['BAR']
            best[2][50000] += ['ASCMADEthird']
            best[2][5000] += ['ChainMetaModelPowell']
            best[2][500] += ['DiscreteDE']
            best[2][50] += ['RFMetaModelOnePlusOne']
            best[2][5] += ['Carola2']
            best[2][500000] += ['BAR3']
            best[2][50000] += ['ChainCMAPowell']
            best[2][5000] += ['ASCMADEthird']
            best[2][500] += ['NeuralMetaModelTwoPointsDE']
            best[2][50] += ['CMandAS2']
            best[2][5] += ['NaiveTBPSA']
            best[2][500000] += ['BAR']
            best[2][50000] += ['ASCMADEthird']
            best[2][5000] += ['CM']
            best[2][500] += ['ChainDiagonalCMAPowell']
            best[2][50] += ['Powell']
            best[2][5] += ['Cobyla']
            best[48][208.333] += ['DiscreteDE']
            best[48][20.8333] += ['pysot']
            best[48][2.08333] += ['MultiCobyla']
            best[48][0.208333] += ['NLOPT_LN_NELDERMEAD']
            best[48][2083.33] += ['RPowell']
            best[48][208.333] += ['CmaFmin2']
            best[48][20.8333] += ['RPowell']
            best[48][2.08333] += ['pysot']
            best[48][0.208333] += ['BOBYQA']
            best[48][20833.3] += ['MetaModelTwoPointsDE']
            best[48][208.333] += ['DiscreteDE']
            best[48][20.8333] += ['ChainNaiveTBPSACMAPowell']
            best[48][2.08333] += ['NLOPT_LN_BOBYQA']
            best[48][0.208333] += ['BOBYQA']
            best[5][200000] += ['RescaledCMA']
            best[5][20000] += ['RFMetaModelDE']
            best[5][2000] += ['DiscreteDE']
            best[5][200] += ['TwoPointsDE']
            best[5][20] += ['NGOpt39']
            best[5][2] += ['OnePlusLambda']
            best[5][200000] += ['BAR']
            best[5][20000] += ['ChainNaiveTBPSACMAPowell']
            best[5][2000] += ['CmaFmin2']
            best[5][200] += ['RotatedTwoPointsDE']
            best[5][20] += ['pysot']
            best[5][2] += ['NLOPT_LN_SBPLX']
            best[5][200000] += ['ASCMADEthird']
            best[5][20000] += ['ASCMADEthird']
            best[5][2000] += ['QOTPDE']
            best[5][200] += ['NGOpt10']
            best[5][20] += ['LQODE']
            best[5][2] += ['NLOPT_LN_SBPLX']
            best[96][104.167] += ['NGOptF']
            best[96][10.4167] += ['NGOpt4']
            best[96][1.04167] += ['RPowell']
            best[96][0.104167] += ['ASCMADEthird']
            best[96][1041.67] += ['NLOPT_LN_NELDERMEAD']
            best[96][104.167] += ['RPowell']
            best[96][10.4167] += ['NGOpt8']
            best[96][1.04167] += ['RPowell']
            best[96][0.104167] += ['NLOPT_LN_NELDERMEAD']
            best[96][1041.67] += ['CMandAS3']
            best[96][104.167] += ['Powell']
            best[96][10.4167] += ['NGOptBase']
            best[96][1.04167] += ['Powell']
            best[96][0.104167] += ['NLOPT_LN_NELDERMEAD']
            bestdist = float('inf')
            for d1 in best:
                for nod2 in best[d1]:
                    dist = (d - d1) ** 2 + (nod - nod2) ** 2
                    if dist < bestdist:
                        bestdist = dist
                        bestalg = best[d1][nod2]
            return bestalg
        if self.fully_continuous and (not self.has_noise):
            algs = recommend_method(self.dimension, self.budget / self.dimension)
            if self.num_workers > 1:
                algs = [a for a in algs if not registry[a].no_parallelization]
                if len(algs) == 0:
                    return SQPCMA

            def most_frequent(List):
                return max(set(List), key=List.count)
            return registry[most_frequent(algs)]
        if self.fully_continuous and (not self.has_noise) and (self.budget >= 12 * self.dimension):
            return ConfPortfolio(optimizers=[GeneticDE, PSO, NGOpt4], warmup_ratio=0.33)
        else:
            return super()._select_optimizer_cls()

@registry.register
class NGOptF5(NGOpt4):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self):
        best = defaultdict(lambda: defaultdict(list))

        def recommend_method(d, nod):
            best[12][8333.33] += ['DiscreteDE']
            best[12][833.333] += ['DiscreteDE']
            best[12][83.3333] += ['MetaModelDE']
            best[12][8.33333] += ['NGOpt']
            best[12][0.833333] += ['NLOPT_GN_DIRECT_L']
            best[12][83333.3] += ['NGOptBase']
            best[12][8333.33] += ['DiscreteDE']
            best[12][833.333] += ['NLOPT_GN_ISRES']
            best[12][83.3333] += ['SMAC3']
            best[12][8.33333] += ['NGOpt']
            best[12][0.833333] += ['NLOPT_LN_SBPLX']
            best[12][83333.3] += ['NGOptBase']
            best[12][8333.33] += ['RPowell']
            best[12][833.333] += ['ChainMetaModelPowell']
            best[12][83.3333] += ['RotatedTwoPointsDE']
            best[12][8.33333] += ['NGOpt39']
            best[12][0.833333] += ['NLOPT_LN_NEWUOA_BOUND']
            best[24][4166.67] += ['CmaFmin2']
            best[24][416.667] += ['CmaFmin2']
            best[24][41.6667] += ['CmaFmin2']
            best[24][4.16667] += ['NLOPT_LN_SBPLX']
            best[24][0.416667] += ['Cobyla']
            best[24][41666.7] += ['MultiCMA']
            best[24][4166.67] += ['CmaFmin2']
            best[24][416.667] += ['RotatedTwoPointsDE']
            best[24][41.6667] += ['CmaFmin2']
            best[24][4.16667] += ['pysot']
            best[24][0.416667] += ['NLOPT_LN_NELDERMEAD']
            best[24][41666.7] += ['MultiCMA']
            best[24][4166.67] += ['Shiwa']
            best[24][416.667] += ['CmaFmin2']
            best[24][41.6667] += ['NLOPT_GN_CRS2_LM']
            best[24][4.16667] += ['NLOPT_GN_CRS2_LM']
            best[24][0.416667] += ['Cobyla']
            best[2][500000] += ['BAR']
            best[2][50000] += ['ASCMADEthird']
            best[2][5000] += ['ChainMetaModelPowell']
            best[2][500] += ['DiscreteDE']
            best[2][50] += ['RFMetaModelOnePlusOne']
            best[2][5] += ['Carola2']
            best[2][500000] += ['BAR3']
            best[2][50000] += ['ChainCMAPowell']
            best[2][5000] += ['ASCMADEthird']
            best[2][500] += ['NeuralMetaModelTwoPointsDE']
            best[2][50] += ['CMandAS2']
            best[2][5] += ['NaiveTBPSA']
            best[2][500000] += ['BAR']
            best[2][50000] += ['ASCMADEthird']
            best[2][5000] += ['CM']
            best[2][500] += ['ChainDiagonalCMAPowell']
            best[2][50] += ['Powell']
            best[2][5] += ['Cobyla']
            best[48][208.333] += ['DiscreteDE']
            best[48][20.8333] += ['pysot']
            best[48][2.08333] += ['MultiCobyla']
            best[48][0.208333] += ['NLOPT_LN_NELDERMEAD']
            best[48][2083.33] += ['RPowell']
            best[48][208.333] += ['CmaFmin2']
            best[48][20.8333] += ['RPowell']
            best[48][2.08333] += ['pysot']
            best[48][0.208333] += ['BOBYQA']
            best[48][20833.3] += ['MetaModelTwoPointsDE']
            best[48][208.333] += ['DiscreteDE']
            best[48][20.8333] += ['ChainNaiveTBPSACMAPowell']
            best[48][2.08333] += ['NLOPT_LN_BOBYQA']
            best[48][0.208333] += ['BOBYQA']
            best[5][200000] += ['RescaledCMA']
            best[5][20000] += ['RFMetaModelDE']
            best[5][2000] += ['DiscreteDE']
            best[5][200] += ['TwoPointsDE']
            best[5][20] += ['NGOpt39']
            best[5][2] += ['OnePlusLambda']
            best[5][200000] += ['BAR']
            best[5][20000] += ['ChainNaiveTBPSACMAPowell']
            best[5][2000] += ['CmaFmin2']
            best[5][200] += ['RotatedTwoPointsDE']
            best[5][20] += ['pysot']
            best[5][2] += ['NLOPT_LN_SBPLX']
            best[5][200000] += ['ASCMADEthird']
            best[5][20000] += ['ASCMADEthird']
            best[5][2000] += ['QOTPDE']
            best[5][200] += ['NGOpt10']
            best[5][20] += ['LQODE']
            best[5][2] += ['NLOPT_LN_SBPLX']
            best[96][104.167] += ['NGOptF']
            best[96][10.4167] += ['NGOpt4']
            best[96][1.04167] += ['RPowell']
            best[96][0.104167] += ['ASCMADEthird']
            best[96][1041.67] += ['NLOPT_LN_NELDERMEAD']
            best[96][104.167] += ['RPowell']
            best[96][10.4167] += ['NGOpt8']
            best[96][1.04167] += ['RPowell']
            best[96][0.104167] += ['NLOPT_LN_NELDERMEAD']
            best[96][1041.67] += ['CMandAS3']
            best[96][104.167] += ['Powell']
            best[96][10.4167] += ['NGOptBase']
            best[96][1.04167] += ['Powell']
            best[96][0.104167] += ['NLOPT_LN_NELDERMEAD']
            bestdist = float('inf')
            for d1 in best:
                for nod2 in best[d1]:
                    dist = (d - d1) ** 2 + (nod - nod2) ** 2
                    if dist < bestdist:
                        bestdist = dist
                        bestalg = best[d1][nod2]
            return bestalg
        if self.fully_continuous and (not self.has_noise):
            algs = recommend_method(self.dimension, self.budget / self.dimension)
            if self.num_workers > 1:
                algs = [a for a in algs if not registry[a].no_parallelization]
                if len(algs) == 0:
                    return SQPCMA

            def most_frequent(List):
                return max(set(List), key=List.count)
            return registry[most_frequent(algs)]
        if self.fully_continuous and (not self.has_noise) and (self.budget >= 12 * self.dimension):
            return ConfPortfolio(optimizers=[GeneticDE, PSO, NGOpt4], warmup_ratio=0.33)
        else:
            return super()._select_optimizer_cls()

@registry.register
class NGOpt(NGOpt4):
    pass

@registry.register
class Wiz(NGOpt16):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self):
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (self.budget < 1000 * self.dimension) and (self.budget > 20 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 100):
            return Carola2
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (self.budget >= 1000 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 50):
            return Carola2
        if self.has_noise and (self.has_discrete_not_softmax or not self.parametrization.function.metrizable):
            return RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        elif self.dimension >= 60 and (not self.parametrization.function.metrizable):
            return CMA
        return super()._select_optimizer_cls()

@registry.register
class NgIoh(NGOptBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self):
        optCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and (not self.fully_continuous) and (not self.has_noise):
            return ConfPortfolio(optimizers=[SuperSmoothDiscreteLenglerOnePlusOne, SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne, DiscreteLenglerOnePlusOne], warmup_ratio=0.4)
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (self.budget < 1000 * self.dimension) and (self.budget > 20 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 100):
            return Carola2
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (self.budget >= 1000 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 50):
            return Carola2
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        elif self.dimension >= 60 and (not funcinfo.metrizable):
            optCls = CMA
        return optCls

@registry.register
class NgIoh2(NGOptBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self):
        optCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and (not self.fully_continuous) and (not self.has_noise):
            return ConfPortfolio(optimizers=[SuperSmoothDiscreteLenglerOnePlusOne, SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne, DiscreteLenglerOnePlusOne], warmup_ratio=0.4)
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (self.budget < 1000 * self.dimension) and (self.budget > 20 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 100):
            return Carola2
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (self.budget >= 1000 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 50):
            return Carola2
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        elif self.dimension >= 60 and (not funcinfo.metrizable):
            optCls = CMA
        return optCls

@registry.register
class NgIoh3(NGOptBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self):
        optCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and (not self.fully_continuous) and (not self.has_noise):
            return ConfPortfolio(optimizers=[SuperSmoothDiscreteLenglerOnePlusOne, SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne, DiscreteLenglerOnePlusOne], warmup_ratio=0.4)
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (self.budget < 1000 * self.dimension) and (self.budget > 20 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 100):
            return Carola2
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (self.budget >= 1000 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 50):
            return Carola2
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        elif self.dimension >= 60 and (not funcinfo.metrizable):
            optCls = CMA
        return optCls

@registry.register
class NgIoh4(NGOptBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self):
        optCls = NGOptBase
        funcinfo = self.parametrization.function
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (self.budget < 1000 * self.dimension) and (self.budget > 20 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 100):
            return Carola2
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (self.budget >= 1000 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 50):
            return Carola2
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        elif self.dimension >= 60 and (not funcinfo.metrizable):
            optCls = CMA
        return optCls

@registry.register
class NgIoh5(NGOptBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self):
        optCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and (not self.fully_continuous) and (not self.has_noise):
            return ConfPortfolio(optimizers=[SuperSmoothDiscreteLenglerOnePlusOne, SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne, DiscreteLenglerOnePlusOne], warmup_ratio=0.4)
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (self.budget < 1000 * self.dimension) and (self.budget > 20 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 100):
            return Carola2
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (self.budget >= 1000 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 50):
            return Carola2
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        elif self.dimension >= 60 and (not funcinfo.metrizable):
            optCls = CMA
        return optCls

@registry.register
class NgIoh6(NGOptBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self):
        optCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and (not self.fully_continuous) and (not self.has_noise):
            return ConfPortfolio(optimizers=[SuperSmoothDiscreteLenglerOnePlusOne, SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne, DiscreteLenglerOnePlusOne], warmup_ratio=0.4)
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (self.budget < 1000 * self.dimension) and (self.budget > 20 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 100):
            return Carola2
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (self.budget >= 1000 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 50):
            return Carola2
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        elif self.dimension >= 60 and (not funcinfo.metrizable):
            optCls = CMA
        return optCls

class _MSR(Portfolio):
    """This code applies multiple copies of NGOpt with random weights for the different objective functions.

    Variants dedicated to multiobjective optimization by multiple singleobjective optimization.
    """

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1, num_single_runs: int = 9, base_optimizer: Optional[NGOpt] = None):
        super().__init__(parametrization, budget=budget, num_workers=num_workers, config=ConfPortfolio(optimizers=[base_optimizer] * num_single_runs))
        self.coeffs = []

    def _internal_tell_candidate(self, candidate, loss):
        if not self.coeffs:
            self.coeffs = [self.parametrization.random_state.uniform(size=self.num_objectives) for _ in self.optims]
        for coeffs, opt in zip(self.coeffs, self.optims):
            this_loss = np.sum(loss * coeffs)
            opt.tell(candidate, this_loss)

class MultipleSingleRuns(base.ConfiguredOptimizer):
    """Multiple single-objective runs, in particular for multi-objective optimization.
    Parameters
    ----------
    num_single_runs: int
        number of single runs.
    """

    def __init__(self, *, num_single_runs: int = 9, base_optimizer: Optional[NGOpt] = None):
        super().__init__(_MSR, locals())
SmoothDiscreteOnePlusOne = ParametrizedOnePlusOne(smoother=True, mutation='discrete').set_name('SmoothDiscreteOnePlusOne', register=True)
SmoothPortfolioDiscreteOnePlusOne = ParametrizedOnePlusOne(smoother=True, mutation='portfolio').set_name('SmoothPortfolioDiscreteOnePlusOne', register=True)
SmoothDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(smoother=True, mutation='lengler').set_name('SmoothDiscreteLenglerOnePlusOne', register=True)
SmoothDiscreteLognormalOnePlusOne = ParametrizedOnePlusOne(smoother=True, mutation='lognormal').set_name('SmoothDiscreteLognormalOnePlusOne', register=True)
SuperSmoothDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(smoother=True, mutation='lengler', antismooth=9).set_name('SuperSmoothDiscreteLenglerOnePlusOne', register=True)
SuperSmoothTinyLognormalDiscreteOnePlusOne = ParametrizedOnePlusOne(smoother=True, mutation='tinylognormal', antismooth=9).set_name('SuperSmoothTinyLognormalDiscreteOnePlusOne', register=True)
UltraSmoothDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(smoother=True, mutation='lengler', antismooth=3).set_name('UltraSmoothDiscreteLenglerOnePlusOne', register=True)
SmootherDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(smoother=True, mutation='lengler', antismooth=2, super_radii=True).set_name('SmootherDiscreteLenglerOnePlusOne', register=True)
YoSmoothDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(smoother=True, mutation='lengler', antismooth=2).set_name('YoSmoothDiscreteLenglerOnePlusOne', register=True)
CMALS = Chaining([CMA, DiscreteLenglerOnePlusOne, UltraSmoothDiscreteLenglerOnePlusOne], ['third', 'third']).set_name('CMALS', register=True)
UltraSmoothDiscreteLognormalOnePlusOne = ParametrizedOnePlusOne(smoother=True, mutation='lognormal', antismooth=3).set_name('UltraSmoothDiscreteLognormalOnePlusOne', register=True)
CMALYS = Chaining([CMA, YoSmoothDiscreteLenglerOnePlusOne], ['tenth']).set_name('CMALYS', register=True)
CLengler = Chaining([CMA, DiscreteLenglerOnePlusOne], ['tenth']).set_name('CLengler', register=True)
CMALL = Chaining([CMA, DiscreteLenglerOnePlusOne, UltraSmoothDiscreteLognormalOnePlusOne], ['third', 'third']).set_name('CMALL', register=True)
CMAILL = Chaining([ImageMetaModel, ImageMetaModelLengler, UltraSmoothDiscreteLognormalOnePlusOne], ['third', 'third']).set_name('CMAILL', register=True)
CMASL = Chaining([CMA, SmootherDiscreteLenglerOnePlusOne], ['tenth']).set_name('CMASL', register=True)
CMASL2 = Chaining([CMA, SmootherDiscreteLenglerOnePlusOne], ['third']).set_name('CMASL2', register=True)
CMASL3 = Chaining([CMA, SmootherDiscreteLenglerOnePlusOne], ['half']).set_name('CMASL3', register=True)
CMAL2 = Chaining([CMA, SmootherDiscreteLenglerOnePlusOne], ['half']).set_name('CMAL2', register=True)
CMAL3 = Chaining([DiagonalCMA, SmootherDiscreteLenglerOnePlusOne], ['half']).set_name('CMAL3', register=True)
SmoothLognormalDiscreteOnePlusOne = ParametrizedOnePlusOne(smoother=True, mutation='lognormal').set_name('SmoothLognormalDiscreteOnePlusOne', register=True)
SmoothAdaptiveDiscreteOnePlusOne = ParametrizedOnePlusOne(smoother=True, mutation='adaptive').set_name('SmoothAdaptiveDiscreteOnePlusOne', register=True)
SmoothRecombiningPortfolioDiscreteOnePlusOne = ParametrizedOnePlusOne(smoother=True, crossover=True, mutation='portfolio').set_name('SmoothRecombiningPortfolioDiscreteOnePlusOne', register=True)
SmoothRecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(smoother=True, crossover=True, mutation='lengler').set_name('SmoothRecombiningDiscreteLenglerOnePlusOne', register=True)
UltraSmoothRecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(smoother=True, crossover=True, mutation='lengler', antismooth=3).set_name('UltraSmoothRecombiningDiscreteLenglerOnePlusOne', register=True)
UltraSmoothElitistRecombiningDiscreteLognormalOnePlusOne = ParametrizedOnePlusOne(smoother=True, crossover=True, mutation='lognormal', antismooth=3, roulette_size=7).set_name('UltraSmoothElitistRecombiningDiscreteLognormalOnePlusOne', register=True)
UltraSmoothElitistRecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(smoother=True, crossover=True, mutation='lengler', antismooth=3, roulette_size=7).set_name('UltraSmoothElitistRecombiningDiscreteLenglerOnePlusOne', register=True)
SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(smoother=True, crossover=True, mutation='lengler', antismooth=9, roulette_size=7).set_name('SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne', register=True)
SuperSmoothRecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(smoother=True, crossover=True, mutation='lengler', antismooth=9).set_name('SuperSmoothRecombiningDiscreteLenglerOnePlusOne', register=True)
SuperSmoothRecombiningDiscreteLognormalOnePlusOne = ParametrizedOnePlusOne(smoother=True, crossover=True, mutation='lognormal', antismooth=9).set_name('SuperSmoothRecombiningDiscreteLognormalOnePlusOne', register=True)
SmoothElitistRecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(smoother=True, crossover=True, mutation='lengler', roulette_size=7).set_name('SmoothElitistRecombiningDiscreteLenglerOnePlusOne', register=True)
SmoothElitistRandRecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(smoother=True, crossover=True, mutation='lengler', roulette_size=7, crossover_type='rand').set_name('SmoothElitistRandRecombiningDiscreteLenglerOnePlusOne', register=True)
SmoothElitistRandRecombiningDiscreteLognormalOnePlusOne = ParametrizedOnePlusOne(smoother=True, crossover=True, mutation='lognormal', roulette_size=7, crossover_type='rand').set_name('SmoothElitistRandRecombiningDiscreteLognormalOnePlusOne', register=True)
RecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(crossover=True, mutation='lengler').set_name('RecombiningDiscreteLenglerOnePlusOne', register=True)
RecombiningDiscreteLognormalOnePlusOne = ParametrizedOnePlusOne(crossover=True, mutation='lognormal').set_name('RecombiningDiscreteLognormalOnePlusOne', register=True)
MaxRecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(crossover=True, mutation='lengler', crossover_type='max').set_name('MaxRecombiningDiscreteLenglerOnePlusOne', register=True)
MinRecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(crossover=True, mutation='lengler', crossover_type='min').set_name('MinRecombiningDiscreteLenglerOnePlusOne', register=True)
OnePtRecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(crossover=True, mutation='lengler', crossover_type='onepoint').set_name('OnePtRecombiningDiscreteLenglerOnePlusOne', register=True)
TwoPtRecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(crossover=True, mutation='lengler', crossover_type='twopoint').set_name('TwoPtRecombiningDiscreteLenglerOnePlusOne', register=True)
RandRecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(crossover=True, mutation='lengler', crossover_type='rand').set_name('RandRecombiningDiscreteLenglerOnePlusOne', register=True)
RandRecombiningDiscreteLognormalOnePlusOne = ParametrizedOnePlusOne(crossover=True, mutation='lognormal', crossover_type='rand').set_name('RandRecombiningDiscreteLognormalOnePlusOne', register=True)

@registry.register
class NgIoh7(NGOptBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self):
        optCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and (not self.fully_continuous) and (not self.has_noise):
            return ConfPortfolio(optimizers=[SuperSmoothDiscreteLenglerOnePlusOne, SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne, DiscreteLenglerOnePlusOne], warmup_ratio=0.4)
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (self.budget < 1000 * self.dimension) and (self.budget > 20 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 100):
            return Carola2
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (self.budget >= 1000 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 50):
            return Carola2
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        elif self.dimension >= 60 and (not funcinfo.metrizable):
            optCls = CMA
        return optCls

@registry.register
class NgDS11(NGOptDSBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self, budget: Optional[int] = None):
        if budget is None:
            budget = self.budget
        else:
            self.budget = budget
        optCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and (not self.fully_continuous) and (not self.has_noise):
            return ConfPortfolio(optimizers=[SuperSmoothDiscreteLenglerOnePlusOne, SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne, DiscreteLenglerOnePlusOne], warmup_ratio=0.4)
        if self.fully_continuous and self.num_workers == 1 and (budget is not None) and (not self.has_noise):
            if 300 * self.dimension < budget < 3000 * self.dimension:
                if self.dimension == 2:
                    return DS14
                if self.dimension < 4:
                    return DS4
                if self.dimension < 8:
                    return DS5
                if self.dimension < 15:
                    return DS9
                if self.dimension < 30:
                    return DS8
                if self.dimension < 60:
                    return DS9
            if 300 * self.dimension < budget < 3000 * self.dimension and self.dimension == 2:
                return DS6
            if 300 * self.dimension < budget < 3000 * self.dimension:
                return DS6
            if 3000 * self.dimension < budget:
                MetaModelDS = ParametrizedMetaModel(multivariate_optimizer=DSproba)
                MetaModelDS.no_parallelization = True
                return MetaModelDS
            if 300 * self.dimension < budget < 3000 * self.dimension and self.dimension <= 3:
                return ChainMetaModelDSSQP
            if self.budget < 30 * self.dimension and self.dimension < 50 and (self.dimension > 30):
                return ChainMetaModelDSSQP
            if self.budget >= 30 * self.dimension and self.budget < 300 * self.dimension and (self.dimension == 2):
                return NLOPT_LN_SBPLX
            if self.budget >= 30 * self.dimension and self.budget < 300 * self.dimension and (self.dimension < 15):
                return ChainMetaModelDSSQP
            if self.budget >= 300 * self.dimension and self.budget < 3000 * self.dimension and (self.dimension < 30):
                return MultiDS
        if self.fully_continuous and self.num_workers == 1 and (budget is not None) and (budget < 1000 * self.dimension) and (budget > 20 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 100):
            return DS2
        if self.fully_continuous and self.num_workers == 1 and (budget is not None) and (budget >= 1000 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 50):
            return DS2
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        return optCls

@registry.register
class NgIoh11(NGOptBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self, budget: Optional[int] = None):
        if budget is None:
            budget = self.budget
        else:
            self.budget = budget
        optCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and (not self.fully_continuous) and (not self.has_noise):
            return ConfPortfolio(optimizers=[SuperSmoothDiscreteLenglerOnePlusOne, SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne, DiscreteLenglerOnePlusOne], warmup_ratio=0.4)
        if self.fully_continuous and self.budget is not None and (not self.has_noise):
            if self.budget > 2000 * self.dimension:
                vlpcma = ParametrizedMetaModel(multivariate_optimizer=VLPCMA)
                return vlpcma
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
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension <= 3:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
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
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        return optCls

@registry.register
class NgIoh14(NgIoh11):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self, budget: Optional[int] = None):
        assert budget is None
        optCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and (not self.fully_continuous) and (not self.has_noise):
            return ConfPortfolio(optimizers=[SuperSmoothDiscreteLenglerOnePlusOne, SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne, DiscreteLenglerOnePlusOne], warmup_ratio=0.4)
        if self.fully_continuous and self.budget is not None and (not self.has_noise):
            num = self.budget // (1000 * self.dimension)
            if self.budget > 2000 * self.dimension and num >= self.num_workers:
                optimizers = []
                optimizers += [Rescaled(base_optimizer=NgIoh11._select_optimizer_cls(self), scale=max(0.01, np.exp(-1.0 / np.random.rand())))]
                while len(optimizers) < num:
                    optimizers += [Rescaled(base_optimizer=NgIoh11._select_optimizer_cls(self), scale=max(0.01, np.exp(-1.0 / np.random.rand())))]
                return ConfPortfolio(optimizers=optimizers, warmup_ratio=0.7, no_crossing=True)
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
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension <= 3:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
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
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        return optCls

@registry.register
class NgIoh13(NgIoh11):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self, budget: Optional[int] = None):
        assert budget is None
        optCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and (not self.fully_continuous) and (not self.has_noise):
            return ConfPortfolio(optimizers=[SuperSmoothDiscreteLenglerOnePlusOne, SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne, DiscreteLenglerOnePlusOne], warmup_ratio=0.4)
        if self.fully_continuous and self.budget is not None and (not self.has_noise):
            num = self.budget // (1000 * self.dimension)
            if self.budget > 2000 * self.dimension and num >= self.num_workers:
                optimizers = []
                optimizers += [Rescaled(base_optimizer=NgIoh11._select_optimizer_cls(self), scale=max(0.01, np.exp(-1.0 / np.random.rand())))]
                while len(optimizers) < num:
                    optimizers += [Rescaled(base_optimizer=NgIoh11._select_optimizer_cls(self), scale=max(0.01, np.exp(-1.0 / np.random.rand())))]
                return ConfPortfolio(optimizers=optimizers, warmup_ratio=1.0, no_crossing=True)
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
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension <= 3:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
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
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        return optCls

@registry.register
class NgIoh15(NgIoh11):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self, budget: Optional[int] = None):
        assert budget is None
        optCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and (not self.fully_continuous) and (not self.has_noise):
            return ConfPortfolio(optimizers=[SuperSmoothDiscreteLenglerOnePlusOne, SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne, DiscreteLenglerOnePlusOne], warmup_ratio=0.4)
        if self.fully_continuous and self.budget is not None and (not self.has_noise):
            num = self.budget // (1000 * self.dimension)
            if self.budget > 2000 * self.dimension and num >= self.num_workers:
                optimizers = []
                for _ in range(num):
                    optimizers += [Rescaled(base_optimizer=NgIoh11._select_optimizer_cls(self), scale=max(0.01, np.exp(-1.0 / np.random.rand())))]
                return ConfPortfolio(optimizers=optimizers, warmup_ratio=0.7, no_crossing=True)
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
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension <= 3:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
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
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        return optCls

@registry.register
class NgIoh12(NgIoh11):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self, budget: Optional[int] = None):
        assert budget is None
        optCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and (not self.fully_continuous) and (not self.has_noise):
            return ConfPortfolio(optimizers=[SuperSmoothDiscreteLenglerOnePlusOne, SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne, DiscreteLenglerOnePlusOne], warmup_ratio=0.4)
        if self.fully_continuous and self.budget is not None and (not self.has_noise):
            num = self.budget // (1000 * self.dimension)
            if self.budget > 2000 * self.dimension and num >= self.num_workers:
                optimizers = []
                for _ in range(num):
                    optimizers += [Rescaled(base_optimizer=NgIoh11._select_optimizer_cls(self), scale=max(0.01, np.exp(-1.0 / np.random.rand())))]
                return ConfPortfolio(optimizers=optimizers, warmup_ratio=1.0, no_crossing=True)
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
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension <= 3:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
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
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        return optCls

@registry.register
class NgIoh16(NgIoh11):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self, budget: Optional[int] = None):
        assert budget is None
        optCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and (not self.fully_continuous) and (not self.has_noise):
            return ConfPortfolio(optimizers=[SuperSmoothDiscreteLenglerOnePlusOne, SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne, DiscreteLenglerOnePlusOne], warmup_ratio=0.4)
        if self.fully_continuous and self.budget is not None and (not self.has_noise):
            num = self.budget // (1000 * self.dimension)
            if self.budget > 2000 * self.dimension and num >= self.num_workers:
                optimizers = []
                optimizers += [Rescaled(base_optimizer=Carola14._select_optimizer_cls(self), scale=max(0.01, np.exp(-1.0 / np.random.rand())))]
                while len(optimizers) < num:
                    optimizers += [Rescaled(base_optimizer=Carola14._select_optimizer_cls(self), scale=max(0.01, np.exp(-1.0 / np.random.rand())))]
                return ConfPortfolio(optimizers=optimizers, warmup_ratio=1.0, no_crossing=True)
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
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension <= 3:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
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
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        return optCls

@registry.register
class NgIoh17(NgIoh11):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self, budget: Optional[int] = None):
        assert budget is None
        optCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and (not self.fully_continuous) and (not self.has_noise):
            return ConfPortfolio(optimizers=[SuperSmoothDiscreteLenglerOnePlusOne, SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne, DiscreteLenglerOnePlusOne], warmup_ratio=0.4)
        if self.fully_continuous and self.budget is not None and (not self.has_noise):
            num = self.budget // (1000 * self.dimension)
            if self.budget > 2000 * self.dimension and num >= self.num_workers:
                optimizers = []
                orig_budget = self.budget
                sub_budget = self.budget // num + (self.budget % num > 0)
                for _ in range(num):
                    optimizers += [Rescaled(base_optimizer=NgIoh11._select_optimizer_cls(self, sub_budget), scale=max(0.01, np.exp(-1.0 / np.random.rand())))]
                self.budget = orig_budget
                return Chaining(optimizers, [sub_budget] * (len(optimizers) - 1), no_crossing=True)
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
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension <= 3:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
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
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        return optCls

@registry.register
class NgDS(NgDS11):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self, budget: Optional[int] = None):
        assert budget is None
        optCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and (not self.fully_continuous) and (not self.has_noise):
            return ConfPortfolio(optimizers=[SuperSmoothDiscreteLenglerOnePlusOne, SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne, DiscreteLenglerOnePlusOne], warmup_ratio=0.4)
        if self.fully_continuous and self.budget is not None and (not self.has_noise):
            if self.budget > 2000 * self.dimension:
                vlpcma = ParametrizedMetaModel(multivariate_optimizer=VLPCMA)
                return vlpcma
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (not self.has_noise):
            if 300 * self.dimension < self.budget < 3000 * self.dimension:
                if self.dimension == 2:
                    return DS14
                if self.dimension < 4:
                    return DS4
                if self.dimension < 8:
                    return DS5
                if self.dimension < 15:
                    return DS9
                if self.dimension < 30:
                    return DS8
                if self.dimension < 60:
                    return DS9
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension == 2:
                return DS6
            if 300 * self.dimension < self.budget < 3000 * self.dimension:
                return DS6
            if 3000 * self.dimension < self.budget:
                MetaModelDS = ParametrizedMetaModel(multivariate_optimizer=DSproba)
                MetaModelDS.no_parallelization = True
                return MetaModelDS
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension <= 3:
                return ChainMetaModelDSSQP
            if self.budget < 30 * self.dimension and self.dimension < 50 and (self.dimension > 30):
                return ChainMetaModelDSSQP
            if self.budget >= 30 * self.dimension and self.budget < 300 * self.dimension and (self.dimension == 2):
                return NLOPT_LN_SBPLX
            if self.budget >= 30 * self.dimension and self.budget < 300 * self.dimension and (self.dimension < 15):
                return ChainMetaModelDSSQP
            if self.budget >= 300 * self.dimension and self.budget < 3000 * self.dimension and (self.dimension < 30):
                return MultiDS
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (self.budget < 1000 * self.dimension) and (self.budget > 20 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 100):
            return DS2
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (self.budget >= 1000 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 50):
            return DS2
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        return optCls

@registry.register
class NgIoh21(NgIoh11):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self, budget: Optional[int] = None):
        assert budget is None
        optCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and (not self.fully_continuous) and (not self.has_noise):
            return ConfPortfolio(optimizers=[SuperSmoothDiscreteLenglerOnePlusOne, SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne, DiscreteLenglerOnePlusOne], warmup_ratio=0.4)
        if self.fully_continuous and self.budget is not None and (not self.has_noise):
            if self.budget > 2000 * self.dimension:
                vlpcma = ParametrizedMetaModel(multivariate_optimizer=VLPCMA)
                return vlpcma
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
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension <= 3:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
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
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        return optCls

@registry.register
class NgDS2(NgDS11):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self, budget: Optional[int] = None):
        if self.budget is not None and self.dimension < self.budget:
            return NgIoh21._select_optimizer_cls(self, budget)
        assert budget is None
        optCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and (not self.fully_continuous) and (not self.has_noise):
            return ConfPortfolio(optimizers=[SuperSmoothDiscreteLenglerOnePlusOne, SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne, DiscreteLenglerOnePlusOne], warmup_ratio=0.4)
        if self.fully_continuous and self.budget is not None and (not self.has_noise):
            if self.budget > 2000 * self.dimension:
                vlpcma = ParametrizedMetaModel(multivariate_optimizer=VLPCMA)
                return vlpcma
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (not self.has_noise):
            if 300 * self.dimension < self.budget < 3000 * self.dimension:
                if self.dimension == 2:
                    return DS14
                if self.dimension < 4:
                    return DS4
                if self.dimension < 8:
                    return DS5
                if self.dimension < 15:
                    return DS9
                if self.dimension < 30:
                    return DS8
                if self.dimension < 60:
                    return DS9
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension == 2:
                return DS6
            if 300 * self.dimension < self.budget < 3000 * self.dimension:
                return DS6
            if 3000 * self.dimension < self.budget:
                MetaModelDS = ParametrizedMetaModel(multivariate_optimizer=DSproba)
                MetaModelDS.no_parallelization = True
                return MetaModelDS
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension <= 3:
                return ChainMetaModelDSSQP
            if self.budget < 30 * self.dimension and self.dimension < 50 and (self.dimension > 30):
                return ChainMetaModelDSSQP
            if self.budget >= 30 * self.dimension and self.budget < 300 * self.dimension and (self.dimension == 2):
                return NLOPT_LN_SBPLX
            if self.budget >= 30 * self.dimension and self.budget < 300 * self.dimension and (self.dimension < 15):
                return ChainMetaModelDSSQP
            if self.budget >= 300 * self.dimension and self.budget < 3000 * self.dimension and (self.dimension < 30):
                return MultiDS
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (self.budget < 1000 * self.dimension) and (self.budget > 20 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 100):
            return DS2
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (self.budget >= 1000 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 50):
            return DS2
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        return optCls

@registry.register
class NGDSRW(NGOpt39):

    def _select_optimizer_cls(self):
        if self.fully_continuous and (not self.has_noise) and (self.budget >= 12 * self.dimension):
            return ConfPortfolio(optimizers=[GeneticDE, PSO, super()._select_optimizer_cls()], warmup_ratio=0.33)
        else:
            if self.budget is not None and self.dimension > self.budget:
                return NgDS2._select_optimizer_cls(self)
            return NGOpt39._select_optimizer_cls(self)

@registry.register
class NgIoh20(NgIoh11):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self, budget: Optional[int] = None):
        assert budget is None
        optCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and (not self.fully_continuous) and (not self.has_noise):
            return ConfPortfolio(optimizers=[SuperSmoothDiscreteLenglerOnePlusOne, SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne, DiscreteLenglerOnePlusOne], warmup_ratio=0.4)
        if self.fully_continuous and self.budget is not None and (not self.has_noise):
            if self.budget > 2000 * self.dimension:
                vlpcma = ParametrizedMetaModel(multivariate_optimizer=VLPCMA) if self.dimension > 4 else ParametrizedMetaModel(multivariate_optimizer=LPCMA)
                return vlpcma
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
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension <= 3:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
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
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        return optCls

@registry.register
class NgIoh19(NgIoh11):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self, budget: Optional[int] = None):
        assert budget is None
        optCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and (not self.fully_continuous) and (not self.has_noise):
            return ConfPortfolio(optimizers=[SuperSmoothDiscreteLenglerOnePlusOne, SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne, DiscreteLenglerOnePlusOne], warmup_ratio=0.4)
        if self.fully_continuous and self.budget is not None and (not self.has_noise):
            if self.budget > 2000 * self.dimension:
                vlpcma = VLPCMA if self.dimension > 4 else LPCMA
                return vlpcma
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
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension <= 3:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
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
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        return optCls

@registry.register
class NgIoh18(NgIoh11):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self, budget: Optional[int] = None):
        assert budget is None
        optCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and (not self.fully_continuous) and (not self.has_noise):
            return ConfPortfolio(optimizers=[SuperSmoothDiscreteLenglerOnePlusOne, SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne, DiscreteLenglerOnePlusOne], warmup_ratio=0.4)
        if self.fully_continuous and self.budget is not None and (not self.has_noise):
            num = self.budget // (1000 * self.dimension)
            if self.budget > 2000 * self.dimension and num >= self.num_workers:
                optimizers = []
                orig_budget = self.budget
                sub_budget = self.budget // num + (self.budget % num > 0)
                for _ in range(num):
                    optimizers += [Rescaled(base_optimizer=NgIoh11._select_optimizer_cls(self, sub_budget), scale=np.random.rand() * 2.0)]
                if len(optimizers) < num:
                    optimizers += [Rescaled(base_optimizer=NgIoh11._select_optimizer_cls(self, sub_budget), shift=np.random.randn())]
                if len(optimizers) < num:
                    optimizers += [Rescaled(base_optimizer=NgIoh11._select_optimizer_cls(self, sub_budget), scale=np.random.rand() * 2.0, shift=np.random.randn())]
                self.budget = orig_budget
                return Chaining(optimizers, [sub_budget] * (len(optimizers) - 1), no_crossing=True)
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
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension <= 3:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
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
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        return optCls

@registry.register
class NgIoh10(NGOptBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self):
        optCls = NGOptBase
        funcinfo = self.parametrization.function
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (not self.has_noise):
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension == 2:
                return FCarola6
            if 300 * self.dimension < self.budget < 3000 * self.dimension:
                return Carola6
            if 3000 * self.dimension < self.budget:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension <= 3:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
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
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        return optCls

@registry.register
class NgIoh9(NGOptBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self):
        optCls = NGOptBase
        funcinfo = self.parametrization.function
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (not self.has_noise):
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension == 2:
                return MetaModelFmin2
            if 300 * self.dimension < self.budget < 3000 * self.dimension:
                return ChainMetaModelSQP
            if 3000 * self.dimension < self.budget:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension <= 3:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
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
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        elif self.dimension >= 60 and (not funcinfo.metrizable):
            optCls = CMA
        return optCls

@registry.register
class NgIoh8(NGOptBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self):
        optCls = NGOptBase
        funcinfo = self.parametrization.function
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (not self.has_noise):
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
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        elif self.dimension >= 60 and (not funcinfo.metrizable):
            optCls = CMA
        return optCls

@registry.register
class MixDeterministicRL(ConfPortfolio):
    """MixDeterministicRL = ConfPortfolio(optimizers=[DiagonalCMA, PSO, GeneticDE]).set_name('MixDeterministicRL', register=True)
    """

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1):
        super().__init__(parametrization, budget=budget, num_workers=num_workers, config=ConfPortfolio(optimizers=[DiagonalCMA, PSO, GeneticDE]))

@registry.register
class SpecialRL(Chaining):
    """SpecialRL = Chaining([MixDeterministicRL, TBPSA], ['half']).set_name('SpecialRL', register=True)
    """

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1):
        super().__init__(parametrization, budget=budget, num_workers=num_workers, config=ConfPortfolio(optimizers=[MixDeterministicRL, TBPSA], warmup_ratio=0.5))

@registry.register
class NoisyRL1(Chaining):
    """NoisyRL1 = Chaining([MixDeterministicRL, NoisyOnePlusOne], ['half']).set_name('NoisyRL1', register=True)
    """

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1):
        super().__init__(parametrization, budget=budget, num_workers=num_workers, config=ConfPortfolio(optimizers=[MixDeterministicRL, NoisyOnePlusOne], warmup_ratio=0.5))

@registry.register
class NoisyRL2(Chaining):
    """NoisyRL2 = Chaining([MixDeterministicRL, RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne], ['half']).set_name('NoisyRL2', register=True)
    """

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1):
        super().__init__(parametrization, budget=budget, num_workers=num_workers, config=ConfPortfolio(optimizers=[MixDeterministicRL, RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne], warmup_ratio=0.5))

@registry.register
class NoisyRL3(Chaining):
    """NoisyRL3 = Chaining([MixDeterministicRL, OptimisticNoisyOnePlusOne], ['half']).set_name('NoisyRL3', register=True)
    """

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1):
        super().__init__(parametrization, budget=budget, num_workers=num_workers, config=ConfPortfolio(optimizers=[MixDeterministicRL, OptimisticNoisyOnePlusOne], warmup_ratio=0.5))

@registry.register
class FCarola6(Chaining):
    """FCarola6 = Chaining([NGOpt, NGOpt, RBFGS], ['tenth', 'most']).set_name('FCarola6', register=True)
    """

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1):
        super().__init__(parametrization, budget=budget, num_workers=num_workers, config=ConfPortfolio(optimizers=[NGOpt, NGOpt, RBFGS], warmup_ratio=0.5))

@registry.register
class Carola11(Chaining):
    """Carola11 = Chaining([MultiCMA, RBFGS], ['most']).set_name('Carola11', register=True)
    """

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1):
        super().__init__(parametrization, budget=budget, num_workers=num_workers, config=ConfPortfolio(optimizers=[MultiCMA, RBFGS], warmup_ratio=0.5))

@registry.register
class Carola14(Chaining):
    """Carola14 = Chaining([MultiCMA, RBFGS], ['most']).set_name('Carola14', register=True)
    """

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1):
        super().__init__(parametrization, budget=budget, num_workers=num_workers, config=ConfPortfolio(optimizers=[MultiCMA, RBFGS], warmup_ratio=0.5))

@registry.register
class DS14(Chaining):
    """DS14 = Chaining([MultiDS, RBFGS], ['most']).set_name('DS14', register=True)
    """

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1):
        super().__init__(parametrization, budget=budget, num_workers=num_workers, config=ConfPortfolio(optimizers=[MultiDS, RBFGS], warmup_ratio=0.5))

@registry.register
class Carola13(Chaining):
    """Carola13 = Chaining([CmaFmin2, RBFGS], ['most']).set_name('Carola13', register=True)
    """

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1):
        super().__init__(parametrization, budget=budget, num_workers=num_workers, config=ConfPortfolio(optimizers=[CmaFmin2, RBFGS], warmup_ratio=0.5))

@registry.register
class Carola15(Chaining):
    """Carola15 = Chaining([Cobyla, MetaModel, RBFGS], ['sqrt', 'most']).set_name('Carola15', register=True)
    """

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1):
        super().__init__(parametrization, budget=budget, num_workers=num_workers, config=ConfPortfolio(optimizers=[Cobyla, MetaModel, RBFGS], warmup_ratio=0.5))

@registry.register
class CSEC(NGOpt39):

    def _select_optimizer_cls(self, budget: Optional[int] = None):
        assert budget is None
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (self.budget >= 3000 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 50):
            return NgIoh21._select_optimizer_cls(self, budget)
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (self.budget >= 30 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 50):
            return NgIoh4._select_optimizer_cls(self)
        if self.fully_continuous and self.budget is not None and (self.num_workers > np.sqrt(self.budget)) and (self.budget >= 30 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 50):
            return NgDS3
        return NgDS2._select_optimizer_cls(self, budget)

@registry.register
class CSEC10(NGOptBase):

    def _select_optimizer_cls(self, budget: Optional[int] = None):
        assert self.budget is not None
        function = self.parametrization
        if isinstance(self.parametrization, p.Array) and (not self.fully_continuous) and (not self.has_noise):
            return GeneticDE
        if function.real_world and (not function.hptuning) and (not function.neural) and (not self.parametrization.function.metrizable):
            return NgDS2._select_optimizer_cls(self, budget)
        if function.has_constraints:
            return NgLn
        if self.num_workers == 1 and function.real_world and (not function.hptuning) and (not function.neural) and (self.dimension > self.budget) and self.fully_continuous and (not self.has_noise):
            return DSproba
        if self.num_workers < np.sqrt(1 + self.dimension) and function.real_world and (not function.hptuning) and (not function.neural) and (8 * self.dimension > self.budget) and self.fully_continuous and (not self.has_noise):
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
            return NgIoh21._select_optimizer_cls(self, budget)
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (self.budget >= 30 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 50):
            return NgIoh4._select_optimizer_cls(self)
        if self.fully_continuous and self.budget is not None and (self.num_workers > np.log(3 + self.budget)) and (self.budget >= 30 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 50):
            return NgDS3
        return NgDS2._select_optimizer_cls(self, budget)

@registry.register
class CSEC11(NGOptBase):

    def _select_optimizer_cls(self, budget: Optional[int] = None):
        assert self.budget is not None
        function = self.parametrization
        if isinstance(self.parametrization, p.Array) and (not self.fully_continuous) and (not self.has_noise):
            return GeneticDE
        if function.real_world and (not function.hptuning) and (not function.neural) and (not self.parametrization.function.metrizable):
            return NgDS2._select_optimizer_cls(self, budget)
        if function.has_constraints:
            return NgLn
        if self.num_workers == 1 and function.real_world and (not function.hptuning) and (not function.neural) and (self.dimension > self.budget) and self.fully_continuous and (not self.has_noise):
            return DSproba
        if self.num_workers < np.sqrt(1 + self.dimension) and function.real_world and (not function.hptuning) and (not function.neural) and (8 * self.dimension > self.budget) and self.fully_continuous and (not self.has_noise):
            return DiscreteLenglerOnePlusOne
        if function.real_world and (not function.hptuning) and (not function.neural) and self.fully_continuous:
            return NGOpt._select_optimizer_cls(self)
        if function.real_world and function.neural and (not function.function.deterministic) and (not function.enforce_determinism):
            return NoisyRL2
        if function.real_world and function.neural and (function.function.deterministic or function.enforce_determinism):
            return SQOPSO
        if function.real_world and (not function.neural):
            return NGDSRW._select_optimizer_cls(self)
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (self.budget >= 300 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 50):
            return NgIoh21._select_optimizer_cls(self, budget)
        if self.fully_continuous and self.num_workers == 1 and (self.budget is not None) and (self.budget >= 30 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 50):
            return NgIoh4._select_optimizer_cls(self)
        if self.fully_continuous and self.budget is not None and (self.num_workers > np.log(3 + self.budget)) and (self.budget >= 30 * self.dimension) and (not self.has_noise) and (self.dimension > 1) and (self.dimension < 50):
            return NgDS3
        return NgDS2._select_optimizer_cls(self, budget)

@registry.register
class NgIohTuned(CSEC11):
    pass

@registry.register
class SplitCSEC11(ConfSplitOptimizer):
    """SplitCSEC11 = ConfSplitOptimizer(multivariate_optimizer=CSEC11, monovariate_optimizer=CSEC11, non_deterministic_descriptor=False).set_name('SplitCSEC11', register=True)
    """

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1):
        super().__init__(parametrization, budget=budget, num_workers=num_workers, config=ConfSplitOptimizer(multivariate_optimizer=CSEC11, monovariate_optimizer=CSEC11, non_deterministic_descriptor=False))

@registry.register
class SplitSQOPSO(ConfSplitOptimizer):
    """SplitSQOPSO = ConfSplitOptimizer(multivariate_optimizer=SQOPSO, monovariate_optimizer=SQOPSO, non_deterministic_descriptor=False).set_name('SplitSQOPSO', register=True)
    """

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1):
        super().__init__(parametrization, budget=budget, num_workers=num_workers, config=ConfSplitOptimizer(multivariate_optimizer=SQOPSO, monovariate_optimizer=SQOPSO, non_deterministic_descriptor=False))

@registry.register
class SplitPSO(ConfSplitOptimizer):
    """SplitPSO = ConfSplitOptimizer(multivariate_optimizer=PSO, monovariate_optimizer=PSO, non_deterministic_descriptor=False).set_name('SplitPSO', register=True)
    """

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1):
        super().__init__(parametrization, budget=budget, num_workers=num_workers, config=ConfSplitOptimizer(multivariate_optimizer=PSO, monovariate_optimizer=PSO, non_deterministic_descriptor=False))

@registry.register
class SplitCMA(ConfSplitOptimizer):
    """SplitCMA = ConfSplitOptimizer(multivariate_optimizer=CMA, monovariate_optimizer=CMA, non_deterministic_descriptor=False).set_name('SplitCMA', register=True)
    """

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1):
        super().__init__(parametrization, budget=budget, num_workers=num_workers, config=ConfSplitOptimizer(multivariate_optimizer=CMA, monovariate_optimizer=CMA, non_deterministic_descriptor=False))

@registry.register
class SplitQODE(ConfSplitOptimizer):
    """SplitQODE = ConfSplitOptimizer(multivariate_optimizer=QODE, monovariate_optimizer=QODE, non_deterministic_descriptor=False).set_name('SplitQODE', register=True)
    """

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1):
        super().__init__(parametrization, budget=budget, num_workers=num_workers, config=ConfSplitOptimizer(multivariate_optimizer=QODE, monovariate_optimizer=QODE, non_deterministic_descriptor=False))

@registry.register
class SplitTwoPointsDE(ConfSplitOptimizer):
    """SplitTwoPointsDE = ConfSplitOptimizer(multivariate_optimizer=TwoPointsDE, monovariate_optimizer=TwoPointsDE, non_deterministic_descriptor=False).set_name('SplitTwoPointsDE', register=True)
    """

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1):
        super().__init__(parametrization, budget=budget, num_workers=num_workers, config=ConfSplitOptimizer(multivariate_optimizer=TwoPointsDE, monovariate_optimizer=TwoPointsDE, non_deterministic_descriptor=False))

@registry.register
class SplitDE(ConfSplitOptimizer):
    """SplitDE = ConfSplitOptimizer(multivariate_optimizer=DE, monovariate_optimizer=DE, non_deterministic_descriptor=False).set_name('SplitDE', register=True)
    """

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1):
        super().__init__(parametrization, budget=budget, num_workers=num_workers, config=ConfSplitOptimizer(multivariate_optimizer=DE, monovariate_optimizer=DE, non_deterministic_descriptor=False))

@registry.register
class SQOPSODCMA(Chaining):
    """SQOPSODCMA = Chaining([SQOPSO, DiagonalCMA], ['half']).set_name('SQOPSODCMA', register=True)
    """

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1):
        super().__init__(parametrization, budget=budget, num_workers=num_workers, config=ConfPortfolio(optimizers=[SQOPSO, DiagonalCMA], warmup_ratio=0.5))

@registry.register
class SQOPSODCMA20(SQOPSODCMA):
    """SQOPSODCMA20 = Chaining(optimizers=[SQOPSODCMA] * 20, budgets=['equal'] * 19, no_crossing=True).set_name('SQOPSODCMA20', register=True)
    """

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1):
        super().__init__(parametrization, budget=budget, num_workers=num_workers, config=ConfPortfolio(optimizers=[SQOPSODCMA] * 20, warmup_ratio=0.5))

@registry.register
class SQOPSODCMA20bar(SQOPSODCMA):
    """SQOPSODCMA20bar = ConfPortfolio(optimizers=[SQOPSODCMA] * 20, warmup_ratio=0.5).set_name('SQOPSODCMA20bar', register=True)
    """

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1):
        super().__init__(parametrization, budget=budget, num_workers=num_workers, config=ConfPortfolio(optimizers=[SQOPSODCMA] * 20, warmup_ratio=0.5))

@registry.register
class NgIohLn(Chaining):
    """NgIohLn = Chaining([LognormalDiscreteOnePlusOne, CSEC11], ['tenth']).set_name('NgIohLn', register=True)
    """

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1):
        super().__init__(parametrization, budget=budget, num_workers=num_workers, config=ConfPortfolio(optimizers=[LognormalDiscreteOnePlusOne, CSEC11], warmup_ratio=0.5))

@registry.register
class CMALn(Chaining):
    """CMALn = Chaining([LognormalDiscreteOnePlusOne, CMA], ['tenth']).set_name('CMALn', register=True)
    """

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1):
        super().__init__(parametrization, budget=budget, num_workers=num_workers, config=ConfPortfolio(optimizers=[LognormalDiscreteOnePlusOne, CMA], warmup_ratio=0.5))

@registry.register
class CMARS(Chaining):
    """CMARS = Chaining([RandomSearch, CMA], ['tenth']).set_name('CMARS', register=True)
    """

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1):
        super().__init__(parametrization, budget=budget, num_workers=num_workers, config=ConfPortfolio(optimizers=[RandomSearch, CMA], warmup_ratio=0.5))

@registry.register
class NgIohRS(Chaining):
    """NgIohRS = Chaining([oneshot.RandomSearch, CSEC11], ['tenth']).set_name('NgIohRS', register=True)
    """

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1):
        super().__init__(parametrization, budget=budget, num_workers=num_workers, config=ConfPortfolio(optimizers=[oneshot.RandomSearch, CSEC11], warmup_ratio=0.5))

@registry.register
class PolyLN(ConfPortfolio):
    """PolyLN = ConfPortfolio(optimizers=[Rescaled(base_optimizer=SmallLognormalDiscreteOnePlusOne, scale=np.random.rand()) for i in range(20)], warmup_ratio=0.5).set_name('PolyLN', register=True)
    """

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1):
        super().__init__(parametrization, budget=budget, num_workers=num_workers, config=ConfPortfolio(optimizers=[Rescaled(base_optimizer=SmallLognormalDiscreteOnePlusOne, scale=np.random.rand()) for i in range(20)], warmup_ratio=0.5))

@registry.register
class MultiLN(ConfPortfolio):
    """MultiLN = ConfPortfolio(optimizers=[Rescaled(base_optimizer=SmallLognormalDiscreteOnePlusOne, scale=2.0 ** (i - 5)) for i in range(6)], warmup_ratio=0.5).set_name('MultiLN', register=True)
    """

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1):
        super().__init__(parametrization, budget=budget, num_workers=num_workers, config=ConfPortfolio(optimizers=[Rescaled(base_optimizer=SmallLognormalDiscreteOnePlusOne, scale=2.0 ** (i - 5)) for i in range(6)], warmup_ratio=0.5))

@registry.register
class ManyLN(ConfPortfolio):
    """ManyLN = ConfPortfolio(optimizers=[SmallLognormalDiscreteOnePlusOne for i in range(20)], warmup_ratio=0.5).set_name('ManyLN', register=True)
    """

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1):
        super().__init__(parametrization, budget=budget, num_workers=num_workers, config=ConfPortfolio(optimizers=[SmallLognormalDiscreteOnePlusOne for i in range(20)], warmup_ratio=0.5))

@registry.register
class NgIohMLn(Chaining):
    """NgIohMLn = Chaining([MultiLN, CSEC11], ['tenth']).set_name('NgIohMLn', register=True)
    """

    def __init__(self, parametrization: p.Parameter, budget: Optional[int] = None, num_workers: int = 1):
        super().__init__(parametrization, budget=budget, num_workers=num_workers, config=ConfPortfolio(optimizers=[MultiLN, CSEC11], warmup_ratio=0.5))
