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

    def __init__(self, parametrization: p.Parameter, budget: int = None, num_workers: int = 1, *, 
                 noise_handling: str = None, tabu_length: int = 0, mutation: str = 'gaussian', 
                 crossover: bool = False, rotation: bool = False, annealing: str = 'none', 
                 use_pareto: bool = False, sparse: bool = False, smoother: bool = False, 
                 super_radii: bool = False, roulette_size: int = 2, antismooth: int = 55, 
                 crossover_type: str = 'none', forced_discretization: bool = False):
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
        if self.fully_continuous:
            self._arity = -1
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

    def _internal_ask_candidate(self) -> p.Parameter:
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
        pessimistic = self.current_bests['pessimistic'].parameter.spawn_child()
        if self.smoother and self._num_ask % max(self.num_workers + 1, self.antismooth) == 0 and isinstance(self.parametrization, p.Array):
            possible_radii = [3] if not self.super_radii else [3, 3 + np.random.randint(int(np.sqrt(np.sqrt(self.dimension))))]
            self.suggest(smooth_copy(pessimistic, possible_radii=possible_radii).value)
        if self.num_objectives > 1 and self.use_pareto:
            pareto = self.pareto_front()
            pessimistic = pareto[self._rng.choice(len(pareto))].spawn_child()
        ref = self.parametrization
        if self.crossover and self._num_ask % 2 == 1 and (len(self.archive) > 2):
            data = mutator.crossover(pessimistic.get_standardized_data(reference=ref), 
                                     mutator.get_roulette(self.archive, num=self.roulette_size), 
                                     rotation=self.rotation, crossover_type=self.crossover_type)
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
                func = {'discrete': mutator.discrete_mutation, 'fastga': mutator.doerr_discrete_mutation, 
                        'doublefastga': mutator.doubledoerr_discrete_mutation, 'rls': mutator.rls_mutation, 
                        'portfolio': mutator.portfolio_discrete_mutation}[mutation]
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

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        if self.annealing != 'none':
            assert isinstance(self.budget, int)
            delta = self._previous_best_loss - loss
            if loss > self._max_loss:
                self._max_loss = loss
            if delta >= 0:
                self._annealing_base = candidate.get_standardized_data(reference=self.parametrization)
            elif self.num_ask < self.budget:
                amplitude = max(1.0, self._max_loss - self._previous_best_loss)
                annealing_dict = {'Exp0.9': 0.33 * amplitude * 0.9 ** self.num_ask, 'Exp0.99': 0.33 * amplitude * 0.99 ** self.num_ask, 
                                  'Exp0.9Auto': 0.33 * amplitude * (0.001 ** (1.0 / self.budget)) ** self.num_ask, 
                                  'Lin100.0': 100.0 * amplitude * (1 - self.num_ask / (self.budget + 1)), 
                                  'Lin1.0': 1.0 * amplitude * (1 - self.num_ask / (self.budget + 1)), 
                                  'LinAuto': 10.0 * amplitude * (1 - self.num_ask / (self.budget + 1))}
                T = annealing_dict[self.annealing]
                if T > 0.0:
                    proba = np.exp(delta / T)
                    if self._rng.rand() < proba:
                        self._annealing_base = candidate.get_standardized_data(reference=self.parametrization)
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

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        """Called whenever calling :code:`tell` on a candidate that was "asked"."""
        data = candidate.get_standardized_data(reference=self.parametrization)
        if self.mutation == 'coordinatewise_adaptive':
            self.inds = candidate._meta['modified_variables'] if 'modified_variables' in candidate._meta else np.array([True] * len(data))
        if 'lognormal' in self.mutation:
            self.imr = candidate._meta['individual_mutation_rate'] if 'individual_mutation_rate' in candidate._meta else 0.2
        self._internal_tell(data, loss)

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

    def __init__(self, *, noise_handling: str = None, tabu_length: int = 0, mutation: str = 'gaussian', 
                 crossover: bool = False, rotation: bool = False, annealing: str = 'none', 
                 use_pareto: bool = False, sparse: bool = False, smoother: bool = False, 
                 super_radii: bool = False, roulette_size: int = 2, antismooth: int = 55, 
                 crossover_type: str = 'none', forced_discretization: bool = False):
        super().__init__(_OnePlusOne, locals())

class _CMA(base.Optimizer):
    _CACHE_KEY = '#CMA#datacache'

    def __init__(self, parametrization: p.Parameter, budget: int = None, num_workers: int = 1, 
                 config: base.ConfiguredOptimizer = None):
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.algorithm = config.algorithm if config is not None else 'quad'
        self._config = config if config is not None else ParametrizedCMA()
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
                inopts = dict(popsize=self._popsize, randn=self._rng.randn, CMA_diagonal=self._config.diagonal, 
                              verbose=-9, seed=np.nan, CMA_elitist=self._config.elitist)
                inopts.update(self._config.inopts if self._config.inopts is not None else {})
                self._es = cma.CMAEvolutionStrategy(x0=self.parametrization.sample().get_standardized_data(reference=self.parametrization) 
                                                   if self._config.random_init else np.zeros(self.dimension, dtype=np.float64), 
                                                   sigma0=self._config.scale * scale_multiplier, inopts=inopts)
            else:
                try:
                    from fcmaes import cmaes
                except ImportError as e:
                    raise ImportError('Please install fcmaes (pip install fcmaes) to use FCMA optimizers') from e
                self._es = cmaes.Cmaes(x0=np.zeros(self.dimension, dtype=np.float64), input_sigma=self._config.scale * scale_multiplier, 
                                       popsize=self._popsize, randn=self._rng.randn)
        return self._es

    def _internal_ask_candidate(self) -> p.Parameter:
        if not self._to_be_asked:
            self._to_be_asked.extend(self.es.ask())
        data = self._to_be_asked.popleft()
        parent = self._parents[self.num_ask % len(self._parents)]
        candidate = parent.spawn_child().set_standardized_data(data, reference=self.parametrization)
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
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

    def _internal_provide_recommendation(self) -> np.ndarray:
        pessimistic = self.current_bests['pessimistic'].parameter.get_standardized_data(reference=self.parametrization)
        d = self.dimension
        n = self.num_ask
        sample_size = int(d * d / 2 + d / 2 + 3)
        if self._config.high_speed and n >= sample_size:
            try:
                data = learn_on_k_best(self.archive, sample_size, self.algorithm)
                return data
            except MetaModelFailure:
                pass
        if self._es is None:
            return pessimistic
        cma_best = self.es.best_x if self._config.fcmaes else self.es.result.xbest
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

    def __init__(self, *, scale: float = 1.0, elitist: bool = False, popsize: int = None, 
                 popsize_factor: float = 3.0, diagonal: bool = False, zero: bool = False, 
                 high_speed: bool = False, fcmaes: bool = False, random_init: bool = False, 
                 inopts: dict = None, algorithm: str = 'quad'):
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

class ChoiceBase(base.Optimizer):
    """Nevergrad optimizer by competence map."""

    def __init__(self, parametrization: p.Parameter, budget: int = None, num_workers: int = 1):
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
    def optim(self) -> base.Optimizer:
        if self._optim is None:
            self._optim = self._select_optimizer_cls()(self.parametrization, self.budget, self.num_workers)
            self._optim = self._optim if not isinstance(self._optim, NGOptBase) else self._optim.optim
            logger.debug('%s selected %s optimizer.', *(x.name for x in (self, self._optim)))
        return self._optim

    def _select_optimizer_cls(self) -> type:
        return CMA

    def _internal_ask_candidate(self) -> p.Parameter:
        return self.optim.ask()

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        self.optim.tell(candidate, loss)

    def recommend(self) -> p.Parameter:
        return self.optim.recommend()

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: float) -> None:
        self.optim.tell(candidate, loss)

    def _info(self) -> dict:
        out = {'sub-optim': self.optim.name}
        out.update(self.optim._info())
        return out

    def enable_pickling(self) -> None:
        self.optim.enable_pickling()

class MetaCMA(ChoiceBase):
    """Nevergrad CMA optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self) -> type:
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
