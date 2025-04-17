import warnings
import nevergrad.common.typing as tp
from nevergrad.parametrization import parameter as p
from nevergrad.optimization.utils import UidQueue
from . import base
from .multiobjective import nsga2 as nsga2


class _EvolutionStrategy(base.Optimizer):
    """Experimental evolution-strategy-like algorithm
    The behavior is going to evolve
    """

    def __init__(self, parametrization, budget=None, num_workers=1, *,
        config: tp.Optional['EvolutionStrategy']=None):
        if budget is not None and budget < 60:
            warnings.warn('ES algorithms are inefficient with budget < 60',
                base.errors.InefficientSettingsWarning)
        super().__init__(parametrization, budget=budget, num_workers=
            num_workers)
        self._population: tp.Dict[str, p.Parameter] = {}
        self._uid_queue = UidQueue()
        self._waiting: tp.List[p.Parameter] = []
        self._config: EvolutionStrategy = EvolutionStrategy(
            ) if config is None else config
        self._rank_method: tp.Optional[tp.Callable[[tp.List[p.Parameter],
            int], tp.List[str]]] = None
        if self._config.ranker == 'nsga2':
            self._rank_method = nsga2.rank
        elif self._config.ranker != 'simple':
            raise NotImplementedError(f'Unknown ranker {self._config.ranker}')
        self._no_hypervolume: bool = self._config.offsprings is None

    def _internal_ask_candidate(self):
        if self.num_ask < self._config.popsize:
            param = self.parametrization.sample()
            assert param.uid == param.heritage['lineage']
            self._uid_queue.asked.add(param.uid)
            self._population[param.uid] = param
            return param
        uid: str = self._uid_queue.ask()
        param: p.Parameter = self._population[uid].spawn_child()
        param.mutate()
        ratio: float = self._config.recombination_ratio
        if ratio and self._rng.rand() < ratio:
            selected_uid: str = self._rng.choice(list(self._population))
            param.recombine(self._population[selected_uid])
        return param

    def _internal_tell_candidate(self, candidate, loss):
        if self._config.offsprings is None:
            uid: str = candidate.heritage['lineage']
            self._uid_queue.tell(uid)
            parent_value: float = float('inf'
                ) if uid not in self._population else base._loss(self.
                _population[uid])
            if loss < parent_value:
                self._population[uid] = candidate
        else:
            no_parent: bool = next(iter(candidate.parents_uids), '#no_parent#'
                ) not in self._population
            if no_parent and len(self._population) < self._config.popsize:
                self._population[candidate.uid] = candidate
                self._uid_queue.tell(candidate.uid)
            else:
                self._waiting.append(candidate)
            if len(self._waiting) >= self._config.offsprings:
                self._select()

    def _select(self):
        choices: tp.List[p.Parameter] = self._waiting + ([] if self._config
            .only_offsprings else list(self._population.values()))
        if self._rank_method is not None and self.num_objectives > 1:
            choices_rank: tp.List[str] = self._rank_method(choices,
                n_selected=self._config.popsize)
            choices = [x for x in choices if x.uid in choices_rank]
        else:
            choices.sort(key=base._loss)
        self._population = {x.uid: x for x in choices[:self._config.popsize]}
        self._uid_queue.clear()
        self._waiting.clear()
        for uid in self._population:
            self._uid_queue.tell(uid)


class EvolutionStrategy(base.ConfiguredOptimizer):
    """Experimental evolution-strategy-like algorithm
    The API is going to evolve

    Parameters
    ----------
    recombination_ratio: float
        probability of using a recombination (after the mutation) for generating new offsprings
    popsize: int
        population size of the parents (lambda)
    offsprings: int
        number of generated offsprings (mu)
    only_offsprings: bool
        use only offsprings for the new generation if True (True: lambda,mu, False: lambda+mu)
    ranker: str
        ranker for the multiobjective case (defaults to NSGA2)
    """

    def __init__(self, *, recombination_ratio: float=0, popsize: int=40,
        offsprings: tp.Optional[int]=None, only_offsprings: bool=False,
        ranker: str='nsga2'):
        super().__init__(_EvolutionStrategy, locals(), as_config=True)
        assert offsprings is None or not only_offsprings or offsprings > popsize
        if only_offsprings:
            assert offsprings is not None, 'only_offsprings only work if offsprings is not None (non-DE mode)'
        assert 0 <= recombination_ratio <= 1
        assert ranker in ['simple', 'nsga2']
        self.recombination_ratio: float = recombination_ratio
        self.popsize: int = popsize
        self.offsprings: tp.Optional[int] = offsprings
        self.only_offsprings: bool = only_offsprings
        self.ranker: str = ranker


RecES: EvolutionStrategy = EvolutionStrategy(recombination_ratio=1,
    only_offsprings=True, offsprings=60).set_name('RecES', register=True)
RecMixES: EvolutionStrategy = EvolutionStrategy(recombination_ratio=1,
    only_offsprings=False, offsprings=20).set_name('RecMixES', register=True)
RecMutDE: EvolutionStrategy = EvolutionStrategy(recombination_ratio=1,
    only_offsprings=False, offsprings=None).set_name('RecMutDE', register=True)
ES: EvolutionStrategy = EvolutionStrategy(recombination_ratio=0,
    only_offsprings=True, offsprings=60).set_name('ES', register=True)
MixES: EvolutionStrategy = EvolutionStrategy(recombination_ratio=0,
    only_offsprings=False, offsprings=20).set_name('MixES', register=True)
MutDE: EvolutionStrategy = EvolutionStrategy(recombination_ratio=0,
    only_offsprings=False, offsprings=None).set_name('MutDE', register=True)
NonNSGAIIES: EvolutionStrategy = EvolutionStrategy(recombination_ratio=0,
    only_offsprings=True, offsprings=60, ranker='simple').set_name(
    'NonNSGAIIES', register=True)
