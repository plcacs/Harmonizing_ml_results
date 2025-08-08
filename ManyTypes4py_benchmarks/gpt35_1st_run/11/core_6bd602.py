import numpy as np
from nevergrad.parametrization import parameter as p
from .hypervolume import HypervolumeIndicator
from typing import Tuple

AUTO_BOUND: int = 15

class HypervolumePareto:
    def __init__(self, *, upper_bounds: tp.Optional[Tuple[float, ...]] = None, auto_bound: int = AUTO_BOUND, seed: tp.Optional[int] = None, no_hypervolume: bool = False) -> None:
        self._auto_bound: int = 0
        self._upper_bounds: np.ndarray = np.asarray([-float('inf')]) if upper_bounds is None else np.array(upper_bounds)
        if upper_bounds is None:
            self._auto_bound = auto_bound
        self._best_volume: float = -float('Inf')
        self._hypervolume: tp.Optional[HypervolumeIndicator] = None
        self._pareto_needs_filtering: bool = False
        self._no_hypervolume: bool = no_hypervolume
        self._pf: ParetoFront = ParetoFront(seed=seed, no_hypervolume=no_hypervolume)

    @property
    def num_objectives(self) -> int:
        return self._upper_bounds.size

    @property
    def best_volume(self) -> float:
        return self._best_volume

    def extend(self, parameters: tp.List[p.Parameter]) -> float:
        output: float = 0.0
        for param in parameters:
            output = self.add(param)
        return output

    def add(self, parameter: p.Parameter) -> float:
        if not isinstance(parameter, p.Parameter):
            raise TypeError(f'{self.__class__.__name__}.add should receive a ng.p.Parameter, but got: {parameter}.')
        losses: np.ndarray = parameter.losses
        if not isinstance(losses, np.ndarray):
            raise TypeError(f'Parameter should have multivalue as losses, but parameter.losses={losses} ({type(losses)}).')
        if self._auto_bound > 0:
            self._auto_bound -= 1
            if (self._upper_bounds > -float('inf')).all() and (losses > self._upper_bounds).all():
                return float('inf')
            self._upper_bounds = np.maximum(self._upper_bounds, losses)
            self._pf.add_to_pareto(parameter)
            return 0.0
        if (losses - self._upper_bounds > 0).any():
            loss: float = -float(np.sum(np.maximum(0, losses - self._upper_bounds)))
            if loss > self._best_volume:
                self._best_volume = loss
            if self._best_volume < 0:
                self._pf.add_to_pareto(parameter)
            return -loss
        if self._no_hypervolume:
            self._pf.add_to_pareto(parameter)
            return 0.0
        return self._calc_hypervolume(parameter, losses)

    def _calc_hypervolume(self, parameter: p.Parameter, losses: np.ndarray) -> float:
        if self._hypervolume is None:
            self._hypervolume = HypervolumeIndicator(self._upper_bounds)
            self._pf._hypervolume = self._hypervolume
        new_volume: float = self._hypervolume.compute([pa.losses for pa in self._pf.get_raw()] + [losses])
        if new_volume > self._best_volume:
            self._best_volume = new_volume
            self._pf.add_to_pareto(parameter)
            return -new_volume
        else:
            distance_to_pareto: float = float('Inf')
            for param in self._pf.get_front():
                stored_losses: np.ndarray = param.losses
                if (stored_losses <= losses).all():
                    distance_to_pareto = min(distance_to_pareto, min(losses - stored_losses))
            assert distance_to_pareto >= 0
            return 0.0 if self._no_hypervolume else -new_volume + distance_to_pareto

    def pareto_front(self, size: tp.Optional[int] = None, subset: str = 'random', subset_tentatives: int = 12) -> tp.List[p.Parameter]:
        return self._pf.get_front(size, subset, subset_tentatives)

    def get_min_losses(self) -> np.ndarray:
        return np.min([p.losses for p in self._pf.get_raw()], axis=0)

class ParetoFront:
    def __init__(self, *, seed: tp.Optional[int] = None, no_hypervolume: bool = False) -> None:
        self._pareto: tp.List[p.Parameter] = []
        self._pareto_needs_filtering: bool = False
        self._no_hypervolume: bool = no_hypervolume
        self._rng: np.random.RandomState = seed if isinstance(seed, np.random.RandomState) else np.random.RandomState(seed)
        self._hypervolume: tp.Optional[HypervolumeIndicator] = None

    def add_to_pareto(self, parameter: p.Parameter) -> None:
        self._pareto.append(parameter)
        self._pareto_needs_filtering = True

    def _filter_pareto_front(self) -> None:
        new_pareto: tp.List[p.Parameter] = []
        for param in self._pareto:
            should_be_added: bool = True
            for other in self._pareto:
                if (other.losses <= param.losses).all() and (other.losses < param.losses).any():
                    should_be_added = False
                    break
            if should_be_added:
                new_pareto.append(param)
        self._pareto = new_pareto
        self._pareto_needs_filtering = False

    def get_raw(self) -> tp.List[p.Parameter]:
        return self._pareto

    def get_front(self, size: tp.Optional[int] = None, subset: str = 'random', subset_tentatives: int = 12) -> tp.List[p.Parameter]:
        if self._pareto_needs_filtering:
            self._filter_pareto_front()
        if size is None or size >= len(self._pareto):
            return self._pareto
        if subset == 'random':
            return self._rng.choice(self._pareto, size).tolist()
        tentatives: tp.List[tp.List[p.Parameter]] = [self._rng.choice(self._pareto, size).tolist() for _ in range(subset_tentatives)]
        if self._hypervolume is None and subset == 'hypervolume':
            raise RuntimeError('Hypervolume subsetting not supported as hypervolume not in use')
        hypervolume: HypervolumeIndicator = self._hypervolume
        scores: tp.List[float] = []
        for tentative in tentatives:
            if subset == 'hypervolume':
                scores += [-hypervolume.compute([pa.losses for pa in tentative])]
            else:
                score: float = 0.0
                for v in self._pareto:
                    best_score: float = float('inf') if subset != 'EPS' else 0.0
                    for pa in tentative:
                        if subset == 'loss-covering':
                            best_score = min(best_score, np.linalg.norm(pa.losses - v.losses))
                        elif subset == 'EPS':
                            best_score = min(best_score, max(pa.losses - v.losses))
                        elif subset == 'domain-covering':
                            best_score = min(best_score, np.linalg.norm(pa.get_standardized_data(reference=v)))
                        else:
                            raise ValueError(f'Unknown subset for Pareto-Set subsampling: "{subset}"')
                    score += best_score ** 2 if subset != 'EPS' else max(score, best_score)
                scores += [score]
        return tentatives[scores.index(min(scores))]
