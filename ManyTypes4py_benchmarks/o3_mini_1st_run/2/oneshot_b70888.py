from typing import Any, Callable, List, Optional, Sequence, Union
import copy
import numpy as np
from scipy.spatial import ConvexHull
import nevergrad.common.typing as tp
from nevergrad.parametrization import parameter as p
from nevergrad.parametrization import transforms as trans
from . import sequences
from . import base
from .base import IntOrParameter
from . import utils


def convex_limit(struct_points: Sequence[Sequence[float]]) -> int:
    """Given points in order from best to worst,
    Returns the length of the maximum initial segment of points such that quasiconvexity is verified.
    """
    points: List[Sequence[float]] = []
    d: int = len(struct_points[0])
    if len(struct_points) < 2 * d + 2:
        return len(struct_points) // 2
    for i in range(0, min(2 * d + 2, len(struct_points)), 2):
        points += [struct_points[i]]
    hull: ConvexHull = ConvexHull(points[: d + 1], incremental=True)
    num_points: int = len(hull.vertices)
    k: int = len(points) - 1
    for i in range(num_points, len(points)):
        hull.add_points(points[i : i + 1])
        num_points += 1
        if len(hull.vertices) != num_points:
            return num_points - 1
        for j in range(i + 1, len(points)):
            hull_copy: ConvexHull = copy.deepcopy(hull)
            hull_copy.add_points(points[j : j + 1])
            if len(hull_copy.vertices) != num_points + 1:
                return num_points - 1
    return k


def hull_center(points: Sequence[Sequence[float]], k: int) -> np.ndarray:
    """Center of the cuboid enclosing the hull."""
    hull: ConvexHull = ConvexHull(points[:k])
    maxi: np.ndarray = np.asarray(hull.vertices[0])
    mini: np.ndarray = np.asarray(hull.vertices[0])
    for v in hull.vertices:
        maxi = np.maximum(np.asarray(v), maxi)
        mini = np.minimum(np.asarray(v), mini)
    return 0.5 * (maxi + mini)


def avg_of_k_best(archive: Any, method: str = 'dimfourth') -> np.ndarray:
    """Operators inspired by the work of Yann Chevaleyre, Laurent Meunier, Clement Royer, Olivier Teytaud, Fabien Teytaud.

    Parameters
    ----------
    archive: utils.Archive[utils.Value]
        Provides a random recommendation instead of the best point so far (for baseline)
    method: str
        If dimfourth, we use the Fteytaud heuristic, i.e. k = min(len(archive) // 4, dimension)
        If exp, we use the Lmeunier method, i.e. k=max(1, len(archive) // (2**dimension))
        If hull, we use the maximum k <= dimfourth-value, such that the function looks quasiconvex on the k best points.
    """
    items: List[Any] = list(archive.items_as_arrays())
    dimension: int = len(items[0][0])
    if method == 'dimfourth':
        k: int = min(len(archive) // 4, dimension)
    elif method == 'exp':
        k = max(1, int(len(archive) // 1.1 ** dimension))
    elif method == 'hull':
        concatenated = np.concatenate(
            sorted(items, key=lambda indiv: archive[indiv[0]].get_estimation('pessimistic')), axis=0
        )
        k = convex_limit(concatenated)
        k = min(len(archive) // 4, min(k, int(len(archive) / 1.1 ** dimension)))
    else:
        raise ValueError(f'{method} not implemented as a method for choosing k in avg_of_k_best.')
    k = 1 if k < 1 else int(k)
    first_k_individuals: List[Any] = sorted(
        items, key=lambda indiv: archive[indiv[0]].get_estimation('pessimistic')
    )[:k]
    assert len(first_k_individuals) == k
    return np.array(sum((p_arr[0] for p_arr in first_k_individuals)) / k)


class OneShotOptimizer(base.Optimizer):
    one_shot = True

    def _internal_ask_candidate(self) -> p.Parameter:
        out: p.Parameter = self.parametrization.spawn_child()
        with p.helpers.deterministic_sampling(out):
            out.set_standardized_data(self._internal_ask())
        return out


class _RandomSearch(OneShotOptimizer):
    def __init__(
        self,
        parametrization: p.Parameter,
        budget: Optional[int] = None,
        num_workers: int = 1,
        middle_point: bool = False,
        stupid: bool = False,
        opposition_mode: Optional[str] = None,
        sampler: str = 'parametrization',
        scale: Union[float, str] = 1.0,
        recommendation_rule: str = 'pessimistic',
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        assert opposition_mode is None or opposition_mode in ['quasi', 'opposite', 'special']
        assert isinstance(scale, (int, float)) or scale in ['auto', 'random', 'autotune']
        self.middle_point: bool = middle_point
        self.opposition_mode: Optional[str] = opposition_mode
        self.stupid: bool = stupid
        self.recommendation_rule: str = recommendation_rule
        self.scale: Union[float, str] = scale
        self.sampler: str = sampler
        self._opposable_data: Optional[np.ndarray] = None
        self._no_hypervolume: bool = True

    def _internal_ask(self) -> np.ndarray:
        mode: Optional[str] = self.opposition_mode
        if self._opposable_data is not None and mode is not None:
            data: np.ndarray = self._opposable_data
            if mode == 'quasi':
                factor = self._rng.uniform(0.0, 1.0)
            elif mode == 'special':
                factor = np.exp(-self._rng.uniform(0.0, 5.0))
            else:
                factor = 1.0
            data = data * -(factor)
            self._opposable_data = None
            return data
        if self.middle_point and (not self._num_ask):
            self._opposable_data = np.zeros(self.dimension)
            return self._opposable_data
        scale: Union[float, str] = self.scale
        if isinstance(scale, str) and scale == 'auto':
            assert self.budget is not None
            scale = (1 + np.log(self.budget)) / (4 * np.log(self.dimension))
        if isinstance(scale, str) and scale == 'autotune':
            assert self.budget is not None
            scale = np.sqrt(np.log(self.budget) / self.dimension)
        if isinstance(scale, str) and scale == 'random':
            scale = np.exp(self._rng.normal(0.0, 1.0) - 2.0) / np.sqrt(self.dimension)
        if self.sampler == 'gaussian':
            point: np.ndarray = self._rng.normal(0, 1, self.dimension)
        elif self.sampler == 'cauchy':
            point = self._rng.standard_cauchy(self.dimension)
        elif self.sampler == 'parametrization':
            point = self.parametrization.sample().get_standardized_data(reference=self.parametrization)
        else:
            raise ValueError(f'Unknown sampler {self.sampler}')
        self._opposable_data = scale * point
        return self._opposable_data

    def _internal_provide_recommendation(self) -> Optional[np.ndarray]:
        if self.stupid:
            return self._internal_ask()
        elif self.archive:
            if self.recommendation_rule == 'average_of_best':
                return avg_of_k_best(self.archive, 'dimfourth')
            if self.recommendation_rule == 'average_of_exp_best':
                return avg_of_k_best(self.archive, 'exp')
            if self.recommendation_rule == 'average_of_hull_best':
                return avg_of_k_best(self.archive, 'hull')
        return None


class RandomSearchMaker(base.ConfiguredOptimizer):
    """Provides random suggestions.

    Parameters
    ----------
    stupid: bool
        Provides a random recommendation instead of the best point so far (for baseline)
    middle_point: bool
        enforces that the first suggested point (ask) is zero.
    opposition_mode: str or None
        symmetrizes exploration wrt the center: (e.g. https://ieeexplore.ieee.org/document/4424748)
             - full symmetry if "opposite"
             - random * symmetric if "quasi"
    sampler: str
        - parametrization: uses the default sample() method of the parametrization, which samples uniformly
          between bounds and a Gaussian otherwise
        - gaussian: uses a Gaussian distribution
        - cauchy: uses a Cauchy distribution
        use a Cauchy distribution instead of Gaussian distribution
    scale: float or "random"
        scalar for multiplying the suggested point values, or string:
         - "random": uses a randomized pattern for the scale.
         - "auto": scales in function of dimension and budget (version 1: sigma = (1+log(budget)) / (4log(dimension)) )
         - "autotune": scales in function of dimension and budget (version 2: sigma = sqrt(log(budget) / dimension) )
    recommendation_rule: str
        "average_of_best" or "pessimistic" or "average_of_exp_best"; "pessimistic" is
        the default and implies selecting the pessimistic best.
    """
    one_shot = True

    def __init__(
        self,
        *,
        middle_point: bool = False,
        stupid: bool = False,
        opposition_mode: Optional[str] = None,
        sampler: str = 'parametrization',
        scale: Union[float, str] = 1.0,
        recommendation_rule: str = 'pessimistic',
    ) -> None:
        assert sampler in ['gaussian', 'cauchy', 'parametrization']
        super().__init__(_RandomSearch, locals())


RandomSearch = RandomSearchMaker().set_name('RandomSearch', register=True)
QORandomSearch = RandomSearchMaker(opposition_mode='quasi').set_name('QORandomSearch', register=True)
ORandomSearch = RandomSearchMaker(opposition_mode='opposite').set_name('ORandomSearch', register=True)
RandomSearchPlusMiddlePoint = RandomSearchMaker(middle_point=True).set_name('RandomSearchPlusMiddlePoint', register=True)


class _SamplingSearch(OneShotOptimizer):
    def __init__(
        self,
        parametrization: p.Parameter,
        budget: Optional[int] = None,
        num_workers: int = 1,
        sampler: str = "Halton",
        scrambled: bool = False,
        middle_point: bool = False,
        opposition_mode: Optional[str] = None,
        cauchy: bool = False,
        autorescale: Union[bool, str] = False,
        scale: Union[float, str] = 1.0,
        rescaled: bool = False,
        recommendation_rule: str = "pessimistic",
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self._sampler_instance: Optional[Any] = None
        self._rescaler: Optional[Any] = None
        self._opposable_data: Optional[np.ndarray] = None
        self._sampler: str = sampler
        self.opposition_mode: Optional[str] = opposition_mode
        self.middle_point: bool = middle_point
        self.scrambled: bool = scrambled
        self.cauchy: bool = cauchy
        self.autorescale: Union[bool, str] = autorescale
        self.scale: Union[float, str] = scale
        self.rescaled: bool = rescaled
        self.recommendation_rule: str = recommendation_rule
        self._no_hypervolume: bool = True
        self._normalizer: p.helpers.Normalizer = p.helpers.Normalizer(self.parametrization)

    @property
    def sampler(self) -> Callable[[], np.ndarray]:
        if self._sampler_instance is None:
            budget: Optional[int] = None if self.budget is None else self.budget - self.middle_point
            samplers: dict = {
                "Halton": sequences.HaltonSampler,
                "Hammersley": sequences.HammersleySampler,
                "LHS": sequences.LHSSampler,
                "Random": sequences.RandomSampler,
            }
            internal_budget: Optional[int] = (
                (budget + 1) // 2 if budget is not None and self.opposition_mode in ['quasi', 'special', 'opposite'] else budget
            )
            self._sampler_instance = samplers[self._sampler](self.dimension, internal_budget, scrambling=self.scrambled, random_state=self._rng)
            assert self._sampler_instance is not None
            if self.rescaled:
                self._rescaler = sequences.Rescaler(self.sampler)
                self._sampler_instance.reinitialize()
        return self._sampler_instance

    def _internal_ask(self) -> np.ndarray:
        if self.middle_point and (not self._num_ask):
            return np.zeros(self.dimension)
        mode: Optional[str] = self.opposition_mode
        if self._opposable_data is not None and mode is not None:
            data: np.ndarray = self._opposable_data
            if mode == 'quasi':
                factor = self._rng.uniform(0.0, 1.0)
            elif mode == 'special':
                factor = np.exp(-self._rng.uniform(0.0, 5.0))
            else:
                factor = 1.0
            data = data * -(factor)
            self._opposable_data = None
            return data
        sample: np.ndarray = self.sampler()
        if self._rescaler is not None:
            sample = self._rescaler.apply(sample)
        if self.autorescale is True or self.autorescale == 'auto':
            assert self.budget is not None
            self.scale = (1 + np.log(self.budget)) / (4 * np.log(self.dimension))
        if self.autorescale == 'autotune':
            assert self.budget is not None
            self.scale = np.sqrt(np.log(self.budget) / self.dimension)
        transf: trans.CumulativeDensity = trans.CumulativeDensity(
            0, 1, scale=self.scale, density='cauchy' if self.cauchy else 'gaussian'
        )
        self._normalizer.unbounded_transform = transf
        self._opposable_data = self._normalizer.backward(sample)
        assert self._opposable_data is not None
        return self._opposable_data

    def _internal_provide_recommendation(self) -> Optional[np.ndarray]:
        if self.archive and self.recommendation_rule == 'average_of_best':
            return avg_of_k_best(self.archive)
        return None


class SamplingSearch(base.ConfiguredOptimizer):
    """This is a one-shot optimization method, hopefully better than random search
    by ensuring more uniformity.

    Parameters
    ----------
    sampler: str
        Choice of the sampler among "Halton", "Hammersley" and "LHS".
    scrambled: bool
        Adds scrambling to the search; much better in high dimension and rarely worse
        than the original search.
    middle_point: bool
        enforces that the first suggested point (ask) is zero.
    cauchy: bool
        use Cauchy inverse distribution instead of Gaussian when fitting points to real space
        (instead of box).
    scale: float or "random"
        scalar for multiplying the suggested point values.
    rescaled: bool or str
        rescales the sampling pattern to reach the boundaries and/or applies automatic rescaling.
    recommendation_rule: str
        "average_of_best" or "pessimistic"; "pessimistic" is the default and implies selecting the pessimistic best.
    """
    one_shot = True

    def __init__(
        self,
        *,
        sampler: str = 'Halton',
        scrambled: bool = False,
        middle_point: bool = False,
        opposition_mode: Optional[str] = None,
        cauchy: bool = False,
        autorescale: Union[bool, str] = False,
        scale: Union[float, str] = 1.0,
        rescaled: bool = False,
        recommendation_rule: str = "pessimistic",
    ) -> None:
        super().__init__(_SamplingSearch, locals())


MetaRecentering = SamplingSearch(cauchy=False, autorescale=True, sampler='Hammersley', scrambled=True).set_name(
    'MetaRecentering', register=True
)
MetaTuneRecentering = SamplingSearch(cauchy=False, autorescale='autotune', sampler='Hammersley', scrambled=True).set_name(
    'MetaTuneRecentering', register=True
)
HullAvgMetaTuneRecentering = SamplingSearch(
    cauchy=False, autorescale='autotune', sampler='Hammersley', scrambled=True, recommendation_rule='average_of_hull_best'
).set_name('HullAvgMetaTuneRecentering', register=True)
HullAvgMetaRecentering = SamplingSearch(
    cauchy=False, autorescale=True, sampler='Hammersley', scrambled=True, recommendation_rule='average_of_hull_best'
).set_name('HullAvgMetaRecentering', register=True)
AvgMetaRecenteringNoHull = SamplingSearch(
    cauchy=False, autorescale=True, sampler='Hammersley', scrambled=True, recommendation_rule='average_of_exp_best'
).set_name('AvgMetaRecenteringNoHull', register=True)
HaltonSearch = SamplingSearch().set_name('HaltonSearch', register=True)
HaltonSearchPlusMiddlePoint = SamplingSearch(middle_point=True).set_name('HaltonSearchPlusMiddlePoint', register=True)
LargeHaltonSearch = SamplingSearch(scale=100.0).set_name('LargeHaltonSearch', register=True)
ScrHaltonSearch = SamplingSearch(scrambled=True).set_name('ScrHaltonSearch', register=True)
ScrHaltonSearchPlusMiddlePoint = SamplingSearch(middle_point=True, scrambled=True).set_name(
    'ScrHaltonSearchPlusMiddlePoint', register=True
)
HammersleySearch = SamplingSearch(sampler='Hammersley').set_name('HammersleySearch', register=True)
HammersleySearchPlusMiddlePoint = SamplingSearch(sampler='Hammersley', middle_point=True).set_name(
    'HammersleySearchPlusMiddlePoint', register=True
)
ScrHammersleySearchPlusMiddlePoint = SamplingSearch(scrambled=True, sampler='Hammersley', middle_point=True).set_name(
    'ScrHammersleySearchPlusMiddlePoint', register=True
)
ScrHammersleySearch = SamplingSearch(sampler='Hammersley', scrambled=True).set_name('ScrHammersleySearch', register=True)
QOScrHammersleySearch = SamplingSearch(sampler='Hammersley', scrambled=True, opposition_mode='quasi').set_name(
    'QOScrHammersleySearch', register=True
)
OScrHammersleySearch = SamplingSearch(sampler='Hammersley', scrambled=True, opposition_mode='opposite').set_name(
    'OScrHammersleySearch', register=True
)
CauchyScrHammersleySearch = SamplingSearch(cauchy=True, sampler='Hammersley', scrambled=True).set_name(
    'CauchyScrHammersleySearch', register=True
)
LHSSearch = SamplingSearch(sampler='LHS').set_name('LHSSearch', register=True)
CauchyLHSSearch = SamplingSearch(sampler='LHS', cauchy=True).set_name('CauchyLHSSearch', register=True)