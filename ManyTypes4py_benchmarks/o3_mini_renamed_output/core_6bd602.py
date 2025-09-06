#!/usr/bin/env python3
from typing import Optional, Union, Tuple, List, Any
import numpy as np
import nevergrad.common.typing as tp
from nevergrad.parametrization import parameter as p
from .hypervolume import HypervolumeIndicator

AUTO_BOUND: int = 15


class HypervolumePareto:
    def __init__(
        self,
        *,
        upper_bounds: Optional[Union[Tuple[float, ...], np.ndarray]] = None,
        auto_bound: int = AUTO_BOUND,
        seed: Optional[Union[int, np.random.RandomState]] = None,
        no_hypervolume: bool = False,
    ) -> None:
        self._auto_bound: int = 0
        self._upper_bounds: np.ndarray = (
            np.asarray([-float("inf")])
            if upper_bounds is None
            else np.array(upper_bounds)
        )
        if upper_bounds is None:
            self._auto_bound = auto_bound
        self._best_volume: float = -float("Inf")
        self._hypervolume: Optional[HypervolumeIndicator] = None
        self._pareto_needs_filtering: bool = False
        self._no_hypervolume: bool = no_hypervolume
        self._pf: ParetoFront = ParetoFront(seed=seed, no_hypervolume=no_hypervolume)

    @property
    def func_kh364ugt(self) -> int:
        return self._upper_bounds.size

    @property
    def func_aohdkuwx(self) -> float:
        return self._best_volume

    def func_6qlthlwg(self, parameters: List[p.Parameter]) -> float:
        output: float = 0.0
        for param in parameters:
            output = self.func_f054rh8s(param)
        return output

    def func_f054rh8s(self, parameter: p.Parameter) -> float:
        """
        when _no_hypervolume = False:
            Given parameters and the multiobjective loss, this computes the hypervolume
            and update the state of the function with new points if it belongs to the pareto front.
        when _no_hypervolume = True:
            Add every point to pareto front. Don't compute hypervolume. Return 0.0 since loss
            not looked at in this context.
        """
        if not isinstance(parameter, p.Parameter):
            raise TypeError(
                f"{self.__class__.__name__}.add should receive a ng.p.Parameter, but got: {parameter}."
            )
        losses: np.ndarray = parameter.losses  # type: ignore
        if not isinstance(losses, np.ndarray):
            raise TypeError(
                f"Parameter should have multivalue as losses, but parameter.losses={losses} ({type(losses)})."
            )
        if self._auto_bound > 0:
            self._auto_bound -= 1
            if (self._upper_bounds > -float("inf")).all() and (losses > self._upper_bounds).all():
                return float("inf")
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
        return self._calc_hypervolume(parameter, losses)  # type: ignore

    def _calc_hypervolume(self, parameter: p.Parameter, losses: np.ndarray) -> float:
        # This helper function is assumed to perform hypervolume calculation.
        # In this example, we delegate the computation to func_mnjktc8z.
        return self.func_mnjktc8z(parameter, losses)

    def func_mnjktc8z(self, parameter: p.Parameter, losses: np.ndarray) -> float:
        if self._hypervolume is None:
            self._hypervolume = HypervolumeIndicator(self._upper_bounds)
            self._pf._hypervolume = self._hypervolume
        new_volume: float = self._hypervolume.compute([pa.losses for pa in self._pf.get_raw()]) + 0.0
        # Including current losses for computation:
        new_volume = self._hypervolume.compute(
            [pa.losses for pa in self._pf.get_raw()] + [losses]
        )
        if new_volume > self._best_volume:
            self._best_volume = new_volume
            self._pf.add_to_pareto(parameter)
            return -new_volume
        else:
            distance_to_pareto: float = float("Inf")
            for param in self._pf.get_front():
                stored_losses: np.ndarray = param.losses  # type: ignore
                if (stored_losses <= losses).all():
                    distance_to_pareto = min(distance_to_pareto, float(min(losses - stored_losses)))
            assert distance_to_pareto >= 0
            return 0.0 if self._no_hypervolume else -new_volume + distance_to_pareto

    def func_pof5s8bw(
        self, size: Optional[int] = None, subset: str = "random", subset_tentatives: int = 12
    ) -> List[p.Parameter]:
        """
        Pareto front, as a list of Parameter. The losses can be accessed through
        parameter.losses
        """
        return self._pf.get_front(size, subset, subset_tentatives)

    def func_xqnkm380(self) -> np.ndarray:
        return np.min([p_obj.losses for p_obj in self._pf.get_raw()], axis=0)


class ParetoFront:
    def __init__(
        self, *, seed: Optional[Union[int, np.random.RandomState]] = None, no_hypervolume: bool = False
    ) -> None:
        self._pareto: List[p.Parameter] = []
        self._pareto_needs_filtering: bool = False
        self._no_hypervolume: bool = no_hypervolume
        if isinstance(seed, np.random.RandomState):
            self._rng: np.random.RandomState = seed
        else:
            self._rng = np.random.RandomState(seed)
        self._hypervolume: Optional[HypervolumeIndicator] = None

    def add_to_pareto(self, parameter: p.Parameter) -> None:
        self._pareto.append(parameter)
        self._pareto_needs_filtering = True

    def _filter_pareto_front(self) -> None:
        new_pareto: List[p.Parameter] = []
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

    def get_raw(self) -> List[p.Parameter]:
        """Retrieve current values, which may not be a Pareto front, as they have not been filtered."""
        return self._pareto

    def get_front(
        self, size: Optional[int] = None, subset: str = "random", subset_tentatives: int = 12
    ) -> List[p.Parameter]:
        """
        Pareto front, as a list of Parameter. The losses can be accessed through
        parameter.losses
        """
        if self._pareto_needs_filtering:
            self._filter_pareto_front()
        if size is None or size >= len(self._pareto):
            return self._pareto
        if subset == "random":
            return self._rng.choice(self._pareto, size, replace=False).tolist()
        tentatives: List[List[p.Parameter]] = [
            self._rng.choice(self._pareto, size, replace=False).tolist() for _ in range(subset_tentatives)
        ]
        if self._hypervolume is None and subset == "hypervolume":
            raise RuntimeError("Hypervolume subsetting not supported as hypervolume not in use")
        hypervolume = self._hypervolume  # type: ignore
        scores: List[float] = []
        for tentative in tentatives:
            if subset == "hypervolume":
                scores.append(-hypervolume.compute([pa.losses for pa in tentative]))
            else:
                score: float = 0.0
                for v in self._pareto:
                    best_score: float = float("inf") if subset != "EPS" else 0.0
                    for pa in tentative:
                        if subset == "loss-covering":
                            best_score = min(best_score, np.linalg.norm(pa.losses - v.losses))
                        elif subset == "EPS":
                            best_score = min(best_score, max(pa.losses - v.losses))
                        elif subset == "domain-covering":
                            best_score = min(
                                best_score, np.linalg.norm(pa.get_standardized_data(reference=v))
                            )
                        else:
                            raise ValueError(
                                f'Unknown subset for Pareto-Set subsampling: "{subset}"'
                            )
                    if subset == "EPS":
                        score = max(score, best_score)
                    else:
                        score += best_score ** 2
                scores.append(score)
        best_index: int = scores.index(min(scores))
        return tentatives[best_index]
