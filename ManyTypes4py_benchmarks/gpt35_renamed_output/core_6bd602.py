import numpy as np
from nevergrad.parametrization import parameter as p
from .hypervolume import HypervolumeIndicator
from typing import Tuple, Optional, Union, List

AUTO_BOUND: int = 15

class HypervolumePareto:
    def __init__(self, *, upper_bounds: Optional[Union[Tuple[float], np.ndarray]] = None, auto_bound: int = AUTO_BOUND, seed: Optional[Union[int, np.random.RandomState]] = None, no_hypervolume: bool = False) -> None:
        self._auto_bound: int = 0
        self._upper_bounds: np.ndarray = np.asarray([-float('inf')]) if upper_bounds is None else np.array(upper_bounds)
        if upper_bounds is None:
            self._auto_bound = auto_bound
        self._best_volume: float = -float('Inf')
        self._hypervolume: Optional[HypervolumeIndicator] = None
        self._pareto_needs_filtering: bool = False
        self._no_hypervolume: bool = no_hypervolume
        self._pf: ParetoFront = ParetoFront(seed=seed, no_hypervolume=no_hypervolume)

    def func_kh364ugt(self) -> int:
        return self._upper_bounds.size

    def func_aohdkuwx(self) -> float:
        return self._best_volume

    def func_6qlthlwg(self, parameters: List[p.Parameter]) -> float:
        output: float = 0.0
        for param in parameters:
            output = self.add(param)
        return output

    def func_f054rh8s(self, parameter: p.Parameter) -> float:
        ...

    def func_mnjktc8z(self, parameter: p.Parameter, losses: np.ndarray) -> float:
        ...

    def func_pof5s8bw(self, size: Optional[int] = None, subset: str = 'random', subset_tentatives: int = 12) -> List[p.Parameter]:
        ...

    def func_xqnkm380(self) -> np.ndarray:
        ...

class ParetoFront:
    def __init__(self, *, seed: Optional[Union[int, np.random.RandomState]] = None, no_hypervolume: bool = False) -> None:
        ...

    def func_ztji00jq(self, parameter: p.Parameter) -> None:
        ...

    def func_clstvj4b(self) -> None:
        ...

    def func_usnsp55y(self) -> List[p.Parameter]:
        ...

    def func_frewt7o1(self, size: Optional[int] = None, subset: str = 'random', subset_tentatives: int = 12) -> List[p.Parameter]:
        ...
