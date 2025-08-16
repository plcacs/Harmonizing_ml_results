from typing import List, Callable, Dict, Type, Optional
from pathlib import Path
import re
import numpy as np
from nevergrad.common.decorators import Registry
from nevergrad.functions import ExperimentFunction
import nevergrad as ng
from . import _core

registry: Registry[Type['LsgoFunction']] = Registry[Type['LsgoFunction']]()

def read_data(name: str) -> Dict[str, np.ndarray]:
    filepaths = (Path(__file__).parent / 'cdatafiles').glob(name + '-*.txt')
    all_data: Dict[str, np.ndarray] = {}
    pattern = re.compile(name + '-(?P<tag>.+?).txt')
    for filepath in filepaths:
        match = pattern.search(filepath.name)
        assert match is not None
        tag = match.group('tag')
        all_data[tag] = np.loadtxt(str(filepath), delimiter=',', dtype=int if tag in ['s', 'p'] else float)
    if 'p' in all_data:
        all_data['p'] -= 1
    return all_data

class FunctionChunk:
    def __init__(self, transforms: List[Callable], loss: Callable):
        self.loss = loss
        self.transforms = transforms
        self._scalar = 1.0

    def _apply_transforms(self, x: np.ndarray) -> np.ndarray:
        if not self.transforms:
            return x
        y = self.transforms[0](x)
        for transf in self.transforms[1:]:
            y = transf(y)
        return y

    def __call__(self, x: np.ndarray) -> float:
        return self._scalar * self.loss(self._apply_transforms(x))

    def __rmul__(self, scalar: float) -> 'FunctionChunk':
        self._scalar *= scalar
        return self

    def __repr__(self) -> str:
        return f'{self._scalar}*FunctionChunck({self.transforms}, {self.loss}'

class LsgoFunction:
    bounds: Tuple[int, int] = (-100, 100)
    tags: List[str] = ['unimodal', 'separable', 'shifted', 'irregular']
    conditionning: float = -1.0

    def __init__(self, xopt: np.ndarray, functions: List[FunctionChunk]):
        self.xopt = xopt
        self.optimum = xopt
        self.functions = functions

    def __call__(self, x: np.ndarray) -> float:
        transformed = x - self.xopt
        return sum((f(transformed) for f in self.functions))

    @property
    def dimension(self) -> int:
        return self.xopt.size

    def instrumented(self, transform: str = 'bouncing') -> ExperimentFunction:
        param = ng.p.Array(shape=(self.dimension,)).set_bounds(self.bounds[0], self.bounds[1], method=transform)
        return ExperimentFunction(self, param.set_name(transform))

@registry.register
class ShiftedElliptic(LsgoFunction):
    number: int = 1
    bounds: Tuple[int, int] = (-100, 100)
    conditionning: float = 1000000.0
    tags: List[str] = ['unimodal', 'separable', 'shifted', 'irregular']

    def __init__(self, xopt: np.ndarray):
        transforms = [_core.irregularity]
        super().__init__(xopt, [FunctionChunk(transforms, _core.Elliptic(xopt.size))])

# Remaining classes follow the same pattern of type annotations
