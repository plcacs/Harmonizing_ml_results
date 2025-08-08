import hashlib
import itertools
import numpy as np
import nevergrad as ng
from nevergrad.common import tools
import nevergrad.common.typing as tp
from .base import ExperimentFunction
from .pbt import PBT as PBT
from . import utils
from . import corefuncs

class ArtificialVariable:
    def __init__(self, dimension: int, num_blocks: int, block_dimension: int, translation_factor: float, rotation: bool, hashing: bool, only_index_transform: bool, random_state: np.random.RandomState, expo: float):
        self._dimension: int = dimension
        self._transforms: tp.List[utils.Transform] = []
        self.rotation: bool = rotation
        self.translation_factor: float = translation_factor
        self.num_blocks: int = num_blocks
        self.block_dimension: int = block_dimension
        self.only_index_transform: bool = only_index_transform
        self.hashing: bool = hashing
        self.dimension: int = self._dimension
        self.random_state: np.random.RandomState = random_state
        self.expo: float = expo

    def _initialize(self) -> None:
        ...

    def process(self, data: np.ndarray, deterministic: bool = True) -> np.ndarray:
        ...

    def _short_repr(self) -> str:
        ...

class ArtificialFunction(ExperimentFunction):
    def __init__(self, name: str, block_dimension: int, num_blocks: int = 1, useless_variables: int = 0, noise_level: float = 0, noise_dissymmetry: bool = False, rotation: bool = False, translation_factor: float = 1.0, hashing: bool = False, aggregator: str = 'max', split: bool = False, bounded: bool = False, expo: float = 1.0, zero_pen: bool = False):
        ...

    @property
    def dimension(self) -> int:
        ...

    @staticmethod
    def list_sorted_function_names() -> tp.List[str]:
        ...

    def _transform(self, x: np.ndarray) -> np.ndarray:
        ...

    def function_from_transform(self, x: np.ndarray) -> float:
        ...

    def evaluation_function(self, *recommendations: tp.Any) -> float:
        ...

    def noisy_function(self, *x: np.ndarray) -> float:
        ...

    def compute_pseudotime(self, input_parameter: tp.Tuple[tp.Tuple[np.ndarray, ...], tp.Dict[str, tp.Any]], loss: float) -> float:
        ...

class FarOptimumFunction(ExperimentFunction):
    def __init__(self, independent_sigma: bool = True, mutable_sigma: bool = True, multiobjective: bool = False, recombination: str = 'crossover', optimum: tp.Tuple[float, float] = (80, 100)):
        ...

    def evaluation_function(self, *recommendations: tp.Any) -> float:
        ...

    @classmethod
    def itercases(cls) -> tp.Iterator['FarOptimumFunction']:
        ...
