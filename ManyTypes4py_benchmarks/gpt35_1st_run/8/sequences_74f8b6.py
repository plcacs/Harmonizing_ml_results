import numpy as np
from numpy.random import RandomState
from nevergrad.common.typing import ArrayLike
from nevergrad.common.decorators import Registry
samplers: Registry = Registry()

def _get_first_primes(num: int) -> np.ndarray:
    ...

class Sampler:

    def __init__(self, dimension: int, budget: int = None, random_state: RandomState = None) -> None:
        ...

    def _internal_sampler(self) -> ArrayLike:
        ...

    def __call__(self) -> ArrayLike:
        ...

    def __iter__(self):
        ...

    def reinitialize(self) -> None:
        ...

    def draw(self) -> None:
        ...

@samplers.register
class LHSSampler(Sampler):

    def __init__(self, dimension: int, budget: int, scrambling: bool = False, random_state: RandomState = None) -> None:
        ...

    def reinitialize(self) -> None:
        ...

    def _internal_sampler(self) -> ArrayLike:
        ...

@samplers.register
class RandomSampler(Sampler):

    def __init__(self, dimension: int, budget: int, scrambling: bool = False, random_state: RandomState = None) -> None:
        ...

    def _internal_sampler(self) -> ArrayLike:
        ...

class HaltonPermutationGenerator:

    def __init__(self, dimension: int, scrambling: bool = False, random_state: RandomState = None) -> None:
        ...

    def get_permutations_generator(self):
        ...

@samplers.register
class HaltonSampler(Sampler):

    def __init__(self, dimension: int, budget: int = None, scrambling: bool = False, random_state: RandomState = None) -> None:
        ...

    def vdc(self, n: int, permut: np.ndarray) -> float:
        ...

    def _internal_sampler(self) -> ArrayLike:
        ...

@samplers.register
class HammersleySampler(HaltonSampler):

    def __init__(self, dimension: int, budget: int = None, scrambling: bool = False, random_state: RandomState = None) -> None:
        ...

    def _internal_sampler(self) -> ArrayLike:
        ...

class Rescaler:

    def __init__(self, points: ArrayLike) -> None:
        ...

    def apply(self, point: ArrayLike) -> np.ndarray:
        ...
