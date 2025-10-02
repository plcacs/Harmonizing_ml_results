"""Samplers in [0,1]^d.
"""
import numpy as np
from numpy.random import RandomState
import nevergrad.common.typing as tp
from nevergrad.common.decorators import Registry
from typing import Optional, Iterator, List, Generator
import math

samplers = Registry()

def _get_first_primes(num: int) -> np.ndarray:
    """Computes the first num primes"""
    if num < 6:
        return np.array([2, 3, 5, 7, 11][:num], dtype=int)
    is_prime = np.ones(int(1 + num * (math.log(math.log(num)) + math.log(num))), dtype=bool)
    is_prime[[0, 1]] = 0
    for index in range(1 + int(np.sqrt(len(is_prime)))):
        if is_prime[index]:
            is_prime[index + index::index] = False
    primes = np.where(is_prime)[0]
    if len(primes) < num:
        raise RuntimeError(f'There is an error on the upper bound of the primes for num={num}')
    return primes[:num]

class Sampler:
    
    def __init__(
        self, 
        dimension: int, 
        budget: Optional[int] = None, 
        random_state: Optional[np.random.RandomState] = None
    ) -> None:
        if random_state is None:
            random_state = np.random.RandomState(np.random.randint(2 ** 32, dtype=np.uint32))
        self.random_state: np.random.RandomState = random_state
        self.dimension: int = dimension
        self.budget: Optional[int] = budget
        self.index: int = 0

    def _internal_sampler(self) -> np.ndarray:
        raise NotImplementedError('Missing sampling function! which is quite necessary for a sampler.')

    def __call__(self) -> np.ndarray:
        assert self.budget is None or self.index < self.budget, 'Over the budget (reinitialize if you want to start over)'
        sample = self._internal_sampler()
        self.index += 1
        return sample

    def __iter__(self) -> Iterator[np.ndarray]:
        assert self.index == 0, 'Reinitialize before iterating again'
        assert self.budget is not None, 'Iterable does not work if budget is not specified'
        return (self() for _ in range(self.budget))

    def reinitialize(self) -> None:
        self.index = 0

    def draw(self) -> None:
        """Simple ASCII drawing of the sampling pattern (for testing/visualization purpose only)"""
        sampler = self.__class__(self.dimension, budget=self.budget)
        assert sampler.budget is not None
        sample: List[np.ndarray] = [sampler() for _ in range(sampler.budget)]
        for i in range(sampler.dimension):
            for j in range(i + 1, sampler.dimension):
                print('plotting coordinates ' + str(i) + ',' + str(j))
                tab: List[List[str]] = [['.' for _ in range(80)] for _ in range(20)]
                for s in sample:
                    x = int(s[i] * 20)
                    y = int(s[j] * 80)
                    if 0 <= x < 20 and 0 <= y < 80:
                        tab[x][y] = '*'
                for t in tab:
                    print(''.join(t))

@samplers.register
class LHSSampler(Sampler):
    
    def __init__(
        self, 
        dimension: int, 
        budget: int, 
        scrambling: bool = False, 
        random_state: Optional[np.random.RandomState] = None
    ) -> None:
        if scrambling:
            raise ValueError('LHSSampler does not support scrambling')
        super().__init__(dimension, budget, random_state=random_state)
        self.permutations: np.ndarray = np.zeros((dimension, budget), dtype=int)
        for k in range(dimension):
            self.permutations[k] = self.random_state.permutation(budget)
        self.seed: int = self.random_state.randint(2 ** 32, dtype=np.uint32)
        self.randg: np.random.RandomState = np.random.RandomState(self.seed)

    def reinitialize(self) -> None:
        super().reinitialize()
        self.randg = np.random.RandomState(self.seed)

    def _internal_sampler(self) -> np.ndarray:
        x: List[int] = self.permutations[:, self.index].tolist()
        assert self.budget is not None
        return (np.array(x) + self.randg.uniform(size=self.dimension)) / float(self.budget)

@samplers.register
class RandomSampler(Sampler):
    
    def __init__(
        self, 
        dimension: int, 
        budget: int, 
        scrambling: bool = False, 
        random_state: Optional[np.random.RandomState] = None
    ) -> None:
        if scrambling:
            raise ValueError('RandomSampler does not support scrambling')
        super().__init__(dimension, budget, random_state=random_state)

    def _internal_sampler(self) -> np.ndarray:
        return self.random_state.uniform(0, 1, self.dimension)

class HaltonPermutationGenerator:
    """Provides a light-memory access to a possibly huge list of permutations
    (at the cost of being slightly slower)
    """

    def __init__(
        self, 
        dimension: int, 
        scrambling: bool = False, 
        random_state: Optional[np.random.RandomState] = None
    ) -> None:
        if random_state is None:
            random_state = np.random.RandomState(np.random.randint(2 ** 32, dtype=np.uint32))
        self.dimension: int = dimension
        self.scrambling: bool = scrambling
        self.primes: List[int] = _get_first_primes(dimension).tolist()
        self.seed: int = random_state.randint(2 ** 32, dtype=np.uint32)
        self.fulllist: np.ndarray = np.arange(self.primes[-1]) if self.primes else np.array([], dtype=int)

    def get_permutations_generator(self) -> Generator[np.ndarray, None, None]:
        if self.scrambling:
            randgen: np.random.RandomState = np.random.RandomState(seed=self.seed)
            for p in self.primes:
                if p <= 1:
                    yield np.array([0], dtype=int)
                else:
                    yield np.concatenate(([0], randgen.choice(self.fulllist[1:p], p - 1, replace=False)), axis=0)
        else:
            for p in self.primes:
                yield self.fulllist[:p]

@samplers.register
class HaltonSampler(Sampler):
    
    def __init__(
        self, 
        dimension: int, 
        budget: Optional[int] = None, 
        scrambling: bool = False, 
        random_state: Optional[np.random.RandomState] = None
    ) -> None:
        super().__init__(dimension, budget, random_state=random_state)
        self.permgen: HaltonPermutationGenerator = HaltonPermutationGenerator(dimension, scrambling, random_state=random_state)

    def vdc(self, n: int, permut: np.ndarray) -> float:
        base: int = len(permut)
        vdc: float = 0.0
        denom: int = 1
        n += 1
        while n:
            denom *= base
            n, remainder = divmod(n, base)
            remainder = permut[remainder]
            vdc += float(remainder) / float(denom)
        return vdc

    def _internal_sampler(self) -> np.ndarray:
        sample: List[float] = [self.vdc(self.index, sigma) for sigma in self.permgen.get_permutations_generator()]
        return np.array(sample)

@samplers.register
class HammersleySampler(HaltonSampler):
    
    def __init__(
        self, 
        dimension: int, 
        budget: int, 
        scrambling: bool = False, 
        random_state: Optional[np.random.RandomState] = None
    ) -> None:
        assert budget is not None
        super().__init__(dimension - 1, budget, scrambling, random_state=random_state)
        self.budget: int = budget  # Ensure budget is int

    def _internal_sampler(self) -> np.ndarray:
        assert self.budget is not None
        first_component: float = (self.index + 0.5) / float(self.budget)
        rest_components: np.ndarray = super()._internal_sampler()
        return np.concatenate(([first_component], rest_components))

class Rescaler:
    
    def __init__(self, points: Iterator[np.ndarray]) -> None:
        iterp = iter(points)
        first_point: np.ndarray = np.asarray(next(iterp))
        self.sample_mins: np.ndarray = first_point
        self.sample_maxs: np.ndarray = first_point
        for point in iterp:
            self.sample_mins = np.minimum(self.sample_mins, point)
            self.sample_maxs = np.maximum(self.sample_maxs, point)
        epsilon_candidates: List[float] = list(self.sample_mins) + [1 - s for s in self.sample_maxs] + [1e-15]
        self.epsilon: float = min(epsilon_candidates)
        assert self.epsilon > 0.0, f'Non-positive epsilon={self.epsilon} from mins {self.sample_mins} and maxs {self.sample_maxs}'

    def apply(self, point: tp.ArrayLike) -> np.ndarray:
        point_arr: np.ndarray = np.asarray(point)
        factor: np.ndarray = (1 - 2 * self.epsilon) / (self.sample_maxs - self.sample_mins)
        return self.epsilon + factor * (point_arr - self.sample_mins)
