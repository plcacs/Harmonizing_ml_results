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

    def __init__(self, dimension: int, num_blocks: int, block_dimension: int, translation_factor: float, rotation: bool, hashing: bool, only_index_transform: bool, random_state: np.random.RandomState, expo: float) -> None:
        self._dimension: int = dimension
        self._transforms: list = []
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
        """Delayed initialization of the transforms to avoid slowing down the instance creation
        (makes unit testing much faster).
        This functions creates the random transform used upon each block (translation + optional rotation).
        """
        indices: list = self.random_state.choice(self._dimension, self.block_dimension * self.num_blocks, replace=False).tolist()
        indices.sort()
        for transform_inds in tools.grouper(indices, n=self.block_dimension):
            self._transforms.append(utils.Transform(transform_inds, translation_factor=self.translation_factor, rotation=self.rotation, random_state=self.random_state, expo=self.expo))

    def process(self, data: np.ndarray, deterministic: bool = True) -> np.ndarray:
        if not self._transforms:
            self._initialize()
        if self.hashing:
            data2: np.ndarray = np.array(data, copy=True)
            for i, y in enumerate(data):
                self.random_state.seed(int(hashlib.md5(str(y).encode()).hexdigest(), 16) % 500000)
                data2[i] = self.random_state.normal(0.0, 1.0)
            data = data2
        data = np.asarray(data)
        output: list = []
        for transform in self._transforms:
            output.append(data[transform.indices] if self.only_index_transform else transform(data))
        return np.array(output)

    def _short_repr(self) -> str:
        return 'Photonics'

class ArtificialFunction(ExperimentFunction):
    """Artificial function object. This allows the creation of functions with different
    dimension and structure to be used for benchmarking in many different settings.

    Parameters
    ----------
    name: str
        name of the underlying function to use (like "sphere" for instance). If a wrong
        name is provided, an error is raised with all existing names.
    block_dimension: int
        the dimension on which the underlying function will be applied.
    num_blocks: int
        the number of blocks of size "block_dimension" on which the underlying function
        will be applied. The number of useful dimension is therefore num_blocks * core_dimension
    useless_variables: int
        the number of additional variables which have no impact on the core function.
        The full dimension of the function is therefore useless_variables + num_blocks * core_dimension
    noise_level: float
        noise level for the additive noise: noise_level * N(0, 1, size=1) * [f(x + N(0, 1, size=dim)) - f(x)]
    noise_dissymmetry: bool
        True if we dissymmetrize the model of noise
    rotation: bool
        whether the block space should be rotated (random rotation)
    hashing: bool
        whether the input data should be hashed. In this case, the function expects an array of size 1 with
        string as element.
    aggregator: str
        how to aggregate the multiple block outputs
    bounded: bool
        bound the search domain to [-5,5]

    Example
    -------
    >>> func = ArtificialFunction("sphere", 5, noise_level=.1)
    >>> x = [1, 2, 1, 0, .5]
    >>> func(x)  # returns a float
    >>> func(x)  # returns a different float since the function is noisy
    >>> func.oracle_call(x)   # returns a float
    >>> func.oracle_call(x)   # returns the same float (no noise for oracles + sphere function is deterministic)
    >>> func2 = ArtificialFunction("sphere", 5, noise_level=.1)
    >>> func2.oracle_call(x)   # returns a different float than before, because a random translation is applied

    Note
    ----
    - The full dimension of the function is available through the dimension attribute.
      Its value is useless_variables + num_blocks * block_dimension
    - The blocks are chosen with random sorted indices (blocks do not overlap)
    - A random translation is always applied to the function at initialization, so that
      instantiating twice the functions will give 2 different functions (unless you use
      seeding)
    - the noise formula is: noise_level * N(0, 1) * (f(x + N(0, 1)) - f(x))
    """

    def __init__(self, name: str, block_dimension: int, num_blocks: int = 1, useless_variables: int = 0, noise_level: float = 0, 
                 noise_dissymmetry: bool = False, rotation: bool = False, translation_factor: float = 1.0, hashing: bool = False, 
                 aggregator: str = 'max', split: bool = False, bounded: bool = False, expo: float = 1.0, zero_pen: bool = False) -> None:
        self.name: str = name
        self.expo: float = expo
        self.translation_factor: float = translation_factor
        self.zero_pen: bool = zero_pen
        self.constraint_violation: list = []
        self._parameters: dict = {x: y for x, y in locals().items() if x not in ['__class__', 'self']}
        assert noise_level >= 0, 'Noise level must be greater or equal to 0'
        if not all((isinstance(x, bool) for x in [noise_dissymmetry, hashing, rotation])):
            raise TypeError('hashing and rotation should be bools')
        for param, mini in [('block_dimension', 1), ('num_blocks', 1), ('useless_variables', 0)]:
            value = self._parameters[param]
            if not isinstance(value, int):
                raise TypeError(f'"{param}" must be an int')
            if value < mini:
                raise ValueError(f'"{param}" must be greater or equal to {mini}')
        if not isinstance(translation_factor, (float, int)):
            raise TypeError(f'Got non-float value {translation_factor}')
        if name not in corefuncs.registry:
            available = ', '.join(self.list_sorted_function_names())
            raise ValueError(f'Unknown core function "{name}". Available names are:\n-----\n{available}')
        self._dimension: int = block_dimension * num_blocks + useless_variables
        self._func = corefuncs.registry[name]
        info = corefuncs.registry.get_info(self._parameters['name'])
        only_index_transform: bool = info.get('no_transform', False)
        array_bounds: dict = dict(upper=5, lower=-5) if bounded else {}
        if not split:
            parametrization = ng.p.Array(shape=(1,) if hashing else (self._dimension,), **array_bounds).set_name('')
        else:
            assert not hashing
            assert not useless_variables
            arrays = [ng.p.Array(shape=(block_dimension,), **array_bounds) for _ in range(num_blocks)]
            parametrization = ng.p.Instrumentation(*arrays)
            parametrization.set_name('split')
        if noise_level > 0:
            parametrization.function.deterministic = False
        super().__init__(self.noisy_function, parametrization)
        self.transform_var = ArtificialVariable(dimension=self._dimension, num_blocks=num_blocks, block_dimension=block_dimension, translation_factor=translation_factor, rotation=rotation, hashing=hashing, only_index_transform=only_index_transform, random_state=self._parametrization.random_state, expo=self.expo)
        self._aggregator = {'max': np.max, 'mean': np.mean, 'sum': np.sum}[aggregator]
        info = corefuncs.registry.get_info(self._parameters['name'])
        self.add_descriptors(useful_dimensions=block_dimension * num_blocks, discrete=any((x in name for x in ['onemax', 'leadingones', 'jump'])))

    @property
    def dimension(self) -> int:
        return self._dimension

    @staticmethod
    def list_sorted_function_names() -> list:
        """Returns a sorted list of function names that can be used for the blocks"""
        return sorted(corefuncs.registry)

    def _transform(self, x: np.ndarray) -> np.ndarray:
        data = self.transform_var.process(x)
        return np.array(data)

    def function_from_transform(self, x: np.ndarray) -> float:
        """Implements the call of the function.
        Under the hood, __call__ delegates to oracle_call + add some noise if noise_level > 0.
        """
        results: list = []
        for block in x:
            results.append(self._func(block))
        try:
            val: float = float(self._aggregator(results))
            if self.zero_pen:
                val += 1000.0 * max(self.translation_factor / (1e-07 + self.translation_factor + np.linalg.norm(x.flatten())) - 0.75, 0.0)
            return val
        except OverflowError:
            return float('inf')

    def evaluation_function(self, *recommendations: tp.Any) -> float:
        """Implements the call of the function.
        Under the hood, __call__ delegates to oracle_call + add some noise if noise_level > 0.
        """
        assert len(recommendations) == 1, 'Should not be a pareto set for a singleobjective function'
        assert not recommendations[0].kwargs
        data: np.ndarray = np.concatenate(recommendations[0].args, axis=0)
        data = self._transform(data)
        return self.function_from_transform(data)

    def noisy_function(self, *x: np.ndarray) -> float:
        data: np.ndarray = np.concatenate(x, axis=0)
        return _noisy_call(x=data, transf=self._transform, func=self.function_from_transform, noise_level=self._parameters['noise_level'], noise_dissymmetry=self._parameters['noise_dissymmetry'], random_state=self._parametrization.random_state)

    def compute_pseudotime(self, input_parameter: tp.Tuple[tp.Tuple, dict], loss: float) -> float:
        """Delay before returning results in steady state mode benchmarks (fake execution time)"""
        args, kwargs = input_parameter
        assert not kwargs
        if hasattr(self._func, 'compute_pseudotime'):
            data = self._transform(np.concatenate(args, axis=0))
            total: float = 0.0
            for block in data:
                total += self._func.compute_pseudotime(((block,), {}), loss)
            return total
        return 1.0

def _noisy_call(x: np.ndarray, transf: tp.Callable[[np.ndarray], np.ndarray], func: tp.Callable[[np.ndarray], float], 
                noise_level: float, noise_dissymmetry: bool, random_state: np.random.RandomState) -> float:
    x_transf: np.ndarray = transf(x)
    fx: float = func(x_transf)
    noise: float = 0.0
    if noise_level:
        if not noise_dissymmetry or x_transf.ravel()[0] <= 0:
            side_point: np.ndarray = transf(x + random_state.normal(0, 1, size=len(x)))
            if noise_dissymmetry:
                noise_level *= 1.0 + x_transf.ravel()[0] * 100.0
            noise = noise_level * random_state.normal(0, 1) * (func(side_point) - fx)
    return fx + noise

class FarOptimumFunction(ExperimentFunction):
    """Very simple 2D norm-1 function with optimal value at (x_optimum, 100)"""

    def __init__(self, independent_sigma: bool = True, mutable_sigma: bool = True, multiobjective: bool = False, 
                 recombination: str = 'crossover', optimum: tp.Tuple[float, float] = (80, 100)) -> None:
        assert recombination in ('crossover', 'average')
        self._optimum: np.ndarray = np.array(optimum, dtype=float)
        parametrization = ng.p.Array(shape=(2,), mutable_sigma=mutable_sigma)
        init: np.ndarray = np.array([1.0, 1.0] if independent_sigma else [1.0], dtype=float)
        sigma = ng.p.Array(init=init).set_mutation(exponent=2.0) if mutable_sigma else init
        parametrization.set_mutation(sigma=sigma)
        if recombination == 'crossover':
            parametrization = ng.ops.mutations.Crossover()(parametrization)
        self.multiobjective_upper_bounds: tp.Optional[np.ndarray] = np.array(2 * self._optimum) if multiobjective else None
        super().__init__(self._multifunc if multiobjective else self._monofunc, parametrization.set_name(''))

    def _multifunc(self, x: np.ndarray) -> np.ndarray:
        return np.abs(x - self._optimum)

    def _monofunc(self, x: np.ndarray) -> float:
        return float(np.sum(self._multifunc(x)))

    def evaluation_function(self, *recommendations: tp.Any) -> float:
        return min((self._monofunc(x.args[0]) for x in recommendations))

    @classmethod
    def itercases(cls) -> tp.Iterator['FarOptimumFunction']:
        options: dict = dict(independent_sigma=[True, False], mutable_sigma=[True, False], multiobjective=[True, False], recombination=['average', 'crossover'], optimum=[(0.8, 1), (80, 100), (0.8, 100)])
        keys: list = sorted(options)
        select = itertools.product(*(options[k] for k in keys))
        cases = (dict(zip(keys, s)) for s in select)
        return (cls(**c) for c in cases)
