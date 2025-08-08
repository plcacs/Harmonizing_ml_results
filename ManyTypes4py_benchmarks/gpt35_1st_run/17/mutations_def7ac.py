import numpy as np
from nevergrad.parametrization import discretization
from . import utils

class Mutator:
    def __init__(self, random_state: np.random.RandomState) -> None:
        self.random_state = random_state

    def significantly_mutate(self, v: float, arity: int) -> float:
        ...

    def doerr_discrete_mutation(self, parent: np.ndarray, arity: int = 2) -> np.ndarray:
        ...

    def doubledoerr_discrete_mutation(self, parent: np.ndarray, max_ratio: float = 1.0, arity: int = 2) -> np.ndarray:
        ...

    def rls_mutation(self, parent: np.ndarray, arity: int = 2) -> np.ndarray:
        ...

    def portfolio_discrete_mutation(self, parent: np.ndarray, intensity: int = None, arity: int = 2) -> np.ndarray:
        ...

    def coordinatewise_mutation(self, parent: np.ndarray, velocity: float, boolean_vector: np.ndarray, arity: int) -> np.ndarray:
        ...

    def discrete_mutation(self, parent: np.ndarray, arity: int = 2) -> np.ndarray:
        ...

    def crossover(self, parent: np.ndarray, donor: np.ndarray, rotation: bool = False, crossover_type: str = 'none') -> np.ndarray:
        ...

    def get_roulette(self, archive, num: int = None) -> np.ndarray:
        ...
