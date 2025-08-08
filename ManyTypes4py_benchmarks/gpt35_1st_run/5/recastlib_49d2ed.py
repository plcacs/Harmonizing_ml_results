from typing import Callable, List

class _NonObjectMinimizeBase(recaster.SequentialRecastOptimizer):

    def __init__(self, parametrization: tp.Parameter, budget: tp.Optional[int] = None, num_workers: int = 1, *, method: str = 'Nelder-Mead', random_restart: bool = False):
        ...

    def _internal_tell_not_asked(self, candidate: tp.Parameter, loss: float):
        ...

    def get_optimization_function(self) -> Callable[[Callable[[List[float]], float]], Callable[[], List[float]]]:
        ...

    @staticmethod
    def _optimization_function(weakself: '_NonObjectMinimizeBase', objective_function: Callable[[List[float]], float]) -> List[float]:
        ...

class NonObjectOptimizer(base.ConfiguredOptimizer):

    def __init__(self, *, method: str = 'Nelder-Mead', random_restart: bool = False):
        ...

class _PymooMinimizeBase(recaster.SequentialRecastOptimizer):

    def __init__(self, parametrization: tp.Parameter, budget: tp.Optional[int] = None, num_workers: int = 1, *, algorithm: str):
        ...

    def get_optimization_function(self) -> Callable[[Callable[[List[float]], float]], Callable[[], None]]:
        ...

    def _internal_ask_candidate(self) -> tp.Parameter:
        ...

    def _internal_tell_candidate(self, candidate: tp.Parameter, loss: float):
        ...

    def _post_loss(self, candidate: tp.Parameter, loss: float) -> List[float]:
        ...

class Pymoo(base.ConfiguredOptimizer):

    def __init__(self, *, algorithm: str):
        ...

class _PymooBatchMinimizeBase(recaster.BatchRecastOptimizer):

    def __init__(self, parametrization: tp.Parameter, budget: tp.Optional[int] = None, num_workers: int = 1, *, algorithm: str):
        ...

    def get_optimization_function(self) -> Callable[[Callable[[List[float]], float]], Callable[[], None]]:
        ...

    def _internal_ask_candidate(self) -> tp.Parameter:
        ...

    def _internal_tell_candidate(self, candidate: tp.Parameter, loss: float):
        ...

    def _post_loss(self, candidate: tp.Parameter, loss: float) -> List[float]:
        ...

class PymooBatch(base.ConfiguredOptimizer):

    def __init__(self, *, algorithm: str):
        ...

def _create_pymoo_problem(optimizer: '_PymooMinimizeBase', objective_function: Callable[[List[float]], float], elementwise: bool = True) -> '_PymooProblem':
    ...

class _PymooProblem(Base):

    def __init__(self, optimizer: '_PymooMinimizeBase', objective_function: Callable[[List[float]], float]):
        ...

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs):
        ...
