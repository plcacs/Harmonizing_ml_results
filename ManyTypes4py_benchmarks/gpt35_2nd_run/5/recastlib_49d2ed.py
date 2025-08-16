from typing import Any

class _NonObjectMinimizeBase(recaster.SequentialRecastOptimizer):
    def __init__(self, parametrization: Any, budget: Any = None, num_workers: int = 1, *, method: str = 'Nelder-Mead', random_restart: bool = False) -> None:
        ...

    def _internal_tell_not_asked(self, candidate: Any, loss: Any) -> None:
        ...

    def get_optimization_function(self) -> Any:
        ...

    @staticmethod
    def _optimization_function(weakself: Any, objective_function: Any) -> Any:
        ...

class NonObjectOptimizer(base.ConfiguredOptimizer):
    def __init__(self, *, method: str = 'Nelder-Mead', random_restart: bool = False) -> None:
        ...

class _PymooMinimizeBase(recaster.SequentialRecastOptimizer):
    def __init__(self, parametrization: Any, budget: Any = None, num_workers: int = 1, *, algorithm: str) -> None:
        ...

    def get_optimization_function(self) -> Any:
        ...

    @staticmethod
    def _optimization_function(weakself: Any, objective_function: Any) -> Any:
        ...

class Pymoo(base.ConfiguredOptimizer):
    def __init__(self, *, algorithm: str) -> None:
        ...

class _PymooBatchMinimizeBase(recaster.BatchRecastOptimizer):
    def __init__(self, parametrization: Any, budget: Any = None, num_workers: int = 1, *, algorithm: str) -> None:
        ...

    def get_optimization_function(self) -> Any:
        ...

    @staticmethod
    def _optimization_function(weakself: Any, objective_function: Any) -> Any:
        ...

class PymooBatch(base.ConfiguredOptimizer):
    def __init__(self, *, algorithm: str) -> None:
        ...

def _create_pymoo_problem(optimizer: Any, objective_function: Any, elementwise: bool = True) -> Any:
    ...

PymooCMAES = Pymoo(algorithm='CMAES').set_name('PymooCMAES', register=True)
PymooBIPOP = Pymoo(algorithm='BIPOP').set_name('PymooBIPOP', register=True)
PymooNSGA2 = Pymoo(algorithm='nsga2').set_name('PymooNSGA2', register=True)
PymooBatchNSGA2 = PymooBatch(algorithm='nsga2').set_name('PymooBatchNSGA2', register=False)
pysot = NonObjectOptimizer(method='pysot').set_name('pysot', register=True)
DSbase = NonObjectOptimizer(method='DSbase').set_name('DSbase', register=True)
DS3p = NonObjectOptimizer(method='DS3p').set_name('DS3p', register=True)
DSsubspace = NonObjectOptimizer(method='DSsubspace').set_name('DSsubspace', register=True)
DSproba = NonObjectOptimizer(method='DSproba').set_name('DSproba', register=True)
