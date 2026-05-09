import functools
import math
import warnings
import weakref
import numpy as np
from scipy import optimize as scipyoptimize
import nevergrad.common.typing as tp
from nevergrad.parametrization import parameter as p
from nevergrad.common import errors
from . import base
from .base import IntOrParameter
from . import recaster

class _NonObjectMinimizeBase(recaster.SequentialRecastOptimizer):
    def __init__(self, parametrization: p.Parameter, budget: tp.Optional[int] = None, num_workers: int = 1, *, method: str, random_restart: bool = False) -> None:
        ...

    def get_optimization_function(self) -> callable:
        ...

    @staticmethod
    def _optimization_function(weakself: weakref.proxy[_NonObjectMinimizeBase], objective_function: callable) -> np.ndarray:
        ...

class NonObjectOptimizer(base.ConfiguredOptimizer):
    ...

    def __init__(self, *, method: str, random_restart: bool = False) -> None:
        ...

class _PymooMinimizeBase(recaster.SequentialRecastOptimizer):
    def __init__(self, parametrization: p.Parameter, budget: tp.Optional[int] = None, num_workers: int = 1, *, algorithm: str) -> None:
        ...

    def get_optimization_function(self) -> callable:
        ...

    @staticmethod
    def _optimization_function(weakself: weakref.proxy[_PymooMinimizeBase], objective_function: callable) -> None:
        ...

class Pymoo(base.ConfiguredOptimizer):
    ...

    def __init__(self, *, algorithm: str) -> None:
        ...

class _PymooBatchMinimizeBase(recaster.BatchRecastOptimizer):
    def __init__(self, parametrization: p.Parameter, budget: tp.Optional[int] = None, num_workers: int = 1, *, algorithm: str) -> None:
        ...

    def get_optimization_function(self) -> callable:
        ...

    @staticmethod
    def _optimization_function(weakself: weakref.proxy[_PymooBatchMinimizeBase], objective_function: callable) -> None:
        ...

class PymooBatch(base.ConfiguredOptimizer):
    ...

    def __init__(self, *, algorithm: str) -> None:
        ...
