import typing as tp
import numpy as np
import nevergrad as ng
from nevergrad.ops import mutations
from . import photonics
from .. import base

def _make_parametrization(name: tp.Str, dimension: int, 
                          bounding_method: tp.Str = 'bouncing', 
                          rolling: bool = False, 
                          as_tuple: bool = False) -> ng.p.Instrumentation:
    ...

class Photonics(base.ExperimentFunction):
    def __init__(self, name: tp.Str, dimension: int, 
                 bounding_method: tp.Str = 'clipping', 
                 rolling: bool = False, 
                 as_tuple: bool = False):
        ...

    def to_array(self, *args: np.ndarray, **kwargs: tp.Dict) -> np.ndarray:
        ...

    def evaluation_function(self, *recommendations: tp.Tuple[ng.p.Recommendation]) -> float:
        ...

    def _compute(self, *args: tp.Any, **kwargs: tp.Dict) -> float:
        ...
