import typing as tp
import numpy as np
import nevergrad as ng
from nevergrad.ops import mutations
from . import photonics
from .. import base
ceviche: Any = photonics.ceviche

def _make_parametrization(name: str, dimension: int, bounding_method: str = 'bouncing', rolling: bool = False, as_tuple: bool = False) -> ng.p.Instrumentation:
    ...

class Photonics(base.ExperimentFunction):
    def __init__(self, name: str, dimension: int, bounding_method: str = 'clipping', rolling: bool = False, as_tuple: bool = False):
        ...

    def to_array(self, *args: np.ndarray, **kwargs: Any) -> np.ndarray:
        ...

    def evaluation_function(self, *recommendations: ng.p.Instrumentation) -> float:
        ...

    def _compute(self, *args: np.ndarray, **kwargs: Any) -> float:
        ...
