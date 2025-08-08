from typing import Optional, Any, Tuple, Union
import numpy as np
import eagerpy as ep
from ..devutils import atleast_kd
from ..models import Model
from ..criteria import TargetedMisclassification
from ..distances import linf
from .base import FixedEpsilonAttack
from .base import T
from .base import get_channel_axis
from .base import raise_if_kwargs
from .base import verify_input_bounds
import math
from .gen_attack_utils import rescale_images

class GenAttack(FixedEpsilonAttack):
    def __init__(self, *, steps: int = 1000, population: int = 10, mutation_probability: float = 0.1, mutation_range: float = 0.15, sampling_temperature: float = 0.3, channel_axis: Optional[int] = None, reduced_dims: Optional[Tuple[int, int]] = None) -> None:
    def apply_noise(self, x: ep.Tensor, noise: ep.Tensor, epsilon: float, channel_axis: Optional[int]) -> ep.Tensor:
    def choice(self, a: np.ndarray, size: int, replace: bool, p: ep.Tensor) -> np.ndarray:
    def run(self, model: Model, inputs: ep.Tensor, criterion: Any, *, epsilon: float, **kwargs: Any) -> ep.Tensor:
