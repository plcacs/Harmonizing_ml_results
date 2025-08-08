from typing import TypeVar, Callable, Optional, Tuple, Any
from abc import ABC, abstractmethod
import copy
import eagerpy as ep
from ..types import Bounds, BoundsInput, Preprocessing
from ..devutils import atleast_kd
T = TypeVar('T')
PreprocessArgs = Tuple[Optional[ep.Tensor], Optional[ep.Tensor], Optional[int]

class Model(ABC):

    @property
    @abstractmethod
    def bounds(self) -> Bounds:
        ...

    @abstractmethod
    def __call__(self, inputs: Any) -> Any:
        ...

    def transform_bounds(self, bounds: Bounds) -> 'TransformBoundsWrapper':
        ...

class TransformBoundsWrapper(Model):

    def __init__(self, model: Model, bounds: BoundsInput):
        ...

    @property
    def bounds(self) -> Bounds:
        ...

    def __call__(self, inputs: Any) -> Any:
        ...

    def transform_bounds(self, bounds: BoundsInput, inplace: bool = False) -> 'TransformBoundsWrapper':
        ...

    def _preprocess(self, inputs: ep.Tensor) -> ep.Tensor:
        ...

    @property
    def data_format(self) -> Any:
        ...

ModelType = TypeVar('ModelType', bound='ModelWithPreprocessing')

class ModelWithPreprocessing(Model):

    def __init__(self, model: Callable, bounds: BoundsInput, dummy: int, preprocessing: Optional[Preprocessing] = None):
        ...

    @property
    def bounds(self) -> Bounds:
        ...

    @property
    def dummy(self) -> int:
        ...

    def __call__(self, inputs: Any) -> Any:
        ...

    def transform_bounds(self, bounds: BoundsInput, inplace: bool = False, wrapper: bool = False) -> 'ModelWithPreprocessing':
        ...

    def _preprocess(self, inputs: ep.Tensor) -> ep.Tensor:
        ...

    def _process_preprocessing(self, preprocessing: Optional[Preprocessing]) -> PreprocessArgs:
        ...
