from typing import TypeVar, Callable, Optional, Tuple, Any, Dict, Set
from abc import ABC, abstractmethod
import copy
import eagerpy as ep
from ..types import Bounds, BoundsInput, Preprocessing
from ..devutils import atleast_kd

T = TypeVar('T')
PreprocessArgs = Tuple[Optional[ep.Tensor], Optional[ep.Tensor], Optional[int]]


class Model(ABC):

    @property
    @abstractmethod
    def bounds(self) -> Bounds:
        ...

    @abstractmethod
    def __call__(self, inputs: Any) -> Any:
        """Passes inputs through the model and returns the model's output"""
        ...

    def transform_bounds(self, bounds: BoundsInput) -> 'TransformBoundsWrapper':
        """Returns a new model with the desired bounds and updates the preprocessing accordingly"""
        return TransformBoundsWrapper(self, bounds)


class TransformBoundsWrapper(Model):

    def __init__(self, model: Model, bounds: BoundsInput) -> None:
        self._model = model
        self._bounds = Bounds(*bounds)

    @property
    def bounds(self) -> Bounds:
        return self._bounds

    def __call__(self, inputs: Any) -> Any:
        x: ep.Tensor
        restore_type: Callable[[ep.Tensor], Any]
        x, restore_type = ep.astensor_(inputs)
        y: ep.Tensor = self._preprocess(x)
        z: Any = self._model(y)
        return restore_type(z)

    def transform_bounds(self, bounds: BoundsInput, inplace: bool = False) -> 'TransformBoundsWrapper':
        if inplace:
            self._bounds = Bounds(*bounds)
            return self
        else:
            return TransformBoundsWrapper(self._model, bounds)

    def _preprocess(self, inputs: ep.Tensor) -> ep.Tensor:
        if self.bounds == self._model.bounds:
            return inputs
        min_: float
        max_: float
        min_, max_ = self.bounds
        x: ep.Tensor = (inputs - min_) / (max_ - min_)
        min_, max_ = self._model.bounds
        return x * (max_ - min_) + min_

    @property
    def data_format(self) -> Any:
        return self._model.data_format


ModelType = TypeVar('ModelType', bound='ModelWithPreprocessing')


class ModelWithPreprocessing(Model):

    def __init__(
        self,
        model: Callable[[ep.Tensor], Any],
        bounds: BoundsInput,
        dummy: ep.Tensor,
        preprocessing: Optional[Preprocessing] = None
    ) -> None:
        if not callable(model):
            raise ValueError('expected model to be callable')
        self._model: Callable[[ep.Tensor], Any] = model
        self._bounds: Bounds = Bounds(*bounds)
        self._dummy: ep.Tensor = dummy
        self._preprocess_args: PreprocessArgs = self._process_preprocessing(preprocessing)

    @property
    def bounds(self) -> Bounds:
        return self._bounds

    @property
    def dummy(self) -> ep.Tensor:
        return self._dummy

    def __call__(self, inputs: Any) -> ep.Tensor:
        x: ep.Tensor
        restore_type: Callable[[ep.Tensor], ep.Tensor]
        x, restore_type = ep.astensor_(inputs)
        y: ep.Tensor = self._preprocess(x)
        z_raw: Any = self._model(y.raw)
        z: ep.Tensor = ep.astensor(z_raw)
        return restore_type(z)

    def transform_bounds(
        self,
        bounds: BoundsInput,
        inplace: bool = False,
        wrapper: bool = False
    ) -> 'ModelWithPreprocessing':
        """Returns a new model with the desired bounds and updates the preprocessing accordingly"""
        if wrapper:
            if inplace:
                raise ValueError('inplace and wrapper cannot both be True')
            return super().transform_bounds(bounds)
        if self.bounds == Bounds(*bounds):
            if inplace:
                return self
            else:
                return copy.copy(self)
        a: float
        b: float
        c: float
        d: float
        a, b = self.bounds
        c, d = bounds
        f: float = (d - c) / (b - a)
        mean: Optional[ep.Tensor]
        std: Optional[ep.Tensor]
        flip_axis: Optional[int]
        mean, std, flip_axis = self._preprocess_args
        if mean is None:
            mean = ep.zeros(self._dummy, 1)
        mean = f * (mean - a) + c
        if std is None:
            std = ep.ones(self._dummy, 1)
        std = f * std
        if inplace:
            model: 'ModelWithPreprocessing' = self
        else:
            model = copy.copy(self)
        model._bounds = Bounds(*bounds)
        model._preprocess_args = (mean, std, flip_axis)
        return model

    def _preprocess(self, inputs: ep.Tensor) -> ep.Tensor:
        mean: Optional[ep.Tensor]
        std: Optional[ep.Tensor]
        flip_axis: Optional[int]
        mean, std, flip_axis = self._preprocess_args
        x: ep.Tensor = inputs
        if flip_axis is not None:
            x = x.flip(axis=flip_axis)
        if mean is not None:
            x = x - mean
        if std is not None:
            x = x / std
        assert x.dtype == inputs.dtype
        return x

    def _process_preprocessing(self, preprocessing: Optional[Preprocessing]) -> PreprocessArgs:
        if preprocessing is None:
            preprocessing = {}
        unsupported: Set[str] = set(preprocessing.keys()) - {'mean', 'std', 'axis', 'flip_axis'}
        if len(unsupported) > 0:
            unsupported_key: str = unsupported.pop()
            raise ValueError(f'unknown preprocessing key: {unsupported_key}')
        mean: Optional[Any] = preprocessing.get('mean', None)
        std: Optional[Any] = preprocessing.get('std', None)
        axis: Optional[int] = preprocessing.get('axis', None)
        flip_axis: Optional[int] = preprocessing.get('flip_axis', None)

        def to_tensor(x: Any) -> Optional[ep.Tensor]:
            if x is None:
                return None
            if isinstance(x, ep.Tensor):
                return x
            try:
                y: ep.Tensor = ep.astensor(x)
                if not isinstance(y, type(self._dummy)):
                    raise ValueError
                return y
            except ValueError:
                return ep.from_numpy(self._dummy, x)

        mean_ = to_tensor(mean)
        std_ = to_tensor(std)

        def apply_axis(x: Optional[ep.Tensor], axis_val: Optional[int]) -> Optional[ep.Tensor]:
            if x is None:
                return None
            if x.ndim != 1:
                raise ValueError(f'non-None axis requires a 1D tensor, got {x.ndim}D')
            if axis_val is not None and axis_val >= 0:
                raise ValueError('expected axis to be None or negative, -1 refers to the last axis')
            return atleast_kd(x, -axis_val) if axis_val is not None else x

        if axis is not None:
            mean_ = apply_axis(mean_, axis)
            std_ = apply_axis(std_, axis)

        return (mean_, std_, flip_axis)
