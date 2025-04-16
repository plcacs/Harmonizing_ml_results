from typing import TypeVar, Callable, Optional, Tuple, Any, Dict
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
    def bounds(self):
        ...

    @abstractmethod
    def __call__(self, inputs):
        """Passes inputs through the model and returns the model's output"""
        ...

    def transform_bounds(self, bounds):
        """Returns a new model with the desired bounds and updates the preprocessing accordingly"""
        return TransformBoundsWrapper(self, bounds)


class TransformBoundsWrapper(Model):

    def __init__(self, model, bounds):
        self._model: Model = model
        self._bounds: Bounds = Bounds(*bounds)

    @property
    def bounds(self):
        return self._bounds

    def __call__(self, inputs):
        x: ep.Tensor
        restore_type: Callable[[ep.Tensor], T]
        x, restore_type = ep.astensor_(inputs)
        y: ep.Tensor = self._preprocess(x)
        z: T = self._model(y)
        return restore_type(z)

    def transform_bounds(self, bounds, inplace=False):
        if inplace:
            self._bounds = Bounds(*bounds)
            return self
        else:
            return TransformBoundsWrapper(self._model, bounds)

    def _preprocess(self, inputs):
        if self.bounds == self._model.bounds:
            return inputs
        min_: float
        max_: float
        min_, max_ = self.bounds
        x: ep.TensorType = (inputs - min_) / (max_ - min_)
        min_, max_ = self._model.bounds
        return x * (max_ - min_) + min_

    @property
    def data_format(self):
        return self._model.data_format


ModelType = TypeVar('ModelType', bound='ModelWithPreprocessing')


class ModelWithPreprocessing(Model):

    def __init__(self, model, bounds, dummy, preprocessing=None):
        if not callable(model):
            raise ValueError('expected model to be callable')
        self._model: Callable[..., ep.types.NativeTensor] = model
        self._bounds: Bounds = Bounds(*bounds)
        self._dummy: ep.Tensor = dummy
        self._preprocess_args: PreprocessArgs = self._process_preprocessing(
            preprocessing)

    @property
    def bounds(self):
        return self._bounds

    @property
    def dummy(self):
        return self._dummy

    def __call__(self, inputs):
        x: ep.Tensor
        restore_type: Callable[[ep.Tensor], T]
        x, restore_type = ep.astensor_(inputs)
        y: ep.Tensor = self._preprocess(x)
        z: ep.Tensor = ep.astensor(self._model(y.raw))
        return restore_type(z)

    def transform_bounds(self, bounds, inplace=False, wrapper=False):
        """Returns a new model with the desired bounds and updates the preprocessing accordingly"""
        if wrapper:
            if inplace:
                raise ValueError('inplace and wrapper cannot both be True')
            return super().transform_bounds(bounds)
        if self.bounds == bounds:
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
            model: ModelWithPreprocessing = self
        else:
            model = copy.copy(self)
        model._bounds = Bounds(*bounds)
        model._preprocess_args = mean, std, flip_axis
        return model

    def _preprocess(self, inputs):
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

    def _process_preprocessing(self, preprocessing):
        if preprocessing is None:
            preprocessing = dict()
        unsupported: set = set(preprocessing.keys()) - {'mean', 'std',
            'axis', 'flip_axis'}
        if len(unsupported) > 0:
            raise ValueError(f'unknown preprocessing key: {unsupported.pop()}')
        mean: Optional[Any] = preprocessing.get('mean', None)
        std: Optional[Any] = preprocessing.get('std', None)
        axis: Optional[int] = preprocessing.get('axis', None)
        flip_axis: Optional[int] = preprocessing.get('flip_axis', None)

        def to_tensor(x):
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

        def apply_axis(x, axis):
            if x is None:
                return None
            if x.ndim != 1:
                raise ValueError(
                    f'non-None axis requires a 1D tensor, got {x.ndim}D')
            if axis >= 0:
                raise ValueError(
                    'expected axis to be None or negative, -1 refers to the last axis'
                    )
            return atleast_kd(x, -axis)
        if axis is not None:
            mean_ = apply_axis(mean_, axis)
            std_ = apply_axis(std_, axis)
        return mean_, std_, flip_axis
