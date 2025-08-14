import inspect
import pickle
from collections.abc import Hashable
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional, TypeVar, Union

import numpy as np
import pandas as pd

from snorkel.types import DataPoint, FieldMap, HashingFunction

MapFunction = Callable[[DataPoint], Optional[DataPoint]]

T = TypeVar('T')

def get_parameters(
    f: Callable[..., Any], allow_args: bool = False, allow_kwargs: bool = False
) -> List[str]:
    """Get names of function parameters."""
    params = inspect.getfullargspec(f)
    if not allow_args and params[1] is not None:
        raise ValueError(f"Function {f.__name__} should not have *args")
    if not allow_kwargs and params[2] is not None:
        raise ValueError(f"Function {f.__name__} should not have **kwargs")
    return params[0]


def is_hashable(obj: Any) -> bool:
    """Test if object is hashable via duck typing."""
    try:
        hash(obj)
        return True
    except Exception:
        return False


def get_hashable(obj: Any) -> Hashable:
    """Get a hashable version of a potentially unhashable object."""
    if is_hashable(obj):
        return obj
    if isinstance(obj, SimpleNamespace):
        obj = vars(obj)
    if isinstance(obj, (dict, pd.Series)):
        return frozenset((k, get_hashable(v)) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return tuple(get_hashable(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return obj.data.tobytes()
    raise ValueError(f"Object {obj} has no hashing proxy.")


class BaseMapper:
    def __init__(
        self,
        name: str,
        pre: List["BaseMapper"],
        memoize: bool,
        memoize_key: Optional[HashingFunction] = None,
    ) -> None:
        if memoize_key is None:
            memoize_key = get_hashable
        self.name = name
        self._pre = pre
        self._memoize_key = memoize_key
        self.memoize = memoize
        self.reset_cache()

    def reset_cache(self) -> None:
        self._cache: Dict[DataPoint, DataPoint] = {}

    def _generate_mapped_data_point(self, x: DataPoint) -> Optional[DataPoint]:
        raise NotImplementedError

    def __call__(self, x: DataPoint) -> Optional[DataPoint]:
        if self.memoize:
            x_hashable = self._memoize_key(x)
            if x_hashable in self._cache:
                return self._cache[x_hashable]
        x_mapped = pickle.loads(pickle.dumps(x))
        for mapper in self._pre:
            x_mapped = mapper(x_mapped)
        x_mapped = self._generate_mapped_data_point(x_mapped)
        if self.memoize:
            self._cache[x_hashable] = x_mapped
        return x_mapped

    def __repr__(self) -> str:
        pre_str = f", Pre: {self._pre}"
        return f"{type(self).__name__} {self.name}{pre_str}"


class Mapper(BaseMapper):
    def __init__(
        self,
        name: str,
        field_names: Optional[Mapping[str, str]] = None,
        mapped_field_names: Optional[Mapping[str, str]] = None,
        pre: Optional[List[BaseMapper]] = None,
        memoize: bool = False,
        memoize_key: Optional[HashingFunction] = None,
    ) -> None:
        if field_names is None:
            field_names = {k: k for k in get_parameters(self.run)[1:]}
        self.field_names = field_names
        self.mapped_field_names = mapped_field_names
        super().__init__(name, pre or [], memoize, memoize_key)

    def run(self, **kwargs: Any) -> Optional[FieldMap]:
        raise NotImplementedError

    def _update_fields(self, x: DataPoint, mapped_fields: FieldMap) -> DataPoint:
        for k, v in mapped_fields.items():
            setattr(x, k, v)
        return x

    def _generate_mapped_data_point(self, x: DataPoint) -> Optional[DataPoint]:
        field_map = {k: getattr(x, v) for k, v in self.field_names.items()}
        mapped_fields = self.run(**field_map)
        if mapped_fields is None:
            return None
        if self.mapped_field_names is not None:
            mapped_fields = {
                v: mapped_fields[k] for k, v in self.mapped_field_names.items()
            }
        return self._update_fields(x, mapped_fields)


class LambdaMapper(BaseMapper):
    def __init__(
        self,
        name: str,
        f: MapFunction,
        pre: Optional[List[BaseMapper]] = None,
        memoize: bool = False,
        memoize_key: Optional[HashingFunction] = None,
    ) -> None:
        self._f = f
        super().__init__(name, pre or [], memoize, memoize_key)

    def _generate_mapped_data_point(self, x: DataPoint) -> Optional[DataPoint]:
        return self._f(x)


class lambda_mapper:
    def __init__(
        self,
        name: Optional[str] = None,
        pre: Optional[List[BaseMapper]] = None,
        memoize: bool = False,
        memoize_key: Optional[HashingFunction] = None,
    ) -> None:
        if callable(name):
            raise ValueError("Looks like this decorator is missing parentheses!")
        self.name = name
        self.pre = pre
        self.memoize = memoize
        self.memoize_key = memoize_key

    def __call__(self, f: MapFunction) -> LambdaMapper:
        name = self.name or f.__name__
        return LambdaMapper(
            name=name,
            f=f,
            pre=self.pre,
            memoize=self.memoize,
            memoize_key=self.memoize_key,
        )
