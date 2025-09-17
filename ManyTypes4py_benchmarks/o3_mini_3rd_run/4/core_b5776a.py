import inspect
import pickle
from collections.abc import Hashable, Mapping
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping as MappingType, Optional, Tuple
import numpy as np
import pandas as pd
from snorkel.types import DataPoint, FieldMap, HashingFunction

MapFunction = Callable[[DataPoint], Optional[DataPoint]]


def get_parameters(f: Callable, allow_args: bool = False, allow_kwargs: bool = False) -> List[str]:
    """Get names of function parameters."""
    params = inspect.getfullargspec(f)
    if not allow_args and params.args is not None and params.varargs is not None:
        raise ValueError(f'Function {f.__name__} should not have *args')
    if not allow_kwargs and params.kwonlyargs is not None and params.varkw is not None:
        raise ValueError(f'Function {f.__name__} should not have **kwargs')
    return params.args


def is_hashable(obj: Any) -> bool:
    """Test if object is hashable via duck typing.

    NB: not using ``collections.Hashable`` as some objects
    (e.g. pandas.Series) have a ``__hash__`` method to throw
    a more specific exception.
    """
    try:
        hash(obj)
        return True
    except Exception:
        return False


def get_hashable(obj: Any) -> Hashable:
    """Get a hashable version of a potentially unhashable object.

    This helper is used for caching mapper outputs of data points.
    For common data point formats (e.g. SimpleNamespace, pandas.Series),
    produces hashable representations of the values using a ``frozenset``.
    For objects like ``pandas.Series``, the name/index indentifier is dropped.

    Parameters
    ----------
    obj
        Object to get hashable version of

    Returns
    -------
    Hashable
        Hashable representation of object values

    Raises
    ------
    ValueError
        No hashable proxy for object
    """
    if is_hashable(obj):
        return obj
    if isinstance(obj, SimpleNamespace):
        obj = vars(obj)
    if isinstance(obj, (dict, pd.Series)):
        return frozenset(((k, get_hashable(v)) for k, v in obj.items()))
    if isinstance(obj, (list, tuple)):
        return tuple((get_hashable(v) for v in obj))
    if isinstance(obj, np.ndarray):
        return obj.data.tobytes()
    raise ValueError(f'Object {obj} has no hashing proxy.')


class BaseMapper:
    """Base class for ``Mapper`` and ``LambdaMapper``.

    Implements nesting, memoization, and deep copy functionality.
    Used primarily for type checking.
    """

    def __init__(
        self,
        name: str,
        pre: List[Callable[[DataPoint], Optional[DataPoint]]],
        memoize: bool,
        memoize_key: Optional[HashingFunction] = None
    ) -> None:
        if memoize_key is None:
            memoize_key = get_hashable
        self.name: str = name
        self._pre: List[Callable[[DataPoint], Optional[DataPoint]]] = pre
        self._memoize_key: HashingFunction = memoize_key
        self.memoize: bool = memoize
        self.reset_cache()

    def reset_cache(self) -> None:
        """Reset the memoization cache."""
        self._cache: Dict[Hashable, Optional[DataPoint]] = {}

    def _generate_mapped_data_point(self, x: DataPoint) -> Optional[DataPoint]:
        raise NotImplementedError

    def __call__(self, x: DataPoint) -> Optional[DataPoint]:
        """Run mapping function on input data point.

        Deep copies the data point first so as not to make
        accidental in-place changes. If ``memoize`` is set to
        ``True``, an internal cache is checked for results. If
        no cached results are found, the computed results are
        added to the cache.

        Parameters
        ----------
        x
            Data point to run mapping function on

        Returns
        -------
        DataPoint
            Mapped data point of same format but possibly different fields
        """
        if self.memoize:
            x_hashable: Hashable = self._memoize_key(x)
            if x_hashable in self._cache:
                return self._cache[x_hashable]
        x_mapped: DataPoint = pickle.loads(pickle.dumps(x))
        for mapper in self._pre:
            x_mapped = mapper(x_mapped)
        x_mapped = self._generate_mapped_data_point(x_mapped)
        if self.memoize:
            self._cache[x_hashable] = x_mapped
        return x_mapped

    def __repr__(self) -> str:
        pre_str: str = f', Pre: {self._pre}'
        return f'{type(self).__name__} {self.name}{pre_str}'


class Mapper(BaseMapper):
    """Base class for any data point to data point mapping in the pipeline.

    Map data points to new data points by transforming, adding
    additional information, or decomposing into primitives. This module
    provides base classes for other operators like ``TransformationFunction``
    and ``Preprocessor``. We don't expect people to construct ``Mapper``
    objects directly.
    """

    def __init__(
        self,
        name: str,
        field_names: Optional[MappingType[str, str]] = None,
        mapped_field_names: Optional[MappingType[str, str]] = None,
        pre: Optional[List[Callable[[DataPoint], Optional[DataPoint]]]] = None,
        memoize: bool = False,
        memoize_key: Optional[HashingFunction] = None
    ) -> None:
        if field_names is None:
            # Exclude the first parameter (self)
            field_names = {k: k for k in get_parameters(self.run)[1:]}
        self.field_names: MappingType[str, str] = field_names
        self.mapped_field_names: Optional[MappingType[str, str]] = mapped_field_names
        super().__init__(name, pre or [], memoize, memoize_key)

    def run(self, **kwargs: Any) -> Optional[FieldMap]:
        """Run the mapping operation using the input fields.

        The inputs to this function are fed by extracting the fields of
        the input data point using the keys of ``field_names``. The output field
        names are converted using ``mapped_field_names`` and added to the
        data point.

        Returns
        -------
        Optional[FieldMap]
            A mapping from canonical output field names to their values.

        Raises
        ------
        NotImplementedError
            Subclasses must implement this method
        """
        raise NotImplementedError

    def _update_fields(self, x: DataPoint, mapped_fields: MappingType[str, Any]) -> DataPoint:
        for k, v in mapped_fields.items():
            setattr(x, k, v)
        return x

    def _generate_mapped_data_point(self, x: DataPoint) -> Optional[DataPoint]:
        field_map: Dict[str, Any] = {k: getattr(x, v) for k, v in self.field_names.items()}
        mapped_fields: Optional[FieldMap] = self.run(**field_map)
        if mapped_fields is None:
            return None
        if self.mapped_field_names is not None:
            mapped_fields = {v: mapped_fields[k] for k, v in self.mapped_field_names.items()}
        return self._update_fields(x, mapped_fields)


class LambdaMapper(BaseMapper):
    """Define a mapper from a function.

    Convenience class for mappers that execute a simple
    function with no set up. The function should map from
    an input data point to a new data point directly, unlike
    ``Mapper.run``. The original data point will not be updated,
    so in-place operations are safe.
    """

    def __init__(
        self,
        name: str,
        f: Callable[[DataPoint], Optional[DataPoint]],
        pre: Optional[List[Callable[[DataPoint], Optional[DataPoint]]]] = None,
        memoize: bool = False,
        memoize_key: Optional[HashingFunction] = None
    ) -> None:
        self._f: Callable[[DataPoint], Optional[DataPoint]] = f
        super().__init__(name, pre or [], memoize, memoize_key)

    def _generate_mapped_data_point(self, x: DataPoint) -> Optional[DataPoint]:
        return self._f(x)


class lambda_mapper:
    """Decorate a function to define a LambdaMapper object.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        pre: Optional[List[Callable[[DataPoint], Optional[DataPoint]]]] = None,
        memoize: bool = False,
        memoize_key: Optional[HashingFunction] = None
    ) -> None:
        if callable(name):
            raise ValueError('Looks like this decorator is missing parentheses!')
        self.name: Optional[str] = name
        self.pre: Optional[List[Callable[[DataPoint], Optional[DataPoint]]]] = pre
        self.memoize: bool = memoize
        self.memoize_key: Optional[HashingFunction] = memoize_key

    def __call__(self, f: Callable[[DataPoint], Optional[DataPoint]]) -> LambdaMapper:
        """Wrap a function to create a ``LambdaMapper``.

        Parameters
        ----------
        f
            Function executing the mapping operation

        Returns
        -------
        LambdaMapper
            New ``LambdaMapper`` executing operation in wrapped function
        """
        name: str = self.name or f.__name__
        return LambdaMapper(name=name, f=f, pre=self.pre, memoize=self.memoize, memoize_key=self.memoize_key)