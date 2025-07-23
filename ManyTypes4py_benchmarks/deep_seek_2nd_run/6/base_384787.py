from __future__ import annotations
import operator
from typing import (
    TYPE_CHECKING, 
    Any, 
    ClassVar, 
    Literal, 
    cast, 
    overload, 
    Union, 
    Sequence, 
    Callable, 
    Iterator, 
    Optional, 
    Tuple, 
    List, 
    Dict, 
    TypeVar, 
    Generic
)
import warnings
import numpy as np
from pandas._libs import algos as libalgos, lib
from pandas.compat import set_function_name
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError
from pandas.util._decorators import Appender, Substitution, cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_bool_kwarg, validate_insert_loc
from pandas.core.dtypes.cast import maybe_cast_pointwise_result
from pandas.core.dtypes.common import is_list_like, is_scalar, pandas_dtype
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCIndex, ABCSeries
from pandas.core.dtypes.missing import isna
from pandas.core import arraylike, missing, roperator
from pandas.core.algorithms import duplicated, factorize_array, isin, map_array, mode, rank, unique
from pandas.core.array_algos.quantile import quantile_with_mask
from pandas.core.missing import _fill_limit_area_1d
from pandas.core.sorting import nargminmax, nargsort

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from pandas._libs.missing import NAType
    from pandas._typing import (
        ArrayLike, 
        AstypeArg, 
        AxisInt, 
        Dtype, 
        DtypeObj, 
        FillnaOptions, 
        InterpolateOptions, 
        NumpySorter, 
        NumpyValueArrayLike, 
        PositionalIndexer, 
        ScalarIndexer, 
        Self, 
        SequenceIndexer, 
        Shape, 
        SortKind, 
        TakeIndexer, 
        npt
    )
    from pandas import Index

_extension_array_shared_docs: Dict[str, str] = {}

T = TypeVar('T')

class ExtensionArray(Generic[T]):
    _typ: ClassVar[str] = 'extension'
    __pandas_priority__: ClassVar[int] = 1000

    @classmethod
    def _from_sequence(cls, scalars: Sequence[T], *, dtype: Optional[Dtype] = None, copy: bool = False) -> ExtensionArray[T]:
        raise AbstractMethodError(cls)

    @classmethod
    def _from_scalars(cls, scalars: Sequence[T], *, dtype: ExtensionDtype) -> ExtensionArray[T]:
        try:
            return cls._from_sequence(scalars, dtype=dtype, copy=False)
        except (ValueError, TypeError):
            raise
        except Exception:
            warnings.warn('_from_scalars should only raise ValueError or TypeError. Consider overriding _from_scalars where appropriate.', stacklevel=find_stack_level())
            raise

    @classmethod
    def _from_sequence_of_strings(cls, strings: Sequence[str], *, dtype: ExtensionDtype, copy: bool = False) -> ExtensionArray[T]:
        raise AbstractMethodError(cls)

    @classmethod
    def _from_factorized(cls, values: np.ndarray, original: ExtensionArray[T]) -> ExtensionArray[T]:
        raise AbstractMethodError(cls)

    @overload
    def __getitem__(self, item: int) -> T: ...

    @overload
    def __getitem__(self, item: slice) -> ExtensionArray[T]: ...

    @overload
    def __getitem__(self, item: np.ndarray) -> ExtensionArray[T]: ...

    def __getitem__(self, item: Union[int, slice, np.ndarray]) -> Union[T, ExtensionArray[T]]:
        raise AbstractMethodError(self)

    def __setitem__(self, key: Union[int, np.ndarray, slice], value: Any) -> None:
        raise NotImplementedError(f'{type(self)} does not implement __setitem__.')

    def __len__(self) -> int:
        raise AbstractMethodError(self)

    def __iter__(self) -> Iterator[T]:
        for i in range(len(self)):
            yield self[i]

    def __contains__(self, item: Any) -> bool:
        if is_scalar(item) and isna(item):
            if not self._can_hold_na:
                return False
            elif item is self.dtype.na_value or isinstance(item, self.dtype.type):
                return self._hasna
            else:
                return False
        else:
            return (item == self).any()

    def __eq__(self, other: Any) -> ExtensionArray[bool]:
        raise AbstractMethodError(self)

    def __ne__(self, other: Any) -> ExtensionArray[bool]:
        return ~(self == other)

    def to_numpy(self, dtype: Optional[Dtype] = None, copy: bool = False, na_value: Any = lib.no_default) -> np.ndarray:
        result = np.asarray(self, dtype=dtype)
        if copy or na_value is not lib.no_default:
            result = result.copy()
        if na_value is not lib.no_default:
            result[self.isna()] = na_value
        return result

    @property
    def dtype(self) -> ExtensionDtype:
        raise AbstractMethodError(self)

    @property
    def shape(self) -> Tuple[int, ...]:
        return (len(self),)

    @property
    def size(self) -> int:
        return np.prod(self.shape)

    @property
    def ndim(self) -> int:
        return 1

    @property
    def nbytes(self) -> int:
        raise AbstractMethodError(self)

    @overload
    def astype(self, dtype: ExtensionDtype, copy: bool = ...) -> ExtensionArray: ...

    @overload
    def astype(self, dtype: np.dtype, copy: bool = ...) -> np.ndarray: ...

    @overload
    def astype(self, dtype: str, copy: bool = ...) -> Union[ExtensionArray, np.ndarray]: ...

    def astype(self, dtype: AstypeArg, copy: bool = True) -> Union[ExtensionArray, np.ndarray]:
        dtype = pandas_dtype(dtype)
        if dtype == self.dtype:
            if not copy:
                return self
            else:
                return self.copy()
        if isinstance(dtype, ExtensionDtype):
            cls = dtype.construct_array_type()
            return cls._from_sequence(self, dtype=dtype, copy=copy)
        elif lib.is_np_dtype(dtype, 'M'):
            from pandas.core.arrays import DatetimeArray
            return DatetimeArray._from_sequence(self, dtype=dtype, copy=copy)
        elif lib.is_np_dtype(dtype, 'm'):
            from pandas.core.arrays import TimedeltaArray
            return TimedeltaArray._from_sequence(self, dtype=dtype, copy=copy)
        if not copy:
            return np.asarray(self, dtype=dtype)
        else:
            return np.array(self, dtype=dtype, copy=copy)

    def isna(self) -> np.ndarray:
        raise AbstractMethodError(self)

    @property
    def _hasna(self) -> bool:
        return bool(self.isna().any())

    def _values_for_argsort(self) -> np.ndarray:
        return np.array(self)

    def argsort(self, *, ascending: bool = True, kind: str = 'quicksort', na_position: str = 'last', **kwargs) -> np.ndarray:
        ascending = nv.validate_argsort_with_ascending(ascending, (), kwargs)
        values = self._values_for_argsort()
        return nargsort(values, kind=kind, ascending=ascending, na_position=na_position, mask=np.asarray(self.isna()))

    def argmin(self, skipna: bool = True) -> int:
        validate_bool_kwarg(skipna, 'skipna')
        if not skipna and self._hasna:
            raise ValueError('Encountered an NA value with skipna=False')
        return nargminmax(self, 'argmin')

    def argmax(self, skipna: bool = True) -> int:
        validate_bool_kwarg(skipna, 'skipna')
        if not skipna and self._hasna:
            raise ValueError('Encountered an NA value with skipna=False')
        return nargminmax(self, 'argmax')

    def interpolate(self, *, method: str, axis: AxisInt, index: Index, limit: Optional[int], limit_direction: str, limit_area: Optional[str], copy: bool, **kwargs) -> ExtensionArray[T]:
        raise NotImplementedError(f'{type(self).__name__} does not implement interpolate')

    def _pad_or_backfill(self, *, method: str, limit: Optional[int] = None, limit_area: Optional[str] = None, copy: bool = True) -> ExtensionArray[T]:
        mask = self.isna()
        if mask.any():
            meth = missing.clean_fill_method(method)
            npmask = np.asarray(mask)
            if limit_area is not None and (not npmask.all()):
                _fill_limit_area_1d(npmask, limit_area)
            if meth == 'pad':
                indexer = libalgos.get_fill_indexer(npmask, limit=limit)
                return self.take(indexer, allow_fill=True)
            else:
                indexer = libalgos.get_fill_indexer(npmask[::-1], limit=limit)[::-1]
                return self[::-1].take(indexer, allow_fill=True)
        else:
            if not copy:
                return self
            new_values = self.copy()
        return new_values

    def fillna(self, value: Any, limit: Optional[int] = None, copy: bool = True) -> ExtensionArray[T]:
        mask = self.isna()
        if limit is not None and limit < len(self):
            modify = mask.cumsum() > limit
            if modify.any():
                mask = mask.copy()
                mask[modify] = False
        value = missing.check_value_size(value, mask, len(self))
        if mask.any():
            if not copy:
                new_values = self[:]
            else:
                new_values = self.copy()
            new_values[mask] = value
        elif not copy:
            new_values = self[:]
        else:
            new_values = self.copy()
        return new_values

    def dropna(self) -> ExtensionArray[T]:
        return self[~self.isna()]

    def duplicated(self, keep: str = 'first') -> np.ndarray:
        mask = self.isna().astype(np.bool_, copy=False)
        return duplicated(values=self, keep=keep, mask=mask)

    def shift(self, periods: int = 1, fill_value: Optional[Any] = None) -> ExtensionArray[T]:
        if not len(self) or periods == 0:
            return self.copy()
        if isna(fill_value):
            fill_value = self.dtype.na_value
        empty = self._from_sequence([fill_value] * min(abs(periods), len(self)), dtype=self.dtype)
        if periods > 0:
            a = empty
            b = self[:-periods]
        else:
            a = self[abs(periods):]
            b = empty
        return self._concat_same_type([a, b])

    def unique(self) -> ExtensionArray[T]:
        uniques = unique(self.astype(object))
        return self._from_sequence(uniques, dtype=self.dtype)

    def searchsorted(self, value: Any, side: str = 'left', sorter: Optional[np.ndarray] = None) -> np.ndarray:
        arr = self.astype(object)
        if isinstance(value, ExtensionArray):
            value = value.astype(object)
        return arr.searchsorted(value, side=side, sorter=sorter)

    def equals(self, other: ExtensionArray[T]) -> bool:
        if type(self) != type(other):
            return False
        other = cast(ExtensionArray[T], other)
        if self.dtype != other.dtype:
            return False
        elif len(self) != len(other):
            return False
        else:
            equal_values = self == other
            if isinstance(equal_values, ExtensionArray):
                equal_values = equal_values.fillna(False)
            equal_na = self.isna() & other.isna()
            return bool((equal_values | equal_na).all())

    def isin(self, values: Union[np.ndarray, ExtensionArray]) -> np.ndarray:
        return isin(np.asarray(self), values)

    def _values_for_factorize(self) -> Tuple[np.ndarray, Any]:
        return (self.astype(object), np.nan)

    def factorize(self, use_na_sentinel: bool = True) -> Tuple[np.ndarray, ExtensionArray[T]]:
        arr, na_value = self._values_for_factorize()
        codes, uniques = factorize_array(arr, use_na_sentinel=use_na_sentinel, na_value=na_value)
        uniques_ea = self._from_factorized(uniques, self)
        return (codes, uniques_ea)

    @Substitution(klass='ExtensionArray')
    @Appender(_extension_array_shared_docs['repeat'])
    def repeat(self, repeats: Union[int, np.ndarray], axis: Optional[int] = None) -> ExtensionArray[T]:
        nv.validate_repeat((), {'axis': axis})
        ind = np.arange(len(self)).repeat(repeats)
        return self.take(ind)

    def take(self, indices: Sequence[int], *, allow_fill: bool = False, fill_value: Optional[Any] = None) -> ExtensionArray[T]:
        raise AbstractMethodError(self)

    def copy(self) -> ExtensionArray[T]:
        raise AbstractMethodError(self)

    def view(self, dtype: Optional[Dtype] = None) -> Union[ExtensionArray[T], np.ndarray]:
        if dtype is not None:
            raise NotImplementedError(dtype)
        return self[:]

    def __repr__(self) -> str:
        if self.ndim > 1:
            return self._repr_2d()
        from pandas.io.formats.printing import format_object_summary
        data = format_object_summary(self, self._formatter(), indent_for_name=False).rstrip(', \n')
        class_name = f'<{type(self).__name__}>\n'
        footer = self._get_repr_footer()
        return f'{class_name}{data}\n{footer}'

    def _get_repr_footer(self) -> str:
        if self.ndim > 1:
            return f'Shape: {self.shape}, dtype: {self.dtype}'
        return f'Length: {len(self)}, dtype: {self.dtype}'

    def _repr_2d(self) -> str:
        from pandas.io.formats.printing import format_object_summary
        lines = [format_object_summary(x, self._formatter(), indent_for_name=False).rstrip(', \n') for x in self]
        data = ',\n'.join(lines)
        class_name = f'<{type(self).__name__}>'
        footer = self._get_repr_footer()
        return f'{class_name}\n[\n{data}\n]\n{footer}'

    def _formatter(self, boxed: bool = False) -> Callable[[Any], str]:
        if boxed:
            return str
        return repr

    def transpose(self, *axes: int) -> ExtensionArray[T]:
        return self[:]

    @property
    def T(self) -> ExtensionArray[T]:
        return self.transpose()

    def ravel(self, order: str = 'C') -> ExtensionArray[T]:
        return self

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[ExtensionArray[T]]) -> ExtensionArray[T]:
        raise AbstractMethodError(cls)

    @cache_readonly
    def _can_hold_na(self) -> bool:
        return self.dtype._can_hold_na

    def _accumulate(self, name: str, *, skipna: bool = True, **kwargs) -> ExtensionArray[T]:
        raise NotImplementedError(f'cannot perform {name} with type {self.dtype}')

    def _reduce(self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs) -> Any:
        meth = getattr(self, name, None)
        if meth is None:
            raise TypeError(f"'{type(self).__name__}' with dtype {self.dtype} does not support operation '{name}'")
        result = meth(skipna=skipna, **kwargs)
        if keepdims:
            if name in ['min', 'max']:
                result = self._from_sequence([result], dtype=self.dtype)
            else:
                result = np.array([result])
        return result

    def _values_for_json(self) -> np.ndarray:
        return np.asarray(self)

    def _hash_pandas_object(self, *, encoding: str, hash_key: str, categorize: bool) -> np.ndarray:
        from pandas.core.util.hashing import hash_array
        values, _ = self._values_for_factorize()
        return hash_array(values, encoding=encoding, hash_key=hash_key, categorize=categorize)

    def _explode(self) -> Tuple[ExtensionArray[T], np.ndarray]:
        values = self.copy()
        counts = np.ones(shape=(len(self),), dtype=np.uint64)
        return (values, counts)

    def tolist(self) -> List[T]:
        if self.ndim > 1:
            return [x.tolist() for x in self]
        return list(self)

    def delete(self, loc: int) -> ExtensionArray[T]:
        indexer = np.delete(np.arange(len(self)), loc)
        return self.take(indexer)

    def insert(self, loc: int, item: T) -> ExtensionArray[T]:
        loc = validate_insert_loc(loc, len(self))
        item_arr = type(self)._from_sequence([item], dtype=self.dtype)
        return type(self)._concat_same_type([self[:loc], item_arr, self[loc:]])

    def _putmask(self, mask: np.ndarray, value: Any) -> None:
        if is_list_like(value):
            val = value[mask]
        else:
            val = value