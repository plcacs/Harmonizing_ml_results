from __future__ import annotations
import decimal
import numbers
import sys
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Type, Union
import numpy as np
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import is_dtype_equal, is_float, is_integer, pandas_dtype
import pandas as pd
from pandas.api.extensions import no_default, register_extension_dtype
from pandas.api.types import is_list_like, is_scalar
from pandas.core import arraylike
from pandas.core.algorithms import value_counts_internal as value_counts
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import ExtensionArray, ExtensionScalarOpsMixin
from pandas.core.indexers import check_array_indexer

if TYPE_CHECKING:
    from pandas._typing import type_t


@register_extension_dtype
class DecimalDtype(ExtensionDtype):
    type: Type[decimal.Decimal] = decimal.Decimal
    name: str = 'decimal'
    na_value: decimal.Decimal = decimal.Decimal('NaN')
    _metadata: tuple[str, ...] = ('context',)

    def __init__(self, context: Optional[decimal.Context] = None) -> None:
        self.context: decimal.Context = context or decimal.getcontext()

    def __repr__(self) -> str:
        return f'DecimalDtype(context={self.context})'

    @classmethod
    def construct_array_type(cls) -> Type[DecimalArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return DecimalArray

    @property
    def _is_numeric(self) -> bool:
        return True


class DecimalArray(OpsMixin, ExtensionScalarOpsMixin, ExtensionArray):
    __array_priority__: float = 1000.0

    _data: np.ndarray
    _dtype: DecimalDtype
    _items: np.ndarray
    data: np.ndarray

    def __init__(
        self,
        values: Sequence[Union[decimal.Decimal, float, int, np.ndarray]],
        dtype: Optional[DecimalDtype] = None,
        copy: bool = False,
        context: Optional[decimal.Context] = None
    ) -> None:
        for i, val in enumerate(values):
            if is_float(val) or is_integer(val):
                if np.isnan(val):
                    values[i] = DecimalDtype.na_value
                else:
                    values[i] = DecimalDtype.type(val)
            elif not isinstance(val, decimal.Decimal):
                raise TypeError('All values must be of type ' + str(decimal.Decimal))
        values = np.asarray(values, dtype=object)
        self._data = values
        self._items = self.data = self._data
        self._dtype = DecimalDtype(context)

    @property
    def dtype(self) -> DecimalDtype:
        return self._dtype

    @classmethod
    def _from_sequence(
        cls,
        scalars: Sequence[Union[decimal.Decimal, float, int]],
        *,
        dtype: Optional[DecimalDtype] = None,
        copy: bool = False
    ) -> DecimalArray:
        return cls(scalars, dtype=dtype, copy=copy)

    @classmethod
    def _from_sequence_of_strings(
        cls,
        strings: Sequence[str],
        *,
        dtype: Optional[DecimalDtype] = None,
        copy: bool = False
    ) -> DecimalArray:
        return cls._from_sequence([decimal.Decimal(x) for x in strings], dtype=dtype, copy=copy)

    @classmethod
    def _from_factorized(
        cls,
        values: Sequence[decimal.Decimal],
        original: ExtensionArray
    ) -> DecimalArray:
        return cls(values)

    _HANDLED_TYPES: tuple[type, ...] = (decimal.Decimal, numbers.Number, np.ndarray)

    def to_numpy(
        self,
        dtype: Optional[Union[str, Type[np.dtype]]] = None,
        copy: bool = False,
        na_value: Any = no_default,
        decimals: Optional[int] = None
    ) -> np.ndarray:
        result = np.asarray(self, dtype=dtype)
        if decimals is not None:
            result = np.asarray([round(x, decimals) for x in result])
        return result

    def __array_ufunc__(
        self,
        ufunc: Callable,
        method: str,
        *inputs: Any,
        **kwargs: Any
    ) -> Any:
        if not all((isinstance(t, self._HANDLED_TYPES + (DecimalArray,)) for t in inputs)):
            return NotImplemented
        result = arraylike.maybe_dispatch_ufunc_to_dunder_op(self, ufunc, method, *inputs, **kwargs)
        if result is not NotImplemented:
            return result
        if 'out' in kwargs:
            return arraylike.dispatch_ufunc_with_out(self, ufunc, method, *inputs, **kwargs)
        inputs = tuple((x._data if isinstance(x, DecimalArray) else x for x in inputs))
        result = getattr(ufunc, method)(*inputs, **kwargs)
        if method == 'reduce':
            result = arraylike.dispatch_reduction_ufunc(self, ufunc, method, *inputs, **kwargs)
            if result is not NotImplemented:
                return result

        def reconstruct(x: Any) -> Union[decimal.Decimal, DecimalArray]:
            if isinstance(x, (decimal.Decimal, numbers.Number)):
                return x
            else:
                return type(self)._from_sequence(x, dtype=self.dtype)

        if ufunc.nout > 1:
            return tuple((reconstruct(x) for x in result))
        else:
            return reconstruct(result)

    def __getitem__(self, item: Union[int, slice, Sequence[int], np.ndarray]) -> Union[decimal.Decimal, DecimalArray]:
        if isinstance(item, numbers.Integral):
            return self._data[item]
        else:
            item = pd.api.indexers.check_array_indexer(self, item)
            return type(self)(self._data[item])

    def take(
        self,
        indexer: Union[np.ndarray, list, pd.Index],
        allow_fill: bool = False,
        fill_value: Optional[decimal.Decimal] = None
    ) -> DecimalArray:
        from pandas.api.extensions import take
        data = self._data
        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value
        result = take(data, indexer, fill_value=fill_value, allow_fill=allow_fill)
        return self._from_sequence(result, dtype=self.dtype)

    def copy(self) -> DecimalArray:
        return type(self)(self._data.copy(), dtype=self.dtype)

    def astype(
        self,
        dtype: Union[str, Type[ExtensionDtype], ExtensionDtype],
        copy: bool = True
    ) -> Union[DecimalArray, ExtensionArray]:
        if is_dtype_equal(dtype, self._dtype):
            if not copy:
                return self
        dtype = pandas_dtype(dtype)
        if isinstance(dtype, type(self.dtype)):
            return type(self)(self._data, copy=copy, context=dtype.context)
        return super().astype(dtype, copy=copy)

    def __setitem__(
        self,
        key: Union[int, slice, list, np.ndarray],
        value: Union[decimal.Decimal, Sequence[decimal.Decimal]]
    ) -> None:
        if is_list_like(value):
            if is_scalar(key):
                raise ValueError('setting an array element with a sequence.')
            value = [decimal.Decimal(v) for v in value]
        else:
            value = decimal.Decimal(value)
        key = check_array_indexer(self, key)
        self._data[key] = value

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, item: Any) -> bool:
        if not isinstance(item, decimal.Decimal):
            return False
        elif item.is_nan():
            return self.isna().any()
        else:
            return super().__contains__(item)

    @property
    def nbytes(self) -> int:
        n = len(self)
        if n:
            return n * sys.getsizeof(self[0])
        return 0

    def isna(self) -> np.ndarray:
        return np.array([x.is_nan() for x in self._data], dtype=bool)

    @property
    def _na_value(self) -> decimal.Decimal:
        return decimal.Decimal('NaN')

    def _formatter(self, boxed: bool = False) -> Callable[[decimal.Decimal], str]:
        if boxed:
            return 'Decimal: {}'.format
        return repr

    @classmethod
    def _concat_same_type(cls, to_concat: list[DecimalArray]) -> DecimalArray:
        return cls(np.concatenate([x._data for x in to_concat]))

    def _reduce(
        self,
        name: str,
        *,
        skipna: bool = True,
        keepdims: bool = False,
        **kwargs: Any
    ) -> Union[decimal.Decimal, DecimalArray]:
        if skipna and self.isna().any():
            other = self[~self.isna()]
            result = other._reduce(name, **kwargs)
        elif name == 'sum' and len(self) == 0:
            result = decimal.Decimal(0)
        else:
            try:
                op: Callable = getattr(self.data, name)
            except AttributeError as err:
                raise NotImplementedError(f'decimal does not support the {name} operation') from err
            result = op(axis=0)
        if keepdims:
            return type(self)([result])
        else:
            return result

    def _cmp_method(
        self,
        other: Any,
        op: Callable[[decimal.Decimal, Any], bool]
    ) -> np.ndarray:
        def convert_values(param: Any) -> Sequence[Any]:
            if isinstance(param, ExtensionArray) or is_list_like(param):
                ovalues = param
            else:
                ovalues = [param] * len(self)
            return ovalues

        lvalues = self
        rvalues = convert_values(other)
        res = [op(a, b) for a, b in zip(lvalues, rvalues)]
        return np.asarray(res, dtype=bool)

    def value_counts(self, dropna: bool = True) -> pd.Series:
        return value_counts(self.to_numpy(), dropna=dropna)

    def fillna(
        self,
        value: Optional[decimal.Decimal] = None,
        limit: Optional[int] = None
    ) -> DecimalArray:
        return super().fillna(value=value, limit=limit, copy=True)


def to_decimal(
    values: Sequence[Union[str, float, int]],
    context: Optional[decimal.Context] = None
) -> DecimalArray:
    return DecimalArray([decimal.Decimal(x) for x in values], context=context)


def make_data() -> list[decimal.Decimal]:
    return [decimal.Decimal(val) for val in np.random.default_rng(2).random(100)]


DecimalArray._add_arithmetic_ops()
