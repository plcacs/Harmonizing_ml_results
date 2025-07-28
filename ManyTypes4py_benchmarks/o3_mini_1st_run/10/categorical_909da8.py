from __future__ import annotations

from csv import QUOTE_NONNUMERIC
from functools import partial
import operator
from shutil import get_terminal_size
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)
import numpy as np

from pandas._config import get_option
from pandas._libs import NaT, algos as libalgos, lib
from pandas._libs.arrays import NDArrayBacked
from pandas.compat.numpy import function as nv
from pandas.util._validators import validate_bool_kwarg
from pandas.core.dtypes.cast import coerce_indexer_dtype, find_common_type
from pandas.core.dtypes.common import (
    ensure_int64,
    ensure_platform_int,
    is_any_real_numeric_dtype,
    is_bool_dtype,
    is_dict_like,
    is_hashable,
    is_integer_dtype,
    is_list_like,
    is_scalar,
    needs_i8_conversion,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import ArrowDtype, CategoricalDtype, CategoricalDtypeType, ExtensionDtype
from pandas.core.dtypes.generic import ABCIndex, ABCSeries
from pandas.core.dtypes.missing import is_valid_na_for_dtype, isna
from pandas.core import algorithms, arraylike, ops
from pandas.core.accessor import PandasDelegate, delegate_names
from pandas.core.algorithms import factorize, take_nd
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray, ravel_compat
from pandas.core.base import ExtensionArray, NoNewAttributesMixin, PandasObject
import pandas.core.common as com
from pandas.core.construction import extract_array, sanitize_array
from pandas.core.ops.common import unpack_zerodim_and_defer
from pandas.core.sorting import nargsort
from pandas.core.strings.object_array import ObjectStringArrayMixin
from pandas.io.formats import console
if TYPE_CHECKING:
    from collections.abc import Callable as AbcCallable, Hashable, Iterator as AbcIterator, Sequence as AbcSequence
    from pandas._typing import ArrayLike, AstypeArg, AxisInt, Dtype, DtypeObj, NpDtype, Ordered, Self, Shape, SortKind, npt
    from pandas import DataFrame, Index, Series

def _cat_compare_op(op: Callable[[Any, Any], Any]) -> Callable[[Categorical, Any], np.ndarray]:
    opname = f'__{op.__name__}__'
    fill_value = op is operator.ne

    @unpack_zerodim_and_defer(opname)
    def func(self: Categorical, other: Any) -> np.ndarray:
        hashable: bool = is_hashable(other)
        if is_list_like(other) and len(other) != len(self) and (not hashable):
            raise ValueError('Lengths must match.')
        if not self.ordered:
            if opname in ['__lt__', '__gt__', '__le__', '__ge__']:
                raise TypeError('Unordered Categoricals can only compare equality or not')
        if isinstance(other, Categorical):
            msg = "Categoricals can only be compared if 'categories' are the same."
            if not self._categories_match_up_to_permutation(other):
                raise TypeError(msg)
            if not self.ordered and (not self.categories.equals(other.categories)):
                other_codes = recode_for_categories(other.codes, other.categories, self.categories, copy=False)
            else:
                other_codes = other._codes
            ret = op(self._codes, other_codes)
            mask = (self._codes == -1) | (other_codes == -1)
            if mask.any():
                ret[mask] = fill_value
            return ret
        if hashable:
            if other in self.categories:
                i = self._unbox_scalar(other)
                ret = op(self._codes, i)
                if opname not in {'__eq__', '__ge__', '__gt__'}:
                    mask = self._codes == -1
                    ret[mask] = fill_value
                return ret
            else:
                return ops.invalid_comparison(self, other, op)
        else:
            if opname not in ['__eq__', '__ne__']:
                raise TypeError(f"Cannot compare a Categorical for op {opname} with type {type(other)}.\nIf you want to compare values, use 'np.asarray(cat) <op> other'.")
            if isinstance(other, ExtensionArray) and needs_i8_conversion(other.dtype):
                return op(other, self)
            return getattr(np.array(self), opname)(np.array(other))
    func.__name__ = opname
    return func

def contains(cat: Any, key: Any, container: Any) -> bool:
    """
    Helper for membership check for ``key`` in ``cat``.
    """
    hash(key)
    try:
        loc = cat.categories.get_loc(key)
    except (KeyError, TypeError):
        return False
    if is_scalar(loc):
        return loc in container
    else:
        return any((loc_ in container for loc_ in loc))

class Categorical(NDArrayBackedExtensionArray, PandasObject, ObjectStringArrayMixin):
    __array_priority__ = 1000
    _hidden_attrs = PandasObject._hidden_attrs | frozenset(['tolist'])
    _typ = 'categorical'

    @classmethod
    def _simple_new(cls, codes: np.ndarray, dtype: CategoricalDtype) -> Categorical:
        codes = coerce_indexer_dtype(codes, dtype.categories)
        dtype = CategoricalDtype(ordered=False).update_dtype(dtype)
        return super()._simple_new(codes, dtype)  # type: ignore[return-value]

    def __init__(
        self,
        values: Any,
        categories: Optional[Any] = None,
        ordered: Optional[bool] = None,
        dtype: Optional[Any] = None,
        copy: bool = True,
    ) -> None:
        dtype = CategoricalDtype._from_values_or_dtype(values, categories, ordered, dtype)
        if not is_list_like(values):
            raise TypeError('Categorical input must be list-like')
        null_mask = np.array(False)
        vdtype = getattr(values, 'dtype', None)
        if isinstance(vdtype, CategoricalDtype):
            if dtype.categories is None:
                dtype = CategoricalDtype(values.categories, dtype.ordered)
        elif isinstance(values, range):
            from pandas.core.indexes.range import RangeIndex
            values = RangeIndex(values)
        elif not isinstance(values, (ABCIndex, ABCSeries, ExtensionArray)):
            values = com.convert_to_list_like(values)
            if isinstance(values, list) and len(values) == 0:
                values = np.array([], dtype=object)
            elif isinstance(values, np.ndarray):
                if values.ndim > 1:
                    raise NotImplementedError('> 1 ndim Categorical are not supported at this time')
                values = sanitize_array(values, None)
            else:
                arr = sanitize_array(values, None)
                null_mask = isna(arr)
                if null_mask.any():
                    arr_list = [values[idx] for idx in np.where(~null_mask)[0]]
                    if arr_list or arr.dtype == 'object':
                        sanitize_dtype = None
                    else:
                        sanitize_dtype = arr.dtype
                    arr = sanitize_array(arr_list, None, dtype=sanitize_dtype)
                values = arr
        if dtype.categories is None:
            if isinstance(values.dtype, ArrowDtype) and issubclass(values.dtype.type, CategoricalDtypeType):
                arr = values._pa_array.combine_chunks()
                categories = arr.dictionary.to_pandas(types_mapper=ArrowDtype)
                codes = arr.indices.to_numpy()
                dtype = CategoricalDtype(categories, values.dtype.pyarrow_dtype.ordered)
            else:
                if not isinstance(values, ABCIndex):
                    values = sanitize_array(values, None)
                try:
                    codes, categories = factorize(values, sort=True)
                except TypeError as err:
                    codes, categories = factorize(values, sort=False)
                    if dtype.ordered:
                        raise TypeError("'values' is not ordered, please explicitly specify the categories order by passing in a categories argument.") from err
                dtype = CategoricalDtype(categories, dtype.ordered)
        elif isinstance(values.dtype, CategoricalDtype):
            old_codes = extract_array(values)._codes
            codes = recode_for_categories(old_codes, values.dtype.categories, dtype.categories, copy=copy)
        else:
            codes = _get_codes_for_values(values, dtype.categories)
        if null_mask.any():
            full_codes = -np.ones(null_mask.shape, dtype=codes.dtype)
            full_codes[~null_mask] = codes
            codes = full_codes
        dtype = CategoricalDtype(ordered=False).update_dtype(dtype)
        arr = coerce_indexer_dtype(codes, dtype.categories)
        super().__init__(arr, dtype)

    @property
    def dtype(self) -> CategoricalDtype:
        return self._dtype

    @property
    def _internal_fill_value(self) -> Any:
        dtype = self._ndarray.dtype
        return dtype.type(-1)

    @classmethod
    def _from_sequence(cls, scalars: Sequence[Any], *, dtype: Optional[Any] = None, copy: bool = False) -> Categorical:
        return cls(scalars, dtype=dtype, copy=copy)

    @classmethod
    def _from_scalars(cls, scalars: Sequence[Any], *, dtype: Any) -> Categorical:
        if dtype is None:
            raise NotImplementedError
        res = cls._from_sequence(scalars, dtype=dtype)
        mask = isna(scalars)
        if not (mask == res.isna()).all():
            raise ValueError
        return res

    @overload
    def astype(self, dtype: Any, copy: bool = ...) -> Categorical: ...
    @overload
    def astype(self, dtype: Any, copy: bool = ...) -> Categorical: ...
    @overload
    def astype(self, dtype: Any, copy: bool = ...) -> Categorical: ...

    def astype(self, dtype: Any, copy: bool = True) -> Union[Categorical, np.ndarray]:
        dtype = pandas_dtype(dtype)
        if self.dtype is dtype:
            result = self.copy() if copy else self
        elif isinstance(dtype, CategoricalDtype):
            dtype = self.dtype.update_dtype(dtype)
            self = self.copy() if copy else self
            result = self._set_dtype(dtype)
        elif isinstance(dtype, ExtensionDtype):
            return super().astype(dtype, copy=copy)
        elif dtype.kind in 'iu' and self.isna().any():
            raise ValueError('Cannot convert float NaN to integer')
        elif len(self.codes) == 0 or len(self.categories) == 0:
            if not copy:
                result = np.asarray(self, dtype=dtype)
            else:
                result = np.array(self, dtype=dtype)
        else:
            new_cats = self.categories._values
            try:
                new_cats = new_cats.astype(dtype=dtype, copy=copy)
                fill_value = self.categories._na_value
                if not is_valid_na_for_dtype(fill_value, dtype):
                    fill_value = lib.item_from_zerodim(np.array(self.categories._na_value).astype(dtype))
            except (TypeError, ValueError) as err:
                msg = f'Cannot cast {self.categories.dtype} dtype to {dtype}'
                raise ValueError(msg) from err
            result = take_nd(new_cats, ensure_platform_int(self._codes), fill_value=fill_value)
        return result

    @classmethod
    def _from_inferred_categories(
        cls, inferred_categories: Any, inferred_codes: Any, dtype: Any, true_values: Optional[List[Any]] = None
    ) -> Categorical:
        from pandas import Index, to_datetime, to_numeric, to_timedelta
        cats = Index(inferred_categories)
        known_categories = isinstance(dtype, CategoricalDtype) and dtype.categories is not None
        if known_categories:
            if is_any_real_numeric_dtype(dtype.categories.dtype):
                cats = to_numeric(inferred_categories, errors='coerce')
            elif lib.is_np_dtype(dtype.categories.dtype, 'M'):
                cats = to_datetime(inferred_categories, errors='coerce')
            elif lib.is_np_dtype(dtype.categories.dtype, 'm'):
                cats = to_timedelta(inferred_categories, errors='coerce')
            elif is_bool_dtype(dtype.categories.dtype):
                if true_values is None:
                    true_values = ['True', 'TRUE', 'true']
                cats = cats.isin(true_values)
        if known_categories:
            categories = dtype.categories
            codes = recode_for_categories(inferred_codes, cats, categories)
        elif not cats.is_monotonic_increasing:
            unsorted = cats.copy()
            categories = cats.sort_values()
            codes = recode_for_categories(inferred_codes, unsorted, categories)
            dtype = CategoricalDtype(categories, ordered=False)
        else:
            dtype = CategoricalDtype(cats, ordered=False)
            codes = inferred_codes
        return cls._simple_new(codes, dtype=dtype)

    @classmethod
    def from_codes(
        cls,
        codes: Sequence[int],
        categories: Optional[Any] = None,
        ordered: Optional[bool] = None,
        dtype: Optional[Any] = None,
        validate: bool = True,
    ) -> Categorical:
        dtype = CategoricalDtype._from_values_or_dtype(categories=categories, ordered=ordered, dtype=dtype)
        if dtype.categories is None:
            msg = "The categories must be provided in 'categories' or 'dtype'. Both were None."
            raise ValueError(msg)
        if validate:
            codes = cls._validate_codes_for_dtype(codes, dtype=dtype)
        return cls._simple_new(np.array(codes), dtype=dtype)

    @property
    def categories(self) -> Any:
        return self.dtype.categories

    @property
    def ordered(self) -> bool:
        return self.dtype.ordered

    @property
    def codes(self) -> np.ndarray:
        v = self._codes.view()
        v.flags.writeable = False
        return v

    def _set_categories(self, categories: Any, fastpath: bool = False) -> None:
        if fastpath:
            new_dtype = CategoricalDtype._from_fastpath(categories, self.ordered)
        else:
            new_dtype = CategoricalDtype(categories, ordered=self.ordered)
        if not fastpath and self.dtype.categories is not None and (len(new_dtype.categories) != len(self.dtype.categories)):
            raise ValueError('new categories need to have the same number of items as the old categories!')
        super().__init__(self._ndarray, new_dtype)

    def _set_dtype(self, dtype: CategoricalDtype) -> Categorical:
        codes = recode_for_categories(self.codes, self.categories, dtype.categories)
        return type(self)._simple_new(codes, dtype=dtype)

    def set_ordered(self, value: bool) -> Categorical:
        new_dtype = CategoricalDtype(self.categories, ordered=value)
        cat = self.copy()
        NDArrayBacked.__init__(cat, cat._ndarray, new_dtype)
        return cat

    def as_ordered(self) -> Categorical:
        return self.set_ordered(True)

    def as_unordered(self) -> Categorical:
        return self.set_ordered(False)

    def set_categories(self, new_categories: Any, ordered: Optional[bool] = None, rename: bool = False) -> Categorical:
        if ordered is None:
            ordered = self.dtype.ordered
        new_dtype = CategoricalDtype(new_categories, ordered=ordered)
        cat = self.copy()
        if rename:
            if cat.dtype.categories is not None and len(new_dtype.categories) < len(cat.dtype.categories):
                cat._codes[cat._codes >= len(new_dtype.categories)] = -1
            codes = cat._codes
        else:
            codes = recode_for_categories(cat.codes, cat.categories, new_dtype.categories)
        NDArrayBacked.__init__(cat, codes, new_dtype)
        return cat

    def rename_categories(self, new_categories: Any) -> Categorical:
        if is_dict_like(new_categories):
            new_categories = [new_categories.get(item, item) for item in self.categories]
        elif callable(new_categories):
            new_categories = [new_categories(item) for item in self.categories]
        cat = self.copy()
        cat._set_categories(new_categories)
        return cat

    def reorder_categories(self, new_categories: Any, ordered: Optional[bool] = None) -> Categorical:
        if len(self.categories) != len(new_categories) or not self.categories.difference(new_categories).empty:
            raise ValueError('items in new_categories are not the same as in old categories')
        return self.set_categories(new_categories, ordered=ordered)

    def add_categories(self, new_categories: Any) -> Categorical:
        if not is_list_like(new_categories):
            new_categories = [new_categories]
        already_included = set(new_categories) & set(self.dtype.categories)
        if len(already_included) != 0:
            raise ValueError(f'new categories must not include old categories: {already_included}')
        if hasattr(new_categories, 'dtype'):
            from pandas import Series
            dtype = find_common_type([self.dtype.categories.dtype, new_categories.dtype])
            new_categories = Series(list(self.dtype.categories) + list(new_categories), dtype=dtype)
        else:
            new_categories = list(self.dtype.categories) + list(new_categories)
        new_dtype = CategoricalDtype(new_categories, self.ordered)
        cat = self.copy()
        codes = coerce_indexer_dtype(cat._ndarray, new_dtype.categories)
        NDArrayBacked.__init__(cat, codes, new_dtype)
        return cat

    def remove_categories(self, removals: Any) -> Categorical:
        from pandas import Index
        if not is_list_like(removals):
            removals = [removals]
        removals = Index(removals).unique().dropna()
        new_categories = (
            self.dtype.categories.difference(removals, sort=False)
            if self.dtype.ordered is True
            else self.dtype.categories.difference(removals)
        )
        not_included = removals.difference(self.dtype.categories)
        if len(not_included) != 0:
            not_included = set(not_included)
            raise ValueError(f'removals must all be in old categories: {not_included}')
        return self.set_categories(new_categories, ordered=self.ordered, rename=False)

    def remove_unused_categories(self) -> Categorical:
        idx, inv = np.unique(self._codes, return_inverse=True)
        if idx.size != 0 and idx[0] == -1:
            idx, inv = (idx[1:], inv - 1)
        new_categories = self.dtype.categories.take(idx)
        new_dtype = CategoricalDtype._from_fastpath(new_categories, ordered=self.ordered)
        new_codes = coerce_indexer_dtype(inv, new_dtype.categories)
        cat = self.copy()
        NDArrayBacked.__init__(cat, new_codes, new_dtype)
        return cat

    def map(self, mapper: Union[Callable[[Any], Any], Dict[Any, Any]], na_action: Optional[str] = None) -> Union[Categorical, np.ndarray]:
        assert callable(mapper) or is_dict_like(mapper)
        new_categories = self.categories.map(mapper)
        has_nans = np.any(self._codes == -1)
        na_val: Any = np.nan
        if na_action is None and has_nans:
            na_val = mapper(np.nan) if callable(mapper) else mapper.get(np.nan, np.nan)
        if new_categories.is_unique and (not new_categories.hasnans) and (na_val is np.nan):
            new_dtype = CategoricalDtype(new_categories, ordered=self.ordered)
            return self.from_codes(self._codes.copy(), dtype=new_dtype, validate=False)
        if has_nans:
            new_categories = new_categories.insert(len(new_categories), na_val)
        return np.take(new_categories, self._codes)
    __eq__ = _cat_compare_op(operator.eq)
    __ne__ = _cat_compare_op(operator.ne)
    __lt__ = _cat_compare_op(operator.lt)
    __gt__ = _cat_compare_op(operator.gt)
    __le__ = _cat_compare_op(operator.le)
    __ge__ = _cat_compare_op(operator.ge)

    def _validate_setitem_value(self, value: Any) -> Any:
        if not is_hashable(value):
            return self._validate_listlike(value)
        else:
            return self._validate_scalar(value)

    def _validate_scalar(self, fill_value: Any) -> int:
        if is_valid_na_for_dtype(fill_value, self.categories.dtype):
            fill_value = -1
        elif fill_value in self.categories:
            fill_value = self._unbox_scalar(fill_value)
        else:
            raise TypeError(f'Cannot setitem on a Categorical with a new category ({fill_value}), set the categories first') from None
        return fill_value

    @classmethod
    def _validate_codes_for_dtype(cls, codes: Any, *, dtype: Any) -> np.ndarray:
        if isinstance(codes, ExtensionArray) and is_integer_dtype(codes.dtype):
            if isna(codes).any():
                raise ValueError('codes cannot contain NA values')
            codes = codes.to_numpy(dtype=np.int64)
        else:
            codes = np.asarray(codes)
        if len(codes) and codes.dtype.kind not in 'iu':
            raise ValueError('codes need to be array-like integers')
        if len(codes) and (codes.max() >= len(dtype.categories) or codes.min() < -1):
            raise ValueError('codes need to be between -1 and len(categories)-1')
        return codes

    @ravel_compat
    def __array__(self, dtype: Optional[Any] = None, copy: Optional[bool] = None) -> np.ndarray:
        if copy is False:
            raise ValueError('Unable to avoid copy while creating an array as requested.')
        ret = take_nd(self.categories._values, self._codes)
        return np.asarray(ret, dtype=dtype)

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        result = arraylike.maybe_dispatch_ufunc_to_dunder_op(self, ufunc, method, *inputs, **kwargs)
        if result is not NotImplemented:
            return result
        if 'out' in kwargs:
            return arraylike.dispatch_ufunc_with_out(self, ufunc, method, *inputs, **kwargs)
        if method == 'reduce':
            result = arraylike.dispatch_reduction_ufunc(self, ufunc, method, *inputs, **kwargs)
            if result is not NotImplemented:
                return result
        raise TypeError(f'Object with dtype {self.dtype} cannot perform the numpy op {ufunc.__name__}')

    def __setstate__(self, state: Any) -> None:
        if not isinstance(state, dict):
            return super().__setstate__(state)
        if '_dtype' not in state:
            state['_dtype'] = CategoricalDtype(state['_categories'], state['_ordered'])
        if '_codes' in state and '_ndarray' not in state:
            state['_ndarray'] = state.pop('_codes')
        super().__setstate__(state)

    @property
    def nbytes(self) -> int:
        return self._codes.nbytes + self.dtype.categories.values.nbytes

    def memory_usage(self, deep: bool = False) -> int:
        return self._codes.nbytes + self.dtype.categories.memory_usage(deep=deep)

    def isna(self) -> np.ndarray:
        return self._codes == -1
    isnull = isna

    def notna(self) -> np.ndarray:
        return ~self.isna()
    notnull = notna

    def value_counts(self, dropna: bool = True) -> Any:
        from pandas import CategoricalIndex, Series
        code = self._codes
        cat = self.categories
        ncat: int = len(cat)
        mask: np.ndarray = code >= 0
        ix: np.ndarray = np.arange(ncat)
        clean: bool = mask.all()
        if dropna or clean:
            obs = code if clean else code[mask]
            count = np.bincount(obs, minlength=ncat or 0)
        else:
            count = np.bincount(np.where(mask, code, ncat))
            ix = np.append(ix, -1)
        ix = coerce_indexer_dtype(ix, self.dtype.categories)
        ix_categorical = self._from_backing_data(ix)
        return Series(count, index=CategoricalIndex(ix_categorical), dtype='int64', name='count', copy=False)

    @classmethod
    def _empty(cls, shape: Tuple[int, ...], dtype: CategoricalDtype) -> Categorical:
        arr = cls._from_sequence([], dtype=dtype)
        backing = np.zeros(shape, dtype=arr._ndarray.dtype)
        return arr._from_backing_data(backing)

    def _internal_get_values(self) -> Union[np.ndarray, ExtensionArray]:
        if needs_i8_conversion(self.categories.dtype):
            return self.categories.take(self._codes, fill_value=NaT)._values
        elif is_integer_dtype(self.categories.dtype) and -1 in self._codes:
            return self.categories.astype('object').take(self._codes, fill_value=np.nan)._values
        return np.array(self)

    def check_for_ordered(self, op: str) -> None:
        if not self.ordered:
            raise TypeError(f'Categorical is not ordered for operation {op}\nyou can use .as_ordered() to change the Categorical to an ordered one\n')

    def argsort(self, *, ascending: bool = True, kind: str = 'quicksort', **kwargs: Any) -> np.ndarray:
        return super().argsort(ascending=ascending, kind=kind, **kwargs)

    @overload
    def sort_values(self, *, inplace: bool, ascending: bool, na_position: str) -> None: ...
    @overload
    def sort_values(self, *, inplace: bool = False, ascending: bool = True, na_position: str = 'last') -> Categorical: ...

    def sort_values(self, *, inplace: bool = False, ascending: bool = True, na_position: str = 'last') -> Optional[Categorical]:
        inplace = validate_bool_kwarg(inplace, 'inplace')
        if na_position not in ['last', 'first']:
            raise ValueError(f'invalid na_position: {na_position!r}')
        sorted_idx = nargsort(self, ascending=ascending, na_position=na_position)
        if not inplace:
            codes = self._codes[sorted_idx]
            return self._from_backing_data(codes)
        self._codes[:] = self._codes[sorted_idx]
        return None

    def _rank(self, *, axis: int = 0, method: str = 'average', na_option: str = 'keep', ascending: bool = True, pct: bool = False) -> np.ndarray:
        if axis != 0:
            raise NotImplementedError
        vff = self._values_for_rank()
        return algorithms.rank(vff, axis=axis, method=method, na_option=na_option, ascending=ascending, pct=pct)

    def _values_for_rank(self) -> np.ndarray:
        from pandas import Series
        if self.ordered:
            values = self.codes
            mask = values == -1
            if mask.any():
                values = values.astype('float64')
                values[mask] = np.nan
        elif is_any_real_numeric_dtype(self.categories.dtype):
            values = np.array(self)
        else:
            values = np.array(self.rename_categories(Series(self.categories, copy=False).rank().values))
        return values

    def _hash_pandas_object(self, *, encoding: str, hash_key: str, categorize: bool) -> np.ndarray:
        from pandas.core.util.hashing import hash_array
        values = np.asarray(self.categories._values)
        hashed = hash_array(values, encoding, hash_key, categorize=False)
        mask = self.isna()
        if len(hashed):
            result = hashed.take(self._codes)
        else:
            result = np.zeros(len(mask), dtype='uint64')
        if mask.any():
            result[mask] = lib.u8max
        return result

    @property
    def _codes(self) -> np.ndarray:
        return self._ndarray

    def _box_func(self, i: int) -> Any:
        if i == -1:
            return np.nan
        return self.categories[i]

    def _unbox_scalar(self, key: Any) -> Any:
        code = self.categories.get_loc(key)
        code = self._ndarray.dtype.type(code)
        return code

    def __iter__(self) -> Iterator[Any]:
        if self.ndim == 1:
            return iter(self._internal_get_values().tolist())
        else:
            return (self[n] for n in range(len(self)))

    def __contains__(self, key: Any) -> bool:
        if is_valid_na_for_dtype(key, self.categories.dtype):
            return bool(self.isna().any())
        return contains(self, key, container=self._codes)

    def _formatter(self, boxed: bool = False) -> Optional[Any]:
        return None

    def _repr_categories(self) -> List[str]:
        max_categories = 10 if get_option('display.max_categories') == 0 else get_option('display.max_categories')
        from pandas.io.formats import format as fmt
        format_array = partial(fmt.format_array, formatter=None, quoting=QUOTE_NONNUMERIC)
        if len(self.categories) > max_categories:
            num = max_categories // 2
            head = format_array(self.categories[:num]._values)
            tail = format_array(self.categories[-num:]._values)
            category_strs = head + ['...'] + tail
        else:
            category_strs = format_array(self.categories._values)
        category_strs = [x.strip() for x in category_strs]
        return category_strs

    def _get_repr_footer(self) -> str:
        category_strs = self._repr_categories()
        dtype_str = str(self.categories.dtype)
        levheader = f'Categories ({len(self.categories)}, {dtype_str}): '
        width, _ = get_terminal_size()
        max_width = get_option('display.width') or width
        if console.in_ipython_frontend():
            max_width = 0
        levstring = ''
        start = True
        cur_col_len = len(levheader)
        sep_len, sep = (3, ' < ') if self.ordered else (2, ', ')
        linesep = f'{sep.rstrip()}\n'
        for val in category_strs:
            if max_width != 0 and cur_col_len + sep_len + len(val) > max_width:
                levstring += linesep + ' ' * (len(levheader) + 1)
                cur_col_len = len(levheader) + 1
            elif not start:
                levstring += sep
                cur_col_len += len(val)
            levstring += val
            start = False
        return f'{levheader}[{levstring.replace(" < ... < ", " ... ")}]'

    def _get_values_repr(self) -> str:
        from pandas.io.formats import format as fmt
        assert len(self) > 0
        vals = self._internal_get_values()
        fmt_values = fmt.format_array(vals, None, float_format=None, na_rep='NaN', quoting=QUOTE_NONNUMERIC)
        fmt_values = [i.strip() for i in fmt_values]
        joined = ', '.join(fmt_values)
        result = '[' + joined + ']'
        return result

    def __repr__(self) -> str:
        footer = self._get_repr_footer()
        length = len(self)
        max_len = 10
        if length > max_len:
            num = max_len // 2
            head = self[:num]._get_values_repr()
            tail = self[-(max_len - num):]._get_values_repr()
            body = f'{head[:-1]}, ..., {tail[1:]}'
            length_info = f'Length: {len(self)}'
            result = f'{body}\n{length_info}\n{footer}'
        elif length > 0:
            body = self._get_values_repr()
            result = f'{body}\n{footer}'
        else:
            body = '[]'
            result = f'{body}, {footer}'
        return result

    def _validate_listlike(self, value: Any) -> np.ndarray:
        value = extract_array(value, extract_numpy=True)
        if isinstance(value, Categorical):
            if self.dtype != value.dtype:
                raise TypeError('Cannot set a Categorical with another, without identical categories')
            value = self._encode_with_my_categories(value)
            return value._codes
        from pandas import Index
        to_add = Index._with_infer(value, tupleize_cols=False).difference(self.categories)
        if len(to_add) and (not isna(to_add).all()):
            raise TypeError('Cannot setitem on a Categorical with a new category, set the categories first')
        codes = self.categories.get_indexer(value)
        return codes.astype(self._ndarray.dtype, copy=False)

    def _reverse_indexer(self) -> Dict[Any, np.ndarray]:
        categories = self.categories
        r, counts = libalgos.groupsort_indexer(ensure_platform_int(self.codes), categories.size)
        counts = ensure_int64(counts).cumsum()
        _result = (r[start:end] for start, end in zip(counts, counts[1:]))
        return dict(zip(categories, _result))

    def _reduce(self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs: Any) -> Any:
        result = super()._reduce(name, skipna=skipna, keepdims=keepdims, **kwargs)
        if name in ['argmax', 'argmin']:
            return result
        if keepdims:
            return type(self)(result, dtype=self.dtype)
        else:
            return result

    def min(self, *, skipna: bool = True, **kwargs: Any) -> Any:
        nv.validate_minmax_axis(kwargs.get('axis', 0))
        nv.validate_min((), kwargs)
        self.check_for_ordered('min')
        if not len(self._codes):
            return self.dtype.na_value
        good = self._codes != -1
        if not good.all():
            if skipna and good.any():
                pointer = self._codes[good].min()
            else:
                return np.nan
        else:
            pointer = self._codes.min()
        return self._wrap_reduction_result(None, pointer)

    def max(self, *, skipna: bool = True, **kwargs: Any) -> Any:
        nv.validate_minmax_axis(kwargs.get('axis', 0))
        nv.validate_max((), kwargs)
        self.check_for_ordered('max')
        if not len(self._codes):
            return self.dtype.na_value
        good = self._codes != -1
        if not good.all():
            if skipna and good.any():
                pointer = self._codes[good].max()
            else:
                return np.nan
        else:
            pointer = self._codes.max()
        return self._wrap_reduction_result(None, pointer)

    def _mode(self, dropna: bool = True) -> Categorical:
        codes = self._codes
        mask: Optional[np.ndarray] = None
        if dropna:
            mask = self.isna()
        res_codes = algorithms.mode(codes, mask=mask)
        res_codes = cast(np.ndarray, res_codes)
        assert res_codes.dtype == codes.dtype
        res = self._from_backing_data(res_codes)
        return res

    def unique(self) -> Categorical:
        return super().unique()

    def equals(self, other: Any) -> bool:
        if not isinstance(other, Categorical):
            return False
        elif self._categories_match_up_to_permutation(other):
            other = self._encode_with_my_categories(other)
            return np.array_equal(self._codes, other._codes)
        return False

    def _accumulate(self, name: str, skipna: bool = True, **kwargs: Any) -> Categorical:
        if name == 'cummin':
            func = np.minimum.accumulate
        elif name == 'cummax':
            func = np.maximum.accumulate
        else:
            raise TypeError(f'Accumulation {name} not supported for {type(self)}')
        self.check_for_ordered(name)
        codes = self.codes.copy()
        mask = self.isna()
        if func == np.minimum.accumulate:
            codes[mask] = np.iinfo(codes.dtype.type).max
        if not skipna:
            mask = np.maximum.accumulate(mask)
        codes = func(codes)
        codes[mask] = -1
        return self._simple_new(codes, dtype=self._dtype)

    @classmethod
    def _concat_same_type(cls, to_concat: List[Categorical], axis: int = 0) -> Categorical:
        from pandas.core.dtypes.concat import union_categoricals
        first = to_concat[0]
        if axis >= first.ndim:
            raise ValueError(f'axis {axis} is out of bounds for array of dimension {first.ndim}')
        if axis == 1:
            if not all((x.ndim == 2 for x in to_concat)):
                raise ValueError
            tc_flat: List[Categorical] = []
            for obj in to_concat:
                tc_flat.extend([obj[:, i] for i in range(obj.shape[1])])
            res_flat = cls._concat_same_type(tc_flat, axis=0)
            result = res_flat.reshape(len(first), -1, order='F')
            return result
        result = union_categoricals(to_concat)
        return result

    def _encode_with_my_categories(self, other: Categorical) -> Categorical:
        codes = recode_for_categories(other.codes, other.categories, self.categories, copy=False)
        return self._from_backing_data(codes)

    def _categories_match_up_to_permutation(self, other: Categorical) -> bool:
        return hash(self.dtype) == hash(other.dtype)

    def describe(self) -> Any:
        counts = self.value_counts(dropna=False)
        freqs = counts / counts.sum()
        from pandas import Index
        from pandas.core.reshape.concat import concat
        result = concat([counts, freqs], ignore_index=True, axis=1)
        result.columns = Index(['counts', 'freqs'])
        result.index.name = 'categories'
        return result

    def isin(self, values: Any) -> np.ndarray:
        null_mask = np.asarray(isna(values))
        code_values = self.categories.get_indexer_for(values)
        code_values = code_values[null_mask | (code_values >= 0)]
        return algorithms.isin(self.codes, code_values)

    def _str_map(
        self,
        f: Callable[[Any], Any],
        na_value: Any = lib.no_default,
        dtype: Any = np.dtype('object'),
        convert: bool = True,
    ) -> np.ndarray:
        categories = self.categories
        codes = self.codes
        if categories.dtype == 'string':
            result = categories.array._str_map(f, na_value, dtype)
            if categories.dtype.na_value is np.nan and is_bool_dtype(dtype) and (na_value is lib.no_default or isna(na_value)):
                na_value = False
        else:
            from pandas.core.arrays import NumpyExtensionArray
            result = NumpyExtensionArray(categories.to_numpy())._str_map(f, na_value, dtype)
        return take_nd(result, codes, fill_value=na_value)

    def _str_get_dummies(self, sep: str = '|', dtype: Optional[Any] = None) -> Any:
        from pandas.core.arrays import NumpyExtensionArray
        return NumpyExtensionArray(self.to_numpy(str, na_value='NaN'))._str_get_dummies(sep, dtype)

    def _groupby_op(self, *, how: str, has_dropped_na: bool, min_count: int, ngroups: int, ids: Any, **kwargs: Any) -> Any:
        from pandas.core.groupby.ops import WrappedCythonOp
        kind = WrappedCythonOp.get_kind_from_how(how)
        op = WrappedCythonOp(how=how, kind=kind, has_dropped_na=has_dropped_na)
        dtype = self.dtype
        if how in ['sum', 'prod', 'cumsum', 'cumprod', 'skew', 'kurt']:
            raise TypeError(f'{dtype} type does not support {how} operations')
        if how in ['min', 'max', 'rank', 'idxmin', 'idxmax'] and (not dtype.ordered):
            raise TypeError(f'Cannot perform {how} with non-ordered Categorical')
        if how not in ['rank', 'any', 'all', 'first', 'last', 'min', 'max', 'idxmin', 'idxmax']:
            if kind == 'transform':
                raise TypeError(f'{dtype} type does not support {how} operations')
            raise TypeError(f"{dtype} dtype does not support aggregation '{how}'")
        result_mask = None
        mask = self.isna()
        if how == 'rank':
            assert self.ordered
            npvalues = self._ndarray
        elif how in ['first', 'last', 'min', 'max', 'idxmin', 'idxmax']:
            npvalues = self._ndarray
            result_mask = np.zeros(ngroups, dtype=bool)
        else:
            npvalues = self.astype(bool)
        res_values = op._cython_op_ndim_compat(npvalues, min_count=min_count, ngroups=ngroups, comp_ids=ids, mask=mask, result_mask=result_mask, **kwargs)
        if how in op.cast_blocklist:
            return res_values
        elif how in ['first', 'last', 'min', 'max']:
            res_values[result_mask == 1] = -1
        return self._from_backing_data(res_values)

@delegate_names(delegate=Categorical, accessors=['categories', 'ordered'], typ='property')
@delegate_names(delegate=Categorical, accessors=['rename_categories', 'reorder_categories', 'add_categories', 'remove_categories', 'remove_unused_categories', 'set_categories', 'as_ordered', 'as_unordered'], typ='method')
class CategoricalAccessor(PandasDelegate, PandasObject, NoNewAttributesMixin):
    def __init__(self, data: Any) -> None:
        self._validate(data)
        self._parent = data.values
        self._index = data.index
        self._name = data.name
        self._freeze()

    @staticmethod
    def _validate(data: Any) -> None:
        if not isinstance(data.dtype, CategoricalDtype):
            raise AttributeError("Can only use .cat accessor with a 'category' dtype")

    def _delegate_property_get(self, name: str) -> Any:
        return getattr(self._parent, name)

    def _delegate_property_set(self, name: str, new_values: Any) -> None:
        setattr(self._parent, name, new_values)

    @property
    def codes(self) -> Any:
        from pandas import Series
        return Series(self._parent.codes, index=self._index)

    def _delegate_method(self, name: str, *args: Any, **kwargs: Any) -> Any:
        from pandas import Series
        method = getattr(self._parent, name)
        res = method(*args, **kwargs)
        if res is not None:
            return Series(res, index=self._index, name=self._name)

def _get_codes_for_values(values: Any, categories: Any) -> np.ndarray:
    codes = categories.get_indexer_for(values)
    return coerce_indexer_dtype(codes, categories)

def recode_for_categories(codes: np.ndarray, old_categories: Any, new_categories: Any, copy: bool = True) -> np.ndarray:
    if len(old_categories) == 0:
        if copy:
            return codes.copy()
        return codes
    elif new_categories.equals(old_categories):
        if copy:
            return codes.copy()
        return codes
    indexer = coerce_indexer_dtype(new_categories.get_indexer_for(old_categories), new_categories)
    new_codes = take_nd(indexer, codes, fill_value=-1)
    return new_codes

def factorize_from_iterable(values: Any) -> Tuple[np.ndarray, Any]:
    from pandas import CategoricalIndex
    if not is_list_like(values):
        raise TypeError('Input must be list-like')
    vdtype = getattr(values, 'dtype', None)
    if isinstance(vdtype, CategoricalDtype):
        values = extract_array(values)
        cat_codes = np.arange(len(values.categories), dtype=values.codes.dtype)
        cat = Categorical.from_codes(cat_codes, dtype=values.dtype, validate=False)
        categories = CategoricalIndex(cat)
        codes = values.codes
    else:
        cat = Categorical(values, ordered=False)
        categories = cat.categories
        codes = cat.codes
    return (codes, categories)

def factorize_from_iterables(iterables: Sequence[Any]) -> Tuple[List[np.ndarray], List[Any]]:
    if len(iterables) == 0:
        return ([], [])
    codes, categories = zip(*(factorize_from_iterable(it) for it in iterables))
    return (list(codes), list(categories))