from __future__ import annotations
import operator
from typing import TYPE_CHECKING, Literal, NoReturn, cast, Any, Dict, Iterable, Optional, Set, Tuple, Type, TypeVar, Union
import numpy as np
from pandas._libs import lib
from pandas._libs.missing import is_matching_na
from pandas._libs.sparse import SparseIndex
import pandas._libs.testing as _testing
from pandas._libs.tslibs.np_datetime import compare_mismatched_resolutions
from pandas.core.dtypes.common import is_bool, is_float_dtype, is_integer_dtype, is_number, is_numeric_dtype, needs_i8_conversion
from pandas.core.dtypes.dtypes import CategoricalDtype, DatetimeTZDtype, ExtensionDtype, NumpyEADtype
from pandas.core.dtypes.missing import array_equivalent
import pandas as pd
from pandas import Categorical, DataFrame, DatetimeIndex, Index, IntervalDtype, IntervalIndex, MultiIndex, PeriodIndex, RangeIndex, Series, TimedeltaIndex
from pandas.core.arrays import DatetimeArray, ExtensionArray, IntervalArray, PeriodArray, TimedeltaArray
from pandas.core.arrays.datetimelike import DatetimeLikeArrayMixin
from pandas.core.arrays.string_ import StringDtype
from pandas.core.indexes.api import safe_sort_index
from pandas.io.formats.printing import pprint_thing

if TYPE_CHECKING:
    from pandas._typing import DtypeObj

T = TypeVar('T')

def assert_almost_equal(
    left: Any,
    right: Any,
    check_dtype: Union[bool, Literal['equiv']] = 'equiv',
    rtol: float = 1e-05,
    atol: float = 1e-08,
    **kwargs: Any
) -> None:
    if isinstance(left, Index):
        assert_index_equal(left, right, check_exact=False, exact=check_dtype, rtol=rtol, atol=atol, **kwargs)
    elif isinstance(left, Series):
        assert_series_equal(left, right, check_exact=False, check_dtype=check_dtype, rtol=rtol, atol=atol, **kwargs)
    elif isinstance(left, DataFrame):
        assert_frame_equal(left, right, check_exact=False, check_dtype=check_dtype, rtol=rtol, atol=atol, **kwargs)
    else:
        if check_dtype:
            if is_number(left) and is_number(right):
                pass
            elif is_bool(left) and is_bool(right):
                pass
            else:
                if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
                    obj = 'numpy array'
                else:
                    obj = 'Input'
                assert_class_equal(left, right, obj=obj)
        _testing.assert_almost_equal(left, right, check_dtype=bool(check_dtype), rtol=rtol, atol=atol, **kwargs)

def _check_isinstance(left: Any, right: Any, cls: Type[T]) -> None:
    cls_name = cls.__name__
    if not isinstance(left, cls):
        raise AssertionError(f'{cls_name} Expected type {cls}, found {type(left)} instead')
    if not isinstance(right, cls):
        raise AssertionError(f'{cls_name} Expected type {cls}, found {type(right)} instead')

def assert_dict_equal(left: Dict[Any, Any], right: Dict[Any, Any], compare_keys: bool = True) -> None:
    _check_isinstance(left, right, dict)
    _testing.assert_dict_equal(left, right, compare_keys=compare_keys)

def assert_index_equal(
    left: Index,
    right: Index,
    exact: Union[bool, Literal['equiv']] = 'equiv',
    check_names: bool = True,
    check_exact: bool = True,
    check_categorical: bool = True,
    check_order: bool = True,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    obj: Optional[str] = None
) -> None:
    __tracebackhide__ = True
    if obj is None:
        obj = 'MultiIndex' if isinstance(left, MultiIndex) else 'Index'

    def _check_types(left: Index, right: Index, obj: str = 'Index') -> None:
        if not exact:
            return
        assert_class_equal(left, right, exact=exact, obj=obj)
        assert_attr_equal('inferred_type', left, right, obj=obj)
        if isinstance(left.dtype, CategoricalDtype) and isinstance(right.dtype, CategoricalDtype):
            if check_categorical:
                assert_attr_equal('dtype', left, right, obj=obj)
                assert_index_equal(left.categories, right.categories, exact=exact)
            return
        assert_attr_equal('dtype', left, right, obj=obj)

    _check_isinstance(left, right, Index)
    _check_types(left, right, obj=obj)
    if left.nlevels != right.nlevels:
        msg1 = f'{obj} levels are different'
        msg2 = f'{left.nlevels}, {left}'
        msg3 = f'{right.nlevels}, {right}'
        raise_assert_detail(obj, msg1, msg2, msg3)
    if len(left) != len(right):
        msg1 = f'{obj} length are different'
        msg2 = f'{len(left)}, {left}'
        msg3 = f'{len(right)}, {right}'
        raise_assert_detail(obj, msg1, msg2, msg3)
    if not check_order:
        left = safe_sort_index(left)
        right = safe_sort_index(right)
    if isinstance(left, MultiIndex):
        right = cast(MultiIndex, right)
        for level in range(left.nlevels):
            lobj = f'{obj} level [{level}]'
            try:
                assert_index_equal(left.levels[level], right.levels[level], exact=exact, check_names=check_names, check_exact=check_exact, check_categorical=check_categorical, rtol=rtol, atol=atol, obj=lobj)
                assert_numpy_array_equal(left.codes[level], right.codes[level])
            except AssertionError:
                llevel = left.get_level_values(level)
                rlevel = right.get_level_values(level)
                assert_index_equal(llevel, rlevel, exact=exact, check_names=check_names, check_exact=check_exact, check_categorical=check_categorical, rtol=rtol, atol=atol, obj=lobj)
            _check_types(left.levels[level], right.levels[level], obj=lobj)
    elif check_exact and check_categorical:
        if not left.equals(right):
            mismatch = left._values != right._values
            if not isinstance(mismatch, np.ndarray):
                mismatch = cast('ExtensionArray', mismatch).fillna(True)
            diff = np.sum(mismatch.astype(int)) * 100.0 / len(left)
            msg = f'{obj} values are different ({np.round(diff, 5)} %)'
            raise_assert_detail(obj, msg, left, right)
    else:
        exact_bool = bool(exact)
        _testing.assert_almost_equal(left.values, right.values, rtol=rtol, atol=atol, check_dtype=exact_bool, obj=obj, lobj=left, robj=right)
    if check_names:
        assert_attr_equal('names', left, right, obj=obj)
    if isinstance(left, PeriodIndex) or isinstance(right, PeriodIndex):
        assert_attr_equal('dtype', left, right, obj=obj)
    if isinstance(left, IntervalIndex) or isinstance(right, IntervalIndex):
        assert_interval_array_equal(left._values, right._values)
    if check_categorical:
        if isinstance(left.dtype, CategoricalDtype) or isinstance(right.dtype, CategoricalDtype):
            assert_categorical_equal(left._values, right._values, obj=f'{obj} category')

def assert_class_equal(left: Any, right: Any, exact: Union[bool, Literal['equiv']] = True, obj: str = 'Input') -> None:
    __tracebackhide__ = True

    def repr_class(x: Any) -> Any:
        if isinstance(x, Index):
            return x
        return type(x).__name__

    def is_class_equiv(idx: Index) -> bool:
        return type(idx) is Index or isinstance(idx, RangeIndex)

    if type(left) == type(right):
        return
    if exact == 'equiv':
        if is_class_equiv(left) and is_class_equiv(right):
            return
    msg = f'{obj} classes are different'
    raise_assert_detail(obj, msg, repr_class(left), repr_class(right))

def assert_attr_equal(attr: str, left: Any, right: Any, obj: str = 'Attributes') -> Optional[None]:
    __tracebackhide__ = True
    left_attr = getattr(left, attr)
    right_attr = getattr(right, attr)
    if left_attr is right_attr or is_matching_na(left_attr, right_attr):
        return None
    try:
        result = left_attr == right_attr
    except TypeError:
        result = False
    if (left_attr is pd.NA) ^ (right_attr is pd.NA):
        result = False
    elif not isinstance(result, bool):
        result = result.all()
    if not result:
        msg = f'Attribute "{attr}" are different'
        raise_assert_detail(obj, msg, left_attr, right_attr)
    return None

def assert_is_sorted(seq: Union[Index, Series, np.ndarray, ExtensionArray]) -> None:
    if isinstance(seq, (Index, Series)):
        seq = seq.values
    if isinstance(seq, np.ndarray):
        assert_numpy_array_equal(seq, np.sort(np.array(seq)))
    else:
        assert_extension_array_equal(seq, seq[seq.argsort()])

def assert_categorical_equal(
    left: Categorical,
    right: Categorical,
    check_dtype: bool = True,
    check_category_order: bool = True,
    obj: str = 'Categorical'
) -> None:
    _check_isinstance(left, right, Categorical)
    if isinstance(left.categories, RangeIndex) or isinstance(right.categories, RangeIndex):
        exact = 'equiv'
    else:
        exact = True
    if check_category_order:
        assert_index_equal(left.categories, right.categories, obj=f'{obj}.categories', exact=exact)
        assert_numpy_array_equal(left.codes, right.codes, check_dtype=check_dtype, obj=f'{obj}.codes')
    else:
        try:
            lc = left.categories.sort_values()
            rc = right.categories.sort_values()
        except TypeError:
            lc, rc = (left.categories, right.categories)
        assert_index_equal(lc, rc, obj=f'{obj}.categories', exact=exact)
        assert_index_equal(left.categories.take(left.codes), right.categories.take(right.codes), obj=f'{obj}.values', exact=exact)
    assert_attr_equal('ordered', left, right, obj=obj)

def assert_interval_array_equal(
    left: IntervalArray,
    right: IntervalArray,
    exact: Union[bool, Literal['equiv']] = 'equiv',
    obj: str = 'IntervalArray'
) -> None:
    _check_isinstance(left, right, IntervalArray)
    kwargs = {}
    if left._left.dtype.kind in 'mM':
        kwargs['check_freq'] = False
    assert_equal(left._left, right._left, obj=f'{obj}.left', **kwargs)
    assert_equal(left._right, right._right, obj=f'{obj}.right', **kwargs)
    assert_attr_equal('closed', left, right, obj=obj)

def assert_period_array_equal(left: PeriodArray, right: PeriodArray, obj: str = 'PeriodArray') -> None:
    _check_isinstance(left, right, PeriodArray)
    assert_numpy_array_equal(left._ndarray, right._ndarray, obj=f'{obj}._ndarray')
    assert_attr_equal('dtype', left, right, obj=obj)

def assert_datetime_array_equal(
    left: DatetimeArray,
    right: DatetimeArray,
    obj: str = 'DatetimeArray',
    check_freq: bool = True
) -> None:
    __tracebackhide__ = True
    _check_isinstance(left, right, DatetimeArray)
    assert_numpy_array_equal(left._ndarray, right._ndarray, obj=f'{obj}._ndarray')
    if check_freq:
        assert_attr_equal('freq', left, right, obj=obj)
    assert_attr_equal('tz', left, right, obj=obj)

def assert_timedelta_array_equal(
    left: TimedeltaArray,
    right: TimedeltaArray,
    obj: str = 'TimedeltaArray',
    check_freq: bool = True
) -> None:
    __tracebackhide__ = True
    _check_isinstance(left, right, TimedeltaArray)
    assert_numpy_array_equal(left._ndarray, right._ndarray, obj=f'{obj}._ndarray')
    if check_freq:
        assert_attr_equal('freq', left, right, obj=obj)

def raise_assert_detail(
    obj: str,
    message: str,
    left: Any,
    right: Any,
    diff: Optional[Any] = None,
    first_diff: Optional[str] = None,
    index_values: Optional[Union[Index, np.ndarray]] = None
) -> NoReturn:
    __tracebackhide__ = True
    msg = f'{obj} are different\n\n{message}'
    if isinstance(index_values, Index):
        index_values = np.asarray(index_values)
    if isinstance(index_values, np.ndarray):
        msg += f'\n[index]: {pprint_thing(index_values)}'
    if isinstance(left, np.ndarray):
        left = pprint_thing(left)
    elif isinstance(left, (CategoricalDtype, NumpyEADtype)):
        left = repr(left)
    elif isinstance(left, StringDtype):
        left = f'StringDtype(storage={left.storage}, na_value={left.na_value})'
    if isinstance(right, np.ndarray):
        right = pprint_thing(right)
    elif isinstance(right, (CategoricalDtype, NumpyEADtype)):
        right = repr(right)
    elif isinstance(right, StringDtype):
        right = f'StringDtype(storage={right.storage}, na_value={right.na_value})'
    msg += f'\n[left]:  {left}\n[right]: {right}'
    if diff is not None:
        msg += f'\n[diff]: {diff}'
    if first_diff is not None:
        msg += f'\n{first_diff}'
    raise AssertionError(msg)

def assert_numpy_array_equal(
    left: np.ndarray,
    right: np.ndarray,
    strict_nan: bool = False,
    check_dtype: bool = True,
    err_msg: Optional[str] = None,
    check_same: Optional[Literal['copy', 'same']] = None,
    obj: str = 'numpy array',
    index_values: Optional[Union[Index, np.ndarray]] = None
) -> None:
    __tracebackhide__ = True
    assert_class_equal(left, right, obj=obj)
    _check_isinstance(left, right, np.ndarray)

    def _get_base(obj: np.ndarray) -> np.ndarray:
        return obj.base if getattr(obj, 'base', None) is not None else obj

    left_base = _get_base(left)
    right_base = _get_base(right)
    if check_same == 'same':
        if left_base is not right_base:
            raise AssertionError(f'{left_base!r} is not {right_base!r}')
    elif check_same == 'copy':
        if left_base is right_base:
            raise AssertionError(f'{left_base!r} is {right_base!r}')

    def _raise(left: np.ndarray, right: np.ndarray, err_msg: Optional[str]) -> NoReturn:
        if err_msg is None:
            if left.shape != right.shape:
                raise_assert_detail(obj, f'{obj} shapes are different', left.shape, right.shape)
            diff = 0
            for left_arr, right_arr in zip(left, right):
                if not array_equivalent(left_arr, right_arr, strict_nan=strict_nan):
                    diff += 1
            diff = diff * 100.0 / left.size
            msg = f'{obj} values are different ({np.round(diff, 5)} %)'
            raise_assert_detail(obj, msg, left, right, index_values=index_values)
        raise AssertionError(err_msg)

    if not array_equivalent(left, right, strict_nan=strict_nan):
        _raise(left, right, err_msg)
    if check_dtype:
        if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
            assert_attr_equal('dtype', left, right, obj=obj)

def assert_extension_array_equal(
    left: ExtensionArray,
    right: ExtensionArray,
    check_dtype: bool = True,
    index_values: Optional[Union[Index, np.ndarray]] = None,
    check_exact: Union[bool, Literal[lib.no_default]] = lib.no_default,
    rtol: Union[float, Literal[lib.no_default]] = lib.no_default,
    atol: Union[float, Literal[lib.no_default]] = lib.no_default,
    obj: str = 'ExtensionArray'
) -> None:
    if check_exact is lib.no_default and rtol is lib.no_default and (atol is lib.no_default):
        check_exact = is_numeric_dtype(left.dtype) and (not is_float_dtype(left.dtype)) or (is_numeric_dtype(right.dtype) and (not is_float_dtype(right.dtype)))
    elif check_exact is lib.no_default:
        check_exact = False
    rtol = rtol if rtol is not lib.no_default else 1e-05
    atol = atol if atol is not lib.no_default else 1e-08
    assert isinstance(left, ExtensionArray), 'left is not an ExtensionArray'
    assert isinstance(right, ExtensionArray