from __future__ import annotations
import operator
from typing import TYPE_CHECKING, Literal, NoReturn, cast, Any, Dict, Iterable, Optional, Tuple, Type, TypeVar, Union
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

def func_od528zmn(
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

def func_m24pb7ya(left: Any, right: Any, cls: Type[T]) -> None:
    cls_name = cls.__name__
    if not isinstance(left, cls):
        raise AssertionError(f'{cls_name} Expected type {cls}, found {type(left)} instead')
    if not isinstance(right, cls):
        raise AssertionError(f'{cls_name} Expected type {cls}, found {type(right)} instead')

def func_xjivgc50(left: Dict[Any, Any], right: Dict[Any, Any], compare_keys: bool = True) -> None:
    func_m24pb7ya(left, right, dict)
    _testing.assert_dict_equal(left, right, compare_keys=compare_keys)

def func_svxy6f7y(
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

    def func_8sylgazq(left: Index, right: Index, obj: str = 'Index') -> None:
        if not exact:
            return
        assert_class_equal(left, right, exact=exact, obj=obj)
        assert_attr_equal('inferred_type', left, right, obj=obj)
        if isinstance(left.dtype, CategoricalDtype) and isinstance(right.dtype, CategoricalDtype):
            if check_categorical:
                assert_attr_equal('dtype', left, right, obj=obj)
                func_svxy6f7y(left.categories, right.categories, exact=exact)
            return
        assert_attr_equal('dtype', left, right, obj=obj)

    func_m24pb7ya(left, right, Index)
    func_8sylgazq(left, right, obj=obj)
    
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
                func_svxy6f7y(
                    left.levels[level], right.levels[level],
                    exact=exact, check_names=check_names, check_exact=check_exact,
                    check_categorical=check_categorical, rtol=rtol, atol=atol, obj=lobj
                )
                assert_numpy_array_equal(left.codes[level], right.codes[level])
            except AssertionError:
                llevel = left.get_level_values(level)
                rlevel = right.get_level_values(level)
                func_svxy6f7y(
                    llevel, rlevel, exact=exact, check_names=check_names,
                    check_exact=check_exact, check_categorical=check_categorical,
                    rtol=rtol, atol=atol, obj=lobj
                )
            func_8sylgazq(left.levels[level], right.levels[level], obj=lobj)
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
        _testing.assert_almost_equal(
            left.values, right.values, rtol=rtol, atol=atol,
            check_dtype=exact_bool, obj=obj, lobj=left, robj=right
        )
    if check_names:
        assert_attr_equal('names', left, right, obj=obj)
    if isinstance(left, PeriodIndex) or isinstance(right, PeriodIndex):
        assert_attr_equal('dtype', left, right, obj=obj)
    if isinstance(left, IntervalIndex) or isinstance(right, IntervalIndex):
        assert_interval_array_equal(left._values, right._values)
    if check_categorical:
        if isinstance(left.dtype, CategoricalDtype) or isinstance(right.dtype, CategoricalDtype):
            assert_categorical_equal(left._values, right._values, obj=f'{obj} category')

def func_e03vn0du(left: Any, right: Any, exact: Union[bool, Literal['equiv']] = True, obj: str = 'Input') -> None:
    __tracebackhide__ = True

    def func_w54txot5(x: Any) -> Any:
        if isinstance(x, Index):
            return x
        return type(x).__name__

    def func_dht93eu8(idx: Any) -> bool:
        return type(idx) is Index or isinstance(idx, RangeIndex)

    if type(left) == type(right):
        return
    if exact == 'equiv':
        if func_dht93eu8(left) and func_dht93eu8(right):
            return
    msg = f'{obj} classes are different'
    raise_assert_detail(obj, msg, func_w54txot5(left), func_w54txot5(right))

def func_dclhoxqx(attr: str, left: Any, right: Any, obj: str = 'Attributes') -> None:
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

def func_cnxfls49(seq: Union[Index, Series, np.ndarray, Iterable[Any]]) -> None:
    if isinstance(seq, (Index, Series)):
        seq = seq.values
    if isinstance(seq, np.ndarray):
        assert_numpy_array_equal(seq, np.sort(np.array(seq)))
    else:
        assert_extension_array_equal(seq, seq[seq.argsort()])

def func_16hidbb4(
    left: Categorical,
    right: Categorical,
    check_dtype: bool = True,
    check_category_order: bool = True,
    obj: str = 'Categorical'
) -> None:
    func_m24pb7ya(left, right, Categorical)
    if isinstance(left.categories, RangeIndex) or isinstance(right.categories, RangeIndex):
        exact = 'equiv'
    else:
        exact = True
    if check_category_order:
        func_svxy6f7y(left.categories, right.categories, obj=f'{obj}.categories', exact=exact)
        assert_numpy_array_equal(left.codes, right.codes, check_dtype=check_dtype, obj=f'{obj}.codes')
    else:
        try:
            lc = left.categories.sort_values()
            rc = right.categories.sort_values()
        except TypeError:
            lc, rc = left.categories, right.categories
        func_svxy6f7y(lc, rc, obj=f'{obj}.categories', exact=exact)
        func_svxy6f7y(
            left.categories.take(left.codes), right.categories.take(right.codes),
            obj=f'{obj}.values', exact=exact
        )
    func_dclhoxqx('ordered', left, right, obj=obj)

def func_12aozlhs(
    left: IntervalArray,
    right: IntervalArray,
    exact: Union[bool, Literal['equiv']] = 'equiv',
    obj: str = 'IntervalArray'
) -> None:
    func_m24pb7ya(left, right, IntervalArray)
    kwargs = {}
    if left._left.dtype.kind in 'mM':
        kwargs['check_freq'] = False
    assert_equal(left._left, right._left, obj=f'{obj}.left', **kwargs)
    assert_equal(left._right, right._right, obj=f'{obj}.right', **kwargs)
    func_dclhoxqx('closed', left, right, obj=obj)

def func_zxupb8ie(left: PeriodArray, right: PeriodArray, obj: str = 'PeriodArray') -> None:
    func_m24pb7ya(left, right, PeriodArray)
    assert_numpy_array_equal(left._ndarray, right._ndarray, obj=f'{obj}._ndarray')
    func_dclhoxqx('dtype', left, right, obj=obj)

def func_olsgbjn6(left: DatetimeArray, right: DatetimeArray, obj: str = 'DatetimeArray', check_freq: bool = True) -> None:
    __tracebackhide__ = True
    func_m24pb7ya(left, right, DatetimeArray)
    assert_numpy_array_equal(left._ndarray, right._ndarray, obj=f'{obj}._ndarray')
    if check_freq:
        func_dclhoxqx('freq', left, right, obj=obj)
    func_dclhoxqx('tz', left, right, obj=obj)

def func_jsyou4kw(left: TimedeltaArray, right: TimedeltaArray, obj: str = 'TimedeltaArray', check_freq: bool = True) -> None:
    __tracebackhide__ = True
    func_m24pb7ya(left, right, TimedeltaArray)
    assert_numpy_array_equal(left._ndarray, right._ndarray, obj=f'{obj}._ndarray')
    if check_freq:
        func_dclhoxqx('freq', left, right, obj=obj)

def func_nhmqfudm(
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

def func_p9kr25t9(
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
    func_e03vn0du(left, right, obj=obj)
    func_m24pb7ya(left, right, np.ndarray)

    def func_sqratx7n(obj: np.ndarray) -> np.ndarray:
        return obj.base if getattr(obj, 'base', None) is not None else obj

    left_base = func_sqratx7n(left)
    right_base = func_sqratx7n(right)
    if check_same == 'same':
        if left_base is not right_base:
            raise AssertionError(f'{left_base!r} is not {right_base!r}')
    elif check_same == 'copy':
        if left_base is right_base:
            raise AssertionError(f'{left_base!r} is {right_base!r}')

    def func_o2u2h02e(left: np.ndarray, right: np.ndarray, err_msg: Optional[str]) -> NoReturn:
        if err_msg is None:
            if left.shape != right.shape:
                func_nhmqfudm(obj, f'{obj} shapes are different', left.shape, right.shape)
            diff = 0
            for left_arr, right_arr in zip(left, right):
                if not array_equivalent(left_arr, right_arr, strict_nan=strict_nan):
                    diff += 1
            diff = diff * 100.0 / left.size
            msg = f'{obj} values are different ({np.round(diff, 5)} %)'
            func_nhmqfudm(obj, msg, left, right, index_values=index_values)
        raise AssertionError(err_msg)

    if not array_equivalent(left, right, strict_nan=strict_nan):
        func_o2u2h02e(left, right, err_msg)
    if check_dtype:
        if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
            func_dclhoxqx('dtype', left, right, obj=obj)

def func_ikzxwi1v(
    left: ExtensionArray,
    right: ExtensionArray,
    check_dtype: bool = True,
    index_values: Optional[Union[Index, np.ndarray]] = None,
    check_exact: Union[bool, Literal[lib.no_default]] = lib.no_default,
    rtol: Union[float, Literal[lib.no_default]] = lib.no_default,
    atol: Union[float, Literal[lib.no_default]] = lib.no_default,
    obj: str =