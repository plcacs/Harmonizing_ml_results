from __future__ import annotations
import operator
from typing import TYPE_CHECKING, Literal, NoReturn, cast
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


def func_od528zmn(left, right, check_dtype='equiv', rtol=1e-05, atol=1e-08,
    **kwargs):
    """
    Check that the left and right objects are approximately equal.

    By approximately equal, we refer to objects that are numbers or that
    contain numbers which may be equivalent to specific levels of precision.

    Parameters
    ----------
    left : object
    right : object
    check_dtype : bool or {'equiv'}, default 'equiv'
        Check dtype if both a and b are the same type. If 'equiv' is passed in,
        then `RangeIndex` and `Index` with int64 dtype are also considered
        equivalent when doing type checking.
    rtol : float, default 1e-5
        Relative tolerance.
    atol : float, default 1e-8
        Absolute tolerance.
    """
    if isinstance(left, Index):
        assert_index_equal(left, right, check_exact=False, exact=
            check_dtype, rtol=rtol, atol=atol, **kwargs)
    elif isinstance(left, Series):
        assert_series_equal(left, right, check_exact=False, check_dtype=
            check_dtype, rtol=rtol, atol=atol, **kwargs)
    elif isinstance(left, DataFrame):
        assert_frame_equal(left, right, check_exact=False, check_dtype=
            check_dtype, rtol=rtol, atol=atol, **kwargs)
    else:
        if check_dtype:
            if is_number(left) and is_number(right):
                pass
            elif is_bool(left) and is_bool(right):
                pass
            else:
                if isinstance(left, np.ndarray) or isinstance(right, np.ndarray
                    ):
                    obj = 'numpy array'
                else:
                    obj = 'Input'
                assert_class_equal(left, right, obj=obj)
        _testing.assert_almost_equal(left, right, check_dtype=bool(
            check_dtype), rtol=rtol, atol=atol, **kwargs)


def func_m24pb7ya(left, right, cls):
    """
    Helper method for our assert_* methods that ensures that
    the two objects being compared have the right type before
    proceeding with the comparison.

    Parameters
    ----------
    left : The first object being compared.
    right : The second object being compared.
    cls : The class type to check against.

    Raises
    ------
    AssertionError : Either `left` or `right` is not an instance of `cls`.
    """
    cls_name = cls.__name__
    if not isinstance(left, cls):
        raise AssertionError(
            f'{cls_name} Expected type {cls}, found {type(left)} instead')
    if not isinstance(right, cls):
        raise AssertionError(
            f'{cls_name} Expected type {cls}, found {type(right)} instead')


def func_xjivgc50(left, right, compare_keys=True):
    func_m24pb7ya(left, right, dict)
    _testing.assert_dict_equal(left, right, compare_keys=compare_keys)


def func_svxy6f7y(left, right, exact='equiv', check_names=True, check_exact
    =True, check_categorical=True, check_order=True, rtol=1e-05, atol=1e-08,
    obj=None):
    """
    Check that left and right Index are equal.

    Parameters
    ----------
    left : Index
        The first index to compare.
    right : Index
        The second index to compare.
    exact : bool or {'equiv'}, default 'equiv'
        Whether to check the Index class, dtype and inferred_type
        are identical. If 'equiv', then RangeIndex can be substituted for
        Index with an int64 dtype as well.
    check_names : bool, default True
        Whether to check the names attribute.
    check_exact : bool, default True
        Whether to compare number exactly.
    check_categorical : bool, default True
        Whether to compare internal Categorical exactly.
    check_order : bool, default True
        Whether to compare the order of index entries as well as their values.
        If True, both indexes must contain the same elements, in the same order.
        If False, both indexes must contain the same elements, but in any order.
    rtol : float, default 1e-5
        Relative tolerance. Only used when check_exact is False.
    atol : float, default 1e-8
        Absolute tolerance. Only used when check_exact is False.
    obj : str, default 'Index' or 'MultiIndex'
        Specify object name being compared, internally used to show appropriate
        assertion message.

    See Also
    --------
    testing.assert_series_equal : Check that two Series are equal.
    testing.assert_frame_equal : Check that two DataFrames are equal.

    Examples
    --------
    >>> from pandas import testing as tm
    >>> a = pd.Index([1, 2, 3])
    >>> b = pd.Index([1, 2, 3])
    >>> tm.assert_index_equal(a, b)
    """
    __tracebackhide__ = True
    if obj is None:
        obj = 'MultiIndex' if isinstance(left, MultiIndex) else 'Index'

    def func_8sylgazq(left, right, obj='Index'):
        if not exact:
            return
        assert_class_equal(left, right, exact=exact, obj=obj)
        assert_attr_equal('inferred_type', left, right, obj=obj)
        if isinstance(left.dtype, CategoricalDtype) and isinstance(right.
            dtype, CategoricalDtype):
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
                func_svxy6f7y(left.levels[level], right.levels[level],
                    exact=exact, check_names=check_names, check_exact=
                    check_exact, check_categorical=check_categorical, rtol=
                    rtol, atol=atol, obj=lobj)
                assert_numpy_array_equal(left.codes[level], right.codes[level])
            except AssertionError:
                llevel = left.get_level_values(level)
                rlevel = right.get_level_values(level)
                func_svxy6f7y(llevel, rlevel, exact=exact, check_names=
                    check_names, check_exact=check_exact, check_categorical
                    =check_categorical, rtol=rtol, atol=atol, obj=lobj)
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
        _testing.assert_almost_equal(left.values, right.values, rtol=rtol,
            atol=atol, check_dtype=exact_bool, obj=obj, lobj=left, robj=right)
    if check_names:
        assert_attr_equal('names', left, right, obj=obj)
    if isinstance(left, PeriodIndex) or isinstance(right, PeriodIndex):
        assert_attr_equal('dtype', left, right, obj=obj)
    if isinstance(left, IntervalIndex) or isinstance(right, IntervalIndex):
        assert_interval_array_equal(left._values, right._values)
    if check_categorical:
        if isinstance(left.dtype, CategoricalDtype) or isinstance(right.
            dtype, CategoricalDtype):
            assert_categorical_equal(left._values, right._values, obj=
                f'{obj} category')


def func_e03vn0du(left, right, exact=True, obj='Input'):
    """
    Checks classes are equal.
    """
    __tracebackhide__ = True

    def func_w54txot5(x):
        if isinstance(x, Index):
            return x
        return type(x).__name__

    def func_dht93eu8(idx):
        """Classes that are a RangeIndex (sub-)instance or exactly an `Index` .

        This only checks class equivalence. There is a separate check that the
        dtype is int64.
        """
        return type(idx) is Index or isinstance(idx, RangeIndex)
    if type(left) == type(right):
        return
    if exact == 'equiv':
        if func_dht93eu8(left) and func_dht93eu8(right):
            return
    msg = f'{obj} classes are different'
    raise_assert_detail(obj, msg, func_w54txot5(left), func_w54txot5(right))


def func_dclhoxqx(attr, left, right, obj='Attributes'):
    """
    Check attributes are equal. Both objects must have attribute.

    Parameters
    ----------
    attr : str
        Attribute name being compared.
    left : object
    right : object
    obj : str, default 'Attributes'
        Specify object name being compared, internally used to show appropriate
        assertion message
    """
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


def func_cnxfls49(seq):
    """Assert that the sequence is sorted."""
    if isinstance(seq, (Index, Series)):
        seq = seq.values
    if isinstance(seq, np.ndarray):
        assert_numpy_array_equal(seq, np.sort(np.array(seq)))
    else:
        assert_extension_array_equal(seq, seq[seq.argsort()])


def func_16hidbb4(left, right, check_dtype=True, check_category_order=True,
    obj='Categorical'):
    """
    Test that Categoricals are equivalent.

    Parameters
    ----------
    left : Categorical
    right : Categorical
    check_dtype : bool, default True
        Check that integer dtype of the codes are the same.
    check_category_order : bool, default True
        Whether the order of the categories should be compared, which
        implies identical integer codes.  If False, only the resulting
        values are compared.  The ordered attribute is
        checked regardless.
    obj : str, default 'Categorical'
        Specify object name being compared, internally used to show appropriate
        assertion message.
    """
    func_m24pb7ya(left, right, Categorical)
    if isinstance(left.categories, RangeIndex) or isinstance(right.
        categories, RangeIndex):
        exact = 'equiv'
    else:
        exact = True
    if check_category_order:
        func_svxy6f7y(left.categories, right.categories, obj=
            f'{obj}.categories', exact=exact)
        assert_numpy_array_equal(left.codes, right.codes, check_dtype=
            check_dtype, obj=f'{obj}.codes')
    else:
        try:
            lc = left.categories.sort_values()
            rc = right.categories.sort_values()
        except TypeError:
            lc, rc = left.categories, right.categories
        func_svxy6f7y(lc, rc, obj=f'{obj}.categories', exact=exact)
        func_svxy6f7y(left.categories.take(left.codes), right.categories.
            take(right.codes), obj=f'{obj}.values', exact=exact)
    func_dclhoxqx('ordered', left, right, obj=obj)


def func_12aozlhs(left, right, exact='equiv', obj='IntervalArray'):
    """
    Test that two IntervalArrays are equivalent.

    Parameters
    ----------
    left, right : IntervalArray
        The IntervalArrays to compare.
    exact : bool or {'equiv'}, default 'equiv'
        Whether to check the Index class, dtype and inferred_type
        are identical. If 'equiv', then RangeIndex can be substituted for
        Index with an int64 dtype as well.
    obj : str, default 'IntervalArray'
        Specify object name being compared, internally used to show appropriate
        assertion message
    """
    func_m24pb7ya(left, right, IntervalArray)
    kwargs = {}
    if left._left.dtype.kind in 'mM':
        kwargs['check_freq'] = False
    assert_equal(left._left, right._left, obj=f'{obj}.left', **kwargs)
    assert_equal(left._right, right._right, obj=f'{obj}.right', **kwargs)
    func_dclhoxqx('closed', left, right, obj=obj)


def func_zxupb8ie(left, right, obj='PeriodArray'):
    func_m24pb7ya(left, right, PeriodArray)
    assert_numpy_array_equal(left._ndarray, right._ndarray, obj=
        f'{obj}._ndarray')
    func_dclhoxqx('dtype', left, right, obj=obj)


def func_olsgbjn6(left, right, obj='DatetimeArray', check_freq=True):
    __tracebackhide__ = True
    func_m24pb7ya(left, right, DatetimeArray)
    assert_numpy_array_equal(left._ndarray, right._ndarray, obj=
        f'{obj}._ndarray')
    if check_freq:
        func_dclhoxqx('freq', left, right, obj=obj)
    func_dclhoxqx('tz', left, right, obj=obj)


def func_jsyou4kw(left, right, obj='TimedeltaArray', check_freq=True):
    __tracebackhide__ = True
    func_m24pb7ya(left, right, TimedeltaArray)
    assert_numpy_array_equal(left._ndarray, right._ndarray, obj=
        f'{obj}._ndarray')
    if check_freq:
        func_dclhoxqx('freq', left, right, obj=obj)


def func_nhmqfudm(obj, message, left, right, diff=None, first_diff=None,
    index_values=None):
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
        right = (
            f'StringDtype(storage={right.storage}, na_value={right.na_value})')
    msg += f'\n[left]:  {left}\n[right]: {right}'
    if diff is not None:
        msg += f'\n[diff]: {diff}'
    if first_diff is not None:
        msg += f'\n{first_diff}'
    raise AssertionError(msg)


def func_p9kr25t9(left, right, strict_nan=False, check_dtype=True, err_msg=
    None, check_same=None, obj='numpy array', index_values=None):
    """
    Check that 'np.ndarray' is equivalent.

    Parameters
    ----------
    left, right : numpy.ndarray or iterable
        The two arrays to be compared.
    strict_nan : bool, default False
        If True, consider NaN and None to be different.
    check_dtype : bool, default True
        Check dtype if both a and b are np.ndarray.
    err_msg : str, default None
        If provided, used as assertion message.
    check_same : None|'copy'|'same', default None
        Ensure left and right refer/do not refer to the same memory area.
    obj : str, default 'numpy array'
        Specify object name being compared, internally used to show appropriate
        assertion message.
    index_values : Index | numpy.ndarray, default None
        optional index (shared by both left and right), used in output.
    """
    __tracebackhide__ = True
    func_e03vn0du(left, right, obj=obj)
    func_m24pb7ya(left, right, np.ndarray)

    def func_sqratx7n(obj):
        return obj.base if getattr(obj, 'base', None) is not None else obj
    left_base = func_sqratx7n(left)
    right_base = func_sqratx7n(right)
    if check_same == 'same':
        if left_base is not right_base:
            raise AssertionError(f'{left_base!r} is not {right_base!r}')
    elif check_same == 'copy':
        if left_base is right_base:
            raise AssertionError(f'{left_base!r} is {right_base!r}')

    def func_o2u2h02e(left, right, err_msg):
        if err_msg is None:
            if left.shape != right.shape:
                func_nhmqfudm(obj, f'{obj} shapes are different', left.
                    shape, right.shape)
            diff = 0
            for left_arr, right_arr in zip(left, right):
                if not array_equivalent(left_arr, right_arr, strict_nan=
                    strict_nan):
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


def func_ikzxwi1v(left, right, check_dtype=True, index_values=None,
    check_exact=lib.no_default, rtol=lib.no_default, atol=lib.no_default,
    obj='ExtensionArray'):
    """
    Check that left and right ExtensionArrays are equal.

    This method compares two ``ExtensionArray`` instances for equality,
    including checks for missing values, the dtype of the arrays, and
    the exactness of the comparison (or tolerance when comparing floats).

    Parameters
    ----------
    left, right : ExtensionArray
        The two arrays to compare.
    check_dtype : bool, default True
        Whether to check if the ExtensionArray dtypes are identical.
    index_values : Index | numpy.ndarray, default None
        Optional index (shared by both left and right), used in output.
    check_exact : bool, default False
        Whether to compare number exactly.

        .. versionchanged:: 2.2.0

            Defaults to True for integer dtypes if none of
            ``check_exact``, ``rtol`` and ``atol`` are specified.
    rtol : float, default 1e-5
        Relative tolerance. Only used when check_exact is False.
    atol : float, default 1e-8
        Absolute tolerance. Only used when check_exact is False.
    obj : str, default 'ExtensionArray'
        Specify object name being compared, internally used to show appropriate
        assertion message.

        .. versionadded:: 2.0.0

    See Also
    --------
    testing.assert_series_equal : Check that left and right ``Series`` are equal.
    testing.assert_frame_equal : Check that left and right ``DataFrame`` are equal.
    testing.assert_index_equal : Check that left and right ``Index`` are equal.

    Notes
    -----
    Missing values are checked separately from valid values.
    A mask of missing values is computed for each and checked to match.
    The remaining all-valid values are cast to object dtype and checked.

    Examples
    --------
    >>> from pandas import testing as tm
    >>> a = pd.Series([1, 2, 3, 4])
    >>> b, c = a.array, a.array
    >>> tm.assert_extension_array_equal(b, c)
    """
    if (check_exact is lib.no_default and rtol is lib.no_default and atol is
        lib.no_default):
        check_exact = is_numeric_dtype(left.dtype) and not is_float_dtype(left
            .dtype) or is_numeric_dtype(right.dtype) and not is_float_dtype(
            right.dtype)
    elif check_exact is lib.no_default:
        check_exact = False
    rtol = rtol if rtol is not lib.no_default else 1e-05
    atol = atol if atol is not lib.no_default else 1e-08
    assert isinstance(left, ExtensionArray), 'left is not an ExtensionArray'
    assert isinstance(right, ExtensionArray), 'right is not an ExtensionArray'
    if check_dtype:
        func_dclhoxqx('dtype', left, right, obj=f'Attributes of {obj}')
    if isinstance(left, DatetimeLikeArrayMixin) and isinstance(right,
        DatetimeLikeArrayMixin) and type(right) == type(left):
        if not check_dtype and left.dtype.kind in 'mM':
            if not isinstance(left.dtype, np.dtype):
                l_unit = cast(DatetimeTZDtype, left.dtype).unit
            else:
                l_unit = np.datetime_data(left.dtype)[0]
            if not isinstance(right.dtype, np.dtype):
                r_unit = cast(DatetimeTZDtype, right.dtype).unit
            else:
                r_unit = np.datetime_data(right.dtype)[0]
            if l_unit != r_unit and compare_mismatched_resolutions(left.
                _ndarray, right._ndarray, operator.eq).all():
                return
        func_p9kr25t9(np.asarray(left.asi8), np.asarray(right.asi8),
            index_values=index_values, obj=obj)
        return
    left_na = np.asarray(left.isna())
    right_na = np.asarray(right.isna())
    func_p9kr25t9(left_na, right_na, obj=f'{obj} NA mask', index_values=
        index_values)
    if isinstance(left.dtype, StringDtype
        ) and left.dtype.storage == 'python' and left.dtype.na_value is np.nan:
        assert np.all([np.isnan(val) for val in left._ndarray[left_na]]
            ), 'wrong missing value sentinels'
    if isinstance(right.dtype, StringDtype
        ) and right.dtype.storage == 'python' and right.dtype.na_value is np.nan:
        assert np.all([np.isnan(val) for val in right._ndarray[right_na]]
            ), 'wrong missing value sentinels'
    left_valid = left[~left_na].to_numpy(dtype=object)
    right_valid = right[~right_na].to_numpy(dtype=object)
    if check_exact:
        func_p9kr25t9(left_valid, right_valid, obj=obj, index_values=
            index_values)
    else:
        _testing.assert_almost_equal(left_valid, right_valid, check_dtype=
            bool(check_dtype), rtol=rtol, atol=atol, obj=obj, index_values=
            index_values)


def func_830pzqsz(left, right, check_dtype=True, check_index_type='equiv',
    check_series_type=True, check_names=True, check_exact=lib.no_default,
    check_datetimelike_compat=False, check_categorical=True,
    check_category_order=True, check_freq=True, check_flags=True, rtol=lib.
    no_default, atol=lib.no_default, obj='Series', *, check_index=True,
    check_like=False):
    """
    Check that left and right Series are equal.

    Parameters
    ----------
    left : Series
        First Series to compare.
    right : Series
        Second Series to compare.
    check_dtype : bool, default True
        Whether to check the Series dtype is identical.
    check_index_type : bool or {'equiv'}, default 'equiv'
        Whether to check the Index class, dtype and inferred_type
        are identical.
    check_series_type : bool, default True
         Whether to check the Series class is identical.
    check_names : bool, default True
        Whether to check the Series and Index names attribute.
    check_exact : bool, default False
        Whether to compare number exactly. This also applies when checking
        Index equivalence.

        .. versionchanged:: 2.2.0

            Defaults to True for integer dtypes if none of
            ``check_exact``, ``rtol`` and ``atol`` are specified.

        .. versionchanged:: 3.0.0

            check_exact for comparing the Indexes defaults to True by
            checking if an Index is of integer dtypes.

    check_datetimelike_compat : bool, default False
        Compare datetime-like which is comparable ignoring dtype.
    check_categorical : bool, default True
        Whether to compare internal Categorical exactly.
    check_category_order : bool, default True
        Whether to compare category order of internal Categoricals.
    check_freq : bool, default True
        Whether to check the `freq` attribute on a DatetimeIndex or TimedeltaIndex.
    check_flags : bool, default True
        Whether to check the `flags` attribute.
    rtol : float, default 1e-5
        Relative tolerance. Only used when check_exact is False.
    atol : float, default 1e-8
        Absolute tolerance. Only used when check_exact is False.
    obj : str, default 'Series'
        Specify object name being compared, internally used to show appropriate
        assertion message.
    check_index : bool, default True
        Whether to check index equivalence. If False, then compare only values.

        .. versionadded:: 1.3.0
    check_like : bool, default False
        If True, ignore the order of the index. Must be False if check_index is False.
        Note: same labels must be with the same data.

        .. versionadded:: 1.5.0

    See Also
    --------
    testing.assert_index_equal : Check that two Indexes are equal.
    testing.assert_frame_equal : Check that two DataFrames are equal.

    Examples
    --------
    >>> from pandas import testing as tm
    >>> a = pd.Series([1, 2, 3, 4])
    >>> b = pd.Series([1, 2, 3, 4])
    >>> tm.assert_series_equal(a, b)
    """
    __tracebackhide__ = True
    if (check_exact is lib.no_default and rtol is lib.no_default and atol is
        lib.no_default):
        check_exact = is_numeric_dtype(left.dtype) and not is_float_dtype(left
            .dtype) or is_numeric_dtype(right.dtype) and not is_float_dtype(
            right.dtype)
        left_index_dtypes = [left.index.dtype
            ] if left.index.nlevels == 1 else left.index.dtypes
        right_index_dtypes = [right.index.dtype
            ] if right.index.nlevels == 1 else right.index.dtypes
        check_exact_index = all(dtype.kind in 'iu' for dtype in
            left_index_dtypes) or all(dtype.kind in 'iu' for dtype in
            right_index_dtypes)
    elif check_exact is lib.no_default:
        check_exact = False
        check_exact_index = False
    else:
        check_exact_index = check_exact
    rtol = rtol if rtol is not lib.no_default else 1e-05
    atol = atol if atol is not lib.no_default else 1e-08
    if not check_index and check_like:
        raise ValueError('check_like must be False if check_index is False')
    func_m24pb7ya(left, right, Series)
    if check_series_type:
        func_e03vn0du(left, right, obj=obj)
    if len(left) != len(right):
        msg1 = f'{len(left)}, {left.index}'
        msg2 = f'{len(right)}, {right.index}'
        func_nhmqfudm(obj, 'Series length are different', msg1, msg2)
    if check_flags:
        assert left.flags == right.flags, f'{left.flags!r} != {right.flags!r}'
    if check_index:
        func_svxy6f7y(left.index, right.index, exact=check_index_type,
            check_names=check_names, check_exact=check_exact_index,
            check_categorical=check_categorical, check_order=not check_like,
            rtol=rtol, atol=atol, obj=f'{obj}.index')
    if check_like:
        left = left.reindex_like(right)
    if check_freq and isinstance(left.index, (DatetimeIndex, TimedeltaIndex)):
        lidx = left.index
        ridx = right.index
        assert lidx.freq == ridx.freq, (lidx.freq, ridx.freq)
    if check_dtype:
        if isinstance(left.dtype, CategoricalDtype) and isinstance(right.
            dtype, CategoricalDtype) and not check_categorical:
            pass
        else:
            func_dclhoxqx('dtype', left, right, obj=f'Attributes of {obj}')
    if check_exact:
        left_values = left._values
        right_values = right._values
        if isinstance(left_values, ExtensionArray) and isinstance(right_values,
            ExtensionArray):
            func_ikzxwi1v(left_values, right_values, check_dtype=
                check_dtype, index_values=left.index, obj=str(obj))
        else:
            lv, rv = left_values, right_values
            if isinstance(left_values, ExtensionArray):
                lv = left_values.to_numpy()
            if isinstance(right_values, ExtensionArray):
                rv = right_values.to_numpy()
            func_p9kr25t9(lv, rv, check_dtype=check_dtype, obj=str(obj),
                index_values=left.index)
    elif check_datetimelike_compat and (needs_i8_conversion(left.dtype) or
        needs_i8_conversion(right.dtype)):
        if not Index(left._values).equals(Index(right._values)):
            msg = (
                f'[datetimelike_compat=True] {left._values} is not equal to {right._values}.'
                )
            raise AssertionError(msg)
    elif isinstance(left.dtype, IntervalDtype) and isinstance(right.dtype,
        IntervalDtype):
        func_12aozlhs(left.array, right.array)
    elif isinstance(left.dtype, CategoricalDtype) or isinstance(right.dtype,
        CategoricalDtype):
        _testing.assert_almost_equal(left._values, right._values, rtol=rtol,
            atol=atol, check_dtype=bool(check_dtype), obj=str(obj),
            index_values=left.index)
    elif isinstance(left.dtype, ExtensionDtype) and isinstance(right.dtype,
        ExtensionDtype):
        func_ikzxwi1v(left._values, right._values, rtol=rtol, atol=atol,
            check_dtype=check_dtype, index_values=left.index, obj=str(obj))
    elif is_extension_array_dtype_and_needs_i8_conversion(left.dtype, right
        .dtype) or is_extension_array_dtype_and_needs_i8_conversion(right.
        dtype, left.dtype):
        func_ikzxwi1v(left._values, right._values, check_dtype=check_dtype,
            index_values=left.index, obj=str(obj))
    elif needs_i8_conversion(left.dtype) and needs_i8_conversion(right.dtype):
        func_ikzxwi1v(left._values, right._values, check_dtype=check_dtype,
            index_values=left.index, obj=str(obj))
    else:
        _testing.assert_almost_equal(left._values, right._values, rtol=rtol,
            atol=atol, check_dtype=bool(check_dtype), obj=str(obj),
            index_values=left.index)
    if check_names:
        func_dclhoxqx('name', left, right, obj=obj)
    if check_categorical:
        if isinstance(left.dtype, CategoricalDtype) or isinstance(right.
            dtype, CategoricalDtype):
            func_16hidbb4(left._values, right._values, obj=
                f'{obj} category', check_category_order=check_category_order)


def func_od2k9qa8(left, right, check_dtype=True, check_index_type='equiv',
    check_column_type='equiv', check_frame_type=True, check_names=True,
    by_blocks=False, check_exact=lib.no_default, check_datetimelike_compat=
    False, check_categorical=True, check_like=False, check_freq=True,
    check_flags=True, rtol=lib.no_default, atol=lib.no_default, obj='DataFrame'
    ):
    """
    Check that left and right DataFrame are equal.

    This function is intended to compare two DataFrames and output any
    differences. It is mostly intended for use in unit tests.
    Additional parameters allow varying the strictness of the
    equality checks performed.

    Parameters
    ----------
    left : DataFrame
        First DataFrame to compare.
    right : DataFrame
        Second DataFrame to compare.
    check_dtype : bool, default True
        Whether to check the DataFrame dtype is identical.
    check_index_type : bool or {'equiv'}, default 'equiv'
        Whether to check the Index class, dtype and inferred_type
        are identical.
    check_column_type : bool or {'equiv'}, default 'equiv'
        Whether to check the columns class, dtype and inferred_type
        are identical. Is passed as the ``exact`` argument of
        :func:`assert_index_equal`.
    check_frame_type : bool, default True
        Whether to check the DataFrame class is identical.
    check_names : bool, default True
        Whether to check that the `names` attribute for both the `index`
        and `column` attributes of the DataFrame is identical.
    by_blocks : bool, default False
        Specify how to compare internal data. If False, compare by columns.
        If True, compare by blocks.
    check_exact : bool, default False
        Whether to compare number exactly. If False, the comparison uses the
        relative tolerance (``rtol``) and absolute tolerance (``atol``)
        parameters to determine if two values are considered close,
        according to the formula: ``|a - b| <= (atol + rtol * |b|)``.

        .. versionchanged:: 2.2.0

            Defaults to True for integer dtypes if none of
            ``check_exact``, ``rtol`` and ``atol`` are specified.
    check_datetimelike_compat : bool, default False
        Compare datetime-like which is comparable ignoring dtype.
    check_categorical : bool, default True
        Whether to compare internal Categorical exactly.
    check_like : bool, default False
        If True, ignore the order of index & columns.
        Note: index labels must match their respective rows
        (same as in columns) - same labels must be with the same data.
    check_freq : bool, default True
        Whether to check the `freq` attribute on a DatetimeIndex or TimedeltaIndex.
    check_flags : bool, default True
        Whether to check the `flags` attribute.
    rtol : float, default 1e-5
        Relative tolerance. Only used when check_exact is False.
    atol : float, default 1e-8
        Absolute tolerance. Only used when check_exact is False.
    obj : str, default 'DataFrame'
        Specify object name being compared, internally used to show appropriate
        assertion message.

    See Also
    --------
    assert_series_equal : Equivalent method for asserting Series equality.
    DataFrame.equals : Check DataFrame equality.

    Examples
    --------
    This example shows comparing two DataFrames that are equal
    but with columns of differing dtypes.

    >>> from pandas.testing import assert_frame_equal
    >>> df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    >>> df2 = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})

    df1 equals itself.

    >>> assert_frame_equal(df1, df1)

    df1 differs from df2 as column 'b' is of a different type.

    >>> assert_frame_equal(df1, df2)
    Traceback (most recent call last):
    ...
    AssertionError: Attributes of DataFrame.iloc[:, 1] (column name="b") are different

    Attribute "dtype" are different
    [left]:  int64
    [right]: float64

    Ignore differing dtypes in columns with check_dtype.

    >>> assert_frame_equal(df1, df2, check_dtype=False)
    """
    __tracebackhide__ = True
    _rtol = rtol if rtol is not lib.no_default else 1e-05
    _atol = atol if atol is not lib.no_default else 1e-08
    _check_exact = check_exact if check_exact is not lib.no_default else False
    func_m24pb7ya(left, right, DataFrame)
    if check_frame_type:
        assert isinstance(left, type(right))
    if left.shape != right.shape:
        func_nhmqfudm(obj, f'{obj} shape mismatch', f'{left.shape!r}',
            f'{right.shape!r}')
    if check_flags:
        assert left.flags == right.flags, f'{left.flags!r} != {right.flags!r}'
    func_svxy6f7y(left.index, right.index, exact=check_index_type,
        check_names=check_names, check_exact=_check_exact,
        check_categorical=check_categorical, check_order=not check_like,
        rtol=_rtol, atol=_atol, obj=f'{obj}.index')
    func_svxy6f7y(left.columns, right.columns, exact=check_column_type,
        check_names=check_names, check_exact=_check_exact,
        check_categorical=check_categorical, check_order=not check_like,
        rtol=_rtol, atol=_atol, obj=f'{obj}.columns')
    if check_like:
        left = left.reindex_like(right)
    if by_blocks:
        rblocks = right._to_dict_of_blocks()
        lblocks = left._to_dict_of_blocks()
        for dtype in list(set(list(lblocks.keys()) + list(rblocks.keys()))):
            assert dtype in lblocks
            assert dtype in rblocks
            func_od2k9qa8(lblocks[dtype], rblocks[dtype], check_dtype=
                check_dtype, obj=obj)
    else:
        for i, col in enumerate(left.columns):
            lcol = left._ixs(i, axis=1)
            rcol = right._ixs(i, axis=1)
            func_830pzqsz(lcol, rcol, check_dtype=check_dtype,
                check_index_type=check_index_type, check_exact=check_exact,
                check_names=check_names, check_datetimelike_compat=
                check_datetimelike_compat, check_categorical=
                check_categorical, check_freq=check_freq, obj=
                f'{obj}.iloc[:, {i}] (column name="{col}")', rtol=rtol,
                atol=atol, check_index=False, check_flags=False)


def func_u5z97rsr(left, right, **kwargs):
    """
    Wrapper for tm.assert_*_equal to dispatch to the appropriate test function.

    Parameters
    ----------
    left, right : Index, Series, DataFrame, ExtensionArray, or np.ndarray
        The two items to be compared.
    **kwargs
        All keyword arguments are passed through to the underlying assert method.
    """
    __tracebackhide__ = True
    if isinstance(left, Index):
        func_svxy6f7y(left, right, **kwargs)
        if isinstance(left, (DatetimeIndex, TimedeltaIndex)):
            assert left.freq == right.freq, (left.freq, right.freq)
    elif isinstance(left, Series):
        func_830pzqsz(left, right, **kwargs)
    elif isinstance(left, DataFrame):
        func_od2k9qa8(left, right, **kwargs)
    elif isinstance(left, IntervalArray):
        func_12aozlhs(left, right, **kwargs)
    elif isinstance(left, PeriodArray):
        func_zxupb8ie(left, right, **kwargs)
    elif isinstance(left, DatetimeArray):
        func_olsgbjn6(left, right, **kwargs)
    elif isinstance(left, TimedeltaArray):
        func_jsyou4kw(left, right, **kwargs)
    elif isinstance(left, ExtensionArray):
        func_ikzxwi1v(left, right, **kwargs)
    elif isinstance(left, np.ndarray):
        func_p9kr25t9(left, right, **kwargs)
    elif isinstance(left, str):
        assert kwargs == {}
        assert left == right
    else:
        assert kwargs == {}
        func_od528zmn(left, right)


def func_43hguk6r(left, right):
    """
    Check that the left and right SparseArray are equal.

    Parameters
    ----------
    left : SparseArray
    right : SparseArray
    """
    func_m24pb7ya(left, right, pd.arrays.SparseArray)
    func_p9kr25t9(left.sp_values, right.sp_values)
    assert isinstance(left.sp_index, SparseIndex)
    assert isinstance(right.sp_index, SparseIndex)
    left_index = left.sp_index
    right_index = right.sp_index
    if not left_index.equals(right_index):
        func_nhmqfudm('SparseArray.index', 'index are not equal',
            left_index, right_index)
    else:
        pass
    func_dclhoxqx('fill_value', left, right)
    func_dclhoxqx('dtype', left, right)
    func_p9kr25t9(left.to_dense(), right.to_dense())


def func_54idg729(iterable, dic):
    for k in iterable:
        assert k in dic, f'Did not contain item: {k!r}'


def func_cqxk4ppd(iter1, iter2, **eql_kwargs):
    """
    iter1, iter2: iterables that produce elements
    comparable with assert_almost_equal

    Checks that the elements are equal, but not
    the same object. (Does not check that items
    in sequences are also not the same object)
    """
    for elem1, elem2 in zip(iter1, iter2):
        func_od528zmn(elem1, elem2, **eql_kwargs)
        msg = (
            f'Expected object {type(elem1)!r} and object {type(elem2)!r} to be different objects, but they were the same object.'
            )
        assert elem1 is not elem2, msg


def func_rao4hcir(left_dtype, right_dtype):
    """
    Checks that we have the combination of an ExtensionArraydtype and
    a dtype that should be converted to int64

    Returns
    -------
    bool

    Related to issue #37609
    """
    return isinstance(left_dtype, ExtensionDtype) and needs_i8_conversion(
        right_dtype)


def func_f975jslv(ser, l_slc, i_slc):
    """
    Check that ser.iloc[i_slc] matches ser.loc[l_slc] and, if applicable,
    ser[l_slc].
    """
    expected = ser.iloc[i_slc]
    func_830pzqsz(ser.loc[l_slc], expected)
    if not is_integer_dtype(ser.index):
        func_830pzqsz(ser[l_slc], expected)


def func_2cgws4ez(left, right=None):
    """
    Check that ._metadata attributes are equivalent.
    """
    for attr in left._metadata:
        val = getattr(left, attr, None)
        if right is None:
            assert val is None
        else:
            assert val == getattr(right, attr, None)
