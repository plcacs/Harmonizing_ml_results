"""
Functions for preparing various inputs passed to the DataFrame or Series
constructors before passing them to a BlockManager.
"""

from __future__ import annotations

from collections import abc
from typing import (
    TYPE_CHECKING,
    Any,
    Union,
)

import numpy as np
from numpy import ma

from pandas._config import using_string_dtype

from pandas._libs import lib

from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.cast import (
    construct_1d_arraylike_from_scalar,
    dict_compat,
    maybe_cast_to_datetime,
    maybe_convert_platform,
    maybe_infer_to_datetimelike,
)
from pandas.core.dtypes.common import (
    is_1d_only_ea_dtype,
    is_integer_dtype,
    is_list_like,
    is_named_tuple,
    is_object_dtype,
    is_scalar,
)
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)
from pandas.core.dtypes.missing import isna

from pandas.core import (
    algorithms,
    common as com,
)
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.string_ import StringDtype
from pandas.core.construction import (
    array as pd_array,
    extract_array,
    range_to_ndarray,
    sanitize_array,
)
from pandas.core.indexes.api import (
    DatetimeIndex,
    Index,
    TimedeltaIndex,
    default_index,
    ensure_index,
    get_objs_combined_axis,
    maybe_sequence_to_range,
    union_indexes,
)
from pandas.core.internals.blocks import (
    BlockPlacement,
    ensure_block_shape,
    new_block,
    new_block_2d,
)
from pandas.core.internals.managers import (
    create_block_manager_from_blocks,
    create_block_manager_from_column_arrays,
)

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Sequence,
    )

    from pandas._typing import (
        ArrayLike,
        DtypeObj,
        Manager,
        npt,
    )
# ---------------------------------------------------------------------
# BlockManager Interface


def arrays_to_mgr(
    arrays: Sequence[ArrayLike],
    columns: Any,
    index: Any,
    *,
    dtype: DtypeObj | None = None,
    verify_integrity: bool = True,
    consolidate: bool = True,
) -> Manager:
    """
    Segregate Series based on type and coerce into matrices.

    Needs to handle a lot of exceptional cases.
    """
    if verify_integrity:
        # figure out the index, if necessary
        if index is None:
            index = _extract_index(arrays)
        else:
            index = ensure_index(index)

        # don't force copy because getting jammed in an ndarray anyway
        arrays, refs = _homogenize(arrays, index, dtype)
        # _homogenize ensures
        #  - all(len(x) == len(index) for x in arrays)
        #  - all(x.ndim == 1 for x in arrays)
        #  - all(isinstance(x, (np.ndarray, ExtensionArray)) for x in arrays)
        #  - all(type(x) is not NumpyExtensionArray for x in arrays)

    else:
        index = ensure_index(index)
        arrays = [extract_array(x, extract_numpy=True) for x in arrays]
        # with _from_arrays, the passed arrays should never be Series objects
        refs = [None] * len(arrays)

        # Reached via DataFrame._from_arrays; we do minimal validation here
        for arr in arrays:
            if (
                not isinstance(arr, (np.ndarray, ExtensionArray))
                or arr.ndim != 1
                or len(arr) != len(index)
            ):
                raise ValueError(
                    "Arrays must be 1-dimensional np.ndarray or ExtensionArray "
                    "with length matching len(index)"
                )

    columns = ensure_index(columns)
    if len(columns) != len(arrays):
        raise ValueError("len(arrays) must match len(columns)")

    # from BlockManager perspective
    axes = [columns, index]

    return create_block_manager_from_column_arrays(
        arrays, axes, consolidate=consolidate, refs=refs
    )


def rec_array_to_mgr(
    data: np.rec.recarray | np.ndarray,
    index: Any,
    columns: Any,
    dtype: DtypeObj | None,
    copy: bool,
) -> Manager:
    """
    Extract from a masked rec array and create the manager.
    """
    # essentially process a record array then fill it
    fdata = ma.getdata(data)
    if index is None:
        index = default_index(len(fdata))
    else:
        index = ensure_index(index)

    if columns is not None:
        columns = ensure_index(columns)
    arrays, arr_columns = to_arrays(fdata, columns)

    # create the manager

    arrays, arr_columns = reorder_arrays(arrays, arr_columns, columns, len(index))
    if columns is None:
        columns = arr_columns

    mgr = arrays_to_mgr(arrays, columns, index, dtype=dtype)

    if copy:
        mgr = mgr.copy()
    return mgr


# ---------------------------------------------------------------------
# DataFrame Constructor Interface


def ndarray_to_mgr(
    values: Any,
    index: Any,
    columns: Any,
    dtype: DtypeObj | None,
    copy: bool,
) -> Manager:
    # used in DataFrame.__init__
    # input must be a ndarray, list, Series, Index, ExtensionArray
    infer_object = not isinstance(values, (ABCSeries, Index, ExtensionArray))

    if isinstance(values, ABCSeries):
        if columns is None:
            if values.name is not None:
                columns = Index([values.name])
        if index is None:
            index = values.index
        else:
            values = values.reindex(index)

        # zero len case (GH #2234)
        if not len(values) and columns is not None and len(columns):
            values = np.empty((0, 1), dtype=object)

    vdtype = getattr(values, "dtype", None)
    refs = None
    if is_1d_only_ea_dtype(vdtype) or is_1d_only_ea_dtype(dtype):
        # GH#19157

        if isinstance(values, (np.ndarray, ExtensionArray)) and values.ndim > 1:
            # GH#12513 a EA dtype passed with a 2D array, split into
            #  multiple EAs that view the values
            # error: No overload variant of "__getitem__" of "ExtensionArray"
            # matches argument type "Tuple[slice, int]"
            values = [
                values[:, n]  # type: ignore[call-overload]
                for n in range(values.shape[1])
            ]
        else:
            values = [values]

        if columns is None:
            columns = Index(range(len(values)))
        else:
            columns = ensure_index(columns)

        return arrays_to_mgr(values, columns, index, dtype=dtype)

    elif isinstance(vdtype, ExtensionDtype):
        # i.e. Datetime64TZ, PeriodDtype; cases with is_1d_only_ea_dtype(vdtype)
        #  are already caught above
        values = extract_array(values, extract_numpy=True)
        if copy:
            values = values.copy()
        if values.ndim == 1:
            values = values.reshape(-1, 1)

    elif isinstance(values, (ABCSeries, Index)):
        if not copy and (dtype is None or astype_is_view(values.dtype, dtype)):
            refs = values._references

        if copy:
            values = values._values.copy()
        else:
            values = values._values

        values = _ensure_2d(values)

    elif isinstance(values, (np.ndarray, ExtensionArray)):
        # drop subclass info
        if copy and (dtype is None or astype_is_view(values.dtype, dtype)):
            # only force a copy now if copy=True was requested
            # and a subsequent `astype` will not already result in a copy
            values = np.array(values, copy=True, order="F")
        else:
            values = np.asarray(values)
        values = _ensure_2d(values)

    else:
        # by definition an array here
        # the dtypes will be coerced to a single dtype
        values = _prep_ndarraylike(values, copy=copy)

    if dtype is not None and values.dtype != dtype:
        # GH#40110 see similar check inside sanitize_array
        values = sanitize_array(
            values,
            None,
            dtype=dtype,
            copy=copy,
            allow_2d=True,
        )

    # _prep_ndarraylike ensures that values.ndim == 2 at this point
    index, columns = _get_axes(
        values.shape[0], values.shape[1], index=index, columns=columns
    )

    _check_values_indices_shape_match(values, index, columns)

    values = values.T

    # if we don't have a dtype specified, then try to convert objects
    # on the entire block; this is to convert if we have datetimelike's
    # embedded in an object type
    if dtype is None and infer_object and is_object_dtype(values.dtype):
        obj_columns = list(values)
        maybe_datetime = [maybe_infer_to_datetimelike(x) for x in obj_columns]
        # don't convert (and copy) the objects if no type inference occurs
        if any(x is not y for x, y in zip(obj_columns, maybe_datetime)):
            block_values = [
                new_block_2d(ensure_block_shape(dval, 2), placement=BlockPlacement(n))
                for n, dval in enumerate(maybe_datetime)
            ]
        else:
            bp = BlockPlacement(slice(len(columns)))
            nb = new_block_2d(values, placement=bp, refs=refs)
            block_values = [nb]
    elif dtype is None and values.dtype.kind == "U" and using_string_dtype():
        dtype = StringDtype(na_value=np.nan)

        obj_columns = list(values)
        block_values = [
            new_block(
                dtype.construct_array_type()._from_sequence(data, dtype=dtype),
                BlockPlacement(slice(i, i + 1)),
                ndim=2,
            )
            for i, data in enumerate(obj_columns)
        ]

    else:
        bp = BlockPlacement(slice(len(columns)))
        nb = new_block_2d(values, placement=bp, refs=refs)
        block_values = [nb]

    if len(columns) == 0:
        # TODO: check len(values) == 0?
        block_values = []

    return create_block_manager_from_blocks(
        block_values, [columns, index], verify_integrity=False
    )


def _check_values_indices_shape_match(
    values: np.ndarray, index: Index, columns: Any
) -> None:
    """
    Check that the shape implied by our axes matches the actual shape of the
    data.
    """
    if values.shape[1] != len(columns) or values.shape[0] != len(index):
        # Could let this raise in Block constructor, but we get a more
        #  helpful exception message this way.
        if values.shape[0] == 0 < len(index):
            raise ValueError("Empty data passed with indices specified.")

        passed = values.shape
        implied = (len(index), len(columns))
        raise ValueError(f"Shape of passed values is {passed}, indices imply {implied}")


def dict_to_mgr(
    data: dict[Any, Any],
    index: Any,
    columns: Any,
    *,
    dtype: DtypeObj | None = None,
    copy: bool = True,
) -> Manager:
    """
    Segregate Series based on type and coerce into matrices.
    Needs to handle a lot of exceptional cases.

    Used in DataFrame.__init__
    """
    arrays: Sequence[Any]

    if columns is not None:
        columns = ensure_index(columns)
        arrays = [np.nan] * len(columns)
        midxs = set()
        data_keys = ensure_index(data.keys())  # type: ignore[arg-type]
        data_values = list(data.values())

        for i, column in enumerate(columns):
            try:
                idx = data_keys.get_loc(column)
            except KeyError:
                midxs.add(i)
                continue
            array = data_values[idx]
            arrays[i] = array
            if is_scalar(array) and isna(array):
                midxs.add(i)

        if index is None:
            # GH10856
            # raise ValueError if only scalars in dict
            if midxs:
                index = _extract_index(
                    [array for i, array in enumerate(arrays) if i not in midxs]
                )
            else:
                index = _extract_index(arrays)
        else:
            index = ensure_index(index)

        # no obvious "empty" int column
        if midxs and not is_integer_dtype(dtype):
            # GH#1783
            for i in midxs:
                arr = construct_1d_arraylike_from_scalar(
                    arrays[i],
                    len(index),
                    dtype if dtype is not None else np.dtype("object"),
                )
                arrays[i] = arr

    else:
        keys = maybe_sequence_to_range(list(data.keys()))
        columns = Index(keys) if keys else default_index(0)
        arrays = [com.maybe_iterable_to_list(data[k]) for k in keys]

    if copy:
        # We only need to copy arrays that will not get consolidated, i.e.
        #  only EA arrays
        arrays = [
            x.copy()
            if isinstance(x, ExtensionArray)
            else x.copy(deep=True)
            if (
                isinstance(x, Index)
                or (isinstance(x, ABCSeries) and is_1d_only_ea_dtype(x.dtype))
            )
            else x
            for x in arrays
        ]

    return arrays_to_mgr(arrays, columns, index, dtype=dtype, consolidate=copy)


def nested_data_to_arrays(
    data: Sequence[Any],
    columns: Index | None,
    index: Index | None,
    dtype: DtypeObj | None,
) -> tuple[list[ArrayLike], Index, Index]:
    """
    Convert a single sequence of arrays to multiple arrays.
    """
    # By the time we get here we have already checked treat_as_nested(data)

    if is_named_tuple(data[0]) and columns is None:
        columns = ensure_index(data[0]._fields)

    arrays, columns = to_arrays(data, columns, dtype=dtype)
    columns = ensure_index(columns)

    if index is None:
        if isinstance(data[0], ABCSeries):
            index = _get_names_from_index(data)
        else:
            index = default_index(len(data))

    return arrays, columns, index


def treat_as_nested(data: Sequence[Any]) -> bool:
    """
    Check if we should use nested_data_to_arrays.
    """
    return (
        len(data) > 0
        and is_list_like(data[0])
        and getattr(data[0], "ndim", 1) == 1
        and not (isinstance(data, ExtensionArray) and data.ndim == 2)
    )


# ---------------------------------------------------------------------


def _prep_ndarraylike(values: Any, copy: bool = True) -> np.ndarray:
    # values is specifically _not_ ndarray, EA, Index, or Series
    # We only get here with `not treat_as_nested(values)`

    if len(values) == 0:
        # TODO: check for length-zero range, in which case return int64 dtype?
        # TODO: reuse anything in try_cast?
        return np.empty((0, 0), dtype=object)
    elif isinstance(values, range):
        arr = range_to_ndarray(values)
        return arr[..., np.newaxis]

    def convert(v: Any) -> Any:
        if not is_list_like(v) or isinstance(v, ABCDataFrame):
            return v

        v = extract_array(v, extract_numpy=True)
        res = maybe_convert_platform(v)
        # We don't do maybe_infer_to_datetimelike here bc we will end up doing
        #  it column-by-column in ndarray_to_mgr
        return res

    # we could have a 1-dim or 2-dim list here
    # this is equiv of np.asarray, but does object conversion
    # and platform dtype preservation
    # does not convert e.g. [1, "a", True] to ["1", "a", "True"] like
    #  np.asarray would
    if is_list_like(values[0]):
        values = np.array([convert(v) for v in values])
    elif isinstance(values[0], np.ndarray) and values[0].ndim == 0:
        # GH#21861 see test_constructor_list_of_lists
        values = np.array([convert(v) for v in values])
    else:
        values = convert(values)

    return _ensure_2d(values)


def _ensure_2d(values: np.ndarray) -> np.ndarray:
    """
    Reshape 1D values, raise on anything else other than 2D.
    """
    if values.ndim == 1:
        values = values.reshape((values.shape[0], 1))
    elif values.ndim != 2:
        raise ValueError(f"Must pass 2-d input. shape={values.shape}")
    return values


def _homogenize(
    data: Sequence[Any], index: Index, dtype: DtypeObj | None
) -> tuple[list[ArrayLike], list[Any]]:
    oindex = None
    homogenized = []
    # if the original array-like in `data` is a Series, keep track of this Series' refs
    refs: list[Any] = []

    for val in data:
        if isinstance(val, (ABCSeries, Index)):
            if dtype is not None:
                val = val.astype(dtype)
            if isinstance(val, ABCSeries) and val.index is not index:
                # Forces alignment. No need to copy data since we
                # are putting it into an ndarray later
                val = val.reindex(index)
            refs.append(val._references)
            val = val._values
        else:
            if isinstance(val, dict):
               