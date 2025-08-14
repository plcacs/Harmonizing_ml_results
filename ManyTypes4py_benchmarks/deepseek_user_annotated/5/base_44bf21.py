from __future__ import annotations

from collections import abc
from datetime import datetime
import functools
from itertools import zip_longest
import operator
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    NoReturn,
    cast,
    final,
    overload,
)
import warnings

import numpy as np

from pandas._config import get_option

from pandas._libs import (
    NaT,
    algos as libalgos,
    index as libindex,
    lib,
    writers,
)
from pandas._libs.internals import BlockValuesRefs
import pandas._libs.join as libjoin
from pandas._libs.lib import (
    is_datetime_array,
    no_default,
)
from pandas._libs.tslibs import (
    IncompatibleFrequency,
    OutOfBoundsDatetime,
    Timestamp,
    tz_compare,
)
from pandas._typing import (
    AnyAll,
    ArrayLike,
    Axes,
    Axis,
    AxisInt,
    DropKeep,
    Dtype,
    DtypeObj,
    F,
    IgnoreRaise,
    IndexLabel,
    IndexT,
    JoinHow,
    Level,
    NaPosition,
    ReindexMethod,
    Self,
    Shape,
    SliceType,
    npt,
)
from pandas.compat.numpy import function as nv
from pandas.errors import (
    DuplicateLabelError,
    InvalidIndexError,
)
from pandas.util._decorators import (
    Appender,
    cache_readonly,
    doc,
    set_module,
)
from pandas.util._exceptions import (
    find_stack_level,
    rewrite_exception,
)

from pandas.core.dtypes.astype import (
    astype_array,
    astype_is_view,
)
from pandas.core.dtypes.cast import (
    LossySetitemError,
    can_hold_element,
    common_dtype_categorical_compat,
    find_result_type,
    infer_dtype_from,
    maybe_cast_pointwise_result,
    np_can_hold_element,
)
from pandas.core.dtypes.common import (
    ensure_int64,
    ensure_object,
    ensure_platform_int,
    is_any_real_numeric_dtype,
    is_bool_dtype,
    is_ea_or_datetimelike_dtype,
    is_float,
    is_hashable,
    is_integer,
    is_iterator,
    is_list_like,
    is_numeric_dtype,
    is_object_dtype,
    is_scalar,
    is_signed_integer_dtype,
    is_string_dtype,
    needs_i8_conversion,
    pandas_dtype,
    validate_all_hashable,
)
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
    ArrowDtype,
    CategoricalDtype,
    DatetimeTZDtype,
    ExtensionDtype,
    IntervalDtype,
    PeriodDtype,
    SparseDtype,
)
from pandas.core.dtypes.generic import (
    ABCCategoricalIndex,
    ABCDataFrame,
    ABCDatetimeIndex,
    ABCIntervalIndex,
    ABCMultiIndex,
    ABCPeriodIndex,
    ABCRangeIndex,
    ABCSeries,
    ABCTimedeltaIndex,
)
from pandas.core.dtypes.inference import is_dict_like
from pandas.core.dtypes.missing import (
    array_equivalent,
    is_valid_na_for_dtype,
    isna,
)

from pandas.core import (
    arraylike,
    nanops,
    ops,
)
from pandas.core.accessor import Accessor
import pandas.core.algorithms as algos
from pandas.core.array_algos.putmask import (
    setitem_datetimelike_compat,
    validate_putmask,
)
from pandas.core.arrays import (
    ArrowExtensionArray,
    BaseMaskedArray,
    Categorical,
    DatetimeArray,
    ExtensionArray,
    TimedeltaArray,
)
from pandas.core.arrays.string_ import (
    StringArray,
    StringDtype,
)
from pandas.core.base import (
    IndexOpsMixin,
    PandasObject,
)
import pandas.core.common as com
from pandas.core.construction import (
    ensure_wrapped_if_datetimelike,
    extract_array,
    sanitize_array,
)
from pandas.core.indexers import (
    disallow_ndim_indexing,
    is_valid_positional_slice,
)
from pandas.core.indexes.frozen import FrozenList
from pandas.core.missing import clean_reindex_fill_method
from pandas.core.ops import get_op_result_name
from pandas.core.sorting import (
    ensure_key_mapped,
    get_group_index_sorter,
    nargsort,
)
from pandas.core.strings.accessor import StringMethods

from pandas.io.formats.printing import (
    PrettyDict,
    default_pprint,
    format_object_summary,
    pprint_thing,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Hashable,
        Iterable,
        Sequence,
    )

    from pandas import (
        CategoricalIndex,
        DataFrame,
        MultiIndex,
        Series,
    )
    from pandas.core.arrays import (
        IntervalArray,
        PeriodArray,
    )

__all__ = ["Index"]

_unsortable_types = frozenset(("mixed", "mixed-integer"))

_index_doc_kwargs: dict[str, str] = {
    "klass": "Index",
    "inplace": "",
    "target_klass": "Index",
    "raises_section": "",
    "unique": "Index",
    "duplicated": "np.ndarray",
}
_index_shared_docs: dict[str, str] = {}
str_t = str

_dtype_obj = np.dtype("object")

_masked_engines = {
    "Complex128": libindex.MaskedComplex128Engine,
    "Complex64": libindex.MaskedComplex64Engine,
    "Float64": libindex.MaskedFloat64Engine,
    "Float32": libindex.MaskedFloat32Engine,
    "UInt64": libindex.MaskedUInt64Engine,
    "UInt32": libindex.MaskedUInt32Engine,
    "UInt16": libindex.MaskedUInt16Engine,
    "UInt8": libindex.MaskedUInt8Engine,
    "Int64": libindex.MaskedInt64Engine,
    "Int32": libindex.MaskedInt32Engine,
    "Int16": libindex.MaskedInt16Engine,
    "Int8": libindex.MaskedInt8Engine,
    "boolean": libindex.MaskedBoolEngine,
    "double[pyarrow]": libindex.MaskedFloat64Engine,
    "float64[pyarrow]": libindex.MaskedFloat64Engine,
    "float32[pyarrow]": libindex.MaskedFloat32Engine,
    "float[pyarrow]": libindex.MaskedFloat32Engine,
    "uint64[pyarrow]": libindex.MaskedUInt64Engine,
    "uint32[pyarrow]": libindex.MaskedUInt32Engine,
    "uint16[pyarrow]": libindex.MaskedUInt16Engine,
    "uint8[pyarrow]": libindex.MaskedUInt8Engine,
    "int64[pyarrow]": libindex.MaskedInt64Engine,
    "int32[pyarrow]": libindex.MaskedInt32Engine,
    "int16[pyarrow]": libindex.MaskedInt16Engine,
    "int8[pyarrow]": libindex.MaskedInt8Engine,
    "bool[pyarrow]": libindex.MaskedBoolEngine,
}


def _maybe_return_indexers(meth: F) -> F:
    """
    Decorator to simplify 'return_indexers' checks in Index.join.
    """

    @functools.wraps(meth)
    def join(
        self,
        other: Index,
        *,
        how: JoinHow = "left",
        level=None,
        return_indexers: bool = False,
        sort: bool = False,
    ):
        join_index, lidx, ridx = meth(self, other, how=how, level=level, sort=sort)
        if not return_indexers:
            return join_index

        if lidx is not None:
            lidx = ensure_platform_int(lidx)
        if ridx is not None:
            ridx = ensure_platform_int(ridx)
        return join_index, lidx, ridx

    return cast(F, join)


def _new_Index(cls, d):
    """
    This is called upon unpickling, rather than the default which doesn't
    have arguments and breaks __new__.
    """
    # required for backward compat, because PI can't be instantiated with
    # ordinals through __new__ GH #13277
    if issubclass(cls, ABCPeriodIndex):
        from pandas.core.indexes.period import _new_PeriodIndex

        return _new_PeriodIndex(cls, **d)

    if issubclass(cls, ABCMultiIndex):
        if "labels" in d and "codes" not in d:
            # GH#23752 "labels" kwarg has been replaced with "codes"
            d["codes"] = d.pop("labels")

        # Since this was a valid MultiIndex at pickle-time, we don't need to
        #  check validty at un-pickle time.
        d["verify_integrity"] = False

    elif "dtype" not in d and "data" in d:
        # Prevent Index.__new__ from conducting inference;
        #  "data" key not in RangeIndex
        d["dtype"] = d["data"].dtype
    return cls.__new__(cls, **d)


@set_module("pandas")
class Index(IndexOpsMixin, PandasObject):
    """
    Immutable sequence used for indexing and alignment.

    The basic object storing axis labels for all pandas objects.

    .. versionchanged:: 2.0.0

       Index can hold all numpy numeric dtypes (except float16). Previously only
       int64/uint64/float64 dtypes were accepted.

    Parameters
    ----------
    data : array-like (1-dimensional)
        An array-like structure containing the data for the index. This could be a
        Python list, a NumPy array, or a pandas Series.
    dtype : str, numpy.dtype, or ExtensionDtype, optional
        Data type for the output Index. If not specified, this will be
        inferred from `data`.
        See the :ref:`user guide <basics.dtypes>` for more usages.
    copy : bool, default False
        Copy input data.
    name : object
        Name to be stored in the index.
    tupleize_cols : bool (default: True)
        When True, attempt to create a MultiIndex if possible.

    See Also
    --------
    RangeIndex : Index implementing a monotonic integer range.
    CategoricalIndex : Index of :class:`Categorical` s.
    MultiIndex : A multi-level, or hierarchical Index.
    IntervalIndex : An Index of :class:`Interval` s.
    DatetimeIndex : Index of datetime64 data.
    TimedeltaIndex : Index of timedelta64 data.
    PeriodIndex : Index of Period data.

    Notes
    -----
    An Index instance can **only** contain hashable objects.
    An Index instance *can not* hold numpy float16 dtype.

    Examples
    --------
    >>> pd.Index([1, 2, 3])
    Index([1, 2, 3], dtype='int64')

    >>> pd.Index(list("abc"))
    Index(['a', 'b', 'c'], dtype='object')

    >>> pd.Index([1, 2, 3], dtype="uint8")
    Index([1, 2, 3], dtype='uint8')
    """

    # similar to __array_priority__, positions Index after Series and DataFrame
    #  but before ExtensionArray.  Should NOT be overridden by subclasses.
    __pandas_priority__ = 2000

    # Cython methods; see github.com/cython/cython/issues/2647
    #  for why we need to wrap these instead of making them class attributes
    # Moreover, cython will choose the appropriate-dtyped sub-function
    #  given the dtypes of the passed arguments

    @final
    def _left_indexer_unique(self, other: Self) -> npt.NDArray[np.intp]:
        # Caller is responsible for ensuring other.dtype == self.dtype
        sv = self._get_join_target()
        ov = other._get_join_target()
        # similar but not identical to ov.searchsorted(sv)
        return libjoin.left_join_indexer_unique(sv, ov)

    @final
    def _left_indexer(
        self, other: Self
    ) -> tuple[ArrayLike, npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        # Caller is responsible for ensuring other.dtype == self.dtype
        sv = self._get_join_target()
        ov = other._get_join_target()
        joined_ndarray, lidx, ridx = libjoin.left_join_indexer(sv, ov)
        joined = self._from_join_target(joined_ndarray)
        return joined, lidx, ridx

    @final
    def _inner_indexer(
        self, other: Self
    ) -> tuple[ArrayLike, npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        # Caller is responsible for ensuring other.dtype == self.dtype
        sv = self._get_join_target()
        ov = other._get_join_target()
        joined_ndarray, lidx, ridx = libjoin.inner_join_indexer(sv, ov)
        joined = self._from_join_target(joined_ndarray)
        return joined, lidx, ridx

    @final
    def _outer_indexer(
        self, other: Self
    ) -> tuple[ArrayLike, npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        # Caller is responsible for ensuring other.dtype == self.dtype
        sv = self._get_join_target()
        ov = other._get_join_target()
        joined_ndarray, lidx, ridx = libjoin.outer_join_indexer(sv, ov)
        joined = self._from_join_target(joined_ndarray)
        return joined, lidx, ridx

    _typ: str = "index"
    _data: ExtensionArray | np.ndarray
    _data_cls: type[ExtensionArray] | tuple[type[np.ndarray], type[ExtensionArray]] = (
        np.ndarray,
        ExtensionArray,
    )
    _id: object | None = None
    _name: Hashable = None
    # MultiIndex.levels previously allowed setting the index name. We
    # don't allow this anymore, and raise if it happens rather than
    # failing silently.
    _no_setting_name: bool = False
    _comparables: list[str] = ["name"]
    _attributes: list[str] = ["name"]

    @cache_readonly
    def _can_hold_strings(self) -> bool:
        return not is_numeric_dtype(self.dtype)

    _engine_types: dict[np.dtype | ExtensionDtype, type[libindex.IndexEngine]] = {
        np.dtype(np.int8): libindex.Int8Engine,
        np.dtype(np.int16): libindex.Int16Engine,
        np.dtype(np.int32): libindex.Int32Engine,
        np.dtype(np.int64): libindex.Int64Engine,
        np.dtype(np.uint8): libindex.UInt8Engine,
        np.dtype(np.uint16): libindex.UInt16