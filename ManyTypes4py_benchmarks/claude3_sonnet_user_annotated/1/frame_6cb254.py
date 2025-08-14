from __future__ import annotations

import collections
from collections import abc
from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
import functools
from io import StringIO
import itertools
import operator
import sys
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
    overload,
)
import warnings

import numpy as np
from numpy import ma

from pandas._config import get_option

from pandas._libs import (
    algos as libalgos,
    lib,
    properties,
)
from pandas._libs.hashtable import duplicated
from pandas._libs.lib import is_range_indexer
from pandas.compat import PYPY
from pandas.compat._constants import REF_COUNT
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.errors import (
    ChainedAssignmentError,
    InvalidIndexError,
)
from pandas.errors.cow import (
    _chained_assignment_method_msg,
    _chained_assignment_msg,
)
from pandas.util._decorators import (
    Appender,
    Substitution,
    deprecate_nonkeyword_arguments,
    doc,
    set_module,
)
from pandas.util._exceptions import (
    find_stack_level,
    rewrite_warning,
)
from pandas.util._validators import (
    validate_ascending,
    validate_bool_kwarg,
    validate_percentile,
)

from pandas.core.dtypes.cast import (
    LossySetitemError,
    can_hold_element,
    construct_1d_arraylike_from_scalar,
    construct_2d_arraylike_from_scalar,
    find_common_type,
    infer_dtype_from_scalar,
    invalidate_string_dtypes,
    maybe_downcast_to_dtype,
)
from pandas.core.dtypes.common import (
    infer_dtype_from_object,
    is_1d_only_ea_dtype,
    is_array_like,
    is_bool_dtype,
    is_dataclass,
    is_dict_like,
    is_float,
    is_float_dtype,
    is_hashable,
    is_integer,
    is_integer_dtype,
    is_iterator,
    is_list_like,
    is_scalar,
    is_sequence,
    needs_i8_conversion,
    pandas_dtype,
)
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
    ArrowDtype,
    BaseMaskedDtype,
    ExtensionDtype,
)
from pandas.core.dtypes.missing import (
    isna,
    notna,
)

from pandas.core import (
    algorithms,
    common as com,
    nanops,
    ops,
    roperator,
)
from pandas.core.accessor import Accessor
from pandas.core.apply import reconstruct_and_relabel_result
from pandas.core.array_algos.take import take_2d_multi
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import (
    BaseMaskedArray,
    DatetimeArray,
    ExtensionArray,
    PeriodArray,
    TimedeltaArray,
)
from pandas.core.arrays.sparse import SparseFrameAccessor
from pandas.core.construction import (
    ensure_wrapped_if_datetimelike,
    sanitize_array,
    sanitize_masked_array,
)
from pandas.core.generic import (
    NDFrame,
    make_doc,
)
from pandas.core.indexers import check_key_length
from pandas.core.indexes.api import (
    DatetimeIndex,
    Index,
    PeriodIndex,
    default_index,
    ensure_index,
    ensure_index_from_sequences,
)
from pandas.core.indexes.multi import (
    MultiIndex,
    maybe_droplevels,
)
from pandas.core.indexing import (
    check_bool_indexer,
    check_dict_or_set_indexers,
)
from pandas.core.internals import BlockManager
from pandas.core.internals.construction import (
    arrays_to_mgr,
    dataclasses_to_dicts,
    dict_to_mgr,
    ndarray_to_mgr,
    nested_data_to_arrays,
    rec_array_to_mgr,
    reorder_arrays,
    to_arrays,
    treat_as_nested,
)
from pandas.core.methods import selectn
from pandas.core.reshape.melt import melt
from pandas.core.series import Series
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import (
    get_group_index,
    lexsort_indexer,
    nargsort,
)

from pandas.io.common import get_handle
from pandas.io.formats import (
    console,
    format as fmt,
)
from pandas.io.formats.info import (
    INFO_DOCSTRING,
    DataFrameInfo,
    frame_sub_kwargs,
)
import pandas.plotting

if TYPE_CHECKING:
    import datetime

    from pandas._libs.internals import BlockValuesRefs
    from pandas._typing import (
        AggFuncType,
        AnyAll,
        AnyArrayLike,
        ArrayLike,
        Axes,
        Axis,
        AxisInt,
        ColspaceArgType,
        CompressionOptions,
        CorrelationMethod,
        DropKeep,
        Dtype,
        DtypeObj,
        FilePath,
        FloatFormatType,
        FormattersType,
        Frequency,
        FromDictOrient,
        HashableT,
        HashableT2,
        IgnoreRaise,
        IndexKeyFunc,
        IndexLabel,
        JoinValidate,
        Level,
        ListLike,
        MergeHow,
        MergeValidate,
        MutableMappingT,
        NaPosition,
        NsmallestNlargestKeep,
        PythonFuncType,
        QuantileInterpolation,
        ReadBuffer,
        ReindexMethod,
        Renamer,
        Scalar,
        Self,
        SequenceNotStr,
        SortKind,
        StorageOptions,
        Suffixes,
        T,
        ToStataByteorder,
        ToTimestampHow,
        UpdateJoin,
        ValueKeyFunc,
        WriteBuffer,
        XMLParsers,
        npt,
    )

    from pandas.core.groupby.generic import DataFrameGroupBy
    from pandas.core.interchange.dataframe_protocol import DataFrame as DataFrameXchg
    from pandas.core.internals.managers import SingleBlockManager

    from pandas.io.formats.style import Styler
