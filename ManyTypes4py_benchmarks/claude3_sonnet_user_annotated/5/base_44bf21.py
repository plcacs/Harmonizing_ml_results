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
