from __future__ import annotations

from datetime import (
    date,
    datetime,
    time,
    timedelta,
)
from decimal import Decimal
import re
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)
import warnings
import zoneinfo

import numpy as np
import numpy.typing as npt

from pandas._config.config import get_option

from pandas._libs import (
    lib,
    missing as libmissing,
)
from pandas._libs.interval import Interval
from pandas._libs.properties import cache_readonly
from pandas._libs.tslibs import (
    BaseOffset,
    NaT,
    NaTType,
    Period,
    Timedelta,
    Timestamp,
    timezones,
    to_offset,
    tz_compare,
)
from pandas._libs.tslibs.dtypes import (
    PeriodDtypeBase,
    abbrev_to_npy_unit,
)
from pandas._libs.tslibs.offsets import BDay
from pandas.compat import pa_version_under10p1
from pandas.errors import PerformanceWarning
from pandas.util._decorators import set_module
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.base import (
    ExtensionDtype,
    StorageExtensionDtype,
    register_extension_dtype,
)
from pandas.core.dtypes.generic import (
    ABCCategoricalIndex,
    ABCIndex,
    ABCRangeIndex,
)
from pandas.core.dtypes.inference import (
    is_bool,
    is_list_like,
)

if not pa_version_under10p1:
    import pyarrow as pa

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from datetime import tzinfo

    import pyarrow as pa  # noqa: TC004

    from pandas._typing import (
        Dtype,
        DtypeObj,
        IntervalClosedType,
        Ordered,
        Scalar,
        Self,
        npt,
        type_t,
    )

    from pandas import (
        Categorical,
        CategoricalIndex,
        DatetimeIndex,
        Index,
        IntervalIndex,
        PeriodIndex,
    )
    from pandas.core.arrays import (
        BaseMaskedArray,
        DatetimeArray,
        IntervalArray,
        NumpyExtensionArray,
        PeriodArray,
        SparseArray,
    )
    from pandas.core.arrays.arrow import ArrowExtensionArray

str_type = str


class PandasExtensionDtype(ExtensionDtype):
    """
    A np.dtype duck-typed class, suitable for holding a custom dtype.
    """
    type: Any
    kind: Any
    subdtype: None = None
    str: str_type
    num: int = 100
    shape: Tuple[int, ...] = ()
    itemsize: int = 8
    base: Optional[DtypeObj] = None
    isbuiltin: int = 0
    isnative: int = 0
    _cache_dtypes: Dict[str_type, PandasExtensionDtype] = {}

    def __repr__(self) -> str_type:
        return str(self)

    def __hash__(self) -> int:
        raise NotImplementedError("sub-classes should implement an __hash__ method")

    def __getstate__(self) -> Dict[str_type, Any]:
        return {k: getattr(self, k, None) for k in self._metadata}

    @classmethod
    def reset_cache(cls) -> None:
        cls._cache_dtypes = {}


class CategoricalDtypeType(type):
    """
    the type of CategoricalDtype, this metaclass determines subclass ability
    """
    pass


@register_extension_dtype
@set_module("pandas")
class CategoricalDtype(PandasExtensionDtype, ExtensionDtype):
    name: str = "category"
    type: Type[CategoricalDtypeType] = CategoricalDtypeType
    kind: str_type = "O"
    str: str = "|O08"
    base: np.dtype = np.dtype("O")
    _metadata: Tuple[str, ...] = ("categories", "ordered")
    _cache_dtypes: Dict[str_type, PandasExtensionDtype] = {}
    _supports_2d: bool = False
    _can_fast_transpose: bool = False

    def __init__(self, categories: Any = None, ordered: Ordered = False) -> None:
        self._finalize(categories, ordered, fastpath=False)

    @classmethod
    def _from_fastpath(
        cls, categories: Any = None, ordered: Optional[bool] = None
    ) -> CategoricalDtype:
        self = cls.__new__(cls)
        self._finalize(categories, ordered, fastpath=True)
        return self

    @classmethod
    def _from_categorical_dtype(
        cls, dtype: CategoricalDtype, categories: Any = None, ordered: Optional[Ordered] = None
    ) -> CategoricalDtype:
        if categories is ordered is None:
            return dtype
        if categories is None:
            categories = dtype.categories
        if ordered is None:
            ordered = dtype.ordered
        return cls(categories, ordered)

    @classmethod
    def _from_values_or_dtype(
        cls,
        values: Any = None,
        categories: Any = None,
        ordered: Optional[bool] = None,
        dtype: Optional[Dtype] = None,
    ) -> CategoricalDtype:
        if dtype is not None:
            if isinstance(dtype, str):
                if dtype == "category":
                    if ordered is None and cls.is_dtype(values):
                        ordered = values.dtype.ordered
                    dtype = CategoricalDtype(categories, ordered)
                else:
                    raise ValueError(f"Unknown dtype {dtype!r}")
            elif categories is not None or ordered is not None:
                raise ValueError(
                    "Cannot specify `categories` or `ordered` together with `dtype`."
                )
            elif not isinstance(dtype, CategoricalDtype):
                raise ValueError(f"Cannot not construct CategoricalDtype from {dtype}")
        elif cls.is_dtype(values):
            dtype = values.dtype._from_categorical_dtype(
                values.dtype, categories, ordered
            )
        else:
            dtype = CategoricalDtype(categories, ordered)
        return cast(CategoricalDtype, dtype)

    @classmethod
    def construct_from_string(cls, string: str_type) -> CategoricalDtype:
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )
        if string != cls.name:
            raise TypeError(f"Cannot construct a 'CategoricalDtype' from '{string}'")
        return cls(ordered=None)

    def _finalize(self, categories: Any, ordered: Ordered, fastpath: bool = False) -> None:
        if ordered is not None:
            self.validate_ordered(ordered)
        if categories is not None:
            categories = self.validate_categories(categories, fastpath=fastpath)
        self._categories = categories
        self._ordered = ordered

    def __setstate__(self, state: MutableMapping[str_type, Any]) -> None:
        self._categories = state.pop("categories", None)
        self._ordered = state.pop("ordered", False)

    def __hash__(self) -> int:
        if self.categories is None:
            if self.ordered:
                return -1
            else:
                return -2
        return int(self._hash_categories)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return other == self.name
        elif other is self:
            return True
        elif not (hasattr(other, "ordered") and hasattr(other, "categories")):
            return False
        elif self.categories is None or other.categories is None:
            return self.categories is other.categories
        elif self.ordered or other.ordered:
            return (self.ordered == other.ordered) and self.categories.equals(
                other.categories
            )
        else:
            left = self.categories
            right = other.categories
            if not left.dtype == right.dtype:
                return False
            if len(left) != len(right):
                return False
            if self.categories.equals(other.categories):
                return True
            if left.dtype != object:
                indexer = left.get_indexer(right)
                return bool((indexer != -1).all())
            return set(left) == set(right)

    def __repr__(self) -> str_type:
        if self.categories is None:
            data = "None"
            dtype = "None"
        else:
            data = self.categories._format_data(name=type(self).__name__)
            if isinstance(self.categories, ABCRangeIndex):
                data = str(self.categories._range)
            data = data.rstrip(", ")
            dtype = self.categories.dtype
        return (
            f"CategoricalDtype(categories={data}, ordered={self.ordered}, "
            f"categories_dtype={dtype})"
        )

    @cache_readonly
    def _hash_categories(self) -> int:
        from pandas.core.util.hashing import (
            combine_hash_arrays,
            hash_array,
            hash_tuples,
        )
        categories = self.categories
        ordered = self.ordered
        if len(categories) and isinstance(categories[0], tuple):
            cat_list = list(categories)
            cat_array = hash_tuples(cat_list)
        else:
            if categories.dtype == "O" and len({type(x) for x in categories}) != 1:
                hashed = hash((tuple(categories), ordered))
                return hashed
            if DatetimeTZDtype.is_dtype(categories.dtype):
                categories = categories.view("datetime64[ns]")
            cat_array = hash_array(np.asarray(categories), categorize=False)
        if ordered:
            cat_array = np.vstack(
                [cat_array, np.arange(len(cat_array), dtype=cat_array.dtype)]
            )
        else:
            cat_array = cat_array.reshape(1, len(cat_array))
        combined_hashed = combine_hash_arrays(iter(cat_array), num_items=len(cat_array))
        return np.bitwise_xor.reduce(combined_hashed)

    @classmethod
    def construct_array_type(cls) -> type_t[Categorical]:
        from pandas import Categorical
        return Categorical

    @staticmethod
    def validate_ordered(ordered: Ordered) -> None:
        if not is_bool(ordered):
            raise TypeError("'ordered' must either be 'True' or 'False'")

    @staticmethod
    def validate_categories(categories: Any, fastpath: bool = False) -> Index:
        from pandas.core.indexes.base import Index
        if not fastpath and not is_list_like(categories):
            raise TypeError(
                f"Parameter 'categories' must be list-like, was {categories!r}"
            )
        if not isinstance(categories, ABCIndex):
            categories = Index._with_infer(categories, tupleize_cols=False)
        if not fastpath:
            if categories.hasnans:
                raise ValueError("Categorical categories cannot be null")
            if not categories.is_unique:
                raise ValueError("Categorical categories must be unique")
        if isinstance(categories, ABCCategoricalIndex):
            categories = categories.categories
        return categories

    def update_dtype(self, dtype: Union[str_type, CategoricalDtype]) -> CategoricalDtype:
        if isinstance(dtype, str) and dtype == "category":
            return self
        elif not self.is_dtype(dtype):
            raise ValueError(
                f"a CategoricalDtype must be passed to perform an update, got {dtype!r}"
            )
        else:
            dtype = cast(CategoricalDtype, dtype)
        if (
            isinstance(dtype, CategoricalDtype)
            and dtype.categories is not None
            and dtype.ordered is not None
        ):
            return dtype
        new_categories = (
            dtype.categories if dtype.categories is not None else self.categories
        )
        new_ordered = dtype.ordered if dtype.ordered is not None else self.ordered
        return CategoricalDtype(new_categories, new_ordered)

    @property
    def categories(self) -> Index:
        return self._categories

    @property
    def ordered(self) -> Ordered:
        return self._ordered

    @property
    def _is_boolean(self) -> bool:
        from pandas.core.dtypes.common import is_bool_dtype
        return is_bool_dtype(self.categories)

    def _get_common_dtype(self, dtypes: List[DtypeObj]) -> Optional[DtypeObj]:
        if all(isinstance(x, CategoricalDtype) for x in dtypes):
            first = dtypes[0]
            if all(first == other for other in dtypes[1:]):
                return first
        non_init_cats = [
            isinstance(x, CategoricalDtype) and x.categories is None for x in dtypes
        ]
        if all(non_init_cats):
            return self
        elif any(non_init_cats):
            return None
        subtypes = (x.subtype if isinstance(x, SparseDtype) else x for x in dtypes)
        non_cat_dtypes = [
            x.categories.dtype if isinstance(x, CategoricalDtype) else x
            for x in subtypes
        ]
        from pandas.core.dtypes.cast import find_common_type
        return find_common_type(non_cat_dtypes)

    @cache_readonly
    def index_class(self) -> type_t[CategoricalIndex]:
        from pandas import CategoricalIndex
        return CategoricalIndex


@register_extension_dtype
@set_module("pandas")
class DatetimeTZDtype(PandasExtensionDtype):
    type: Type[Timestamp] = Timestamp
    kind: str_type = "M"
    num: int = 101
    _metadata: Tuple[str, ...] = ("unit", "tz")
    _match = re.compile(r"(datetime64|M8)\[(?P<unit>.+), (?P<tz>.+)\]")
    _cache_dtypes: Dict[str_type, PandasExtensionDtype] = {}
    _supports_2d: bool = True
    _can_fast_transpose: bool = True

    @property
    def na_value(self) -> NaTType:
        return NaT

    @cache_readonly
    def base(self) -> DtypeObj:
        return np.dtype(f"M8[{self.unit}]")

    @cache_readonly
    def str(self) -> str:
        return f"|M8[{self.unit}]"

    def __init__(self, unit: Union[str_type, DatetimeTZDtype] = "ns", tz: Any = None) -> None:
        if isinstance(unit, DatetimeTZDtype):
            unit, tz = unit.unit, unit.tz
        if unit != "ns":
            if isinstance(unit, str) and tz is None:
                result = type(self).construct_from_string(unit)
                unit = result.unit
                tz = result.tz
                msg = (
                    f"Passing a dtype alias like 'datetime64[ns, {tz}]' "
                    "to DatetimeTZDtype is no longer supported. Use "
                    "'DatetimeTZDtype.construct_from_string()' instead."
                )
                raise ValueError(msg)
            if unit not in ["s", "ms", "us", "ns"]:
                raise ValueError("DatetimeTZDtype only supports s, ms, us, ns units")
        if tz:
            tz = timezones.maybe_get_tz(tz)
            tz = timezones.tz_standardize(tz)
        elif tz is not None:
            raise zoneinfo.ZoneInfoNotFoundError(tz)
        if tz is None:
            raise TypeError("A 'tz' is required.")
        self._unit = unit
        self._tz = tz

    @cache_readonly
    def _creso(self) -> int:
        return abbrev_to_npy_unit(self.unit)

    @property
    def unit(self) -> str_type:
        return self._unit

    @property
    def tz(self) -> tzinfo:
        return self._tz

    @classmethod
    def construct_array_type(cls) -> type_t[DatetimeArray]:
        from pandas.core.arrays import DatetimeArray
        return DatetimeArray

    @classmethod
    def construct_from_string(cls, string: str_type) -> DatetimeTZDtype:
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )
        msg = f"Cannot construct a 'DatetimeTZDtype' from '{string}'"
        match = cls._match.match(string)
        if match:
            d = match.groupdict()
            try:
                return cls(unit=d["unit"], tz=d["tz"])
            except (KeyError, TypeError, ValueError) as err:
                raise TypeError(msg) from err
        raise TypeError(msg)

    def __str__(self) -> str_type:
        return f"datetime64[{self.unit}, {self.tz}]"

    @property
    def name(self) -> str_type:
        return str(self)

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            if other.startswith("M8["):
                other = f"datetime64[{other[3:]}"
            return other == self.name
        return (
            isinstance(other, DatetimeTZDtype)
            and self.unit == other.unit
            and tz_compare(self.tz, other.tz)
        )

    def __from_arrow__(self, array: Union[pa.Array, pa.ChunkedArray]) -> DatetimeArray:
        import pyarrow
        from pandas.core.arrays import DatetimeArray
        array = array.cast(pyarrow.timestamp(unit=self._unit), safe=True)
        if isinstance(array, pyarrow.Array):
            np_arr = array.to_numpy(zero_copy_only=False)
        else:
            np_arr = array.to_numpy()
        return DatetimeArray._simple_new(np_arr, dtype=self)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self._tz = state["tz"]
        self._unit = state["unit"]

    def _get_common_dtype(self, dtypes: List[DtypeObj])