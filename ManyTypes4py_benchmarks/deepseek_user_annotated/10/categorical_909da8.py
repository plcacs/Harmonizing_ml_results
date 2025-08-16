from __future__ import annotations

from csv import QUOTE_NONNUMERIC
from functools import partial
import operator
from shutil import get_terminal_size
from typing import (
    TYPE_CHECKING,
    Literal,
    cast,
    overload,
    Any,
    Callable,
    Hashable,
    Iterator,
    Sequence,
    TypeVar,
    Union,
    Optional,
    Dict,
    List,
    Tuple,
)

import numpy as np
import numpy.typing as npt

from pandas._config import get_option

from pandas._libs import (
    NaT,
    algos as libalgos,
    lib,
)
from pandas._libs.arrays import NDArrayBacked
from pandas.compat.numpy import function as nv
from pandas.util._validators import validate_bool_kwarg

from pandas.core.dtypes.cast import (
    coerce_indexer_dtype,
    find_common_type,
)
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
from pandas.core.dtypes.dtypes import (
    ArrowDtype,
    CategoricalDtype,
    CategoricalDtypeType,
    ExtensionDtype,
)
from pandas.core.dtypes.generic import (
    ABCIndex,
    ABCSeries,
)
from pandas.core.dtypes.missing import (
    is_valid_na_for_dtype,
    isna,
)

from pandas.core import (
    algorithms,
    arraylike,
    ops,
)
from pandas.core.accessor import (
    PandasDelegate,
    delegate_names,
)
from pandas.core.algorithms import (
    factorize,
    take_nd,
)
from pandas.core.arrays._mixins import (
    NDArrayBackedExtensionArray,
    ravel_compat,
)
from pandas.core.base import (
    ExtensionArray,
    NoNewAttributesMixin,
    PandasObject,
)
import pandas.core.common as com
from pandas.core.construction import (
    extract_array,
    sanitize_array,
)
from pandas.core.ops.common import unpack_zerodim_and_defer
from pandas.core.sorting import nargsort
from pandas.core.strings.object_array import ObjectStringArrayMixin

from pandas.io.formats import console

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Hashable,
        Iterator,
        Sequence,
    )

    from pandas._typing import (
        ArrayLike,
        AstypeArg,
        AxisInt,
        Dtype,
        DtypeObj,
        NpDtype,
        Ordered,
        Self,
        Shape,
        SortKind,
        npt,
    )

    from pandas import (
        DataFrame,
        Index,
        Series,
    )

T = TypeVar('T')

def _cat_compare_op(op: Callable[..., bool]) -> Callable[[Categorical, Any], np.ndarray]:
    opname = f"__{op.__name__}__"
    fill_value = op is operator.ne

    @unpack_zerodim_and_defer(opname)
    def func(self: Categorical, other: Any) -> np.ndarray:
        hashable = is_hashable(other)
        if is_list_like(other) and len(other) != len(self) and not hashable:
            raise ValueError("Lengths must match.")

        if not self.ordered:
            if opname in ["__lt__", "__gt__", "__le__", "__ge__"]:
                raise TypeError(
                    "Unordered Categoricals can only compare equality or not"
                )
        if isinstance(other, Categorical):
            msg = "Categoricals can only be compared if 'categories' are the same."
            if not self._categories_match_up_to_permutation(other):
                raise TypeError(msg)

            if not self.ordered and not self.categories.equals(other.categories):
                other_codes = recode_for_categories(
                    other.codes, other.categories, self.categories, copy=False
                )
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

                if opname not in {"__eq__", "__ge__", "__gt__"}:
                    mask = self._codes == -1
                    ret[mask] = fill_value
                return ret
            else:
                return ops.invalid_comparison(self, other, op)
        else:
            if opname not in ["__eq__", "__ne__"]:
                raise TypeError(
                    f"Cannot compare a Categorical for op {opname} with "
                    f"type {type(other)}.\nIf you want to compare values, "
                    "use 'np.asarray(cat) <op> other'."
                )

            if isinstance(other, ExtensionArray) and needs_i8_conversion(other.dtype):
                return op(other, self)
            return getattr(np.array(self), opname)(np.array(other))

    func.__name__ = opname
    return func

def contains(cat: Union[Categorical, 'CategoricalIndex'], key: Hashable, container: Any) -> bool:
    hash(key)

    try:
        loc = cat.categories.get_loc(key)
    except (KeyError, TypeError):
        return False

    if is_scalar(loc):
        return loc in container
    else:
        return any(loc_ in container for loc_ in loc)

class Categorical(NDArrayBackedExtensionArray, PandasObject, ObjectStringArrayMixin):
    _dtype: CategoricalDtype
    _typ: str = "categorical"
    _hidden_attrs = PandasObject._hidden_attrs | frozenset(["tolist"])
    __array_priority__ = 1000

    @classmethod
    def _simple_new(
        cls, codes: np.ndarray, dtype: CategoricalDtype
    ) -> Self:
        codes = coerce_indexer_dtype(codes, dtype.categories)
        dtype = CategoricalDtype(ordered=False).update_dtype(dtype)
        return super()._simple_new(codes, dtype)

    def __init__(
        self,
        values: Any,
        categories: Optional[Any] = None,
        ordered: Optional[bool] = None,
        dtype: Optional[Dtype] = None,
        copy: bool = True,
    ) -> None:
        dtype = CategoricalDtype._from_values_or_dtype(
            values, categories, ordered, dtype
        )
        if not is_list_like(values):
            raise TypeError("Categorical input must be list-like")

        null_mask = np.array(False)
        vdtype = getattr(values, "dtype", None)
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
                    raise NotImplementedError(
                        "> 1 ndim Categorical are not supported at this time"
                    )
                values = sanitize_array(values, None)
            else:
                arr = sanitize_array(values, None)
                null_mask = isna(arr)
                if null_mask.any():
                    arr_list = [values[idx] for idx in np.where(~null_mask)[0]]
                    sanitize_dtype = None if arr_list or arr.dtype == "object" else arr.dtype
                    arr = sanitize_array(arr_list, None, dtype=sanitize_dtype)
                values = arr

        if dtype.categories is None:
            if isinstance(values.dtype, ArrowDtype) and issubclass(
                values.dtype.type, CategoricalDtypeType
            ):
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
                        raise TypeError(
                            "'values' is not ordered, please "
                            "explicitly specify the categories order "
                            "by passing in a categories argument."
                        ) from err
                dtype = CategoricalDtype(categories, dtype.ordered)

        elif isinstance(values.dtype, CategoricalDtype):
            old_codes = extract_array(values)._codes
            codes = recode_for_categories(
                old_codes, values.dtype.categories, dtype.categories, copy=copy
            )
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
    def _internal_fill_value(self) -> int:
        return self._ndarray.dtype.type(-1)

    @classmethod
    def _from_sequence(
        cls, scalars: Sequence[Any], *, dtype: Optional[Dtype] = None, copy: bool = False
    ) -> Self:
        return cls(scalars, dtype=dtype, copy=copy)

    @classmethod
    def _from_scalars(cls, scalars: Sequence[Any], *, dtype: DtypeObj) -> Self:
        if dtype is None:
            raise NotImplementedError
        res = cls._from_sequence(scalars, dtype=dtype)
        mask = isna(scalars)
        if not (mask == res.isna()).all():
            raise ValueError
        return res

    @overload
    def astype(self, dtype: npt.DTypeLike, copy: bool = ...) -> np.ndarray: ...
    @overload
    def astype(self, dtype: ExtensionDtype, copy: bool = ...) -> ExtensionArray: ...
    @overload
    def astype(self, dtype: AstypeArg, copy: bool = ...) -> ArrayLike: ...
    def astype(self, dtype: AstypeArg, copy: bool = True) -> ArrayLike:
        dtype = pandas_dtype(dtype)
        result: Union[Categorical, np.ndarray]
        if self.dtype is dtype:
            result = self.copy() if copy else self
        elif isinstance(dtype, CategoricalDtype):
            dtype = self.dtype.update_dtype(dtype)
            self = self.copy() if copy else self
            result = self._set_dtype(dtype)
        elif isinstance(dtype, ExtensionDtype):
            return super().astype(dtype, copy=copy)
        elif dtype.kind in "iu" and self.isna().any():
            raise ValueError("Cannot convert float NaN to integer")
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
                    fill_value = lib.item_from_zerodim(
                        np.array(self.categories._na_value).astype(dtype)
                    )
            except (TypeError, ValueError) as err:
                msg = f"Cannot cast {self.categories.dtype} dtype to {dtype}"
                raise ValueError(msg) from err
            result = take_nd(
                new_cats, ensure_platform_int(self._codes), fill_value=fill_value
            )
        return result

    @classmethod
    def _from_inferred_categories(
        cls,
        inferred_categories: Index,
        inferred_codes: Index,
        dtype: Union[CategoricalDtype, str],
        true_values: Optional[List[str]] = None,
    ) -> Self:
        from pandas import (
            Index,
            to_datetime,
            to_numeric,
            to_timedelta,
        )
        cats = Index(inferred_categories)
        known_categories = (
            isinstance(dtype, CategoricalDtype) and dtype.categories is not None
        )
        if known_categories:
            if is_any_real_numeric_dtype(dtype.categories.dtype):
                cats = to_numeric(inferred_categories, errors="coerce")
            elif lib.is_np_dtype(dtype.categories.dtype, "M"):
                cats = to_datetime(inferred_categories, errors="coerce")
            elif lib.is_np_dtype(dtype.categories.dtype, "m"):
                cats = to_timedelta(inferred_categories, errors="coerce")
            elif is_bool_dtype(dtype.categories.dtype):
                if true_values is None:
                    true_values = ["True", "TRUE", "true"]
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
        codes: ArrayLike,
        categories: Optional[Any] = None,
        ordered: Optional[bool] = None,
        dtype: Optional[Dtype] = None,
        validate: bool = True,
    ) -> Self:
        dtype = CategoricalDtype._from_values_or_dtype(
            categories=categories, ordered=ordered, dtype=dtype
        )
        if dtype.categories is None:
            raise ValueError(
                "The categories must be provided in 'categories' or "
                "'dtype'. Both were None."
            )
        if validate:
            codes = cls._validate_codes_for_dtype(codes, dtype=dtype)
        return cls._simple_new(codes, dtype=dtype)

    @property
    def categories(self) -> Index:
        return self.dtype.categories

    @property
    def ordered(self) -> Ordered:
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
            if (
                self.dtype.categories is not None
                and len(new_dtype.categories) != len(self.dtype.categories)
            ):
                raise ValueError(
                    "new categories need to have the same number of "
                    "items as the old categories!"
                )
        super().__init__(self._ndarray, new_dtype)

    def _set_dtype(self, dtype: CategoricalDtype) -> Self:
        codes = recode_for_categories(self.codes, self.categories, dtype.categories)
        return type(self)._simple_new(codes, dtype=dtype)

    def set_ordered(self, value: bool) -> Self:
        new_dtype = CategoricalDtype(self.categories, ordered=value)
        cat = self.copy()
        NDArrayBacked.__init__(cat, cat._ndarray, new_dtype)
        return cat

    def as_ordered(self) -> Self:
        return self.set_ordered(True)

    def as_unordered(self) -> Self:
        return self.set_ordered(False)

    def set_categories(
        self,
        new_categories: Any,
        ordered: Optional[bool] = None,
        rename: bool = False,
    ) -> Self:
        if ordered is None:
            ordered = self.dtype.ordered
        new_dtype = CategoricalDtype(new_categories, ordered=ordered)
        cat = self.copy()
        if rename:
            if cat.dtype.categories is not None and len(new_dtype.categories) < len(
                cat.dtype.categories
            ):
                cat._codes[cat._codes >= len(new_dtype.categories)] = -1
            codes = cat._codes
        else:
            codes = recode_for_categories(
                cat.codes, cat.categories, new_dtype.categories
            )
        NDArrayBacked.__init__(cat, codes, new_dtype)
        return cat

    def rename_categories(self, new_categories: Any) -> Self:
        if is_dict_like(new_categories):
            new_categories = [
                new_categories.get(item, item) for item in self.categories
            ]
        elif callable(new_categories):
            new_categories = [new_categories(item) for item in self.categories]
        cat = self.copy()
        cat._set_categories(new_categories)
        return cat

    def reorder_categories(self, new_categories: Any, ordered: Optional[bool] = None) -> Self:
        if (
            len(self.categories) != len(new_categories)
            or not self.categories.difference(new_categories).empty
        ):
            raise ValueError(
               