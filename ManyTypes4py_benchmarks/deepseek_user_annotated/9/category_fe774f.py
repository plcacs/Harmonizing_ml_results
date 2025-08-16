from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
    Optional,
    Union,
    List,
    Tuple,
    Dict,
)

import numpy as np

from pandas._libs import index as libindex
from pandas.util._decorators import (
    cache_readonly,
    doc,
    set_module,
)

from pandas.core.dtypes.common import is_scalar
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas.core.dtypes.missing import (
    is_valid_na_for_dtype,
    isna,
)

from pandas.core.arrays.categorical import (
    Categorical,
    contains,
)
from pandas.core.construction import extract_array
from pandas.core.indexes.base import (
    Index,
    maybe_extract_name,
)
from pandas.core.indexes.extension import (
    NDArrayBackedExtensionIndex,
    inherit_names,
)

if TYPE_CHECKING:
    from collections.abc import Hashable

    from pandas._typing import (
        Dtype,
        DtypeObj,
        Self,
        npt,
    )


@inherit_names(
    [
        "argsort",
        "tolist",
        "codes",
        "categories",
        "ordered",
        "_reverse_indexer",
        "searchsorted",
        "min",
        "max",
    ],
    Categorical,
)
@inherit_names(
    [
        "rename_categories",
        "reorder_categories",
        "add_categories",
        "remove_categories",
        "remove_unused_categories",
        "set_categories",
        "as_ordered",
        "as_unordered",
    ],
    Categorical,
    wrap=True,
)
@set_module("pandas")
class CategoricalIndex(NDArrayBackedExtensionIndex):
    _typ: str = "categoricalindex"
    _data_cls: type[Categorical] = Categorical

    @property
    def _can_hold_strings(self) -> bool:
        return self.categories._can_hold_strings

    @cache_readonly
    def _should_fallback_to_positional(self) -> bool:
        return self.categories._should_fallback_to_positional

    codes: np.ndarray
    categories: Index
    ordered: Optional[bool]
    _data: Categorical
    _values: Categorical

    @property
    def _engine_type(self) -> type[libindex.IndexEngine]:
        return {
            np.int8: libindex.Int8Engine,
            np.int16: libindex.Int16Engine,
            np.int32: libindex.Int32Engine,
            np.int64: libindex.Int64Engine,
        }[self.codes.dtype.type]

    def __new__(
        cls,
        data: Optional[Any] = None,
        categories: Optional[Any] = None,
        ordered: Optional[bool] = None,
        dtype: Optional[Dtype] = None,
        copy: bool = False,
        name: Optional[Hashable] = None,
    ) -> Self:
        name = maybe_extract_name(name, data, cls)

        if is_scalar(data):
            cls._raise_scalar_data_error(data)

        data = Categorical(
            data, categories=categories, ordered=ordered, dtype=dtype, copy=copy
        )

        return cls._simple_new(data, name=name)

    def _is_dtype_compat(self, other: Index) -> Categorical:
        if isinstance(other.dtype, CategoricalDtype):
            cat = extract_array(other)
            cat = cast(Categorical, cat)
            if not cat._categories_match_up_to_permutation(self._values):
                raise TypeError(
                    "categories must match existing categories when appending"
                )

        elif other._is_multi:
            raise TypeError("MultiIndex is not dtype-compatible with CategoricalIndex")
        else:
            values = other

            cat = Categorical(other, dtype=self.dtype)
            other = CategoricalIndex(cat)
            if not other.isin(values).all():
                raise TypeError(
                    "cannot append a non-category item to a CategoricalIndex"
                )
            cat = other._values

            if not ((cat == values) | (isna(cat) & isna(values))).all():
                raise TypeError(
                    "categories must match existing categories when appending"
                )

        return cat

    def equals(self, other: object) -> bool:
        if self.is_(other):
            return True

        if not isinstance(other, Index):
            return False

        try:
            other = self._is_dtype_compat(other)
        except (TypeError, ValueError):
            return False

        return self._data.equals(other)

    @property
    def _formatter_func(self):
        return self.categories._formatter_func

    def _format_attrs(self) -> List[Tuple[str, Union[str, int, bool, None]]:
        attrs: List[Tuple[str, Union[str, int, bool, None]] = [
            (
                "categories",
                f"[{', '.join(self._data._repr_categories())}]",
            ),
            ("ordered", self.ordered),
        ]
        extra = super()._format_attrs()
        return attrs + extra

    @property
    def inferred_type(self) -> str:
        return "categorical"

    def __contains__(self, key: Any) -> bool:
        if is_valid_na_for_dtype(key, self.categories.dtype):
            return self.hasnans
        if self.categories._typ == "rangeindex":
            container: Union[Index, libindex.IndexEngine, libindex.ExtensionEngine] = (
                self.categories
            )
        else:
            container = self._engine
        return contains(self, key, container=container)

    def reindex(
        self,
        target: Any,
        method: Optional[str] = None,
        level: Optional[Any] = None,
        limit: Optional[int] = None,
        tolerance: Optional[Any] = None,
    ) -> Tuple[Index, Optional[npt.NDArray[np.intp]]]:
        if method is not None:
            raise NotImplementedError(
                "argument method is not implemented for CategoricalIndex.reindex"
            )
        if level is not None:
            raise NotImplementedError(
                "argument level is not implemented for CategoricalIndex.reindex"
            )
        if limit is not None:
            raise NotImplementedError(
                "argument limit is not implemented for CategoricalIndex.reindex"
            )
        return super().reindex(target)

    def _maybe_cast_indexer(self, key: Any) -> int:
        try:
            return self._data._unbox_scalar(key)
        except KeyError:
            if is_valid_na_for_dtype(key, self.categories.dtype):
                return -1
            raise

    def _maybe_cast_listlike_indexer(self, values: Any) -> CategoricalIndex:
        if isinstance(values, CategoricalIndex):
            values = values._data
        if isinstance(values, Categorical):
            cat = self._data._encode_with_my_categories(values)
            codes = cat._codes
        else:
            codes = self.categories.get_indexer(values)
            codes = codes.astype(self.codes.dtype, copy=False)
            cat = self._data._from_backing_data(codes)
        return type(self)._simple_new(cat)

    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        return self.categories._is_comparable_dtype(dtype)

    def map(
        self,
        mapper: Union[Dict, Any],
        na_action: Optional[Literal["ignore"]] = None,
    ) -> Union[CategoricalIndex, Index]:
        mapped = self._values.map(mapper, na_action=na_action)
        return Index(mapped, name=self.name)

    def _concat(self, to_concat: List[Index], name: Hashable) -> Index:
        try:
            cat = Categorical._concat_same_type(
                [self._is_dtype_compat(c) for c in to_concat]
            )
        except TypeError:
            res = concat_compat([x._values for x in to_concat])
            return Index(res, name=name)
        else:
            return type(self)._simple_new(cat, name=name)
