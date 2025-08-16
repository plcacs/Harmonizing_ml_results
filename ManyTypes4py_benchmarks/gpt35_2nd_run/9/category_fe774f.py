from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal, cast, Union
import numpy as np
from pandas._libs import index as libindex
from pandas.util._decorators import cache_readonly, doc, set_module
from pandas.core.dtypes.common import is_scalar
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas.core.dtypes.missing import is_valid_na_for_dtype, isna
from pandas.core.arrays.categorical import Categorical, contains
from pandas.core.construction import extract_array
from pandas.core.indexes.base import Index, maybe_extract_name
from pandas.core.indexes.extension import NDArrayBackedExtensionIndex, inherit_names
if TYPE_CHECKING:
    from collections.abc import Hashable
    from pandas._typing import Dtype, DtypeObj, Self, npt

@inherit_names(['argsort', 'tolist', 'codes', 'categories', 'ordered', '_reverse_indexer', 'searchsorted', 'min', 'max'], Categorical)
@inherit_names(['rename_categories', 'reorder_categories', 'add_categories', 'remove_categories', 'remove_unused_categories', 'set_categories', 'as_ordered', 'as_unordered'], Categorical, wrap=True)
@set_module('pandas')
class CategoricalIndex(NDArrayBackedExtensionIndex):
    _typ: Literal['categoricalindex'] = 'categoricalindex'
    _data_cls: type[Categorical] = Categorical

    @property
    def _can_hold_strings(self) -> bool:
        return self.categories._can_hold_strings

    @cache_readonly
    def _should_fallback_to_positional(self) -> bool:
        return self.categories._should_fallback_to_positional

    @property
    def _engine_type(self) -> Union[libindex.Int8Engine, libindex.Int16Engine, libindex.Int32Engine, libindex.Int64Engine]:
        return {np.int8: libindex.Int8Engine, np.int16: libindex.Int16Engine, np.int32: libindex.Int32Engine, np.int64: libindex.Int64Engine}[self.codes.dtype.type]

    def __new__(cls, data=None, categories=None, ordered=None, dtype=None, copy=False, name=None) -> CategoricalIndex:
        name = maybe_extract_name(name, data, cls)
        if is_scalar(data):
            cls._raise_scalar_data_error(data)
        data = Categorical(data, categories=categories, ordered=ordered, dtype=dtype, copy=copy)
        return cls._simple_new(data, name=name)

    def _is_dtype_compat(self, other: Index) -> Categorical:
        ...

    def equals(self, other: object) -> bool:
        ...

    @property
    def _formatter_func(self) -> Any:
        return self.categories._formatter_func

    def _format_attrs(self) -> list[tuple[str, Any]]:
        ...

    @property
    def inferred_type(self) -> str:
        return 'categorical'

    @doc(Index.__contains__)
    def __contains__(self, key) -> bool:
        ...

    def reindex(self, target, method=None, level=None, limit=None, tolerance=None) -> tuple[Index, np.ndarray[np.intp]]:
        ...

    def _maybe_cast_indexer(self, key) -> Any:
        ...

    def _maybe_cast_listlike_indexer(self, values) -> CategoricalIndex:
        ...

    def _is_comparable_dtype(self, dtype) -> bool:
        ...

    def map(self, mapper, na_action=None) -> Union[CategoricalIndex, Index]:
        ...

    def _concat(self, to_concat, name) -> Union[CategoricalIndex, Index]:
        ...
