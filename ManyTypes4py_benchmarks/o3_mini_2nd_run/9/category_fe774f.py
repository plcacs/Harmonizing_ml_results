from __future__ import annotations
from typing import Any, Callable, List, Optional, Tuple, Union, TYPE_CHECKING
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
    """
    Index based on an underlying :class:`Categorical`.

    CategoricalIndex, like Categorical, can only take on a limited,
    and usually fixed, number of possible values (`categories`). Also,
    like Categorical, it might have an order, but numerical operations
    (additions, divisions, ...) are not possible.
    """
    _typ: str = 'categoricalindex'
    _data_cls = Categorical

    @property
    def _can_hold_strings(self) -> bool:
        return self.categories._can_hold_strings

    @cache_readonly
    def _should_fallback_to_positional(self) -> bool:
        return self.categories._should_fallback_to_positional

    @property
    def _engine_type(self) -> type:
        return {
            np.int8: libindex.Int8Engine,
            np.int16: libindex.Int16Engine,
            np.int32: libindex.Int32Engine,
            np.int64: libindex.Int64Engine
        }[self.codes.dtype.type]

    def __new__(
        cls: type[Self],
        data: Any = None,
        categories: Any = None,
        ordered: Optional[bool] = None,
        dtype: Optional[Union[CategoricalDtype, str]] = None,
        copy: bool = False,
        name: Any = None
    ) -> Self:
        name = maybe_extract_name(name, data, cls)
        if is_scalar(data):
            cls._raise_scalar_data_error(data)
        data = Categorical(data, categories=categories, ordered=ordered, dtype=dtype, copy=copy)
        return cls._simple_new(data, name=name)

    def _is_dtype_compat(self, other: Index) -> Categorical:
        """
        *this is an internal non-public method*

        provide a comparison between the dtype of self and other (coercing if
        needed)
        """
        if isinstance(other.dtype, CategoricalDtype):
            cat = extract_array(other)
            cat = cast(Categorical, cat)
            if not cat._categories_match_up_to_permutation(self._values):
                raise TypeError('categories must match existing categories when appending')
        elif other._is_multi:
            raise TypeError('MultiIndex is not dtype-compatible with CategoricalIndex')
        else:
            values = other
            cat = Categorical(other, dtype=self.dtype)
            other = CategoricalIndex(cat)
            if not other.isin(values).all():
                raise TypeError('cannot append a non-category item to a CategoricalIndex')
            cat = other._values
            if not ((cat == values) | isna(cat) & isna(values)).all():
                raise TypeError('categories must match existing categories when appending')
        return cat

    def equals(self, other: Any) -> bool:
        """
        Determine if two CategoricalIndex objects contain the same elements.
        """
        if self.is_(other):
            return True
        if not isinstance(other, Index):
            return False
        try:
            other_cat = self._is_dtype_compat(other)
        except (TypeError, ValueError):
            return False
        return self._data.equals(other_cat)

    @property
    def _formatter_func(self) -> Callable[[Any], Any]:
        return self.categories._formatter_func

    def _format_attrs(self) -> List[Tuple[str, Any]]:
        """
        Return a list of tuples of the (attr,formatted_value)
        """
        attrs = [
            ('categories', f"[{', '.join(self._data._repr_categories())}]"),
            ('ordered', self.ordered)
        ]
        extra = super()._format_attrs()
        return attrs + extra

    @property
    def inferred_type(self) -> str:
        return 'categorical'

    @doc(Index.__contains__)
    def __contains__(self, key: Any) -> bool:
        if is_valid_na_for_dtype(key, self.categories.dtype):
            return self.hasnans
        if self.categories._typ == 'rangeindex':
            container = self.categories
        else:
            container = self._engine
        return contains(self, key, container=container)

    def reindex(
        self,
        target: Any,
        method: Optional[Any] = None,
        level: Optional[Any] = None,
        limit: Optional[Any] = None,
        tolerance: Optional[Any] = None
    ) -> Tuple[Index, Optional[np.ndarray]]:
        """
        Create index with target's values (move/add/delete values as necessary)
        """
        if method is not None:
            raise NotImplementedError('argument method is not implemented for CategoricalIndex.reindex')
        if level is not None:
            raise NotImplementedError('argument level is not implemented for CategoricalIndex.reindex')
        if limit is not None:
            raise NotImplementedError('argument limit is not implemented for CategoricalIndex.reindex')
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

    def _is_comparable_dtype(self, dtype: Any) -> bool:
        return self.categories._is_comparable_dtype(dtype)

    def map(
        self,
        mapper: Union[Callable[[Any], Any], dict[Any, Any], Any],
        na_action: Optional[Literal[None, "ignore"]] = None
    ) -> Index:
        """
        Map values using input an input mapping or function.
        """
        mapped = self._values.map(mapper, na_action=na_action)
        return Index(mapped, name=self.name)

    def _concat(self, to_concat: List[Index], name: Any) -> CategoricalIndex:
        try:
            cat = Categorical._concat_same_type([self._is_dtype_compat(c) for c in to_concat])
        except TypeError:
            res = concat_compat([x._values for x in to_concat])
            return Index(res, name=name)
        else:
            return type(self)._simple_new(cat, name=name)