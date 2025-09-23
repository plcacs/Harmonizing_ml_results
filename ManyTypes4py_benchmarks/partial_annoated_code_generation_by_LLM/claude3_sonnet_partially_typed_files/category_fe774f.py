from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal, cast, Optional, Union, List, Tuple, Dict, Callable
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

    Parameters
    ----------
    data : array-like (1-dimensional)
        The values of the categorical. If `categories` are given, values not in
        `categories` will be replaced with NaN.
    categories : index-like, optional
        The categories for the categorical. Items need to be unique.
        If the categories are not given here (and also not in `dtype`), they
        will be inferred from the `data`.
    ordered : bool, optional
        Whether or not this categorical is treated as an ordered
        categorical. If not given here or in `dtype`, the resulting
        categorical will be unordered.
    dtype : CategoricalDtype or "category", optional
        If :class:`CategoricalDtype`, cannot be used together with
        `categories` or `ordered`.
    copy : bool, default False
        Make a copy of input ndarray.
    name : object, optional
        Name to be stored in the index.

    Attributes
    ----------
    codes
    categories
    ordered

    Methods
    -------
    rename_categories
    reorder_categories
    add_categories
    remove_categories
    remove_unused_categories
    set_categories
    as_ordered
    as_unordered
    map

    Raises
    ------
    ValueError
        If the categories do not validate.
    TypeError
        If an explicit ``ordered=True`` is given but no `categories` and the
        `values` are not sortable.

    See Also
    --------
    Index : The base pandas Index type.
    Categorical : A categorical array.
    CategoricalDtype : Type for categorical data.

    Notes
    -----
    See the `user guide
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#categoricalindex>`__
    for more.

    Examples
    --------
    >>> pd.CategoricalIndex(["a", "b", "c", "a", "b", "c"])
    CategoricalIndex(['a', 'b', 'c', 'a', 'b', 'c'],
                     categories=['a', 'b', 'c'], ordered=False, dtype='category')

    ``CategoricalIndex`` can also be instantiated from a ``Categorical``:

    >>> c = pd.Categorical(["a", "b", "c", "a", "b", "c"])
    >>> pd.CategoricalIndex(c)
    CategoricalIndex(['a', 'b', 'c', 'a', 'b', 'c'],
                     categories=['a', 'b', 'c'], ordered=False, dtype='category')

    Ordered ``CategoricalIndex`` can have a min and max value.

    >>> ci = pd.CategoricalIndex(
    ...     ["a", "b", "c", "a", "b", "c"], ordered=True, categories=["c", "b", "a"]
    ... )
    >>> ci
    CategoricalIndex(['a', 'b', 'c', 'a', 'b', 'c'],
                     categories=['c', 'b', 'a'], ordered=True, dtype='category')
    >>> ci.min()
    'c'
    """
    _typ: str = 'categoricalindex'
    _data_cls = Categorical

    @property
    def _can_hold_strings(self) -> bool:
        return self.categories._can_hold_strings

    @cache_readonly
    def _should_fallback_to_positional(self) -> bool:
        return self.categories._should_fallback_to_positional
    codes: np.ndarray
    categories: Index
    ordered: bool | None
    _data: Categorical
    _values: Categorical

    @property
    def _engine_type(self) -> type:
        return {np.int8: libindex.Int8Engine, np.int16: libindex.Int16Engine, np.int32: libindex.Int32Engine, np.int64: libindex.Int64Engine}[self.codes.dtype.type]

    def __new__(cls, data: Any = None, categories: Optional[Any] = None, ordered: Optional[bool] = None, 
                dtype: Optional[Union[CategoricalDtype, str]] = None, copy: bool = False, 
                name: Optional[Hashable] = None) -> Self:
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

        Parameters
        ----------
        other : Index

        Returns
        -------
        Categorical

        Raises
        ------
        TypeError if the dtypes are not compatible
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

    def equals(self, other: object) -> bool:
        """
        Determine if two CategoricalIndex objects contain the same elements.

        The order and orderedness of elements matters. The categories matter,
        but the order of the categories matters only when ``ordered=True``.

        Parameters
        ----------
        other : object
            The CategoricalIndex object to compare with.

        Returns
        -------
        bool
            ``True`` if two :class:`pandas.CategoricalIndex` objects have equal
            elements, ``False`` otherwise.

        See Also
        --------
        Categorical.equals : Returns True if categorical arrays are equal.

        Examples
        --------
        >>> ci = pd.CategoricalIndex(["a", "b", "c", "a", "b", "c"])
        >>> ci2 = pd.CategoricalIndex(pd.Categorical(["a", "b", "c", "a", "b", "c"]))
        >>> ci.equals(ci2)
        True

        The order of elements matters.

        >>> ci3 = pd.CategoricalIndex(["c", "b", "a", "a", "b", "c"])
        >>> ci.equals(ci3)
        False

        The orderedness also matters.

        >>> ci4 = ci.as_ordered()
        >>> ci.equals(ci4)
        False

        The categories matter, but the order of the categories matters only when
        ``ordered=True``.

        >>> ci5 = ci.set_categories(["a", "b", "c", "d"])
        >>> ci.equals(ci5)
        False

        >>> ci6 = ci.set_categories(["b", "c", "a"])
        >>> ci.equals(ci6)
        True
        >>> ci_ordered = pd.CategoricalIndex(
        ...     ["a", "b", "c", "a", "b", "c"], ordered=True
        ... )
        >>> ci2_ordered = ci_ordered.set_categories(["b", "c", "a"])
        >>> ci_ordered.equals(ci2_ordered)
        False
        """
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
    def _formatter_func(self) -> Callable:
        return self.categories._formatter_func

    def _format_attrs(self) -> List[Tuple[str, str | int | bool | None]]:
        """
        Return a list of tuples of the (attr,formatted_value)
        """
        attrs: List[Tuple[str, str | int | bool | None]]
        attrs = [('categories', f"[{', '.join(self._data._repr_categories())}]"), ('ordered', self.ordered)]
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
            container: Union[Index, libindex.IndexEngine, libindex.ExtensionEngine] = self.categories
        else:
            container = self._engine
        return contains(self, key, container=container)

    def reindex(self, target: Any, method: Optional[str] = None, level: Optional[int] = None, 
                limit: Optional[int] = None, tolerance: Optional[Any] = None) -> Tuple[Index, Optional[npt.NDArray[np.intp]]]:
        """
        Create index with target's values (move/add/delete values as necessary)

        Returns
        -------
        new_index : pd.Index
            Resulting index
        indexer : np.ndarray[np.intp] or None
            Indices of output values in original index

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

    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        return self.categories._is_comparable_dtype(dtype)

    def map(self, mapper: Union[Dict, Callable, Index], na_action: Optional[Literal['ignore']] = None) -> Index:
        """
        Map values using input an input mapping or function.

        Maps the values (their categories, not the codes) of the index to new
        categories. If the mapping correspondence is one-to-one the result is a
        :class:`~pandas.CategoricalIndex` which has the same order property as
        the original, otherwise an :class:`~pandas.Index` is returned.

        If a `dict` or :class:`~pandas.Series` is used any unmapped category is
        mapped to `NaN`. Note that if this happens an :class:`~pandas.Index`
        will be returned.

        Parameters
        ----------
        mapper : function, dict, or Series
            Mapping correspondence.
        na_action : {None, 'ignore'}, default 'ignore'
            If 'ignore', propagate NaN values, without passing them to
            the mapping correspondence.

        Returns
        -------
        pandas.CategoricalIndex or pandas.Index
            Mapped index.

        See Also
        --------
        Index.map : Apply a mapping correspondence on an
            :class:`~pandas.Index`.
        Series.map : Apply a mapping correspondence on a
            :class:`~pandas.Series`.
        Series.apply : Apply more complex functions on a
            :class:`~pandas.Series`.

        Examples
        --------
        >>> idx = pd.CategoricalIndex(["a", "b", "c"])
        >>> idx
        CategoricalIndex(['a', 'b', 'c'], categories=['a', 'b', 'c'],
                          ordered=False, dtype='category')
        >>> idx.map(lambda x: x.upper())
        CategoricalIndex(['A', 'B', 'C'], categories=['A', 'B', 'C'],
                         ordered=False, dtype='category')
        >>> idx.map({"a": "first", "b": "second", "c": "third"})
        CategoricalIndex(['first', 'second', 'third'], categories=['first',
                         'second', 'third'], ordered=False, dtype='category')

        If the mapping is one-to-one the ordering of the categories is
        preserved:

        >>> idx = pd.CategoricalIndex(["a", "b", "c"], ordered=True)
        >>> idx
        CategoricalIndex(['a', 'b', 'c'], categories=['a', 'b', 'c'],
                         ordered=True, dtype='category')
        >>> idx.map({"a": 3, "b": 2, "c": 1})
        CategoricalIndex([3, 2, 1], categories=[3, 2, 1], ordered=True,
                         dtype='category')

        If the mapping is not one-to-one an :class:`~pandas.Index` is returned:

        >>> idx.map({"a": "first", "b": "second", "c": "first"})
        Index(['first', 'second', 'first'], dtype='object')

        If a `dict` is used, all unmapped categories are mapped to `NaN` and
        the result is an :class:`~pandas.Index`:

        >>> idx.map({"a": "first", "b": "second"})
        Index(['first', 'second', nan], dtype='object')
        """
        mapped = self._values.map(mapper, na_action=na_action)
        return Index(mapped, name=self.name)

    def _concat(self, to_concat: List[Index], name: Hashable) -> Index:
        try:
            cat = Categorical._concat_same_type([self._is_dtype_compat(c) for c in to_concat])
        except TypeError:
            res = concat_compat([x._values for x in to_concat])
            return Index(res, name=name)
        else:
            return type(self)._simple_new(cat, name=name)
