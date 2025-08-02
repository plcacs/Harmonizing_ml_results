from __future__ import annotations
from contextlib import suppress
import sys
from typing import TYPE_CHECKING, Any, TypeVar, cast, final, Optional, Tuple, Union, List, Dict, Sequence, Callable
import warnings
import numpy as np
from pandas._libs.indexing import NDFrameIndexerBase
from pandas._libs.lib import item_from_zerodim
from pandas.compat import PYPY
from pandas.errors import AbstractMethodError, ChainedAssignmentError, IndexingError, InvalidIndexError, LossySetitemError
from pandas.errors.cow import _chained_assignment_msg
from pandas.util._decorators import doc
from pandas.core.dtypes.cast import can_hold_element, maybe_promote
from pandas.core.dtypes.common import (
    is_array_like,
    is_bool_dtype,
    is_hashable,
    is_integer,
    is_iterator,
    is_list_like,
    is_numeric_dtype,
    is_object_dtype,
    is_scalar,
    is_sequence,
)
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.dtypes.missing import (
    construct_1d_array_from_inferred_fill_value,
    infer_fill_value,
    is_valid_na_for_dtype,
    isna,
    na_value_for_dtype,
)
from pandas.core import algorithms as algos
import pandas.core.common as com
from pandas.core.construction import array as pd_array, extract_array
from pandas.core.indexers import (
    check_array_indexer,
    is_list_like_indexer,
    is_scalar_indexer,
    length_of_indexer,
)
from pandas.core.indexes.api import Index, MultiIndex

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence
    from pandas._typing import Axis, AxisInt, Self, npt
    from pandas import DataFrame, Series

T = TypeVar('T')
_NS = slice(None, None)
_one_ellipsis_message = "indexer may only contain one '...' entry"


class _IndexSlice:
    """
    Create an object to more easily perform multi-index slicing.

    See Also
    --------
    MultiIndex.remove_unused_levels : New MultiIndex with no unused levels.

    Notes
    -----
    See :ref:`Defined Levels <advanced.shown_levels>`
    for further info on slicing a MultiIndex.

    Examples
    --------
    >>> midx = pd.MultiIndex.from_product([["A0", "A1"], ["B0", "B1", "B2", "B3"]])
    >>> columns = ["foo", "bar"]
    >>> dfmi = pd.DataFrame(
    ...     np.arange(16).reshape((len(midx), len(columns))),
    ...     index=midx,
    ...     columns=columns,
    ... )

    Using the default slice command:

    >>> dfmi.loc[(slice(None), slice("B0", "B1")), :]
               foo  bar
        A0 B0    0    1
           B1    2    3
        A1 B0    8    9
           B1   10   11

    Using the IndexSlice class for a more intuitive command:

    >>> idx = pd.IndexSlice
    >>> dfmi.loc[idx[:, "B0":"B1"], :]
               foo  bar
        A0 B0    0    1
           B1    2    3
        A1 B0    8    9
           B1   10   11
    """

    def __getitem__(self, arg: Any) -> Any:
        return arg


IndexSlice = _IndexSlice()


class IndexingMixin:
    """
    Mixin for adding .loc/.iloc/.at/.iat to Dataframes and Series.
    """

    @property
    def iloc(self) -> _iLocIndexer:
        """
        Purely integer-location based indexing for selection by position.

        .. versionchanged:: 3.0

           Callables which return a tuple are deprecated as input.

        ``.iloc[]`` is primarily integer position based (from ``0`` to
        ``length-1`` of the axis), but may also be used with a boolean
        array.

        Allowed inputs are:

        - An integer, e.g. ``5``.
        - A list or array of integers, e.g. ``[4, 3, 0]``.
        - A slice object with ints, e.g. ``1:7``.
        - A boolean array.
        - A ``callable`` function with one argument (the calling Series or
          DataFrame) and that returns valid output for indexing (one of the above).
          This is useful in method chains, when you don't have a reference to the
          calling object, but would like to base your selection on
          some value.
        - A tuple of row and column indexes. The tuple elements consist of one of the
          above inputs, e.g. ``(0, 1)``.

        ``.iloc`` will raise ``IndexError`` if a requested indexer is
        out-of-bounds, except *slice* indexers which allow out-of-bounds
        indexing (this conforms with python/numpy *slice* semantics).

        See more at :ref:`Selection by Position <indexing.integer>`.

        See Also
        --------
        DataFrame.iat : Fast integer location scalar accessor.
        DataFrame.loc : Purely label-location based indexer for selection by label.
        Series.iloc : Purely integer-location based indexing for
                       selection by position.

        Examples
        --------
        >>> mydict = [
        ...     {"a": 1, "b": 2, "c": 3, "d": 4},
        ...     {"a": 100, "b": 200, "c": 300, "d": 400},
        ...     {"a": 1000, "b": 2000, "c": 3000, "d": 4000},
        ... ]
        >>> df = pd.DataFrame(mydict)
        >>> df
              a     b     c     d
        0     1     2     3     4
        1   100   200   300   400
        2  1000  2000  3000  4000

        **Indexing just the rows**

        With a scalar integer.

        >>> type(df.iloc[0])
        <class 'pandas.Series'>
        >>> df.iloc[0]
        a    1
        b    2
        c    3
        d    4
        Name: 0, dtype: int64

        With a list of integers.

        >>> df.iloc[[0]]
           a  b  c  d
        0  1  2  3  4
        >>> type(df.iloc[[0]])
        <class 'pandas.DataFrame'>

        >>> df.iloc[[0, 1]]
             a    b    c    d
        0    1    2    3    4
        1  100  200  300  400

        With a `slice` object.

        >>> df.iloc[:3]
              a     b     c     d
        0     1     2     3     4
        1   100   200   300   400
        2  1000  2000  3000  4000

        With a boolean mask the same length as the index.

        >>> df.iloc[[True, False, True]]
              a     b     c     d
        0     1     2     3     4
        2  1000  2000  3000  4000

        With a callable, useful in method chains. The `x` passed
        to the ``lambda`` is the DataFrame being sliced. This selects
        the rows whose index label even.

        >>> df.iloc[lambda x: x.index % 2 == 0]
              a     b     c     d
        0     1     2     3     4
        2  1000  2000  3000  4000

        **Indexing both axes**

        You can mix the indexer types for the index and columns. Use ``:`` to
        select the entire axis.

        With scalar integers.

        >>> df.iloc[0, 1]
        2

        With lists of integers.

        >>> df.iloc[[0, 2], [1, 3]]
              b     d
        0     2     4
        2  2000  4000

        With `slice` objects.

        >>> df.iloc[1:3, 0:3]
              a     b     c
        1   100   200   300
        2  1000  2000  3000

        With a boolean array whose length matches the columns.

        >>> df.iloc[:, [True, False, True, False]]
              a     c
        0     1     3
        1   100   300
        2  1000  3000

        With a callable function that expects the Series or DataFrame.

        >>> df.iloc[:, lambda df: [0, 2]]
              a     c
        0     1     3
        1   100   300
        2  1000  3000
        """
        return _iLocIndexer('iloc', self)

    @property
    def loc(self) -> _LocIndexer:
        """
        Access a group of rows and columns by label(s) or a boolean array.

        ``.loc[]`` is primarily label based, but may also be used with a
        boolean array.

        Allowed inputs are:

        - A single label, e.g. ``5`` or ``'a'``, (note that ``5`` is
          interpreted as a *label* of the index, and **never** as an
          integer position along the index).
        - A list or array of labels, e.g. ``['a', 'b', 'c']``.
        - A slice object with labels, e.g. ``'a':'f'``.

          .. warning:: Note that contrary to usual python slices, **both** the
              start and the stop are included

        - A boolean array of the same length as the axis being sliced,
          e.g. ``[True, False, True]``.
        - An alignable boolean Series. The index of the key will be aligned before
          masking.
        - An alignable Index. The Index of the returned selection will be the input.
        - A ``callable`` function with one argument (the calling Series or
          DataFrame) and that returns valid output for indexing (one of the above)

        See more at :ref:`Selection by Label <indexing.label>`.

        Raises
        ------
        KeyError
            If any items are not found.
        IndexingError
            If an indexed key is passed and its index is unalignable to the frame index.

        See Also
        --------
        DataFrame.at : Access a single value for a row/column label pair.
        DataFrame.iloc : Access group of rows and columns by integer position(s).
        DataFrame.xs : Returns a cross-section (row(s) or column(s)) from the
                       Series/DataFrame.
        Series.loc : Access group of values using labels.

        Examples
        --------
        **Getting values**

        >>> df = pd.DataFrame(
        ...     [[1, 2], [4, 5], [7, 8]],
        ...     index=["cobra", "viper", "sidewinder"],
        ...     columns=["max_speed", "shield"],
        ... )
        >>> df
                    max_speed  shield
        cobra               1       2
        viper               4       5
        sidewinder          7       8

        Single label. Note this returns the row as a Series.

        >>> df.loc["viper"]
        max_speed    4
        shield       5
        Name: viper, dtype: int64

        List of labels. Note using ``[[]]`` returns a DataFrame.

        >>> df.loc[["viper", "sidewinder"]]
                    max_speed  shield
        viper               4       5
        sidewinder          7       8

        Single label for row and column

        >>> df.loc["cobra", "shield"]
        2

        Slice with labels for row and single label for column. As mentioned
        above, note that both the start and stop of the slice are included.

        >>> df.loc["cobra":"viper", "max_speed"]
        cobra    1
        viper    4
        Name: max_speed, dtype: int64

        Boolean list with the same length as the row axis

        >>> df.loc[[False, False, True]]
                    max_speed  shield
        sidewinder          7       8

        Alignable boolean Series:

        >>> df.loc[
        ...     pd.Series([False, True, False], index=["viper", "sidewinder", "cobra"])
        ... ]
                                 max_speed  shield
        sidewinder          7       8

        Index (same behavior as ``df.reindex``)

        >>> df.loc[pd.Index(["cobra", "viper"], name="foo")]
               max_speed  shield
        foo
        cobra          1       2
        viper          4       5

        Conditional that returns a boolean Series

        >>> df.loc[df["shield"] > 6]
                    max_speed  shield
        sidewinder          7       8

        Conditional that returns a boolean Series with column labels specified

        >>> df.loc[df["shield"] > 6, ["max_speed"]]
                    max_speed
        sidewinder          7

        Multiple conditional using ``&`` that returns a boolean Series

        >>> df.loc[(df["max_speed"] > 1) & (df["shield"] < 8)]
                    max_speed  shield
        viper          4       5

        Multiple conditional using ``|`` that returns a boolean Series

        >>> df.loc[(df["max_speed"] > 4) | (df["shield"] < 5)]
                    max_speed  shield
        cobra               1       2
        sidewinder          7       8

        Please ensure that each condition is wrapped in parentheses ``()``.
        See the :ref:`user guide<indexing.boolean>`
        for more details and explanations of Boolean indexing.

        .. note::
            If you find yourself using 3 or more conditionals in ``.loc[]``,
            consider using :ref:`advanced indexing<advanced.advanced_hierarchical>`.

            See below for using ``.loc[]`` on MultiIndex DataFrames.

        Callable that returns a boolean Series

        >>> df.loc[lambda df: df["shield"] == 8]
                    max_speed  shield
        sidewinder          7       8

        **Setting values**

        Set value for all items matching the list of labels

        >>> df.loc[["viper", "sidewinder"], ["shield"]] = 50
        >>> df
                    max_speed  shield
        cobra               1       2
        viper               4      50
        sidewinder          7      50

        Set value for an entire row

        >>> df.loc["cobra"] = 10
        >>> df
                    max_speed  shield
        cobra              10      10
        viper               4      50
        sidewinder          7      50

        Set value for an entire column

        >>> df.loc[:, "max_speed"] = 30
        >>> df
                    max_speed  shield
        cobra              30      10
        viper              30      50
        sidewinder         30      50

        Set value for rows matching callable condition

        >>> df.loc[df["shield"] > 35] = 0
        >>> df
                    max_speed  shield
        cobra              30      10
        viper               0       0
        sidewinder          0       0

        Add value matching location

        >>> df.loc["viper", "shield"] += 5
        >>> df
                    max_speed  shield
        cobra              30      10
        viper               0       5
        sidewinder          0       0

        Setting using a ``Series`` or a ``DataFrame`` sets the values matching the
        index labels, not the index positions.

        >>> shuffled_df = df.loc[["viper", "cobra", "sidewinder"]]
        >>> df.loc[:] += shuffled_df
        >>> df
                    max_speed  shield
        cobra              60      20
        viper               0      10
        sidewinder          0       0

        **Getting values on a DataFrame with an index that has integer labels**

        Another example using integers for the index

        >>> df = pd.DataFrame(
        ...     [[1, 2], [4, 5], [7, 8]],
        ...     index=[7, 8, 9],
        ...     columns=["max_speed", "shield"],
        ... )
        >>> df
           max_speed  shield
        7          1       2
        8          4       5
        9          7       8

        Slice with integer labels for rows. As mentioned above, note that both
        the start and stop of the slice are included.

        >>> df.loc[7:9]
           max_speed  shield
        7          1       2
        8          4       5
        9          7       8

        **Getting values with a MultiIndex**

        A number of examples using a DataFrame with a MultiIndex

        >>> tuples = [
        ...     ("cobra", "mark i"),
        ...     ("cobra", "mark ii"),
        ...     ("sidewinder", "mark i"),
        ...     ("sidewinder", "mark ii"),
        ...     ("viper", "mark ii"),
        ...     ("viper", "mark iii"),
        ... ]
        >>> index = pd.MultiIndex.from_tuples(tuples)
        >>> values = [[12, 2], [0, 4], [10, 20], [1, 4], [7, 1], [16, 36]]
        >>> df = pd.DataFrame(values, columns=["max_speed", "shield"], index=index)
        >>> df
                                 max_speed  shield
        cobra      mark i           12       2
                   mark ii           0       4
        sidewinder mark i           10      20
                   mark ii           1       4
        viper      mark ii           7       1
                   mark iii         16      36

        Single label. Note this returns a DataFrame with a single index.

        >>> df.loc["cobra"]
                 max_speed  shield
        mark i          12       2
        mark ii          0       4

        Single index tuple. Note this returns a Series.

        >>> df.loc[("cobra", "mark ii")]
        max_speed    0
        shield       4
        Name: (cobra, mark ii), dtype: int64

        Single label for row and column. Similar to passing in a tuple, this
        returns a Series.

        >>> df.loc["cobra", "mark i"]
        max_speed    12
        shield        2
        Name: (cobra, mark i), dtype: int64

        Single tuple. Note using ``[[]]`` returns a DataFrame.

        >>> df.loc[[("cobra", "mark ii")]]
                       max_speed  shield
        cobra mark ii          0       4

        Single tuple for the index with a single label for the column

        >>> df.loc[("cobra", "mark i"), "shield"]
        2

        Slice from index tuple to single label

        >>> df.loc[("cobra", "mark i") : "viper"]
                                 max_speed  shield
        cobra      mark i           12       2
                   mark ii           0       4
        sidewinder mark i           10      20
                   mark ii           1       4
        viper      mark ii           7       1
                   mark iii         16      36

        Slice from index tuple to index tuple

        >>> df.loc[("cobra", "mark i") : ("viper", "mark ii")]
                                max_speed  shield
        cobra      mark i          12       2
                   mark ii          0       4
        sidewinder mark i          10      20
                   mark ii          1       4
        viper      mark ii          7       1

        Please see the :ref:`user guide<advanced.advanced_hierarchical>`
        for more details and explanations of advanced indexing.
        """
        return _LocIndexer('loc', self)

    @property
    def at(self) -> _AtIndexer:
        """
        Access a single value for a row/column label pair.

        Similar to ``loc``, in that both provide label-based lookups. Use
        ``at`` if you only need to get or set a single value in a DataFrame
        or Series.

        Raises
        ------
        KeyError
            If getting a value and 'label' does not exist in a DataFrame or Series.

        ValueError
            If row/column label pair is not a tuple or if any label
            from the pair is not a scalar for DataFrame.
            If label is list-like (*excluding* NamedTuple) for Series.

        See Also
        --------
        DataFrame.at : Access a single value for a row/column pair by label.
        DataFrame.iat : Access a single value for a row/column pair by integer
            position.
        DataFrame.loc : Access a group of rows and columns by label(s).
        DataFrame.iloc : Access a group of rows and columns by integer position(s).
        Series.at : Access a single value by label.
        Series.iat : Access a single value by integer position.
        Series.loc : Access a group of rows by label(s).
        Series.iloc : Access a group of rows by integer position(s).

        Notes
        -----
        See :ref:`Fast scalar value getting and setting <indexing.basics.get_value>`
        for more details.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     [[0, 2, 3], [0, 4, 1], [10, 20, 30]],
        ...     index=[4, 5, 6],
        ...     columns=["A", "B", "C"],
        ... )
        >>> df
            A   B   C
        4   0   2   3
        5   0   4   1
        6  10  20  30

        Get value at specified row/column pair

        >>> df.at[4, "B"]
        2

        Set value at specified row/column pair

        >>> df.at[4, "B"] = 10
        >>> df.at[4, "B"]
        10

        Get value within a Series

        >>> df.loc[5].at["B"]
        4
        """
        return _AtIndexer('at', self)

    @property
    def iat(self) -> _iAtIndexer:
        """
        Access a single value for a row/column pair by integer position.

        Similar to ``iloc``, in that both provide integer-based lookups. Use
        ``iat`` if you only need to get or set a single value in a DataFrame
        or Series.

        Raises
        ------
        IndexError
            When integer position is out of bounds.

        See Also
        --------
        DataFrame.at : Access a single value for a row/column label pair.
        DataFrame.loc : Access a group of rows and columns by label(s).
        DataFrame.iloc : Access a group of rows and columns by integer position(s).

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     [[0, 2, 3], [0, 4, 1], [10, 20, 30]], columns=["A", "B", "C"]
        ... )
        >>> df
            A   B   C
        0   0   2   3
        1   0   4   1
        2  10  20  30

        Get value at specified row/column pair

        >>> df.iat[1, 2]
        1

        Set value at specified row/column pair

        >>> df.iat[1, 2] = 10
        >>> df.iat[1, 2]
        10

        Get value within a series

        >>> df.loc[0].iat[1]
        2
        """
        return _iAtIndexer('iat', self)


class _LocationIndexer(NDFrameIndexerBase):
    axis: Optional[int] = None

    @final
    def __call__(self, axis: Optional[Axis] = None) -> _LocationIndexer:
        new_self = type(self)(self.name, self.obj)
        if axis is not None:
            axis_int_none = self.obj._get_axis_number(axis)
        else:
            axis_int_none = axis
        new_self.axis = axis_int_none
        return new_self

    def _convert_key(self, key: Any) -> Any:
        # Placeholder for conversion logic
        return key

    @final
    def _convert_tuple(self, key: Tuple[Any, ...]) -> Any:
        # Placeholder for tuple conversion logic
        return key

    def _validate_key_length(self, key: Any) -> Any:
        # Placeholder for key length validation
        return key

    def _expand_ellipsis(self, key: Tuple[Any, ...]) -> Tuple[Any, ...]:
        # Placeholder for ellipsis expansion
        return key

    def _validate_key(self, key: Any, axis: int) -> None:
        # Placeholder for key validation
        pass

    def _get_setitem_indexer(self, key: Any) -> Any:
        # Placeholder for getting setitem indexer
        return key

    def _has_valid_setitem_indexer(self, key: Any) -> None:
        # Placeholder for setitem indexer validation
        pass

    def _raise_callable_usage(self, key: Any, maybe_callable: Any) -> Any:
        # Placeholder for raising callable usage
        return maybe_callable

    def _is_nested_tuple_indexer(self, tup: Tuple[Any, ...]) -> bool:
        # Placeholder for nested tuple indexer check
        return False

    def __setitem__(self, key: Any, value: Any) -> None:
        if not PYPY:
            if sys.getrefcount(self.obj) <= 2:
                warnings.warn(_chained_assignment_msg, ChainedAssignmentError, stacklevel=2)
        check_dict_or_set_indexers(key)
        if isinstance(key, tuple):
            key = tuple(
                list(x) if is_iterator(x) else x for x in key
            )
            key = tuple(com.apply_if_callable(x, self.obj) for x in key)
        else:
            maybe_callable = com.apply_if_callable(key, self.obj)
            key = self._raise_callable_usage(key, maybe_callable)
        indexer = self._get_setitem_indexer(key)
        self._has_valid_setitem_indexer(key)
        iloc = cast('_iLocIndexer', self) if self.name == 'iloc' else self.obj.iloc
        self.func_nnm8v7yr._setitem_with_indexer(indexer, value, self.name)

    @final
    def _ensure_listlike_indexer(self, key: Any, axis: int) -> None:
        # Placeholder for ensuring list-like indexer
        pass

    def _convert_to_indexer(self, key: Any, axis: int) -> Any:
        # Placeholder for converting to indexer
        return key

    @final
    def _setitem_with_indexer(self, key: Any, value: Any, name: str) -> None:
        # Placeholder for setitem with indexer
        pass

    def _setitem_with_indexer_split_path(
        self, indexer: Any, value: Any, name: str
    ) -> None:
        # Placeholder for split path setitem
        pass

    def _setitem_single_block(
        self, indexer: Any, value: Any, name: str
    ) -> None:
        # Placeholder for single block setitem
        pass

    def _get_listlike_indexer(self, key: Any, axis: int) -> Tuple[Index, Any]:
        # Placeholder for getting list-like indexer
        return (Index(key), key)

    def _getitem_axis(self, key: Any, axis: int) -> Any:
        # Placeholder for getitem axis
        return key

    def _getitem_tuple(self, key: Tuple[Any, ...]) -> Any:
        # Placeholder for getitem tuple
        return key

    def _multi_take_opportunity(self, tup: Tuple[Any, ...]) -> bool:
        # Placeholder for multi take opportunity
        return False

    def _getitem_tuple_same_dim(self, tup: Tuple[Any, ...]) -> Any:
        # Placeholder for getitem tuple same dimension
        return tup

    def _handle_lowerdim_multi_index_axis0(self, tup: Tuple[Any, ...]) -> Any:
        # Placeholder for handling lower dimension multi-index
        return tup

    @final
    def _multi_take(self, tup: Tuple[Any, ...]) -> Any:
        # Placeholder for multi take
        return tup
     
    @final
    def __getitem__(self, key: Any) -> Any:
        check_dict_or_set_indexers(key)
        if isinstance(key, tuple):
            key = tuple(com.apply_if_callable(x, self.obj) for x in key)
            if self._is_scalar_access(key):
                return self.obj._get_value(*key, takeable=self._takeable)
            return self._getitem_tuple(key)
        else:
            axis: int = self.axis or 0
            maybe_callable = com.apply_if_callable(key, self.obj)
            maybe_callable = self._raise_callable_usage(key, maybe_callable)
            return self._getitem_axis(maybe_callable, axis=axis)

    def _validate_key_length(self, key: Any) -> Any:
        # Placeholder for key length validation
        return key

    def _validate_tuple_indexer(self, tup: Tuple[Any, ...]) -> Tuple[Any, ...]:
        # Placeholder for tuple indexer validation
        return tup

    def _convert_key_if_needed(self, key: Any) -> Any:
        # Placeholder for key conversion
        return key

    @final
    def __setitem__(self, key: Any, value: Any) -> None:
        # Placeholder for setitem logic
        pass

    @final
    def _validate_key_length(self, key: Any) -> Any:
        # Placeholder for key length validation
        return key

    @final
    def _validate_key(self, key: Any, axis: int) -> None:
        # Placeholder for key validation
        pass

    @final
    def _get_label(self, key: Any, axis: int) -> Any:
        # Placeholder for get_label
        return key

    @final
    def _get_listlike_indexer(self, key: Any, axis: int) -> Tuple[Index, Any]:
        # Placeholder for getting list-like indexer
        return (Index(key), key)

    @final
    def _get_slice_axis(self, slice_obj: slice, axis: int) -> Any:
        # Placeholder for getting slice on axis
        return slice_obj

    @final
    def _getbool_axis(self, key: np.ndarray, axis: int) -> Any:
        # Placeholder for getting boolean axis
        return key

    @final
    def _get_list_axis(self, key: Any, axis: int) -> Any:
        # Placeholder for getting list axis
        return key


@doc(IndexingMixin.loc)
class _LocIndexer(_LocationIndexer):
    _takeable: bool = False
    _valid_types: Tuple[str, ...] = (
        'labels (MUST BE IN THE INDEX), slices of labels (BOTH endpoints included! Can be slices of integers if the index is integers), listlike of labels, boolean'
    )

    @doc(_LocationIndexer._validate_key)
    def _validate_key(self, key: Any, axis: int) -> None:
        ax = self.obj._get_axis(axis)
        if isinstance(key, bool) and not (
            is_bool_dtype(ax.dtype)
            or ax.dtype.name == 'boolean'
            or (isinstance(ax, MultiIndex) and is_bool_dtype(ax.get_level_values(0).dtype))
        ):
            raise KeyError(
                f'{key}: boolean label can not be used without a boolean index'
            )
        if isinstance(key, slice) and (isinstance(key.start, bool) or isinstance(key.stop, bool)):
            raise TypeError(f'{key}: boolean values can not be used in a slice')

    def _has_valid_setitem_indexer(self, indexer: Any) -> bool:
        return True

    def _is_scalar_access(self, key: Tuple[Any, ...]) -> bool:
        if len(key) != self.ndim:
            return False
        for i, k in enumerate(key):
            if not is_scalar(k):
                return False
            ax = self.obj.axes[i]
            if isinstance(ax, MultiIndex):
                return False
            if isinstance(k, str) and ax._supports_partial_string_indexing:
                return False
            if not ax._index_as_unique:
                return False
        return True

    def _soft_apply_multi_take(self, tup: Tuple[Any, ...]) -> bool:
        if not all(is_list_like_indexer(x) for x in tup):
            return False
        return not any(com.is_bool_indexer(x) for x in tup)

    def _multi_take_opportunity(self, tup: Tuple[Any, ...]) -> bool:
        return self._soft_apply_multi_take(tup)

    def _multi_take(self, tup: Tuple[Any, ...]) -> Any:
        d: Dict[int, Any] = {axis: self._get_listlike_indexer(key, axis) for key, axis in zip(tup, self.obj._AXIS_ORDERS)}
        return self.obj._reindex_with_indexers(d, allow_dups=True)

    def _getitem_tuple_same_dim(self, tup: Tuple[Any, ...]) -> Any:
        return self.obj._reindex_with_indexers({axis: key for axis, key in enumerate(tup)}, allow_dups=True)

    def _handle_lowerdim_multi_index_axis0(self, tup: Tuple[Any, ...]) -> Any:
        return self.obj.xs(tup, axis=0)

    def _handle_lowerdim_multi_index_axis1(self, tup: Tuple[Any, ...]) -> Any:
        return self.obj.xs(tup, axis=1)

    def _get_label(self, label: Any, axis: int) -> Any:
        try:
            return self.obj.xs(label, axis=axis)
        except KeyError as ek:
            if self.ndim < len(label) <= self.obj.index.nlevels:
                raise ek
            raise IndexingError('No label returned') from ek

    def _validate_key_length(self, key: Any) -> Tuple[Any, ...]:
        return super()._validate_key_length(key)

    def _expand_ellipsis(self, key: Tuple[Any, ...]) -> Tuple[Any, ...]:
        return super()._expand_ellipsis(key)

    @final
    def __getitem__(self, key: Any) -> Any:
        check_dict_or_set_indexers(key)
        if isinstance(key, tuple):
            key = tuple(com.apply_if_callable(x, self.obj) for x in key)
            try:
                return self.obj.xs(key, axis=0)
            except IndexingError:
                return super().__getitem__(key)
        else:
            axis: int = self.axis or 0
            maybe_callable = com.apply_if_callable(key, self.obj)
            maybe_callable = self._raise_callable_usage(key, maybe_callable)
            return self._getitem_axis(maybe_callable, axis=axis)


@doc(IndexingMixin.iloc)
class _iLocIndexer(_LocationIndexer):
    _valid_types: Tuple[str, ...] = (
        'integer, integer slice (START point is INCLUDED, END point is EXCLUDED), listlike of integers, boolean array'
    )
    _takeable: bool = True

    def _validate_key(self, key: Any, axis: int) -> None:
        if com.is_bool_indexer(key):
            if hasattr(key, 'index') and isinstance(key.index, Index):
                if key.index.inferred_type == 'integer':
                    raise NotImplementedError(
                        'iLocation based boolean indexing on an integer type is not available'
                    )
                raise ValueError(
                    'iLocation based boolean indexing cannot use an indexable as a mask'
                )
            return
        if isinstance(key, slice):
            return
        elif is_integer(key):
            self._validate_integer(key, axis)
        elif isinstance(key, tuple):
            raise IndexingError('Too many indexers')
        elif is_list_like_indexer(key):
            if isinstance(key, ABCSeries):
                arr = key._values
            elif is_array_like(key):
                arr = key
            else:
                arr = np.array(key)
            len_axis = len(self.obj._get_axis(axis))
            if not is_numeric_dtype(arr.dtype):
                raise IndexError(f'.iloc requires numeric indexers, got {arr}')
            if len(arr) and (arr.max() >= len_axis or arr.min() < -len_axis):
                raise IndexError('positional indexers are out-of-bounds')
        else:
            raise ValueError(
                f'Can only index by location with a [{self._valid_types}]'
            )

    def _has_valid_setitem_indexer(self, indexer: Any) -> bool:
        if isinstance(indexer, dict):
            raise IndexError('iloc cannot enlarge its target object')
        if isinstance(indexer, ABCDataFrame):
            raise TypeError(
                'DataFrame indexer for .iloc is not supported. Consider using .loc with a DataFrame indexer for automatic alignment.'
            )
        if not isinstance(indexer, tuple):
            indexer = _tuplify(self.ndim, indexer)
        for ax, i in zip(self.obj.axes, indexer):
            if isinstance(i, slice):
                pass
            elif is_list_like_indexer(i):
                pass
            elif is_integer(i):
                if i >= len(ax):
                    raise IndexError('iloc cannot enlarge its target object')
            elif isinstance(i, dict):
                raise IndexError('iloc cannot enlarge its target object')
        return True

    def _is_scalar_access(self, key: Tuple[Any, ...]) -> bool:
        if len(key) != self.ndim:
            return False
        return all(is_integer(k) for k in key)

    def _validate_integer(self, key: int, axis: int) -> None:
        len_axis = len(self.obj._get_axis(axis))
        if key >= len_axis or key < -len_axis:
            raise IndexError('single positional indexer is out-of-bounds')

    def _getitem_tuple_same_dim(self, tup: Tuple[Any, ...]) -> Any:
        return self.obj._reindex_with_indexers({axis: key for axis, key in enumerate(tup)}, allow_dups=True)

    @final
    def __getitem__(self, key: Any) -> Any:
        check_dict_or_set_indexers(key)
        if isinstance(key, tuple):
            tup = self._validate_tuple_indexer(key)
            try:
                return self._getitem_lowerdim(tup)
            except IndexingError:
                return self._getitem_tuple_same_dim(tup)
        else:
            axis: int = self.axis or 0
            maybe_callable = com.apply_if_callable(key, self.obj)
            maybe_callable = self._raise_callable_usage(key, maybe_callable)
            return self._getitem_axis(maybe_callable, axis=axis)

    def _getitem_axis(self, key: Any, axis: int) -> Any:
        if key is Ellipsis:
            key = slice(None)
        elif isinstance(key, ABCDataFrame):
            raise IndexError(
                """DataFrame indexer is not allowed for .iloc
Consider using .loc for automatic alignment."""
            )
        if isinstance(key, slice):
            return self._get_slice_axis(key, axis=axis)
        if is_iterator(key):
            key = list(key)
        if isinstance(key, list):
            key = np.asarray(key)
        if com.is_bool_indexer(key):
            self._validate_key(key, axis=axis)
            return self._getbool_axis(key, axis=axis)
        elif is_list_like_indexer(key):
            return self._get_list_axis(key, axis=axis)
        else:
            key = item_from_zerodim(key)
            if not is_integer(key):
                raise TypeError(
                    'Cannot index by location index with a non-integer key'
                )
            self._validate_integer(key, axis)
            return self.obj._ixs(key, axis=axis)

    def _get_slice_axis(self, slice_obj: slice, axis: int) -> Any:
        obj = self.obj
        if not (slice_obj.start or slice_obj.stop or (slice_obj.step and slice_obj.step != 1)):
            return obj.copy(deep=False)
        labels = obj._get_axis(axis)
        labels._validate_positional_slice(slice_obj)
        return self.obj._slice(slice_obj, axis=axis)

    def _get_label(self, label: Any, axis: int) -> Any:
        return self.obj._ixs(label, axis=axis)


class _ScalarAccessIndexer(NDFrameIndexerBase):
    """
    Access scalars quickly.
    """

    def _convert_key(self, key: Any) -> Any:
        raise AbstractMethodError(self)

    def _get_value(self, *args: Any, takeable: bool) -> Any:
        raise AbstractMethodError(self)

    def _set_value(self, *args: Any, value: Any, takeable: bool) -> None:
        raise AbstractMethodError(self)

    def __getitem__(self, key: Any) -> Any:
        if not isinstance(key, tuple):
            if not is_list_like_indexer(key):
                key = (key,)
            else:
                raise ValueError('Invalid call for scalar access (getting)!')
        key = self._convert_key(key)
        return self.obj._get_value(*key, takeable=self._takeable)

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, tuple):
            key = tuple(com.apply_if_callable(x, self.obj) for x in key)
        else:
            key = com.apply_if_callable(key, self.obj)
        if not isinstance(key, tuple):
            key = _tuplify(self.ndim, key)
        key = list(self._convert_key(key))
        if len(key) != self.ndim:
            raise ValueError('Not enough indexers for scalar access (setting)!')
        self.obj._set_value(*key, value=value, takeable=self._takeable)


@doc(IndexingMixin.at)
class _AtIndexer(_ScalarAccessIndexer):
    _takeable: bool = False

    def _convert_key(self, key: Any) -> Any:
        if self.ndim == 1 and len(key) > 1:
            key = (key,)
        return key

    @property
    def _is_unique(self) -> bool:
        assert self.ndim == 2
        return self.obj.index.is_unique and self.obj.columns.is_unique

    def __getitem__(self, key: Any) -> Any:
        if self.ndim == 2 and not self._is_unique:
            if not isinstance(key, tuple) or not all(is_scalar(x) for x in key):
                raise ValueError('Invalid call for scalar access (getting)!')
            return self.obj.loc[key]
        return super().__getitem__(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        if self.ndim == 2 and not self._is_unique:
            if not isinstance(key, tuple) or not all(is_scalar(x) for x in key):
                raise ValueError('Invalid call for scalar access (setting)!')
            self.obj.loc[key] = value
            return
        super().__setitem__(key, value)


@doc(IndexingMixin.iat)
class _iAtIndexer(_ScalarAccessIndexer):
    _takeable: bool = True

    def _convert_key(self, key: Any) -> Any:
        if not all(is_integer(k) for k in key):
            raise ValueError(
                'iAt based indexing can only have integer indexers'
            )
        return key
