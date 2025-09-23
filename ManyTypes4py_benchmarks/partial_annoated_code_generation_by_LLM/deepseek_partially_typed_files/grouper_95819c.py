"""
Provide user facing operators for doing the split part of the
split-apply-combine paradigm.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, final, Any, Callable, Union, Optional, cast
import numpy as np
from pandas._libs.tslibs import OutOfBoundsDatetime
from pandas.errors import InvalidIndexError
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.common import is_list_like, is_scalar
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas.core import algorithms
from pandas.core.arrays import Categorical, ExtensionArray
import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.core.groupby import ops
from pandas.core.groupby.categorical import recode_for_groupby
from pandas.core.indexes.api import Index, MultiIndex, default_index
from pandas.core.series import Series
from pandas.io.formats.printing import pprint_thing
if TYPE_CHECKING:
    from collections.abc import Hashable, Iterator
    from pandas._typing import ArrayLike, NDFrameT, npt
    from pandas.core.generic import NDFrame

class Grouper:
    """
    A Grouper allows the user to specify a groupby instruction for an object.

    This specification will select a column via the key parameter, or if the
    level parameter is given, a level of the index of the target
    object.

    If ``level`` is passed as a keyword to both `Grouper` and
    `groupby`, the values passed to `Grouper` take precedence.

    Parameters
    ----------
    *args
        Currently unused, reserved for future use.
    **kwargs
        Dictionary of the keyword arguments to pass to Grouper.

    Attributes
    ----------
    key : str, defaults to None
        Groupby key, which selects the grouping column of the target.
    level : name/number, defaults to None
        The level for the target index.
    freq : str / frequency object, defaults to None
        This will groupby the specified frequency if the target selection
        (via key or level) is a datetime-like object. For full specification
        of available frequencies, please see :ref:`here<timeseries.offset_aliases>`.
    sort : bool, default to False
        Whether to sort the resulting labels.
    closed : {'left' or 'right'}
        Closed end of interval. Only when `freq` parameter is passed.
    label : {'left' or 'right'}
        Interval boundary to use for labeling.
        Only when `freq` parameter is passed.
    convention : {'start', 'end', 'e', 's'}
        If grouper is PeriodIndex and `freq` parameter is passed.

    origin : Timestamp or str, default 'start_day'
        The timestamp on which to adjust the grouping. The timezone of origin must
        match the timezone of the index.
        If string, must be one of the following:

        - 'epoch': `origin` is 1970-01-01
        - 'start': `origin` is the first value of the timeseries
        - 'start_day': `origin` is the first day at midnight of the timeseries

        - 'end': `origin` is the last value of the timeseries
        - 'end_day': `origin` is the ceiling midnight of the last day

        .. versionadded:: 1.3.0

    offset : Timedelta or str, default is None
        An offset timedelta added to the origin.

    dropna : bool, default True
        If True, and if group keys contain NA values, NA values together with
        row/column will be dropped. If False, NA values will also be treated as
        the key in groups.

    Returns
    -------
    Grouper or pandas.api.typing.TimeGrouper
        A TimeGrouper is returned if ``freq`` is not ``None``. Otherwise, a Grouper
        is returned.

    See Also
    --------
    Series.groupby : Apply a function groupby to a Series.
    DataFrame.groupby : Apply a function groupby.

    Examples
    --------
    ``df.groupby(pd.Grouper(key="Animal"))`` is equivalent to ``df.groupby('Animal')``

    >>> df = pd.DataFrame(
    ...     {
    ...         "Animal": ["Falcon", "Parrot", "Falcon", "Falcon", "Parrot"],
    ...         "Speed": [100, 5, 200, 300, 15],
    ...     }
    ... )
    >>> df
       Animal  Speed
    0  Falcon    100
    1  Parrot      5
    2  Falcon    200
    3  Falcon    300
    4  Parrot     15
    >>> df.groupby(pd.Grouper(key="Animal")).mean()
            Speed
    Animal
    Falcon  200.0
    Parrot   10.0

    Specify a resample operation on the column 'Publish date'

    >>> df = pd.DataFrame(
    ...     {
    ...         "Publish date": [
    ...             pd.Timestamp("2000-01-02"),
    ...             pd.Timestamp("2000-01-02"),
    ...             pd.Timestamp("2000-01-09"),
    ...             pd.Timestamp("2000-01-16"),
    ...         ],
    ...         "ID": [0, 1, 2, 3],
    ...         "Price": [10, 20, 30, 40],
    ...     }
    ... )
    >>> df
      Publish date  ID  Price
    0   2000-01-02   0     10
    1   2000-01-02   1     20
    2   2000-01-09   2     30
    3   2000-01-16   3     40
    >>> df.groupby(pd.Grouper(key="Publish date", freq="1W")).mean()
                   ID  Price
    Publish date
    2000-01-02    0.5   15.0
    2000-01-09    2.0   30.0
    2000-01-16    3.0   40.0

    If you want to adjust the start of the bins based on a fixed timestamp:

    >>> start, end = "2000-10-01 23:30:00", "2000-10-02 00:30:00"
    >>> rng = pd.date_range(start, end, freq="7min")
    >>> ts = pd.Series(np.arange(len(rng)) * 3, index=rng)
    >>> ts
    2000-10-01 23:30:00     0
    2000-10-01 23:37:00     3
    2000-10-01 23:44:00     6
    2000-10-01 23:51:00     9
    2000-10-01 23:58:00    12
    2000-10-02 00:05:00    15
    2000-10-02 00:12:00    18
    2000-10-02 00:19:00    21
    2000-10-02 00:26:00    24
    Freq: 7min, dtype: int64

    >>> ts.groupby(pd.Grouper(freq="17min")).sum()
    2000-10-01 23:14:00     0
    2000-10-01 23:31:00     9
    2000-10-01 23:48:00    21
    2000-10-02 00:05:00    54
    2000-10-02 00:22:00    24
    Freq: 17min, dtype: int64

    >>> ts.groupby(pd.Grouper(freq="17min", origin="epoch")).sum()
    2000-10-01 23:18:00     0
    2000-10-01 23:35:00    18
    2000-10-01 23:52:00    27
    2000-10-02 00:09:00    39
    2000-10-02 00:26:00    24
    Freq: 17min, dtype: int64

    >>> ts.groupby(pd.Grouper(freq="17min", origin="2000-01-01")).sum()
    2000-10-01 23:24:00     3
    2000-10-01 23:41:00    15
    2000-10-01 23:58:00    45
    2000-10-02 00:15:00    45
    Freq: 17min, dtype: int64

    If you want to adjust the start of the bins with an `offset` Timedelta, the two
    following lines are equivalent:

    >>> ts.groupby(pd.Grouper(freq="17min", origin="start")).sum()
    2000-10-01 23:30:00     9
    2000-10-01 23:47:00    21
    2000-10-02 00:04:00    54
    2000-10-02 00:21:00    24
    Freq: 17min, dtype: int64

    >>> ts.groupby(pd.Grouper(freq="17min", offset="23h30min")).sum()
    2000-10-01 23:30:00     9
    2000-10-01 23:47:00    21
    2000-10-02 00:04:00    54
    2000-10-02 00:21:00    24
    Freq: 17min, dtype: int64

    To replace the use of the deprecated `base` argument, you can now use `offset`,
    in this example it is equivalent to have `base=2`:

    >>> ts.groupby(pd.Grouper(freq="17min", offset="2min")).sum()
    2000-10-01 23:16:00     0
    2000-10-01 23:33:00     9
    2000-10-01 23:50:00    36
    2000-10-02 00:07:00    39
    2000-10-02 00:24:00    24
    Freq: 17min, dtype: int64
    """
    sort: bool
    dropna: bool
    _grouper: Optional[Index]
    _attributes: tuple[str, ...] = ('key', 'level', 'freq', 'sort', 'dropna')
    key: Optional[Hashable]
    level: Optional[Union[Hashable, int]]
    freq: Optional[Any]
    _indexer_deprecated: Optional[npt.NDArray[np.intp]]
    binner: Optional[Any]
    _indexer: Optional[npt.NDArray[np.intp]]

    def __new__(cls, *args: Any, **kwargs: Any) -> Grouper:
        if kwargs.get('freq') is not None:
            from pandas.core.resample import TimeGrouper
            cls = TimeGrouper
        return super().__new__(cls)

    def __init__(self, key: Optional[Hashable] = None, level: Optional[Union[Hashable, int]] = None, freq: Optional[Any] = None, sort: bool = False, dropna: bool = True) -> None:
        self.key = key
        self.level = level
        self.freq = freq
        self.sort = sort
        self.dropna = dropna
        self._indexer_deprecated = None
        self.binner = None
        self._grouper = None
        self._indexer = None

    def _get_grouper(self, obj: NDFrameT, validate: bool = True) -> tuple[ops.BaseGrouper, NDFrameT]:
        """
        Parameters
        ----------
        obj : Series or DataFrame
        validate : bool, default True
            if True, validate the grouper

        Returns
        -------
        a tuple of grouper, obj (possibly sorted)
        """
        (obj, _, _) = self._set_grouper(obj)
        (grouper, _, obj) = get_grouper(obj, [self.key], level=self.level, sort=self.sort, validate=validate, dropna=self.dropna)
        return (grouper, obj)

    def _set_grouper(self, obj: NDFrameT, sort: bool = False, *, gpr_index: Optional[Index] = None) -> tuple[NDFrameT, Index, Optional[npt.NDArray[np.intp]]]:
        """
        given an object and the specifications, setup the internal grouper
        for this particular specification

        Parameters
        ----------
        obj : Series or DataFrame
        sort : bool, default False
            whether the resulting grouper should be sorted
        gpr_index : Index or None, default None

        Returns
        -------
        NDFrame
        Index
        np.ndarray[np.intp] | None
        """
        assert obj is not None
        if self.key is not None and self.level is not None:
            raise ValueError('The Grouper cannot specify both a key and a level!')
        if self._grouper is None:
            self._grouper = gpr_index
            self._indexer = self._indexer_deprecated
        if self.key is not None:
            key = self.key
            if getattr(gpr_index, 'name', None) == key and isinstance(obj, Series):
                assert self._grouper is not None
                if self._indexer is not None:
                    reverse_indexer = self._indexer.argsort()
                    unsorted_ax = self._grouper.take(reverse_indexer)
                    ax = unsorted_ax.take(obj.index)
                else:
                    ax = self._grouper.take(obj.index)
            else:
                if key not in obj._info_axis:
                    raise KeyError(f'The grouper name {key} is not found')
                ax = Index(obj[key], name=key)
        else:
            ax = obj.index
            if self.level is not None:
                level = self.level
                if isinstance(ax, MultiIndex):
                    level = ax._get_level_number(level)
                    ax = Index(ax._get_level_values(level), name=ax.names[level])
                elif level not in (0, ax.name):
                    raise ValueError(f'The level {level} is not valid')
        indexer: Optional[npt.NDArray[np.intp]] = None
        if (self.sort or sort) and (not ax.is_monotonic_increasing):
            indexer = self._indexer_deprecated = ax.array.argsort(kind='mergesort', na_position='first')
            ax = ax.take(indexer)
            obj = obj.take(indexer, axis=0)
        return (obj, ax, indexer)

    @final
    def __repr__(self) -> str:
        attrs_list = (f'{attr_name}={getattr(self, attr_name)!r}' for attr_name in self._attributes if getattr(self, attr_name) is not None)
        attrs = ', '.join(attrs_list)
        cls_name = type(self).__name__
        return f'{cls_name}({attrs})'

@final
class Grouping:
    """
    Holds the grouping information for a single key

    Parameters
    ----------
    index : Index
    grouper :
    obj : DataFrame or Series
    name : Label
    level :
    observed : bool, default False
        If we are a Categorical, use the observed values
    in_axis : if the Grouping is a column in self.obj and hence among
        Groupby.exclusions list
    dropna : bool, default True
        Whether to drop NA groups.
    uniques : Array-like, optional
        When specified, will be used for unique values. Enables including empty groups
        in the result for a BinGrouper. Must not contain duplicates.

    Attributes
    -------
    indices : dict
        Mapping of {group -> index_list}
    codes : ndarray
        Group codes
    group_index : Index or None
        unique groups
    groups : dict
        Mapping of {group -> label_list}
    """
    _codes: Optional[npt.NDArray[np.signedinteger]]
    _orig_cats: Optional[Index]
    _index: Index
    level: Optional[Union[Hashable, int]]
    _orig_grouper: Any
    _sort: bool
    obj: Optional[NDFrame]
    _observed: bool
    in_axis: bool
    _dropna: bool
    _uniques: Optional[ArrayLike]
    grouping_vector: Any

    def __init__(self, index: Index, grouper: Any = None, obj: Optional[NDFrame] = None, level: Optional[Union[Hashable, int]] = None, sort: bool = True, observed: bool = False, in_axis: bool = False, dropna: bool = True, uniques: Optional[ArrayLike] = None) -> None:
        self.level = level
        self._orig_grouper = grouper
        grouping_vector = _convert_grouper(index, grouper)
        self._orig_cats = None
        self._index = index
        self._sort = sort
        self.obj = obj
        self._observed = observed
        self.in_axis = in_axis
        self._dropna = dropna
        self._uniques = uniques
        self._codes = None
        ilevel = self._ilevel
        if ilevel is not None:
            if isinstance(index, MultiIndex):
                index_level = index.get_level_values(ilevel)
            else:
                index_level = index
            if grouping_vector is None:
                grouping_vector = index_level
            else:
                mapper = grouping_vector
                grouping_vector = index_level.map(mapper)
        elif isinstance(grouping_vector, Grouper):
            assert self.obj is not None
            (newgrouper, newobj) = grouping_vector._get_grouper(self.obj, validate=False)
            self.obj = newobj
            if isinstance(newgrouper