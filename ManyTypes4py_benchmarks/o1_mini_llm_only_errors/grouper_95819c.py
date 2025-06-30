"""
Provide user facing operators for doing the split part of the
split-apply-combine paradigm.
"""
from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    final,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
import numpy as np
import pandas as pd
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
    from collections.abc import Hashable
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
    _attributes: Tuple[str, ...] = ('key', 'level', 'freq', 'sort', 'dropna')

    def __new__(cls, *args: Any, **kwargs: Any) -> Grouper:
        if kwargs.get('freq') is not None:
            from pandas.core.resample import TimeGrouper

            cls = TimeGrouper
        return super().__new__(cls)

    def __init__(
        self,
        key: Optional[str] = None,
        level: Optional[Union[str, int]] = None,
        freq: Optional[Union[str, Any]] = None,
        sort: bool = False,
        dropna: bool = True,
    ) -> None:
        self.key: Optional[str] = key
        self.level: Optional[Union[str, int]] = level
        self.freq: Optional[Union[str, Any]] = freq
        self.sort: bool = sort
        self.dropna: bool = dropna
        self._indexer_deprecated: Optional[np.ndarray] = None
        self.binner: Optional[Any] = None
        self._grouper: Optional[Any] = None
        self._indexer: Optional[np.ndarray] = None

    def _get_grouper(
        self, obj: Union[DataFrame, Series], validate: bool = True
    ) -> Tuple[Any, Union[DataFrame, Series]]:
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
        obj, _, _ = self._set_grouper(obj)
        grouper, _, obj = get_grouper(
            obj,
            [self.key],
            level=self.level,
            sort=self.sort,
            validate=validate,
            dropna=self.dropna,
        )
        return (grouper, obj)

    def _set_grouper(
        self,
        obj: Union[DataFrame, Series],
        sort: bool = False,
        *,
        gpr_index: Optional[Index] = None,
    ) -> Tuple[Union[DataFrame, Series], Index, Optional[np.ndarray]]:
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
                ax: Index = Index(obj[key], name=key)
        else:
            ax = obj.index
            if self.level is not None:
                level = self.level
                if isinstance(ax, MultiIndex):
                    level = ax._get_level_number(level)
                    ax = Index(ax._get_level_values(level), name=ax.names[level])
                elif level not in (0, ax.name):
                    raise ValueError(f'The level {level} is not valid')
        indexer: Optional[np.ndarray] = None
        if (self.sort or sort) and (not ax.is_monotonic_increasing):
            indexer = self._indexer_deprecated = ax.array.argsort(kind='mergesort', na_position='first')
            ax = ax.take(indexer)
            obj = obj.take(indexer, axis=0)
        return (obj, ax, indexer)

    @final
    def __repr__(self) -> str:
        attrs_list = (
            f'{attr_name}={getattr(self, attr_name)!r}'
            for attr_name in self._attributes
            if getattr(self, attr_name) is not None
        )
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
    _codes: Optional[np.ndarray] = None

    def __init__(
        self,
        index: Index,
        grouper: Optional[Any] = None,
        obj: Optional[Union[DataFrame, Series]] = None,
        level: Optional[Union[str, int]] = None,
        sort: bool = True,
        observed: bool = False,
        in_axis: bool = False,
        dropna: bool = True,
        uniques: Optional[ArrayLike] = None,
    ) -> None:
        self.level: Optional[Union[str, int]] = level
        self._orig_grouper: Optional[Any] = grouper
        grouping_vector: Union[Any, Callable, np.ndarray, Series, Index, ExtensionArray] = _convert_grouper(index, grouper)
        self._orig_cats: Optional[Index] = None
        self._index: Index = index
        self._sort: bool = sort
        self.obj: Optional[Union[DataFrame, Series]] = obj
        self._observed: bool = observed
        self.in_axis: bool = in_axis
        self._dropna: bool = dropna
        self._uniques: Optional[ArrayLike] = uniques
        ilevel: Optional[int] = self._ilevel
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
            newgrouper, newobj = grouping_vector._get_grouper(self.obj, validate=False)
            self.obj = newobj
            if isinstance(newgrouper, ops.BinGrouper):
                grouping_vector = newgrouper
            else:
                ng = newgrouper.groupings[0].grouping_vector
                grouping_vector = Index(ng, name=newgrouper.result_index.name)
        elif not isinstance(grouping_vector, (Series, Index, ExtensionArray, np.ndarray)):
            if getattr(grouping_vector, 'ndim', 1) != 1:
                t = str(type(grouping_vector))
                raise ValueError(f"Grouper for '{t}' not 1-dimensional")
            grouping_vector = index.map(grouping_vector)
            if not (hasattr(grouping_vector, '__len__') and len(grouping_vector) == len(index)):
                grper = pprint_thing(grouping_vector)
                errmsg = f'Grouper result violates len(labels) == len(data)\nresult: {grper}'
                raise AssertionError(errmsg)
        if isinstance(grouping_vector, np.ndarray):
            if grouping_vector.dtype.kind in 'mM':
                grouping_vector = Series(grouping_vector).to_numpy()
        elif isinstance(getattr(grouping_vector, 'dtype', None), CategoricalDtype):
            self._orig_cats = grouping_vector.categories
            grouping_vector = recode_for_groupby(grouping_vector, sort, observed)
        self.grouping_vector: Union[Any, Callable, np.ndarray, Series, Index, ExtensionArray] = grouping_vector

    def __repr__(self) -> str:
        return f'Grouping({self.name})'

    def __iter__(self) -> Iterator[Any]:
        return iter(self.indices)

    @cache_readonly
    def _passed_categorical(self) -> bool:
        dtype = getattr(self.grouping_vector, 'dtype', None)
        return isinstance(dtype, CategoricalDtype)

    @cache_readonly
    def name(self) -> Optional[str]:
        ilevel = self._ilevel
        if ilevel is not None:
            return self._index.names[ilevel]
        if isinstance(self._orig_grouper, (Index, Series)):
            return self._orig_grouper.name
        elif isinstance(self.grouping_vector, ops.BaseGrouper):
            return self.grouping_vector.result_index.name
        elif isinstance(self.grouping_vector, Index):
            return self.grouping_vector.name
        return None

    @cache_readonly
    def _ilevel(self) -> Optional[int]:
        """
        If necessary, converted index level name to index level position.
        """
        level = self.level
        if level is None:
            return None
        if not isinstance(level, int):
            index = self._index
            if level not in index.names:
                raise AssertionError(f'Level {level} not in index')
            return index.names.index(level)
        return level

    @property
    def ngroups(self) -> int:
        return len(self.uniques)

    @cache_readonly
    def indices(self) -> Dict[Any, np.ndarray]:
        if isinstance(self.grouping_vector, ops.BaseGrouper):
            return self.grouping_vector.indices
        values = Categorical(self.grouping_vector)
        return values._reverse_indexer()

    @property
    def codes(self) -> np.ndarray:
        return self._codes_and_uniques[0]

    @property
    def uniques(self) -> Union[Index, Categorical]:
        return self._codes_and_uniques[1]

    @cache_readonly
    def _codes_and_uniques(self) -> Tuple[np.ndarray, Union[Index, Categorical]]:
        if self._passed_categorical:
            cat: Categorical = self.grouping_vector  # type: ignore[assignment]
            categories = cat.categories
            if self._observed:
                ucodes = algorithms.unique1d(cat.codes)
                ucodes = ucodes[ucodes != -1]
                if self._sort:
                    ucodes = np.sort(ucodes)
            else:
                ucodes = np.arange(len(categories))
            has_dropped_na = False
            if not self._dropna:
                na_mask = cat.isna()
                if np.any(na_mask):
                    has_dropped_na = True
                    if self._sort:
                        na_code = len(categories)
                    else:
                        na_idx = na_mask.argmax()
                        na_code = algorithms.nunique_ints(cat.codes[:na_idx])
                    ucodes = np.insert(ucodes, na_code, -1)
            uniques = Categorical.from_codes(
                codes=ucodes, categories=categories, ordered=cat.ordered, validate=False
            )
            codes = cat.codes
            if has_dropped_na:
                if not self._sort:
                    codes = np.where(codes >= na_code, codes + 1, codes)
                codes = np.where(na_mask, na_code, codes)
            return (codes, uniques)
        elif isinstance(self.grouping_vector, ops.BaseGrouper):
            codes = self.grouping_vector.codes_info
            uniques = self.grouping_vector.result_index._values
        elif self._uniques is not None:
            cat = Categorical(self.grouping_vector, categories=self._uniques)
            codes = cat.codes
            uniques = self._uniques
        else:
            codes, uniques = algorithms.factorize(
                self.grouping_vector, sort=self._sort, use_na_sentinel=self._dropna
            )
        return (codes, uniques)

    @cache_readonly
    def groups(self) -> Dict[Any, Any]:
        codes, uniques = self._codes_and_uniques
        uniques = Index._with_infer(uniques, name=self.name)
        cats = Categorical.from_codes(codes, uniques, validate=False)
        return self._index.groupby(cats)

    @property
    def observed_grouping(self) -> Grouping:
        if self._observed:
            return self
        return self._observed_grouping

    @cache_readonly
    def _observed_grouping(self) -> Grouping:
        grouping = Grouping(
            self._index,
            self._orig_grouper,
            obj=self.obj,
            level=self.level,
            sort=self._sort,
            observed=True,
            in_axis=self.in_axis,
            dropna=self._dropna,
            uniques=self._uniques,
        )
        return grouping


def get_grouper(
    obj: Union[DataFrame, Series],
    key: Optional[Union[Any, List[Any]]] = None,
    level: Optional[Union[str, int, List[Union[str, int]]]] = None,
    sort: bool = True,
    observed: bool = False,
    validate: bool = True,
    dropna: bool = True,
) -> Tuple[ops.BaseGrouper, frozenset, Union[DataFrame, Series]]:
    """
    Create and return a BaseGrouper, which is an internal
    mapping of how to create the grouper indexers.
    This may be composed of multiple Grouping objects, indicating
    multiple groupers

    Groupers are ultimately index mappings. They can originate as:
    index mappings, keys to columns, functions, or Groupers

    Groupers enable local references to level,sort, while
    the passed in level, and sort are 'global'.

    This routine tries to figure out what the passing in references
    are and then creates a Grouping for each one, combined into
    a BaseGrouper.

    If observed & we have a categorical grouper, only show the observed
    values.

    If validate, then check for key/level overlaps.

    """
    group_axis: Index = obj.index
    if level is not None:
        if isinstance(group_axis, MultiIndex):
            if is_list_like(level) and len(level) == 1:
                level = level[0]
            if key is None and is_scalar(level):
                key = group_axis.get_level_values(level)
                level = None
        else:
            if is_list_like(level):
                nlevels = len(level)
                if nlevels == 1:
                    level = level[0]
                elif nlevels == 0:
                    raise ValueError('No group keys passed!')
                else:
                    raise ValueError('multiple levels only valid with MultiIndex')
            if isinstance(level, str):
                if obj.index.name != level:
                    raise ValueError(f'level name {level} is not the name of the index')
            elif level > 0 or level < -1:
                raise ValueError('level > 0 or level < -1 only valid with MultiIndex')
            level = None
            key = group_axis
    if isinstance(key, Grouper):
        grouper, obj = key._get_grouper(obj, validate=False)
        if key.key is None:
            return (grouper, frozenset(), obj)
        else:
            return (grouper, frozenset({key.key}), obj)
    elif isinstance(key, ops.BaseGrouper):
        return (key, frozenset(), obj)
    if not isinstance(key, list):
        keys: List[Any] = [key]
        match_axis_length: bool = False
    else:
        keys = key
        match_axis_length = len(keys) == len(group_axis)
    any_callable: bool = any(
        (callable(g) or isinstance(g, dict) for g in keys)
    )
    any_groupers: bool = any(
        (isinstance(g, (Grouper, Grouping)) for g in keys)
    )
    any_arraylike: bool = any(
        (isinstance(g, (list, tuple, Series, Index, np.ndarray)) for g in keys)
    )
    if not any_callable and (not any_arraylike) and (not any_groupers) and match_axis_length and (level is None):
        if isinstance(obj, DataFrame):
            all_in_columns_index: bool = all(
                (g in obj.columns or g in obj.index.names for g in keys)
            )
        else:
            assert isinstance(obj, Series)
            all_in_columns_index = all(
                (g in obj.index.names for g in keys)
            )
        if not all_in_columns_index:
            keys = [com.asarray_tuplesafe(keys)]
    if isinstance(level, (tuple, list)):
        if key is None:
            keys = [None] * len(level)
        levels: Iterable[Union[str, int]] = level
    else:
        levels = [level] * len(keys)
    groupings: List[Grouping] = []
    exclusions: Set[Any] = set()

    def is_in_axis(key_val: Any) -> bool:
        if not _is_label_like(key_val):
            if obj.ndim == 1:
                return False
            items = obj.axes[-1]
            try:
                items.get_loc(key_val)
            except (KeyError, TypeError, InvalidIndexError):
                return False
        return True

    def is_in_obj(gpr: Any) -> bool:
        if not hasattr(gpr, 'name'):
            return False
        try:
            obj_gpr_column = obj[gpr.name]
        except (KeyError, IndexError, InvalidIndexError, OutOfBoundsDatetime):
            return False
        if isinstance(gpr, Series) and isinstance(obj_gpr_column, Series):
            return gpr._mgr.references_same_values(obj_gpr_column._mgr, 0)
        return False

    for gpr, lvl in zip(keys, levels):
        if is_in_obj(gpr):
            in_axis = True
            exclusions.add(gpr.name)
        elif is_in_axis(gpr):
            if obj.ndim != 1 and gpr in obj:
                if validate:
                    obj._check_label_or_level_ambiguity(gpr, axis=0)
                in_axis, name, gpr_val = True, gpr, obj[gpr]
                if gpr_val.ndim != 1:
                    raise ValueError(f"Grouper for '{name}' not 1-dimensional")
                exclusions.add(name)
            elif obj._is_level_reference(gpr, axis=0):
                in_axis, lvl, gpr = False, gpr, None
            else:
                raise KeyError(gpr)
        elif isinstance(gpr, Grouper) and gpr.key is not None:
            exclusions.add(gpr.key)
            in_axis = True
        else:
            in_axis = False
        ping: Grouping
        if not isinstance(gpr, Grouping):
            ping = Grouping(
                group_axis, gpr, obj=obj, level=lvl, sort=sort, observed=observed, in_axis=in_axis, dropna=dropna
            )
        else:
            ping = gpr
        groupings.append(ping)
    if len(groupings) == 0 and len(obj):
        raise ValueError('No group keys passed!')
    if len(groupings) == 0:
        groupings.append(Grouping(default_index(0), np.array([], dtype=np.intp)))
    grouper = ops.BaseGrouper(
        group_axis, groupings, sort=sort, dropna=dropna
    )
    return (grouper, frozenset(exclusions), obj)


def _is_label_like(val: Any) -> bool:
    return isinstance(val, (str, tuple)) or (val is not None and is_scalar(val))


def _convert_grouper(
    axis: Index, grouper: Any
) -> Union[Any, Callable[[Any], Any], np.ndarray, Series, Index, ExtensionArray]:
    if isinstance(grouper, dict):
        return grouper.get
    elif isinstance(grouper, Series):
        if grouper.index.equals(axis):
            return grouper._values
        else:
            return grouper.reindex(axis)._values
    elif isinstance(grouper, MultiIndex):
        return grouper._values
    elif isinstance(grouper, (list, tuple, Index, Categorical, np.ndarray)):
        if len(grouper) != len(axis):
            raise ValueError('Grouper and axis must be same length')
        if isinstance(grouper, (list, tuple)):
            grouper = com.asarray_tuplesafe(grouper)
        return grouper
    else:
        return grouper
