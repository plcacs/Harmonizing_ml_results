from __future__ import annotations
from typing import Any, Iterator, Optional, Tuple, Union, Set
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
if False:
    from collections.abc import Hashable, Iterator
    from pandas._typing import ArrayLike, NDFrameT, npt
    from pandas.core.generic import NDFrame  # type: ignore


class Grouper:
    _attributes = ('key', 'level', 'freq', 'sort', 'dropna')

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get('freq') is not None:
            from pandas.core.resample import TimeGrouper
            cls = TimeGrouper  # type: ignore[assignment]
        return super().__new__(cls)

    def __init__(
        self, 
        key: Optional[Any] = None, 
        level: Optional[Any] = None, 
        freq: Optional[Any] = None, 
        sort: bool = False, 
        dropna: bool = True
    ) -> None:
        self.key: Optional[Any] = key
        self.level: Optional[Any] = level
        self.freq: Optional[Any] = freq
        self.sort: bool = sort
        self.dropna: bool = dropna
        self._indexer_deprecated: Optional[np.ndarray] = None
        self.binner: Optional[Any] = None
        self._grouper: Optional[Any] = None
        self._indexer: Optional[np.ndarray] = None

    def _get_grouper(self, obj: Union[Series, DataFrame], validate: bool = True) -> Tuple[ops.BaseGrouper, Union[Series, DataFrame]]:
        obj, _, _ = self._set_grouper(obj)
        grouper, _, obj = get_grouper(
            obj, 
            [self.key], 
            level=self.level, 
            sort=self.sort, 
            validate=validate, 
            dropna=self.dropna
        )
        return (grouper, obj)

    def _set_grouper(
        self, 
        obj: Union[Series, DataFrame], 
        sort: bool = False, 
        *, 
        gpr_index: Optional[Index] = None
    ) -> Tuple[Union[Series, DataFrame], Index, Optional[np.ndarray]]:
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
        indexer: Optional[np.ndarray] = None
        if (self.sort or sort) and (not ax.is_monotonic_increasing):
            indexer = self._indexer_deprecated = ax.array.argsort(kind='mergesort', na_position='first')
            ax = ax.take(indexer)
            obj = obj.take(indexer, axis=0)
        return (obj, ax, indexer)

    def __repr__(self) -> str:
        attrs_list = (
            f'{attr_name}={getattr(self, attr_name)!r}' 
            for attr_name in self._attributes 
            if getattr(self, attr_name) is not None
        )
        attrs = ', '.join(attrs_list)
        cls_name = type(self).__name__
        return f'{cls_name}({attrs})'


class Grouping:
    _codes: Optional[Any] = None

    def __init__(
        self, 
        index: Index, 
        grouper: Optional[Any] = None, 
        obj: Optional[Union[DataFrame, Series]] = None, 
        level: Optional[Any] = None, 
        sort: bool = True, 
        observed: bool = False, 
        in_axis: bool = False, 
        dropna: bool = True, 
        uniques: Optional[Any] = None
    ) -> None:
        self.level: Optional[Any] = level
        self._orig_grouper: Optional[Any] = grouper
        grouping_vector = _convert_grouper(index, grouper)
        self._orig_cats: Optional[Any] = None
        self._index: Index = index
        self._sort: bool = sort
        self.obj: Optional[Union[DataFrame, Series]] = obj
        self._observed: bool = observed
        self.in_axis: bool = in_axis
        self._dropna: bool = dropna
        self._uniques: Optional[Any] = uniques
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
        self.grouping_vector: Any = grouping_vector

    def __repr__(self) -> str:
        return f'Grouping({self.name})'

    def __iter__(self) -> Iterator[Any]:
        return iter(self.indices)

    @cache_readonly
    def _passed_categorical(self) -> bool:
        dtype = getattr(self.grouping_vector, 'dtype', None)
        return isinstance(dtype, CategoricalDtype)

    @cache_readonly
    def name(self) -> Optional[Any]:
        ilevel = self._ilevel
        if ilevel is not None:
            return self._index.names[ilevel]
        if isinstance(self._orig_grouper, (Index, Series)):
            return self._orig_grouper.name
        elif hasattr(self.grouping_vector, 'result_index'):
            return self.grouping_vector.result_index.name
        elif isinstance(self.grouping_vector, Index):
            return self.grouping_vector.name
        return None

    @cache_readonly
    def _ilevel(self) -> Optional[int]:
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
    def indices(self) -> dict[Any, Any]:
        if isinstance(self.grouping_vector, ops.BaseGrouper):
            return self.grouping_vector.indices
        values = Categorical(self.grouping_vector)
        return values._reverse_indexer()

    @property
    def codes(self) -> Any:
        return self._codes_and_uniques[0]

    @property
    def uniques(self) -> Any:
        return self._codes_and_uniques[1]

    @cache_readonly
    def _codes_and_uniques(self) -> Tuple[np.ndarray, Any]:
        if self._passed_categorical:
            cat = self.grouping_vector
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
            uniques = Categorical.from_codes(codes=ucodes, categories=categories, ordered=cat.ordered, validate=False)
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
            codes, uniques = algorithms.factorize(self.grouping_vector, sort=self._sort, use_na_sentinel=self._dropna)
        return (codes, uniques)

    @cache_readonly
    def groups(self) -> dict[Any, Any]:
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
            uniques=self._uniques
        )
        return grouping


def get_grouper(
    obj: Union[DataFrame, Series],
    key: Optional[Any] = None,
    level: Optional[Any] = None,
    sort: bool = True,
    observed: bool = False,
    validate: bool = True,
    dropna: bool = True
) -> Tuple[ops.BaseGrouper, frozenset, Union[DataFrame, Series]]:
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
        keys = [key]
        match_axis_length = False
    else:
        keys = key
        match_axis_length = len(keys) == len(group_axis)
    any_callable = any((callable(g) or isinstance(g, dict) for g in keys))
    any_groupers = any((isinstance(g, (Grouper, Grouping)) for g in keys))
    any_arraylike = any((isinstance(g, (list, tuple, Series, Index, np.ndarray)) for g in keys))
    if not any_callable and (not any_arraylike) and (not any_groupers) and match_axis_length and (level is None):
        if isinstance(obj, DataFrame):
            all_in_columns_index = all((g in obj.columns or g in obj.index.names for g in keys))
        else:
            assert isinstance(obj, Series)
            all_in_columns_index = all((g in obj.index.names for g in keys))
        if not all_in_columns_index:
            keys = [com.asarray_tuplesafe(keys)]
    if isinstance(level, (tuple, list)):
        if key is None:
            keys = [None] * len(level)
        levels = level
    else:
        levels = [level] * len(keys)
    groupings = []
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
                in_axis, name, gpr = (True, gpr, obj[gpr])
                if gpr.ndim != 1:
                    raise ValueError(f"Grouper for '{name}' not 1-dimensional")
                exclusions.add(name)
            elif obj._is_level_reference(gpr, axis=0):
                in_axis, lvl, gpr = (False, gpr, None)
            else:
                raise KeyError(gpr)
        elif isinstance(gpr, Grouper) and gpr.key is not None:
            exclusions.add(gpr.key)
            in_axis = True
        else:
            in_axis = False
        ping = Grouping(
            group_axis,
            gpr,
            obj=obj,
            level=lvl,
            sort=sort,
            observed=observed,
            in_axis=in_axis,
            dropna=dropna
        ) if not isinstance(gpr, Grouping) else gpr
        groupings.append(ping)
    if len(groupings) == 0 and len(obj):
        raise ValueError('No group keys passed!')
    if len(groupings) == 0:
        groupings.append(Grouping(default_index(0), np.array([], dtype=np.intp)))
    grouper = ops.BaseGrouper(group_axis, groupings, sort=sort, dropna=dropna)
    return (grouper, frozenset(exclusions), obj)


def _is_label_like(val: Any) -> bool:
    return isinstance(val, (str, tuple)) or (val is not None and is_scalar(val))


def _convert_grouper(axis: Index, grouper: Any) -> Any:
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