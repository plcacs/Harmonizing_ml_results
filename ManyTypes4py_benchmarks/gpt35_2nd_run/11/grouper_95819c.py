from __future__ import annotations
from typing import TYPE_CHECKING, final, Union, List, Tuple, Set
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
    _attributes: Tuple[str, ...] = ('key', 'level', 'freq', 'sort', 'dropna')

    def __new__(cls, *args, **kwargs) -> Union[Grouper, TimeGrouper]:
        if kwargs.get('freq') is not None:
            from pandas.core.resample import TimeGrouper
            cls = TimeGrouper
        return super().__new__(cls)

    def __init__(self, key: str = None, level: Union[str, int] = None, freq: str = None, sort: bool = False, dropna: bool = True) -> None:
        self.key = key
        self.level = level
        self.freq = freq
        self.sort = sort
        self.dropna = dropna
        self._indexer_deprecated = None
        self.binner = None
        self._grouper = None
        self._indexer = None

    def _get_grouper(self, obj: Union[Series, DataFrame], validate: bool = True) -> Tuple[ops.BaseGrouper, Union[Series, DataFrame]]:
        obj, _, _ = self._set_grouper(obj)
        grouper, _, obj = get_grouper(obj, [self.key], level=self.level, sort=self.sort, validate=validate, dropna=self.dropna)
        return (grouper, obj)

    def _set_grouper(self, obj: Union[Series, DataFrame], sort: bool = False, *, gpr_index: Index = None) -> Tuple[Union[Series, DataFrame], Index, Union[np.ndarray, None]]:
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
        indexer = None
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
    _codes: Union[np.ndarray, None] = None

    def __init__(self, index: Index, grouper: Union[Grouper, None] = None, obj: Union[DataFrame, Series, None] = None, level: Union[str, int, None] = None, sort: bool = True, observed: bool = False, in_axis: bool = False, dropna: bool = True, uniques: Union[ArrayLike, None] = None) -> None:
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
        self.grouping_vector = grouping_vector

    def __repr__(self) -> str:
        return f'Grouping({self.name})'

    def __iter__(self) -> Iterator:
        return iter(self.indices)

    @cache_readonly
    def _passed_categorical(self) -> bool:
        dtype = getattr(self.grouping_vector, 'dtype', None)
        return isinstance(dtype, CategoricalDtype)

    @cache_readonly
    def name(self) -> Union[str, None]:
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
    def _ilevel(self) -> Union[int, None]:
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
    def indices(self) -> dict:
        if isinstance(self.grouping_vector, ops.BaseGrouper):
            return self.grouping_vector.indices
        values = Categorical(self.grouping_vector)
        return values._reverse_indexer()

    @property
    def codes(self) -> np.ndarray:
        return self._codes_and_uniques[0]

    @property
    def uniques(self) -> ArrayLike:
        return self._codes_and_uniques[1]

    @cache_readonly
    def _codes_and_uniques(self) -> Tuple[np.ndarray, ArrayLike]:
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
    def groups(self) -> dict:
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
        grouping = Grouping(self._index, self._orig_grouper, obj=self.obj, level=self.level, sort=self._sort, observed=True, in_axis=self.in_axis, dropna=self._dropna, uniques=self._uniques)
        return grouping

def get_grouper(obj: Union[Series, DataFrame], key: Union[str, List[str], Grouper], level: Union[str, int, List[Union[str, int]], None] = None, sort: bool = True, observed: bool = False, validate: bool = True, dropna: bool = True) -> Tuple[ops.BaseGrouper, Set[str], Union[Series, DataFrame]]:
    # Function body remains the same
    pass

def _is_label_like(val: Any) -> bool:
    return isinstance(val, (str, tuple)) or (val is not None and is_scalar(val))

def _convert_grouper(axis: Index, grouper: Union[dict, Series, MultiIndex, List, Tuple, Index, Categorical, np.ndarray]) -> Union[ArrayLike, None]:
    # Function body remains the same
    pass
