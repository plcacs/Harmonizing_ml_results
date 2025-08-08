from __future__ import annotations
from typing import TYPE_CHECKING, final, List, Tuple, Union
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
        ...

    def _set_grouper(self, obj: Union[Series, DataFrame], sort: bool = False, *, gpr_index: Index = None) -> Tuple[Union[Series, DataFrame], Index, Union[np.ndarray, None]]:
        ...

    @final
    def __repr__(self) -> str:
        ...

@final
class Grouping:
    _codes: Union[np.ndarray, None] = None

    def __init__(self, index: Index, grouper: Union[Grouper, None] = None, obj: Union[DataFrame, Series, None] = None, level: Union[str, int, None] = None, sort: bool = True, observed: bool = False, in_axis: bool = False, dropna: bool = True, uniques: Union[ArrayLike, None] = None) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def __iter__(self) -> Iterator:
        ...

    @cache_readonly
    def _passed_categorical(self) -> bool:
        ...

    @cache_readonly
    def name(self) -> Union[str, None]:
        ...

    @cache_readonly
    def _ilevel(self) -> Union[int, None]:
        ...

    @property
    def ngroups(self) -> int:
        ...

    @cache_readonly
    def indices(self) -> dict:
        ...

    @property
    def codes(self) -> np.ndarray:
        ...

    @property
    def uniques(self) -> ArrayLike:
        ...

    @cache_readonly
    def _codes_and_uniques(self) -> Tuple[np.ndarray, ArrayLike]:
        ...

    @cache_readonly
    def groups(self) -> dict:
        ...

    @property
    def observed_grouping(self) -> Union[Grouping, Grouping]:
        ...

    @cache_readonly
    def _observed_grouping(self) -> Grouping:
        ...

def get_grouper(obj: Union[Series, DataFrame], key: Union[str, List[str], Grouper, ops.BaseGrouper], level: Union[str, int, List[Union[str, int]], None] = None, sort: bool = True, observed: bool = False, validate: bool = True, dropna: bool = True) -> Tuple[ops.BaseGrouper, frozenset, Union[Series, DataFrame]]:
    ...

def _is_label_like(val: Any) -> bool:
    ...

def _convert_grouper(axis: Index, grouper: Union[dict, Series, MultiIndex, List, Tuple, Index, Categorical, np.ndarray]) -> Union[Callable, np.ndarray]:
    ...
