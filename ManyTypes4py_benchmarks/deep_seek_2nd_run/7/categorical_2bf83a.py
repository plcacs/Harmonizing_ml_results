from typing import TYPE_CHECKING, Any, List, Optional, Union, cast
import pandas as pd
from pandas.api.types import CategoricalDtype
from pandas.core.indexes.base import Index
if TYPE_CHECKING:
    import databricks.koalas as ks
    from databricks.koalas import Series

class CategoricalAccessor(object):
    def __init__(self, series: "Series") -> None:
        if not isinstance(series.dtype, CategoricalDtype):
            raise ValueError('Cannot call CategoricalAccessor on type {}'.format(series.dtype))
        self._data = series

    @property
    def categories(self) -> Index:
        return self._data.dtype.categories

    @categories.setter
    def categories(self, categories: Any) -> None:
        raise NotImplementedError()

    @property
    def ordered(self) -> bool:
        return self._data.dtype.ordered

    @property
    def codes(self) -> "Series":
        return self._data._with_new_scol(self._data.spark.column).rename()

    def add_categories(self, new_categories: Any, inplace: bool = False) -> Optional["Series"]:
        raise NotImplementedError()

    def as_ordered(self, inplace: bool = False) -> Optional["Series"]:
        raise NotImplementedError()

    def as_unordered(self, inplace: bool = False) -> Optional["Series"]:
        raise NotImplementedError()

    def remove_categories(self, removals: Any, inplace: bool = False) -> Optional["Series"]:
        raise NotImplementedError()

    def remove_unused_categories(self) -> "Series":
        raise NotImplementedError()

    def rename_categories(self, new_categories: Any, inplace: bool = False) -> Optional["Series"]:
        raise NotImplementedError()

    def reorder_categories(
        self, 
        new_categories: Any, 
        ordered: Optional[bool] = None, 
        inplace: bool = False
    ) -> Optional["Series"]:
        raise NotImplementedError()

    def set_categories(
        self, 
        new_categories: Any, 
        ordered: Optional[bool] = None, 
        rename: bool = False, 
        inplace: bool = False
    ) -> Optional["Series"]:
        raise NotImplementedError()
