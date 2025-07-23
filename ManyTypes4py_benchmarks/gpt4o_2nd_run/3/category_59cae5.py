from functools import partial
from typing import Any, Optional, Union
import pandas as pd
from pandas.api.types import is_hashable
from databricks import koalas as ks
from databricks.koalas.indexes.base import Index
from databricks.koalas.missing.indexes import MissingPandasLikeCategoricalIndex
from databricks.koalas.series import Series

class CategoricalIndex(Index):
    def __new__(cls, data: Optional[Union[Series, Index, Any]] = None, 
                categories: Optional[Any] = None, 
                ordered: Optional[bool] = None, 
                dtype: Optional[Union[str, Any]] = None, 
                copy: bool = False, 
                name: Optional[Any] = None) -> 'CategoricalIndex':
        if not is_hashable(name):
            raise TypeError('Index.name must be a hashable type')
        if isinstance(data, (Series, Index)):
            if dtype is None:
                dtype = 'category'
            return Index(data, dtype=dtype, copy=copy, name=name)
        return ks.from_pandas(pd.CategoricalIndex(data=data, categories=categories, ordered=ordered, dtype=dtype, name=name))

    @property
    def codes(self) -> Index:
        return self._with_new_scol(self.spark.column).rename(None)

    @property
    def categories(self) -> Index:
        return self.dtype.categories

    @categories.setter
    def categories(self, categories: Any) -> None:
        raise NotImplementedError()

    @property
    def ordered(self) -> bool:
        return self.dtype.ordered

    def __getattr__(self, item: str) -> Any:
        if hasattr(MissingPandasLikeCategoricalIndex, item):
            property_or_func = getattr(MissingPandasLikeCategoricalIndex, item)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)
            else:
                return partial(property_or_func, self)
        raise AttributeError("'CategoricalIndex' object has no attribute '{}'".format(item))
