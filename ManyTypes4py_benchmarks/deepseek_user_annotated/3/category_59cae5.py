#
# Copyright (C) 2019 Databricks, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from functools import partial
from typing import Any, Optional, Union

import pandas as pd
from pandas.api.types import is_hashable
from pandas.core.dtypes.base import ExtensionDtype

from databricks import koalas as ks
from databricks.koalas.indexes.base import Index
from databricks.koalas.missing.indexes import MissingPandasLikeCategoricalIndex
from databricks.koalas.series import Series


class CategoricalIndex(Index):
    def __new__(
        cls,
        data: Optional[Union[pd.Categorical, pd.Index, Series, Index, list, tuple]] = None,
        categories: Optional[pd.Index] = None,
        ordered: Optional[bool] = None,
        dtype: Optional[Union[ExtensionDtype, str]] = None,
        copy: bool = False,
        name: Optional[Any] = None
    ) -> 'CategoricalIndex':
        if not is_hashable(name):
            raise TypeError("Index.name must be a hashable type")

        if isinstance(data, (Series, Index)):
            if dtype is None:
                dtype = "category"
            return Index(data, dtype=dtype, copy=copy, name=name)

        return ks.from_pandas(
            pd.CategoricalIndex(
                data=data, categories=categories, ordered=ordered, dtype=dtype, name=name
            )
        )

    @property
    def codes(self) -> Index:
        return self._with_new_scol(self.spark.column).rename(None)

    @property
    def categories(self) -> pd.Index:
        return self.dtype.categories

    @categories.setter
    def categories(self, categories: pd.Index) -> None:
        raise NotImplementedError()

    @property
    def ordered(self) -> bool:
        return self.dtype.ordered

    def __getattr__(self, item: str) -> Any:
        if hasattr(MissingPandasLikeCategoricalIndex, item):
            property_or_func = getattr(MissingPandasLikeCategoricalIndex, item)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)  # type: ignore
            else:
                return partial(property_or_func, self)
        raise AttributeError("'CategoricalIndex' object has no attribute '{}'".format(item))
