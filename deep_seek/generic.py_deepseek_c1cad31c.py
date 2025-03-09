```python
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

"""
A base class of DataFrame/Column to behave similar to pandas DataFrame/Series.
"""
from abc import ABCMeta, abstractmethod
from collections import Counter
from collections.abc import Iterable
from distutils.version import LooseVersion
from functools import reduce
from typing import Any, List, Optional, Tuple, Union, TYPE_CHECKING, cast, Callable, Dict, NoReturn
import warnings

import numpy as np  # noqa: F401
import pandas as pd
from pandas.api.types import is_list_like

import pyspark
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    FloatType,
    IntegralType,
    LongType,
    NumericType,
)

from databricks import koalas as ks  # For running doctests and reference resolution in PyCharm.
from databricks.koalas.indexing import AtIndexer, iAtIndexer, iLocIndexer, LocIndexer
from databricks.koalas.internal import InternalFrame
from databricks.koalas.spark import functions as SF
from databricks.koalas.typedef import Scalar, spark_type_to_pandas_dtype
from databricks.koalas.utils import (
    is_name_like_tuple,
    is_name_like_value,
    name_like_string,
    scol_for,
    sql_conf,
    validate_arguments_and_invoke_function,
    validate_axis,
    SPARK_CONF_ARROW_ENABLED,
)
from databricks.koalas.window import Rolling, Expanding

if TYPE_CHECKING:
    from databricks.koalas.frame import DataFrame
    from databricks.koalas.groupby import DataFrameGroupBy, SeriesGroupBy
    from databricks.koalas.series import Series
    from databricks.koalas.indexes import Index


class Frame(object, metaclass=ABCMeta):
    """
    The base class for both DataFrame and Series.
    """

    @abstractmethod
    def __getitem__(self, key: Any) -> Any:
        pass

    @property
    @abstractmethod
    def _internal(self) -> InternalFrame:
        pass

    @abstractmethod
    def _apply_series_op(self, op: Callable[..., Any], should_resolve: bool = False) -> Any:
        pass

    @abstractmethod
    def _reduce_for_stat_function(
        self,
        sfun: Callable[..., Any],
        name: str,
        axis: Optional[Union[int, str]] = None,
        numeric_only: bool = True,
        **kwargs: Any
    ) -> Union["Series", Scalar]:
        pass

    @property
    @abstractmethod
    def dtypes(self) -> Union[pd.Series, Any]:
        pass

    @abstractmethod
    def to_pandas(self) -> pd.DataFrame:
        pass

    @property
    @abstractmethod
    def index(self) -> "Index":
        pass

    @abstractmethod
    def copy(self) -> "Frame":
        pass

    @abstractmethod
    def _to_internal_pandas(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def head(self, n: int = 5) -> "Frame":
        pass

    def cummin(self, skipna: bool = True) -> Union["Series", "DataFrame"]:
        # Method implementation remains unchanged
        pass

    def cummax(self, skipna: bool = True) -> Union["Series", "DataFrame"]:
        # Method implementation remains unchanged
        pass

    def cumsum(self, skipna: bool = True) -> Union["Series", "DataFrame"]:
        # Method implementation remains unchanged
        pass

    def cumprod(self, skipna: bool = True) -> Union["Series", "DataFrame"]:
        # Method implementation remains unchanged
        pass

    def get_dtype_counts(self) -> pd.Series:
        # Method implementation remains unchanged
        pass

    def pipe(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        # Method implementation remains unchanged
        pass

    def to_csv(
        self,
        path: Optional[str] = None,
        sep: str = ",",
        na_rep: str = "",
        columns: Optional[List[str]] = None,
        header: Union[bool, List[str]] = True,
        quotechar: str = '"',
        date_format: Optional[str] = None,
        escapechar: Optional[str] = None,
        num_files: Optional[int] = None,
        mode: str = "overwrite",
        partition_cols: Optional[Union[str, List[str]]] = None,
        index_col: Optional[Union[str, List[str]]] = None,
        **options: Any
    ) -> Optional[str]:
        # Method implementation remains unchanged
        pass

    def to_json(
        self,
        path: Optional[str] = None,
        compression: str = "uncompressed",
        num_files: Optional[int] = None,
        mode: str = "overwrite",
        orient: str = "records",
        lines: bool = True,
        partition_cols: Optional[Union[str, List[str]]] = None,
        index_col: Optional[Union[str, List[str]]] = None,
        **options: Any
    ) -> Optional[str]:
        # Method implementation remains unchanged
        pass

    def to_excel(
        self,
        excel_writer: Any,
        sheet_name: str = "Sheet1",
        na_rep: str = "",
        float_format: Optional[str] = None,
        columns: Optional[List[str]] = None,
        header: Union[bool, List[str]] = True,
        index: bool = True,
        index_label: Optional[Union[str, List[str]]] = None,
        startrow: int = 0,
        startcol: int = 0,
        engine: Optional[str] = None,
        merge_cells: bool = True,
        encoding: Optional[str] = None,
        inf_rep: str = "inf",
        verbose: bool = True,
        freeze_panes: Optional[Tuple[int, int]] = None,
    ) -> None:
        # Method implementation remains unchanged
        pass

    def mean(
        self, axis: Optional[Union[int, str]] = None, numeric_only: Optional[bool] = None
    ) -> Union[Scalar, "Series"]:
        # Method implementation remains unchanged
        pass

    def sum(
        self,
        axis: Optional[Union[int, str]] = None,
        numeric_only: Optional[bool] = None,
        min_count: int = 0,
    ) -> Union[Scalar, "Series"]:
        # Method implementation remains unchanged
        pass

    def product(
        self,
        axis: Optional[Union[int, str]] = None,
        numeric_only: Optional[bool] = None,
        min_count: int = 0,
    ) -> Union[Scalar, "Series"]:
        # Method implementation remains unchanged
        pass

    def skew(
        self, axis: Optional[Union[int, str]] = None, numeric_only: Optional[bool] = None
    ) -> Union[Scalar, "Series"]:
        # Method implementation remains unchanged
        pass

    def kurtosis(
        self, axis: Optional[Union[int, str]] = None, numeric_only: Optional[bool] = None
    ) -> Union[Scalar, "Series"]:
        # Method implementation remains unchanged
        pass

    def min(
        self, axis: Optional[Union[int, str]] = None, numeric_only: Optional[bool] = None
    ) -> Union[Scalar, "Series"]:
        # Method implementation remains unchanged
        pass

    def max(
        self, axis: Optional[Union[int, str]] = None, numeric_only: Optional[bool] = None
    ) -> Union[Scalar, "Series"]:
        # Method implementation remains unchanged
        pass

    def count(
        self, axis: Optional[Union[int, str]] = None, numeric_only: bool = False
    ) -> Union[Scalar, "Series"]:
        # Method implementation remains unchanged
        pass

    def std(
        self,
        axis: Optional[Union[int, str]] = None,
        ddof: int = 1,
        numeric_only: Optional[bool] = None,
    ) -> Union[Scalar, "Series"]:
        # Method implementation remains unchanged
        pass

    def var(
        self,
        axis: Optional[Union[int, str]] = None,
        ddof: int = 1,
        numeric_only: Optional[bool] = None,
    ) -> Union[Scalar, "Series"]:
        # Method implementation remains unchanged
        pass

    def median(
        self,
        axis: Optional[Union[int, str]] = None,
        numeric_only: Optional[bool] = None,
        accuracy: int = 10000,
    ) -> Union[Scalar, "Series"]:
        # Method implementation remains unchanged
        pass

    def sem(
        self,
        axis: Optional[Union[int, str]] = None,
        ddof: int = 1,
        numeric_only: Optional[bool] = None,
    ) -> Union[Scalar, "Series"]:
        # Method implementation remains unchanged
        pass

    @property
    def size(self) -> int:
        # Method implementation remains unchanged
        pass

    def abs(self) -> Union["DataFrame", "Series"]:
        # Method implementation remains unchanged
        pass

    def groupby(
        self,
        by: Union[Any, List[Any]],
        axis: Union[int, str] = 0,
        as_index: bool = True,
        dropna: bool = True,
    ) -> Union["DataFrameGroupBy", "SeriesGroupBy"]:
        # Method implementation remains unchanged
        pass

    def bool(self) -> bool:
        # Method implementation remains unchanged
        pass

    def first_valid_index(self) -> Optional[Union[Scalar, Tuple[Scalar, ...]]]:
        # Method implementation remains unchanged
        pass

    def last_valid_index(self) -> Optional[Union[Scalar, Tuple[Scalar, ...]]]:
        # Method implementation remains unchanged
        pass

    def rolling(self, window: int, min_periods: Optional[int] = None) -> Rolling:
        # Method implementation remains unchanged
        pass

    def expanding(self, min_periods: int = 1) -> Expanding:
        # Method implementation remains unchanged
        pass

    def get(self, key: Any, default: Optional[Any] = None) -> Any:
        # Method implementation remains unchanged
        pass

    def squeeze(self, axis: Optional[Union[int, str]] = None) -> Union[Scalar, "DataFrame", "Series"]:
        # Method implementation remains unchanged
        pass

    def truncate(
        self,
        before: Optional[Any] = None,
        after: Optional[Any] = None,
        axis: Optional[Union[int, str]] = None,
        copy: bool = True,
    ) -> Union["DataFrame", "Series"]:
        # Method implementation remains unchanged
        pass

    def to_markdown(self, buf: Optional[Any] = None, mode: Optional[str] = None) -> str:
        # Method implementation remains unchanged
        pass

    @abstractmethod
    def fillna(
        self,
        value: Optional[Any] = None,
        method: Optional[str] = None,
        axis: Optional[Union[int, str]] = None,
        inplace: bool = False,
        limit: Optional[int] = None,
    ) -> Union["DataFrame", "Series"]:
        pass

    def bfill(
        self,
        axis: Optional[Union[int, str]] = None,
        inplace: bool = False,
        limit: Optional[int] = None,
    ) -> Union["DataFrame", "Series"]:
        # Method implementation remains unchanged
        pass

    def ffill(
        self,
        axis: Optional[Union[int, str]] = None,
        inplace: bool = False,
        limit: Optional[int] = None,
    ) -> Union["DataFrame", "Series"]:
        # Method implementation remains unchanged
        pass

    @property
    def at(self) -> AtIndexer:
        # Method implementation remains unchanged
        pass

    @property
    def iat(self) -> iAtIndexer:
        # Method implementation remains unchanged
        pass

    @property
    def iloc(self) -> iLocIndexer:
        # Method implementation remains unchanged
        pass

    @property
    def loc(self) -> LocIndexer:
        # Method implementation remains unchanged
        pass

    def __bool__(self) -> NoReturn:
        # Method implementation remains unchanged
        pass

    @staticmethod
    def _count_expr(spark_column: Any, spark_type: Any) -> Any:
        # Method implementation remains unchanged
        pass
```