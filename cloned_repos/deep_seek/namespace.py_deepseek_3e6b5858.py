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
Wrappers around spark that correspond to common pandas functions.
"""
from typing import Any, Optional, Union, List, Tuple, Sized, cast, Dict, Iterable, Set, Callable, OrderedDict as OrderedDictType, TYPE_CHECKING
from collections import OrderedDict
from collections.abc import Iterable as ABCIterable
from distutils.version import LooseVersion
from functools import reduce
from io import BytesIO
import json

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_dtype, is_datetime64tz_dtype, is_list_like
import pyarrow as pa
import pyarrow.parquet as pq
import pyspark
from pyspark import sql as spark
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import (
    ByteType,
    ShortType,
    IntegerType,
    LongType,
    FloatType,
    DoubleType,
    BooleanType,
    TimestampType,
    DecimalType,
    StringType,
    DateType,
    StructType,
    DecimalType,
)

from databricks import koalas as ks  # noqa: F401
from databricks.koalas.base import IndexOpsMixin
from databricks.koalas.utils import (
    align_diff_frames,
    default_session,
    is_name_like_tuple,
    name_like_string,
    same_anchor,
    scol_for,
    validate_axis,
)
from databricks.koalas.frame import DataFrame, _reduce_spark_multi
from databricks.koalas.internal import (
    InternalFrame,
    DEFAULT_SERIES_NAME,
    HIDDEN_COLUMNS,
)
from databricks.koalas.series import Series, first_series
from databricks.koalas.spark.utils import as_nullable_spark_type, force_decimal_precision_scale
from databricks.koalas.indexes import Index, DatetimeIndex

if TYPE_CHECKING:
    from pyspark.sql.column import Column as SparkColumn
    from pyspark.sql.dataframe import DataFrame as SparkDataFrame

__all__ = [
    "from_pandas",
    "range",
    "read_csv",
    "read_delta",
    "read_table",
    "read_spark_io",
    "read_parquet",
    "read_clipboard",
    "read_excel",
    "read_html",
    "to_datetime",
    "date_range",
    "get_dummies",
    "concat",
    "melt",
    "isna",
    "isnull",
    "notna",
    "notnull",
    "read_sql_table",
    "read_sql_query",
    "read_sql",
    "read_json",
    "merge",
    "to_numeric",
    "broadcast",
    "read_orc",
]


def from_pandas(pobj: Union[pd.DataFrame, pd.Series, pd.Index]) -> Union[Series, DataFrame, Index]:
    # ... (existing code remains the same)


def range(
    start: int, end: Optional[int] = None, step: int = 1, num_partitions: Optional[int] = None
) -> DataFrame:
    # ... (existing code remains the same)


def read_csv(
    path: str,
    sep: str = ",",
    header: Union[str, int, List[int]] = "infer",
    names: Optional[Union[str, List[str]]] = None,
    index_col: Optional[Union[str, List[str]]] = None,
    usecols: Optional[Union[List[Union[int, str]], Callable[[Any], bool]]] = None,
    squeeze: bool = False,
    mangle_dupe_cols: bool = True,
    dtype: Optional[Union[str, Dict[str, str]]] = None,
    nrows: Optional[int] = None,
    parse_dates: bool = False,
    quotechar: Optional[str] = None,
    escapechar: Optional[str] = None,
    comment: Optional[str] = None,
    **options: Any
) -> Union[DataFrame, Series]:
    # ... (existing code remains the same)


def read_json(
    path: str, lines: bool = True, index_col: Optional[Union[str, List[str]]] = None, **options: Any
) -> DataFrame:
    # ... (existing code remains the same)


def read_delta(
    path: str,
    version: Optional[str] = None,
    timestamp: Optional[str] = None,
    index_col: Optional[Union[str, List[str]]] = None,
    **options: Any
) -> DataFrame:
    # ... (existing code remains the same)


def read_table(name: str, index_col: Optional[Union[str, List[str]]] = None) -> DataFrame:
    # ... (existing code remains the same)


def read_spark_io(
    path: Optional[str] = None,
    format: Optional[str] = None,
    schema: Union[str, StructType] = None,
    index_col: Optional[Union[str, List[str]]] = None,
    **options: Any
) -> DataFrame:
    # ... (existing code remains the same)


def read_parquet(
    path: str,
    columns: Optional[List[str]] = None,
    index_col: Optional[Union[str, List[str]]] = None,
    pandas_metadata: bool = False,
    **options: Any
) -> DataFrame:
    # ... (existing code remains the same)


def read_clipboard(sep: str = r"\s+", **kwargs: Any) -> DataFrame:
    # ... (existing code remains the same)


def read_excel(
    io: Union[str, bytes, "ExcelFile", "xlrd.Book"],
    sheet_name: Union[str, int, List[Union[str, int]]] = 0,
    header: int = 0,
    names: Optional[List[str]] = None,
    index_col: Optional[Union[int, List[int]]] = None,
    usecols: Optional[Union[str, List[int], List[str], Callable[[str], bool]]] = None,
    squeeze: bool = False,
    dtype: Optional[Dict[str, Union[str, type]]] = None,
    engine: Optional[str] = None,
    converters: Optional[Dict[str, Callable]] = None,
    true_values: Optional[List] = None,
    false_values: Optional[List] = None,
    skiprows: Optional[Union[int, List[int], Callable[[int], bool]]] = None,
    nrows: Optional[int] = None,
    na_values: Optional[Union[Any, Dict[str, Any]]] = None,
    keep_default_na: bool = True,
    verbose: bool = False,
    parse_dates: Union[bool, List, Dict] = False,
    date_parser: Optional[Callable] = None,
    thousands: Optional[str] = None,
    comment: Optional[str] = None,
    skipfooter: int = 0,
    convert_float: bool = True,
    mangle_dupe_cols: bool = True,
    **kwds: Any
) -> Union[DataFrame, Series, OrderedDictType[str, DataFrame]]:
    # ... (existing code remains the same)


def read_html(
    io: Union[str, "PathLike[str]"],
    match: str = ".+",
    flavor: Optional[str] = None,
    header: Optional[int] = None,
    index_col: Optional[Union[int, List[int]]] = None,
    skiprows: Optional[Union[int, List[int], slice]] = None,
    attrs: Optional[Dict[str, str]] = None,
    parse_dates: bool = False,
    thousands: Optional[str] = ",",
    encoding: Optional[str] = None,
    decimal: str = ".",
    converters: Optional[Dict] = None,
    na_values: Optional[Iterable] = None,
    keep_default_na: bool = True,
    displayed_only: bool = True,
) -> List[DataFrame]:
    # ... (existing code remains the same)


def read_sql_table(
    table_name: str,
    con: str,
    schema: Optional[str] = None,
    index_col: Optional[Union[str, List[str]]] = None,
    columns: Optional[List[str]] = None,
    **options: Any
) -> DataFrame:
    # ... (existing code remains the same)


def read_sql_query(
    sql: str, con: str, index_col: Optional[Union[str, List[str]]] = None, **options: Any
) -> DataFrame:
    # ... (existing code remains the same)


def read_sql(
    sql: str,
    con: str,
    index_col: Optional[Union[str, List[str]]] = None,
    columns: Optional[List[str]] = None,
    **options: Any
) -> DataFrame:
    # ... (existing code remains the same)


def to_datetime(
    arg: Union[DataFrame, Series, Iterable, Any],
    errors: str = "raise",
    format: Optional[str] = None,
    unit: Optional[str] = None,
    infer_datetime_format: bool = False,
    origin: str = "unix",
) -> Union[Series, DataFrame, pd.Timestamp, Any]:
    # ... (existing code remains the same)


def date_range(
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    periods: Optional[int] = None,
    freq: Optional[Union[str, pd.DateOffset]] = None,
    tz: Optional[str] = None,
    normalize: bool = False,
    name: Optional[str] = None,
    closed: Optional[str] = None,
    **kwargs: Any
) -> DatetimeIndex:
    # ... (existing code remains the same)


def get_dummies(
    data: Union[DataFrame, Series],
    prefix: Optional[Union[str, List[str], Dict[str, str]]] = None,
    prefix_sep: str = "_",
    dummy_na: bool = False,
    columns: Optional[List[str]] = None,
    sparse: bool = False,
    drop_first: bool = False,
    dtype: Optional[type] = None,
) -> DataFrame:
    # ... (existing code remains the same)


def concat(
    objs: Union[Iterable[Union[DataFrame, Series]], axis: int = 0, join: str = "outer", ignore_index: bool = False, sort: bool = False
) -> Union[Series, DataFrame]:
    # ... (existing code remains the same)


def melt(
    frame: DataFrame,
    id_vars: Optional[List[str]] = None,
    value_vars: Optional[List[str]] = None,
    var_name: Optional[str] = None,
    value_name: str = "value",
) -> DataFrame:
    # ... (existing code remains the same)


def isna(obj: Union[DataFrame, Series, Any]) -> Union[Series, DataFrame, bool, np.ndarray]:
    # ... (existing code remains the same)


isnull = isna


def notna(obj: Union[DataFrame, Series, Any]) -> Union[Series, DataFrame, bool, np.ndarray]:
    # ... (existing code remains the same)


notnull = notna


def merge(
    obj: DataFrame,
    right: DataFrame,
    how: str = "inner",
    on: Optional[Union[str, List[str], Tuple, List[Tuple]]] = None,
    left_on: Optional[Union[str, List[str], Tuple, List[Tuple]]] = None,
    right_on: Optional[Union[str, List[str], Tuple, List[Tuple]]] = None,
    left_index: bool = False,
    right_index: bool = False,
    suffixes: Tuple[str, str] = ("_x", "_y"),
) -> DataFrame:
    # ... (existing code remains the same)


def to_numeric(arg: Union[Series, List, Tuple, np.ndarray, Any]) -> Union[Series, float, np.ndarray]:
    # ... (existing code remains the same)


def broadcast(obj: DataFrame) -> DataFrame:
    # ... (existing code remains the same)


def read_orc(
    path: str,
    columns: Optional[List[str]] = None,
    index_col: Optional[Union[str, List[str]]] = None,
    **options: Any
) -> DataFrame:
    # ... (existing code remains the same)


def _get_index_map(
    sdf: "SparkDataFrame", index_col: Optional[Union[str, List[str]]] = None
) -> Tuple[Optional[List["SparkColumn"]], Optional[List[Tuple]]]:
    # ... (existing code remains the same)


_get_dummies_default_accept_types: Tuple[type, ...] = (DecimalType, StringType, DateType)
_get_dummies_acceptable_types: Tuple[type, ...] = _get_dummies_default_accept_types + (
    ByteType,
    ShortType,
    IntegerType,
    LongType,
    FloatType,
    DoubleType,
    BooleanType,
    TimestampType,
)
```