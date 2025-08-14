from typing import Any, Optional, Union, List, Tuple, Sized, cast, Dict, Iterable, OrderedDict
from collections import OrderedDict
from collections.abc import Iterable
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
    ...

def range(
    start: int, 
    end: Optional[int] = None, 
    step: int = 1, 
    num_partitions: Optional[int] = None
) -> DataFrame:
    ...

def read_csv(
    path: str,
    sep: str = ",",
    header: Union[int, List[int], str] = "infer",
    names: Optional[Union[str, List[str]]] = None,
    index_col: Optional[Union[str, List[str]]] = None,
    usecols: Optional[Union[List[Union[int, str]], Callable]] = None,
    squeeze: bool = False,
    mangle_dupe_cols: bool = True,
    dtype: Optional[Union[str, Dict[str, Any]]] = None,
    nrows: Optional[int] = None,
    parse_dates: bool = False,
    quotechar: Optional[str] = None,
    escapechar: Optional[str] = None,
    comment: Optional[str] = None,
    **options: Any
) -> Union[DataFrame, Series]:
    ...

def read_json(
    path: str, 
    lines: bool = True, 
    index_col: Optional[Union[str, List[str]]] = None, 
    **options: Any
) -> DataFrame:
    ...

def read_delta(
    path: str,
    version: Optional[str] = None,
    timestamp: Optional[str] = None,
    index_col: Optional[Union[str, List[str]]] = None,
    **options: Any
) -> DataFrame:
    ...

def read_table(name: str, index_col: Optional[Union[str, List[str]]] = None) -> DataFrame:
    ...

def read_spark_io(
    path: Optional[str] = None,
    format: Optional[str] = None,
    schema: Union[str, StructType] = None,
    index_col: Optional[Union[str, List[str]]] = None,
    **options: Any
) -> DataFrame:
    ...

def read_parquet(
    path: str,
    columns: Optional[List[str]] = None,
    index_col: Optional[Union[str, List[str]]] = None,
    pandas_metadata: bool = False,
    **options: Any
) -> DataFrame:
    ...

def read_clipboard(sep: str = r"\s+", **kwargs: Any) -> DataFrame:
    ...

def read_excel(
    io: Union[str, bytes, ExcelFile, xlrd.Book, BinaryIO],
    sheet_name: Union[str, int, List[Union[str, int]], None] = 0,
    header: Union[int, List[int], None] = 0,
    names: Optional[List[str]] = None,
    index_col: Union[int, List[int], None] = None,
    usecols: Union[int, str, List[Union[int, str]], None] = None,
    squeeze: bool = False,
    dtype: Optional[Dict[str, Any]] = None,
    engine: Optional[str] = None,
    converters: Optional[Dict[str, Any]] = None,
    true_values: Optional[List[Any]] = None,
    false_values: Optional[List[Any]] = None,
    skiprows: Optional[Union[int, List[int], slice]] = None,
    nrows: Optional[int] = None,
    na_values: Optional[Union[Any, List[Any], Dict[str, Any]]] = None,
    keep_default_na: bool = True,
    verbose: bool = False,
    parse_dates: Union[bool, List[Union[int, str]], List[List[Union[int, str]]], Dict[str, List[Union[int, str]]]] = False,
    date_parser: Optional[Callable] = None,
    thousands: Optional[str] = None,
    comment: Optional[str] = None,
    skipfooter: int = 0,
    convert_float: bool = True,
    mangle_dupe_cols: bool = True,
    **kwds: Any
) -> Union[DataFrame, Series, OrderedDict]:
    ...

def read_html(
    io: Union[str, BinaryIO],
    match: Union[str, Pattern] = ".+",
    flavor: Optional[str] = None,
    header: Union[int, List[int], None] = None,
    index_col: Union[int, List[int], None] = None,
    skiprows: Union[int, List[int], slice, None] = None,
    attrs: Optional[Dict[str, str]] = None,
    parse_dates: bool = False,
    thousands: str = ",",
    encoding: Optional[str] = None,
    decimal: str = ".",
    converters: Optional[Dict[str, Any]] = None,
    na_values: Optional[Union[Any, List[Any]]] = None,
    keep_default_na: bool = True,
    displayed_only: bool = True,
) -> List[DataFrame]:
    ...

def read_sql_table(
    table_name: str,
    con: str,
    schema: Optional[str] = None,
    index_col: Optional[Union[str, List[str]]] = None,
    columns: Optional[List[str]] = None,
    **options: Any
) -> DataFrame:
    ...

def read_sql_query(
    sql: str,
    con: str,
    index_col: Optional[Union[str, List[str]]] = None,
    **options: Any
) -> DataFrame:
    ...

def read_sql(
    sql: str,
    con: str,
    index_col: Optional[Union[str, List[str]]] = None,
    columns: Optional[List[str]] = None,
    **options: Any
) -> DataFrame:
    ...

def to_datetime(
    arg: Union[DataFrame, Series, List, np.ndarray, pd.Series, pd.DataFrame, str, int, float],
    errors: str = "raise",
    format: Optional[str] = None,
    unit: Optional[str] = None,
    infer_datetime_format: bool = False,
    origin: str = "unix",
) -> Union[DatetimeIndex, Series, pd.Timestamp, pd.Series, pd.DatetimeIndex]:
    ...

def date_range(
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    periods: Optional[int] = None,
    freq: Optional[str] = None,
    tz: Optional[str] = None,
    normalize: bool = False,
    name: Optional[str] = None,
    closed: Optional[str] = None,
    **kwargs: Any
) -> DatetimeIndex:
    ...

def get_dummies(
    data: Union[DataFrame, Series],
    prefix: Optional[Union[str, List[str], Dict[str, str]]] = None,
    prefix_sep: str = "_",
    dummy_na: bool = False,
    columns: Optional[List[str]] = None,
    sparse: bool = False,
    drop_first: bool = False,
    dtype: type = np.uint8,
) -> DataFrame:
    ...

def concat(
    objs: Union[Iterable[Union[DataFrame, Series]], DataFrame, Series],
    axis: int = 0,
    join: str = "outer",
    ignore_index: bool = False,
    sort: bool = False,
) -> Union[DataFrame, Series]:
    ...

def melt(
    frame: DataFrame,
    id_vars: Optional[Union[str, List[str]]] = None,
    value_vars: Optional[Union[str, List[str]]] = None,
    var_name: Optional[str] = None,
    value_name: str = "value",
) -> DataFrame:
    ...

def isna(obj: Any) -> Union[bool, Series, DataFrame]:
    ...

isnull = isna

def notna(obj: Any) -> Union[bool, Series, DataFrame]:
    ...

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
    ...

def to_numeric(arg: Union[Series, List, Tuple, np.ndarray, str, int, float]) -> Union[Series, np.ndarray, float]:
    ...

def broadcast(obj: DataFrame) -> DataFrame:
    ...

def read_orc(
    path: str,
    columns: Optional[List[str]] = None,
    index_col: Optional[Union[str, List[str]]] = None,
    **options: Any
) -> DataFrame:
    ...

def _get_index_map(
    sdf: spark.DataFrame, 
    index_col: Optional[Union[str, List[str]]] = None
) -> Tuple[Optional[List[spark.Column]], Optional[List[Tuple]]]:
    ...
