import functools
from collections import OrderedDict
from contextlib import contextmanager
from distutils.version import LooseVersion
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings
import pyarrow
import pyspark
from pyspark import sql as spark
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
import pandas as pd
from pandas.api.types import is_list_like
from databricks import koalas as ks
from databricks.koalas.typedef.typehints import as_spark_type, extension_dtypes, spark_type_to_pandas_dtype

def same_anchor(this: Union['InternalFrame', 'DataFrame', 'IndexOpsMixin'], that: Union['InternalFrame', 'DataFrame', 'IndexOpsMixin']) -> bool:
    ...

def combine_frames(this: 'DataFrame', *args: Union['Series', 'DataFrame'], how: str='full', preserve_order_column: bool=False) -> 'DataFrame':
    ...

def align_diff_frames(resolve_func: Callable, this: 'DataFrame', that: 'DataFrame', fillna: bool=True, how: str='full', preserve_order_column: bool=False) -> 'DataFrame':
    ...

def is_testing() -> bool:
    ...

def default_session(conf: Optional[Dict[str, Any]]=None) -> spark.SparkSession:
    ...

@contextmanager
def sql_conf(pairs: Dict[str, Any], *, spark: Optional[spark.SparkSession]=None):
    ...

def validate_arguments_and_invoke_function(pobj: Any, koalas_func: Callable, pandas_func: Callable, input_args: Dict[str, Any]) -> Any:
    ...

def lazy_property(fn: Callable) -> Callable:
    ...

def scol_for(sdf: spark.DataFrame, column_name: str) -> Any:
    ...

def column_labels_level(column_labels: List[Tuple[str]]) -> int:
    ...

def name_like_string(name: Union[str, Tuple[str]]) -> str:
    ...

def is_name_like_tuple(value: Union[Tuple, None], allow_none: bool=True, check_type: bool=False) -> bool:
    ...

def is_name_like_value(value: Union[str, int, float, Tuple, None], allow_none: bool=True, allow_tuple: bool=True, check_type: bool=False) -> bool:
    ...

def validate_axis(axis: Union[int, str]=0, none_axis: int=0) -> int:
    ...

def validate_bool_kwarg(value: Union[bool, None], arg_name: str) -> bool:
    ...

def validate_how(how: str) -> str:
    ...

def verify_temp_column_name(df: Union['DataFrame', spark.DataFrame], column_name_or_label: Union[str, Tuple[str]]) -> Union[str, Tuple[str]]:
    ...

def compare_null_first(left: Any, right: Any, comp: Callable) -> Any:
    ...

def compare_null_last(left: Any, right: Any, comp: Callable) -> Any:
    ...

def compare_disallow_null(left: Any, right: Any, comp: Callable) -> Any:
    ...

def compare_allow_null(left: Any, right: Any, comp: Callable) -> Any:
    ...
