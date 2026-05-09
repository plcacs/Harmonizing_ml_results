from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import functools
from collections import OrderedDict
from contextlib import contextmanager
from distutils.version import LooseVersion
import os
import pandas as pd
from databricks import koalas as ks
from databricks.koalas.typedef.typehints import as_spark_type, extension_dtypes, spark_type_to_pandas_dtype
from databricks.koalas.frame import DataFrame, InternalFrame
from databricks.koalas.internal import HIDDEN_COLUMNS, NATURAL_ORDER_COLUMN_NAME, SPARK_INDEX_NAME_FORMAT

def same_anchor(this: DataFrame, that: DataFrame) -> bool:
    # ...

def combine_frames(this: DataFrame, *args: DataFrame, how: str = 'full', preserve_order_column: bool = False) -> DataFrame:
    # ...

def align_diff_frames(resolve_func: Callable[[DataFrame, List[Any], List[Any]], List[Tuple[Series, str]]], 
                       this: DataFrame, 
                       that: DataFrame, 
                       fillna: bool = True, 
                       how: str = 'full', 
                       preserve_order_column: bool = False) -> DataFrame:
    # ...

def default_session(conf: Optional[Dict[str, str]] = None) -> spark.Session:
    # ...

@contextmanager
def sql_conf(pairs: Dict[str, str], spark: Optional[spark.Session] = None) -> None:
    # ...

def scol_for(sdf: DataFrame, column_name: str) -> spark.Column:
    # ...

def column_labels_level(column_labels: List[Any]) -> int:
    # ...

def name_like_string(name: Any) -> str:
    # ...

def is_name_like_tuple(value: Any, allow_none: bool = True, check_type: bool = False) -> bool:
    # ...

def is_name_like_value(value: Any, allow_none: bool = True, allow_tuple: bool = True, check_type: bool = False) -> bool:
    # ...

def validate_axis(axis: Any, none_axis: int = 0) -> int:
    # ...

def validate_bool_kwarg(value: bool, arg_name: str) -> bool:
    # ...

def validate_how(how: str) -> str:
    # ...

def verify_temp_column_name(df: DataFrame, column_name_or_label: Union[str, Tuple[str]]) -> Union[str, Tuple[str]]:
    # ...
