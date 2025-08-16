from __future__ import annotations
import re
from typing import TYPE_CHECKING, List, Union, Tuple
import numpy as np
from pandas.core.dtypes.common import is_iterator, is_list_like
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.missing import notna
import pandas.core.algorithms as algos
from pandas.core.indexes.api import MultiIndex
from pandas.core.reshape.concat import concat
from pandas.core.tools.numeric import to_numeric
if TYPE_CHECKING:
    from collections.abc import Hashable
    from pandas._typing import AnyArrayLike
    from pandas import DataFrame

def ensure_list_vars(arg_vars: Union[None, AnyArrayLike], variable: str, columns: MultiIndex) -> List:
    if arg_vars is not None:
        if not is_list_like(arg_vars):
            return [arg_vars]
        elif isinstance(columns, MultiIndex) and (not isinstance(arg_vars, list)):
            raise ValueError(f'{variable} must be a list of tuples when columns are a MultiIndex')
        else:
            return list(arg_vars)
    else:
        return []

def melt(frame: DataFrame, id_vars: Union[None, Hashable, Tuple[Hashable, ...]], value_vars: Union[None, Hashable, Tuple[Hashable, ...]], var_name: Union[None, Hashable, Tuple[Hashable, ...]], value_name: str = 'value', col_level: Union[None, int] = None, ignore_index: bool = True) -> DataFrame:
    ...

def lreshape(data: DataFrame, groups: dict, dropna: bool = True) -> DataFrame:
    ...

def wide_to_long(df: DataFrame, stubnames: Union[str, List[str]], i: Union[str, List[str]], j: str, sep: str = '', suffix: str = '\\d+') -> DataFrame:
    ...
