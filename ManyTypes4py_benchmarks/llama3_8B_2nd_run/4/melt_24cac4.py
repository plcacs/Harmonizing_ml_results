from __future__ import annotations
import re
from typing import TYPE_CHECKING, List, Any, Dict
import numpy as np
from pandas.core.dtypes.common import is_iterator, is_list_like
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.missing import notna
import pandas.core.algorithms as algos
from pandas.core.indexes.api import MultiIndex
from pandas.core.reshape.concat import concat
from pandas.core.tools.numeric import to_numeric
from pandas import DataFrame

def ensure_list_vars(arg_vars: Any, variable: str, columns: List[str]) -> List[str]:
    ...

def melt(frame: DataFrame, id_vars: Any, value_vars: Any, var_name: str, value_name: str = 'value', col_level: int = None, ignore_index: bool = True) -> DataFrame:
    ...

def lreshape(data: DataFrame, groups: Dict[str, List[str]], dropna: bool = True) -> DataFrame:
    ...

def wide_to_long(df: DataFrame, stubnames: List[str], i: List[str], j: str, sep: str = '', suffix: str = '\\d+') -> DataFrame:
    if not is_list_like(stubnames):
        stubnames = [stubnames]
    else:
        stubnames = list(stubnames)
    ...

    def get_var_names(df: DataFrame, stub: str, sep: str, suffix: str) -> List[str]:
        regex = f'^{re.escape(stub)}{re.escape(sep)}{suffix}$'
        return df.columns[df.columns.str.match(regex)]

    def melt_stub(df: DataFrame, stub: str, i: List[str], j: str, value_vars: List[str], sep: str) -> DataFrame:
        ...

    _melted = []
    value_vars_flattened = []
    for stub in stubnames:
        value_var = get_var_names(df, stub, sep, suffix)
        value_vars_flattened.extend(value_var)
        _melted.append(melt_stub(df, stub, i, j, value_var, sep))
    melted = concat(_melted, axis=1)
    id_vars = df.columns.difference(value_vars_flattened)
    new = df[id_vars]
    if len(i) == 1:
        return new.set_index(i).join(melted)
    else:
        return new.merge(melted.reset_index(), on=i).set_index(i + [j])
