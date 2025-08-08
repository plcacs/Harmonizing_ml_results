from __future__ import annotations
import itertools
from typing import TYPE_CHECKING, Literal, cast, List, Union, Dict, Any, Tuple
import numpy as np
from pandas._libs import lib
from pandas.core.dtypes.cast import maybe_downcast_to_dtype
from pandas.core.dtypes.common import is_list_like, is_nested_list_like, is_scalar
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
import pandas.core.common as com
from pandas.core.groupby import Grouper
from pandas.core.indexes.api import Index, MultiIndex, get_objs_combined_axis
from pandas.core.reshape.concat import concat
from pandas.core.series import Series
if TYPE_CHECKING:
    from collections.abc import Callable, Hashable
    from pandas._typing import AggFuncType, AggFuncTypeBase, AggFuncTypeDict, IndexLabel, SequenceNotStr
    from pandas import DataFrame

def pivot_table(data: DataFrame, values: Union[List[str], str] = None, index: Union[str, Grouper, np.ndarray, List[Any]] = None, columns: Union[str, Grouper, np.ndarray, List[Any]] = None, aggfunc: Union[Callable, List[Callable], Dict[str, Union[Callable, List[Callable]]]] = 'mean', fill_value: Any = None, margins: bool = False, dropna: bool = True, margins_name: str = 'All', observed: bool = True, sort: bool = True, **kwargs: Any) -> DataFrame:
    ...

def __internal_pivot_table(data: DataFrame, values: Union[List[str], str], index: Union[str, Grouper, np.ndarray, List[Any]], columns: Union[str, Grouper, np.ndarray, List[Any]], aggfunc: Union[Callable, List[Callable], Dict[str, Union[Callable, List[Callable]]], fill_value: Any, margins: bool, dropna: bool, margins_name: str, observed: bool, sort: bool, kwargs: Dict[str, Any]) -> DataFrame:
    ...

def _add_margins(table: DataFrame, data: DataFrame, values: Union[List[str], str], rows: Union[str, Grouper, np.ndarray, List[Any]], cols: Union[str, Grouper, np.ndarray, List[Any]], aggfunc: Union[Callable, List[Callable], Dict[str, Union[Callable, List[Callable]]], kwargs: Dict[str, Any], observed: bool, margins_name: str, fill_value: Any) -> DataFrame:
    ...

def _compute_grand_margin(data: DataFrame, values: Union[List[str], str], aggfunc: Union[Callable, Dict[str, Union[Callable, str]], kwargs: Dict[str, Any], margins_name: str = 'All') -> Dict[str, Any]:
    ...

def _generate_marginal_results(table: DataFrame, data: DataFrame, values: Union[List[str], str], rows: Union[str, Grouper, np.ndarray, List[Any]], cols: Union[str, Grouper, np.ndarray, List[Any]], aggfunc: Union[Callable, List[Callable], Dict[str, Union[Callable, List[Callable]]], kwargs: Dict[str, Any], observed: bool, margins_name: str) -> Tuple[DataFrame, List[str], Series]:
    ...

def _generate_marginal_results_without_values(table: DataFrame, data: DataFrame, rows: Union[str, Grouper, np.ndarray, List[Any]], cols: Union[str, Grouper, np.ndarray, List[Any]], aggfunc: Union[Callable, List[Callable], Dict[str, Union[Callable, List[Callable]]], kwargs: Dict[str, Any], observed: bool, margins_name: str) -> Tuple[DataFrame, List[str], Series]:
    ...

def _convert_by(by: Union[str, Grouper, np.ndarray, List[Any]]) -> List[Any]:
    ...

def pivot(data: DataFrame, columns: Union[str, List[str]], index: Union[str, Grouper, np.ndarray, List[Any]] = lib.no_default, values: Union[str, List[str]] = lib.no_default) -> DataFrame:
    ...

def crosstab(index: Union[np.ndarray, Series, List[Union[np.ndarray, Series]]], columns: Union[np.ndarray, Series, List[Union[np.ndarray, Series]]], values: Union[np.ndarray, Series] = None, rownames: Union[str, List[str]] = None, colnames: Union[str, List[str]] = None, aggfunc: Callable = None, margins: bool = False, margins_name: str = 'All', dropna: bool = True, normalize: Union[bool, str, Literal['all', 'index', 'columns']] = False) -> DataFrame:
    ...

def _normalize(table: DataFrame, normalize: Union[bool, str, Literal['all', 'index', 'columns']], margins: bool, margins_name: str = 'All') -> DataFrame:
    ...

def _get_names(arrs: List[Union[np.ndarray, Series]], names: Union[str, List[str]], prefix: str = 'row') -> List[str]:
    ...

def _build_names_mapper(rownames: List[str], colnames: List[str]) -> Tuple[Dict[str, str], List[str], Dict[str, str], List[str]:
    ...
