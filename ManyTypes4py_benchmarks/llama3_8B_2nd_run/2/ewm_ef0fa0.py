from __future__ import annotations
import numpy as np
from pandas._libs.tslibs import Timedelta
from pandas._libs.window.aggregations import window_aggregations
from pandas.util._decorators import doc
from pandas.core.dtypes.common import is_datetime64_dtype, is_numeric_dtype
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import isna
from pandas.core import common
from pandas.core.arrays.datetimelike import dtype_to_unit
from pandas.core.indexers.objects import BaseIndexer, ExponentialMovingWindowIndexer, GroupbyIndexer
from pandas.core.window.common import zsqrt
from pandas.core.window.doc import _shared_docs, create_section_header, kwargs_numeric_only, numba_notes, template_header, template_returns, template_see_also, window_agg_numba_parameters
from pandas.core.window.numba import generate_numba_ewm_func, generate_numba_ewm_table_func
from pandas.core.window.online import EWMMeanState, generate_online_numba_ewma_func

class ExponentialMovingWindow(BaseWindow):
    ...

class ExponentialMovingWindowGroupby(BaseWindowGroupby, ExponentialMovingWindow):
    ...

class OnlineExponentialMovingWindow(ExponentialMovingWindow):
    def __init__(self, 
                 obj: 'Series' | 'DataFrame', 
                 com: float | None = None, 
                 span: float | None = None, 
                 halflife: float | str | np.timedelta64 | None = None, 
                 alpha: float | None = None, 
                 min_periods: int = 0, 
                 adjust: bool = True, 
                 ignore_na: bool = False, 
                 times: np.ndarray | Series | None = None, 
                 engine: str = 'numba', 
                 engine_kwargs: dict | None = None, 
                 selection: dict | None = None) -> None:
        ...

    def mean(self, 
             *args: tuple, 
             update: 'DataFrame' | 'Series' | None = None, 
             update_times: np.ndarray | Series | None = None, 
             **kwargs: dict) -> 'DataFrame' | 'Series':
        ...

    def std(self, 
             bias: bool = False, 
             *args: tuple, 
             **kwargs: dict) -> 'DataFrame' | 'Series':
        ...

    def corr(self, 
             other: 'Series' | 'DataFrame' | None = None, 
             pairwise: bool | None = None, 
             numeric_only: bool = False) -> 'DataFrame':
        ...

    def cov(self, 
             other: 'Series' | 'DataFrame' | None = None, 
             pairwise: bool | None = None, 
             bias: bool = False, 
             numeric_only: bool = False) -> 'DataFrame':
        ...

    def var(self, 
             bias: bool = False, 
             numeric_only: bool = False) -> 'DataFrame':
        ...

    def aggregate(self, 
                   func: callable | None = None, 
                   *args: tuple, 
                   **kwargs: dict) -> 'DataFrame':
        ...
