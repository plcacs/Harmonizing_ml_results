from __future__ import annotations
from datetime import timedelta
import numpy as np
from pandas._libs.tslibs import BaseOffset
from pandas._libs.window.indexers import calculate_variable_window_bounds
from pandas.util._decorators import Appender
from pandas.core.dtypes.common import ensure_platform_int
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.tseries.offsets import Nano

get_window_bounds_doc: str = '\nComputes the bounds of a window.\n\nParameters\n----------\nnum_values : int, default 0\n    number of values that will be aggregated over\nwindow_size : int, default 0\n    the number of rows in a window\nmin_periods : int, default None\n    min_periods passed from the top level rolling API\ncenter : bool, default None\n    center passed from the top level rolling API\nclosed : str, default None\n    closed passed from the top level rolling API\nstep : int, default None\n    step passed from the top level rolling API\n    .. versionadded:: 1.5\nwin_type : str, default None\n    win_type passed from the top level rolling API\n\nReturns\n-------\nA tuple of ndarray[int64]s, indicating the boundaries of each\nwindow\n'

class BaseIndexer:
    def __init__(self, index_array: np.ndarray = None, window_size: int = 0, **kwargs):
    def get_window_bounds(self, num_values: int = 0, min_periods: int = None, center: bool = None, closed: str = None, step: int = None) -> tuple:

class FixedWindowIndexer(BaseIndexer):
    def get_window_bounds(self, num_values: int = 0, min_periods: int = None, center: bool = None, closed: str = None, step: int = None) -> tuple:

class VariableWindowIndexer(BaseIndexer):
    def get_window_bounds(self, num_values: int = 0, min_periods: int = None, center: bool = None, closed: str = None, step: int = None) -> tuple:

class VariableOffsetWindowIndexer(BaseIndexer):
    def __init__(self, index_array: np.ndarray = None, window_size: int = 0, index: DatetimeIndex = None, offset: BaseOffset = None, **kwargs):
    def get_window_bounds(self, num_values: int = 0, min_periods: int = None, center: bool = None, closed: str = None, step: int = None) -> tuple:

class ExpandingIndexer(BaseIndexer):
    def get_window_bounds(self, num_values: int = 0, min_periods: int = None, center: bool = None, closed: str = None, step: int = None) -> tuple:

class FixedForwardWindowIndexer(BaseIndexer):
    def get_window_bounds(self, num_values: int = 0, min_periods: int = None, center: bool = None, closed: str = None, step: int = None) -> tuple:

class GroupbyIndexer(BaseIndexer):
    def __init__(self, index_array: np.ndarray = None, window_size: int = 0, groupby_indices: dict = None, window_indexer: BaseIndexer = None, indexer_kwargs: dict = None, **kwargs):
    def get_window_bounds(self, num_values: int = 0, min_periods: int = None, center: bool = None, closed: str = None, step: int = None) -> tuple:

class ExponentialMovingWindowIndexer(BaseIndexer):
    def get_window_bounds(self, num_values: int = 0, min_periods: int = None, center: bool = None, closed: str = None, step: int = None) -> tuple:
