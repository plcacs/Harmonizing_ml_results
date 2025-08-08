from __future__ import annotations
from typing import TYPE_CHECKING, Callable
import warnings
import numpy as np
from pandas._libs import missing as libmissing
from pandas.core.nanops import check_below_min_count
if TYPE_CHECKING:
    from collections.abc import Callable
    from pandas._typing import AxisInt, npt

def _reductions(func: Callable, values: np.ndarray, mask: np.ndarray[bool], *, skipna: bool = True, min_count: int = 0, axis: AxisInt = None, **kwargs) -> npt:
    def sum(values: np.ndarray, mask: np.ndarray[bool], *, skipna: bool = True, min_count: int = 0, axis: AxisInt = None) -> npt:
    def prod(values: np.ndarray, mask: np.ndarray[bool], *, skipna: bool = True, min_count: int = 0, axis: AxisInt = None) -> npt:
    def _minmax(func: Callable, values: np.ndarray, mask: np.ndarray[bool], *, skipna: bool = True, axis: AxisInt = None) -> npt:
    def min(values: np.ndarray, mask: np.ndarray[bool], *, skipna: bool = True, axis: AxisInt = None) -> npt:
    def max(values: np.ndarray, mask: np.ndarray[bool], *, skipna: bool = True, axis: AxisInt = None) -> npt:
    def mean(values: np.ndarray, mask: np.ndarray[bool], *, skipna: bool = True, axis: AxisInt = None) -> npt:
    def var(values: np.ndarray, mask: np.ndarray[bool], *, skipna: bool = True, axis: AxisInt = None, ddof: int = 1) -> npt:
    def std(values: np.ndarray, mask: np.ndarray[bool], *, skipna: bool = True, axis: AxisInt = None, ddof: int = 1) -> npt:
