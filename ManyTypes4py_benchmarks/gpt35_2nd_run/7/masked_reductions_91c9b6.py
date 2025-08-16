def _reductions(func: Callable, values: npt.ArrayLike, mask: npt.ArrayLike, *, skipna: bool = True, min_count: int = 0, axis: AxisInt = None, **kwargs) -> npt.ArrayLike:
def sum(values: npt.ArrayLike, mask: npt.ArrayLike, *, skipna: bool = True, min_count: int = 0, axis: AxisInt = None) -> npt.ArrayLike:
def prod(values: npt.ArrayLike, mask: npt.ArrayLike, *, skipna: bool = True, min_count: int = 0, axis: AxisInt = None) -> npt.ArrayLike:
def _minmax(func: Callable, values: npt.ArrayLike, mask: npt.ArrayLike, *, skipna: bool = True, axis: AxisInt = None) -> npt.ArrayLike:
def min(values: npt.ArrayLike, mask: npt.ArrayLike, *, skipna: bool = True, axis: AxisInt = None) -> npt.ArrayLike:
def max(values: npt.ArrayLike, mask: npt.ArrayLike, *, skipna: bool = True, axis: AxisInt = None) -> npt.ArrayLike:
def mean(values: npt.ArrayLike, mask: npt.ArrayLike, *, skipna: bool = True, axis: AxisInt = None) -> npt.ArrayLike:
def var(values: npt.ArrayLike, mask: npt.ArrayLike, *, skipna: bool = True, axis: AxisInt = None, ddof: int = 1) -> npt.ArrayLike:
def std(values: npt.ArrayLike, mask: npt.ArrayLike, *, skipna: bool = True, axis: AxisInt = None, ddof: int = 1) -> npt.ArrayLike:
