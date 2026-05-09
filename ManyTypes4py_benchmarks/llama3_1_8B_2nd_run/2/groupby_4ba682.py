from typing import Any, Union, TypeVar, Tuple, Sequence, Dict, List, Callable, Optional
import pandas as pd

T = TypeVar('T')
NDFrameT = TypeVar('NDFrameT', bound=pd.DataFrame)
OutputFrameOrSeries = TypeVar('OutputFrameOrSeries', bound=NDFrameT)

class GroupByPlot(PandasObject):
    ...

class BaseGroupBy(PandasObject, SelectionMixin[NDFrameT], GroupByIndexingMixin):
    ...

class GroupBy(BaseGroupBy[NDFrameT]):
    ...

def get_groupby(obj: pd.Series | pd.DataFrame, by: Optional[Union[Hashable, Sequence[Hashable]]] = None, grouper: Optional[GroupBy] = None, group_keys: bool = True) -> GroupBy:
    ...

def _insert_quantile_level(idx: pd.Index, qs: np.ndarray[float64]) -> pd.MultiIndex:
    ...
