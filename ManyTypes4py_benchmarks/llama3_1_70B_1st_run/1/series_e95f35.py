from typing import Any, Generic, Iterable, List, Optional, Tuple, TypeVar, Union, cast
import numpy as np
import pandas as pd

T = TypeVar('T')

class Series(Generic[T]):
    def __init__(self, data: Any = None, index: Any = None, dtype: Any = None, name: Any = None, copy: bool = False, fastpath: bool = False):
        pass

    def add(self, other: Any) -> 'Series[T]':
        pass

    def radd(self, other: Any) -> 'Series[T]':
        pass

    def div(self, other: Any) -> 'Series[T]':
        pass

    def rdiv(self, other: Any) -> 'Series[T]':
        pass

    def truediv(self, other: Any) -> 'Series[T]':
        pass

    def rtruediv(self, other: Any) -> 'Series[T]':
        pass

    def mul(self, other: Any) -> 'Series[T]':
        pass

    def rmul(self, other: Any) -> 'Series[T]':
        pass

    def sub(self, other: Any) -> 'Series[T]':
        pass

    def rsub(self, other: Any) -> 'Series[T]':
        pass

    def mod(self, other: Any) -> 'Series[T]':
        pass

    def rmod(self, other: Any) -> 'Series[T]':
        pass

    def pow(self, other: Any) -> 'Series[T]':
        pass

    def rpow(self, other: Any) -> 'Series[T]':
        pass

    def floordiv(self, other: Any) -> 'Series[T]':
        pass

    def rfloordiv(self, other: Any) -> 'Series[T]':
        pass

    def eq(self, other: Any) -> 'Series[T]':
        pass

    def gt(self, other: Any) -> 'Series[T]':
        pass

    def ge(self, other: Any) -> 'Series[T]':
        pass

    def lt(self, other: Any) -> 'Series[T]':
        pass

    def le(self, other: Any) -> 'Series[T]':
        pass

    def ne(self, other: Any) -> 'Series[T]':
        pass

    def divmod(self, other: Any) -> Tuple['Series[T]', 'Series[T]']:
        pass

    def rdivmod(self, other: Any) -> Tuple['Series[T]', 'Series[T]']:
        pass

    def between(self, left: Any, right: Any, inclusive: bool = True) -> 'Series[T]':
        pass

    def map(self, arg: Any) -> 'Series[T]':
        pass

    def alias(self, name: Any) -> 'Series[T]':
        pass

    @property
    def shape(self) -> Tuple[int]:
        pass

    @property
    def name(self) -> Any:
        pass

    @name.setter
    def name(self, name: Any) -> None:
        pass

    def rename(self, index: Any = None, **kwargs: Any) -> 'Series[T]':
        pass

    @property
    def index(self) -> 'Index':
        pass

    @property
    def is_unique(self) -> bool:
        pass

    def reset_index(self, level: Any = None, drop: bool = False, name: Any = None, inplace: bool = False) -> Optional['Series[T]']:
        pass

    def to_frame(self, name: Any = None) -> 'DataFrame':
        pass

    def to_string(self, buf: Any = None, na_rep: str = 'NaN', float_format: Any = None, header: bool = True, index: bool = True, length: bool = False, dtype: bool = False, name: bool = False, max_rows: int = None) -> str:
        pass

    def to_clipboard(self, excel: bool = True, sep: Any = None, **kwargs: Any) -> None:
        pass

    def to_dict(self, into: Any = dict) -> Any:
        pass

    def to_latex(self, buf: Any = None, columns: Any = None, col_space: Any = None, header: bool = True, index: bool = True, na_rep: str = 'NaN', formatters: Any = None, float_format: Any = None, sparsify: Any = None, index_names: bool = True, bold_rows: bool = False, column_format: Any = None, longtable: Any = None, escape: Any = None, encoding: Any = None, decimal: str = '.', multicolumn: Any = None, multicolumn_format: Any = None, multirow: Any = None) -> Any:
        pass

    def to_pandas(self) -> pd.Series:
        pass

    def toPandas(self) -> pd.Series:
        pass

    def to_list(self) -> List[Any]:
        pass

    def drop_duplicates(self, keep: str = 'first', inplace: bool = False) -> Optional['Series[T]']:
        pass

    def reindex(self, index: Any = None, fill_value: Any = None) -> 'Series[T]':
        pass

    def reindex_like(self, other: Any) -> 'Series[T]':
        pass

    def fillna(self, value: Any = None, method: Any = None, axis: Any = None, inplace: bool = False, limit: Any = None) -> Optional['Series[T]']:
        pass

    def dropna(self, axis: Any = 0, inplace: bool = False, **kwargs: Any) -> Optional['Series[T]']:
        pass

    def clip(self, lower: Any = None, upper: Any = None) -> 'Series[T]':
        pass

    def drop(self, labels: Any = None, index: Any = None, level: Any = None) -> 'Series[T]':
        pass

    def head(self, n: int = 5) -> 'Series[T]':
        pass

    def last(self, offset: Any) -> 'Series[T]':
        pass

    def first(self, offset: Any) -> 'Series[T]':
        pass

    def unique(self) -> 'Series[T]':
        pass

    def sort_values(self, ascending: bool = True, inplace: bool = False, na_position: str = 'last') -> Optional['Series[T]']:
        pass

    def sort_index(self, axis: Any = 0, level: Any = None, ascending: bool = True, inplace: bool = False, kind: Any = None, na_position: str = 'last') -> Optional['Series[T]']:
        pass

    def swaplevel(self, i: int = -2, j: int = -1, copy: bool = True) -> 'Series[T]':
        pass

    def swapaxes(self, i: Any, j: Any, copy: bool = True) -> 'Series[T]':
        pass

    def corr(self, other: 'Series[T]', method: str = 'pearson') -> float:
        pass

    def nsmallest(self, n: int = 5) -> 'Series[T]':
        pass

    def nlargest(self, n: int = 5) -> 'Series[T]':
        pass

    def append(self, to_append: Any, ignore_index: bool = False, verify_integrity: bool = False) -> 'Series[T]':
        pass

    def sample(self, n: Any = None, frac: Any = None, replace: bool = False, random_state: Any = None) -> 'Series[T]':
        pass

    def hist(self, bins: int = 10, **kwds: Any) -> Any:
        pass

    def apply(self, func: Any, args: Any = (), **kwds: Any) -> 'Series[T]':
        pass

    def aggregate(self, func: Any) -> Any:
        pass

    def agg(self, func: Any) -> Any:
        pass

    def transpose(self, *args: Any, **kwargs: Any) -> 'Series[T]':
        pass

    def T(self) -> 'Series[T]':
        pass

    def transform(self, func: Any, axis: Any = 0, *args: Any, **kwargs: Any) -> Any:
        pass

    def transform_batch(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        pass

    def round(self, decimals: int = 0) -> 'Series[T]':
        pass

    def quantile(self, q: Any = 0.5, accuracy: int = 10000) -> Any:
        pass

    def rank(self, method: str = 'average', ascending: bool = True) -> 'Series[T]':
        pass

    def filter(self, items: Any = None, like: Any = None, regex: Any = None, axis: Any = None) -> 'Series[T]':
        pass

    def describe(self, percentiles: Any = None) -> 'Series[T]':
        pass

    def diff(self, periods: int = 1) -> 'Series[T]':
        pass

    def idxmax(self, skipna: bool = True) -> Any:
        pass

    def idxmin(self, skipna: bool = True) -> Any:
        pass

    def pop(self, item: Any) -> Any:
        pass

    def copy(self, deep: Any = None) -> 'Series[T]':
        pass

    def mode(self, dropna: bool = True) -> 'Series[T]':
        pass

    def replace(self, to_replace: Any = None, value: Any = None, regex: bool = False) -> 'Series[T]':
        pass

    def update(self, other: 'Series[T]') -> None:
        pass

    def where(self, cond: 'Series[T]', other: Any = np.nan) -> 'Series[T]':
        pass

    def mask(self, cond: 'Series[T]', other: Any = np.nan) -> 'Series[T]':
        pass

    def xs(self, key: Any, level: Any = None) -> 'Series[T]':
        pass

    def pct_change(self, periods: int = 1) -> 'Series[T]':
        pass

    def combine_first(self, other: 'Series[T]') -> 'Series[T]':
        pass

    def dot(self, other: Any) -> Any:
        pass

    def __matmul__(self, other: Any) -> Any:
        pass

    def repeat(self, repeats: Any) -> 'Series[T]':
        pass

    def asof(self, where: Any) -> Any:
        pass

    def mad(self) -> float:
        pass

    def unstack(self, level: int = -1) -> 'DataFrame':
        pass

    def item(self) -> Any:
        pass

    def iteritems(self) -> Iterable[Tuple[Any, Any]]:
        pass

    def items(self) -> Iterable[Tuple[Any, Any]]:
        pass

    def droplevel(self, level: Any) -> 'Series[T]':
        pass

    def tail(self, n: int = 5) -> 'Series[T]':
        pass

    def explode(self) -> 'Series[T]':
        pass

    def argsort(self) -> 'Series[T]':
        pass

    def argmax(self) -> int:
        pass

    def argmin(self) -> int:
        pass

    def compare(self, other: 'Series[T]', keep_shape: bool = False, keep_equal: bool = False) -> 'DataFrame':
        pass

    def align(self, other: Any, join: str = 'outer', axis: Any = None, copy: bool = True) -> Tuple['Series[T]', Any]:
        pass

    def between_time(self, start_time: Any, end_time: Any, include_start: bool = True, include_end: bool = True, axis: Any = 0) -> 'Series[T]':
        pass

    def at_time(self, time: Any, asof: bool = False, axis: Any = 0) -> 'Series[T]':
        pass

    def _cum(self, func: Any, skipna: bool, part_cols: Any = (), ascending: bool = True) -> 'Series[T]':
        pass

    def _cumsum(self, skipna: bool, part_cols: Any = ()) -> 'Series[T]':
        pass

    def _cumprod(self, skipna: bool, part_cols: Any = ()) -> 'Series[T]':
        pass

    def _reduce_for_stat_function(self, sfun: Any, name: str, axis: Any = None, numeric_only: Any = None, **kwargs: Any) -> Any:
        pass

    def __getitem__(self, key: Any) -> Any:
        pass

    def __getattr__(self, item: str) -> Any:
        pass

    def _to_internal_pandas(self) -> pd.Series:
        pass

    def __repr__(self) -> str:
        pass

    def __dir__(self) -> List[str]:
        pass

    def __iter__(self) -> Iterable[Any]:
        pass

    def __class_getitem__(cls, params: Any) -> Any:
        pass

def unpack_scalar(sdf: Any) -> Any:
    pass

def first_series(df: Any) -> 'Series[T]':
    pass
