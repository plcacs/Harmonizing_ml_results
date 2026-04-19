from typing import Any, Callable, Iterable, Sequence, TypeVar, Literal
import numpy as np
import numpy.typing as npt
from pandas import Series, MultiIndex, DataFrame
from pandas.core.arrays import ExtensionArray
from pandas.core.indexes.base import Index

T = TypeVar("T", Index, Series, DataFrame, npt.NDArray[Any])


def get_indexer_indexer(
    target: Index,
    level: int | str | Sequence[int | str] | None,
    ascending: bool | Sequence[bool] | npt.NDArray[np.bool_],
    kind: Literal["quicksort", "mergesort", "heapsort", "stable"],
    na_position: Literal["first", "last"],
    sort_remaining: bool,
    key: Callable[[Index], Any] | None,
) -> npt.NDArray[np.intp] | None: ...
def get_group_index(
    labels: Sequence[npt.NDArray[np.signedinteger[Any]]],
    shape: tuple[int, ...],
    sort: bool,
    xnull: bool,
) -> npt.NDArray[np.int64]: ...
def get_compressed_ids(
    labels: Sequence[npt.NDArray[np.signedinteger[Any]]],
    sizes: tuple[int, ...],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]: ...
def is_int64_overflow_possible(shape: Sequence[int]) -> bool: ...
def _decons_group_index(
    comp_labels: npt.NDArray[np.signedinteger[Any]],
    shape: tuple[int, ...],
) -> list[npt.NDArray[np.signedinteger[Any]]]: ...
def decons_obs_group_ids(
    comp_ids: npt.NDArray[np.signedinteger[Any]],
    obs_ids: npt.NDArray[np.signedinteger[Any]],
    shape: tuple[int, ...],
    labels: Sequence[npt.NDArray[np.signedinteger[Any]]],
    xnull: bool,
) -> list[npt.NDArray[np.intp]]: ...
def lexsort_indexer(
    keys: Sequence[Index | Series | npt.NDArray[Any]],
    orders: bool | Sequence[bool] | None = ...,
    na_position: Literal["first", "last"] = ...,
    key: Callable[[Any], Any] | None = ...,
    codes_given: bool = ...,
) -> npt.NDArray[np.intp]: ...
def nargsort(
    items: npt.NDArray[Any] | ExtensionArray | Index | Series,
    kind: Literal["quicksort", "mergesort", "heapsort", "stable"] = ...,
    ascending: bool = ...,
    na_position: Literal["first", "last"] = ...,
    key: Callable[[Any], Any] | None = ...,
    mask: npt.NDArray[np.bool_] | None = ...,
) -> npt.NDArray[np.intp]: ...
def nargminmax(
    values: ExtensionArray,
    method: Literal["argmax", "argmin"],
    axis: int = ...,
) -> int | npt.NDArray[np.intp]: ...
def _nanargminmax(
    values: npt.NDArray[Any],
    mask: npt.NDArray[np.bool_],
    func: Callable[[npt.NDArray[Any]], int],
) -> int: ...
def _ensure_key_mapped_multiindex(
    index: MultiIndex,
    key: Callable[[Index], Any],
    level: Sequence[int | str] | int | str | None = ...,
) -> MultiIndex: ...
def ensure_key_mapped(
    values: T,
    key: Callable[[Any], Any] | None,
    levels: Sequence[object] | None = ...,
) -> T: ...
def get_indexer_dict(
    label_list: Sequence[npt.NDArray[np.intp]],
    keys: Sequence[Sequence[object]],
) -> dict[object, npt.NDArray[np.intp]]: ...
def get_group_index_sorter(
    group_index: npt.NDArray[np.signedinteger[Any]],
    ngroups: int | None = ...,
) -> npt.NDArray[np.intp]: ...
def compress_group_index(
    group_index: npt.NDArray[np.signedinteger[Any]],
    sort: bool = ...,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]: ...
def _reorder_by_uniques(
    uniques: npt.NDArray[np.int64],
    labels: npt.NDArray[np.intp],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.intp]]: ...