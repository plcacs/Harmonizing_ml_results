from __future__ import annotations
from collections import defaultdict
from copy import copy
import csv
from enum import Enum
import itertools
from typing import TYPE_CHECKING, Any, cast, final, overload, Sequence, Mapping, Callable, Iterable, Hashable, HashableT, Scalar, SequenceT

class ParserBase:
    class BadLineHandleMethod(Enum):
        ERROR = 0
        WARN = 1
        SKIP = 2

    def __init__(self, kwds: Mapping[str, Any]) -> None:
        # ... rest of the method ...

    @final
    def _should_parse_dates(self, i: int) -> bool:
        # ... rest of the method ...

    @final
    def _extract_multi_indexer_columns(self, header: Sequence[Any], index_names: Sequence[Any], passed_names: bool) -> Tuple[Sequence[Any], Sequence[Any], Sequence[Any], bool]:
        # ... rest of the method ...

    @final
    def _make_index(self, alldata: Sequence[Any], columns: Sequence[Any], indexnamerow: Sequence[Any]) -> Tuple[Optional[Index], Sequence[Any]]:
        # ... rest of the method ...

    @final
    def _set_noconvert_dtype_columns(self, col_indices: Sequence[int], names: Sequence[Any]) -> Set[int]:
        # ... rest of the method ...

    @final
    def _infer_types(self, values: np.ndarray, na_values: Set[Any], no_dtype_specified: bool, try_num_bool: bool) -> Tuple[Any, int]:
        # ... rest of the method ...

    @final
    def _do_date_conversions(self, names: Sequence[Any], data: Sequence[Any]) -> Sequence[Any]:
        # ... rest of the method ...

    @final
    def _check_data_length(self, columns: Sequence[Any], data: Sequence[Any]) -> None:
        # ... rest of the method ...

    @final
    def _validate_usecols_names(self, usecols: Sequence[Any], names: Sequence[Any]) -> Sequence[Any]:
        # ... rest of the method ...

    @final
    def _get_empty_meta(self, columns: Sequence[Any], dtype: Optional[dict]) -> Tuple[Optional[Index], Sequence[Any], dict]:
        # ... rest of the method ...

def get_na_values(col: str, na_values: Union[Sequence[Any], dict], na_fvalues: Sequence[Any], keep_default_na: bool) -> Tuple[Set[Any], Set[Any]]:
    # ... rest of the method ...

def is_index_col(col: Any) -> bool:
    # ... rest of the method ...

def validate_parse_dates_presence(parse_dates: Sequence[Any], columns: Sequence[Any]) -> Set[Any]:
    # ... rest of the method ...

def _validate_usecols_arg(usecols: Any) -> Tuple[Set[int], Optional[str]]:
    # ... rest of the method ...

@overload
def evaluate_callable_usecols(usecols: Callable[[str], bool], names: Sequence[Any]) -> Set[int]:
    ...

@overload
def evaluate_callable_usecols(usecols: Any, names: Sequence[Any]) -> Any:
    ...

def evaluate_callable_usecols(usecols: Any, names: Sequence[Any]) -> Any:
    # ... rest of the method ...
