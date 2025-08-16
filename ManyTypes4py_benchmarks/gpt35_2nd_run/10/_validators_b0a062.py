from __future__ import annotations
from collections.abc import Iterable, Sequence
from typing import TypeVar, overload, Tuple, Dict, Any

BoolishT = TypeVar('BoolishT', bool, int)
BoolishNoneT = TypeVar('BoolishNoneT', bool, int, None)

def _check_arg_length(fname: str, args: Tuple, max_fname_arg_count: int, compat_args: Tuple) -> None:
    ...

def _check_for_default_values(fname: str, arg_val_dict: Dict, compat_args: Dict) -> None:
    ...

def validate_args(fname: str, args: Tuple, max_fname_arg_count: int, compat_args: Dict) -> None:
    ...

def _check_for_invalid_keys(fname: str, kwargs: Dict, compat_args: Dict) -> None:
    ...

def validate_kwargs(fname: str, kwargs: Dict, compat_args: Dict) -> None:
    ...

def validate_args_and_kwargs(fname: str, args: Tuple, kwargs: Dict, max_fname_arg_count: int, compat_args: Dict) -> None:
    ...

def validate_bool_kwarg(value: Any, arg_name: str, none_allowed: bool = True, int_allowed: bool = False) -> Any:
    ...

def validate_fillna_kwargs(value: Any, method: Any, validate_scalar_dict_value: bool = True) -> Tuple:
    ...

def validate_percentile(q: Any) -> np.ndarray:
    ...

def validate_ascending(ascending: Any) -> Any:
    ...

def validate_endpoints(closed: Any) -> Tuple[bool, bool]:
    ...

def validate_inclusive(inclusive: Any) -> Tuple[bool, bool]:
    ...

def validate_insert_loc(loc: int, length: int) -> int:
    ...

def check_dtype_backend(dtype_backend: Any) -> None:
    ...
