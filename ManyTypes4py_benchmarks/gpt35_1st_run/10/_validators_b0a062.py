from __future__ import annotations
from collections.abc import Iterable, Sequence
from typing import TypeVar, overload, Tuple, Dict, Any

BoolishT = TypeVar('BoolishT', bool, int)
BoolishNoneT = TypeVar('BoolishNoneT', bool, int, None)

def _check_arg_length(fname: str, args: Tuple, max_fname_arg_count: int, compat_args: Dict[str, Any]) -> None:
    ...

def _check_for_default_values(fname: str, arg_val_dict: Dict[str, Any], compat_args: Dict[str, Any]) -> None:
    ...

def validate_args(fname: str, args: Tuple, max_fname_arg_count: int, compat_args: Dict[str, Any]) -> None:
    ...

def _check_for_invalid_keys(fname: str, kwargs: Dict[str, Any], compat_args: Dict[str, Any]) -> None:
    ...

def validate_kwargs(fname: str, kwargs: Dict[str, Any], compat_args: Dict[str, Any]) -> None:
    ...

def validate_args_and_kwargs(fname: str, args: Tuple, kwargs: Dict[str, Any], max_fname_arg_count: int, compat_args: Dict[str, Any]) -> None:
    ...

def validate_bool_kwarg(value: bool, arg_name: str, none_allowed: bool = True, int_allowed: bool = False) -> bool:
    ...

def validate_fillna_kwargs(value: Any, method: Any, validate_scalar_dict_value: bool = True) -> Tuple[Any, Any]:
    ...

def validate_percentile(q: Union[float, Iterable[float]]) -> np.ndarray:
    ...

def validate_ascending(ascending: Union[Sequence, bool]) -> Union[bool, List[bool]]:
    ...

def validate_endpoints(closed: Union[None, str]) -> Tuple[bool, bool]:
    ...

def validate_inclusive(inclusive: Union[str, Tuple[bool, bool]]) -> Tuple[bool, bool]:
    ...

def validate_insert_loc(loc: int, length: int) -> int:
    ...

def check_dtype_backend(dtype_backend: Any) -> None:
    ...
