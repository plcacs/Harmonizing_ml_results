from __future__ import annotations
from collections.abc import Iterable, Sequence
from typing import Any, Dict, List, Tuple, Union, Optional, overload
import numpy as np
from pandas._libs import lib
from pandas.core.dtypes.common import is_bool, is_integer
from numpy.typing import NDArray

BoolishT = TypeVar('BoolishT', bool, int)
BoolishNoneT = TypeVar('BoolishNoneT', bool, int, None)


def _check_arg_length(
    fname: str,
    args: Tuple[Any, ...],
    max_fname_arg_count: int,
    compat_args: Dict[str, Any]
) -> None:
    if max_fname_arg_count < 0:
        raise ValueError("'max_fname_arg_count' must be non-negative")
    if len(args) > len(compat_args):
        max_arg_count: int = len(compat_args) + max_fname_arg_count
        actual_arg_count: int = len(args) + max_fname_arg_count
        argument: str = 'argument' if max_arg_count == 1 else 'arguments'
        raise TypeError(f'{fname}() takes at most {max_arg_count} {argument} ({actual_arg_count} given)')


def _check_for_default_values(
    fname: str,
    arg_val_dict: Dict[str, Any],
    compat_args: Dict[str, Any]
) -> None:
    for key in arg_val_dict:
        try:
            v1: Any = arg_val_dict[key]
            v2: Any = compat_args[key]
            if v1 is not None and v2 is None or (v1 is None and v2 is not None):
                match: bool = False
            else:
                match = v1 == v2
            if not is_bool(match):
                raise ValueError("'match' is not a boolean")
        except ValueError:
            match = arg_val_dict[key] is compat_args[key]
        if not match:
            raise ValueError(f"the '{key}' parameter is not supported in the pandas implementation of {fname}()")


def validate_args(
    fname: str,
    args: Tuple[Any, ...],
    max_fname_arg_count: int,
    compat_args: Dict[str, Any]
) -> None:
    _check_arg_length(fname, args, max_fname_arg_count, compat_args)
    kwargs: Dict[str, Any] = dict(zip(compat_args, args))
    _check_for_default_values(fname, kwargs, compat_args)


def _check_for_invalid_keys(
    fname: str,
    kwargs: Dict[str, Any],
    compat_args: Dict[str, Any]
) -> None:
    diff = set(kwargs) - set(compat_args)
    if diff:
        bad_arg = next(iter(diff))
        raise TypeError(f"{fname}() got an unexpected keyword argument '{bad_arg}'")


def validate_kwargs(
    fname: str,
    kwargs: Dict[str, Any],
    compat_args: Dict[str, Any]
) -> None:
    kwds: Dict[str, Any] = kwargs.copy()
    _check_for_invalid_keys(fname, kwargs, compat_args)
    _check_for_default_values(fname, kwds, compat_args)


def validate_args_and_kwargs(
    fname: str,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    max_fname_arg_count: int,
    compat_args: Dict[str, Any]
) -> None:
    _check_arg_length(fname, args + tuple(kwargs.values()), max_fname_arg_count, compat_args)
    args_dict: Dict[str, Any] = dict(zip(compat_args, args))
    for key in args_dict:
        if key in kwargs:
            raise TypeError(f"{fname}() got multiple values for keyword argument '{key}'")
    kwargs.update(args_dict)
    validate_kwargs(fname, kwargs, compat_args)


def validate_bool_kwarg(
    value: Any,
    arg_name: str,
    none_allowed: bool = True,
    int_allowed: bool = False
) -> Union[bool, int, None]:
    good_value: bool = is_bool(value)
    if none_allowed:
        good_value = good_value or value is None
    if int_allowed:
        good_value = good_value or isinstance(value, int)
    if not good_value:
        raise ValueError(f'For argument "{arg_name}" expected type bool, received type {type(value).__name__}.')
    return value


def validate_fillna_kwargs(
    value: Any,
    method: Any,
    validate_scalar_dict_value: bool = True
) -> Tuple[Any, Any]:
    from pandas.core.missing import clean_fill_method  # type: ignore
    if value is None and method is None:
        raise ValueError("Must specify a fill 'value' or 'method'.")
    if value is None and method is not None:
        method = clean_fill_method(method)
    elif value is not None and method is None:
        if validate_scalar_dict_value and isinstance(value, (list, tuple)):
            raise TypeError(f'"value" parameter must be a scalar or dict, but you passed a "{type(value).__name__}"')
    elif value is not None and method is not None:
        raise ValueError("Cannot specify both 'value' and 'method'.")
    return (value, method)


def validate_percentile(q: Union[float, Iterable[float]]) -> NDArray[Any]:
    q_arr: NDArray[Any] = np.asarray(q)
    msg: str = 'percentiles should all be in the interval [0, 1]'
    if q_arr.ndim == 0:
        if not 0 <= q_arr <= 1:
            raise ValueError(msg)
    elif not all((0 <= qs <= 1 for qs in q_arr)):
        raise ValueError(msg)
    return q_arr


@overload
def validate_ascending(ascending: Union[bool, int]) -> Union[bool, int]:
    ...


@overload
def validate_ascending(ascending: Sequence[Union[bool, int]]) -> List[Union[bool, int]]:
    ...


def validate_ascending(
    ascending: Union[bool, int, Sequence[Union[bool, int]]]
) -> Union[bool, int, List[Union[bool, int]]]:
    kwargs: Dict[str, bool] = {'none_allowed': False, 'int_allowed': True}
    if not isinstance(ascending, Sequence):
        return validate_bool_kwarg(ascending, 'ascending', **kwargs)
    return [validate_bool_kwarg(item, 'ascending', **kwargs) for item in ascending]


def validate_endpoints(closed: Optional[str]) -> Tuple[bool, bool]:
    left_closed: bool = False
    right_closed: bool = False
    if closed is None:
        left_closed = True
        right_closed = True
    elif closed == 'left':
        left_closed = True
    elif closed == 'right':
        right_closed = True
    else:
        raise ValueError("Closed has to be either 'left', 'right' or None")
    return (left_closed, right_closed)


def validate_inclusive(inclusive: str) -> Tuple[bool, bool]:
    left_right_inclusive: Optional[Tuple[bool, bool]] = None
    if isinstance(inclusive, str):
        left_right_inclusive = {
            'both': (True, True),
            'left': (True, False),
            'right': (False, True),
            'neither': (False, False)
        }.get(inclusive)
    if left_right_inclusive is None:
        raise ValueError("Inclusive has to be either 'both', 'neither', 'left' or 'right'")
    return left_right_inclusive


def validate_insert_loc(loc: int, length: int) -> int:
    if not is_integer(loc):
        raise TypeError(f'loc must be an integer between -{length} and {length}')
    if loc < 0:
        loc += length
    if not 0 <= loc <= length:
        raise IndexError(f'loc must be an integer between -{length} and {length}')
    return loc


def check_dtype_backend(dtype_backend: Any) -> None:
    if dtype_backend is not lib.no_default:
        if dtype_backend not in ['numpy_nullable', 'pyarrow']:
            raise ValueError(f"dtype_backend {dtype_backend} is invalid, only 'numpy_nullable' and 'pyarrow' are allowed.")