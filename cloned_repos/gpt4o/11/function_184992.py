from __future__ import annotations
from typing import TYPE_CHECKING, Any, TypeVar, cast, overload, Optional, Tuple, Dict
import numpy as np
from numpy import ndarray
from pandas._libs.lib import is_bool, is_integer
from pandas.errors import UnsupportedFunctionCall
from pandas.util._validators import validate_args, validate_args_and_kwargs, validate_kwargs

if TYPE_CHECKING:
    from pandas._typing import Axis, AxisInt
    AxisNoneT = TypeVar('AxisNoneT', Axis, None)

class CompatValidator:
    def __init__(self, defaults: Dict[str, Any], fname: Optional[str] = None, method: Optional[str] = None, max_fname_arg_count: Optional[int] = None) -> None:
        self.fname = fname
        self.method = method
        self.defaults = defaults
        self.max_fname_arg_count = max_fname_arg_count

    def __call__(self, args: Tuple[Any, ...], kwargs: Dict[str, Any], fname: Optional[str] = None, max_fname_arg_count: Optional[int] = None, method: Optional[str] = None) -> None:
        if not args and (not kwargs):
            return None
        fname = self.fname if fname is None else fname
        max_fname_arg_count = self.max_fname_arg_count if max_fname_arg_count is None else max_fname_arg_count
        method = self.method if method is None else method
        if method == 'args':
            validate_args(fname, args, max_fname_arg_count, self.defaults)
        elif method == 'kwargs':
            validate_kwargs(fname, kwargs, self.defaults)
        elif method == 'both':
            validate_args_and_kwargs(fname, args, kwargs, max_fname_arg_count, self.defaults)
        else:
            raise ValueError(f"invalid validation method '{method}'")

ARGMINMAX_DEFAULTS: Dict[str, Any] = {'out': None}
validate_argmin = CompatValidator(ARGMINMAX_DEFAULTS, fname='argmin', method='both', max_fname_arg_count=1)
validate_argmax = CompatValidator(ARGMINMAX_DEFAULTS, fname='argmax', method='both', max_fname_arg_count=1)

def process_skipna(skipna: Any, args: Tuple[Any, ...]) -> Tuple[bool, Tuple[Any, ...]]:
    if isinstance(skipna, ndarray) or skipna is None:
        args = (skipna,) + args
        skipna = True
    return (skipna, args)

def validate_argmin_with_skipna(skipna: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> bool:
    skipna, args = process_skipna(skipna, args)
    validate_argmin(args, kwargs)
    return skipna

def validate_argmax_with_skipna(skipna: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> bool:
    skipna, args = process_skipna(skipna, args)
    validate_argmax(args, kwargs)
    return skipna

ARGSORT_DEFAULTS: Dict[str, Any] = {'axis': -1, 'kind': None, 'order': None, 'stable': None}
validate_argsort = CompatValidator(ARGSORT_DEFAULTS, fname='argsort', max_fname_arg_count=0, method='both')

ARGSORT_DEFAULTS_KIND: Dict[str, Any] = {'axis': -1, 'order': None, 'stable': None}
validate_argsort_kind = CompatValidator(ARGSORT_DEFAULTS_KIND, fname='argsort', max_fname_arg_count=0, method='both')

def validate_argsort_with_ascending(ascending: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> bool:
    if is_integer(ascending) or ascending is None:
        args = (ascending,) + args
        ascending = True
    validate_argsort_kind(args, kwargs, max_fname_arg_count=3)
    ascending = cast(bool, ascending)
    return ascending

CLIP_DEFAULTS: Dict[str, Any] = {'out': None}
validate_clip = CompatValidator(CLIP_DEFAULTS, fname='clip', method='both', max_fname_arg_count=3)

@overload
def validate_clip_with_axis(axis: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Optional[AxisInt]:
    ...

@overload
def validate_clip_with_axis(axis: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Optional[AxisInt]:
    ...

def validate_clip_with_axis(axis: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Optional[AxisInt]:
    if isinstance(axis, ndarray):
        args = (axis,) + args
        axis = None
    validate_clip(args, kwargs)
    return axis

CUM_FUNC_DEFAULTS: Dict[str, Any] = {'dtype': None, 'out': None}
validate_cum_func = CompatValidator(CUM_FUNC_DEFAULTS, method='both', max_fname_arg_count=1)
validate_cumsum = CompatValidator(CUM_FUNC_DEFAULTS, fname='cumsum', method='both', max_fname_arg_count=1)

def validate_cum_func_with_skipna(skipna: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any], name: str) -> bool:
    if not is_bool(skipna):
        args = (skipna,) + args
        skipna = True
    elif isinstance(skipna, np.bool_):
        skipna = bool(skipna)
    validate_cum_func(args, kwargs, fname=name)
    return skipna

ALLANY_DEFAULTS: Dict[str, Any] = {'dtype': None, 'out': None, 'keepdims': False, 'axis': None}
validate_all = CompatValidator(ALLANY_DEFAULTS, fname='all', method='both', max_fname_arg_count=1)
validate_any = CompatValidator(ALLANY_DEFAULTS, fname='any', method='both', max_fname_arg_count=1)

LOGICAL_FUNC_DEFAULTS: Dict[str, Any] = {'out': None, 'keepdims': False}
validate_logical_func = CompatValidator(LOGICAL_FUNC_DEFAULTS, method='kwargs')

MINMAX_DEFAULTS: Dict[str, Any] = {'axis': None, 'dtype': None, 'out': None, 'keepdims': False}
validate_min = CompatValidator(MINMAX_DEFAULTS, fname='min', method='both', max_fname_arg_count=1)
validate_max = CompatValidator(MINMAX_DEFAULTS, fname='max', method='both', max_fname_arg_count=1)

REPEAT_DEFAULTS: Dict[str, Any] = {'axis': None}
validate_repeat = CompatValidator(REPEAT_DEFAULTS, fname='repeat', method='both', max_fname_arg_count=1)

ROUND_DEFAULTS: Dict[str, Any] = {'out': None}
validate_round = CompatValidator(ROUND_DEFAULTS, fname='round', method='both', max_fname_arg_count=1)

STAT_FUNC_DEFAULTS: Dict[str, Any] = {'dtype': None, 'out': None, 'keepdims': False}
SUM_DEFAULTS = STAT_FUNC_DEFAULTS.copy()
SUM_DEFAULTS.update({'axis': None, 'keepdims': False, 'initial': None})
PROD_DEFAULTS = SUM_DEFAULTS.copy()
MEAN_DEFAULTS = SUM_DEFAULTS.copy()
MEDIAN_DEFAULTS = STAT_FUNC_DEFAULTS.copy()
MEDIAN_DEFAULTS.update({'overwrite_input': False, 'keepdims': False})
validate_stat_func = CompatValidator(STAT_FUNC_DEFAULTS, method='kwargs')
validate_sum = CompatValidator(SUM_DEFAULTS, fname='sum', method='both', max_fname_arg_count=1)
validate_prod = CompatValidator(PROD_DEFAULTS, fname='prod', method='both', max_fname_arg_count=1)
validate_mean = CompatValidator(MEAN_DEFAULTS, fname='mean', method='both', max_fname_arg_count=1)
validate_median = CompatValidator(MEDIAN_DEFAULTS, fname='median', method='both', max_fname_arg_count=1)

STAT_DDOF_FUNC_DEFAULTS: Dict[str, Any] = {'dtype': None, 'out': None, 'keepdims': False}
validate_stat_ddof_func = CompatValidator(STAT_DDOF_FUNC_DEFAULTS, method='kwargs')

TAKE_DEFAULTS: Dict[str, Any] = {'out': None, 'mode': 'raise'}
validate_take = CompatValidator(TAKE_DEFAULTS, fname='take', method='kwargs')

TRANSPOSE_DEFAULTS: Dict[str, Any] = {'axes': None}
validate_transpose = CompatValidator(TRANSPOSE_DEFAULTS, fname='transpose', method='both', max_fname_arg_count=0)

def validate_groupby_func(name: str, args: Tuple[Any, ...], kwargs: Dict[str, Any], allowed: Optional[list[str]] = None) -> None:
    if allowed is None:
        allowed = []
    kwargs = set(kwargs) - set(allowed)
    if len(args) + len(kwargs) > 0:
        raise UnsupportedFunctionCall(f'numpy operations are not valid with groupby. Use .groupby(...).{name}() instead')

def validate_minmax_axis(axis: Optional[int], ndim: int = 1) -> None:
    if axis is None:
        return
    if axis >= ndim or (axis < 0 and ndim + axis < 0):
        raise ValueError(f'`axis` must be fewer than the number of dimensions ({ndim})')

_validation_funcs: Dict[str, CompatValidator] = {
    'median': validate_median,
    'mean': validate_mean,
    'min': validate_min,
    'max': validate_max,
    'sum': validate_sum,
    'prod': validate_prod
}

def validate_func(fname: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
    if fname not in _validation_funcs:
        return validate_stat_func(args, kwargs, fname=fname)
    validation_func = _validation_funcs[fname]
    return validation_func(args, kwargs)
