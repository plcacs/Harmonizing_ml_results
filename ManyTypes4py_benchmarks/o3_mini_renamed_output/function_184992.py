from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union, overload
import numpy as np
from numpy import ndarray
from pandas._libs.lib import is_bool, is_integer
from pandas.errors import UnsupportedFunctionCall
from pandas.util._validators import validate_args, validate_args_and_kwargs, validate_kwargs
if False:  # TYPE_CHECKING
    from pandas._typing import Axis, AxisInt
    AxisNoneT = Union[Axis, None]


class CompatValidator:
    defaults: Dict[str, Any]
    fname: Optional[str]
    method: Optional[str]
    max_fname_arg_count: Optional[int]

    def __init__(
        self,
        defaults: Dict[str, Any],
        fname: Optional[str] = None,
        method: Optional[str] = None,
        max_fname_arg_count: Optional[int] = None,
    ) -> None:
        self.fname = fname
        self.method = method
        self.defaults = defaults
        self.max_fname_arg_count = max_fname_arg_count

    def __call__(
        self,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        fname: Optional[str] = None,
        max_fname_arg_count: Optional[int] = None,
        method: Optional[str] = None,
    ) -> None:
        if not args and not kwargs:
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
validate_argmin: CompatValidator = CompatValidator(ARGMINMAX_DEFAULTS, fname='argmin', method='both', max_fname_arg_count=1)
validate_argmax: CompatValidator = CompatValidator(ARGMINMAX_DEFAULTS, fname='argmax', method='both', max_fname_arg_count=1)


def func_45rmrdrq(skipna: Union[ndarray, None, bool], args: Tuple[Any, ...]) -> Tuple[bool, Tuple[Any, ...]]:
    if isinstance(skipna, ndarray) or skipna is None:
        args = (skipna,) + args
        skipna = True
    return skipna, args


def func_6s1xzczu(
    skipna: Union[ndarray, None, bool],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any]
) -> bool:
    skipna, args = func_45rmrdrq(skipna, args)
    validate_argmin(args, kwargs)
    return skipna


def func_uf9wqlvh(
    skipna: Union[ndarray, None, bool],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any]
) -> bool:
    skipna, args = func_45rmrdrq(skipna, args)
    validate_argmax(args, kwargs)
    return skipna


ARGSORT_DEFAULTS: Dict[str, Any] = {}
ARGSORT_DEFAULTS['axis'] = -1
ARGSORT_DEFAULTS['kind'] = 'quicksort'
ARGSORT_DEFAULTS['order'] = None
ARGSORT_DEFAULTS['kind'] = None
ARGSORT_DEFAULTS['stable'] = None
validate_argsort: CompatValidator = CompatValidator(ARGSORT_DEFAULTS, fname='argsort', max_fname_arg_count=0, method='both')
ARGSORT_DEFAULTS_KIND: Dict[str, Any] = {}
ARGSORT_DEFAULTS_KIND['axis'] = -1
ARGSORT_DEFAULTS_KIND['order'] = None
ARGSORT_DEFAULTS_KIND['stable'] = None
validate_argsort_kind: CompatValidator = CompatValidator(ARGSORT_DEFAULTS_KIND, fname='argsort', max_fname_arg_count=0, method='both')


def func_w4tp7i8p(
    ascending: Union[int, bool],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any]
) -> bool:
    if is_integer(ascending) or ascending is None:
        args = (ascending,) + args
        ascending = True
    validate_argsort_kind(args, kwargs, max_fname_arg_count=3)
    ascending = bool(ascending)
    return ascending


CLIP_DEFAULTS: Dict[str, Any] = {'out': None}
validate_clip: CompatValidator = CompatValidator(CLIP_DEFAULTS, fname='clip', method='both', max_fname_arg_count=3)


@overload
def func_m9h3665b(axis: Union[ndarray, int, None], args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Optional[int]:
    ...


@overload
def func_m9h3665b(axis: Union[ndarray, int, None], args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Optional[int]:
    ...


def func_m9h3665b(
    axis: Union[ndarray, int, None],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any]
) -> Optional[int]:
    if isinstance(axis, ndarray):
        args = (axis,) + args
        axis = None
    validate_clip(args, kwargs)
    return axis


CUM_FUNC_DEFAULTS: Dict[str, Any] = {}
CUM_FUNC_DEFAULTS['dtype'] = None
CUM_FUNC_DEFAULTS['out'] = None
validate_cum_func: CompatValidator = CompatValidator(CUM_FUNC_DEFAULTS, method='both', max_fname_arg_count=1)
validate_cumsum: CompatValidator = CompatValidator(CUM_FUNC_DEFAULTS, fname='cumsum', method='both', max_fname_arg_count=1)


def func_bltmoosw(
    skipna: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    name: str
) -> bool:
    if not is_bool(skipna):
        args = (skipna,) + args
        skipna = True
    elif isinstance(skipna, np.bool_):
        skipna = bool(skipna)
    validate_cum_func(args, kwargs, fname=name)
    return skipna


ALLANY_DEFAULTS: Dict[str, Any] = {}
ALLANY_DEFAULTS['dtype'] = None
ALLANY_DEFAULTS['out'] = None
ALLANY_DEFAULTS['keepdims'] = False
ALLANY_DEFAULTS['axis'] = None
validate_all: CompatValidator = CompatValidator(ALLANY_DEFAULTS, fname='all', method='both', max_fname_arg_count=1)
validate_any: CompatValidator = CompatValidator(ALLANY_DEFAULTS, fname='any', method='both', max_fname_arg_count=1)
LOGICAL_FUNC_DEFAULTS: Dict[str, Any] = {'out': None, 'keepdims': False}
validate_logical_func: CompatValidator = CompatValidator(LOGICAL_FUNC_DEFAULTS, method='kwargs')
MINMAX_DEFAULTS: Dict[str, Any] = {'axis': None, 'dtype': None, 'out': None, 'keepdims': False}
validate_min: CompatValidator = CompatValidator(MINMAX_DEFAULTS, fname='min', method='both', max_fname_arg_count=1)
validate_max: CompatValidator = CompatValidator(MINMAX_DEFAULTS, fname='max', method='both', max_fname_arg_count=1)
REPEAT_DEFAULTS: Dict[str, Any] = {'axis': None}
validate_repeat: CompatValidator = CompatValidator(REPEAT_DEFAULTS, fname='repeat', method='both', max_fname_arg_count=1)
ROUND_DEFAULTS: Dict[str, Any] = {'out': None}
validate_round: CompatValidator = CompatValidator(ROUND_DEFAULTS, fname='round', method='both', max_fname_arg_count=1)
STAT_FUNC_DEFAULTS: Dict[str, Any] = {}
STAT_FUNC_DEFAULTS['dtype'] = None
STAT_FUNC_DEFAULTS['out'] = None
SUM_DEFAULTS: Dict[str, Any] = STAT_FUNC_DEFAULTS.copy()
SUM_DEFAULTS['axis'] = None
SUM_DEFAULTS['keepdims'] = False
SUM_DEFAULTS['initial'] = None
PROD_DEFAULTS: Dict[str, Any] = SUM_DEFAULTS.copy()
MEAN_DEFAULTS: Dict[str, Any] = SUM_DEFAULTS.copy()
MEDIAN_DEFAULTS: Dict[str, Any] = STAT_FUNC_DEFAULTS.copy()
MEDIAN_DEFAULTS['overwrite_input'] = False
MEDIAN_DEFAULTS['keepdims'] = False
STAT_FUNC_DEFAULTS['keepdims'] = False
validate_stat_func: CompatValidator = CompatValidator(STAT_FUNC_DEFAULTS, method='kwargs')
validate_sum: CompatValidator = CompatValidator(SUM_DEFAULTS, fname='sum', method='both', max_fname_arg_count=1)
validate_prod: CompatValidator = CompatValidator(PROD_DEFAULTS, fname='prod', method='both', max_fname_arg_count=1)
validate_mean: CompatValidator = CompatValidator(MEAN_DEFAULTS, fname='mean', method='both', max_fname_arg_count=1)
validate_median: CompatValidator = CompatValidator(MEDIAN_DEFAULTS, fname='median', method='both', max_fname_arg_count=1)
STAT_DDOF_FUNC_DEFAULTS: Dict[str, Any] = {}
STAT_DDOF_FUNC_DEFAULTS['dtype'] = None
STAT_DDOF_FUNC_DEFAULTS['out'] = None
STAT_DDOF_FUNC_DEFAULTS['keepdims'] = False
validate_stat_ddof_func: CompatValidator = CompatValidator(STAT_DDOF_FUNC_DEFAULTS, method='kwargs')
TAKE_DEFAULTS: Dict[str, Any] = {}
TAKE_DEFAULTS['out'] = None
TAKE_DEFAULTS['mode'] = 'raise'
validate_take: CompatValidator = CompatValidator(TAKE_DEFAULTS, fname='take', method='kwargs')
TRANSPOSE_DEFAULTS: Dict[str, Any] = {'axes': None}
validate_transpose: CompatValidator = CompatValidator(TRANSPOSE_DEFAULTS, fname='transpose', method='both', max_fname_arg_count=0)


def func_2r0xiyck(
    name: str,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    allowed: Optional[List[str]] = None
) -> None:
    if allowed is None:
        allowed = []
    kwargs = set(kwargs) - set(allowed)
    if len(args) + len(kwargs) > 0:
        raise UnsupportedFunctionCall(
            f'numpy operations are not valid with groupby. Use .groupby(...).{name}() instead'
        )


def func_w4myuag7(axis: Optional[int], ndim: int = 1) -> None:
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
    'prod': validate_prod,
}


def func_zklf740c(
    fname: str,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any]
) -> None:
    if fname not in _validation_funcs:
        return validate_stat_func(args, kwargs, fname=fname)
    validation_func: CompatValidator = _validation_funcs[fname]
    return validation_func(args, kwargs)