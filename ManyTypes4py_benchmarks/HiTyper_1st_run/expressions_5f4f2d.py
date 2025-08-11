"""
Expressions
-----------

Offer fast expression evaluation through numexpr

"""
from __future__ import annotations
import operator
from typing import TYPE_CHECKING
import warnings
import numpy as np
from pandas._config import get_option
from pandas.util._exceptions import find_stack_level
from pandas.core import roperator
from pandas.core.computation.check import NUMEXPR_INSTALLED
if NUMEXPR_INSTALLED:
    import numexpr as ne
if TYPE_CHECKING:
    from pandas._typing import FuncType
_TEST_MODE = None
_TEST_RESULT = []
USE_NUMEXPR = NUMEXPR_INSTALLED
_evaluate = None
_where = None
_ALLOWED_DTYPES = {'evaluate': {'int64', 'int32', 'float64', 'float32', 'bool'}, 'where': {'int64', 'float64', 'bool'}}
_MIN_ELEMENTS = 1000000

def set_use_numexpr(v: bool=True) -> None:
    global USE_NUMEXPR
    if NUMEXPR_INSTALLED:
        USE_NUMEXPR = v
    global _evaluate, _where
    _evaluate = _evaluate_numexpr if USE_NUMEXPR else _evaluate_standard
    _where = _where_numexpr if USE_NUMEXPR else _where_standard

def set_numexpr_threads(n: Union[None, int, typing.Iterable[typing.Optional[typing.Any]]]=None) -> None:
    if NUMEXPR_INSTALLED and USE_NUMEXPR:
        if n is None:
            n = ne.detect_number_of_cores()
        ne.set_num_threads(n)

def _evaluate_standard(op: Union[str, typing.Type, None, bytes], op_str: Union[str, list[typing.Union[int,str]]], left_op: Union[str, typing.Type, None, bytes], right_op: Union[str, typing.Type, None, bytes]) -> Union[bool, str]:
    """
    Standard evaluation.
    """
    if _TEST_MODE:
        _store_test_result(False)
    return op(left_op, right_op)

def _can_use_numexpr(op: Union[bool, typing.Iterable[str], core.DependencyContainer], op_str: Union[bool, str, None], left_op: bool, right_op: bool, dtype_check: Union[bool, static_frame.core.util.IndexConstructor, typing.AbstractSet]) -> bool:
    """return left_op boolean if we WILL be using numexpr"""
    if op_str is not None:
        if left_op.size > _MIN_ELEMENTS:
            dtypes = set()
            for o in [left_op, right_op]:
                if hasattr(o, 'dtype'):
                    dtypes |= {o.dtype.name}
            if not len(dtypes) or _ALLOWED_DTYPES[dtype_check] >= dtypes:
                return True
    return False

def _evaluate_numexpr(op: str, op_str: Union[str, typing.Type], left_op: Union[str, typing.Type, typing.Callable[str, str]], right_op: Union[str, typing.Callable[str, str]]) -> Union[dict, None, list, list[None]]:
    result = None
    if _can_use_numexpr(op, op_str, left_op, right_op, 'evaluate'):
        is_reversed = op.__name__.strip('_').startswith('r')
        if is_reversed:
            left_op, right_op = (right_op, left_op)
        left_value = left_op
        right_value = right_op
        try:
            result = ne.evaluate(f'left_value {op_str} right_value', local_dict={'left_value': left_value, 'right_value': right_value}, casting='safe')
        except TypeError:
            pass
        except NotImplementedError:
            if _bool_arith_fallback(op_str, left_op, right_op):
                pass
            else:
                raise
        if is_reversed:
            left_op, right_op = (right_op, left_op)
    if _TEST_MODE:
        _store_test_result(result is not None)
    if result is None:
        result = _evaluate_standard(op, op_str, left_op, right_op)
    return result
_op_str_mapping = {operator.add: '+', roperator.radd: '+', operator.mul: '*', roperator.rmul: '*', operator.sub: '-', roperator.rsub: '-', operator.truediv: '/', roperator.rtruediv: '/', operator.floordiv: None, roperator.rfloordiv: None, operator.mod: None, roperator.rmod: None, operator.pow: '**', roperator.rpow: '**', operator.eq: '==', operator.ne: '!=', operator.le: '<=', operator.lt: '<', operator.ge: '>=', operator.gt: '>', operator.and_: '&', roperator.rand_: '&', operator.or_: '|', roperator.ror_: '|', operator.xor: '^', roperator.rxor: '^', divmod: None, roperator.rdivmod: None}

def _where_standard(cond: Union[pandas._ArrayLike, typing.Callable, Series], left_op: Union[pandas._ArrayLike, typing.Callable, Series], right_op: Union[pandas._ArrayLike, typing.Callable, Series]):
    return np.where(cond, left_op, right_op)

def _where_numexpr(cond: Union[bool, str, None], left_op: Union[str, None], right_op: Union[str, None]) -> Union[list, list[typing.Optional[bool]], list[None]]:
    result = None
    if _can_use_numexpr(None, 'where', left_op, right_op, 'where'):
        result = ne.evaluate('where(cond_value, a_value, b_value)', local_dict={'cond_value': cond, 'a_value': left_op, 'b_value': right_op}, casting='safe')
    if result is None:
        result = _where_standard(cond, left_op, right_op)
    return result
set_use_numexpr(get_option('compute.use_numexpr'))

def _has_bool_dtype(x: Union[numpy.ndarray, None]) -> bool:
    try:
        return x.dtype == bool
    except AttributeError:
        return isinstance(x, (bool, np.bool_))
_BOOL_OP_UNSUPPORTED = {'+': '|', '*': '&', '-': '^'}

def _bool_arith_fallback(op_str: Any, left_op: bool, right_op: bool) -> bool:
    """
    Check if we should fallback to the python `_evaluate_standard` in case
    of an unsupported operation by numexpr, which is the case for some
    boolean ops.
    """
    if _has_bool_dtype(left_op) and _has_bool_dtype(right_op):
        if op_str in _BOOL_OP_UNSUPPORTED:
            warnings.warn(f'evaluating in Python space because the {op_str!r} operator is not supported by numexpr for the bool dtype, use {_BOOL_OP_UNSUPPORTED[op_str]!r} instead.', stacklevel=find_stack_level())
            return True
    return False

def evaluate(op: Union[str, static_frame.core.util.UFunc, bool], left_op: Union[bool, typing.Type, None], right_op: Union[bool, typing.Type, None], use_numexpr: bool=True) -> Union[dict[str, str], str, list[str]]:
    """
    Evaluate and return the expression of the op on left_op and right_op.

    Parameters
    ----------
    op : the actual operand
    left_op : left operand
    right_op : right operand
    use_numexpr : bool, default True
        Whether to try to use numexpr.
    """
    op_str = _op_str_mapping[op]
    if op_str is not None:
        if use_numexpr:
            return _evaluate(op, op_str, left_op, right_op)
    return _evaluate_standard(op, op_str, left_op, right_op)

def where(cond: bool, left_op: bool, right_op: bool, use_numexpr: bool=True):
    """
    Evaluate the where condition cond on left_op and right_op.

    Parameters
    ----------
    cond : np.ndarray[bool]
    left_op : return if cond is True
    right_op : return if cond is False
    use_numexpr : bool, default True
        Whether to try to use numexpr.
    """
    assert _where is not None
    if use_numexpr:
        return _where(cond, left_op, right_op)
    else:
        return _where_standard(cond, left_op, right_op)

def set_test_mode(v: bool=True) -> None:
    """
    Keeps track of whether numexpr was used.

    Stores an additional ``True`` for every successful use of evaluate with
    numexpr since the last ``get_test_result``.
    """
    global _TEST_MODE, _TEST_RESULT
    _TEST_MODE = v
    _TEST_RESULT = []

def _store_test_result(used_numexpr: Union[bool, typing.Iterable[int]]) -> None:
    if used_numexpr:
        _TEST_RESULT.append(used_numexpr)

def get_test_result() -> Union[list, list[tuple[typing.Any]], dict[str, typing.Any]]:
    """
    Get test result and reset test_results.
    """
    global _TEST_RESULT
    res = _TEST_RESULT
    _TEST_RESULT = []
    return res