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

_TEST_MODE: bool = None
_TEST_RESULT: list[bool] = []
USE_NUMEXPR: bool = NUMEXPR_INSTALLED
_evaluate: callable = None
_where: callable = None
_ALLOWED_DTYPES: dict[str, set[str]] = {'evaluate': {'int64', 'int32', 'float64', 'float32', 'bool'}, 'where': {'int64', 'float64', 'bool'}}
_MIN_ELEMENTS: int = 1000000

def set_use_numexpr(v: bool = True) -> None:
    global USE_NUMEXPR
    if NUMEXPR_INSTALLED:
        USE_NUMEXPR = v
    global _evaluate, _where
    _evaluate = _evaluate_numexpr if USE_NUMEXPR else _evaluate_standard
    _where = _where_numexpr if USE_NUMEXPR else _where_standard

def set_numexpr_threads(n: int | None = None) -> None:
    if NUMEXPR_INSTALLED and USE_NUMEXPR:
        if n is None:
            n = ne.detect_number_of_cores()
        ne.set_num_threads(n)

def _evaluate_standard(
    op: operator.Add | operator.Sub | operator.Mul | operator.Truediv | operator.FloorDiv | operator.Pow | operator.Eq | operator.Ne | operator.Le | operator.Lt | operator.Ge | operator.Gt | operator.And | operator.Or | operator.Xor | roperator.RAdd | roperator.RSub | roperator.RMul | roperator.RTruediv | roperator.RFloorDiv | roperator.RPow | roperator.RAnd | roperator.ROr | roperator.RXor,
    op_str: str,
    left_op: np.ndarray,
    right_op: np.ndarray,
) -> np.ndarray:
    """
    Standard evaluation.
    """
    if _TEST_MODE:
        _store_test_result(False)
    return op(left_op, right_op)

def _can_use_numexpr(
    op: operator.Add | operator.Sub | operator.Mul | operator.Truediv | operator.FloorDiv | operator.Pow | operator.Eq | operator.Ne | operator.Le | operator.Lt | operator.Ge | operator.Gt | operator.And | operator.Or | operator.Xor | roperator.RAdd | roperator.RSub | roperator.RMul | roperator.RTruediv | roperator.RFloorDiv | roperator.RPow | roperator.RAnd | roperator.ROr | roperator.RXor | None,
    op_str: str,
    left_op: np.ndarray,
    right_op: np.ndarray,
    dtype_check: str,
) -> bool:
    """return left_op boolean if we WILL be using numexpr"""
    if op_str is not None:
        if left_op.size > _MIN_ELEMENTS:
            dtypes: set[str] = set()
            for o in [left_op, right_op]:
                if hasattr(o, 'dtype'):
                    dtypes |= {o.dtype.name}
            if not len(dtypes) or _ALLOWED_DTYPES[dtype_check] >= dtypes:
                return True
    return False

def _evaluate_numexpr(
    op: operator.Add | operator.Sub | operator.Mul | operator.Truediv | operator.FloorDiv | operator.Pow | operator.Eq | operator.Ne | operator.Le | operator.Lt | operator.Ge | operator.Gt | operator.And | operator.Or | operator.Xor | roperator.RAdd | roperator.RSub | roperator.RMul | roperator.RTruediv | roperator.RFloorDiv | roperator.RPow | roperator.RAnd | roperator.ROr | roperator.RXor | None,
    op_str: str,
    left_op: np.ndarray,
    right_op: np.ndarray,
) -> np.ndarray:
    result: np.ndarray | None = None
    if _can_use_numexpr(op, op_str, left_op, right_op, 'evaluate'):
        is_reversed: bool = op.__name__.strip('_').startswith('r')
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

_op_str_mapping: dict[operator.Add | operator.Sub | operator.Mul | operator.Truediv | operator.FloorDiv | operator.Pow | operator.Eq | operator.Ne | operator.Le | operator.Lt | operator.Ge | operator.Gt | operator.And | operator.Or | operator.Xor | roperator.RAdd | roperator.RSub | roperator.RMul | roperator.RTruediv | roperator.RFloorDiv | roperator.RPow | roperator.RAnd | roperator.ROr | roperator.RXor, str] = {
    operator.add: '+', roperator.radd: '+', operator.mul: '*', roperator.rmul: '*', operator.sub: '-', roperator.rsub: '-', operator.truediv: '/', roperator.rtruediv: '/', operator.floordiv: None, roperator.rfloordiv: None, operator.mod: None, roperator.rmod: None, operator.pow: '**', roperator.rpow: '**', operator.eq: '==', operator.ne: '!=', operator.le: '<=', operator.lt: '<', operator.ge: '>=', operator.gt: '>', operator.and_: '&', roperator.rand_: '&', operator.or_: '|', roperator.ror_: '|', operator.xor: '^', roperator.rxor: '^', divmod: None, roperator.rdivmod: None
}

def _where_standard(cond: np.ndarray, left_op: np.ndarray, right_op: np.ndarray) -> np.ndarray:
    return np.where(cond, left_op, right_op)

def _where_numexpr(cond: np.ndarray, left_op: np.ndarray, right_op: np.ndarray) -> np.ndarray:
    result: np.ndarray | None = None
    if _can_use_numexpr(None, 'where', left_op, right_op, 'where'):
        result = ne.evaluate('where(cond_value, a_value, b_value)', local_dict={'cond_value': cond, 'a_value': left_op, 'b_value': right_op}, casting='safe')
    if result is None:
        result = _where_standard(cond, left_op, right_op)
    return result

def set_test_mode(v: bool = True) -> None:
    """
    Keeps track of whether numexpr was used.

    Stores an additional ``True`` for every successful use of evaluate with
    numexpr since the last ``get_test_result``.
    """
    global _TEST_MODE, _TEST_RESULT
    _TEST_MODE = v
    _TEST_RESULT = []

def _store_test_result(used_numexpr: bool) -> None:
    if used_numexpr:
        _TEST_RESULT.append(used_numexpr)

def get_test_result() -> list[bool]:
    """
    Get test result and reset test_results.
    """
    global _TEST_RESULT
    res = _TEST_RESULT
    _TEST_RESULT = []
    return res

def _has_bool_dtype(x: np.ndarray | bool) -> bool:
    try:
        return x.dtype == bool
    except AttributeError:
        return isinstance(x, (bool, np.bool_))

_BOOL_OP_UNSUPPORTED: dict[str, str] = {'+': '|', '*': '&', '-': '^'}

def _bool_arith_fallback(op_str: str, left_op: np.ndarray, right_op: np.ndarray) -> bool:
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

def evaluate(
    op: operator.Add | operator.Sub | operator.Mul | operator.Truediv | operator.FloorDiv | operator.Pow | operator.Eq | operator.Ne | operator.Le | operator.Lt | operator.Ge | operator.Gt | operator.And | operator.Or | operator.Xor | roperator.RAdd | roperator.RSub | roperator.RMul | roperator.RTruediv | roperator.RFloorDiv | roperator.RPow | roperator.RAnd | roperator.ROr | roperator.RXor,
    left_op: np.ndarray,
    right_op: np.ndarray,
    use_numexpr: bool = True,
) -> np.ndarray:
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

def where(
    cond: np.ndarray,
    left_op: np.ndarray,
    right_op: np.ndarray,
    use_numexpr: bool = True,
) -> np.ndarray:
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
