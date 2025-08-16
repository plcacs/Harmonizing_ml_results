#!/usr/bin/env python3
"""
Expressions
-----------

Offer fast expression evaluation through numexpr

"""

from __future__ import annotations

import operator
from typing import Any, Callable, List, Optional, Set
import warnings

import numpy as np

from pandas._config import get_option
from pandas.util._exceptions import find_stack_level

from pandas.core import roperator
from pandas.core.computation.check import NUMEXPR_INSTALLED

if NUMEXPR_INSTALLED:
    import numexpr as ne

# type alias for a function taking two arguments and returning Any
FuncType = Callable[[Any, Any], Any]

_TEST_MODE: Optional[bool] = None
_TEST_RESULT: List[bool] = []
USE_NUMEXPR: bool = NUMEXPR_INSTALLED
_evaluate: Optional[Callable[[Callable[[Any, Any], Any], Optional[str], Any, Any], Any]] = None
_where: Optional[Callable[[Any, Any, Any], Any]] = None

# the set of dtypes that we will allow pass to numexpr
_ALLOWED_DTYPES: dict[str, Set[str]] = {
    "evaluate": {"int64", "int32", "float64", "float32", "bool"},
    "where": {"int64", "float64", "bool"},
}

# the minimum prod shape that we will use numexpr
_MIN_ELEMENTS: int = 1_000_000


def set_use_numexpr(v: bool = True) -> None:
    global USE_NUMEXPR, _evaluate, _where
    # set/unset to use numexpr
    if NUMEXPR_INSTALLED:
        USE_NUMEXPR = v

    # choose what we are going to do
    _evaluate = _evaluate_numexpr if USE_NUMEXPR else _evaluate_standard
    _where = _where_numexpr if USE_NUMEXPR else _where_standard


def set_numexpr_threads(n: Optional[int] = None) -> None:
    # if we are using numexpr, set the threads to n
    # otherwise reset
    if NUMEXPR_INSTALLED and USE_NUMEXPR:
        if n is None:
            n = ne.detect_number_of_cores()
        ne.set_num_threads(n)


def _evaluate_standard(op: Callable[[Any, Any], Any],
                       op_str: Optional[str],
                       left_op: Any,
                       right_op: Any) -> Any:
    """
    Standard evaluation.
    """
    if _TEST_MODE:
        _store_test_result(False)
    return op(left_op, right_op)


def _can_use_numexpr(op: Optional[Callable[[Any, Any], Any]],
                     op_str: Optional[str],
                     left_op: Any,
                     right_op: Any,
                     dtype_check: str) -> bool:
    """return left_op boolean if we WILL be using numexpr"""
    if op_str is not None:
        # required min elements (otherwise we are adding overhead)
        if left_op.size > _MIN_ELEMENTS:
            # check for dtype compatibility
            dtypes: Set[str] = set()
            for o in [left_op, right_op]:
                # ndarray and Series Case
                if hasattr(o, "dtype"):
                    dtypes |= {o.dtype.name}

            # allowed are a superset
            if not len(dtypes) or _ALLOWED_DTYPES[dtype_check] >= dtypes:
                return True

    return False


def _evaluate_numexpr(op: Callable[[Any, Any], Any],
                      op_str: Optional[str],
                      left_op: Any,
                      right_op: Any) -> Any:
    result: Any = None

    if _can_use_numexpr(op, op_str, left_op, right_op, "evaluate"):
        is_reversed: bool = op.__name__.strip("_").startswith("r")
        if is_reversed:
            # we were originally called by a reversed op method
            left_op, right_op = right_op, left_op

        left_value: Any = left_op
        right_value: Any = right_op

        try:
            result = ne.evaluate(
                f"left_value {op_str} right_value",
                local_dict={"left_value": left_value, "right_value": right_value},
                casting="safe",
            )
        except TypeError:
            # numexpr raises eg for array ** array with integers
            # (https://github.com/pydata/numexpr/issues/379)
            pass
        except NotImplementedError:
            if _bool_arith_fallback(op_str, left_op, right_op):
                pass
            else:
                raise

        if is_reversed:
            # reverse order to original for fallback
            left_op, right_op = right_op, left_op

    if _TEST_MODE:
        _store_test_result(result is not None)

    if result is None:
        result = _evaluate_standard(op, op_str, left_op, right_op)

    return result


_op_str_mapping: dict[Callable[[Any, Any], Any], Optional[str]] = {
    operator.add: "+",
    roperator.radd: "+",
    operator.mul: "*",
    roperator.rmul: "*",
    operator.sub: "-",
    roperator.rsub: "-",
    operator.truediv: "/",
    roperator.rtruediv: "/",
    # floordiv not supported by numexpr 2.x
    operator.floordiv: None,
    roperator.rfloordiv: None,
    # we require Python semantics for mod of negative for backwards compatibility
    # see https://github.com/pydata/numexpr/issues/365
    # so sticking with unaccelerated for now GH#36552
    operator.mod: None,
    roperator.rmod: None,
    operator.pow: "**",
    roperator.rpow: "**",
    operator.eq: "==",
    operator.ne: "!=",
    operator.le: "<=",
    operator.lt: "<",
    operator.ge: ">=",
    operator.gt: ">",
    operator.and_: "&",
    roperator.rand_: "&",
    operator.or_: "|",
    roperator.ror_: "|",
    operator.xor: "^",
    roperator.rxor: "^",
    divmod: None,
    roperator.rdivmod: None,
}


def _where_standard(cond: Any, left_op: Any, right_op: Any) -> Any:
    # Caller is responsible for extracting ndarray if necessary
    return np.where(cond, left_op, right_op)


def _where_numexpr(cond: Any, left_op: Any, right_op: Any) -> Any:
    # Caller is responsible for extracting ndarray if necessary
    result: Any = None

    if _can_use_numexpr(None, "where", left_op, right_op, "where"):
        result = ne.evaluate(
            "where(cond_value, a_value, b_value)",
            local_dict={"cond_value": cond, "a_value": left_op, "b_value": right_op},
            casting="safe",
        )

    if result is None:
        result = _where_standard(cond, left_op, right_op)

    return result


# turn myself on
set_use_numexpr(get_option("compute.use_numexpr"))


def _has_bool_dtype(x: Any) -> bool:
    try:
        return x.dtype == bool
    except AttributeError:
        return isinstance(x, (bool, np.bool_))


_BOOL_OP_UNSUPPORTED: dict[str, str] = {"+": "|", "*": "&", "-": "^"}


def _bool_arith_fallback(op_str: str, left_op: Any, right_op: Any) -> bool:
    """
    Check if we should fallback to the python `_evaluate_standard` in case
    of an unsupported operation by numexpr, which is the case for some
    boolean ops.
    """
    if _has_bool_dtype(left_op) and _has_bool_dtype(right_op):
        if op_str in _BOOL_OP_UNSUPPORTED:
            warnings.warn(
                f"evaluating in Python space because the {op_str!r} "
                "operator is not supported by numexpr for the bool dtype, "
                f"use {_BOOL_OP_UNSUPPORTED[op_str]!r} instead.",
                stacklevel=find_stack_level(),
            )
            return True
    return False


def evaluate(op: Callable[[Any, Any], Any],
             left_op: Any,
             right_op: Any,
             use_numexpr: bool = True) -> Any:
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
    op_str: Optional[str] = _op_str_mapping[op]
    if op_str is not None:
        if use_numexpr:
            # error: "None" not callable
            return _evaluate(op, op_str, left_op, right_op)  # type: ignore[misc]
    return _evaluate_standard(op, op_str, left_op, right_op)


def where(cond: Any,
          left_op: Any,
          right_op: Any,
          use_numexpr: bool = True) -> Any:
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


def get_test_result() -> List[bool]:
    """
    Get test result and reset test_results.
    """
    global _TEST_RESULT
    res: List[bool] = _TEST_RESULT
    _TEST_RESULT = []
    return res
