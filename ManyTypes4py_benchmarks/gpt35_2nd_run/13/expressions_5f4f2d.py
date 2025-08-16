from __future__ import annotations
import operator
from typing import TYPE_CHECKING, Union
import numpy as np

_TEST_MODE: Union[None, bool] = None
_TEST_RESULT: list[bool] = []
USE_NUMEXPR: bool = NUMEXPR_INSTALLED
_evaluate: Union[None, FuncType] = None
_where: Union[None, FuncType] = None
_ALLOWED_DTYPES: dict[str, set[str]] = {'evaluate': {'int64', 'int32', 'float64', 'float32', 'bool'}, 'where': {'int64', 'float64', 'bool'}}
_MIN_ELEMENTS: int = 1000000

def set_use_numexpr(v: bool = True) -> None:
    global USE_NUMEXPR
    global _evaluate, _where

def set_numexpr_threads(n: Union[None, int] = None) -> None:
    pass

def _evaluate_standard(op: FuncType, op_str: str, left_op: np.ndarray, right_op: np.ndarray) -> np.ndarray:
    pass

def _can_use_numexpr(op: FuncType, op_str: str, left_op: np.ndarray, right_op: np.ndarray, dtype_check: str) -> bool:
    pass

def _evaluate_numexpr(op: FuncType, op_str: str, left_op: np.ndarray, right_op: np.ndarray) -> np.ndarray:
    pass

_op_str_mapping: dict[FuncType, Union[str, None]] = {operator.add: '+', operator.mul: '*', operator.sub: '-', operator.truediv: '/', operator.floordiv: None, operator.mod: None, operator.pow: '**', operator.eq: '==', operator.ne: '!=', operator.le: '<=', operator.lt: '<', operator.ge: '>=', operator.gt: '>', operator.and_: '&', operator.or_: '|', operator.xor: '^', divmod: None}

def _where_standard(cond: np.ndarray, left_op: np.ndarray, right_op: np.ndarray) -> np.ndarray:
    pass

def _where_numexpr(cond: np.ndarray, left_op: np.ndarray, right_op: np.ndarray) -> np.ndarray:
    pass

def evaluate(op: FuncType, left_op: np.ndarray, right_op: np.ndarray, use_numexpr: bool = True) -> np.ndarray:
    pass

def where(cond: np.ndarray, left_op: np.ndarray, right_op: np.ndarray, use_numexpr: bool = True) -> np.ndarray:
    pass

def set_test_mode(v: bool = True) -> None:
    global _TEST_MODE, _TEST_RESULT

def _store_test_result(used_numexpr: bool) -> None:
    pass

def get_test_result() -> list[bool]:
    pass
