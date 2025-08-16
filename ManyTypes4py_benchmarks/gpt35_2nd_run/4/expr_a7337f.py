from __future__ import annotations
import ast
from functools import partial, reduce
from keyword import iskeyword
import tokenize
from typing import TYPE_CHECKING, ClassVar, TypeVar
import numpy as np
from pandas.errors import UndefinedVariableError
from pandas.core.dtypes.common import is_string_dtype
import pandas.core.common as com
from pandas.core.computation.ops import ARITH_OPS_SYMS, BOOL_OPS_SYMS, CMP_OPS_SYMS, LOCAL_TAG, MATHOPS, REDUCTIONS, UNARY_OPS_SYMS, BinOp, Constant, FuncNode, Op, Term, UnaryOp, is_term
from pandas.core.computation.parsing import clean_backtick_quoted_toks, tokenize_string
from pandas.core.computation.scope import Scope
from pandas.io.formats import printing
if TYPE_CHECKING:
    from collections.abc import Callable

def _rewrite_assign(tok: tuple[int, str]) -> tuple[int, str]:
    ...

def _replace_booleans(tok: tuple[int, str]) -> tuple[int, str]:
    ...

def _replace_locals(tok: tuple[int, str]) -> tuple[int, str]:
    ...

def _compose2(f: Callable, g: Callable) -> Callable:
    ...

def _compose(*funcs: Callable) -> Callable:
    ...

def _preparse(source: str, f: Callable = _compose(_replace_locals, _replace_booleans, _rewrite_assign, clean_backtick_quoted_toks)) -> str:
    ...

def _is_type(t: type) -> Callable:
    ...

def _filter_nodes(superclass: type, all_nodes: frozenset) -> frozenset:
    ...

def _node_not_implemented(node_name: str) -> Callable:
    ...

_T = TypeVar('_T')

def disallow(nodes: frozenset) -> Callable:
    ...

def _op_maker(op_class: type, op_symbol: str) -> Callable:
    ...

def add_ops(op_classes: dict) -> Callable:
    ...

class BaseExprVisitor(ast.NodeVisitor):
    ...

def _try_visit_binop(bop: Union[Op, Term]) -> Union[Op, Term]:
    ...

class PandasExprVisitor(BaseExprVisitor):
    ...

class PythonExprVisitor(BaseExprVisitor):
    ...

class Expr:
    ...

PARSERS: dict[str, type] = {'python': PythonExprVisitor, 'pandas': PandasExprVisitor}
