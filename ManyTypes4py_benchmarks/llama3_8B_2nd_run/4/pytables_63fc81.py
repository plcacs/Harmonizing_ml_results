from __future__ import annotations
import ast
from decimal import Decimal, InvalidOperation
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, Type
import numpy as np
from pandas._libs.tslibs import Timedelta, Timestamp
from pandas.errors import UndefinedVariableError
from pandas.core.dtypes.common import is_list_like
import pandas.core.common as com
from pandas.core.computation import expr, ops, scope as _scope
from pandas.core.computation.common import ensure_decoded
from pandas.core.computation.expr import BaseExprVisitor
from pandas.core.computation.ops import is_term
from pandas.core.construction import extract_array
from pandas.io.formats.printing import pprint_thing, pprint_thing_encoded
if TYPE_CHECKING:
    from pandas._typing import Self, npt

class PyTablesScope(_scope.Scope):
    __slots__: tuple[str, ...] = ('queryables',)

    def __init__(self, level: int, global_dict: Any = None, local_dict: Any = None, queryables: dict[str, Any] = None) -> None:
        super().__init__(level + 1, global_dict=global_dict, local_dict=local_dict)
        self.queryables: dict[str, Any] = queryables or {}

class Term(ops.Term):
    ...

class Constant(Term):
    ...

class BinOp(ops.BinOp):
    ...

class FilterBinOp(BinOp):
    ...

class JointFilterBinOp(FilterBinOp):
    ...

class ConditionBinOp(BinOp):
    ...

class JointConditionBinOp(ConditionBinOp):
    ...

class UnaryOp(ops.UnaryOp):
    ...

class PyTablesExprVisitor(BaseExprVisitor):
    const_type: Type[Constant]
    term_type: Type[Term]

    def __init__(self, env: PyTablesScope, engine: str, parser: str, **kwargs: Any) -> None:
        ...

    def visit_UnaryOp(self, node: ast.UnaryOp, **kwargs: Any) -> UnaryOp:
        ...

    def visit_Index(self, node: ast.Index, **kwargs: Any) -> Term:
        ...

    def visit_Assign(self, node: ast.Assign, **kwargs: Any) -> Term:
        ...

    def visit_Subscript(self, node: ast.Subscript, **kwargs: Any) -> Term:
        ...

    def visit_Attribute(self, node: ast.Attribute, **kwargs: Any) -> Term:
        ...

    def translate_In(self, op: ast.AST) -> ast.AST:
        ...

    def _rewrite_membership_op(self, node: ast.AST, left: Term, right: Term) -> tuple[ast.AST, ...]:
        ...

def _validate_where(w: Any) -> Any:
    ...

class PyTablesExpr(expr.Expr):
    ...

    def __init__(self, where: Any, queryables: dict[str, Any] = None, encoding: str = None, scope_level: int = 0) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def evaluate(self) -> tuple[ConditionBinOp, FilterBinOp]:
        ...

class TermValue:
    ...

    def tostring(self, encoding: str) -> str:
        ...

def maybe_expression(s: str) -> bool:
    ...
