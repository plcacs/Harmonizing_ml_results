```python
from __future__ import annotations

import ast
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Frozen, TypeVar

import numpy as np

from pandas.core.computation.ops import BinOp, Constant, FuncNode, Op, Term, UnaryOp
from pandas.core.computation.scope import Scope

if TYPE_CHECKING:
    from collections.abc import Callable as CallableABC

_T = TypeVar('_T')

def _rewrite_assign(tok: tuple[int, str]) -> tuple[int, str]: ...
def _replace_booleans(tok: tuple[int, str]) -> tuple[int, str]: ...
def _replace_locals(tok: tuple[int, str]) -> tuple[int, str]: ...
def _compose2(f: Callable[..., Any], g: Callable[..., Any]) -> Callable[..., Any]: ...
def _compose(*funcs: Callable[..., Any]) -> Callable[..., Any]: ...
def _preparse(
    source: str,
    f: Callable[[tuple[int, str]], tuple[int, str]] = ...,
) -> str: ...
def _is_type(t: type) -> Callable[[Any], bool]: ...

_is_list: Callable[[Any], bool]
_is_str: Callable[[Any], bool]
_all_nodes: frozenset[type[ast.AST]]
_all_node_names: frozenset[str]
_mod_nodes: frozenset[str]
_stmt_nodes: frozenset[str]
_expr_nodes: frozenset[str]
_expr_context_nodes: frozenset[str]
_boolop_nodes: frozenset[str]
_operator_nodes: frozenset[str]
_unary_op_nodes: frozenset[str]
_cmp_op_nodes: frozenset[str]
_comprehension_nodes: frozenset[str]
_handler_nodes: frozenset[str]
_arguments_nodes: frozenset[str]
_keyword_nodes: frozenset[str]
_alias_nodes: frozenset[str]
_hacked_nodes: frozenset[str]
_unsupported_expr_nodes: frozenset[str]
_unsupported_nodes: frozenset[str]
_base_supported_nodes: frozenset[str]
intersection: frozenset[str]

def _filter_nodes(superclass: type[ast.AST], all_nodes: frozenset[type[ast.AST]] = ...) -> frozenset[str]: ...
def _node_not_implemented(node_name: str) -> Callable[..., Any]: ...
def disallow(nodes: frozenset[str]) -> Callable[[type], type]: ...
def _op_maker(op_class: type[Op], op_symbol: str) -> Callable[..., Callable[..., Any]]: ...
def add_ops(op_classes: dict[str, type[Op]]) -> Callable[[type], type]: ...

_op_classes: dict[str, type[Op]]
_python_not_supported: frozenset[str]
_numexpr_supported_calls: frozenset[str]

class BaseExprVisitor(ast.NodeVisitor):
    const_type: type[Constant]
    term_type: type[Term]
    binary_ops: tuple[str, ...]
    binary_op_nodes: tuple[str, ...]
    binary_op_nodes_map: dict[str, str]
    unary_ops: tuple[str, ...]
    unary_op_nodes: tuple[str, ...]
    unary_op_nodes_map: dict[str, str]
    rewrite_map: dict[type[ast.cmpop], type[ast.cmpop]]
    env: Scope
    engine: str
    parser: str
    preparser: Callable[[str], str]
    assigner: str | None

    def __init__(
        self,
        env: Scope,
        engine: str,
        parser: str,
        preparser: Callable[[str], str] = ...,
    ) -> None: ...
    def visit(self, node: ast.AST | str, **kwargs: Any) -> Any: ...
    def visit_Module(self, node: ast.Module, **kwargs: Any) -> Any: ...
    def visit_Expr(self, node: ast.Expr, **kwargs: Any) -> Any: ...
    def _rewrite_membership_op(
        self,
        node: ast.Compare,
        left: Any,
        right: Any,
    ) -> tuple[Callable[..., Any], type[ast.cmpop], Any, Any]: ...
    def _maybe_transform_eq_ne(
        self,
        node: ast.Compare,
        left: Any = ...,
        right: Any = ...,
    ) -> tuple[Callable[..., Any], type[ast.cmpop], Any, Any]: ...
    def _maybe_downcast_constants(self, left: Any, right: Any) -> tuple[Any, Any]: ...
    def _maybe_eval(self, binop: BinOp, eval_in_python: tuple[str, ...]) -> Any: ...
    def _maybe_evaluate_binop(
        self,
        op: Callable[..., Any],
        op_class: type[ast.cmpop],
        lhs: Any,
        rhs: Any,
        eval_in_python: tuple[str, ...] = ...,
        maybe_eval_in_python: tuple[str, ...] = ...,
    ) -> Any: ...
    def visit_BinOp(self, node: ast.BinOp, **kwargs: Any) -> Any: ...
    def visit_UnaryOp(self, node: ast.UnaryOp, **kwargs: Any) -> Any: ...
    def visit_Name(self, node: ast.Name, **kwargs: Any) -> Term: ...
    def visit_NameConstant(self, node: ast.NameConstant, **kwargs: Any) -> Constant: ...
    def visit_Num(self, node: ast.Num, **kwargs: Any) -> Constant: ...
    def visit_Constant(self, node: ast.Constant, **kwargs: Any) -> Constant: ...
    def visit_Str(self, node: ast.Str, **kwargs: Any) -> Term: ...
    def visit_List(self, node: ast.List, **kwargs: Any) -> Term: ...
    def visit_Tuple(self, node: ast.Tuple, **kwargs: Any) -> Term: ...
    def visit_Index(self, node: ast.Index, **kwargs: Any) -> Any: ...
    def visit_Subscript(self, node: ast.Subscript, **kwargs: Any) -> Term: ...
    def visit_Slice(self, node: ast.Slice, **kwargs: Any) -> slice: ...
    def visit_Assign(self, node: ast.Assign, **kwargs: Any) -> Any: ...
    def visit_Attribute(self, node: ast.Attribute, **kwargs: Any) -> Term: ...
    def visit_Call(self, node: ast.Call, side: str | None = ..., **kwargs: Any) -> Any: ...
    def translate_In(self, op: ast.cmpop) -> ast.cmpop: ...
    def visit_Compare(self, node: ast.Compare, **kwargs: Any) -> Any: ...
    def _try_visit_binop(self, bop: Any) -> Any: ...
    def visit_BoolOp(self, node: ast.BoolOp, **kwargs: Any) -> Any: ...

class PandasExprVisitor(BaseExprVisitor):
    def __init__(
        self,
        env: Scope,
        engine: str,
        parser: str,
        preparser: Callable[[str], str] = ...,
    ) -> None: ...

class PythonExprVisitor(BaseExprVisitor):
    def __init__(
        self,
        env: Scope,
        engine: str,
        parser: str,
        preparser: Callable[[str], str] = ...,
    ) -> None: ...

class Expr:
    expr: str
    env: Scope
    engine: str
    parser: str
    _visitor: BaseExprVisitor
    terms: Any

    def __init__(
        self,
        expr: str,
        engine: str = ...,
        parser: str = ...,
        env: Scope | None = ...,
        level: int = ...,
    ) -> None: ...
    @property
    def assigner(self) -> str | None: ...
    def __call__(self) -> Any: ...
    def __repr__(self) -> str: ...
    def __len__(self) -> int: ...
    def parse(self) -> Any: ...
    @property
    def names(self) -> frozenset[str]: ...

PARSERS: dict[str, type[BaseExprVisitor]]
```