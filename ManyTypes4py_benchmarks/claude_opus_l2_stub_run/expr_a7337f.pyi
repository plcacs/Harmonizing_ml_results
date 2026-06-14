from __future__ import annotations

import ast
from functools import partial
from typing import TYPE_CHECKING, ClassVar, TypeVar

from pandas.core.computation.ops import (
    BinOp,
    Constant,
    FuncNode,
    Op,
    Term,
    UnaryOp,
)
from pandas.core.computation.scope import Scope

if TYPE_CHECKING:
    from collections.abc import Callable

def _rewrite_assign(tok: tuple[int, str]) -> tuple[int, str]: ...
def _replace_booleans(tok: tuple[int, str]) -> tuple[int, str]: ...
def _replace_locals(tok: tuple[int, str]) -> tuple[int, str]: ...
def _compose2(f: Callable[..., object], g: Callable[..., object]) -> Callable[..., object]: ...
def _compose(*funcs: Callable[..., object]) -> Callable[..., object]: ...
def _preparse(
    source: str,
    f: Callable[[tuple[int, str]], tuple[int, str]] = ...,
) -> str: ...
def _is_type(t: type | tuple[type, ...]) -> Callable[[Term], bool]: ...

_is_list: Callable[[Term], bool]
_is_str: Callable[[Term], bool]

_all_nodes: frozenset[type]

def _filter_nodes(
    superclass: type, all_nodes: frozenset[type] = ...
) -> frozenset[str]: ...

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
_msg: str

def _node_not_implemented(node_name: str) -> Callable[..., None]: ...

_T = TypeVar("_T")

def disallow(nodes: frozenset[str]) -> Callable[[type[_T]], type[_T]]: ...

def _op_maker(
    op_class: type[BinOp] | type[UnaryOp], op_symbol: str
) -> Callable[..., partial[BinOp | UnaryOp]]: ...

_op_classes: dict[str, type[BinOp] | type[UnaryOp]]

def add_ops(
    op_classes: dict[str, type[BinOp] | type[UnaryOp]],
) -> Callable[[type[_T]], type[_T]]: ...

class BaseExprVisitor(ast.NodeVisitor):
    const_type: ClassVar[type[Constant]]
    term_type: ClassVar[type[Term]]
    binary_ops: ClassVar[tuple[str, ...]]
    binary_op_nodes: ClassVar[tuple[str, ...]]
    binary_op_nodes_map: ClassVar[dict[str, str]]
    unary_ops: ClassVar[tuple[str, ...]]
    unary_op_nodes: ClassVar[tuple[str, ...]]
    unary_op_nodes_map: ClassVar[dict[str, str]]
    rewrite_map: ClassVar[dict[type[ast.cmpop], type[ast.cmpop]]]
    unsupported_nodes: tuple[str, ...]

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
        preparser: Callable[..., str] = ...,
    ) -> None: ...
    def visit(self, node: ast.AST | str, **kwargs: object) -> object: ...
    def visit_Module(self, node: ast.Module, **kwargs: object) -> object: ...
    def visit_Expr(self, node: ast.Expr, **kwargs: object) -> object: ...
    def _rewrite_membership_op(self, node: ast.BinOp, left: Term, right: Term) -> tuple[partial[BinOp], ast.AST, Term, Term]: ...
    def _maybe_transform_eq_ne(self, node: ast.BinOp, left: Term | None = ..., right: Term | None = ...) -> tuple[partial[BinOp], ast.AST, Term, Term]: ...
    def _maybe_downcast_constants(self, left: Term, right: Term) -> tuple[Term, Term]: ...
    def _maybe_eval(self, binop: BinOp, eval_in_python: tuple[str, ...]) -> object: ...
    def _maybe_evaluate_binop(self, op: partial[BinOp], op_class: object, lhs: Term, rhs: Term, eval_in_python: tuple[str, ...] = ..., maybe_eval_in_python: tuple[str, ...] = ...) -> object: ...
    def visit_BinOp(self, node: ast.BinOp, **kwargs: object) -> object: ...
    def visit_UnaryOp(self, node: ast.UnaryOp, **kwargs: object) -> object: ...
    def visit_Name(self, node: ast.Name, **kwargs: object) -> Term: ...
    def visit_NameConstant(self, node: ast.NameConstant, **kwargs: object) -> Constant: ...
    def visit_Num(self, node: ast.Num, **kwargs: object) -> Constant: ...
    def visit_Constant(self, node: ast.Constant, **kwargs: object) -> Constant: ...
    def visit_Str(self, node: ast.Str, **kwargs: object) -> Term: ...
    def visit_List(self, node: ast.List, **kwargs: object) -> Term: ...
    visit_Tuple: Callable[..., Term]
    def visit_Index(self, node: ast.Index, **kwargs: object) -> object: ...
    def visit_Subscript(self, node: ast.Subscript, **kwargs: object) -> Term: ...
    def visit_Slice(self, node: ast.Slice, **kwargs: object) -> slice: ...
    def visit_Assign(self, node: ast.Assign, **kwargs: object) -> object: ...
    def visit_Attribute(self, node: ast.Attribute, **kwargs: object) -> Term: ...
    def visit_Call(self, node: ast.Call, side: str | None = ..., **kwargs: object) -> object: ...
    def translate_In(self, op: ast.cmpop) -> ast.cmpop: ...
    def visit_Compare(self, node: ast.Compare, **kwargs: object) -> object: ...
    def _try_visit_binop(self, bop: object) -> object: ...
    def visit_BoolOp(self, node: ast.BoolOp, **kwargs: object) -> object: ...

_python_not_supported: frozenset[str]
_numexpr_supported_calls: frozenset[str]

class PandasExprVisitor(BaseExprVisitor):
    def __init__(
        self,
        env: Scope,
        engine: str,
        parser: str,
        preparser: Callable[..., str] = ...,
    ) -> None: ...

class PythonExprVisitor(BaseExprVisitor):
    def __init__(
        self,
        env: Scope,
        engine: str,
        parser: str,
        preparser: Callable[..., str] = ...,
    ) -> None: ...

class Expr:
    expr: str
    env: Scope
    engine: str
    parser: str
    _visitor: BaseExprVisitor
    terms: object

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
    def __call__(self) -> object: ...
    def __repr__(self) -> str: ...
    def __len__(self) -> int: ...
    def parse(self) -> object: ...
    @property
    def names(self) -> frozenset[str]: ...

PARSERS: dict[str, type[PythonExprVisitor] | type[PandasExprVisitor]]