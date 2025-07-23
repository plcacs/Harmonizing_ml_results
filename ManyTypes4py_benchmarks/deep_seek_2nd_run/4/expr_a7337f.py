"""
:func:`~pandas.eval` parsers.
"""
from __future__ import annotations
import ast
from functools import partial, reduce
from keyword import iskeyword
import tokenize
from typing import (
    TYPE_CHECKING, 
    ClassVar, 
    TypeVar, 
    Any, 
    Callable, 
    Dict, 
    FrozenSet, 
    List, 
    Optional, 
    Tuple, 
    Type, 
    Union, 
    cast,
    overload
)
import numpy as np
from pandas.errors import UndefinedVariableError
from pandas.core.dtypes.common import is_string_dtype
import pandas.core.common as com
from pandas.core.computation.ops import (
    ARITH_OPS_SYMS, 
    BOOL_OPS_SYMS, 
    CMP_OPS_SYMS, 
    LOCAL_TAG, 
    MATHOPS, 
    REDUCTIONS, 
    UNARY_OPS_SYMS, 
    BinOp, 
    Constant, 
    FuncNode, 
    Op, 
    Term, 
    UnaryOp, 
    is_term
)
from pandas.core.computation.parsing import clean_backtick_quoted_toks, tokenize_string
from pandas.core.computation.scope import Scope
from pandas.io.formats import printing

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pandas.core.computation.ops import Node
    from pandas.core.computation.scope import ScopeType

_T = TypeVar('_T')
_AST = TypeVar('_AST', bound=ast.AST)
_Token = Tuple[int, str]
_Tokenizer = Callable[[str], Any]

def _rewrite_assign(tok: _Token) -> _Token:
    """
    Rewrite the assignment operator for PyTables expressions that use ``=``
    as a substitute for ``==``.

    Parameters
    ----------
    tok : tuple of int, str
        ints correspond to the all caps constants in the tokenize module

    Returns
    -------
    tuple of int, str
        Either the input or token or the replacement values
    """
    toknum, tokval = tok
    return (toknum, '==' if tokval == '=' else tokval)

def _replace_booleans(tok: _Token) -> _Token:
    """
    Replace ``&`` with ``and`` and ``|`` with ``or`` so that bitwise
    precedence is changed to boolean precedence.

    Parameters
    ----------
    tok : tuple of int, str
        ints correspond to the all caps constants in the tokenize module

    Returns
    -------
    tuple of int, str
        Either the input or token or the replacement values
    """
    toknum, tokval = tok
    if toknum == tokenize.OP:
        if tokval == '&':
            return (tokenize.NAME, 'and')
        elif tokval == '|':
            return (tokenize.NAME, 'or')
        return (toknum, tokval)
    return (toknum, tokval)

def _replace_locals(tok: _Token) -> _Token:
    """
    Replace local variables with a syntactically valid name.

    Parameters
    ----------
    tok : tuple of int, str
        ints correspond to the all caps constants in the tokenize module

    Returns
    -------
    tuple of int, str
        Either the input or token or the replacement values

    Notes
    -----
    This is somewhat of a hack in that we rewrite a string such as ``'@a'`` as
    ``'__pd_eval_local_a'`` by telling the tokenizer that ``__pd_eval_local_``
    is a ``tokenize.OP`` and to replace the ``'@'`` symbol with it.
    """
    toknum, tokval = tok
    if toknum == tokenize.OP and tokval == '@':
        return (tokenize.OP, LOCAL_TAG)
    return (toknum, tokval)

def _compose2(f: Callable[..., _T], g: Callable[..., Any]) -> Callable[..., _T]:
    """
    Compose 2 callables.
    """
    return lambda *args, **kwargs: f(g(*args, **kwargs))

def _compose(*funcs: Callable[..., Any]) -> Callable[..., Any]:
    """
    Compose 2 or more callables.
    """
    assert len(funcs) > 1, 'At least 2 callables must be passed to compose'
    return reduce(_compose2, funcs)

def _preparse(
    source: str, 
    f: Callable[[_Token], _Token] = _compose(
        _replace_locals, _replace_booleans, _rewrite_assign, clean_backtick_quoted_toks
    )
) -> str:
    """
    Compose a collection of tokenization functions.

    Parameters
    ----------
    source : str
        A Python source code string
    f : callable
        This takes a tuple of (toknum, tokval) as its argument and returns a
        tuple with the same structure but possibly different elements. Defaults
        to the composition of ``_rewrite_assign``, ``_replace_booleans``, and
        ``_replace_locals``.

    Returns
    -------
    str
        Valid Python source code

    Notes
    -----
    The `f` parameter can be any callable that takes *and* returns input of the
    form ``(toknum, tokval)``, where ``toknum`` is one of the constants from
    the ``tokenize`` module and ``tokval`` is a string.
    """
    assert callable(f), 'f must be callable'
    return tokenize.untokenize((f(x) for x in tokenize_string(source)))

def _is_type(t: Union[type, Tuple[type, ...]]) -> Callable[[Any], bool]:
    """
    Factory for a type checking function of type ``t`` or tuple of types.
    """
    return lambda x: isinstance(x.value, t)

_is_list: Callable[[Any], bool] = _is_type(list)
_is_str: Callable[[Any], bool] = _is_type(str)
_all_nodes: FrozenSet[Type[ast.AST]] = frozenset(
    node for node in (getattr(ast, name) for name in dir(ast)) 
    if isinstance(node, type) and issubclass(node, ast.AST)
)

def _filter_nodes(
    superclass: Type[ast.AST], 
    all_nodes: FrozenSet[Type[ast.AST]] = _all_nodes
) -> FrozenSet[str]:
    """
    Filter out AST nodes that are subclasses of ``superclass``.
    """
    node_names = (node.__name__ for node in all_nodes if issubclass(node, superclass))
    return frozenset(node_names)

_all_node_names: FrozenSet[str] = frozenset(x.__name__ for x in _all_nodes)
_mod_nodes: FrozenSet[str] = _filter_nodes(ast.mod)
_stmt_nodes: FrozenSet[str] = _filter_nodes(ast.stmt)
_expr_nodes: FrozenSet[str] = _filter_nodes(ast.expr)
_expr_context_nodes: FrozenSet[str] = _filter_nodes(ast.expr_context)
_boolop_nodes: FrozenSet[str] = _filter_nodes(ast.boolop)
_operator_nodes: FrozenSet[str] = _filter_nodes(ast.operator)
_unary_op_nodes: FrozenSet[str] = _filter_nodes(ast.unaryop)
_cmp_op_nodes: FrozenSet[str] = _filter_nodes(ast.cmpop)
_comprehension_nodes: FrozenSet[str] = _filter_nodes(ast.comprehension)
_handler_nodes: FrozenSet[str] = _filter_nodes(ast.excepthandler)
_arguments_nodes: FrozenSet[str] = _filter_nodes(ast.arguments)
_keyword_nodes: FrozenSet[str] = _filter_nodes(ast.keyword)
_alias_nodes: FrozenSet[str] = _filter_nodes(ast.alias)
_hacked_nodes: FrozenSet[str] = frozenset(['Assign', 'Module', 'Expr'])
_unsupported_expr_nodes: FrozenSet[str] = frozenset([
    'Yield', 'GeneratorExp', 'IfExp', 'DictComp', 'SetComp', 'Repr', 
    'Lambda', 'Set', 'AST', 'Is', 'IsNot'
])
_unsupported_nodes: FrozenSet[str] = (
    _stmt_nodes | _mod_nodes | _handler_nodes | _arguments_nodes | 
    _keyword_nodes | _alias_nodes | _expr_context_nodes | _unsupported_expr_nodes
) - _hacked_nodes
_base_supported_nodes: FrozenSet[str] = _all_node_names - _unsupported_nodes | _hacked_nodes
intersection: FrozenSet[str] = _unsupported_nodes & _base_supported_nodes
_msg: str = f'cannot both support and not support {intersection}'
assert not intersection, _msg

def _node_not_implemented(node_name: str) -> Callable[..., None]:
    """
    Return a function that raises a NotImplementedError with a passed node name.
    """
    def f(self: Any, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(f"'{node_name}' nodes are not implemented")
    return f

def disallow(nodes: FrozenSet[str]) -> Callable[[Type[_T]], Type[_T]]:
    """
    Decorator to disallow certain nodes from parsing. Raises a
    NotImplementedError instead.

    Returns
    -------
    callable
    """
    def disallowed(cls: Type[_T]) -> Type[_T]:
        cls.unsupported_nodes = ()
        for node in nodes:
            new_method = _node_not_implemented(node)
            name = f'visit_{node}'
            cls.unsupported_nodes += (name,)
            setattr(cls, name, new_method)
        return cls
    return disallowed

def _op_maker(op_class: Type[Op], op_symbol: str) -> Callable[..., partial[Op]]:
    """
    Return a function to create an op class with its symbol already passed.

    Returns
    -------
    callable
    """
    def f(self: Any, node: ast.AST, *args: Any, **kwargs: Any) -> partial[Op]:
        """
        Return a partial function with an Op subclass with an operator already passed.

        Returns
        -------
        callable
        """
        return partial(op_class, op_symbol, *args, **kwargs)
    return f

_op_classes: Dict[str, Type[Op]] = {'binary': BinOp, 'unary': UnaryOp}

def add_ops(op_classes: Dict[str, Type[Op]]) -> Callable[[Type[_T]], Type[_T]]:
    """
    Decorator to add default implementation of ops.
    """
    def f(cls: Type[_T]) -> Type[_T]:
        for op_attr_name, op_class in op_classes.items():
            ops = getattr(cls, f'{op_attr_name}_ops')
            ops_map = getattr(cls, f'{op_attr_name}_op_nodes_map')
            for op in ops:
                op_node = ops_map[op]
                if op_node is not None:
                    made_op = _op_maker(op_class, op)
                    setattr(cls, f'visit_{op_node}', made_op)
        return cls
    return f

@disallow(_unsupported_nodes)
@add_ops(_op_classes)
class BaseExprVisitor(ast.NodeVisitor):
    """
    Custom ast walker. Parsers of other engines should subclass this class
    if necessary.

    Parameters
    ----------
    env : Scope
    engine : str
    parser : str
    preparser : callable
    """
    const_type: Type[Constant] = Constant
    term_type: Type[Term] = Term
    binary_ops: List[str] = CMP_OPS_SYMS + BOOL_OPS_SYMS + ARITH_OPS_SYMS
    binary_op_nodes: Tuple[str, ...] = (
        'Gt', 'Lt', 'GtE', 'LtE', 'Eq', 'NotEq', 'In', 'NotIn', 
        'BitAnd', 'BitOr', 'And', 'Or', 'Add', 'Sub', 'Mult', 
        'Div', 'Pow', 'FloorDiv', 'Mod'
    )
    binary_op_nodes_map: Dict[str, str] = dict(zip(binary_ops, binary_op_nodes))
    unary_ops: List[str] = UNARY_OPS_SYMS
    unary_op_nodes: Tuple[str, ...] = ('UAdd', 'USub', 'Invert', 'Not')
    unary_op_nodes_map: Dict[str, str] = dict(zip(unary_ops, unary_op_nodes))
    rewrite_map: Dict[Type[ast.AST], Type[ast.AST]] = {
        ast.Eq: ast.In, 
        ast.NotEq: ast.NotIn, 
        ast.In: ast.In, 
        ast.NotIn: ast.NotIn
    }
    unsupported_nodes: Tuple[str, ...]
    assigner: Optional[str] = None

    def __init__(
        self, 
        env: Scope, 
        engine: str, 
        parser: str, 
        preparser: Callable[[str, Optional[Callable[..., Any]]], str] = _preparse
    ):
        self.env: Scope = env
        self.engine: str = engine
        self.parser: str = parser
        self.preparser: Callable[[str, Optional[Callable[..., Any]]], str] = preparser
        self.assigner: Optional[str] = None

    def visit(self, node: Union[str, ast.AST], **kwargs: Any) -> Any:
        if isinstance(node, str):
            clean = self.preparser(node)
            try:
                node = ast.fix_missing_locations(ast.parse(clean))
            except SyntaxError as e:
                if any(iskeyword(x) for x in clean.split()):
                    e.msg = 'Python keyword not valid identifier in numexpr query'
                raise e
        method = f'visit_{type(node).__name__}'
        visitor = getattr(self, method)
        return visitor(node, **kwargs)

    def visit_Module(self, node: ast.Module, **kwargs: Any) -> Any:
        if len(node.body) != 1:
            raise SyntaxError('only a single expression is allowed')
        expr = node.body[0]
        return self.visit(expr, **kwargs)

    def visit_Expr(self, node: ast.Expr, **kwargs: Any) -> Any:
        return self.visit(node.value, **kwargs)

    def _rewrite_membership_op(
        self, 
        node: ast.BinOp, 
        left: Term, 
        right: Term
    ) -> Tuple[Op, ast.AST, Term, Term]:
        op_instance = node.op
        op_type = type(op_instance)
        if is_term(left) and is_term(right) and (op_type in self.rewrite_map):
            left_list, right_list = map(_is_list, (left, right))
            left_str, right_str = map(_is_str, (left, right))
            if left_list or right_list or left_str or right_str:
                op_instance = self.rewrite_map[op_type]()
            if right_str:
                name = self.env.add_tmp([right.value])
                right = self.term_type(name, self.env)
            if left_str:
                name = self.env.add_tmp([left.value])
                left = self.term_type(name, self.env)
        op = self.visit(op_instance)
        return (op, op_instance, left, right)

    def _maybe_transform_eq_ne(
        self, 
        node: ast.BinOp, 
        left: Optional[Term] = None, 
        right: Optional[Term] = None
    ) -> Tuple[Op, ast.AST, Term, Term]:
        if left is None:
            left = self.visit(node.left, side='left')
        if right is None:
            right = self.visit(node.right, side='right')
        op, op_class, left, right = self._rewrite_membership_op(node, left, right)
        return (op, op_class, left, right)

    def _maybe_downcast_constants(self, left: Term, right: Term) -> Tuple[Term, Term]:
        f32 = np.dtype(np.float32)
        if left.is_scalar and hasattr(left, 'value') and (not right.is_scalar) and (right.return_type == f32):
            name = self.env.add_tmp(np.float32(left.value))
            left = self.term_type(name, self.env)
        if right.is_scalar and hasattr(right, 'value') and (not left.is_scalar) and (left.return_type == f32):
            name = self.env.add_tmp(np.float32(right.value))
            right = self.term_type(name, self.env)
        return (left, right)

    def _maybe_eval(self, binop: BinOp, eval_in_python: Union[bool, Tuple[str, ...]]) -> Any:
        return binop.evaluate(self.env, self.engine, self.parser, self.term_type, eval_in_python)

    def _maybe_evaluate_binop(
        self, 
        op: Op, 
        op_class: ast.AST, 
        lhs: Term, 
        rhs: Term, 
        eval_in_python: Tuple[str, ...] = ('in', 'not in'), 
        maybe_eval_in_python: Tuple[str, ...] = ('==', '!=', '<', '>', '<=', '>=')
    ) -> Any:
        res = op(lhs, rhs)
        if res.has_invalid_return_type:
            raise TypeError(f"unsupported operand type(s) for {res.op}: '{lhs.type}' and '{rhs.type}'")
        if self.engine != 'pytables' and (res.op in CMP_OPS_SYMS and getattr(lhs, 'is_datetime', False) or getattr(rhs, 'is_datetime', False)):
            return self._maybe_eval(res, self.binary_ops)
        if res.op in eval_in_python:
            return self._maybe_eval(res, eval_in_python)
        elif self.engine != 'pytables':
            if (getattr(lhs, 'return_type', None) == object or 
                is_string_dtype