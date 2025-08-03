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
    Union, 
    cast
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
    from pandas.core.computation.ops import BaseOp

Token = Tuple[int, str]
Node = ast.AST
VisitorMethod = Callable[..., Any]

def func_y8fuuh5t(tok: Token) -> Token:
    """
    Rewrite the assignment operator for PyTables expressions that use ``=``
    as a substitute for ``==``.
    """
    toknum, tokval = tok
    return toknum, '==' if tokval == '=' else tokval

def func_sgaf5d32(tok: Token) -> Token:
    """
    Replace ``&`` with ``and`` and ``|`` with ``or`` so that bitwise
    precedence is changed to boolean precedence.
    """
    toknum, tokval = tok
    if toknum == tokenize.OP:
        if tokval == '&':
            return tokenize.NAME, 'and'
        elif tokval == '|':
            return tokenize.NAME, 'or'
        return toknum, tokval
    return toknum, tokval

def func_jax6ebwv(tok: Token) -> Token:
    """
    Replace local variables with a syntactically valid name.
    """
    toknum, tokval = tok
    if toknum == tokenize.OP and tokval == '@':
        return tokenize.OP, LOCAL_TAG
    return toknum, tokval

def func_netq6my4(f: Callable, g: Callable) -> Callable:
    """
    Compose 2 callables.
    """
    return lambda *args, **kwargs: f(g(*args, **kwargs))

def func_x0jxop4z(*funcs: Callable) -> Callable:
    """
    Compose 2 or more callables.
    """
    assert len(funcs) > 1, 'At least 2 callables must be passed to compose'
    return reduce(func_netq6my4, funcs)

def func_upu1eds3(
    source: str, 
    f: Callable[[Token], Token] = func_x0jxop4z(
        func_jax6ebwv, 
        func_sgaf5d32, 
        func_y8fuuh5t, 
        clean_backtick_quoted_toks
    )
) -> str:
    """
    Compose a collection of tokenization functions.
    """
    assert callable(f), 'f must be callable'
    return tokenize.untokenize(f(x) for x in tokenize_string(source))

def func_0fg7d3ug(t: type) -> Callable[[Any], bool]:
    """
    Factory for a type checking function of type ``t`` or tuple of types.
    """
    return lambda x: isinstance(x.value, t)

_is_list: Callable[[Any], bool] = func_0fg7d3ug(list)
_is_str: Callable[[Any], bool] = func_0fg7d3ug(str)
_all_nodes: FrozenSet[type] = frozenset(
    node for node in (getattr(ast, name) for name in dir(ast)) 
    if isinstance(node, type) and issubclass(node, ast.AST)
)

def func_q7shhtrh(superclass: type, all_nodes: FrozenSet[type] = _all_nodes) -> FrozenSet[str]:
    """
    Filter out AST nodes that are subclasses of ``superclass``.
    """
    node_names = (node.__name__ for node in all_nodes if issubclass(node, superclass))
    return frozenset(node_names)

_all_node_names: FrozenSet[str] = frozenset(x.__name__ for x in _all_nodes)
_mod_nodes: FrozenSet[str] = func_q7shhtrh(ast.mod)
_stmt_nodes: FrozenSet[str] = func_q7shhtrh(ast.stmt)
_expr_nodes: FrozenSet[str] = func_q7shhtrh(ast.expr)
_expr_context_nodes: FrozenSet[str] = func_q7shhtrh(ast.expr_context)
_boolop_nodes: FrozenSet[str] = func_q7shhtrh(ast.boolop)
_operator_nodes: FrozenSet[str] = func_q7shhtrh(ast.operator)
_unary_op_nodes: FrozenSet[str] = func_q7shhtrh(ast.unaryop)
_cmp_op_nodes: FrozenSet[str] = func_q7shhtrh(ast.cmpop)
_comprehension_nodes: FrozenSet[str] = func_q7shhtrh(ast.comprehension)
_handler_nodes: FrozenSet[str] = func_q7shhtrh(ast.excepthandler)
_arguments_nodes: FrozenSet[str] = func_q7shhtrh(ast.arguments)
_keyword_nodes: FrozenSet[str] = func_q7shhtrh(ast.keyword)
_alias_nodes: FrozenSet[str] = func_q7shhtrh(ast.alias)
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

_T = TypeVar('_T')

def func_54zjf998(node_name: str) -> Callable:
    """
    Return a function that raises a NotImplementedError with a passed node name.
    """
    def func_62znxl73(self: Any, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(f"'{node_name}' nodes are not implemented")
    return func_62znxl73

def func_3t5xqf78(nodes: FrozenSet[str]) -> Callable:
    """
    Decorator to disallow certain nodes from parsing.
    """
    def func_74kukzar(cls: _T) -> _T:
        cls.unsupported_nodes = ()
        for node in nodes:
            new_method = func_54zjf998(node)
            name = f'visit_{node}'
            cls.unsupported_nodes += (name,)
            setattr(cls, name, new_method)
        return cls
    return func_74kukzar

def func_q2broc3g(op_class: type, op_symbol: str) -> Callable:
    """
    Return a function to create an op class with its symbol already passed.
    """
    def func_62znxl73(self: Any, node: Node, *args: Any, **kwargs: Any) -> partial:
        """
        Return a partial function with an Op subclass with an operator already passed.
        """
        return partial(op_class, op_symbol, *args, **kwargs)
    return func_62znxl73

_op_classes: Dict[str, type] = {'binary': BinOp, 'unary': UnaryOp}

def func_584p78zy(op_classes: Dict[str, type]) -> Callable:
    """
    Decorator to add default implementation of ops.
    """
    def func_62znxl73(cls: _T) -> _T:
        for op_attr_name, op_class in op_classes.items():
            ops = getattr(cls, f'{op_attr_name}_ops')
            ops_map = getattr(cls, f'{op_attr_name}_op_nodes_map')
            for op in ops:
                op_node = ops_map[op]
                if op_node is not None:
                    made_op = func_q2broc3g(op_class, op)
                    setattr(cls, f'visit_{op_node}', made_op)
        return cls
    return func_62znxl73

@func_3t5xqf78(_unsupported_nodes)
@func_584p78zy(_op_classes)
class BaseExprVisitor(ast.NodeVisitor):
    """
    Custom ast walker. Parsers of other engines should subclass this class
    if necessary.
    """
    const_type: ClassVar[type] = Constant
    term_type: ClassVar[type] = Term
    binary_ops: ClassVar[Tuple[str, ...]] = CMP_OPS_SYMS + BOOL_OPS_SYMS + ARITH_OPS_SYMS
    binary_op_nodes: ClassVar[Tuple[str, ...]] = (
        'Gt', 'Lt', 'GtE', 'LtE', 'Eq', 'NotEq', 'In', 'NotIn', 
        'BitAnd', 'BitOr', 'And', 'Or', 'Add', 'Sub', 'Mult', 
        'Div', 'Pow', 'FloorDiv', 'Mod'
    )
    binary_op_nodes_map: ClassVar[Dict[str, str]] = dict(zip(binary_ops, binary_op_nodes))
    unary_ops: ClassVar[Tuple[str, ...]] = UNARY_OPS_SYMS
    unary_op_nodes: ClassVar[Tuple[str, ...]] = ('UAdd', 'USub', 'Invert', 'Not')
    unary_op_nodes_map: ClassVar[Dict[str, str]] = dict(zip(unary_ops, unary_op_nodes))
    rewrite_map: ClassVar[Dict[type, type]] = {
        ast.Eq: ast.In, 
        ast.NotEq: ast.NotIn, 
        ast.In: ast.In, 
        ast.NotIn: ast.NotIn
    }

    def __init__(
        self, 
        env: Scope, 
        engine: str, 
        parser: str, 
        preparser: Callable[[str, Optional[Callable]], str] = func_upu1eds3
    ) -> None:
        self.env: Scope = env
        self.engine: str = engine
        self.parser: str = parser
        self.preparser: Callable[[str, Optional[Callable]], str] = preparser
        self.assigner: Optional[str] = None

    def func_pfrsxk6o(self, node: Union[str, Node], **kwargs: Any) -> Any:
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

    def func_jszipbqb(self, node: ast.Module, **kwargs: Any) -> Any:
        if len(node.body) != 1:
            raise SyntaxError('only a single expression is allowed')
        expr = node.body[0]
        return self.visit(expr, **kwargs)

    def func_b65dbn8y(self, node: ast.Expr, **kwargs: Any) -> Any:
        return self.visit(node.value, **kwargs)

    def func_m8wasilv(self, node: ast.BinOp, left: Term, right: Term) -> Tuple[BaseOp, Any, Term, Term]:
        op_instance = node.op
        op_type = type(op_instance)
        if is_term(left) and is_term(right) and op_type in self.rewrite_map:
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
        return op, op_instance, left, right

    def func_81qpti6n(self, node: ast.BinOp, left: Optional[Term] = None, right: Optional[Term] = None) -> Tuple[BaseOp, Any, Term, Term]:
        if left is None:
            left = self.visit(node.left, side='left')
        if right is None:
            right = self.visit(node.right, side='right')
        op, op_class, left, right = self._rewrite_membership_op(node, left, right)
        return op, op_class, left, right

    def func_n4tipv6t(self, left: Term, right: Term) -> Tuple[Term, Term]:
        f32 = np.dtype(np.float32)
        if left.is_scalar and hasattr(left, 'value') and not right.is_scalar and right.return_type == f32:
            name = self.env.add_tmp(np.float32(left.value))
            left = self.term_type(name, self.env)
        if right.is_scalar and hasattr(right, 'value') and not left.is_scalar and left.return_type == f32:
            name = self.env.add_tmp(np.float32(right.value))
            right = self.term_type(name, self.env)
        return left, right

    def func_klizmj28(self, binop: BinOp, eval_in_python: bool) -> Any:
        return binop.evaluate(self.env, self.engine, self.parser, self.term_type, eval_in_python)

    def func_a2tjhx9k(
        self, 
        op: BaseOp, 
        op_class: Any, 
        lhs: Term, 
        rhs: Term, 
        eval_in_python: Tuple[str, ...] = ('in', 'not in'), 
        maybe_eval_in_python: Tuple[str, ...] = ('==', '!=', '<', '>', '<=', '>=')
    ) -> Any:
        res = op(lhs, rhs)
        if res.has_invalid_return_type:
            raise TypeError(
                f"unsupported operand type(s) for {res.op}: '{lhs.type}' and '{rhs.type}'"
            )
        if self.engine != 'pytables' and (
            res.op in CMP_OPS_SYMS and 
            getattr(lhs, 'is_datetime', False) or 
            getattr(rhs, 'is_datetime', False)
        ):
            return self._maybe_eval(res, self.binary_ops)
        if res.op in eval_in_python:
            return self._maybe_eval(res, eval_in_python)
        elif self.engine != 'pytables':
            if (
                getattr(lhs, 'return_type', None) == object or 
                is_string_dtype(getattr(lhs, 'return_type', None)) or 
                getattr(rhs, 'return_type', None) == object or 
                is_string_dtype(getattr(rhs, 'return_type', None))
            ):
                return self._maybe_eval(res, eval_in_python + maybe_eval_in_python)
        return res

    def func_r7u90jwf(self, node: ast.BinOp, **kwargs: Any) -> Any:
        op, op_class, left, right = self._maybe_transform_eq_ne(node)
        left, right = self._maybe_downcast_constants(left, right)
        return self._maybe_evaluate_binop(op, op_class, left, right)

    def func_wo1obbsk(self, node: ast.UnaryOp, **kwargs: Any) -> Any:
        op = self.visit(node.op)
        operand = self.visit(node.operand)
        return op(operand)

    def func_s97xxk9l(self, node: ast.Name, **kwargs: Any) -> Term:
        return self.term_type(node.id, self.env, **kwargs)

    def func_6iaadf4h(self, node: ast.Num, **kwargs: Any) -> Constant:
        return self.const_type(node.value, self.env)

    def func_k54hfg4o(self, node: ast.Str, **kwargs: Any) -> Constant:
        return self.const_type(node.value, self.env)

    def func_cgxclb95(self, node: ast.NameConstant, **kwargs: Any) -> Constant:
        return self.const_type(node.value, self.env)

    def func_fikxvsfo(self, node: ast.Str, **kwargs: Any) -> Term:
        name = self.env.add_tmp(node.s)
        return self.term_type(name, self.env)

    def func_7lvu5gcy(self, node: ast.List, **kwargs: Any) -> Term:
        name = self.env.add_tmp([self.visit(e)(self.env) for e in node.elts])
        return self.term_type(name, self.env)

    def func_ft6sn6xu(self,