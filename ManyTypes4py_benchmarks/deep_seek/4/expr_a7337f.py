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
    from pandas.core.computation.ops import Node
    from pandas.core.computation.scope import ScalarType

_T = TypeVar('_T')
_ExprT = TypeVar('_ExprT', bound='Expr')
_Token = Tuple[int, str]
_ASTNode = TypeVar('_ASTNode', bound=ast.AST)
_OpT = TypeVar('_OpT', bound=Op)

def _rewrite_assign(tok: _Token) -> _Token:
    toknum, tokval = tok
    return (toknum, '==' if tokval == '=' else tokval)

def _replace_booleans(tok: _Token) -> _Token:
    toknum, tokval = tok
    if toknum == tokenize.OP:
        if tokval == '&':
            return (tokenize.NAME, 'and')
        elif tokval == '|':
            return (tokenize.NAME, 'or')
        return (toknum, tokval)
    return (toknum, tokval)

def _replace_locals(tok: _Token) -> _Token:
    toknum, tokval = tok
    if toknum == tokenize.OP and tokval == '@':
        return (tokenize.OP, LOCAL_TAG)
    return (toknum, tokval)

def _compose2(f: Callable[..., _T], g: Callable[..., Any]) -> Callable[..., _T]:
    return lambda *args, **kwargs: f(g(*args, **kwargs))

def _compose(*funcs: Callable[..., Any]) -> Callable[..., Any]:
    assert len(funcs) > 1, 'At least 2 callables must be passed to compose'
    return reduce(_compose2, funcs)

def _preparse(
    source: str, 
    f: Callable[[_Token], _Token] = _compose(
        _replace_locals, 
        _replace_booleans, 
        _rewrite_assign, 
        clean_backtick_quoted_toks
    )
) -> str:
    assert callable(f), 'f must be callable'
    return tokenize.untokenize((f(x) for x in tokenize_string(source)))

def _is_type(t: type) -> Callable[[Any], bool]:
    return lambda x: isinstance(x.value, t)

_is_list: Callable[[Any], bool] = _is_type(list)
_is_str: Callable[[Any], bool] = _is_type(str)

_all_nodes = frozenset(
    node for node in (getattr(ast, name) for name in dir(ast)) 
    if isinstance(node, type) and issubclass(node, ast.AST)
)

def _filter_nodes(superclass: type, all_nodes: FrozenSet[type] = _all_nodes) -> FrozenSet[str]:
    node_names = (node.__name__ for node in all_nodes if issubclass(node, superclass))
    return frozenset(node_names)

_all_node_names = frozenset(x.__name__ for x in _all_nodes)
_mod_nodes = _filter_nodes(ast.mod)
_stmt_nodes = _filter_nodes(ast.stmt)
_expr_nodes = _filter_nodes(ast.expr)
_expr_context_nodes = _filter_nodes(ast.expr_context)
_boolop_nodes = _filter_nodes(ast.boolop)
_operator_nodes = _filter_nodes(ast.operator)
_unary_op_nodes = _filter_nodes(ast.unaryop)
_cmp_op_nodes = _filter_nodes(ast.cmpop)
_comprehension_nodes = _filter_nodes(ast.comprehension)
_handler_nodes = _filter_nodes(ast.excepthandler)
_arguments_nodes = _filter_nodes(ast.arguments)
_keyword_nodes = _filter_nodes(ast.keyword)
_alias_nodes = _filter_nodes(ast.alias)
_hacked_nodes = frozenset(['Assign', 'Module', 'Expr'])
_unsupported_expr_nodes = frozenset([
    'Yield', 'GeneratorExp', 'IfExp', 'DictComp', 'SetComp', 
    'Repr', 'Lambda', 'Set', 'AST', 'Is', 'IsNot'
])
_unsupported_nodes = (
    _stmt_nodes | _mod_nodes | _handler_nodes | _arguments_nodes | 
    _keyword_nodes | _alias_nodes | _expr_context_nodes | _unsupported_expr_nodes
) - _hacked_nodes
_base_supported_nodes = _all_node_names - _unsupported_nodes | _hacked_nodes
intersection = _unsupported_nodes & _base_supported_nodes
_msg = f'cannot both support and not support {intersection}'
assert not intersection, _msg

def _node_not_implemented(node_name: str) -> Callable[..., None]:
    def f(self: Any, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(f"'{node_name}' nodes are not implemented")
    return f

def disallow(nodes: FrozenSet[str]) -> Callable[[_T], _T]:
    def disallowed(cls: _T) -> _T:
        cls.unsupported_nodes = ()  # type: ignore
        for node in nodes:
            new_method = _node_not_implemented(node)
            name = f'visit_{node}'
            cls.unsupported_nodes += (name,)  # type: ignore
            setattr(cls, name, new_method)
        return cls
    return disallowed

def _op_maker(op_class: type, op_symbol: str) -> Callable[..., partial[_OpT]]:
    def f(self: Any, node: _ASTNode, *args: Any, **kwargs: Any) -> partial[_OpT]:
        return partial(op_class, op_symbol, *args, **kwargs)
    return f

_op_classes: Dict[str, type] = {'binary': BinOp, 'unary': UnaryOp}

def add_ops(op_classes: Dict[str, type]) -> Callable[[_T], _T]:
    def f(cls: _T) -> _T:
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
        preparser: Callable[[str, Any], str] = _preparse
    ) -> None:
        self.env = env
        self.engine = engine
        self.parser = parser
        self.preparser = preparser
        self.assigner: Optional[str] = None

    def visit(self, node: Union[str, _ASTNode], **kwargs: Any) -> Any:
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
        node: ast.Compare, 
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
        node: Union[ast.BinOp, ast.BoolOp], 
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
            if getattr(lhs, 'return_type', None) == object or is_string_dtype(getattr(lhs, 'return_type', None)) or getattr(rhs, 'return_type', None) == object or is_string_dtype(getattr(rhs, 'return_type', None)):
                return self._maybe_eval(res, eval_in_python + maybe_eval_in_python)
        return res

    def visit_BinOp(self, node: ast.BinOp, **kwargs: Any) -> Any:
        op, op_class, left, right = self._maybe_transform_eq_ne(node)
        left, right = self._maybe_downcast_constants(left, right)
        return self._maybe_evaluate_binop(op, op_class, left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp, **kwargs: Any) -> Any:
        op = self.visit(node.op)
        operand = self.visit(node.operand)
        return op(operand)

    def visit_Name(self, node: ast.Name, **kwargs: Any) -> Term:
        return self.term_type(node.id, self.env, **kwargs)

    def visit_NameConstant(self, node: ast.NameConstant, **kwargs: Any) -> Constant:
        return self.const_type(node.value, self.env)

    def visit_Num(self, node: ast.Num, **kwargs: Any) -> Constant:
        return self.const_type(node.value, self.env)

    def visit_Constant(self, node: ast.Constant, **kwargs: Any) -> Constant:
        return self.const_type(node.value, self.env)

    def visit_Str(self, node: ast.Str, **kwargs: Any) -> Term:
        name = self.env.add_tmp(node.s)
        return self.term_type(name, self.env)

    def visit_List(self, node: ast.List, **kwargs: Any) -> Term:
        name = self.env.add_tmp([self.visit(e)(self.env) for e in node.elts])
        return self.term_type(name, self.env)

    def visit_Tuple(self, node: ast.Tuple, **kwargs: Any) -> Term:
        return self.visit_List(node, **kwargs)

    def visit_Index(self, node: ast.Index, **kwargs: Any) -> Any:
        return self.visit(node.value)

    def visit_Subscript(self, node: ast.Subscript, **kwargs: Any) -> Term:
        from pandas import eval as pd_eval
        value = self.visit(node.value)
        slobj = self.visit(node.slice)
        result = pd_eval(slobj, local_dict=self.env, engine=self.engine, parser=self.parser)
        try:
            v = value.value[result]
        except AttributeError:
            lhs = pd_eval(value, local_dict=self.env, engine=self.engine, parser=self.parser)
            v = lhs[result]
        name = self.env.add_tmp(v)
        return self.term_type(name, env=self.env)

    def visit_Slice(self, node: ast.Slice, **kwargs: Any) -> slice:
        lower = node.lower
        if lower is not None:
            lower = self.visit(lower).value
        upper = node.upper
        if upper is not None:
            upper = self.visit(upper).value
        step = node.step
        if step is not None:
            step = self.visit(step).value
        return slice(lower, upper, step)

    def visit_Assign(self, node: ast.Assign, **kwargs: Any) -> Any:
        if len(node.targets) != 1:
            raise SyntaxError('can only assign a single expression')
        if not isinstance(node.targets[0], ast.Name):
            raise SyntaxError('left hand side of an assignment must be a single name')
        if self.env.target is None:
            raise ValueError('cannot assign without a target object')
        try:
            assigner = self.visit(node.targets[0], **kwargs)
        except UndefinedVariableError:
            assigner = node.targets[0].id
        self.assigner = getattr(assigner, 'name', assigner)
       