"""
:func:`~pandas.eval` parsers.
"""
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


def func_y8fuuh5t(tok):
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
    return toknum, '==' if tokval == '=' else tokval


def func_sgaf5d32(tok):
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
            return tokenize.NAME, 'and'
        elif tokval == '|':
            return tokenize.NAME, 'or'
        return toknum, tokval
    return toknum, tokval


def func_jax6ebwv(tok):
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
        return tokenize.OP, LOCAL_TAG
    return toknum, tokval


def func_netq6my4(f, g):
    """
    Compose 2 callables.
    """
    return lambda *args, **kwargs: f(g(*args, **kwargs))


def func_x0jxop4z(*funcs):
    """
    Compose 2 or more callables.
    """
    assert len(funcs) > 1, 'At least 2 callables must be passed to compose'
    return reduce(_compose2, funcs)


def func_upu1eds3(source, f=func_x0jxop4z(_replace_locals,
    _replace_booleans, _rewrite_assign, clean_backtick_quoted_toks)):
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
    return tokenize.untokenize(f(x) for x in tokenize_string(source))


def func_0fg7d3ug(t):
    """
    Factory for a type checking function of type ``t`` or tuple of types.
    """
    return lambda x: isinstance(x.value, t)


_is_list = func_0fg7d3ug(list)
_is_str = func_0fg7d3ug(str)
_all_nodes = frozenset(node for node in (getattr(ast, name) for name in dir
    (ast)) if isinstance(node, type) and issubclass(node, ast.AST))


def func_q7shhtrh(superclass, all_nodes=_all_nodes):
    """
    Filter out AST nodes that are subclasses of ``superclass``.
    """
    node_names = (node.__name__ for node in all_nodes if issubclass(node,
        superclass))
    return frozenset(node_names)


_all_node_names = frozenset(x.__name__ for x in _all_nodes)
_mod_nodes = func_q7shhtrh(ast.mod)
_stmt_nodes = func_q7shhtrh(ast.stmt)
_expr_nodes = func_q7shhtrh(ast.expr)
_expr_context_nodes = func_q7shhtrh(ast.expr_context)
_boolop_nodes = func_q7shhtrh(ast.boolop)
_operator_nodes = func_q7shhtrh(ast.operator)
_unary_op_nodes = func_q7shhtrh(ast.unaryop)
_cmp_op_nodes = func_q7shhtrh(ast.cmpop)
_comprehension_nodes = func_q7shhtrh(ast.comprehension)
_handler_nodes = func_q7shhtrh(ast.excepthandler)
_arguments_nodes = func_q7shhtrh(ast.arguments)
_keyword_nodes = func_q7shhtrh(ast.keyword)
_alias_nodes = func_q7shhtrh(ast.alias)
_hacked_nodes = frozenset(['Assign', 'Module', 'Expr'])
_unsupported_expr_nodes = frozenset(['Yield', 'GeneratorExp', 'IfExp',
    'DictComp', 'SetComp', 'Repr', 'Lambda', 'Set', 'AST', 'Is', 'IsNot'])
_unsupported_nodes = (_stmt_nodes | _mod_nodes | _handler_nodes |
    _arguments_nodes | _keyword_nodes | _alias_nodes | _expr_context_nodes |
    _unsupported_expr_nodes) - _hacked_nodes
_base_supported_nodes = _all_node_names - _unsupported_nodes | _hacked_nodes
intersection = _unsupported_nodes & _base_supported_nodes
_msg = f'cannot both support and not support {intersection}'
assert not intersection, _msg


def func_54zjf998(node_name):
    """
    Return a function that raises a NotImplementedError with a passed node name.
    """

    def func_62znxl73(self, *args, **kwargs):
        raise NotImplementedError(f"'{node_name}' nodes are not implemented")
    return f


_T = TypeVar('_T')


def func_3t5xqf78(nodes):
    """
    Decorator to disallow certain nodes from parsing. Raises a
    NotImplementedError instead.

    Returns
    -------
    callable
    """

    def func_74kukzar(cls):
        cls.unsupported_nodes = ()
        for node in nodes:
            new_method = func_54zjf998(node)
            name = f'visit_{node}'
            cls.unsupported_nodes += name,
            setattr(cls, name, new_method)
        return cls
    return disallowed


def func_q2broc3g(op_class, op_symbol):
    """
    Return a function to create an op class with its symbol already passed.

    Returns
    -------
    callable
    """

    def func_62znxl73(self, node, *args, **kwargs):
        """
        Return a partial function with an Op subclass with an operator already passed.

        Returns
        -------
        callable
        """
        return partial(op_class, op_symbol, *args, **kwargs)
    return f


_op_classes = {'binary': BinOp, 'unary': UnaryOp}


def func_584p78zy(op_classes):
    """
    Decorator to add default implementation of ops.
    """

    def func_62znxl73(cls):
        for op_attr_name, op_class in op_classes.items():
            ops = getattr(cls, f'{op_attr_name}_ops')
            ops_map = getattr(cls, f'{op_attr_name}_op_nodes_map')
            for op in ops:
                op_node = ops_map[op]
                if op_node is not None:
                    made_op = func_q2broc3g(op_class, op)
                    setattr(cls, f'visit_{op_node}', made_op)
        return cls
    return f


@func_3t5xqf78(_unsupported_nodes)
@func_584p78zy(_op_classes)
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
    const_type = Constant
    term_type = Term
    binary_ops = CMP_OPS_SYMS + BOOL_OPS_SYMS + ARITH_OPS_SYMS
    binary_op_nodes = ('Gt', 'Lt', 'GtE', 'LtE', 'Eq', 'NotEq', 'In',
        'NotIn', 'BitAnd', 'BitOr', 'And', 'Or', 'Add', 'Sub', 'Mult',
        'Div', 'Pow', 'FloorDiv', 'Mod')
    binary_op_nodes_map = dict(zip(binary_ops, binary_op_nodes))
    unary_ops = UNARY_OPS_SYMS
    unary_op_nodes = 'UAdd', 'USub', 'Invert', 'Not'
    unary_op_nodes_map = dict(zip(unary_ops, unary_op_nodes))
    rewrite_map = {ast.Eq: ast.In, ast.NotEq: ast.NotIn, ast.In: ast.In,
        ast.NotIn: ast.NotIn}

    def __init__(self, env, engine, parser, preparser=_preparse):
        self.env = env
        self.engine = engine
        self.parser = parser
        self.preparser = preparser
        self.assigner = None

    def func_pfrsxk6o(self, node, **kwargs):
        if isinstance(node, str):
            clean = self.preparser(node)
            try:
                node = ast.fix_missing_locations(ast.parse(clean))
            except SyntaxError as e:
                if any(iskeyword(x) for x in clean.split()):
                    e.msg = (
                        'Python keyword not valid identifier in numexpr query')
                raise e
        method = f'visit_{type(node).__name__}'
        visitor = getattr(self, method)
        return visitor(node, **kwargs)

    def func_jszipbqb(self, node, **kwargs):
        if len(node.body) != 1:
            raise SyntaxError('only a single expression is allowed')
        expr = node.body[0]
        return self.visit(expr, **kwargs)

    def func_b65dbn8y(self, node, **kwargs):
        return self.visit(node.value, **kwargs)

    def func_m8wasilv(self, node, left, right):
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

    def func_81qpti6n(self, node, left=None, right=None):
        if left is None:
            left = self.visit(node.left, side='left')
        if right is None:
            right = self.visit(node.right, side='right')
        op, op_class, left, right = self._rewrite_membership_op(node, left,
            right)
        return op, op_class, left, right

    def func_n4tipv6t(self, left, right):
        f32 = np.dtype(np.float32)
        if left.is_scalar and hasattr(left, 'value'
            ) and not right.is_scalar and right.return_type == f32:
            name = self.env.add_tmp(np.float32(left.value))
            left = self.term_type(name, self.env)
        if right.is_scalar and hasattr(right, 'value'
            ) and not left.is_scalar and left.return_type == f32:
            name = self.env.add_tmp(np.float32(right.value))
            right = self.term_type(name, self.env)
        return left, right

    def func_klizmj28(self, binop, eval_in_python):
        return binop.evaluate(self.env, self.engine, self.parser, self.
            term_type, eval_in_python)

    def func_a2tjhx9k(self, op, op_class, lhs, rhs, eval_in_python=('in',
        'not in'), maybe_eval_in_python=('==', '!=', '<', '>', '<=', '>=')):
        res = op(lhs, rhs)
        if res.has_invalid_return_type:
            raise TypeError(
                f"unsupported operand type(s) for {res.op}: '{lhs.type}' and '{rhs.type}'"
                )
        if self.engine != 'pytables' and (res.op in CMP_OPS_SYMS and
            getattr(lhs, 'is_datetime', False) or getattr(rhs,
            'is_datetime', False)):
            return self._maybe_eval(res, self.binary_ops)
        if res.op in eval_in_python:
            return self._maybe_eval(res, eval_in_python)
        elif self.engine != 'pytables':
            if getattr(lhs, 'return_type', None) == object or is_string_dtype(
                getattr(lhs, 'return_type', None)) or getattr(rhs,
                'return_type', None) == object or is_string_dtype(getattr(
                rhs, 'return_type', None)):
                return self._maybe_eval(res, eval_in_python +
                    maybe_eval_in_python)
        return res

    def func_r7u90jwf(self, node, **kwargs):
        op, op_class, left, right = self._maybe_transform_eq_ne(node)
        left, right = self._maybe_downcast_constants(left, right)
        return self._maybe_evaluate_binop(op, op_class, left, right)

    def func_wo1obbsk(self, node, **kwargs):
        op = self.visit(node.op)
        operand = self.visit(node.operand)
        return op(operand)

    def func_s97xxk9l(self, node, **kwargs):
        return self.term_type(node.id, self.env, **kwargs)

    def func_6iaadf4h(self, node, **kwargs):
        return self.const_type(node.value, self.env)

    def func_k54hfg4o(self, node, **kwargs):
        return self.const_type(node.value, self.env)

    def func_cgxclb95(self, node, **kwargs):
        return self.const_type(node.value, self.env)

    def func_fikxvsfo(self, node, **kwargs):
        name = self.env.add_tmp(node.s)
        return self.term_type(name, self.env)

    def func_7lvu5gcy(self, node, **kwargs):
        name = self.env.add_tmp([self.visit(e)(self.env) for e in node.elts])
        return self.term_type(name, self.env)
    visit_Tuple = visit_List

    def func_ft6sn6xu(self, node, **kwargs):
        """df.index[4]"""
        return self.visit(node.value)

    def func_n9c4253c(self, node, **kwargs):
        from pandas import eval as pd_eval
        value = self.visit(node.value)
        slobj = self.visit(node.slice)
        result = pd_eval(slobj, local_dict=self.env, engine=self.engine,
            parser=self.parser)
        try:
            v = value.value[result]
        except AttributeError:
            lhs = pd_eval(value, local_dict=self.env, engine=self.engine,
                parser=self.parser)
            v = lhs[result]
        name = self.env.add_tmp(v)
        return self.term_type(name, env=self.env)

    def func_uugplu7j(self, node, **kwargs):
        """df.index[slice(4,6)]"""
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

    def func_ax4dwwni(self, node, **kwargs):
        """
        support a single assignment node, like

        c = a + b

        set the assigner at the top level, must be a Name node which
        might or might not exist in the resolvers

        """
        if len(node.targets) != 1:
            raise SyntaxError('can only assign a single expression')
        if not isinstance(node.targets[0], ast.Name):
            raise SyntaxError(
                'left hand side of an assignment must be a single name')
        if self.env.target is None:
            raise ValueError('cannot assign without a target object')
        try:
            assigner = self.visit(node.targets[0], **kwargs)
        except UndefinedVariableError:
            assigner = node.targets[0].id
        self.assigner = getattr(assigner, 'name', assigner)
        if self.assigner is None:
            raise SyntaxError(
                'left hand side of an assignment must be a single resolvable name'
                )
        return self.visit(node.value, **kwargs)

    def func_3okhcnt1(self, node, **kwargs):
        attr = node.attr
        value = node.value
        ctx = node.ctx
        if isinstance(ctx, ast.Load):
            resolved = self.visit(value).value
            try:
                v = getattr(resolved, attr)
                name = self.env.add_tmp(v)
                return self.term_type(name, self.env)
            except AttributeError:
                if isinstance(value, ast.Name) and value.id == attr:
                    return resolved
                raise
        raise ValueError(f'Invalid Attribute context {type(ctx).__name__}')

    def func_qmbwfoyz(self, node, side=None, **kwargs):
        if isinstance(node.func, ast.Attribute
            ) and node.func.attr != '__call__':
            res = self.visit_Attribute(node.func)
        elif not isinstance(node.func, ast.Name):
            raise TypeError('Only named functions are supported')
        else:
            try:
                res = self.visit(node.func)
            except UndefinedVariableError:
                try:
                    res = FuncNode(node.func.id)
                except ValueError:
                    raise
        if res is None:
            raise ValueError(f'Invalid function call {node.func.id}')
        if hasattr(res, 'value'):
            res = res.value
        if isinstance(res, FuncNode):
            new_args = [self.visit(arg) for arg in node.args]
            if node.keywords:
                raise TypeError(
                    f'Function "{res.name}" does not support keyword arguments'
                    )
            return res(*new_args)
        else:
            new_args = [self.visit(arg)(self.env) for arg in node.args]
            for key in node.keywords:
                if not isinstance(key, ast.keyword):
                    raise ValueError(
                        f"keyword error in function call '{node.func.id}'")
                if key.arg:
                    kwargs[key.arg] = self.visit(key.value)(self.env)
            name = self.env.add_tmp(res(*new_args, **kwargs))
            return self.term_type(name=name, env=self.env)

    def func_vsbsczoj(self, op):
        return op

    def func_dyuqzr8a(self, node, **kwargs):
        ops = node.ops
        comps = node.comparators
        if len(comps) == 1:
            op = self.translate_In(ops[0])
            binop = ast.BinOp(op=op, left=node.left, right=comps[0])
            return self.visit(binop)
        left = node.left
        values = []
        for op, comp in zip(ops, comps):
            new_node = self.visit(ast.Compare(comparators=[comp], left=left,
                ops=[self.translate_In(op)]))
            left = comp
            values.append(new_node)
        return self.visit(ast.BoolOp(op=ast.And(), values=values))

    def func_3c2w5g69(self, bop):
        if isinstance(bop, (Op, Term)):
            return bop
        return self.visit(bop)

    def func_dp85ss82(self, node, **kwargs):

        def func_8qwq32e5(x, y):
            lhs = self._try_visit_binop(x)
            rhs = self._try_visit_binop(y)
            op, op_class, lhs, rhs = self._maybe_transform_eq_ne(node, lhs, rhs
                )
            return self._maybe_evaluate_binop(op, node.op, lhs, rhs)
        operands = node.values
        return reduce(visitor, operands)


_python_not_supported = frozenset(['Dict', 'BoolOp', 'In', 'NotIn'])
_numexpr_supported_calls = frozenset(REDUCTIONS + MATHOPS)


@func_3t5xqf78((_unsupported_nodes | _python_not_supported) - (
    _boolop_nodes | frozenset(['BoolOp', 'Attribute', 'In', 'NotIn', 'Tuple']))
    )
class PandasExprVisitor(BaseExprVisitor):

    def __init__(self, env, engine, parser, preparser=partial(_preparse, f=
        func_x0jxop4z(_replace_locals, _replace_booleans,
        clean_backtick_quoted_toks))):
        super().__init__(env, engine, parser, preparser)


@func_3t5xqf78(_unsupported_nodes | _python_not_supported | frozenset(['Not']))
class PythonExprVisitor(BaseExprVisitor):

    def __init__(self, env, engine, parser, preparser=lambda source, f=None:
        source):
        super().__init__(env, engine, parser, preparser=preparser)


class Expr:
    """
    Object encapsulating an expression.

    Parameters
    ----------
    expr : str
    engine : str, optional, default 'numexpr'
    parser : str, optional, default 'pandas'
    env : Scope, optional, default None
    level : int, optional, default 2
    """

    def __init__(self, expr, engine='numexpr', parser='pandas', env=None,
        level=0):
        self.expr = expr
        self.env = env or Scope(level=level + 1)
        self.engine = engine
        self.parser = parser
        self._visitor = PARSERS[parser](self.env, self.engine, self.parser)
        self.terms = self.parse()

    @property
    def func_q50f5v9v(self):
        return getattr(self._visitor, 'assigner', None)

    def __call__(self):
        return self.terms(self.env)

    def __repr__(self):
        return printing.pprint_thing(self.terms)

    def __len__(self):
        return len(self.expr)

    def func_ho0p3hlk(self):
        """
        Parse an expression.
        """
        return self._visitor.visit(self.expr)

    @property
    def func_5ov4ul59(self):
        """
        Get the names in an expression.
        """
        if is_term(self.terms):
            return frozenset([self.terms.name])
        return frozenset(term.name for term in com.flatten(self.terms))


PARSERS = {'python': PythonExprVisitor, 'pandas': PandasExprVisitor}
