"""manage PyTables query interface via Expressions"""
from __future__ import annotations
import ast
from decimal import Decimal, InvalidOperation
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union
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
from pandas.core.indexes.base import Index
from pandas.io.formats.printing import pprint_thing, pprint_thing_encoded
if TYPE_CHECKING:
    from pandas._typing import Self, npt

class PyTablesScope(_scope.Scope):
    __slots__ = ('queryables',)
    queryables: Dict[str, Any]

    def __init__(self, level, global_dict=None, local_dict=None, queryables=None):
        super().__init__(level + 1, global_dict=global_dict, local_dict=local_dict)
        self.queryables = queryables or {}

class Term(ops.Term):
    env: PyTablesScope

    def __new__(cls, name, env, side=None, encoding=None):
        if isinstance(name, str):
            klass = cls
        else:
            klass = Constant
        return object.__new__(klass)

    def __init__(self, name, env, side=None, encoding=None):
        super().__init__(name, env, side=side, encoding=encoding)

    def _resolve_name(self):
        if self.side == 'left':
            if self.name not in self.env.queryables:
                raise NameError(f'name {self.name!r} is not defined')
            return self.name
        try:
            return self.env.resolve(self.name, is_local=False)
        except UndefinedVariableError:
            return self.name

    @property
    def value(self):
        return self._value

class Constant(Term):

    def __init__(self, name, env, side=None, encoding=None):
        assert isinstance(env, PyTablesScope), type(env)
        super().__init__(name, env, side=side, encoding=encoding)

    def _resolve_name(self):
        return self._name

class BinOp(ops.BinOp):
    _max_selectors: ClassVar[int] = 31
    op: str
    queryables: Dict[str, Any]
    condition: Optional[str]

    def __init__(self, op, lhs, rhs, queryables, encoding):
        super().__init__(op, lhs, rhs)
        self.queryables = queryables
        self.encoding = encoding
        self.condition: Optional[str] = None

    def _disallow_scalar_only_bool_ops(self):
        pass

    def prune(self, klass):

        def pr(left, right):
            """create and return a new specialized BinOp from myself"""
            if left is None:
                return right
            elif right is None:
                return left
            k = klass
            if isinstance(left, ConditionBinOp):
                if isinstance(right, ConditionBinOp):
                    k = JointConditionBinOp
                elif isinstance(left, k):
                    return left
                elif isinstance(right, k):
                    return right
            elif isinstance(left, FilterBinOp):
                if isinstance(right, FilterBinOp):
                    k = JointFilterBinOp
                elif isinstance(left, k):
                    return left
                elif isinstance(right, k):
                    return right
            return k(self.op, left, right, queryables=self.queryables, encoding=self.encoding).evaluate()
        left, right = (self.lhs, self.rhs)
        if is_term(left) and is_term(right):
            res = pr(left.value, right.value)
        elif not is_term(left) and is_term(right):
            res = pr(left.prune(klass), right.value)
        elif is_term(left) and (not is_term(right)):
            res = pr(left.value, right.prune(klass))
        elif not (is_term(left) or is_term(right)):
            res = pr(left.prune(klass), right.prune(klass))
        return res

    def conform(self, rhs):
        """inplace conform rhs"""
        if not is_list_like(rhs):
            rhs = [rhs]
        if isinstance(rhs, np.ndarray):
            rhs = rhs.ravel()
        return rhs

    @property
    def is_valid(self):
        """return True if this is a valid field"""
        return self.lhs in self.queryables

    @property
    def is_in_table(self):
        """
        return True if this is a valid column name for generation (e.g. an
        actual column in the table)
        """
        return self.queryables.get(self.lhs) is not None

    @property
    def kind(self):
        """the kind of my field"""
        return getattr(self.queryables.get(self.lhs), 'kind', None)

    @property
    def meta(self):
        """the meta of my field"""
        return getattr(self.queryables.get(self.lhs), 'meta', None)

    @property
    def metadata(self):
        """the metadata of my field"""
        return getattr(self.queryables.get(self.lhs), 'metadata', None)

    def generate(self, v):
        """create and return the op string for this TermValue"""
        val = v.tostring(self.encoding)
        return f'({self.lhs} {self.op} {val})'

    def convert_value(self, conv_val):
        """
        convert the expression that is in the term to something that is
        accepted by pytables
        """

        def stringify(value):
            if self.encoding is not None:
                return pprint_thing_encoded(value, encoding=self.encoding)
            return pprint_thing(value)
        kind = ensure_decoded(self.kind)
        meta = ensure_decoded(self.meta)
        if kind == 'datetime' or (kind and kind.startswith('datetime64')):
            if isinstance(conv_val, (int, float)):
                conv_val = stringify(conv_val)
            conv_val = ensure_decoded(conv_val)
            conv_val = Timestamp(conv_val).as_unit('ns')
            if conv_val.tz is not None:
                conv_val = conv_val.tz_convert('UTC')
            return TermValue(conv_val, conv_val._value, kind)
        elif kind in ('timedelta64', 'timedelta'):
            if isinstance(conv_val, str):
                conv_val = Timedelta(conv_val)
            else:
                conv_val = Timedelta(conv_val, unit='s')
            conv_val = conv_val.as_unit('ns')._value
            return TermValue(int(conv_val), conv_val, kind)
        elif meta == 'category':
            metadata = extract_array(self.metadata, extract_numpy=True)
            result: Union[npt.NDArray[np.intp], np.intp, int]
            if conv_val not in metadata:
                result = -1
            else:
                result = metadata.searchsorted(conv_val, side='left')
            return TermValue(result, result, 'integer')
        elif kind == 'integer':
            try:
                v_dec = Decimal(conv_val)
            except InvalidOperation:
                float(conv_val)
            else:
                conv_val = int(v_dec.to_integral_exact(rounding='ROUND_HALF_EVEN'))
            return TermValue(conv_val, conv_val, kind)
        elif kind == 'float':
            conv_val = float(conv_val)
            return TermValue(conv_val, conv_val, kind)
        elif kind == 'bool':
            if isinstance(conv_val, str):
                conv_val = conv_val.strip().lower() not in ['false', 'f', 'no', 'n', 'none', '0', '[]', '{}', '']
            else:
                conv_val = bool(conv_val)
            return TermValue(conv_val, conv_val, kind)
        elif isinstance(conv_val, str):
            return TermValue(conv_val, stringify(conv_val), 'string')
        else:
            raise TypeError(f'Cannot compare {conv_val} of type {type(conv_val)} to {kind} column')

    def convert_values(self):
        pass

class FilterBinOp(BinOp):
    filter: Optional[Tuple[Any, Callable[[Any, Any], Any], Index]] = None

    def __repr__(self):
        if self.filter is None:
            return 'Filter: Not Initialized'
        return pprint_thing(f'[Filter : [{self.filter[0]}] -> [{self.filter[1]}]')

    def invert(self):
        """invert the filter"""
        if self.filter is not None:
            self.filter = (self.filter[0], self.generate_filter_op(invert=True), self.filter[2])
        return self

    def format(self):
        """return the actual filter format"""
        return [self.filter] if self.filter else []

    def evaluate(self):
        if not self.is_valid:
            raise ValueError(f'query term is not valid [{self}]')
        rhs = self.conform(self.rhs)
        values = list(rhs)
        if self.is_in_table:
            if self.op in ['==', '!='] and len(values) > self._max_selectors:
                filter_op = self.generate_filter_op()
                self.filter = (self.lhs, filter_op, Index(values))
                return self
            return None
        if self.op in ['==', '!=']:
            filter_op = self.generate_filter_op()
            self.filter = (self.lhs, filter_op, Index(values))
        else:
            raise TypeError(f'passing a filterable condition to a non-table indexer [{self}]')
        return self

    def generate_filter_op(self, invert=False):
        if self.op == '!=' and (not invert) or (self.op == '==' and invert):
            return lambda axis, vals: ~axis.isin(vals)
        else:
            return lambda axis, vals: axis.isin(vals)

class JointFilterBinOp(FilterBinOp):

    def format(self):
        raise NotImplementedError('unable to collapse Joint Filters')

    def evaluate(self):
        return self

class ConditionBinOp(BinOp):
    condition: Optional[str] = None

    def __repr__(self):
        return pprint_thing(f'[Condition : [{self.condition}]]')

    def invert(self):
        """invert the condition"""
        raise NotImplementedError('cannot use an invert condition when passing to numexpr')

    def format(self):
        """return the actual ne format"""
        return self.condition

    def evaluate(self):
        if not self.is_valid:
            raise ValueError(f'query term is not valid [{self}]')
        if not self.is_in_table:
            return None
        rhs = self.conform(self.rhs)
        values = [self.convert_value(v) for v in rhs]
        if self.op in ['==', '!=']:
            if len(values) <= self._max_selectors:
                vs = [self.generate(v) for v in values]
                self.condition = f'({' | '.join(vs)})'
            else:
                return None
        else:
            self.condition = self.generate(values[0])
        return self

class JointConditionBinOp(ConditionBinOp):

    def evaluate(self):
        self.condition = f'({self.lhs.condition} {self.op} {self.rhs.condition})'
        return self

class UnaryOp(ops.UnaryOp):

    def prune(self, klass):
        if self.op != '~':
            raise NotImplementedError('UnaryOp only support invert type ops')
        operand = self.operand
        operand = operand.prune(klass)
        if operand is not None and (issubclass(klass, ConditionBinOp) and operand.condition is not None or (not issubclass(klass, ConditionBinOp) and issubclass(klass, FilterBinOp) and (operand.filter is not None))):
            return operand.invert()
        return None

class PyTablesExprVisitor(BaseExprVisitor):
    const_type: ClassVar[type[ops.Term]] = Constant
    term_type: ClassVar[type[Term]] = Term
    binary_ops: ClassVar[List[str]] = ops.BINARY_OPS
    unary_ops: ClassVar[List[str]] = ops.UNARY_OPS
    binary_op_nodes_map: ClassVar[Dict[str, type[ast.operator]]] = {}

    def __init__(self, env, engine, parser, **kwargs: Any):
        super().__init__(env, engine, parser)
        for bin_op in self.binary_ops:
            bin_node = self.binary_op_nodes_map.get(bin_op, ast.BinOp)
            setattr(self, f'visit_{bin_node.__name__}', lambda node, bin_op=bin_op: partial(BinOp, bin_op, **kwargs))

    def visit_UnaryOp(self, node, **kwargs: Any):
        if isinstance(node.op, (ast.Not, ast.Invert)):
            operand = self.visit(node.operand)
            if operand is not None:
                return UnaryOp('~', operand)
        elif isinstance(node.op, ast.USub):
            operand = self.visit(node.operand)
            if operand is not None:
                return self.const_type(-operand.value, self.env)
        elif isinstance(node.op, ast.UAdd):
            raise NotImplementedError('Unary addition not supported')
        return None

    def visit_Index(self, node, **kwargs: Any):
        return self.visit(node.value).value

    def visit_Assign(self, node, **kwargs: Any):
        cmpr = ast.Compare(ops=[ast.Eq()], left=node.targets[0], comparators=[node.value])
        return self.visit(cmpr)

    def visit_Subscript(self, node, **kwargs: Any):
        value = self.visit(node.value)
        slobj = self.visit(node.slice)
        try:
            value = value.value
        except AttributeError:
            pass
        if isinstance(slobj, Term):
            slobj = slobj.value
        try:
            return self.const_type(value[slobj], self.env)
        except TypeError as err:
            raise ValueError(f'cannot subscript {value!r} with {slobj!r}') from err

    def visit_Attribute(self, node, **kwargs: Any):
        attr = node.attr
        value = node.value
        ctx = type(node.ctx)
        if ctx == ast.Load:
            resolved = self.visit(value)
            try:
                resolved = resolved.value
            except AttributeError:
                pass
            try:
                return self.term_type(getattr(resolved, attr), self.env)
            except AttributeError:
                if isinstance(value, ast.Name) and value.id == attr:
                    return resolved
        raise ValueError(f'Invalid Attribute context {ctx.__name__}')

    def translate_In(self, op):
        return ast.Eq() if isinstance(op, ast.In) else op

    def _rewrite_membership_op(self, node, left, right):
        return (self.visit(node.op), node.op, left, right)

def _validate_where(w):
    """
    Validate that the where statement is of the right type.

    The type may either be String, Expr, or list-like of Exprs.

    Parameters
    ----------
    w : String term expression, Expr, or list-like of Exprs.

    Returns
    -------
    where : The original where clause if the check was successful.

    Raises
    ------
    TypeError : An invalid data type was passed in for w (e.g. dict).
    """
    if not (isinstance(w, (PyTablesExpr, str)) or is_list_like(w)):
        raise TypeError('where must be passed as a string, PyTablesExpr, or list-like of PyTablesExpr')
    return w

class PyTablesExpr(expr.Expr):
    """
    Hold a pytables-like expression, comprised of possibly multiple 'terms'.

    Parameters
    ----------
    where : string term expression, PyTablesExpr, or list-like of PyTablesExprs
    queryables : a "kinds" map (dict of column name -> kind), or None if column
        is non-indexable
    encoding : an encoding that will encode the query terms

    Returns
    -------
    a PyTablesExpr object

    Examples
    --------
    'index>=date'
    "columns=['A', 'D']"
    'columns=A'
    'columns==A'
    "~(columns=['A','B'])"
    'index>df.index[3] & string="bar"'
    '(index>df.index[3] & index<=df.index[6]) | string="bar"'
    "ts>=Timestamp('2012-02-01')"
    "major_axis>=20130101"
    """
    _visitor: Optional[PyTablesExprVisitor]
    env: PyTablesScope
    expr: str

    def __init__(self, where, queryables=None, encoding=None, scope_level=0):
        where = _validate_where(where)
        self.encoding = encoding
        self.condition: Optional[str] = None
        self.filter: Optional[Tuple[Any, Callable[[Any, Any], Any], Index]] = None
        self.terms: Optional[Union[BinOp, ConditionBinOp, FilterBinOp, JointConditionBinOp, JointFilterBinOp, UnaryOp]] = None
        self._visitor = None
        local_dict: Optional[_scope.DeepChainMap[Any, Any]] = None
        if isinstance(where, PyTablesExpr):
            local_dict = where.env.scope
            _where = where.expr
        elif is_list_like(where):
            where = list(where)
            for idx, w in enumerate(where):
                if isinstance(w, PyTablesExpr):
                    local_dict = w.env.scope
                else:
                    where[idx] = _validate_where(w)
            _where = ' & '.join([f'({w})' for w in com.flatten(where)])
        else:
            _where = where
        self.expr = _where
        self.env = PyTablesScope(scope_level + 1, global_dict=None, local_dict=local_dict)
        if queryables is not None and isinstance(self.expr, str):
            self.env.queryables.update(queryables)
            self._visitor = PyTablesExprVisitor(self.env, engine='pytables', parser='pytables', queryables=queryables, encoding=encoding)
            self.terms = self.parse()

    def __repr__(self):
        if self.terms is not None:
            return pprint_thing(self.terms)
        return pprint_thing(self.expr)

    def evaluate(self):
        """create and return the numexpr condition and filter"""
        try:
            self.condition = self.terms.prune(ConditionBinOp)
        except AttributeError as err:
            raise ValueError(f'cannot process expression [{self.expr}], [{self}] is not a valid condition') from err
        try:
            self.filter = self.terms.prune(FilterBinOp)
        except AttributeError as err:
            raise ValueError(f'cannot process expression [{self.expr}], [{self}] is not a valid filter') from err
        return (self.condition, self.filter)

class TermValue:
    """hold a term value the we use to construct a condition/filter"""

    def __init__(self, value, converted, kind):
        assert isinstance(kind, str), kind
        self.value: Any = value
        self.converted: Any = converted
        self.kind: str = kind

    def tostring(self, encoding):
        """quote the string if not encoded else encode and return"""
        if self.kind == 'string':
            if encoding is not None:
                return str(self.converted)
            return f'"{self.converted}"'
        elif self.kind == 'float':
            return repr(self.converted)
        return str(self.converted)

def maybe_expression(s):
    """loose checking if s is a pytables-acceptable expression"""
    if not isinstance(s, str):
        return False
    operations = PyTablesExprVisitor.binary_ops + PyTablesExprVisitor.unary_ops + ('=',)
    return any((op in s for op in operations))