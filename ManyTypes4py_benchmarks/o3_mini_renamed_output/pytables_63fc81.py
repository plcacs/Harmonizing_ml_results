from __future__ import annotations
import ast
from decimal import Decimal, InvalidOperation
from functools import partial
from typing import Any, Callable, ClassVar, Dict, List, NoReturn, Optional, Tuple, Type, Union
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
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pandas._typing import npt

class PyTablesScope(_scope.Scope):
    __slots__ = 'queryables',

    def __init__(self, level: int, global_dict: Optional[Dict[str, Any]] = None, local_dict: Optional[Dict[str, Any]] = None, queryables: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(level + 1, global_dict=global_dict, local_dict=local_dict)
        self.queryables: Dict[str, Any] = queryables or {}

class Term(ops.Term):

    def __new__(cls: Type[Term], name: Any, env: PyTablesScope, side: Optional[str] = None, encoding: Optional[str] = None) -> Term:
        if isinstance(name, str):
            klass: Type[Term] = cls
        else:
            klass = Constant
        return object.__new__(klass)

    def __init__(self, name: Any, env: PyTablesScope, side: Optional[str] = None, encoding: Optional[str] = None) -> None:
        super().__init__(name, env, side=side, encoding=encoding)

    def func_vi6m90be(self) -> Any:
        if self.side == 'left':
            if self.name not in self.env.queryables:
                raise NameError(f'name {self.name!r} is not defined')
            return self.name
        try:
            return self.env.resolve(self.name, is_local=False)
        except UndefinedVariableError:
            return self.name

    @property
    def func_qoxvb5y7(self) -> Any:
        return self._value

class Constant(Term):

    def __init__(self, name: Any, env: PyTablesScope, side: Optional[str] = None, encoding: Optional[str] = None) -> None:
        assert isinstance(env, PyTablesScope), type(env)
        super().__init__(name, env, side=side, encoding=encoding)

    def func_vi6m90be(self) -> Any:
        return self._name

class BinOp(ops.BinOp):
    _max_selectors: int = 31

    def __init__(self, op: str, lhs: Any, rhs: Any, queryables: Dict[str, Any], encoding: Optional[str]) -> None:
        super().__init__(op, lhs, rhs)
        self.queryables: Dict[str, Any] = queryables
        self.encoding: Optional[str] = encoding
        self.condition: Optional[Any] = None

    def func_ay5l3zjm(self) -> None:
        pass

    def func_5512fwle(self, klass: Type) -> Any:
        def func_ia84vlq3(left: Any, right: Any) -> Any:
            """create and return a new specialized BinOp from mysel"""
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
        left, right = self.lhs, self.rhs
        if is_term(left) and is_term(right):
            res = func_ia84vlq3(left.value, right.value)
        elif not is_term(left) and is_term(right):
            res = func_ia84vlq3(left.prune(klass), right.value)
        elif is_term(left) and not is_term(right):
            res = func_ia84vlq3(left.value, right.prune(klass))
        elif not (is_term(left) or is_term(right)):
            res = func_ia84vlq3(left.prune(klass), right.prune(klass))
        return res

    def func_f55c0179(self, rhs: Any) -> Any:
        """inplace conform rhs"""
        if not is_list_like(rhs):
            rhs = [rhs]
        if isinstance(rhs, np.ndarray):
            rhs = rhs.ravel()
        return rhs

    @property
    def func_vrbq2eng(self) -> bool:
        """return True if this is a valid field"""
        return self.lhs in self.queryables

    @property
    def func_6t38l2e6(self) -> bool:
        """
        return True if this is a valid column name for generation (e.g. an
        actual column in the table)
        """
        return self.queryables.get(self.lhs) is not None

    @property
    def func_uklceppt(self) -> Optional[Any]:
        """the kind of my field"""
        return getattr(self.queryables.get(self.lhs), 'kind', None)

    @property
    def func_17xh1im2(self) -> Optional[Any]:
        """the meta of my field"""
        return getattr(self.queryables.get(self.lhs), 'meta', None)

    @property
    def func_nmx7mqio(self) -> Optional[Any]:
        """the metadata of my field"""
        return getattr(self.queryables.get(self.lhs), 'metadata', None)

    def func_owh3jfp9(self, v: Any) -> str:
        """create and return the op string for this TermValue"""
        val: str = v.tostring(self.encoding)
        return f'({self.lhs} {self.op} {val})'

    def func_vz99i461(self, conv_val: Any) -> TermValue:
        """
        convert the expression that is in the term to something that is
        accepted by pytables
        """
        def func_qct7x1hi(value: Any) -> str:
            if self.encoding is not None:
                return pprint_thing_encoded(value, encoding=self.encoding)
            return pprint_thing(value)
        kind: Optional[str] = ensure_decoded(self.kind)
        meta: Optional[str] = ensure_decoded(self.meta)
        if kind == 'datetime' or (kind and self.func_uklceppt and str(self.func_uklceppt).startswith('datetime64')):
            if isinstance(conv_val, (int, float)):
                conv_val = func_qct7x1hi(conv_val)
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
            if conv_val not in metadata:
                result = -1
            else:
                result = self.func_nmx7mqio.searchsorted(conv_val, side='left')  # type: ignore
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
                conv_val = conv_val.strip().lower() not in ['false', '', 'no', 'n', 'none', '0', '[]', '{}', '']
            else:
                conv_val = bool(conv_val)
            return TermValue(conv_val, conv_val, kind)
        elif isinstance(conv_val, str):
            return TermValue(conv_val, func_qct7x1hi(conv_val), 'string')
        else:
            raise TypeError(f'Cannot compare {conv_val} of type {type(conv_val)} to {kind} column')

    def func_prxokmg7(self) -> None:
        pass

class FilterBinOp(BinOp):
    filter: Optional[Tuple[Any, Any, Any]] = None

    def __repr__(self) -> str:
        if self.filter is None:
            return 'Filter: Not Initialized'
        return pprint_thing(f'[Filter : [{self.filter[0]}] -> [{self.filter[1]}]')

    def func_lq1xiwgj(self) -> FilterBinOp:
        """invert the filter"""
        if self.filter is not None:
            self.filter = (self.filter[0], self.generate_filter_op(invert=True), self.filter[2])
        return self

    def func_9fh2p7ks(self) -> List[Any]:
        """return the actual filter format"""
        return [self.filter]

    def func_hpbgcqu3(self) -> Optional[Any]:
        if not self.is_valid:
            raise ValueError(f'query term is not valid [{self}]')
        rhs = self.conform(self.rhs)
        values: List[Any] = list(rhs)
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

    def func_bgn0vxaj(self, invert: bool = False) -> Callable[[Any, Any], Any]:
        if self.op == '!=' and not invert or self.op == '==' and invert:
            return lambda axis, vals: ~axis.isin(vals)
        else:
            return lambda axis, vals: axis.isin(vals)

class JointFilterBinOp(FilterBinOp):

    def func_9fh2p7ks(self) -> Any:
        raise NotImplementedError('unable to collapse Joint Filters')

    def func_hpbgcqu3(self) -> JointFilterBinOp:
        return self

class ConditionBinOp(BinOp):

    def __repr__(self) -> str:
        return pprint_thing(f'[Condition : [{self.condition}]]')

    def func_lq1xiwgj(self) -> Any:
        """invert the condition"""
        raise NotImplementedError('cannot use an invert condition when passing to numexpr')

    def func_9fh2p7ks(self) -> Any:
        """return the actual ne format"""
        return self.condition

    def func_hpbgcqu3(self) -> Any:
        if not self.is_valid:
            raise ValueError(f'query term is not valid [{self}]')
        if not self.is_in_table:
            return None
        rhs = self.conform(self.rhs)
        values = [self.convert_value(v) for v in rhs]
        if self.op in ['==', '!=']:
            if len(values) <= self._max_selectors:
                vs = [self.generate(v) for v in values]
                self.condition = '({' + ' | '.join(vs) + '})'
            else:
                return None
        else:
            self.condition = self.generate(values[0])
        return self

class JointConditionBinOp(ConditionBinOp):

    def func_hpbgcqu3(self) -> JointConditionBinOp:
        self.condition = f'({self.lhs.condition} {self.op} {self.rhs.condition})'
        return self

class UnaryOp(ops.UnaryOp):

    def func_5512fwle(self, klass: Type) -> Optional[Any]:
        if self.op != '~':
            raise NotImplementedError('UnaryOp only support invert type ops')
        operand = self.operand
        operand = operand.prune(klass)
        if operand is not None and ((issubclass(klass, ConditionBinOp) and operand.condition is not None) or (not issubclass(klass, ConditionBinOp) and issubclass(klass, FilterBinOp) and operand.filter is not None)):
            return operand.invert()
        return None

class PyTablesExprVisitor(BaseExprVisitor):
    const_type: ClassVar[Type[Constant]] = Constant
    term_type: ClassVar[Type[Term]] = Term

    def __init__(self, env: PyTablesScope, engine: Any, parser: Any, **kwargs: Any) -> None:
        super().__init__(env, engine, parser)
        for bin_op in self.binary_ops:
            bin_node = self.binary_op_nodes_map[bin_op]
            setattr(self, f'visit_{bin_node}', lambda node, bin_op=bin_op: partial(BinOp, bin_op, **kwargs))

    def func_5pjjyjey(self, node: ast.AST, **kwargs: Any) -> Any:
        if isinstance(node.op, (ast.Not, ast.Invert)):
            return UnaryOp('~', self.visit(node.operand))
        elif isinstance(node.op, ast.USub):
            return self.const_type(-self.visit(node.operand).value, self.env)
        elif isinstance(node.op, ast.UAdd):
            raise NotImplementedError('Unary addition not supported')
        return None

    def func_26hn0fcy(self, node: ast.AST, **kwargs: Any) -> Any:
        return self.visit(node.value).value

    def func_zi9cb7kz(self, node: ast.AST, **kwargs: Any) -> Any:
        cmpr = ast.Compare(ops=[ast.Eq()], left=node.targets[0], comparators=[node.value])
        return self.visit(cmpr)

    def func_p7d7vhto(self, node: ast.AST, **kwargs: Any) -> Any:
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

    def func_ol6770tf(self, node: ast.AST, **kwargs: Any) -> Any:
        attr: str = node.attr
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

    def func_rp3z6kqc(self, op: Any) -> Any:
        return ast.Eq() if isinstance(op, ast.In) else op

    def func_q347mtjd(self, node: ast.AST, left: Any, right: Any) -> Tuple[Any, Any, Any, Any]:
        return self.visit(node.op), node.op, left, right

def func_spwb9twh(w: Any) -> Any:
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
    """

    def __init__(self, where: Union[str, PyTablesExpr, List[Any]], queryables: Optional[Dict[str, Any]] = None, encoding: Optional[str] = None, scope_level: int = 0) -> None:
        where = func_spwb9twh(where)
        self.encoding: Optional[str] = encoding
        self.condition: Optional[Any] = None
        self.filter: Optional[Any] = None
        self.terms: Optional[Any] = None
        self._visitor: Optional[PyTablesExprVisitor] = None
        local_dict: Optional[Dict[str, Any]] = None
        if isinstance(where, PyTablesExpr):
            local_dict = where.env.scope
            _where: Union[str, Any] = where.expr
        elif is_list_like(where):
            where = list(where)
            for idx, w in enumerate(where):
                if isinstance(w, PyTablesExpr):
                    local_dict = w.env.scope
                else:
                    where[idx] = func_spwb9twh(w)
            _where = ' & '.join(['(' + w + ')' for w in com.flatten(where)])
        else:
            _where = where
        self.expr: Union[str, Any] = _where
        self.env: PyTablesScope = PyTablesScope(scope_level + 1, local_dict=local_dict)
        if queryables is not None and isinstance(self.expr, str):
            self.env.queryables.update(queryables)
            self._visitor = PyTablesExprVisitor(self.env, queryables, parser='pytables', engine='pytables', encoding=encoding)
            self.terms = self.parse()

    def __repr__(self) -> str:
        if self.terms is not None:
            return pprint_thing(self.terms)
        return pprint_thing(self.expr)

    def func_hpbgcqu3(self) -> Tuple[Any, Any]:
        """create and return the numexpr condition and filter"""
        try:
            self.condition = self.terms.prune(ConditionBinOp)
        except AttributeError as err:
            raise ValueError(f'cannot process expression [{self.expr}], [{self}] is not a valid condition') from err
        try:
            self.filter = self.terms.prune(FilterBinOp)
        except AttributeError as err:
            raise ValueError(f'cannot process expression [{self.expr}], [{self}] is not a valid filter') from err
        return self.condition, self.filter

class TermValue:
    """hold a term value that we use to construct a condition/filter"""

    def __init__(self, value: Any, converted: Any, kind: str) -> None:
        assert isinstance(kind, str), kind
        self.value: Any = value
        self.converted: Any = converted
        self.kind: str = kind

    def func_dd6a5lxa(self, encoding: Optional[str]) -> str:
        """quote the string if not encoded else encode and return"""
        if self.kind == 'string':
            if encoding is not None:
                return str(self.converted)
            return f'"{self.converted}"'
        elif self.kind == 'float':
            return repr(self.converted)
        return str(self.converted)

def func_w8w8wns6(s: Any) -> bool:
    """loose checking if s is a pytables-acceptable expression"""
    if not isinstance(s, str):
        return False
    operations = (PyTablesExprVisitor.binary_ops + PyTablesExprVisitor.unary_ops + ('=',))
    return any(op in s for op in operations)