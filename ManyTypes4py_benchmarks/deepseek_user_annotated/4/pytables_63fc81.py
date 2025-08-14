"""manage PyTables query interface via Expressions"""

from __future__ import annotations

import ast
from decimal import (
    Decimal,
    InvalidOperation,
)
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Callable,
    Type,
    TypeVar,
    cast,
)

import numpy as np
from numpy.typing import NDArray

from pandas._libs.tslibs import (
    Timedelta,
    Timestamp,
)
from pandas.errors import UndefinedVariableError

from pandas.core.dtypes.common import is_list_like

import pandas.core.common as com
from pandas.core.computation import (
    expr,
    ops,
    scope as _scope,
)
from pandas.core.computation.common import ensure_decoded
from pandas.core.computation.expr import BaseExprVisitor
from pandas.core.computation.ops import is_term
from pandas.core.construction import extract_array
from pandas.core.indexes.base import Index

from pandas.io.formats.printing import (
    pprint_thing,
    pprint_thing_encoded,
)

if TYPE_CHECKING:
    from pandas._typing import (
        Self,
        npt,
    )


class PyTablesScope(_scope.Scope):
    __slots__ = ("queryables",)

    queryables: Dict[str, Any]

    def __init__(
        self,
        level: int,
        global_dict: Optional[Dict[str, Any]] = None,
        local_dict: Optional[_scope.DeepChainMap[Any, Any]] = None,
        queryables: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(level + 1, global_dict=global_dict, local_dict=local_dict)
        self.queryables = queryables or {}


class Term(ops.Term):
    env: PyTablesScope

    def __new__(
        cls,
        name: Union[str, Any],
        env: PyTablesScope,
        side: Optional[str] = None,
        encoding: Optional[str] = None,
    ) -> Term:
        if isinstance(name, str):
            klass = cls
        else:
            klass = Constant
        return object.__new__(klass)

    def __init__(
        self,
        name: Union[str, Any],
        env: PyTablesScope,
        side: Optional[str] = None,
        encoding: Optional[str] = None,
    ) -> None:
        super().__init__(name, env, side=side, encoding=encoding)

    def _resolve_name(self) -> Any:
        if self.side == "left":
            if isinstance(self.name, str) and self.name not in self.env.queryables:
                raise NameError(f"name {self.name!r} is not defined")
            return self.name

        try:
            return self.env.resolve(self.name, is_local=False)
        except UndefinedVariableError:
            return self.name

    @property
    def value(self) -> Any:
        return self._value


class Constant(Term):
    def __init__(
        self,
        name: Any,
        env: PyTablesScope,
        side: Optional[str] = None,
        encoding: Optional[str] = None,
    ) -> None:
        assert isinstance(env, PyTablesScope), type(env)
        super().__init__(name, env, side=side, encoding=encoding)

    def _resolve_name(self) -> Any:
        return self._name


class BinOp(ops.BinOp):
    _max_selectors: ClassVar[int] = 31

    op: str
    queryables: Dict[str, Any]
    condition: Optional[str]
    encoding: Optional[str]

    def __init__(
        self,
        op: str,
        lhs: Any,
        rhs: Any,
        queryables: Dict[str, Any],
        encoding: Optional[str],
    ) -> None:
        super().__init__(op, lhs, rhs)
        self.queryables = queryables
        self.encoding = encoding
        self.condition = None

    def _disallow_scalar_only_bool_ops(self) -> None:
        pass

    def prune(
        self, klass: Type[Union[ConditionBinOp, FilterBinOp]]
    ) -> Optional[Union[ConditionBinOp, FilterBinOp]]:
        def pr(
            left: Optional[Union[ConditionBinOp, FilterBinOp]],
            right: Optional[Union[ConditionBinOp, FilterBinOp]],
        ) -> Optional[Union[ConditionBinOp, FilterBinOp]]:
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

            return k(
                self.op, left, right, queryables=self.queryables, encoding=self.encoding
            ).evaluate()

        left, right = self.lhs, self.rhs

        if is_term(left) and is_term(right):
            res = pr(left.value, right.value)
        elif not is_term(left) and is_term(right):
            res = pr(left.prune(klass), right.value)
        elif is_term(left) and not is_term(right):
            res = pr(left.value, right.prune(klass))
        elif not (is_term(left) or is_term(right)):
            res = pr(left.prune(klass), right.prune(klass))

        return res

    def conform(self, rhs: Any) -> Any:
        if not is_list_like(rhs):
            rhs = [rhs]
        if isinstance(rhs, np.ndarray):
            rhs = rhs.ravel()
        return rhs

    @property
    def is_valid(self) -> bool:
        return self.lhs in self.queryables

    @property
    def is_in_table(self) -> bool:
        return self.queryables.get(self.lhs) is not None

    @property
    def kind(self) -> Optional[str]:
        return getattr(self.queryables.get(self.lhs), "kind", None)

    @property
    def meta(self) -> Optional[str]:
        return getattr(self.queryables.get(self.lhs), "meta", None)

    @property
    def metadata(self) -> Any:
        return getattr(self.queryables.get(self.lhs), "metadata", None)

    def generate(self, v: TermValue) -> str:
        val = v.tostring(self.encoding)
        return f"({self.lhs} {self.op} {val})"

    def convert_value(self, conv_val: Any) -> TermValue:
        def stringify(value: Any) -> str:
            if self.encoding is not None:
                return pprint_thing_encoded(value, encoding=self.encoding)
            return pprint_thing(value)

        kind = ensure_decoded(self.kind)
        meta = ensure_decoded(self.meta)
        if kind == "datetime" or (kind and kind.startswith("datetime64")):
            if isinstance(conv_val, (int, float)):
                conv_val = stringify(conv_val)
            conv_val = ensure_decoded(conv_val)
            conv_val = Timestamp(conv_val).as_unit("ns")
            if conv_val.tz is not None:
                conv_val = conv_val.tz_convert("UTC")
            return TermValue(conv_val, conv_val._value, kind)
        elif kind in ("timedelta64", "timedelta"):
            if isinstance(conv_val, str):
                conv_val = Timedelta(conv_val)
            else:
                conv_val = Timedelta(conv_val, unit="s")
            conv_val = conv_val.as_unit("ns")._value
            return TermValue(int(conv_val), conv_val, kind)
        elif meta == "category":
            metadata = extract_array(self.metadata, extract_numpy=True)
            result: Union[NDArray[np.intp], np.intp, int]
            if conv_val not in metadata:
                result = -1
            else:
                result = metadata.searchsorted(conv_val, side="left")
            return TermValue(result, result, "integer")
        elif kind == "integer":
            try:
                v_dec = Decimal(conv_val)
            except InvalidOperation:
                float(conv_val)
            else:
                conv_val = int(v_dec.to_integral_exact(rounding="ROUND_HALF_EVEN"))
            return TermValue(conv_val, conv_val, kind)
        elif kind == "float":
            conv_val = float(conv_val)
            return TermValue(conv_val, conv_val, kind)
        elif kind == "bool":
            if isinstance(conv_val, str):
                conv_val = conv_val.strip().lower() not in [
                    "false",
                    "f",
                    "no",
                    "n",
                    "none",
                    "0",
                    "[]",
                    "{}",
                    "",
                ]
            else:
                conv_val = bool(conv_val)
            return TermValue(conv_val, conv_val, kind)
        elif isinstance(conv_val, str):
            return TermValue(conv_val, stringify(conv_val), "string")
        else:
            raise TypeError(
                f"Cannot compare {conv_val} of type {type(conv_val)} to {kind} column"
            )

    def convert_values(self) -> None:
        pass


class FilterBinOp(BinOp):
    filter: Optional[Tuple[Any, Any, Index]] = None

    def __repr__(self) -> str:
        if self.filter is None:
            return "Filter: Not Initialized"
        return pprint_thing(f"[Filter : [{self.filter[0]}] -> [{self.filter[1]}]")

    def invert(self) -> Self:
        if self.filter is not None:
            self.filter = (
                self.filter[0],
                self.generate_filter_op(invert=True),
                self.filter[2],
            )
        return self

    def format(self) -> List[Tuple[Any, Any, Index]]:
        return [self.filter] if self.filter is not None else []

    def evaluate(self) -> Optional[Self]:
        if not self.is_valid:
            raise ValueError(f"query term is not valid [{self}]")

        rhs = self.conform(self.rhs)
        values = list(rhs)

        if self.is_in_table:
            if self.op in ["==", "!="] and len(values) > self._max_selectors:
                filter_op = self.generate_filter_op()
                self.filter = (self.lhs, filter_op, Index(values))
                return self
            return None

        if self.op in ["==", "!="]:
            filter_op = self.generate_filter_op()
            self.filter = (self.lhs, filter_op, Index(values))
        else:
            raise TypeError(
                f"passing a filterable condition to a non-table indexer [{self}]"
            )

        return self

    def generate_filter_op(
        self, invert: bool = False
    ) -> Callable[[Any, Any], Any]:
        if (self.op == "!=" and not invert) or (self.op == "==" and invert):
            return lambda axis, vals: ~axis.isin(vals)
        else:
            return lambda axis, vals: axis.isin(vals)


class JointFilterBinOp(FilterBinOp):
    def format(self) -> List[Any]:
        raise NotImplementedError("unable to collapse Joint Filters")

    def evaluate(self) -> Self:
        return self


class ConditionBinOp(BinOp):
    def __repr__(self) -> str:
        return pprint_thing(f"[Condition : [{self.condition}]]")

    def invert(self) -> Self:
        raise NotImplementedError(
            "cannot use an invert condition when passing to numexpr"
        )

    def format(self) -> str:
        if self.condition is None:
            raise ValueError("Condition is not initialized")
        return self.condition

    def evaluate(self) -> Optional[Self]:
        if not self.is_valid:
            raise ValueError(f"query term is not valid [{self}]")

        if not self.is_in_table:
            return None

        rhs = self.conform(self.rhs)
        values = [self.convert_value(v) for v in rhs]

        if self.op in ["==", "!="]:
            if len(values) <= self._max_selectors:
                vs = [self.generate(v) for v in values]
                self.condition = f"({' | '.join(vs)})"
            else:
                return None
        else:
            self.condition = self.generate(values[0])

        return self


class JointConditionBinOp(ConditionBinOp):
    def evaluate(self) -> Self:
        self.condition = f"({self.lhs.condition} {self.op} {self.rhs.condition})"
        return self


class UnaryOp(ops.UnaryOp):
    def prune(
        self, klass: Type[Union[ConditionBinOp, FilterBinOp]]
    ) -> Optional[Union[ConditionBinOp, FilterBinOp]]:
        if self.op != "~":
            raise NotImplementedError("UnaryOp only support invert type ops")

        operand = self.operand
        operand = operand.prune(klass)

        if operand is not None and (
            (issubclass(klass, ConditionBinOp) and operand.condition is not None)
            or (
                not issubclass(klass, ConditionBinOp)
                and issubclass(klass, FilterBinOp)
                and operand.filter is not None
            )
        ):
            return operand.invert()
        return None


class PyTablesExprVisitor(BaseExprVisitor):
    const_type: ClassVar[Type[ops.Term]] = Constant
    term_type: ClassVar[Type[Term]] = Term

    def __init__(
        self,
        env: PyTablesScope,
        engine: str,
        parser: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(env, engine, parser)
        for bin_op in self.binary_ops:
            bin_node = self.binary_op_nodes_map[bin_op]
            setattr(
                self,
                f"visit_{bin_node}",
                lambda node, bin_op=bin_op: partial(BinOp, bin_op, **kwargs),
            )

    def visit_UnaryOp(
        self, node: ast.UnaryOp, **kwargs: Any
    ) -> Optional[Union[ops.Term, UnaryOp]]:
        if isinstance(node.op, (ast.Not, ast.Invert)):
            return UnaryOp("~", self.visit(node.operand))
        elif isinstance(node.op, ast.USub):
            return self.const_type(-self.visit(node.operand).value, self.env)
        elif isinstance(node.op, ast.UAdd):
            raise NotImplementedError("Unary addition not supported")
        return None

    def visit_Index(self, node: ast.Index, **kwargs: Any) -> Any:
        return self.visit(node.value).value

    def visit_Assign(self, node: ast.Assign, **kwargs: Any) -> Any:
        cmpr = ast.Compare(
            ops=[ast.Eq()], left=node.targets[0], comparators=[node.value]
        )
        return self.visit(cmpr)

    def visit_Subscript(self, node: ast.Subscript, **kwargs: Any) -> ops.Term:
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
            raise ValueError(f"cannot subscript {value!r} with {slobj!r}") from err

    def visit_Attribute(self, node: ast.Attribute, **kwargs: Any) -> Term:
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

        raise ValueError(f"Invalid Attribute context {ctx.__name__}")

    def translate_In(self, op: ast.AST) -> ast.AST:
        return ast.Eq() if isinstance(op, ast.In) else op

    def _rewrite_membership_op(
        self, node: ast.Compare, left: Any, right: Any
    ) -> Tuple[Any, ast.AST, Any, Any]:
        return self.visit(node.op), node.op, left, right


def _validate_where(
    w: Union[str, PyTablesExpr, List[Union[str, PyTablesExpr]]]
) -> Union[str, PyTablesExpr, List[Union[str, PyTablesExpr]]:
    if not (isinstance(w, (PyTablesExpr, str)) or is_list_like(w)):
        raise TypeError(
            "where must be passed as a string, PyTablesExpr, "
            "or list-like of PyTablesExpr"
        )
    return w


class PyTablesExpr(expr.Expr):
    _visitor: Optional[PyTablesExprVisitor]
    env: PyTablesScope
    expr: str
    condition: Optional[str]
    filter: Optional[Any]
    terms: Optional[Union[BinOp, UnaryOp]]
    encoding: Optional[str]

    def __init__(
        self,
        where: Union[str, PyTablesExpr, List[Union[str, PyTablesExpr]]],
        queryables: Optional[Dict[str, Any]] = None,
        encoding: Optional[str] = None,
        scope_level: int = 0,
    ) -> None:
        where = _validate_where(where)
        self.encoding = encoding
        self.condition = None
        self.filter = None
        self.terms = None
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
