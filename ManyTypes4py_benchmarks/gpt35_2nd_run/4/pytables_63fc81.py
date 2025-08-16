from __future__ import annotations
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from pandas._typing import Self, npt

class PyTablesScope(_scope.Scope):
    __slots__: tuple[str] = ('queryables',)

    def __init__(self, level: int, global_dict: dict = None, local_dict: dict = None, queryables: dict = None) -> None:
        super().__init__(level + 1, global_dict=global_dict, local_dict=local_dict)
        self.queryables = queryables or {}

class Term(ops.Term):
    def __new__(cls, name: str, env: PyTablesScope, side: Any = None, encoding: Any = None) -> Term:
        if isinstance(name, str):
            klass = cls
        else:
            klass = Constant
        return object.__new__(klass)

    def __init__(self, name: str, env: PyTablesScope, side: Any = None, encoding: Any = None) -> None:
        super().__init__(name, env, side=side, encoding=encoding)

    def _resolve_name(self) -> str:
        if self.side == 'left':
            if self.name not in self.env.queryables:
                raise NameError(f'name {self.name!r} is not defined')
            return self.name
        try:
            return self.env.resolve(self.name, is_local=False)
        except UndefinedVariableError:
            return self.name

    @property
    def value(self) -> Any:
        return self._value

class Constant(Term):
    def __init__(self, name: str, env: PyTablesScope, side: Any = None, encoding: Any = None) -> None:
        assert isinstance(env, PyTablesScope), type(env)
        super().__init__(name, env, side=side, encoding=encoding)

    def _resolve_name(self) -> str:
        return self._name

class BinOp(ops.BinOp):
    _max_selectors: ClassVar[int] = 31

    def __init__(self, op: str, lhs: Any, rhs: Any, queryables: dict, encoding: Any) -> None:
        super().__init__(op, lhs, rhs)
        self.queryables = queryables
        self.encoding = encoding
        self.condition = None

    def _disallow_scalar_only_bool_ops(self) -> None:
        pass

    def prune(self, klass: Any) -> Any:
        ...

    def conform(self, rhs: Any) -> Any:
        ...

    @property
    def is_valid(self) -> bool:
        ...

    @property
    def is_in_table(self) -> bool:
        ...

    @property
    def kind(self) -> Any:
        ...

    @property
    def meta(self) -> Any:
        ...

    @property
    def metadata(self) -> Any:
        ...

    def generate(self, v: Any) -> str:
        ...

    def convert_value(self, conv_val: Any) -> TermValue:
        ...

    def convert_values(self) -> None

class FilterBinOp(BinOp):
    filter: Any = None

    def __repr__(self) -> str:
        ...

    def invert(self) -> FilterBinOp:
        ...

    def format(self) -> list:
        ...

    def evaluate(self) -> Any:
        ...

    def generate_filter_op(self, invert: bool = False) -> Any:
        ...

class JointFilterBinOp(FilterBinOp):
    def format(self) -> None:
        ...

    def evaluate(self) -> Any:
        ...

class ConditionBinOp(BinOp):
    def __repr__(self) -> str:
        ...

    def invert(self) -> None:
        ...

    def format(self) -> Any:
        ...

    def evaluate(self) -> Any:
        ...

class JointConditionBinOp(ConditionBinOp):
    def evaluate(self) -> Any:
        ...

class UnaryOp(ops.UnaryOp):
    def prune(self, klass: Any) -> Any:
        ...

class PyTablesExprVisitor(BaseExprVisitor):
    const_type: ClassVar = Constant
    term_type: ClassVar = Term

    def __init__(self, env: PyTablesScope, engine: str, parser: str, **kwargs: Any) -> None:
        ...

    def visit_UnaryOp(self, node: Any, **kwargs: Any) -> Any:
        ...

    def visit_Index(self, node: Any, **kwargs: Any) -> Any:
        ...

    def visit_Assign(self, node: Any, **kwargs: Any) -> Any:
        ...

    def visit_Subscript(self, node: Any, **kwargs: Any) -> Any:
        ...

    def visit_Attribute(self, node: Any, **kwargs: Any) -> Any:
        ...

    def translate_In(self, op: Any) -> Any:
        ...

    def _rewrite_membership_op(self, node: Any, left: Any, right: Any) -> Any:
        ...

def _validate_where(w: Any) -> Any:
    ...

class PyTablesExpr(expr.Expr):
    def __init__(self, where: Any, queryables: dict = None, encoding: Any = None, scope_level: int = 0) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def evaluate(self) -> tuple:
        ...

class TermValue:
    def __init__(self, value: Any, converted: Any, kind: str) -> None:
        ...

    def tostring(self, encoding: Any) -> str:
        ...

def maybe_expression(s: str) -> bool:
    ...
