from __future__ import annotations
from datetime import datetime
from functools import partial
import operator
from typing import Any, Callable, Dict, FrozenSet, Iterable, Iterator, List, Optional, Tuple, Type, Union, Literal
import numpy as np
from pandas._libs.tslibs import Timestamp
from pandas.core.dtypes.common import is_list_like, is_scalar
import pandas.core.common as com
from pandas.core.computation.common import ensure_decoded, result_type_many
from pandas.core.computation.scope import DEFAULT_GLOBALS
from pandas.io.formats.printing import pprint_thing, pprint_thing_encoded
if False:
    from collections.abc import Callable, Iterable, Iterator

REDUCTIONS: Tuple[str, str, str, str] = ('sum', 'prod', 'min', 'max')
_unary_math_ops: Tuple[str, ...] = (
    'sin', 'cos', 'tan', 'exp', 'log', 'expm1', 'log1p',
    'sqrt', 'sinh', 'cosh', 'tanh', 'arcsin', 'arccos', 'arctan', 'arccosh',
    'arcsinh', 'arctanh', 'abs', 'log10', 'floor', 'ceil'
)
_binary_math_ops: Tuple[str, ...] = ('arctan2',)
MATHOPS: Tuple[str, ...] = _unary_math_ops + _binary_math_ops
LOCAL_TAG: str = '__pd_eval_local_'


class Term:
    def __new__(cls: Type[Term], name: Any, env: Any, side: Any = None, encoding: Any = None) -> Term:
        klass = Constant if not isinstance(name, str) else cls
        supr_new = super(Term, klass).__new__
        return supr_new(klass)

    def __init__(self, name: Any, env: Any, side: Any = None, encoding: Any = None) -> None:
        self._name: Any = name
        self.env: Any = env
        self.side: Any = side
        tname: str = str(name)
        self.is_local: bool = tname.startswith(LOCAL_TAG) or tname in DEFAULT_GLOBALS
        self._value: Any = self._resolve_name()
        self.encoding: Any = encoding

    def _resolve_name(self) -> Any:
        # Placeholder for name resolution logic.
        return self._name

    @property
    def name(self) -> Any:
        return self._name

    @property
    def func_wyr61k4h(self) -> str:
        return str(self.name).replace(LOCAL_TAG, '')

    def __repr__(self) -> str:
        return pprint_thing(self.name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.value

    def func_qqezu6cn(self, *args: Any, **kwargs: Any) -> Term:
        return self

    def func_93lfeq2f(self) -> Any:
        local_name: str = str(self.local_name)
        is_local: bool = self.is_local
        if local_name in self.env.scope and isinstance(self.env.scope[local_name], type):
            is_local = False
        res: Any = self.env.resolve(local_name, is_local=is_local)
        self.update(res)
        if hasattr(res, 'ndim') and isinstance(res.ndim, int) and res.ndim > 2:
            raise NotImplementedError(
                'N-dimensional objects, where N > 2, are not supported with eval'
            )
        return res

    def update(self, value: Any) -> None:
        self._value = value

    def func_u52b7k94(self, value: Any) -> None:
        key: Any = self.name
        if isinstance(key, str):
            self.env.swapkey(self.local_name, key, new_value=value)
        self.value = value

    @property
    def func_m5svyf38(self) -> Any:
        return func_m5svyf38(self._value)

    @property
    def type(self) -> Any:
        try:
            return self._value.values.dtype
        except AttributeError:
            try:
                return self._value.dtype
            except AttributeError:
                return type(self._value)
    return_type = type

    @property
    def func_1xdklqp1(self) -> str:
        return f'{type(self).__name__}(name={self.name!r}, type={self.type})'

    @property
    def func_j4x8bma6(self) -> bool:
        try:
            t = self.type.type
        except AttributeError:
            t = self.type
        return issubclass(t, (datetime, np.datetime64))

    @property
    def func_lqiixkem(self) -> Any:
        return self._value

    @func_lqiixkem.setter
    def func_lqiixkem(self, new_value: Any) -> None:
        self._value = new_value

    @property
    def func_e7jy3xl6(self) -> Any:
        return self._name

    @property
    def func_cxipyu8l(self) -> int:
        return self._value.ndim

    @property
    def local_name(self) -> Any:
        # Placeholder for retrieving local name. Using self._name for simplicity.
        return self._name

    @property
    def value(self) -> Any:
        return self._value

    @value.setter
    def value(self, new_value: Any) -> None:
        self._value = new_value


class Constant(Term):
    def func_93lfeq2f(self) -> Any:
        return self._name

    @property
    def func_e7jy3xl6(self) -> Any:
        return self.value

    def __repr__(self) -> str:
        return repr(self.name)


_bool_op_map: Dict[str, str] = {'not': '~', 'and': '&', 'or': '|'}


class Op:
    def __init__(self, op: str, operands: Iterable[Any], encoding: Optional[Any] = None) -> None:
        self.op: str = _bool_op_map.get(op, op)
        self.operands: Tuple[Any, ...] = tuple(operands)
        self.encoding: Optional[Any] = encoding

    def __iter__(self) -> Iterator[Any]:
        return iter(self.operands)

    def __repr__(self) -> str:
        parened = ('({})'.format(pprint_thing(opr)) for opr in self.operands)
        return pprint_thing(' {} '.format(self.op).join(parened))

    @property
    def func_4eij6wyi(self) -> Any:
        if self.op in CMP_OPS_SYMS + BOOL_OPS_SYMS:
            return np.bool_
        return result_type_many(*(term.type for term in com.flatten(self)))

    @property
    def operand_types(self) -> FrozenSet[Any]:
        return frozenset(getattr(term, 'type', type(term)) for term in com.flatten(self))

    @property
    def func_4dfis32w(self) -> bool:
        types = self.operand_types
        obj_dtype_set = frozenset([np.dtype('object')])
        return self.return_type == object and types - obj_dtype_set  # type: ignore

    @property
    def func_j00vo7sl(self) -> FrozenSet[Any]:
        return frozenset(term.type for term in com.flatten(self))

    @property
    def func_m5svyf38(self) -> bool:
        return all(getattr(operand, 'is_scalar', False) for operand in self.operands)

    @property
    def func_j4x8bma6(self) -> bool:
        try:
            t = self.return_type.type
        except AttributeError:
            t = self.return_type
        return issubclass(t, (datetime, np.datetime64))

    @property
    def return_type(self) -> Any:
        # Placeholder for the return type, could be overridden.
        return Any


def func_4rtux7gg(x: Any, y: Any) -> Any:
    try:
        return x.isin(y)
    except AttributeError:
        if is_list_like(x):
            try:
                return y.isin(x)
            except AttributeError:
                pass
        return x in y


def func_hd1fxlvq(x: Any, y: Any) -> Any:
    try:
        return ~x.isin(y)
    except AttributeError:
        if is_list_like(x):
            try:
                return ~y.isin(x)
            except AttributeError:
                pass
        return x not in y


CMP_OPS_SYMS: Tuple[str, ...] = ('>', '<', '>=', '<=', '==', '!=', 'in', 'not in')
_cmp_ops_funcs: Tuple[Callable[[Any, Any], Any], ...] = (
    operator.gt, operator.lt, operator.ge, operator.le,
    operator.eq, operator.ne, func_4rtux7gg, func_hd1fxlvq
)
_cmp_ops_dict: Dict[str, Callable[[Any, Any], Any]] = dict(zip(CMP_OPS_SYMS, _cmp_ops_funcs))
BOOL_OPS_SYMS: Tuple[str, ...] = ('&', '|', 'and', 'or')
_bool_ops_funcs: Tuple[Callable[[Any, Any], Any], ...] = (operator.and_, operator.or_, operator.and_, operator.or_)
_bool_ops_dict: Dict[str, Callable[[Any, Any], Any]] = dict(zip(BOOL_OPS_SYMS, _bool_ops_funcs))
ARITH_OPS_SYMS: Tuple[str, ...] = ('+', '-', '*', '/', '**', '//', '%')
_arith_ops_funcs: Tuple[Callable[[Any, Any], Any], ...] = (
    operator.add, operator.sub, operator.mul, operator.truediv,
    operator.pow, operator.floordiv, operator.mod
)
_arith_ops_dict: Dict[str, Callable[[Any, Any], Any]] = dict(zip(ARITH_OPS_SYMS, _arith_ops_funcs))
_binary_ops_dict: Dict[str, Callable[[Any, Any], Any]] = {}
for d in (_cmp_ops_dict, _bool_ops_dict, _arith_ops_dict):
    _binary_ops_dict.update(d)


def func_rhsz47d3(obj: Any) -> bool:
    return isinstance(obj, Term)


class BinOp(Op):
    def __init__(self, op: str, lhs: Union[Term, Op], rhs: Union[Term, Op]) -> None:
        super().__init__(op, (lhs, rhs))
        self.lhs: Union[Term, Op] = lhs
        self.rhs: Union[Term, Op] = rhs
        self._disallow_scalar_only_bool_ops()
        self.convert_values()
        try:
            self.func: Callable[[Any, Any], Any] = _binary_ops_dict[op]
        except KeyError as err:
            keys = list(_binary_ops_dict.keys())
            raise ValueError(
                f'Invalid binary operator {op!r}, valid operators are {keys}'
            ) from err

    def _disallow_scalar_only_bool_ops(self) -> None:
        # Placeholder for disallow_scalar_only_bool_ops logic.
        pass

    def convert_values(self) -> None:
        # Placeholder for convert_values logic.
        pass

    def __call__(self, env: Any) -> Any:
        left: Any = self.lhs(env)
        right: Any = self.rhs(env)
        return self.func(left, right)

    def func_qqezu6cn(
        self, env: Any, engine: str, parser: str, term_type: Any, eval_in_python: List[Any]
    ) -> Any:
        if engine == 'python':
            res: Any = self(env)
        else:
            left = self.lhs.evaluate(env, engine=engine, parser=parser,
                                       term_type=term_type, eval_in_python=eval_in_python)
            right = self.rhs.evaluate(env, engine=engine, parser=parser,
                                        term_type=term_type, eval_in_python=eval_in_python)
            if self.op in eval_in_python:
                res = self.func(left.value, right.value)
            else:
                from pandas.core.computation.eval import eval  # type: ignore
                res = eval(self, local_dict=env, engine=engine, parser=parser)
        name = env.add_tmp(res)
        return term_type(name, env=env)

    def func_48t5hjlw(self) -> None:

        def func_pys5j30y(value: Any) -> str:
            if self.encoding is not None:
                encoder = partial(pprint_thing_encoded, encoding=self.encoding)
            else:
                encoder = pprint_thing
            return encoder(value)

        lhs = self.lhs
        rhs = self.rhs
        if func_rhsz47d3(lhs) and getattr(lhs, 'is_datetime', False) and func_rhsz47d3(rhs) and getattr(rhs, 'is_scalar', False):
            v: Any = rhs.value
            if isinstance(v, (int, float)):
                v = func_pys5j30y(v)
            v = Timestamp(ensure_decoded(v))
            if v.tz is not None:
                v = v.tz_convert('UTC')
            self.rhs.update(v)
        if func_rhsz47d3(rhs) and getattr(rhs, 'is_datetime', False) and func_rhsz47d3(lhs) and getattr(lhs, 'is_scalar', False):
            v = lhs.value
            if isinstance(v, (int, float)):
                v = func_pys5j30y(v)
            v = Timestamp(ensure_decoded(v))
            if v.tz is not None:
                v = v.tz_convert('UTC')
            self.lhs.update(v)

    def func_3qv91zdn(self) -> None:
        rhs = self.rhs
        lhs = self.lhs
        rhs_rt: Any = getattr(rhs, 'return_type', type(rhs))
        rhs_rt = getattr(rhs_rt, 'type', rhs_rt)
        lhs_rt: Any = getattr(lhs, 'return_type', type(lhs))
        lhs_rt = getattr(lhs_rt, 'type', lhs_rt)
        if (getattr(lhs, 'is_scalar', False) or getattr(rhs, 'is_scalar', False)
            ) and self.op in _bool_ops_dict and not (issubclass(rhs_rt, (bool, np.bool_)) and issubclass(lhs_rt, (bool, np.bool_))):
            raise NotImplementedError('cannot evaluate scalar only bool ops')


UNARY_OPS_SYMS: Tuple[str, ...] = ('+', '-', '~', 'not')
_unary_ops_funcs: Tuple[Callable[[Any], Any], ...] = (operator.pos, operator.neg, operator.invert, operator.invert)
_unary_ops_dict: Dict[str, Callable[[Any], Any]] = dict(zip(UNARY_OPS_SYMS, _unary_ops_funcs))


class UnaryOp(Op):
    def __init__(self, op: str, operand: Union[Term, Op]) -> None:
        super().__init__(op, (operand,))
        self.operand: Union[Term, Op] = operand
        try:
            self.func: Callable[[Any], Any] = _unary_ops_dict[op]
        except KeyError as err:
            raise ValueError(
                f'Invalid unary operator {op!r}, valid operators are {UNARY_OPS_SYMS}'
            ) from err

    def __call__(self, env: Any) -> Any:
        operand: Any = self.operand(env)
        return self.func(operand)

    def __repr__(self) -> str:
        return pprint_thing(f'{self.op}({self.operand})')

    @property
    def func_4eij6wyi(self) -> Any:
        operand = self.operand
        if getattr(operand, 'return_type', None) == np.dtype('bool'):
            return np.dtype('bool')
        if isinstance(operand, Op) and (operand.op in _cmp_ops_dict or operand.op in _bool_ops_dict):
            return np.dtype('bool')
        return np.dtype('int')


class MathCall(Op):
    def __init__(self, func: Any, args: Iterable[Any]) -> None:
        super().__init__(func.name, args)
        self.func: Any = func

    def __call__(self, env: Any) -> Any:
        operands: List[Any] = [op(env) for op in self.operands]
        return self.func.func(*operands)

    def __repr__(self) -> str:
        operands_str = ', '.join(map(str, self.operands))
        return pprint_thing(f'{self.op}({operands_str})')


class FuncNode:
    def __init__(self, name: str) -> None:
        if name not in MATHOPS:
            raise ValueError(f'"{name}" is not a supported function')
        self.name: str = name
        self.func: Any = getattr(np, name)

    def __call__(self, *args: Any) -> MathCall:
        return MathCall(self, args)


def func_m5svyf38(value: Any) -> Any:
    # Placeholder implementation for func_m5svyf38.
    return value
