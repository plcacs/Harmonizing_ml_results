import abc
from abc import abstractmethod, abstractproperty
import collections
import functools
import re as stdlib_re
import sys
import types
from typing import (
    Any, Dict, List, Set, Tuple, TypeVar, Union, Optional, Callable, Generic,
    Iterable, Iterator, Mapping, MutableMapping, Sequence, MutableSequence,
    AbstractSet, MutableSet, Sized, Hashable, Container, ByteString,
    SupportsAbs, SupportsFloat, SupportsInt, SupportsRound, Reversible,
    SupportsBytes, SupportsComplex, MappingView, KeysView, ItemsView, ValuesView,
    Generator, NamedTuple, IO, TextIO, BinaryIO, Pattern, Match, cast,
    get_type_hints, no_type_check, no_type_check_decorator, overload
)
try:
    import collections.abc as collections_abc
except ImportError:
    import collections as collections_abc

__all__ = [
    'Any', 'Callable', 'Generic', 'Optional', 'TypeVar', 'Union', 'Tuple',
    'AbstractSet', 'ByteString', 'Container', 'Hashable', 'ItemsView',
    'Iterable', 'Iterator', 'KeysView', 'Mapping', 'MappingView',
    'MutableMapping', 'MutableSequence', 'MutableSet', 'Sequence', 'Sized',
    'ValuesView', 'Reversible', 'SupportsAbs', 'SupportsFloat', 'SupportsInt',
    'SupportsRound', 'Dict', 'List', 'Set', 'NamedTuple', 'Generator',
    'AnyStr', 'cast', 'get_type_hints', 'no_type_check',
    'no_type_check_decorator', 'overload', 'io', 're'
]

T = TypeVar('T')
KT = TypeVar('KT')
VT = TypeVar('VT')
T_co = TypeVar('T_co', covariant=True)
V_co = TypeVar('V_co', covariant=True)
VT_co = TypeVar('VT_co', covariant=True)
T_contra = TypeVar('T_contra', contravariant=True)
AnyStr = TypeVar('AnyStr', bytes, str)

def _qualname(x: type) -> str:
    if sys.version_info[:2] >= (3, 3):
        return x.__qualname__
    else:
        return x.__name__

class TypingMeta(type):
    _is_protocol: bool = False

    def __new__(cls, name: str, bases: Tuple[type, ...], namespace: Dict[str, Any], *, _root: bool = False) -> 'TypingMeta':
        if not _root:
            raise TypeError('Cannot subclass %s' % (', '.join(map(_type_repr, bases)) or '()')
        return super().__new__(cls, name, bases, namespace)

    def __init__(self, *args: Any, **kwds: Any) -> None:
        pass

    def _eval_type(self, globalns: Optional[Dict[str, Any]], localns: Optional[Dict[str, Any]]) -> type:
        return self

    def _has_type_var(self) -> bool:
        return False

    def __repr__(self) -> str:
        return '%s.%s' % (self.__module__, _qualname(self))

class Final:
    __slots__: Tuple[str, ...] = ()

    def __new__(self, *args: Any, **kwds: Any) -> None:
        raise TypeError('Cannot instantiate %r' % self.__class__)

class _ForwardRef(TypingMeta):
    def __new__(cls, arg: str) -> '_ForwardRef':
        if not isinstance(arg, str):
            raise TypeError('ForwardRef must be a string -- got %r' % (arg,))
        try:
            code = compile(arg, '<string>', 'eval')
        except SyntaxError:
            raise SyntaxError('ForwardRef must be an expression -- got %r' % (arg,))
        self = super().__new__(cls, arg, (), {}, _root=True)
        self.__forward_arg__ = arg
        self.__forward_code__ = code
        self.__forward_evaluated__ = False
        self.__forward_value__ = None
        typing_globals = globals()
        frame = sys._getframe(1)
        while frame is not None and frame.f_globals is typing_globals:
            frame = frame.f_back
        assert frame is not None
        self.__forward_frame__ = frame
        return self

    def _eval_type(self, globalns: Optional[Dict[str, Any]], localns: Optional[Dict[str, Any]]) -> type:
        if not isinstance(localns, dict):
            raise TypeError('ForwardRef localns must be a dict -- got %r' % (localns,))
        if not isinstance(globalns, dict):
            raise TypeError('ForwardRef globalns must be a dict -- got %r' % (globalns,))
        if not self.__forward_evaluated__:
            if globalns is None and localns is None:
                globalns = localns = {}
            elif globalns is None:
                globalns = localns
            elif localns is None:
                localns = globalns
            self.__forward_value__ = _type_check(eval(self.__forward_code__, globalns, localns), 'Forward references must evaluate to types.')
            self.__forward_evaluated__ = True
        return self.__forward_value__

    def __instancecheck__(self, obj: Any) -> bool:
        raise TypeError('Forward references cannot be used with isinstance().')

    def __subclasscheck__(self, cls: type) -> bool:
        if not self.__forward_evaluated__:
            globalns = self.__forward_frame__.f_globals
            localns = self.__forward_frame__.f_locals
            try:
                self._eval_type(globalns, localns)
            except NameError:
                return False
        return issubclass(cls, self.__forward_value__)

    def __repr__(self) -> str:
        return '_ForwardRef(%r)' % (self.__forward_arg__,)

class _TypeAlias:
    __slots__: Tuple[str, ...] = ('name', 'type_var', 'impl_type', 'type_checker')

    def __new__(cls, *args: Any, **kwds: Any) -> '_TypeAlias':
        if len(args) == 3 and isinstance(args[0], str) and isinstance(args[1], tuple):
            raise TypeError('A type alias cannot be subclassed')
        return object.__new__(cls)

    def __init__(self, name: str, type_var: type, impl_type: type, type_checker: Callable[[Any], Any]) -> None:
        assert isinstance(name, str), repr(name)
        assert isinstance(type_var, type), repr(type_var)
        assert isinstance(impl_type, type), repr(impl_type)
        assert not isinstance(impl_type, TypingMeta), repr(impl_type)
        self.name = name
        self.type_var = type_var
        self.impl_type = impl_type
        self.type_checker = type_checker

    def __repr__(self) -> str:
        return '%s[%s]' % (self.name, _type_repr(self.type_var))

    def __getitem__(self, parameter: type) -> '_TypeAlias':
        assert isinstance(parameter, type), repr(parameter)
        if not isinstance(self.type_var, TypeVar):
            raise TypeError('%s cannot be further parameterized.' % self)
        if self.type_var.__constraints__:
            if not issubclass(parameter, Union[self.type_var.__constraints__]):
                raise TypeError('%s is not a valid substitution for %s.' % (parameter, self.type_var))
        return self.__class__(self.name, parameter, self.impl_type, self.type_checker)

    def __instancecheck__(self, obj: Any) -> bool:
        raise TypeError('Type aliases cannot be used with isinstance().')

    def __subclasscheck__(self, cls: type) -> bool:
        if cls is Any:
            return True
        if isinstance(cls, _TypeAlias):
            return cls.name == self.name and issubclass(cls.type_var, self.type_var)
        else:
            return issubclass(cls, self.impl_type)

def _has_type_var(t: Optional[type]) -> bool:
    return t is not None and isinstance(t, TypingMeta) and t._has_type_var()

def _eval_type(t: type, globalns: Optional[Dict[str, Any]], localns: Optional[Dict[str, Any]]) -> type:
    if isinstance(t, TypingMeta):
        return t._eval_type(globalns, localns)
    else:
        return t

def _type_check(arg: Any, msg: str) -> type:
    if arg is None:
        return type(None)
    if isinstance(arg, str):
        arg = _ForwardRef(arg)
    if not isinstance(arg, (type, _TypeAlias)):
        raise TypeError(msg + ' Got %.100r.' % (arg,))
    return arg

def _type_repr(obj: Any) -> str:
    if isinstance(obj, type) and (not isinstance(obj, TypingMeta)):
        if obj.__module__ == 'builtins':
            return _qualname(obj)
        else:
            return '%s.%s' % (obj.__module__, _qualname(obj))
    else:
        return repr(obj)

class AnyMeta(TypingMeta):
    def __new__(cls, name: str, bases: Tuple[type, ...], namespace: Dict[str, Any], _root: bool = False) -> 'AnyMeta':
        self = super().__new__(cls, name, bases, namespace, _root=_root)
        return self

    def __instancecheck__(self, obj: Any) -> bool:
        raise TypeError('Any cannot be used with isinstance().')

    def __subclasscheck__(self, cls: type) -> bool:
        if not isinstance(cls, type):
            return super().__subclasscheck__(cls)
        return True

class Any(Final, metaclass=AnyMeta, _root=True):
    __slots__: Tuple[str, ...] = ()

class TypeVar(TypingMeta, metaclass=TypingMeta, _root=True):
    def __new__(
        cls,
        name: str,
        *constraints: type,
        bound: Optional[type] = None,
        covariant: bool = False,
        contravariant: bool = False
    ) -> 'TypeVar':
        self = super().__new__(cls, name, (Final,), {}, _root=True)
        if covariant and contravariant:
            raise ValueError('Bivariant type variables are not supported.')
        self.__covariant__ = bool(covariant)
        self.__contravariant__ = bool(contravariant)
        if constraints and bound is not None:
            raise TypeError('Constraints cannot be combined with bound=...')
        if constraints and len(constraints) == 1:
            raise TypeError('A single constraint is not allowed')
        msg = 'TypeVar(name, constraint, ...): constraints must be types.'
        self.__constraints__ = tuple((_type_check(t, msg) for t in constraints))
        if bound:
            self.__bound__ = _type_check(bound, 'Bound must be a type.')
        else:
            self.__bound__ = None
        return self

    def _has_type_var(self) -> bool:
        return True

    def __repr__(self) -> str:
        if self.__covariant__:
            prefix = '+'
        elif self.__contravariant__:
            prefix = '-'
        else:
            prefix = '~'
        return prefix + self.__name__

    def __instancecheck__(self, instance: Any) -> bool:
        raise TypeError('Type variables cannot be used with isinstance().')

    def __subclasscheck__(self, cls: type) -> bool:
        if cls is self:
            return True
        if cls is Any:
            return True
        if self.__bound__ is not None:
            return issubclass(cls, self.__bound__)
        if self.__constraints__:
            return any((issubclass(cls, c) for c in self.__constraints__))
        return True

class UnionMeta(TypingMeta):
    def __new__(
        cls,
        name: str,
        bases: Tuple[type, ...],
        namespace: Dict[str, Any],
        parameters: Optional[Tuple[type, ...]] = None,
        _root: bool = False
    ) -> 'UnionMeta':
        if parameters is None:
            return super().__new__(cls, name, bases, namespace, _root=_root)
        if not isinstance(parameters, tuple):
            raise TypeError('Expected parameters=<tuple>')
        params = []
        msg = 'Union[arg, ...]: each arg must be a type.'
        for p in parameters:
            if isinstance(p, UnionMeta):
                params.extend(p.__union_params__)
            else:
                params.append(_type_check(p, msg))
        all_params = set(params)
        if len(all_params) < len(params):
            new_params = []
            for t in params:
                if t in all_params:
                    new_params.append(t)
                    all_params.remove(t)
            params = new_params
            assert not all_params, all_params
        all_params = set(params)
        for t1 in params:
            if t1 is Any:
                return Any
            if isinstance(t1, TypeVar):
                continue
            if isinstance(t1, _TypeAlias):
                continue
            if any((issubclass(t1, t2) for t2 in all_params - {t1} if not isinstance(t2, TypeVar))):
                all_params.remove(t1)
        if len(all_params) == 1:
            return all_params.pop()
        self = super().__new__(cls, name, bases, {}, _root=True)
        self.__union_params__ = tuple((t for t in params if t in all_params))
        self.__union_set_params__ = frozenset(self.__union_params__)
        return self

    def _eval_type(self, globalns: Optional[Dict[str, Any]], localns: Optional[Dict[str, Any]]) -> type:
        p = tuple((_eval_type(t, globalns, localns) for t in self.__union_params__))
        if p == self.__union_params__:
            return self
        else:
            return self.__class__(self.__name__, self.__bases__, {}, p, _root=True)

    def _has_type_var(self) -> bool:
        if self.__union_params__:
            for t in self.__union_params__:
                if _has_type_var(t):
                    return True
        return False

    def __repr__(self) -> str:
        r = super().__repr__()
        if self.__union_params__:
            r += '[%s]' % ', '.join((_type_repr(t) for t in self.__union_params__))
        return r

    def __getitem__(self, parameters: Union[type, Tuple[type, ...]]) -> 'UnionMeta':
        if self.__union_params__ is not None:
            raise TypeError('Cannot subscript an existing Union. Use Union[u, t] instead.')
        if parameters == ():
            raise TypeError('Cannot take a Union of no types.')
        if not isinstance(parameters, tuple):
            parameters = (parameters,)
        return self.__class__(self.__name__, self.__bases__, dict(self.__dict__), parameters, _root=True)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, UnionMeta):
            return NotImplemented
        return self.__union_set_params__ == other.__union_set_params__

    def __hash__(self) -> int:
        return hash(self.__union_set_params__)

    def __instancecheck__(self, obj: Any) -> bool:
        raise TypeError('Unions cannot be used with isinstance().')

    def __subclasscheck__(self, cls: type) -> bool:
        if cls is Any:
            return True
        if self.__union_params__ is None:
            return isinstance(cls, UnionMeta)
        elif isinstance(cls, UnionMeta):
            if cls.__union_params__ is None:
                return False
            return all((issubclass(c, self) for c in cls.__union_params__))
        elif isinstance(cls, TypeVar):
            if cls in self.__union_params__:
                return True
            if cls.__constraints__:
                return issubclass(Union[cls.__constraints__], self)
            return False
        else:
            return any((issubclass(cls, t) for t in self.__union_params__))

class Union(Final, metaclass=UnionMeta, _root=True):
    __union_params__: Optional[Tuple[type, ...]] = None
    __union_set_params__: Optional[frozenset] = None

class OptionalMeta(TypingMeta):
    def __new__(cls, name: str, bases: Tuple[type, ...], namespace: Dict[str, Any], _root: bool = False) -> 'OptionalMeta':
        return super().__new__(cls, name, bases, namespace, _root=_root)

    def __getitem__(self, arg: type) -> Union[type, Tuple[type, type(None)]]:
        arg = _type_check(arg, 'Optional[t] requires a single type.')
        return Union[arg, type(None)]

class Optional(Final, metaclass=OptionalMeta, _root=True):
    __slots__: Tuple[str, ...] = ()

class TupleMeta(TypingMeta):
    def __new__(
        cls,
        name: str,
        bases: Tuple[type, ...],
        namespace: Dict[str, Any],
        parameters: Optional[Tuple[type, ...]] = None,
        use_ellipsis: bool = False,
        _root: bool = False
    ) -> 'TupleMeta':
        self = super().__new__(cls, name, bases, namespace, _root=_root)
        self.__tuple_params__ = parameters
        self.__tuple_use_ellipsis__ = use_ellipsis
        return self

    def _has_type_var(self) -> bool:
        if self.__tuple_params__:
            for t in self.__tuple_params__:
                if _has_type_var(t):
                    return True
        return False

    def _eval_type(self, globalns: Optional[Dict[str, Any]], localns: Optional[Dict[str, Any]]) -> type:
        tp = self.__tuple_params__
        if tp is None:
            return self
        p = tuple((_eval_type(t, globalns, localns) for t in tp))
        if p == self.__tuple_params__:
            return self
        else:
            return self.__class__(self