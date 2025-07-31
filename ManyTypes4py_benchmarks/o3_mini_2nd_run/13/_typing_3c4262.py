from __future__ import annotations
import abc
from abc import abstractmethod, abstractproperty
import collections
import functools
import re as stdlib_re
import sys
import types
from typing import Any as PyAny, Callable as PyCallable, Dict, List, Optional, Tuple, Type, Union

try:
    import collections.abc as collections_abc
except ImportError:
    import collections as collections_abc

__all__ = ['Any', 'Callable', 'Generic', 'Optional', 'TypeVar', 'Union', 'Tuple', 'AbstractSet', 'ByteString', 'Container', 'Hashable', 'ItemsView', 'Iterable', 'Iterator', 'KeysView', 'Mapping', 'MappingView', 'MutableMapping', 'MutableSequence', 'MutableSet', 'Sequence', 'Sized', 'ValuesView', 'Reversible', 'SupportsAbs', 'SupportsFloat', 'SupportsInt', 'SupportsRound', 'Dict', 'List', 'Set', 'NamedTuple', 'Generator', 'AnyStr', 'cast', 'get_type_hints', 'no_type_check', 'no_type_check_decorator', 'overload', 'io', 're']

def _qualname(x: PyAny) -> str:
    if sys.version_info[:2] >= (3, 3):
        return x.__qualname__  # type: ignore[attr-defined]
    else:
        return x.__name__  # type: ignore[attr-defined]

class TypingMeta(type):
    """Metaclass for every type defined below.

    This overrides __new__() to require an extra keyword parameter
    '_root', which serves as a guard against naive subclassing of the
    typing classes.  Any legitimate class defined using a metaclass
    derived from TypingMeta (including internal subclasses created by
    e.g.  Union[X, Y]) must pass _root=True.

    This also defines a dummy constructor (all the work is done in
    __new__) and a nicer repr().
    """
    _is_protocol: bool = False

    def __new__(cls: Type[TypingMeta], name: str, bases: Tuple[Type, ...],
                namespace: Dict[str, PyAny], *, _root: bool = False) -> PyAny:
        if not _root:
            raise TypeError('Cannot subclass %s' % (', '.join(map(_type_repr, bases)) or '()'))
        return super().__new__(cls, name, bases, namespace)

    def __init__(self, *args: PyAny, **kwds: PyAny) -> None:
        pass

    def _eval_type(self, globalns: Dict[str, PyAny], localns: Dict[str, PyAny]) -> PyAny:
        """Override this in subclasses to interpret forward references."""
        return self

    def _has_type_var(self) -> bool:
        return False

    def __repr__(self) -> str:
        return '%s.%s' % (self.__module__, _qualname(self))

class Final:
    """Mix-in class to prevent instantiation."""
    __slots__ = ()

    def __new__(cls: Type[Final], *args: PyAny, **kwds: PyAny) -> PyAny:
        raise TypeError('Cannot instantiate %r' % self.__class__)

class _ForwardRef(TypingMeta):
    """Wrapper to hold a forward reference."""

    def __new__(cls: Type[_ForwardRef], arg: str) -> _ForwardRef:
        if not isinstance(arg, str):
            raise TypeError('ForwardRef must be a string -- got %r' % (arg,))
        try:
            code = compile(arg, '<string>', 'eval')
        except SyntaxError:
            raise SyntaxError('ForwardRef must be an expression -- got %r' % (arg,))
        self = super().__new__(cls, arg, (), {}, _root=True)  # type: _ForwardRef
        self.__forward_arg__ = arg
        self.__forward_code__ = code
        self.__forward_evaluated__ = False
        self.__forward_value__ = None
        typing_globals: Dict[str, PyAny] = globals()
        frame = sys._getframe(1)
        while frame is not None and frame.f_globals is typing_globals:
            frame = frame.f_back
        assert frame is not None
        self.__forward_frame__ = frame
        return self

    def _eval_type(self, globalns: Dict[str, PyAny], localns: Dict[str, PyAny]) -> PyAny:
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
            self.__forward_value__ = _type_check(eval(self.__forward_code__, globalns, localns),
                                                  'Forward references must evaluate to types.')
            self.__forward_evaluated__ = True
        return self.__forward_value__

    def __instancecheck__(self, obj: PyAny) -> bool:
        raise TypeError('Forward references cannot be used with isinstance().')

    def __subclasscheck__(self, cls: PyAny) -> bool:
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
    """Internal helper class for defining generic variants of concrete types.

    Note that this is not a type; let's call it a pseudo-type.  It can
    be used in instance and subclass checks, e.g. isinstance(m, Match)
    or issubclass(type(m), Match).  However, it cannot be itself the
    target of an issubclass() call; e.g. issubclass(Match, C) (for
    some arbitrary class C) raises TypeError rather than returning
    False.
    """
    __slots__ = ('name', 'type_var', 'impl_type', 'type_checker')

    def __new__(cls, *args: PyAny, **kwds: PyAny) -> PyAny:
        if len(args) == 3 and isinstance(args[0], str) and isinstance(args[1], tuple):
            raise TypeError('A type alias cannot be subclassed')
        return object.__new__(cls)

    def __init__(self, name: str, type_var: type, impl_type: type, type_checker: PyCallable[[PyAny], PyAny]) -> None:
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

    def __getitem__(self, parameter: type) -> _TypeAlias:
        assert isinstance(parameter, type), repr(parameter)
        if not isinstance(self.type_var, TypeVar):
            raise TypeError('%s cannot be further parameterized.' % self)
        if self.type_var.__constraints__:
            if not issubclass(parameter, Union[self.type_var.__constraints__]):  # type: ignore
                raise TypeError('%s is not a valid substitution for %s.' % (parameter, self.type_var))
        return self.__class__(self.name, parameter, self.impl_type, self.type_checker)

    def __instancecheck__(self, obj: PyAny) -> bool:
        raise TypeError('Type aliases cannot be used with isinstance().')

    def __subclasscheck__(self, cls: PyAny) -> bool:
        if cls is Any:
            return True
        if isinstance(cls, _TypeAlias):
            return cls.name == self.name and issubclass(cls.type_var, self.type_var)
        else:
            return issubclass(cls, self.impl_type)

def _has_type_var(t: PyAny) -> bool:
    return t is not None and isinstance(t, TypingMeta) and t._has_type_var()

def _eval_type(t: PyAny, globalns: Dict[str, PyAny], localns: Dict[str, PyAny]) -> PyAny:
    if isinstance(t, TypingMeta):
        return t._eval_type(globalns, localns)
    else:
        return t

def _type_check(arg: PyAny, msg: str) -> Union[type, _TypeAlias]:
    """Check that the argument is a type, and return it."""
    if arg is None:
        return type(None)
    if isinstance(arg, str):
        arg = _ForwardRef(arg)
    if not isinstance(arg, (type, _TypeAlias)):
        raise TypeError(msg + ' Got %.100r.' % (arg,))
    return arg

def _type_repr(obj: PyAny) -> str:
    """Return the repr() of an object, special-casing types."""
    if isinstance(obj, type) and (not isinstance(obj, TypingMeta)):
        if obj.__module__ == 'builtins':
            return _qualname(obj)
        else:
            return '%s.%s' % (obj.__module__, _qualname(obj))
    else:
        return repr(obj)

class AnyMeta(TypingMeta):
    """Metaclass for Any."""

    def __new__(cls: Type[AnyMeta], name: str, bases: Tuple[Type, ...],
                namespace: Dict[str, PyAny], *, _root: bool = False) -> PyAny:
        self = super().__new__(cls, name, bases, namespace, _root=_root)
        return self

    def __instancecheck__(self, obj: PyAny) -> bool:
        raise TypeError('Any cannot be used with isinstance().')

    def __subclasscheck__(self, cls: PyAny) -> bool:
        if not isinstance(cls, type):
            return super().__subclasscheck__(cls)
        return True

class Any(Final, metaclass=AnyMeta, _root=True):
    """Special type indicating an unconstrained type.

    - Any object is an instance of Any.
    - Any class is a subclass of Any.
    - As a special case, Any and object are subclasses of each other.
    """
    __slots__ = ()

class TypeVar(TypingMeta, metaclass=TypingMeta, _root=True):
    """Type variable.

    See module docstring for details.
    """

    def __new__(cls: Type[TypeVar], name: str, *constraints: PyAny,
                bound: Optional[PyAny] = None, covariant: bool = False,
                contravariant: bool = False) -> TypeVar:
        self = super().__new__(cls, name, (Final,), {}, _root=True)  # type: ignore
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
        self.__name__ = name
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

    def __instancecheck__(self, instance: PyAny) -> bool:
        raise TypeError('Type variables cannot be used with isinstance().')

    def __subclasscheck__(self, cls: PyAny) -> bool:
        if cls is self:
            return True
        if cls is Any:
            return True
        if self.__bound__ is not None:
            return issubclass(cls, self.__bound__)
        if self.__constraints__:
            return any((issubclass(cls, c) for c in self.__constraints__))
        return True

T = TypeVar('T')
KT = TypeVar('KT')
VT = TypeVar('VT')
T_co = TypeVar('T_co', covariant=True)
V_co = TypeVar('V_co', covariant=True)
VT_co = TypeVar('VT_co', covariant=True)
T_contra = TypeVar('T_contra', contravariant=True)
AnyStr = TypeVar('AnyStr', bytes, str)

class UnionMeta(TypingMeta):
    """Metaclass for Union."""

    def __new__(cls: Type[UnionMeta], name: str, bases: Tuple[Type, ...],
                namespace: Dict[str, PyAny], parameters: Optional[Tuple[PyAny, ...]] = None,
                _root: bool = False) -> PyAny:
        if parameters is None:
            return super().__new__(cls, name, bases, namespace, _root=_root)
        if not isinstance(parameters, tuple):
            raise TypeError('Expected parameters=<tuple>')
        params: List[PyAny] = []
        msg = 'Union[arg, ...]: each arg must be a type.'
        for p in parameters:
            if isinstance(p, UnionMeta):
                params.extend(p.__union_params__)  # type: ignore
            else:
                params.append(_type_check(p, msg))
        all_params = set(params)
        if len(all_params) < len(params):
            new_params: List[PyAny] = []
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
        self_obj = super().__new__(cls, name, bases, {}, _root=True)
        self_obj.__union_params__ = tuple((t for t in params if t in all_params))
        self_obj.__union_set_params__ = frozenset(self_obj.__union_params__)
        return self_obj

    def _eval_type(self, globalns: Dict[str, PyAny], localns: Dict[str, PyAny]) -> PyAny:
        p = tuple((_eval_type(t, globalns, localns) for t in self.__union_params__))  # type: ignore
        if p == self.__union_params__:
            return self
        else:
            return self.__class__(self.__name__, self.__bases__, {}, p, _root=True)

    def _has_type_var(self) -> bool:
        if self.__union_params__:
            for t in self.__union_params__:  # type: ignore
                if _has_type_var(t):
                    return True
        return False

    def __repr__(self) -> str:
        r = super().__repr__()
        if hasattr(self, '__union_params__') and self.__union_params__:  # type: ignore
            r += '[%s]' % ', '.join((_type_repr(t) for t in self.__union_params__))  # type: ignore
        return r

    def __getitem__(self, parameters: PyAny) -> PyAny:
        if self.__union_params__ is not None:  # type: ignore
            raise TypeError('Cannot subscript an existing Union. Use Union[u, t] instead.')
        if parameters == ():
            raise TypeError('Cannot take a Union of no types.')
        if not isinstance(parameters, tuple):
            parameters = (parameters,)
        return self.__class__(self.__name__, self.__bases__, dict(self.__dict__), parameters, _root=True)

    def __eq__(self, other: PyAny) -> bool:
        if not isinstance(other, UnionMeta):
            return NotImplemented  # type: ignore
        return self.__union_set_params__ == other.__union_set_params__  # type: ignore

    def __hash__(self) -> int:
        return hash(self.__union_set_params__)  # type: ignore

    def __instancecheck__(self, obj: PyAny) -> bool:
        raise TypeError('Unions cannot be used with isinstance().')

    def __subclasscheck__(self, cls: PyAny) -> bool:
        if cls is Any:
            return True
        if self.__union_params__ is None:  # type: ignore
            return isinstance(cls, UnionMeta)
        elif isinstance(cls, UnionMeta):
            if cls.__union_params__ is None:  # type: ignore
                return False
            return all((issubclass(c, self) for c in cls.__union_params__))  # type: ignore
        elif isinstance(cls, TypeVar):
            if cls in self.__union_params__:  # type: ignore
                return True
            if cls.__constraints__:
                return issubclass(Union[cls.__constraints__], self)  # type: ignore
            return False
        else:
            return any((issubclass(cls, t) for t in self.__union_params__))  # type: ignore

class Union(Final, metaclass=UnionMeta, _root=True):
    """Union type; Union[X, Y] means either X or Y.
    
    See module docstring for further details.
    """
    __union_params__ = None
    __union_set_params__ = None

class OptionalMeta(TypingMeta):
    """Metaclass for Optional."""

    def __new__(cls: Type[OptionalMeta], name: str, bases: Tuple[Type, ...],
                namespace: Dict[str, PyAny], _root: bool = False) -> PyAny:
        return super().__new__(cls, name, bases, namespace, _root=_root)

    def __getitem__(self, arg: PyAny) -> PyAny:
        arg = _type_check(arg, 'Optional[t] requires a single type.')
        return Union[arg, type(None)]

class Optional(Final, metaclass=OptionalMeta, _root=True):
    """Optional type.

    Optional[X] is equivalent to Union[X, type(None)].
    """
    __slots__ = ()

class TupleMeta(TypingMeta):
    """Metaclass for Tuple."""

    def __new__(cls: Type[TupleMeta], name: str, bases: Tuple[Type, ...],
                namespace: Dict[str, PyAny], parameters: Optional[Tuple[PyAny, ...]] = None,
                use_ellipsis: bool = False, _root: bool = False) -> PyAny:
        self_obj = super().__new__(cls, name, bases, namespace, _root=_root)
        self_obj.__tuple_params__ = parameters
        self_obj.__tuple_use_ellipsis__ = use_ellipsis
        return self_obj

    def _has_type_var(self) -> bool:
        if self.__tuple_params__:
            for t in self.__tuple_params__:  # type: ignore
                if _has_type_var(t):
                    return True
        return False

    def _eval_type(self, globalns: Dict[str, PyAny], localns: Dict[str, PyAny]) -> PyAny:
        tp = self.__tuple_params__
        if tp is None:
            return self
        p = tuple((_eval_type(t, globalns, localns) for t in tp))
        if p == self.__tuple_params__:
            return self
        else:
            return self.__class__(self.__name__, self.__bases__, {}, p, _root=True)
    
    def __repr__(self) -> str:
        r = super().__repr__()
        if self.__tuple_params__ is not None:
            params = [_type_repr(p) for p in self.__tuple_params__]  # type: ignore
            if self.__tuple_use_ellipsis__:
                params.append('...')
            r += '[%s]' % ', '.join(params)
        return r

    def __getitem__(self, parameters: PyAny) -> PyAny:
        if self.__tuple_params__ is not None:
            raise TypeError('Cannot re-parameterize %r' % (self,))
        if not isinstance(parameters, tuple):
            parameters = (parameters,)
        if len(parameters) == 2 and parameters[1] == Ellipsis:
            parameters = parameters[:1]
            use_ellipsis = True
            msg = 'Tuple[t, ...]: t must be a type.'
        else:
            use_ellipsis = False
            msg = 'Tuple[t0, t1, ...]: each t must be a type.'
        parameters = tuple((_type_check(p, msg) for p in parameters))
        return self.__class__(self.__name__, self.__bases__, dict(self.__dict__), parameters, use_ellipsis=use_ellipsis, _root=True)

    def __eq__(self, other: PyAny) -> bool:
        if not isinstance(other, TupleMeta):
            return NotImplemented  # type: ignore
        return self.__tuple_params__ == other.__tuple_params__

    def __hash__(self) -> int:
        return hash(self.__tuple_params__)

    def __instancecheck__(self, obj: PyAny) -> bool:
        raise TypeError('Tuples cannot be used with isinstance().')

    def __subclasscheck__(self, cls: PyAny) -> bool:
        if cls is Any:
            return True
        if not isinstance(cls, type):
            return super().__subclasscheck__(cls)
        if issubclass(cls, tuple):
            return True
        if not isinstance(cls, TupleMeta):
            return super().__subclasscheck__(cls)
        if self.__tuple_params__ is None:
            return True
        if cls.__tuple_params__ is None:
            return False
        if cls.__tuple_use_ellipsis__ != self.__tuple_use_ellipsis__:
            return False
        return len(self.__tuple_params__) == len(cls.__tuple_params__) and all((issubclass(x, p) for x, p in zip(cls.__tuple_params__, self.__tuple_params__)))  # type: ignore

class Tuple(Final, metaclass=TupleMeta, _root=True):
    """Tuple type; Tuple[X, Y] is the cross-product type of X and Y."""
    __slots__ = ()

class CallableMeta(TypingMeta):
    """Metaclass for Callable."""

    def __new__(cls: Type[CallableMeta], name: str, bases: Tuple[Type, ...],
                namespace: Dict[str, PyAny], _root: bool = False,
                args: Optional[Union[Ellipsis, List[PyAny]]] = None,
                result: Optional[PyAny] = None) -> PyAny:
        if args is None and result is None:
            pass
        else:
            if args is not Ellipsis:
                if not isinstance(args, list):
                    raise TypeError('Callable[args, result]: args must be a list. Got %.100r.' % (args,))
                msg = 'Callable[[arg, ...], result]: each arg must be a type.'
                args = tuple((_type_check(arg, msg) for arg in args))
            msg = 'Callable[args, result]: result must be a type.'
            result = _type_check(result, msg)
        self_obj = super().__new__(cls, name, bases, namespace, _root=_root)
        self_obj.__args__ = args
        self_obj.__result__ = result
        return self_obj

    def _has_type_var(self) -> bool:
        if self.__args__:
            for t in self.__args__:  # type: ignore
                if _has_type_var(t):
                    return True
        return _has_type_var(self.__result__)

    def _eval_type(self, globalns: Dict[str, PyAny], localns: Dict[str, PyAny]) -> PyAny:
        if self.__args__ is None and self.__result__ is None:
            return self
        if self.__args__ is Ellipsis:
            args = self.__args__
        else:
            args = [_eval_type(t, globalns, localns) for t in self.__args__]  # type: ignore
        result = _eval_type(self.__result__, globalns, localns)
        if args == self.__args__ and result == self.__result__:
            return self
        else:
            return self.__class__(self.__name__, self.__bases__, {}, args=args, result=result, _root=True)

    def __repr__(self) -> str:
        r = super().__repr__()
        if self.__args__ is not None or self.__result__ is not None:
            if self.__args__ is Ellipsis:
                args_r = '...'
            else:
                args_r = '[%s]' % ', '.join((_type_repr(t) for t in self.__args__))  # type: ignore
            r += '[%s, %s]' % (args_r, _type_repr(self.__result__))
        return r

    def __getitem__(self, parameters: PyAny) -> PyAny:
        if self.__args__ is not None or self.__result__ is not None:
            raise TypeError('This Callable type is already parameterized.')
        if not isinstance(parameters, tuple) or len(parameters) != 2:
            raise TypeError('Callable must be used as Callable[[arg, ...], result].')
        args, result = parameters
        return self.__class__(self.__name__, self.__bases__, dict(self.__dict__), _root=True, args=args, result=result)

    def __eq__(self, other: PyAny) -> bool:
        if not isinstance(other, CallableMeta):
            return NotImplemented  # type: ignore
        return self.__args__ == other.__args__ and self.__result__ == other.__result__

    def __hash__(self) -> int:
        return hash(self.__args__) ^ hash(self.__result__)

    def __instancecheck__(self, obj: PyAny) -> bool:
        if self.__args__ is None and self.__result__ is None:
            return isinstance(obj, collections_abc.Callable)
        else:
            raise TypeError('Callable[] cannot be used with isinstance().')

    def __subclasscheck__(self, cls: PyAny) -> bool:
        if cls is Any:
            return True
        if not isinstance(cls, CallableMeta):
            return super().__subclasscheck__(cls)
        if self.__args__ is None and self.__result__ is None:
            return True
        return self == cls

class Callable(Final, metaclass=CallableMeta, _root=True):
    """Callable type; Callable[[int], str] is a function of (int) -> str."""
    __slots__ = ()

def _gorg(a: GenericMeta) -> GenericMeta:
    """Return the farthest origin of a generic class."""
    assert isinstance(a, GenericMeta)
    while a.__origin__ is not None:
        a = a.__origin__
    return a

def _geqv(a: GenericMeta, b: GenericMeta) -> bool:
    """Return whether two generic classes are equivalent."""
    assert isinstance(a, GenericMeta) and isinstance(b, GenericMeta)
    return _gorg(a) is _gorg(b)

class GenericMeta(TypingMeta, abc.ABCMeta):
    """Metaclass for generic types."""
    __extra__: Optional[Type] = None

    def __new__(cls: Type[GenericMeta], name: str, bases: Tuple[Type, ...],
                namespace: Dict[str, PyAny], parameters: Optional[Tuple[PyAny, ...]] = None,
                origin: Optional[GenericMeta] = None, extra: Optional[Type] = None) -> PyAny:
        if parameters is None:
            params: Optional[List[PyAny]] = None
            for base in bases:
                if isinstance(base, TypingMeta):
                    if not isinstance(base, GenericMeta):
                        raise TypeError('You cannot inherit from magic class %s' % repr(base))
                    if base.__parameters__ is None:
                        continue
                    for bp in base.__parameters__:
                        if _has_type_var(bp) and (not isinstance(bp, TypeVar)):
                            raise TypeError('Cannot inherit from a generic class parameterized with non-type-variable %s' % bp)
                        if params is None:
                            params = []
                        if bp not in params:
                            params.append(bp)
            if params is not None:
                parameters = tuple(params)
        self_obj = super().__new__(cls, name, bases, namespace, _root=True)
        self_obj.__parameters__ = parameters
        if extra is not None:
            self_obj.__extra__ = extra
        self_obj.__origin__ = origin
        return self_obj

    def _has_type_var(self) -> bool:
        if self.__parameters__:
            for t in self.__parameters__:
                if _has_type_var(t):
                    return True
        return False

    def __repr__(self) -> str:
        r = super().__repr__()
        if self.__parameters__ is not None:
            r += '[%s]' % ', '.join((_type_repr(p) for p in self.__parameters__))
        return r

    def __eq__(self, other: PyAny) -> bool:
        if not isinstance(other, GenericMeta):
            return NotImplemented  # type: ignore
        return _geqv(self, other) and self.__parameters__ == other.__parameters__

    def __hash__(self) -> int:
        return hash((self.__name__, self.__parameters__))

    def __getitem__(self, params: PyAny) -> PyAny:
        if not isinstance(params, tuple):
            params = (params,)
        if not params:
            raise TypeError('Cannot have empty parameter list')
        msg = 'Parameters to generic types must be types.'
        params = tuple((_type_check(p, msg) for p in params))
        if self.__parameters__ is None:
            for p in params:
                if not isinstance(p, TypeVar):
                    raise TypeError('Initial parameters must be type variables; got %s' % p)
            if len(set(params)) != len(params):
                raise TypeError('All type variables in Generic[...] must be distinct.')
        else:
            if len(params) != len(self.__parameters__):
                raise TypeError('Cannot change parameter count from %d to %d' % (len(self.__parameters__), len(params)))
            for new, old in zip(params, self.__parameters__):
                if isinstance(old, TypeVar):
                    if not old.__constraints__:
                        continue
                    if issubclass(new, Union[old.__constraints__]):  # type: ignore
                        continue
                if not issubclass(new, old):
                    raise TypeError('Cannot substitute %s for %s in %s' % (_type_repr(new), _type_repr(old), self))
        return self.__class__(self.__name__, (self,) + self.__bases__, dict(self.__dict__), parameters=params, origin=self, extra=self.__extra__)

    def __instancecheck__(self, instance: PyAny) -> bool:
        return self.__subclasscheck__(instance.__class__)

    def __subclasscheck__(self, cls: PyAny) -> bool:
        if cls is Any:
            return True
        if isinstance(cls, GenericMeta):
            origin = self.__origin__
            if origin is not None and origin is cls.__origin__:
                assert len(self.__parameters__) == len(origin.__parameters__)
                assert len(cls.__parameters__) == len(origin.__parameters__)
                for p_self, p_cls, p_origin in zip(self.__parameters__, cls.__parameters__, origin.__parameters__):
                    if isinstance(p_origin, TypeVar):
                        if p_origin.__covariant__:
                            if not issubclass(p_cls, p_self):
                                break
                        elif p_origin.__contravariant__:
                            if not issubclass(p_self, p_cls):
                                break
                        elif p_self != p_cls:
                            break
                    elif p_self != p_cls:
                        break
                else:
                    return True
        if super().__subclasscheck__(cls):
            return True
        if self.__extra__ is None or isinstance(cls, GenericMeta):
            return False
        return issubclass(cls, self.__extra__)

class Generic(metaclass=GenericMeta):
    """Abstract base class for generic types."""
    __slots__ = ()

    def __new__(cls: Type[Generic], *args: PyAny, **kwds: PyAny) -> PyAny:
        next_in_mro = object
        for i, c in enumerate(cls.__mro__[:-1]):
            if isinstance(c, GenericMeta) and _gorg(c) is Generic:
                next_in_mro = cls.__mro__[i + 1]
        return next_in_mro.__new__(_gorg(cls))

def cast(typ: PyAny, val: PyAny) -> PyAny:
    """Cast a value to a type."""
    return val

def _get_defaults(func: PyCallable[..., PyAny]) -> Dict[str, PyAny]:
    """Internal helper to extract the default arguments, by name."""
    code = func.__code__
    pos_count = code.co_argcount
    kw_count = code.co_kwonlyargcount
    arg_names = code.co_varnames
    kwarg_names = arg_names[pos_count:pos_count + kw_count]
    arg_names = arg_names[:pos_count]
    defaults = func.__defaults__ or ()
    kwdefaults = func.__kwdefaults__
    res: Dict[str, PyAny] = dict(kwdefaults) if kwdefaults else {}
    pos_offset = pos_count - len(defaults)
    for name, value in zip(arg_names[pos_offset:], defaults):
        assert name not in res
        res[name] = value
    return res

def get_type_hints(obj: PyAny, globalns: Optional[Dict[str, PyAny]] = None,
                   localns: Optional[Dict[str, PyAny]] = None) -> Dict[str, PyAny]:
    """Return type hints for a function or method object."""
    if getattr(obj, '__no_type_check__', None):
        return {}
    if globalns is None:
        globalns = getattr(obj, '__globals__', {})
        if localns is None:
            localns = globalns
    elif localns is None:
        localns = globalns
    defaults = _get_defaults(obj)
    hints: Dict[str, PyAny] = dict(obj.__annotations__)
    for name, value in hints.items():
        if isinstance(value, str):
            value = _ForwardRef(value)
        value = _eval_type(value, globalns, localns)
        if name in defaults and defaults[name] is None:
            value = Optional[value]
        hints[name] = value
    return hints

def no_type_check(arg: Union[PyCallable[..., PyAny], type]) -> Union[PyCallable[..., PyAny], type]:
    """Decorator to indicate that annotations are not type hints."""
    if isinstance(arg, type):
        for obj in arg.__dict__.values():
            if isinstance(obj, types.FunctionType):
                obj.__no_type_check__ = True  # type: ignore
    else:
        arg.__no_type_check__ = True  # type: ignore
    return arg

def no_type_check_decorator(decorator: PyCallable[..., PyAny]) -> PyCallable[..., PyAny]:
    """Decorator to give another decorator the @no_type_check effect."""
    @functools.wraps(decorator)
    def wrapped_decorator(*args: PyAny, **kwds: PyAny) -> PyAny:
        func = decorator(*args, **kwds)
        func = no_type_check(func)
        return func
    return wrapped_decorator

def overload(func: PyCallable[..., PyAny]) -> PyCallable[..., PyAny]:
    raise RuntimeError('Overloading is only supported in library stubs')

class _ProtocolMeta(GenericMeta):
    """Internal metaclass for _Protocol."""

    def __instancecheck__(self, obj: PyAny) -> bool:
        raise TypeError('Protocols cannot be used with isinstance().')

    def __subclasscheck__(self, cls: PyAny) -> bool:
        if not self._is_protocol:
            return NotImplemented  # type: ignore
        if self is _Protocol:
            return True
        attrs = self._get_protocol_attrs()
        for attr in attrs:
            if not any((attr in d.__dict__ for d in cls.__mro__)):
                return False
        return True

    def _get_protocol_attrs(self) -> set[str]:
        protocol_bases: List[type] = []
        for c in self.__mro__:
            if getattr(c, '_is_protocol', False) and c.__name__ != '_Protocol':
                protocol_bases.append(c)
        attrs: set[str] = set()
        for base in protocol_bases:
            for attr in base.__dict__.keys():
                for c in self.__mro__:
                    if c is not base and attr in c.__dict__ and (not getattr(c, '_is_protocol', False)):
                        break
                else:
                    if not attr.startswith('_abc_') and attr != '__abstractmethods__' and (attr != '_is_protocol') and (attr != '__dict__') and (attr != '__slots__') and (attr != '_get_protocol_attrs') and (attr != '__parameters__') and (attr != '__origin__') and (attr != '__module__'):
                        attrs.add(attr)
        return attrs

class _Protocol(metaclass=_ProtocolMeta):
    """Internal base class for protocol classes."""
    __slots__ = ()
    _is_protocol: bool = True

Hashable = collections_abc.Hashable

class Iterable(Generic[T_co], extra=collections_abc.Iterable):
    __slots__ = ()

class Iterator(Iterable[T_co], extra=collections_abc.Iterator):
    __slots__ = ()

class SupportsInt(_Protocol):
    __slots__ = ()

    @abstractmethod
    def __int__(self) -> int:
        pass

class SupportsFloat(_Protocol):
    __slots__ = ()

    @abstractmethod
    def __float__(self) -> float:
        pass

class SupportsComplex(_Protocol):
    __slots__ = ()

    @abstractmethod
    def __complex__(self) -> complex:
        pass

class SupportsBytes(_Protocol):
    __slots__ = ()

    @abstractmethod
    def __bytes__(self) -> bytes:
        pass

class SupportsAbs(_Protocol[T_co]):
    __slots__ = ()

    @abstractmethod
    def __abs__(self) -> T_co:
        pass

class SupportsRound(_Protocol[T_co]):
    __slots__ = ()

    @abstractmethod
    def __round__(self, ndigits: int = 0) -> T_co:
        pass

class Reversible(_Protocol[T_co]):
    __slots__ = ()

    @abstractmethod
    def __reversed__(self) -> PyAny:
        pass

Sized = collections_abc.Sized

class Container(Generic[T_co], extra=collections_abc.Container):
    __slots__ = ()

class AbstractSet(Sized, Iterable[T_co], Container[T_co], extra=collections_abc.Set):
    pass

class MutableSet(AbstractSet[T], extra=collections_abc.MutableSet):
    pass

class Mapping(Sized, Iterable[KT], Container[KT], Generic[VT_co], extra=collections_abc.Mapping):
    pass

class MutableMapping(Mapping[KT, VT], extra=collections_abc.MutableMapping):
    pass

class Sequence(Sized, Iterable[T_co], Container[T_co], extra=collections_abc.Sequence):
    pass

class MutableSequence(Sequence[T], extra=collections_abc.MutableSequence):
    pass

class ByteString(Sequence[int], extra=collections_abc.ByteString):
    pass

ByteString.register(type(memoryview(b'')))

class List(list, MutableSequence[T]):

    def __new__(cls: Type[List], *args: PyAny, **kwds: PyAny) -> list:
        if _geqv(cls, List):
            raise TypeError('Type List cannot be instantiated; use list() instead')
        return list.__new__(cls, *args, **kwds)

class Set(set, MutableSet[T]):

    def __new__(cls: Type[Set], *args: PyAny, **kwds: PyAny) -> set:
        if _geqv(cls, Set):
            raise TypeError('Type Set cannot be instantiated; use set() instead')
        return set.__new__(cls, *args, **kwds)

class _FrozenSetMeta(GenericMeta):
    """This metaclass ensures set is not a subclass of FrozenSet."""
    def __subclasscheck__(self, cls: PyAny) -> bool:
        if issubclass(cls, Set):
            return False
        return super().__subclasscheck__(cls)

class FrozenSet(frozenset, AbstractSet[T_co], metaclass=_FrozenSetMeta):
    __slots__ = ()

    def __new__(cls: Type[FrozenSet], *args: PyAny, **kwds: PyAny) -> frozenset:
        if _geqv(cls, FrozenSet):
            raise TypeError('Type FrozenSet cannot be instantiated; use frozenset() instead')
        return frozenset.__new__(cls, *args, **kwds)

class MappingView(Sized, Iterable[T_co], extra=collections_abc.MappingView):
    pass

class KeysView(MappingView[KT], AbstractSet[KT], extra=collections_abc.KeysView):
    pass

class ItemsView(MappingView, Generic[KT, VT_co], extra=collections_abc.ItemsView):
    pass

class ValuesView(MappingView[VT_co], extra=collections_abc.ValuesView):
    pass

class Dict(dict, MutableMapping[KT, VT]):

    def __new__(cls: Type[Dict], *args: PyAny, **kwds: PyAny) -> dict:
        if _geqv(cls, Dict):
            raise TypeError('Type Dict cannot be instantiated; use dict() instead')
        return dict.__new__(cls, *args, **kwds)

if hasattr(collections_abc, 'Generator'):
    _G_base = collections_abc.Generator
else:
    _G_base = types.GeneratorType

class Generator(Iterator[T_co], Generic[T_co, T_contra, V_co], extra=_G_base):
    __slots__ = ()

    def __new__(cls: Type[Generator], *args: PyAny, **kwds: PyAny) -> PyAny:
        if _geqv(cls, Generator):
            raise TypeError('Type Generator cannot be instantiated; create a subclass instead')
        return super().__new__(cls, *args, **kwds)

def NamedTuple(typename: str, fields: List[Union[Tuple[str, PyAny], str]]) -> Type:
    """Typed version of namedtuple."""
    fields_fixed = [(n, t) for n, t in fields if isinstance(n, str)]  # type: ignore
    cls = collections.namedtuple(typename, [n for n, t in fields_fixed])
    cls._field_types = dict(fields_fixed)
    try:
        cls.__module__ = sys._getframe(1).f_globals.get('__name__', '__main__')
    except (AttributeError, ValueError):
        pass
    return cls

class IO(Generic[AnyStr]):
    """Generic base class for TextIO and BinaryIO."""
    __slots__ = ()

    @abstractproperty
    def mode(self) -> str:
        pass

    @abstractproperty
    def name(self) -> str:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def closed(self) -> bool:
        pass

    @abstractmethod
    def fileno(self) -> int:
        pass

    @abstractmethod
    def flush(self) -> None:
        pass

    @abstractmethod
    def isatty(self) -> bool:
        pass

    @abstractmethod
    def read(self, n: int = -1) -> AnyStr:
        pass

    @abstractmethod
    def readable(self) -> bool:
        pass

    @abstractmethod
    def readline(self, limit: int = -1) -> AnyStr:
        pass

    @abstractmethod
    def readlines(self, hint: int = -1) -> List[AnyStr]:
        pass

    @abstractmethod
    def seek(self, offset: int, whence: int = 0) -> int:
        pass

    @abstractmethod
    def seekable(self) -> bool:
        pass

    @abstractmethod
    def tell(self) -> int:
        pass

    @abstractmethod
    def truncate(self, size: Optional[int] = None) -> int:
        pass

    @abstractmethod
    def writable(self) -> bool:
        pass

    @abstractmethod
    def write(self, s: AnyStr) -> int:
        pass

    @abstractmethod
    def writelines(self, lines: List[AnyStr]) -> None:
        pass

    @abstractmethod
    def __enter__(self) -> IO[AnyStr]:
        pass

    @abstractmethod
    def __exit__(self, type: PyAny, value: PyAny, traceback: PyAny) -> Optional[bool]:
        pass

class BinaryIO(IO[bytes]):
    """Typed version of the return of open() in binary mode."""
    __slots__ = ()

    @abstractmethod
    def write(self, s: bytes) -> int:
        pass

    @abstractmethod
    def __enter__(self) -> BinaryIO:
        pass

class TextIO(IO[str]):
    """Typed version of the return of open() in text mode."""
    __slots__ = ()

    @abstractproperty
    def buffer(self) -> BinaryIO:
        pass

    @abstractproperty
    def encoding(self) -> str:
        pass

    @abstractproperty
    def errors(self) -> Optional[str]:
        pass

    @abstractproperty
    def line_buffering(self) -> bool:
        pass

    @abstractproperty
    def newlines(self) -> Optional[Union[str, Tuple[str, ...]]]:
        pass

    @abstractmethod
    def __enter__(self) -> TextIO:
        pass

class io:
    """Wrapper namespace for IO generic classes."""
    __all__ = ['IO', 'TextIO', 'BinaryIO']
    IO = IO
    TextIO = TextIO
    BinaryIO = BinaryIO

io.__name__ = __name__ + '.io'
sys.modules[io.__name__] = io

Pattern = _TypeAlias('Pattern', AnyStr, type(stdlib_re.compile('')), lambda p: p.pattern)
Match = _TypeAlias('Match', AnyStr, type(stdlib_re.match('', '')), lambda m: m.re.pattern)

class re:
    """Wrapper namespace for re type aliases."""
    __all__ = ['Pattern', 'Match']
    Pattern = Pattern
    Match = Match

re.__name__ = __name__ + '.re'
sys.modules[re.__name__] = re