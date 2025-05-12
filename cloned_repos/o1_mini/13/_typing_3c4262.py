import abc
from abc import abstractmethod, abstractproperty
import collections
import functools
import re as stdlib_re
import sys
import types
from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    TypeVar,
    Union,
    Tuple as TypingTuple,
    AbstractSet,
    ByteString,
    Container,
    Hashable,
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    Mapping,
    MappingView,
    MutableMapping,
    MutableSequence,
    MutableSet,
    Sequence,
    Sized,
    ValuesView,
    Reversible,
    SupportsAbs,
    SupportsFloat,
    SupportsInt,
    SupportsRound,
    Dict,
    List,
    Set,
    NamedTuple as TypingNamedTuple,
    Generator,
    AnyStr,
    cast,
    get_type_hints,
    no_type_check,
    no_type_check_decorator,
    overload,
)
import io as stdlib_io
import re as stdlib_re_alias

try:
    import collections.abc as collections_abc
except ImportError:
    import collections as collections_abc

__all__ = [
    'Any',
    'Callable',
    'Generic',
    'Optional',
    'TypeVar',
    'Union',
    'Tuple',
    'AbstractSet',
    'ByteString',
    'Container',
    'Hashable',
    'ItemsView',
    'Iterable',
    'Iterator',
    'KeysView',
    'Mapping',
    'MappingView',
    'MutableMapping',
    'MutableSequence',
    'MutableSet',
    'Sequence',
    'Sized',
    'ValuesView',
    'Reversible',
    'SupportsAbs',
    'SupportsFloat',
    'SupportsInt',
    'SupportsRound',
    'Dict',
    'List',
    'Set',
    'NamedTuple',
    'Generator',
    'AnyStr',
    'cast',
    'get_type_hints',
    'no_type_check',
    'no_type_check_decorator',
    'overload',
    'io',
    're',
]

def _qualname(x: type) -> str:
    if sys.version_info[:2] >= (3, 3):
        return x.__qualname__
    else:
        return x.__name__

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

    def __new__(
        cls,
        name: str,
        bases: tuple,
        namespace: dict,
        *,
        _root: bool = False
    ) -> 'TypingMeta':
        if not _root:
            raise TypeError('Cannot subclass %s' % (', '.join(map(_type_repr, bases)) or '()'))
        return super().__new__(cls, name, bases, namespace)

    def __init__(self, *args: Any, **kwds: Any) -> None:
        pass

    def _eval_type(self, globalns: dict, localns: dict) -> Any:
        """Override this in subclasses to interpret forward references.

        For example, Union['C'] is internally stored as
        Union[_ForwardRef('C')], which should evaluate to _Union[C],
        where C is an object found in globalns or localns (searching
        localns first, of course).
        """
        return self

    def _has_type_var(self) -> bool:
        return False

    def __repr__(self) -> str:
        return '%s.%s' % (self.__module__, _qualname(self))

class Final:
    """Mix-in class to prevent instantiation."""
    __slots__ = ()

    def __new__(cls, *args: Any, **kwds: Any) -> None:
        raise TypeError('Cannot instantiate %r' % cls.__class__)

class _ForwardRef(TypingMeta):
    """Wrapper to hold a forward reference."""

    __forward_arg__: str
    __forward_code__: Any
    __forward_evaluated__: bool
    __forward_value__: Any
    __forward_frame__: Any

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

    def _eval_type(self, globalns: dict, localns: dict) -> Any:
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
    """Internal helper class for defining generic variants of concrete types.

    Note that this is not a type; let's call it a pseudo-type.  It can
    be used in instance and subclass checks, e.g. isinstance(m, Match)
    or issubclass(type(m), Match).  However, it cannot be itself the
    target of an issubclass() call; e.g. issubclass(Match, C) (for
    some arbitrary class C) raises TypeError rather than returning
    False.
    """
    __slots__ = ('name', 'type_var', 'impl_type', 'type_checker')

    def __new__(cls, *args: Any, **kwds: Any) -> '_TypeAlias':
        """Constructor.

        This only exists to give a better error message in case
        someone tries to subclass a type alias (not a good idea).
        """
        if len(args) == 3 and isinstance(args[0], str) and isinstance(args[1], tuple):
            raise TypeError('A type alias cannot be subclassed')
        return object.__new__(cls)

    def __init__(self, name: str, type_var: type, impl_type: type, type_checker: Callable[[Any], Any]) -> None:
        """Initializer.

        Args:
            name: The name, e.g. 'Pattern'.
            type_var: The type parameter, e.g. AnyStr, or the
                specific type, e.g. str.
            impl_type: The implementation type.
            type_checker: Function that takes an impl_type instance.
                and returns a value that should be a type_var instance.
        """
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

def _has_type_var(t: Any) -> bool:
    return t is not None and isinstance(t, TypingMeta) and t._has_type_var()

def _eval_type(t: Any, globalns: dict, localns: dict) -> Any:
    if isinstance(t, TypingMeta):
        return t._eval_type(globalns, localns)
    else:
        return t

def _type_check(arg: Any, msg: str) -> Any:
    """Check that the argument is a type, and return it.

    As a special case, accept None and return type(None) instead.
    Also, _TypeAlias instances (e.g. Match, Pattern) are acceptable.

    The msg argument is a human-readable error message, e.g.

        "Union[arg, ...]: arg should be a type."

    We append the repr() of the actual value (truncated to 100 chars).
    """
    if arg is None:
        return type(None)
    if isinstance(arg, str):
        arg = _ForwardRef(arg)
    if not isinstance(arg, (type, _TypeAlias)):
        raise TypeError(msg + ' Got %.100r.' % (arg,))
    return arg

def _type_repr(obj: Any) -> str:
    """Return the repr() of an object, special-casing types.

    If obj is a type, we return a shorter version than the default
    type.__repr__, based on the module and qualified name, which is
    typically enough to uniquely identify a type.  For everything
    else, we fall back on repr(obj).
    """
    if isinstance(obj, type) and (not isinstance(obj, TypingMeta)):
        if obj.__module__ == 'builtins':
            return _qualname(obj)
        else:
            return '%s.%s' % (obj.__module__, _qualname(obj))
    else:
        return repr(obj)

class AnyMeta(TypingMeta):
    """Metaclass for Any."""

    def __new__(
        cls,
        name: str,
        bases: tuple,
        namespace: dict,
        _root: bool = False
    ) -> 'AnyMeta':
        self = super().__new__(cls, name, bases, namespace, _root=_root)
        return self

    def __instancecheck__(self, obj: Any) -> bool:
        raise TypeError('Any cannot be used with isinstance().')

    def __subclasscheck__(self, cls: type) -> bool:
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

    Usage::

      T = TypeVar('T')  # Can be anything
      A = TypeVar('A', str, bytes)  # Must be str or bytes

    Type variables exist primarily for the benefit of static type
    checkers.  They serve as the parameters for generic types as well
    as for generic function definitions.  See class Generic for more
    information on generic types.  Generic functions work as follows:

      def repeat(x: T, n: int) -> Sequence[T]:
          '''Return a list containing n references to x.'''
          return [x]*n

      def longest(x: A, y: A) -> A:
          '''Return the longest of two strings.'''
          return x if len(x) >= len(y) else y

    The latter example's signature is essentially the overloading
    of (str, str) -> str and (bytes, bytes) -> bytes.  Also note
    that if the arguments are instances of some subclass of str,
    the return type is still plain str.

    At runtime, isinstance(x, T) will raise TypeError.  However,
    issubclass(C, T) is true for any class C, and issubclass(str, A)
    and issubclass(bytes, A) are true, and issubclass(int, A) is
    false.

    Type variables may be marked covariant or contravariant by passing
    covariant=True or contravariant=True.  See PEP 484 for more
    details.  By default type variables are invariant.

    Type variables can be introspected. e.g.::

      T.__name__ == 'T'
      T.__constraints__ == ()
      T.__covariant__ == False
      T.__contravariant__ = False
      A.__constraints__ == (str, bytes)
    """

    __covariant__: bool
    __contravariant__: bool
    __constraints__: tuple
    __bound__: Optional[type]

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

    __union_params__: Optional[tuple]
    __union_set_params__: Optional[frozenset]

    def __new__(
        cls,
        name: str,
        bases: tuple,
        namespace: dict,
        parameters: Optional[tuple] = None,
        _root: bool = False
    ) -> 'UnionMeta':
        if parameters is None:
            return super().__new__(cls, name, bases, namespace, _root=_root)
        if not isinstance(parameters, tuple):
            raise TypeError('Expected parameters=<tuple>')
        params: list = []
        msg = 'Union[arg, ...]: each arg must be a type.'
        for p in parameters:
            if isinstance(p, UnionMeta):
                params.extend(p.__union_params__)
            else:
                params.append(_type_check(p, msg))
        all_params = set(params)
        if len(all_params) < len(params):
            new_params: list = []
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

    def _eval_type(self, globalns: dict, localns: dict) -> Any:
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

    def __getitem__(self, parameters: Any) -> 'UnionMeta':
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
    """Union type; Union[X, Y] means either X or Y.

    To define a union, use e.g. Union[int, str].  Details:

    - The arguments must be types and there must be at least one.

    - None as an argument is a special case and is replaced by
      type(None).

    - Unions of unions are flattened, e.g.::

        Union[Union[int, str], float] == Union[int, str, float]

    - Unions of a single argument vanish, e.g.::

        Union[int] == int  # The constructor actually returns int

    - Redundant arguments are skipped, e.g.::

        Union[int, str, int] == Union[int, str]

    - When comparing unions, the argument order is ignored, e.g.::

        Union[int, str] == Union[str, int]

    - When two arguments have a subclass relationship, the least
      derived argument is kept, e.g.::

        class Employee: pass
        class Manager(Employee): pass
        Union[int, Employee, Manager] == Union[int, Employee]
        Union[Manager, int, Employee] == Union[int, Employee]
        Union[Employee, Manager] == Employee

    - Corollary: if Any is present it is the sole survivor, e.g.::

        Union[int, Any] == Any

    - Similar for object::

        Union[int, object] == object

    - To cut a tie: Union[object, Any] == Union[Any, object] == Any.

    - You cannot subclass or instantiate a union.

    - You cannot write Union[X][Y] (what would it mean?).

    - You can use Optional[X] as a shorthand for Union[X, None].
    """
    __union_params__: Optional[tuple] = None
    __union_set_params__: Optional[frozenset] = None

class OptionalMeta(TypingMeta):
    """Metaclass for Optional."""

    def __new__(
        cls,
        name: str,
        bases: tuple,
        namespace: dict,
        _root: bool = False
    ) -> 'OptionalMeta':
        return super().__new__(cls, name, bases, namespace, _root=_root)

    def __getitem__(self, arg: type) -> UnionMeta:
        arg = _type_check(arg, 'Optional[t] requires a single type.')
        return Union[arg, type(None)]

class Optional(Final, metaclass=OptionalMeta, _root=True):
    """Optional type.

    Optional[X] is equivalent to Union[X, type(None)].
    """
    __slots__ = ()

class TupleMeta(TypingMeta):
    """Metaclass for Tuple."""

    __tuple_params__: Optional[tuple]
    __tuple_use_ellipsis__: bool

    def __new__(
        cls,
        name: str,
        bases: tuple,
        namespace: dict,
        parameters: Optional[tuple] = None,
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

    def _eval_type(self, globalns: dict, localns: dict) -> Any:
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
            params = [_type_repr(p) for p in self.__tuple_params__]
            if self.__tuple_use_ellipsis__:
                params.append('...')
            r += '[%s]' % ', '.join(params)
        return r

    def __getitem__(self, parameters: Any) -> 'TupleMeta':
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

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TupleMeta):
            return NotImplemented
        return self.__tuple_params__ == other.__tuple_params__

    def __hash__(self) -> int:
        return hash(self.__tuple_params__)

    def __instancecheck__(self, obj: Any) -> bool:
        raise TypeError('Tuples cannot be used with isinstance().')

    def __subclasscheck__(self, cls: type) -> bool:
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
        return len(self.__tuple_params__) == len(cls.__tuple_params__) and all(
            (issubclass(x, p) for x, p in zip(cls.__tuple_params__, self.__tuple_params__))
        )

class Tuple(Final, metaclass=TupleMeta, _root=True):
    """Tuple type; Tuple[X, Y] is the cross-product type of X and Y.

    Example: Tuple[T1, T2] is a tuple of two elements corresponding
    to type variables T1 and T2.  Tuple[int, float, str] is a tuple
    of an int, a float and a string.

    To specify a variable-length tuple of homogeneous type, use Sequence[T].
    """
    __slots__ = ()

class CallableMeta(TypingMeta):
    """Metaclass for Callable."""

    __args__: Optional[Union[list, Ellipsis]]
    __result__: Optional[type]

    def __new__(
        cls,
        name: str,
        bases: tuple,
        namespace: dict,
        _root: bool = False,
        args: Optional[list] = None,
        result: Optional[type] = None
    ) -> 'CallableMeta':
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
        self = super().__new__(cls, name, bases, namespace, _root=_root)
        self.__args__ = args
        self.__result__ = result
        return self

    def _has_type_var(self) -> bool:
        if self.__args__:
            for t in self.__args__:
                if _has_type_var(t):
                    return True
        return _has_type_var(self.__result__)

    def _eval_type(self, globalns: dict, localns: dict) -> Any:
        if self.__args__ is None and self.__result__ is None:
            return self
        if self.__args__ is Ellipsis:
            args = self.__args__
        else:
            args = [_eval_type(t, globalns, localns) for t in self.__args__]
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
                args_r = '[%s]' % ', '.join((_type_repr(t) for t in self.__args__))
            r += '[%s, %s]' % (args_r, _type_repr(self.__result__))
        return r

    def __getitem__(self, parameters: Any) -> 'CallableMeta':
        if self.__args__ is not None or self.__result__ is not None:
            raise TypeError('This Callable type is already parameterized.')
        if not isinstance(parameters, tuple) or len(parameters) != 2:
            raise TypeError('Callable must be used as Callable[[arg, ...], result].')
        args, result = parameters
        if isinstance(args, list):
            args = tuple(args)
        return self.__class__(self.__name__, self.__bases__, dict(self.__dict__), _root=True, args=args, result=result)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CallableMeta):
            return NotImplemented
        return self.__args__ == other.__args__ and self.__result__ == other.__result__

    def __hash__(self) -> int:
        return hash(self.__args__) ^ hash(self.__result__)

    def __instancecheck__(self, obj: Any) -> bool:
        if self.__args__ is None and self.__result__ is None:
            return isinstance(obj, collections_abc.Callable)
        else:
            raise TypeError('Callable[] cannot be used with isinstance().')

    def __subclasscheck__(self, cls: type) -> bool:
        if cls is Any:
            return True
        if not isinstance(cls, CallableMeta):
            return super().__subclasscheck__(cls)
        if self.__args__ is None and self.__result__ is None:
            return True
        return self == cls

class Callable(Final, metaclass=CallableMeta, _root=True):
    """Callable type; Callable[[int], str] is a function of (int) -> str.

    The subscription syntax must always be used with exactly two
    values: the argument list and the return type.  The argument list
    must be a list of types; the return type must be a single type.

    There is no syntax to indicate optional or keyword arguments,
    such function types are rarely used as callback types.
    """
    __slots__ = ()

def _gorg(a: 'GenericMeta') -> 'GenericMeta':
    """Return the farthest origin of a generic class."""
    assert isinstance(a, GenericMeta)
    while a.__origin__ is not None:
        a = a.__origin__
    return a

def _geqv(a: 'GenericMeta', b: 'GenericMeta') -> bool:
    """Return whether two generic classes are equivalent.

    The intention is to consider generic class X and any of its
    parameterized forms (X[T], X[int], etc.)  as equivalent.

    However, X is not equivalent to a subclass of X.

    The relation is reflexive, symmetric and transitive.
    """
    assert isinstance(a, GenericMeta) and isinstance(b, GenericMeta)
    return _gorg(a) is _gorg(b)

class GenericMeta(TypingMeta, abc.ABCMeta):
    """Metaclass for generic types."""
    __extra__: Optional[type] = None

    def __new__(
        cls,
        name: str,
        bases: tuple,
        namespace: dict,
        parameters: Optional[tuple] = None,
        origin: Optional[type] = None,
        extra: Optional[type] = None
    ) -> 'GenericMeta':
        if parameters is None:
            params: Optional[list] = None
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
        self = super().__new__(cls, name, bases, namespace, _root=True)
        self.__parameters__ = parameters
        if extra is not None:
            self.__extra__ = extra
        self.__origin__ = origin
        return self

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

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, GenericMeta):
            return NotImplemented
        return _geqv(self, other) and self.__parameters__ == other.__parameters__

    def __hash__(self) -> int:
        return hash((self.__name__, self.__parameters__))

    def __getitem__(self, params: Any) -> 'GenericMeta':
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
                    if issubclass(new, Union[old.__constraints__]):
                        continue
                if not issubclass(new, old):
                    raise TypeError('Cannot substitute %s for %s in %s' % (_type_repr(new), _type_repr(old), self))
        return self.__class__(self.__name__, (self,) + self.__bases__, dict(self.__dict__), parameters=params, origin=self, extra=self.__extra__)

    def __instancecheck__(self, instance: Any) -> bool:
        return self.__subclasscheck__(instance.__class__)

    def __subclasscheck__(self, cls: type) -> bool:
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
    """Abstract base class for generic types.

    A generic type is typically declared by inheriting from an
    instantiation of this class with one or more type variables.
    For example, a generic mapping type might be defined as::

      class Mapping(Generic[KT, VT]):
          def __getitem__(self, key: KT) -> VT:
              ...
          # Etc.

    This class can then be used as follows::

      def lookup_name(mapping: Mapping, key: KT, default: VT) -> VT:
          try:
              return mapping[key]
          except KeyError:
              return default

    For clarity the type variables may be redefined, e.g.::

      X = TypeVar('X')
      Y = TypeVar('Y')
      def lookup_name(mapping: Mapping[X, Y], key: X, default: Y) -> Y:
          # Same body as above.
    """
    __slots__ = ()

    def __new__(cls, *args: Any, **kwds: Any) -> Any:
        next_in_mro = object
        for i, c in enumerate(cls.__mro__[:-1]):
            if isinstance(c, GenericMeta) and _gorg(c) is Generic:
                next_in_mro = cls.__mro__[i + 1]
        return next_in_mro.__new__(_gorg(cls))

def cast(typ: Any, val: Any) -> Any:
    """Cast a value to a type.

    This returns the value unchanged.  To the type checker this
    signals that the return value has the designated type, but at
    runtime we intentionally don't check anything (we want this
    to be as fast as possible).
    """
    return val

def _get_defaults(func: types.FunctionType) -> dict:
    """Internal helper to extract the default arguments, by name."""
    code = func.__code__
    pos_count = code.co_argcount
    kw_count = code.co_kwonlyargcount
    arg_names = code.co_varnames
    kwarg_names = arg_names[pos_count:pos_count + kw_count]
    arg_names = arg_names[:pos_count]
    defaults = func.__defaults__ or ()
    kwdefaults = func.__kwdefaults__
    res: dict = dict(kwdefaults) if kwdefaults else {}
    pos_offset = pos_count - len(defaults)
    for name, value in zip(arg_names[pos_offset:], defaults):
        assert name not in res
        res[name] = value
    return res

def get_type_hints(obj: Any, globalns: Optional[dict] = None, localns: Optional[dict] = None) -> dict:
    """Return type hints for a function or method object.

    This is often the same as obj.__annotations__, but it handles
    forward references encoded as string literals, and if necessary
    adds Optional[t] if a default value equal to None is set.

    BEWARE -- the behavior of globalns and localns is counterintuitive
    (unless you are familiar with how eval() and exec() work).  The
    search order is locals first, then globals.

    - If no dict arguments are passed, an attempt is made to use the
      globals from obj, and these are also used as the locals.  If the
      object does not appear to have globals, an exception is raised.

    - If one dict argument is passed, it is used for both globals and
      locals.

    - If two dict arguments are passed, they specify globals and
      locals, respectively.
    """
    if getattr(obj, '__no_type_check__', None):
        return {}
    if globalns is None:
        globalns = getattr(obj, '__globals__', {})
        if localns is None:
            localns = globalns
    elif localns is None:
        localns = globalns
    defaults = _get_defaults(obj)
    hints: dict = dict(obj.__annotations__)
    for name, value in hints.items():
        if isinstance(value, str):
            value = _ForwardRef(value)
        value = _eval_type(value, globalns, localns)
        if name in defaults and defaults[name] is None:
            value = Optional[value]
        hints[name] = value
    return hints

def no_type_check(arg: Any) -> Any:
    """Decorator to indicate that annotations are not type hints.

    The argument must be a class or function; if it is a class, it
    applies recursively to all methods defined in that class (but not
    to methods defined in its superclasses or subclasses).

    This mutates the function(s) in place.
    """
    if isinstance(arg, type):
        for obj in arg.__dict__.values():
            if isinstance(obj, types.FunctionType):
                obj.__no_type_check__ = True
    else:
        arg.__no_type_check__ = True
    return arg

def no_type_check_decorator(decorator: Callable[..., Callable[..., Any]]) -> Callable[..., Callable[..., Any]]:
    """Decorator to give another decorator the @no_type_check effect.

    This wraps the decorator with something that wraps the decorated
    function in @no_type_check.
    """

    @functools.wraps(decorator)
    def wrapped_decorator(*args: Any, **kwds: Any) -> Callable[..., Any]:
            func = decorator(*args, **kwds)
            func = no_type_check(func)
            return func
    return wrapped_decorator

def overload(func: Any) -> Any:
    raise RuntimeError('Overloading is only supported in library stubs')

class _ProtocolMeta(GenericMeta):
    """Internal metaclass for _Protocol.

    This exists so _Protocol classes can be generic without deriving
    from Generic.
    """

    def __instancecheck__(self, obj: Any) -> bool:
        raise TypeError('Protocols cannot be used with isinstance().')

    def __subclasscheck__(self, cls: type) -> bool:
        if not self._is_protocol:
            return NotImplemented
        if self is _Protocol:
            return True
        attrs = self._get_protocol_attrs()
        for attr in attrs:
            if not any((attr in d.__dict__ for d in cls.__mro__)):
                return False
        return True

    def _get_protocol_attrs(self) -> set:
        protocol_bases: list = []
        for c in self.__mro__:
            if getattr(c, '_is_protocol', False) and c.__name__ != '_Protocol':
                protocol_bases.append(c)
        attrs: set = set()
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
    """Internal base class for protocol classes.

    This implements a simple-minded structural isinstance check
    (similar but more general than the one-offs in collections.abc
    such as Hashable).
    """
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
    def __reversed__(self) -> Iterator[T_co]:
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
    __slots__ = ()

class MutableSequence(Sequence[T], extra=collections_abc.MutableSequence):
    __slots__ = ()

class ByteString(Sequence[int], extra=collections_abc.ByteString):
    __slots__ = ()
ByteString.register(type(memoryview(b'')))

class List(list, MutableSequence[T]):
    __slots__ = ()

    def __new__(cls, *args: Any, **kwds: Any) -> list:
        if _geqv(cls, List):
            raise TypeError('Type List cannot be instantiated; use list() instead')
        return list.__new__(cls, *args, **kwds)

class Set(set, MutableSet[T]):
    __slots__ = ()

    def __new__(cls, *args: Any, **kwds: Any) -> set:
        if _geqv(cls, Set):
            raise TypeError('Type Set cannot be instantiated; use set() instead')
        return set.__new__(cls, *args, **kwds)

class _FrozenSetMeta(GenericMeta):
    """This metaclass ensures set is not a subclass of FrozenSet.

    Without this metaclass, set would be considered a subclass of
    FrozenSet, because FrozenSet.__extra__ is collections.abc.Set, and
    set is a subclass of that.
    """

    def __subclasscheck__(self, cls: type) -> bool:
        if issubclass(cls, Set):
            return False
        return super().__subclasscheck__(cls)

class FrozenSet(frozenset, AbstractSet[T_co], metaclass=_FrozenSetMeta):
    __slots__ = ()

    def __new__(cls, *args: Any, **kwds: Any) -> frozenset:
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
    __slots__ = ()

    def __new__(cls, *args: Any, **kwds: Any) -> dict:
        if _geqv(cls, Dict):
            raise TypeError('Type Dict cannot be instantiated; use dict() instead')
        return dict.__new__(cls, *args, **kwds)

if hasattr(collections_abc, 'Generator'):
    _G_base = collections_abc.Generator
else:
    _G_base = types.GeneratorType

class Generator(Iterator[T_co], Generic[T_co, T_contra, V_co], extra=_G_base):
    __slots__ = ()

    def __new__(cls, *args: Any, **kwds: Any) -> Iterator[T_co]:
        if _geqv(cls, Generator):
            raise TypeError('Type Generator cannot be instantiated; create a subclass instead')
        return super().__new__(cls, *args, **kwds)

def NamedTuple(typename: str, fields: list) -> type:
    """Typed version of namedtuple.

    Usage::

        Employee = typing.NamedTuple('Employee', [('name', str), 'id', int)])

    This is equivalent to::

        Employee = collections.namedtuple('Employee', ['name', 'id'])

    The resulting class has one extra attribute: _field_types,
    giving a dict mapping field names to types.  (The field names
    are in the _fields attribute, which is part of the namedtuple
    API.)
    """
    fields = [(n, t) for n, t in fields]
    cls = collections.namedtuple(typename, [n for n, t in fields])
    cls._field_types = dict(fields)
    try:
        cls.__module__ = sys._getframe(1).f_globals.get('__name__', '__main__')
    except (AttributeError, ValueError):
        pass
    return cls

class IO(Generic[AnyStr]):
    """Generic base class for TextIO and BinaryIO.

    This is an abstract, generic version of the return of open().

    NOTE: This does not distinguish between the different possible
    classes (text vs. binary, read vs. write vs. read/write,
    append-only, unbuffered).  The TextIO and BinaryIO subclasses
    below capture the distinctions between text vs. binary, which is
    pervasive in the interface; however we currently do not offer a
    way to track the other distinctions in the type system.
    """
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
    def readlines(self, hint: int = -1) -> list:
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
    def writelines(self, lines: Iterable[AnyStr]) -> None:
        pass

    @abstractmethod
    def __enter__(self) -> 'IO[AnyStr]':
        pass

    @abstractmethod
    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        pass

class BinaryIO(IO[bytes]):
    """Typed version of the return of open() in binary mode."""
    __slots__ = ()

    @abstractmethod
    def write(self, s: bytes) -> int:
        pass

    @abstractmethod
    def __enter__(self) -> 'BinaryIO':
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
    def newlines(self) -> Optional[Union[str, tuple, None]]:
        pass

    @abstractmethod
    def __enter__(self) -> 'TextIO':
        pass

class io:
    """Wrapper namespace for IO generic classes."""
    __all__ = ['IO', 'TextIO', 'BinaryIO']
    IO = IO
    TextIO = TextIO
    BinaryIO = BinaryIO
io.__name__ = __name__ + '.io'
sys.modules[io.__name__] = io

Pattern: _TypeAlias = _TypeAlias('Pattern', AnyStr, type(stdlib_re.compile('')), lambda p: p.pattern)
Match: _TypeAlias = _TypeAlias('Match', AnyStr, type(stdlib_re.match('', '')), lambda m: m.re.pattern)

class re:
    """Wrapper namespace for re type aliases."""
    __all__ = ['Pattern', 'Match']
    Pattern = Pattern
    Match = Match
re.__name__ = __name__ + '.re'
sys.modules[re.__name__] = re
