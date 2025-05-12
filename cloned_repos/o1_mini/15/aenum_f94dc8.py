"""Python Enumerations"""
import sys as _sys
from collections import OrderedDict
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    NoReturn,
)

__all__ = ['Enum', 'IntEnum', 'unique']
version: Tuple[int, int, int] = (1, 1, 6)
pyver: float = float('%s.%s' % _sys.version_info[:2])

try:
    any  # type: ignore
except NameError:

    def any(iterable: Iterable[Any]) -> bool:
        for element in iterable:
            if element:
                return True
        return False

try:
    from collections import OrderedDict
except ImportError:
    OrderedDict = None

try:
    basestring
except NameError:
    basestring = str

try:
    unicode
except NameError:
    unicode = str


class _RouteClassAttributeToGetattr:
    """Route attribute access on a class to __getattr__.

    This is a descriptor, used to define attributes that act differently when
    accessed through an instance and through a class.  Instance access remains
    normal, but access to an attribute through a class will be routed to the
    class's __getattr__ method; this is done by raising AttributeError.
    """

    def __init__(self, fget: Optional[Callable[['Enum'], Any]] = None) -> None:
        self.fget = fget

    def __get__(
        self, instance: Optional['Enum'], ownerclass: Optional[Type[Any]] = None
    ) -> Any:
        if instance is None:
            raise AttributeError()
        return self.fget(instance)  # type: ignore

    def __set__(self, instance: Any, value: Any) -> None:
        raise AttributeError("can't set attribute")

    def __delete__(self, instance: Any) -> None:
        raise AttributeError("can't delete attribute")


def _is_descriptor(obj: Any) -> bool:
    """Returns True if obj is a descriptor, False otherwise."""
    return hasattr(obj, '__get__') or hasattr(obj, '__set__') or hasattr(obj, '__delete__')


def _is_dunder(name: str) -> bool:
    """Returns True if a __dunder__ name, False otherwise."""
    return (
        name.startswith('__')
        and name.endswith('__')
        and name[2:3] != '_'
        and name[-3:-2] != '_'
        and len(name) > 4
    )


def _is_sunder(name: str) -> bool:
    """Returns True if a _sunder_ name, False otherwise."""
    return (
        name.startswith('_')
        and name.endswith('_')
        and name[1:2] != '_'
        and name[-2:-1] != '_'
        and len(name) > 2
    )


def _make_class_unpicklable(cls: Type[Any]) -> None:
    """Make the given class un-picklable."""

    def _break_on_call_reduce(self: Any, protocol: Any = None) -> Any:
        raise TypeError('%r cannot be pickled' % self)

    cls.__reduce_ex__ = _break_on_call_reduce
    cls.__module__ = '<unknown>'


class _EnumDict(dict):
    """Track enum member order and ensure member names are not reused.

    EnumMeta will use the names found in self._member_names as the
    enumeration member names.
    """

    def __init__(self) -> None:
        super(_EnumDict, self).__init__()
        self._member_names: List[str] = []

    def __setitem__(self, key: str, value: Any) -> None:
        """Changes anything not dundered or not a descriptor.

        If a descriptor is added with the same name as an enum member, the name
        is removed from _member_names (this may leave a hole in the numerical
        sequence of values).

        If an enum member name is used twice, an error is raised; duplicate
        values are not checked for.

        Single underscore (sunder) names are reserved.

        Note:   in 3.x __order__ is simply discarded as a not necessary piece
                leftover from 2.x
        """
        if pyver >= 3.0 and key in ('_order_', '__order__'):
            return
        elif key == '__order__':
            key = '_order_'
        if _is_sunder(key):
            if key != '_order_':
                raise ValueError('_names_ are reserved for future Enum use')
        elif _is_dunder(key):
            pass
        elif key in self._member_names:
            raise TypeError(f'Attempted to reuse key: {key!r}')
        elif not _is_descriptor(value):
            if key in self:
                raise TypeError(f'Key already defined as: {self[key]!r}')
            self._member_names.append(key)
        super(_EnumDict, self).__setitem__(key, value)


Enum: Optional[Type[Any]] = None  # To be defined later


class EnumMeta(type):
    """Metaclass for Enum"""

    @classmethod
    def __prepare__(metacls, cls: str, bases: Tuple[Type[Any], ...]) -> _EnumDict:
        return _EnumDict()

    def __new__(
        metacls,
        cls: str,
        bases: Tuple[Type[Any], ...],
        classdict: _EnumDict,
    ) -> Type[Any]:
        if type(classdict) is dict:
            original_dict = classdict
            classdict = _EnumDict()
            for k, v in original_dict.items():
                classdict[k] = v
        member_type, first_enum = metacls._get_mixins_(bases)
        __new__, save_new, use_args = metacls._find_new_(classdict, member_type, first_enum)
        members: Dict[str, Any] = {k: classdict[k] for k in classdict._member_names}
        for name in classdict._member_names:
            del classdict[name]
        _order_ = classdict.get('_order_')
        if _order_ is None:
            if pyver < 3.0:
                try:
                    _order_ = [name for name, value in sorted(members.items(), key=lambda item: item[1])]
                except TypeError:
                    _order_ = sorted(members.keys())
            else:
                _order_ = classdict._member_names
        else:
            del classdict['_order_']
            if pyver < 3.0:
                _order_ = _order_.replace(',', ' ').split()
                aliases = [name for name in members if name not in _order_]
                _order_ += aliases
        invalid_names = set(members) & {'mro'}
        if invalid_names:
            raise ValueError(f'Invalid enum member name(s): {", ".join(invalid_names)}')
        base_attributes = set(a for b in bases for a in b.__dict__)
        enum_class = super(EnumMeta, metacls).__new__(metacls, cls, bases, classdict)
        enum_class._member_names_: List[str] = []
        if OrderedDict is not None:
            enum_class._member_map_: Union[OrderedDict[str, Any], Dict[str, Any]] = OrderedDict()
        else:
            enum_class._member_map_: Dict[str, Any] = {}
        enum_class._member_type_: Type[Any] = member_type
        enum_class._value2member_map_: Dict[Any, Any] = {}
        if __new__ is None:
            __new__ = enum_class.__new__
        for member_name in _order_:
            value = members[member_name]
            if not isinstance(value, tuple):
                args = (value,)
            else:
                args = value
            if member_type is tuple:
                args = (args,)
            if not use_args or not args:
                enum_member = __new__(enum_class)
                if not hasattr(enum_member, '_value_'):
                    enum_member._value_ = value
            else:
                enum_member = __new__(enum_class, *args)
                if not hasattr(enum_member, '_value_'):
                    enum_member._value_ = member_type(*args)
            value = enum_member._value_
            enum_member._name_ = member_name
            enum_member.__objclass__ = enum_class
            enum_member.__init__(*args)
            for name, canonical_member in enum_class._member_map_.items():
                if canonical_member.value == enum_member._value_:
                    enum_member = canonical_member
                    break
            else:
                enum_class._member_names_.append(member_name)
            if member_name not in base_attributes:
                setattr(enum_class, member_name, enum_member)
            enum_class._member_map_[member_name] = enum_member
            try:
                enum_class._value2member_map_[value] = enum_member
            except TypeError:
                pass
        unpicklable = False
        if '__reduce_ex__' not in classdict:
            if member_type is not object:
                methods = ('__getnewargs_ex__', '__getnewargs__', '__reduce_ex__', '__reduce__')
                if not any(m in member_type.__dict__ for m in methods):
                    _make_class_unpicklable(enum_class)
                    unpicklable = True
        for name in ('__repr__', '__str__', '__format__', '__reduce_ex__'):
            class_method = getattr(enum_class, name)
            obj_method = getattr(member_type, name, None)
            enum_method = getattr(first_enum, name, None)
            if name not in classdict and class_method is not enum_method:
                if name == '__reduce_ex__' and unpicklable:
                    continue
                setattr(enum_class, name, enum_method)
        if pyver < 2.6:
            if issubclass(enum_class, int):
                setattr(enum_class, '__cmp__', getattr(int, '__cmp__'))
        elif pyver < 3.0:
            if issubclass(enum_class, int):
                for method in ('__le__', '__lt__', '__gt__', '__ge__', '__eq__', '__ne__', '__hash__'):
                    setattr(enum_class, method, getattr(int, method))
        if Enum is not None:
            if save_new:
                setattr(enum_class, '__member_new__', enum_class.__dict__['__new__'])
            setattr(enum_class, '__new__', Enum.__dict__['__new__'])
        return enum_class

    def __bool__(cls) -> bool:
        """
        classes/types should always be True.
        """
        return True

    def __call__(
        cls,
        value: Any,
        names: Optional[Union[str, Iterable[Any]]] = None,
        module: Optional[str] = None,
        type: Optional[Type[Any]] = None,
        start: int = 1,
    ) -> Any:
        """Either returns an existing member, or creates a new enum class.

        This method is used both when an enum class is given a value to match
        to an enumeration member (i.e. Color(3)) and for the functional API
        (i.e. Color = Enum('Color', names='red green blue')).

        When used for the functional API: `module`, if set, will be stored in
        the new class' __module__ attribute; `type`, if set, will be mixed in
        as the first base class.

        Note: if `module` is not set this routine will attempt to discover the
        calling module by walking the frame stack; if this is unsuccessful
        the resulting class will not be pickleable.
        """
        if names is None:
            return cls.__new__(cls, value)
        return cls._create_(value, names, module=module, type=type, start=start)

    def __contains__(cls, member: Any) -> bool:
        return isinstance(member, cls) and member.name in cls._member_map_

    def __delattr__(cls, attr: str) -> None:
        if attr in cls._member_map_:
            raise AttributeError(f'{cls.__name__}: cannot delete Enum member.')
        super(EnumMeta, cls).__delattr__(attr)

    def __dir__(self) -> List[str]:
        return ['__class__', '__doc__', '__members__', '__module__'] + self._member_names_

    @property
    def __members__(cls) -> Dict[str, Any]:
        """Returns a mapping of member name->value.

        This mapping lists all enum members, including aliases. Note that this
        is a copy of the internal mapping.
        """
        return cls._member_map_.copy()

    def __getattr__(cls, name: str) -> Any:
        """Return the enum member matching `name`

        We use __getattr__ instead of descriptors or inserting into the enum
        class' __dict__ in order to support `name` and `value` being both
        properties for enum members (which live in the class' __dict__) and
        enum members themselves.
        """
        if _is_dunder(name):
            raise AttributeError(name)
        try:
            return cls._member_map_[name]
        except KeyError:
            raise AttributeError(name)

    def __getitem__(cls, name: str) -> Any:
        return cls._member_map_[name]

    def __iter__(cls) -> Iterator[Any]:
        return (cls._member_map_[name] for name in cls._member_names_)

    def __reversed__(cls) -> Iterator[Any]:
        return (cls._member_map_[name] for name in reversed(cls._member_names_))

    def __len__(cls) -> int:
        return len(cls._member_names_)

    __nonzero__ = __bool__

    def __repr__(cls) -> str:
        return f'<enum {cls.__name__!r}>'

    def __setattr__(cls, name: str, value: Any) -> None:
        """Block attempts to reassign Enum members.

        A simple assignment to the class namespace only changes one of the
        several possible ways to get an Enum member from the Enum class,
        resulting in an inconsistent Enumeration.
        """
        member_map = cls.__dict__.get('_member_map_', {})
        if name in member_map:
            raise AttributeError('Cannot reassign members.')
        super(EnumMeta, cls).__setattr__(name, value)

    def _create_(
        cls,
        class_name: str,
        names: Union[str, Iterable[Any]],
        module: Optional[str] = None,
        type: Optional[Type[Any]] = None,
        start: int = 1,
    ) -> Type[Any]:
        """Convenience method to create a new Enum class.

        `names` can be:

        * A string containing member names, separated either with spaces or
          commas.  Values are auto-numbered from 1.
        * An iterable of member names.  Values are auto-numbered from 1.
        * An iterable of (member name, value) pairs.
        * A mapping of member name -> value.
        """
        if pyver < 3.0:
            if isinstance(class_name, unicode):
                try:
                    class_name = class_name.encode('ascii')  # type: ignore
                except UnicodeEncodeError:
                    raise TypeError(f'{class_name!r} is not representable in ASCII')
        metacls = cls.__class__
        if type is None:
            bases: Tuple[Type[Any], ...] = (cls,)
        else:
            bases = (type, cls)
        classdict: _EnumDict = metacls.__prepare__(class_name, bases)
        _order_: List[str] = []
        if isinstance(names, str):
            names = names.replace(',', ' ').split()
        if isinstance(names, (tuple, list)) and isinstance(names[0], str):
            names = [(e, i + start) for i, e in enumerate(names)]
        item: Any = None
        for item in names:
            if isinstance(item, str):
                member_name, member_value = (item, names[item])  # type: ignore
            else:
                member_name, member_value = item
            classdict[member_name] = member_value
            _order_.append(member_name)
        if not isinstance(item, str):
            classdict['_order_'] = ' '.join(_order_)
        enum_class: Type[Any] = metacls.__new__(metacls, class_name, bases, classdict)
        if module is None:
            try:
                module = _sys._getframe(2).f_globals['__name__']
            except (AttributeError, ValueError):
                pass
        if module is None:
            _make_class_unpicklable(enum_class)
        else:
            enum_class.__module__ = module
        return enum_class

    @staticmethod
    def _get_mixins_(bases: Tuple[Type[Any], ...]) -> Tuple[Type[Any], Optional[Type[Any]]]:
        """Returns the type for creating enum members, and the first inherited
        enum class.

        bases: the tuple of bases that was given to __new__
        """
        if not bases or Enum is None:
            return (object, Enum)
        member_type: Optional[Type[Any]] = None
        first_enum: Optional[Type[Any]] = None
        for base in bases:
            if base is not Enum and issubclass(base, Enum) and base._member_names_:
                raise TypeError('Cannot extend enumerations')
        base = bases[-1]
        if not issubclass(base, Enum):
            raise TypeError('new enumerations must be created as `ClassName([mixin_type,] enum_type)`')
        if not issubclass(bases[0], Enum):
            member_type = bases[0]
            first_enum = bases[-1]
        else:
            for b in bases[0].__mro__:
                if issubclass(b, Enum):
                    if first_enum is None:
                        first_enum = b
                elif member_type is None:
                    member_type = b
        return (member_type, first_enum)

    if pyver < 3.0:

        @staticmethod
        def _find_new_(
            classdict: _EnumDict, member_type: Type[Any], first_enum: Optional[Type[Any]]
        ) -> Tuple[Optional[Callable[..., Any]], bool, bool]:
            """Returns the __new__ to be used for creating the enum members.

            classdict: the class dictionary given to __new__
            member_type: the data type whose __new__ will be used by default
            first_enum: enumeration to check for an overriding __new__
            """
            __new__ = classdict.get('__new__', None)
            if __new__:
                return (None, True, True)
            N__new__ = getattr(None, '__new__')
            O__new__ = getattr(object, '__new__')
            if Enum is None:
                E__new__ = N__new__
            else:
                E__new__ = Enum.__dict__['__new__']
            for method in ('__member_new__', '__new__'):
                for possible in (member_type, first_enum):
                    try:
                        target = possible.__dict__[method]  # type: ignore
                    except (AttributeError, KeyError):
                        target = getattr(possible, method, None)
                    if target not in [None, N__new__, O__new__, E__new__]:
                        if method == '__member_new__':
                            classdict['__new__'] = target
                            return (None, False, True)
                        if isinstance(target, staticmethod):
                            target = target.__get__(member_type)
                        __new__ = target
                        break
                if __new__ is not None:
                    break
            else:
                __new__ = object.__new__
            if __new__ is object.__new__:
                use_args = False
            else:
                use_args = True
            return (__new__, False, use_args)

    else:

        @staticmethod
        def _find_new_(
            classdict: _EnumDict, member_type: Type[Any], first_enum: Optional[Type[Any]]
        ) -> Tuple[Callable[..., Any], bool, bool]:
            """Returns the __new__ to be used for creating the enum members.

            classdict: the class dictionary given to __new__
            member_type: the data type whose __new__ will be used by default
            first_enum: enumeration to check for an overriding __new__
            """
            __new__ = classdict.get('__new__', None)
            save_new = __new__ is not None
            if __new__ is None:
                for method in ('__member_new__', '__new__'):
                    for possible in (member_type, first_enum):
                        target = getattr(possible, method, None)
                        if target not in (None, None.__new__, object.__new__, Enum.__new__):
                            __new__ = target
                            break
                    if __new__ is not None:
                        break
                else:
                    __new__ = object.__new__
            if __new__ is object.__new__:
                use_args = False
            else:
                use_args = True
            return (__new__, save_new, use_args)


temp_enum_dict: Dict[str, Any] = {}
temp_enum_dict['__doc__'] = (
    'Generic enumeration.\n\n    Derive from this class to define new enumerations.\n\n'
)


def __new__(cls: Type[Any], value: Any) -> Any:
    if type(value) is cls:
        value = value.value
    try:
        if value in cls._value2member_map_:
            return cls._value2member_map_[value]
    except TypeError:
        for member in cls._member_map_.values():
            if member.value == value:
                return member
    raise ValueError(f'{value} is not a valid {cls.__name__}')


temp_enum_dict['__new__'] = __new__
del __new__


def __repr__(self: Any) -> str:
    return f'<{self.__class__.__name__}.{self._name_}: {self._value_!r}>'


temp_enum_dict['__repr__'] = __repr__
del __repr__


def __str__(self: Any) -> str:
    return f'{self.__class__.__name__}.{self._name_}'


temp_enum_dict['__str__'] = __str__
del __str__

if pyver >= 3.0:

    def __dir__(self: Any) -> List[str]:
        added_behavior = [
            m
            for cls in self.__class__.mro()
            for m in cls.__dict__
            if m[0] != '_' and m not in self._member_map_
        ]
        return ['__class__', '__doc__', '__module__'] + added_behavior

    temp_enum_dict['__dir__'] = __dir__
    del __dir__


def __format__(self: Any, format_spec: str) -> str:
    if self._member_type_ is object:
        cls = str
        val = str(self)
    else:
        cls = self._member_type_
        val = self.value
    return cls.__format__(val, format_spec)


temp_enum_dict['__format__'] = __format__
del __format__

if pyver < 2.6:

    def __cmp__(self: Any, other: Any) -> int:
        if type(other) is self.__class__:
            if self is other:
                return 0
            return -1
        return NotImplemented
        raise TypeError(
            f'unorderable types: {self.__class__.__name__}() and {other.__class__.__name__}()'
        )

    temp_enum_dict['__cmp__'] = __cmp__
    del __cmp__

else:

    def __le__(self: Any, other: Any) -> NoReturn:
        raise TypeError(
            f'unorderable types: {self.__class__.__name__}() <= {other.__class__.__name__}()'
        )

    temp_enum_dict['__le__'] = __le__
    del __le__

    def __lt__(self: Any, other: Any) -> NoReturn:
        raise TypeError(
            f'unorderable types: {self.__class__.__name__}() < {other.__class__.__name__}()'
        )

    temp_enum_dict['__lt__'] = __lt__
    del __lt__

    def __ge__(self: Any, other: Any) -> NoReturn:
        raise TypeError(
            f'unorderable types: {self.__class__.__name__}() >= {other.__class__.__name__}()'
        )

    temp_enum_dict['__ge__'] = __ge__
    del __ge__

    def __gt__(self: Any, other: Any) -> NoReturn:
        raise TypeError(
            f'unorderable types: {self.__class__.__name__}() > {other.__class__.__name__}()'
        )

    temp_enum_dict['__gt__'] = __gt__
    del __gt__


def __eq__(self: Any, other: Any) -> Union[bool, NotImplemented]:
    if type(other) is self.__class__:
        return self is other
    return NotImplemented


temp_enum_dict['__eq__'] = __eq__
del __eq__


def __ne__(self: Any, other: Any) -> Union[bool, NotImplemented]:
    if type(other) is self.__class__:
        return self is not other
    return NotImplemented


temp_enum_dict['__ne__'] = __ne__
del __ne__


def __hash__(self: Any) -> int:
    return hash(self._name_)


temp_enum_dict['__hash__'] = __hash__
del __hash__


def __reduce_ex__(self: Any, proto: int) -> Tuple[Any, Tuple[Any, ...]]:
    return (self.__class__, (self._value_,))


temp_enum_dict['__reduce_ex__'] = __reduce_ex__
del __reduce_ex__


@_RouteClassAttributeToGetattr
def name(self: Any) -> str:
    return self._name_


temp_enum_dict['name'] = name
del name


@_RouteClassAttributeToGetattr
def value(self: Any) -> Any:
    return self._value_


temp_enum_dict['value'] = value
del value


@classmethod
def _convert(
    cls: Type[Any],
    name: str,
    module: str,
    filter: Callable[[str], bool],
    source: Optional[Any] = None,
) -> Type[Any]:
    """
    Create a new Enum subclass that replaces a collection of global constants
    """
    module_globals = vars(_sys.modules[module])
    if source:
        source = vars(source)
    else:
        source = module_globals
    members = {name: value for name, value in source.items() if filter(name)}
    cls_new = cls(name, members, module=module)
    cls_new.__reduce_ex__ = _reduce_ex_by_name  # type: ignore
    module_globals.update(cls_new.__members__)
    module_globals[name] = cls_new
    return cls_new


temp_enum_dict['_convert'] = _convert
del _convert


Enum = EnumMeta('Enum', (object,), temp_enum_dict)
del temp_enum_dict


class IntEnum(int, Enum):
    """Enum where members are also (and must be) ints"""
    pass


def _reduce_ex_by_name(self: Any, proto: int) -> Any:
    return self.name


def unique(enumeration: Type[Any]) -> Type[Any]:
    """Class decorator that ensures only unique members exist in an enumeration."""
    duplicates: List[Tuple[str, str]] = []
    for name, member in enumeration.__members__.items():
        if name != member.name:
            duplicates.append((name, member.name))
    if duplicates:
        duplicate_names = ', '.join([f'{alias} -> {name}' for alias, name in duplicates])
        raise ValueError(f'duplicate names found in {enumeration!r}: {duplicate_names}')
    return enumeration
