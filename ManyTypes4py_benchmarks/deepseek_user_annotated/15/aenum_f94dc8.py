"""Python Enumerations"""

import sys as _sys
from typing import (
    Any, Dict, List, Tuple, Set, Iterable, Iterator, Mapping, Optional, TypeVar,
    Union, Callable, Type, cast, Generic, Sequence, OrderedDict as OrderedDictType
)

__all__ = ['Enum', 'IntEnum', 'unique']

version = 1, 1, 6

pyver = float('%s.%s' % _sys.version_info[:2])

T = TypeVar('T')
E = TypeVar('E', bound='Enum')
V = TypeVar('V')
K = TypeVar('K')

try:
    any
except NameError:
    def any(iterable: Iterable[object]) -> bool:
        for element in iterable:
            if element:
                return True
        return False

try:
    from collections import OrderedDict
except ImportError:
    OrderedDict: Optional[Type[OrderedDictType[Any, Any]]] = None

try:
    basestring
except NameError:
    basestring = str

try:
    unicode
except NameError:
    unicode = str

class _RouteClassAttributeToGetattr(object):
    def __init__(self, fget: Optional[Callable[[Any], Any]] = None) -> None:
        self.fget = fget

    def __get__(self, instance: Any, ownerclass: Optional[Type[Any]] = None) -> Any:
        if instance is None:
            raise AttributeError()
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        return self.fget(instance)

    def __set__(self, instance: Any, value: Any) -> None:
        raise AttributeError("can't set attribute")

    def __delete__(self, instance: Any) -> None:
        raise AttributeError("can't delete attribute")

def _is_descriptor(obj: object) -> bool:
    return (
            hasattr(obj, '__get__') or
            hasattr(obj, '__set__') or
            hasattr(obj, '__delete__'))

def _is_dunder(name: str) -> bool:
    return (name[:2] == name[-2:] == '__' and
            name[2:3] != '_' and
            name[-3:-2] != '_' and
            len(name) > 4)

def _is_sunder(name: str) -> bool:
    return (name[0] == name[-1] == '_' and
            name[1:2] != '_' and
            name[-2:-1] != '_' and
            len(name) > 2)

def _make_class_unpicklable(cls: Type[Any]) -> None:
    def _break_on_call_reduce(self: Any, protocol: Optional[int] = None) -> None:
        raise TypeError('%r cannot be pickled' % self)
    cls.__reduce_ex__ = _break_on_call_reduce
    cls.__module__ = '<unknown>'

class _EnumDict(dict):
    def __init__(self) -> None:
        super(_EnumDict, self).__init__()
        self._member_names: List[str] = []

    def __setitem__(self, key: str, value: Any) -> None:
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
            raise TypeError('Attempted to reuse key: %r' % key)
        elif not _is_descriptor(value):
            if key in self:
                raise TypeError('Key already defined as: %r' % self[key])
            self._member_names.append(key)
        super(_EnumDict, self).__setitem__(key, value)

Enum = None

class EnumMeta(type):
    @classmethod
    def __prepare__(metacls: Type['EnumMeta'], cls: str, bases: Tuple[Type[Any], ...]) -> _EnumDict:
        return _EnumDict()

    def __new__(metacls: Type['EnumMeta'],
                cls: str,
                bases: Tuple[Type[Any], ...],
                classdict: _EnumDict) -> 'EnumMeta':
        if type(classdict) is dict:
            original_dict = classdict
            classdict = _EnumDict()
            for k, v in original_dict.items():
                classdict[k] = v

        member_type, first_enum = metacls._get_mixins_(bases)
        __new__, save_new, use_args = metacls._find_new_(classdict, member_type, first_enum)
        members = dict((k, classdict[k]) for k in classdict._member_names)
        for name in classdict._member_names:
            del classdict[name]

        _order_ = classdict.get('_order_')
        if _order_ is None:
            if pyver < 3.0:
                try:
                    _order_ = [name for (name, value) in sorted(members.items(), key=lambda item: item[1])]
                except TypeError:
                    _order_ = [name for name in sorted(members.keys())]
            else:
                _order_ = classdict._member_names
        else:
            del classdict['_order_']
            if pyver < 3.0:
                _order_ = _order_.replace(',', ' ').split()
                aliases = [name for name in members if name not in _order_]
                _order_ += aliases

        invalid_names = set(members) & set(['mro'])
        if invalid_names:
            raise ValueError('Invalid enum member name(s): %s' % (', '.join(invalid_names), ))

        base_attributes = set([a for b in bases for a in b.__dict__])
        enum_class = super(EnumMeta, metacls).__new__(metacls, cls, bases, classdict)
        enum_class._member_names_: List[str] = []
        if OrderedDict is not None:
            enum_class._member_map_: OrderedDictType[str, Any] = OrderedDict()
        else:
            enum_class._member_map_: Dict[str, Any] = {}
        enum_class._member_type_ = member_type
        enum_class._value2member_map_: Dict[Any, Any] = {}

        if __new__ is None:
            __new__ = enum_class.__new__
        for member_name in _order_:
            value = members[member_name]
            if not isinstance(value, tuple):
                args = (value, )
            else:
                args = value
            if member_type is tuple:
                args = (args, )
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
                methods = ('__getnewargs_ex__', '__getnewargs__',
                        '__reduce_ex__', '__reduce__')
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
                for method in (
                        '__le__',
                        '__lt__',
                        '__gt__',
                        '__ge__',
                        '__eq__',
                        '__ne__',
                        '__hash__',
                        ):
                    setattr(enum_class, method, getattr(int, method))

        if Enum is not None:
            if save_new:
                setattr(enum_class, '__member_new__', enum_class.__dict__['__new__'])
            setattr(enum_class, '__new__', Enum.__dict__['__new__'])
        return enum_class

    def __bool__(cls) -> bool:
        return True

    def __call__(cls: Type[E],
                 value: Any,
                 names: Optional[Union[str, Iterable[Union[str, Tuple[str, Any]]]] = None,
                 module: Optional[str] = None,
                 type: Optional[Type[Any]] = None,
                 start: int = 1) -> Union[E, Type[E]]:
        if names is None:
            return cls.__new__(cls, value)
        return cls._create_(value, names, module=module, type=type, start=start)

    def __contains__(cls: Type[E], member: object) -> bool:
        return isinstance(member, cls) and member.name in cls._member_map_

    def __delattr__(cls: Type[E], attr: str) -> None:
        if attr in cls._member_map_:
            raise AttributeError("%s: cannot delete Enum member." % cls.__name__)
        super(EnumMeta, cls).__delattr__(attr)

    def __dir__(self) -> List[str]:
        return (['__class__', '__doc__', '__members__', '__module__'] +
                self._member_names_)

    @property
    def __members__(cls: Type[E]) -> Dict[str, E]:
        return cls._member_map_.copy()

    def __getattr__(cls: Type[E], name: str) -> E:
        if _is_dunder(name):
            raise AttributeError(name)
        try:
            return cls._member_map_[name]
        except KeyError:
            raise AttributeError(name)

    def __getitem__(cls: Type[E], name: str) -> E:
        return cls._member_map_[name]

    def __iter__(cls: Type[E]) -> Iterator[E]:
        return (cls._member_map_[name] for name in cls._member_names_)

    def __reversed__(cls: Type[E]) -> Iterator[E]:
        return (cls._member_map_[name] for name in reversed(cls._member_names_))

    def __len__(cls: Type[E]) -> int:
        return len(cls._member_names_)

    __nonzero__ = __bool__

    def __repr__(cls) -> str:
        return "<enum %r>" % cls.__name__

    def __setattr__(cls: Type[E], name: str, value: Any) -> None:
        member_map = cls.__dict__.get('_member_map_', {})
        if name in member_map:
            raise AttributeError('Cannot reassign members.')
        super(EnumMeta, cls).__setattr__(name, value)

    def _create_(cls: Type[E],
                 class_name: str,
                 names: Optional[Union[str, Iterable[Union[str, Tuple[str, Any]]]] = None,
                 module: Optional[str] = None,
                 type: Optional[Type[Any]] = None,
                 start: int = 1) -> Type[E]:
        if pyver < 3.0:
            if isinstance(class_name, unicode):
                try:
                    class_name = class_name.encode('ascii')
                except UnicodeEncodeError:
                    raise TypeError('%r is not representable in ASCII' % class_name)
        metacls = cls.__class__
        if type is None:
            bases = (cls, )
        else:
            bases = (type, cls)
        classdict = metacls.__prepare__(class_name, bases)
        _order_: List[str] = []

        if isinstance(names, basestring):
            names = names.replace(',', ' ').split()
        if isinstance(names, (tuple, list)) and isinstance(names[0], basestring):
            names = [(e, i+start) for (i, e) in enumerate(names)]

        item = None
        for item in names:
            if isinstance(item, basestring):
                member_name, member_value = item, names[item]
            else:
                member_name, member_value = item
            classdict[member_name] = member_value
            _order_.append(member_name)
        if not isinstance(item, basestring):
            classdict['_order_'] = ' '.join(_order_)
        enum_class = metacls.__new__(metacls, class_name, bases, classdict)

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
    def _get_mixins_(bases: Tuple[Type[Any], ...]) -> Tuple[Type[Any], Optional[Type['Enum']]]:
        if not bases or Enum is None:
            return object, Enum

        member_type = first_enum = None
        for base in bases:
            if  (base is not Enum and
                    issubclass(base, Enum) and
                    base._member_names_):
                raise TypeError("Cannot extend enumerations")
        if not issubclass(bases[-1], Enum):
            raise TypeError("new enumerations must be created as "
                    "`ClassName([mixin_type,] enum_type)`")

        if not issubclass(bases[0], Enum):
            member_type = bases[0]
            first_enum = bases[-1]
        else:
            for base in bases[0].__mro__:
                if issubclass(base, Enum):
                    if first_enum is None:
                        first_enum = base
                else:
                    if member_type is None:
                        member_type = base

        return member_type, first_enum

    if pyver < 3.0:
        @staticmethod
        def _find_new_(classdict: _EnumDict,
                      member_type: Type[Any],
                      first_enum: Optional[Type['Enum']]) -> Tuple[Optional[Callable[..., Any]], bool, bool]:
            __new__ = classdict.get('__new__', None)
            if __new__:
                return None, True, True

            N__new__ = getattr(None, '__new__')
            O__new__ = getattr(object, '__new__')
            if Enum is None:
                E__new__ = N__new__
            else:
                E__new__ = Enum.__dict__['__new__']
            for method in ('__member_new__', '__new__'):
                for possible in (member_type, first_enum):
                    try:
                        target = possible.__dict__[method]
                    except (AttributeError, KeyError):
                        target = getattr(possible, method, None)
                    if target not in [
                            None,
                            N__new__,
                            O__new__,
                            E__new__,
                            ]:
                        if method == '__member_new__':
                            classdict['__new__'] = target
                            return None, False, True
                        if isinstance(target, staticmethod):
                            target = target.__get__(member_type)
                        __new__ = target
                        break
                if __new__ is not None:
                    break
            else:
                __new__ = object.__new__

            use_args = __new__ is not object.__new__
            return __new__, False, use_args
    else:
        @staticmethod
        def _find_new_(classdict: _EnumDict,
                      member_type: Type[Any],
                      first_enum: Optional[Type['Enum']]) -> Tuple[Optional[Callable[..., Any]], bool, bool]:
            __new__ = classdict.get('__new__', None)
            save_new = __new__ is not None

            if __new__ is None:
                for method in ('__member_new__', '__new__'):
                    for possible in (member_type, first_enum):
                        target = getattr(possible, method, None)
                        if target not in (
                                None,
                                None.__new__,
                                object.__new__,
                                Enum.__new__,
                                ):
                            __new__ = target
                            break
                    if __new__ is not None:
                        break
                else:
                    __new__ = object.__new__

            use_args = __new__ is not object.__new__
            return __new__, save_new, use_args

temp_enum_dict: Dict[str, Any] = {}
temp_enum_dict['__doc__'] = "Generic enumeration.\n\n    Derive from this class to define new enumerations.\n\n"

def __new__(cls: Type[E], value: Any) -> E:
    if type(value) is cls:
        value = value.value
    try:
        if value in cls._value2member_map_:
            return cls._value2member_map_[value]
    except TypeError:
        for member in cls._member_map_.values():
            if member.value == value:
                return member
    raise ValueError("%s is not a valid %s" % (value, cls.__name__))
