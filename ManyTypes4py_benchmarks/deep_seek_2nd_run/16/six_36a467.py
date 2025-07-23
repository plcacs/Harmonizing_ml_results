"""Utilities for writing code that runs on Python 2 and 3"""
from __future__ import absolute_import
import functools
import itertools
import operator
import sys
import types
from typing import (
    Any, TypeVar, Callable, Dict, List, Tuple, Union, Optional, Iterator, 
    Iterable, Mapping, Sequence, Set, Text, BinaryIO, IO, Type, cast
)

__author__ = 'Benjamin Peterson <benjamin@python.org>'
__version__ = '1.10.0'

PY2: bool = sys.version_info[0] == 2
PY3: bool = sys.version_info[0] == 3
PY34: bool = sys.version_info[0:2] >= (3, 4)

if PY3:
    string_types = (str,)
    integer_types = (int,)
    class_types = (type,)
    text_type = str
    binary_type = bytes
    MAXSIZE = sys.maxsize
else:
    string_types = (basestring,)  # type: ignore
    integer_types = (int, long)  # type: ignore
    class_types = (type, types.ClassType)  # type: ignore
    text_type = unicode  # type: ignore
    binary_type = str
    if sys.platform.startswith('java'):
        MAXSIZE = int((1 << 31) - 1)
    else:
        class X(object):
            def __len__(self) -> int:
                return 1 << 31
        try:
            len(X())
        except OverflowError:
            MAXSIZE = int((1 << 31) - 1)
        else:
            MAXSIZE = int((1 << 63) - 1)
        del X

def _add_doc(func: Callable[..., Any], doc: str) -> None:
    """Add documentation to a function."""
    func.__doc__ = doc

def _import_module(name: str) -> types.ModuleType:
    """Import module, returning the module after the last dot."""
    __import__(name)
    return sys.modules[name]

class _LazyDescr(object):
    def __init__(self, name: str) -> None:
        self.name = name

    def __get__(self, obj: Any, tp: Any) -> Any:
        result = self._resolve()
        setattr(obj, self.name, result)
        try:
            delattr(obj.__class__, self.name)
        except AttributeError:
            pass
        return result

class MovedModule(_LazyDescr):
    def __init__(self, name: str, old: str, new: Optional[str] = None) -> None:
        super(MovedModule, self).__init__(name)
        if PY3:
            if new is None:
                new = name
            self.mod = new
        else:
            self.mod = old

    def _resolve(self) -> types.ModuleType:
        return _import_module(self.mod)

    def __getattr__(self, attr: str) -> Any:
        _module = self._resolve()
        value = getattr(_module, attr)
        setattr(self, attr, value)
        return value

class _LazyModule(types.ModuleType):
    def __init__(self, name: str) -> None:
        super(_LazyModule, self).__init__(name)
        self.__doc__ = self.__class__.__doc__

    def __dir__(self) -> List[str]:
        attrs = ['__doc__', '__name__']
        attrs += [attr.name for attr in self._moved_attributes]
        return attrs
    
    _moved_attributes: List[Any] = []

class MovedAttribute(_LazyDescr):
    def __init__(
        self, 
        name: str, 
        old_mod: str, 
        new_mod: Optional[str], 
        old_attr: Optional[str] = None, 
        new_attr: Optional[str] = None
    ) -> None:
        super(MovedAttribute, self).__init__(name)
        if PY3:
            if new_mod is None:
                new_mod = name
            self.mod = new_mod
            if new_attr is None:
                if old_attr is None:
                    new_attr = name
                else:
                    new_attr = old_attr
            self.attr = new_attr
        else:
            self.mod = old_mod
            if old_attr is None:
                old_attr = name
            self.attr = old_attr

    def _resolve(self) -> Any:
        module = _import_module(self.mod)
        return getattr(module, self.attr)

class _SixMetaPathImporter(object):
    """
    A meta path importer to import six.moves and its submodules.
    """
    def __init__(self, six_module_name: str) -> None:
        self.name = six_module_name
        self.known_modules: Dict[str, Any] = {}

    def _add_module(self, mod: Any, *fullnames: str) -> None:
        for fullname in fullnames:
            self.known_modules[self.name + '.' + fullname] = mod

    def _get_module(self, fullname: str) -> Any:
        return self.known_modules[self.name + '.' + fullname]

    def find_module(self, fullname: str, path: Optional[str] = None) -> Optional['_SixMetaPathImporter']:
        if fullname in self.known_modules:
            return self
        return None

    def __get_module(self, fullname: str) -> Any:
        try:
            return self.known_modules[fullname]
        except KeyError:
            raise ImportError('This loader does not know module ' + fullname)

    def load_module(self, fullname: str) -> Any:
        try:
            return sys.modules[fullname]
        except KeyError:
            pass
        mod = self.__get_module(fullname)
        if isinstance(mod, MovedModule):
            mod = mod._resolve()
        else:
            mod.__loader__ = self
        sys.modules[fullname] = mod
        return mod

    def is_package(self, fullname: str) -> bool:
        """
        Return true, if the named module is a package.
        """
        return hasattr(self.__get_module(fullname), '__path__')

    def get_code(self, fullname: str) -> None:
        """Return None"""
        self.__get_module(fullname)
        return None
    
    get_source = get_code

_importer = _SixMetaPathImporter(__name__)

class _MovedItems(_LazyModule):
    """Lazy loading of moved objects"""
    __path__: List[str] = []

_moved_attributes = [
    MovedAttribute('cStringIO', 'cStringIO', 'io', 'StringIO'),
    MovedAttribute('filter', 'itertools', 'builtins', 'ifilter', 'filter'),
    # ... (rest of the moved attributes remain the same)
]

if sys.platform == 'win32':
    _moved_attributes += [MovedModule('winreg', '_winreg')]

for attr in _moved_attributes:
    setattr(_MovedItems, attr.name, attr)
    if isinstance(attr, MovedModule):
        _importer._add_module(attr, 'moves.' + attr.name)
del attr

_MovedItems._moved_attributes = _moved_attributes
moves = _MovedItems(__name__ + '.moves')
_importer._add_module(moves, 'moves')

# ... (rest of the classes and functions with similar type annotations)

def with_metaclass(meta: Type[Any], *bases: Type[Any]) -> Any:
    """Create a base class with a metaclass."""
    class metaclass(meta):
        def __new__(cls: Type[Any], name: str, this_bases: Tuple[Type[Any], d: Dict[str, Any]) -> Any:
            return meta(name, bases, d)
    return type.__new__(metaclass, 'temporary_class', (), {})

def add_metaclass(metaclass: Type[Any]) -> Callable[[Type[Any]], Type[Any]]:
    """Class decorator for creating a class with a metaclass."""
    def wrapper(cls: Type[Any]) -> Type[Any]:
        orig_vars = cls.__dict__.copy()
        slots = orig_vars.get('__slots__')
        if slots is not None:
            if isinstance(slots, str):
                slots = [slots]
            for slots_var in slots:
                orig_vars.pop(slots_var)
        orig_vars.pop('__dict__', None)
        orig_vars.pop('__weakref__', None)
        return metaclass(cls.__name__, cls.__bases__, orig_vars)
    return wrapper

def python_2_unicode_compatible(klass: Type[Any]) -> Type[Any]:
    """
    A decorator that defines __unicode__ and __str__ methods under Python 2.
    Under Python 3 it does nothing.
    """
    if PY2:
        if '__str__' not in klass.__dict__:
            raise ValueError("@python_2_unicode_compatible cannot be applied to %s because it doesn't define __str__()." % klass.__name__)
        klass.__unicode__ = klass.__str__
        klass.__str__ = lambda self: self.__unicode__().encode('utf-8')
    return klass

__path__: List[str] = []
__package__: str = __name__
if globals().get('__spec__') is not None:
    __spec__.submodule_search_locations = []  # type: ignore

if sys.meta_path:
    for i, importer in enumerate(sys.meta_path):
        if type(importer).__name__ == '_SixMetaPathImporter' and importer.name == __name__:
            del sys.meta_path[i]
            break
    del i, importer

sys.meta_path.append(_importer)
