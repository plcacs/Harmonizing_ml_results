"""Utilities for writing code that runs on Python 2 and 3"""
from __future__ import absolute_import
import functools
import itertools
import operator
import sys
import types
from typing import (
    Any, Dict, List, Tuple, Union, Optional, Callable, TypeVar, Type, Iterator as TIterator,
    Iterable, Mapping, Set, Sequence, TextIO, BinaryIO, cast, overload
)

__author__ = 'Benjamin Peterson <benjamin@python.org>'
__version__ = '1.10.0'

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

PY2: bool = sys.version_info[0] == 2
PY3: bool = sys.version_info[0] == 3
PY34: bool = sys.version_info[0:2] >= (3, 4)

if PY3:
    string_types: Tuple[type, ...] = (str,)
    integer_types: Tuple[type, ...] = (int,)
    class_types: Tuple[type, ...] = (type,)
    text_type: type = str
    binary_type: type = bytes
    MAXSIZE: int = sys.maxsize
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

def _add_doc(func: F, doc: str) -> F:
    """Add documentation to a function."""
    func.__doc__ = doc
    return func

def _import_module(name: str) -> types.ModuleType:
    """Import module, returning the module after the last dot."""
    __import__(name)
    return sys.modules[name]

class _LazyDescr(object):
    def __init__(self, name: str) -> None:
        self.name = name

    def __get__(self, obj: Any, tp: type) -> Any:
        result = self._resolve()
        setattr(obj, self.name, result)
        try:
            delattr(obj.__class__, self.name)
        except AttributeError:
            pass
        return result

    def _resolve(self) -> Any:
        raise NotImplementedError

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

    def load_module(self, fullname: str) -> types.ModuleType:
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
        return hasattr(self.__get_module(fullname), '__path__')

    def get_code(self, fullname: str) -> None:
        self.__get_module(fullname)
        return None
    
    get_source = get_code

_importer: _SixMetaPathImporter = _SixMetaPathImporter(__name__)

class _MovedItems(_LazyModule):
    """Lazy loading of moved objects"""
    __path__: List[str] = []

_moved_attributes: List[MovedAttribute] = [
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
moves: _MovedItems = _MovedItems(__name__ + '.moves')
_importer._add_module(moves, 'moves')

# ... (rest of the module classes remain similar with type annotations added)

def add_move(move: Union[MovedAttribute, MovedModule]) -> None:
    """Add an item to six.moves."""
    setattr(_MovedItems, move.name, move)

def remove_move(name: str) -> None:
    """Remove item from six.moves."""
    try:
        delattr(_MovedItems, name)
    except AttributeError:
        try:
            del moves.__dict__[name]
        except KeyError:
            raise AttributeError('no such move, %r' % (name,))

if PY3:
    _meth_func: str = '__func__'
    _meth_self: str = '__self__'
    _func_closure: str = '__closure__'
    _func_code: str = '__code__'
    _func_defaults: str = '__defaults__'
    _func_globals: str = '__globals__'
else:
    _meth_func = 'im_func'
    _meth_self = 'im_self'
    _func_closure = 'func_closure'
    _func_code = 'func_code'
    _func_defaults = 'func_defaults'
    _func_globals = 'func_globals'

try:
    advance_iterator: Callable[[TIterator[T]], T] = next
except NameError:
    def advance_iterator(it: TIterator[T]) -> T:
        return it.next()

next = advance_iterator

try:
    callable: Callable[[Any], bool] = callable  # type: ignore
except NameError:
    def callable(obj: Any) -> bool:
        return any(('__call__' in klass.__dict__ for klass in type(obj).__mro__))

if PY3:
    def get_unbound_function(unbound: Callable[..., Any]) -> Callable[..., Any]:
        return unbound
    
    create_bound_method: Callable[[Callable[..., Any], Any], Callable[..., Any]] = types.MethodType

    def create_unbound_method(func: Callable[..., Any], cls: type) -> Callable[..., Any]:
        return func
    
    Iterator: type = object
else:
    def get_unbound_function(unbound: Callable[..., Any]) -> Callable[..., Any]:
        return unbound.im_func

    def create_bound_method(func: Callable[..., Any], obj: Any) -> Callable[..., Any]:
        return types.MethodType(func, obj, obj.__class__)

    def create_unbound_method(func: Callable[..., Any], cls: type) -> Callable[..., Any]:
        return types.MethodType(func, None, cls)

    class Iterator(object):
        def next(self) -> Any:
            return type(self).__next__(self)

_add_doc(get_unbound_function, 'Get the function out of a possibly unbound function')

get_method_function: Callable[[Any], Callable[..., Any]] = operator.attrgetter(_meth_func)
get_method_self: Callable[[Any], Any] = operator.attrgetter(_meth_self)
get_function_closure: Callable[[Any], Any] = operator.attrgetter(_func_closure)
get_function_code: Callable[[Any], Any] = operator.attrgetter(_func_code)
get_function_defaults: Callable[[Any], Any] = operator.attrgetter(_func_defaults)
get_function_globals: Callable[[Any], Dict[str, Any]] = operator.attrgetter(_func_globals)

if PY3:
    def iterkeys(d: Mapping[T, Any], **kw: Any) -> TIterator[T]:
        return iter(d.keys(**kw))

    def itervalues(d: Mapping[Any, T], **kw: Any) -> TIterator[T]:
        return iter(d.values(**kw))

    def iteritems(d: Mapping[T, V], **kw: Any) -> TIterator[Tuple[T, V]]:
        return iter(d.items(**kw))

    def iterlists(d: Mapping[T, Sequence[V]], **kw: Any) -> TIterator[Tuple[T, Sequence[V]]]:
        return iter(d.lists(**kw))
    
    viewkeys: Callable[[Mapping[T, Any]], Any] = operator.methodcaller('keys')
    viewvalues: Callable[[Mapping[Any, T]], Any] = operator.methodcaller('values')
    viewitems: Callable[[Mapping[T, V]], Any] = operator.methodcaller('items')
else:
    def iterkeys(d: Mapping[T, Any], **kw: Any) -> TIterator[T]:
        return d.iterkeys(**kw)  # type: ignore

    def itervalues(d: Mapping[Any, T], **kw: Any) -> TIterator[T]:
        return d.itervalues(**kw)  # type: ignore

    def iteritems(d: Mapping[T, V], **kw: Any) -> TIterator[Tuple[T, V]]:
        return d.iteritems(**kw)  # type: ignore

    def iterlists(d: Mapping[T, Sequence[V]], **kw: Any) -> TIterator[Tuple[T, Sequence[V]]]:
        return d.iterlists(**kw)  # type: ignore
    
    viewkeys: Callable[[Mapping[T, Any]], Any] = operator.methodcaller('viewkeys')
    viewvalues: Callable[[Mapping[Any, T]], Any] = operator.methodcaller('viewvalues')
    viewitems: Callable[[Mapping[T, V]], Any] = operator.methodcaller('viewitems')

_add_doc(iterkeys, 'Return an iterator over the keys of a dictionary.')
_add_doc(itervalues, 'Return an iterator over the values of a dictionary.')
_add_doc(iteritems, 'Return an iterator over the (key, value) pairs of a dictionary.')
_add_doc(iterlists, 'Return an iterator over the (key, [values]) pairs of a dictionary.')

if PY3:
    def b(s: Union[str, bytes]) -> bytes:
        if isinstance(s, bytes):
            return s
        return s.encode('latin-1')

    def u(s: str) -> str:
        return s
    
    unichr: Callable[[int], str] = chr
    
    import struct
    int2byte: Callable[[int], bytes] = struct.Struct('>B').pack
    del struct
    
    byte2int: Callable[[bytes], int] = operator.itemgetter(0)
    indexbytes: Callable[[bytes, int], int] = operator.getitem
    iterbytes: Callable[[bytes], TIterator[int]] = iter
    
    import io
    StringIO: Type[TextIO] = io.StringIO
    BytesIO: Type[BinaryIO] = io.BytesIO
    
    _assertCountEqual: str = 'assertCountEqual'
    if sys.version_info[1] <= 1:
        _assertRaisesRegex: str = 'assertRaisesRegexp'
        _assertRegex: str = 'assertRegexpMatches'
    else:
        _assertRaisesRegex = 'assertRaisesRegex'
        _assertRegex = 'assertRegex'
else:
    def b(s: str) -> str:
        return s

    def u(s: str) -> unicode:  # type: ignore
        return unicode(s.replace('\\\\', '\\\\\\\\'), 'unicode_escape')
    
    unichr = unichr  # type: ignore
    int2byte: Callable[[int], str] = chr  # type: ignore

    def byte2int(bs: str) -> int:  # type: ignore
        return ord(bs[0])

    def indexbytes(buf: str, i: int) -> int:  # type: ignore
        return ord(buf[i])
    
    iterbytes: Callable[[str], TIterator[int]] = functools.partial(itertools.imap, ord)  # type: ignore
    
    import StringIO  # type: ignore
    StringIO = BytesIO = StringIO.StringIO  # type: ignore
    
    _assertCountEqual: str = 'assertItemsEqual'
    _assertRaisesRegex: str = 'assertRaisesRegexp'
    _assertRegex: str = 'assertRegexpMatches'

_add_doc(b, 'Byte literal')
_add_doc(u, 'Text literal')

def assertCountEqual(self: Any, *args: Any, **kwargs: Any) -> None:
    return getattr(self, _assertCountEqual)(*args, **kwargs)

def assertRaisesRegex(self: Any, *args: Any, **kwargs: Any) -> None:
    return getattr(self, _assertRaisesRegex)(*args, **kwargs)

def assertRegex(self: Any, *args: Any, **kwargs: Any) -> None:
    return getattr(self, _assertRegex)(*args, **kwargs)

if PY3:
    exec_: Callable[..., None] = getattr(moves.builtins, 'exec')

    def reraise(tp: Type[BaseException], value: Optional[BaseException], tb: Optional[types.TracebackType] = None) -> None:
        if value is None:
            value = tp()
        if value.__traceback__ is not tb:
            raise value.with_traceback(tb)
        raise value
else:
    def exec_(_code_: str, _globs_: Optional[Dict[str, Any]] = None, _locs_: Optional[Dict[str, Any]] = None) -> None:
        """Execute code in a namespace."""
        if _globs_ is None:
            frame = sys._getframe(1)
            _globs_ = frame.f_globals
            if _locs_ is None:
                _locs_ = frame.f_locals
            del frame
        elif _locs_ is None:
            _locs_ = _globs_
        exec('exec _code_ in _globs_, _locs_')  # type: ignore
    
    exec_('def reraise(tp, value, tb=None):\n    raise tp, value, tb\n')  # type: ignore

if sys.version_info[:2] == (3, 2):
    exec_('def raise_from(value, from_value):\n    if from_value is None:\n        raise value\n    raise value from from_value\n')  # type: ignore
elif sys.version_info[:2] > (3, 2):
    exec_('def raise_from(value, from_value):\n    raise value from from_value\n')  # type: ignore
else:
    def raise_from(value: BaseException, from_value: Optional[BaseException]) -> None:
        raise value

print_: Callable[..., None] = getattr(moves.builtins, 'print', None)
if print_ is None:
    def print_(*args: Any, **kwargs: Any) -> None:
        """The new-style print function for Python 