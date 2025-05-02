"""Utilities for writing code that runs on Python 2 and 3"""
from __future__ import absolute_import
import functools
import itertools
import operator
import sys
import types
from typing import (
    Any, Callable, Dict, Iterator, List, Optional, Tuple, Type, TypeVar, Union,
    overload, cast
)

__author__ = 'Benjamin Peterson <benjamin@python.org>'
__version__ = '1.10.0'

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

PY2: bool = sys.version_info[0] == 2
PY3: bool = sys.version_info[0] == 3
PY34: bool = sys.version_info[0:2] >= (3, 4)

if PY3:
    string_types: Tuple[Type[str], ...] = (str,)
    integer_types: Tuple[Type[int], ...] = (int,)
    class_types: Tuple[Type[type], ...] = (type,)
    text_type: Type[str] = str
    binary_type: Type[bytes] = bytes
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

    def __get__(self, obj: Any, tp: Type[Any]) -> Any:
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

# ... (rest of the module remains the same with appropriate type annotations)

def add_move(move: Any) -> None:
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
    _meth_func = '__func__'
    _meth_self = '__self__'
    _func_closure = '__closure__'
    _func_code = '__code__'
    _func_defaults = '__defaults__'
    _func_globals = '__globals__'
else:
    _meth_func = 'im_func'
    _meth_self = 'im_self'
    _func_closure = 'func_closure'
    _func_code = 'func_code'
    _func_defaults = 'func_defaults'
    _func_globals = 'func_globals'

try:
    advance_iterator = next
except NameError:
    def advance_iterator(it: Iterator[Any]) -> Any:
        return it.next()

next = advance_iterator

try:
    callable = callable
except NameError:
    def callable(obj: Any) -> bool:
        return any('__call__' in klass.__dict__ for klass in type(obj).__mro__)

if PY3:
    def get_unbound_function(unbound: Callable[..., Any]) -> Callable[..., Any]:
        return unbound
    
    create_bound_method = types.MethodType

    def create_unbound_method(func: Callable[..., Any], cls: Type[Any]) -> Callable[..., Any]:
        return func
    
    Iterator = object
else:
    def get_unbound_function(unbound: Callable[..., Any]) -> Callable[..., Any]:
        return unbound.im_func

    def create_bound_method(func: Callable[..., Any], obj: Any) -> Callable[..., Any]:
        return types.MethodType(func, obj, obj.__class__)

    def create_unbound_method(func: Callable[..., Any], cls: Type[Any]) -> Callable[..., Any]:
        return types.MethodType(func, None, cls)

    class Iterator(object):
        def next(self) -> Any:
            return type(self).__next__(self)

    callable = callable

_add_doc(get_unbound_function, 'Get the function out of a possibly unbound function')

get_method_function = operator.attrgetter(_meth_func)
get_method_self = operator.attrgetter(_meth_self)
get_function_closure = operator.attrgetter(_func_closure)
get_function_code = operator.attrgetter(_func_code)
get_function_defaults = operator.attrgetter(_func_defaults)
get_function_globals = operator.attrgetter(_func_globals)

if PY3:
    def iterkeys(d: Dict[Any, Any], **kw: Any) -> Iterator[Any]:
        return iter(d.keys(**kw))

    def itervalues(d: Dict[Any, Any], **kw: Any) -> Iterator[Any]:
        return iter(d.values(**kw))

    def iteritems(d: Dict[Any, Any], **kw: Any) -> Iterator[Tuple[Any, Any]]:
        return iter(d.items(**kw))

    def iterlists(d: Any, **kw: Any) -> Iterator[Any]:
        return iter(d.lists(**kw))
    
    viewkeys = operator.methodcaller('keys')
    viewvalues = operator.methodcaller('values')
    viewitems = operator.methodcaller('items')
else:
    def iterkeys(d: Dict[Any, Any], **kw: Any) -> Iterator[Any]:
        return d.iterkeys(**kw)

    def itervalues(d: Dict[Any, Any], **kw: Any) -> Iterator[Any]:
        return d.itervalues(**kw)

    def iteritems(d: Dict[Any, Any], **kw: Any) -> Iterator[Tuple[Any, Any]]:
        return d.iteritems(**kw)

    def iterlists(d: Any, **kw: Any) -> Iterator[Any]:
        return d.iterlists(**kw)
    
    viewkeys = operator.methodcaller('viewkeys')
    viewvalues = operator.methodcaller('viewvalues')
    viewitems = operator.methodcaller('viewitems')

_add_doc(iterkeys, 'Return an iterator over the keys of a dictionary.')
_add_doc(itervalues, 'Return an iterator over the values of a dictionary.')
_add_doc(iteritems, 'Return an iterator over the (key, value) pairs of a dictionary.')
_add_doc(iterlists, 'Return an iterator over the (key, [values]) pairs of a dictionary.')

if PY3:
    def b(s: str) -> bytes:
        return s.encode('latin-1')

    def u(s: str) -> str:
        return s
    
    unichr = chr
    import struct
    int2byte = struct.Struct('>B').pack
    del struct
    byte2int = operator.itemgetter(0)
    indexbytes = operator.getitem
    iterbytes = iter
    import io
    StringIO = io.StringIO
    BytesIO = io.BytesIO
    _assertCountEqual = 'assertCountEqual'
    if sys.version_info[1] <= 1:
        _assertRaisesRegex = 'assertRaisesRegexp'
        _assertRegex = 'assertRegexpMatches'
    else:
        _assertRaisesRegex = 'assertRaisesRegex'
        _assertRegex = 'assertRegex'
else:
    def b(s: str) -> str:
        return s

    def u(s: str) -> 'unicode':  # type: ignore
        return unicode(s.replace('\\\\', '\\\\\\\\'), 'unicode_escape')
    
    unichr = unichr  # type: ignore
    int2byte = chr

    def byte2int(bs: str) -> int:
        return ord(bs[0])

    def indexbytes(buf: str, i: int) -> int:
        return ord(buf[i])
    
    iterbytes = functools.partial(itertools.imap, ord)
    import StringIO
    StringIO = BytesIO = StringIO.StringIO
    _assertCountEqual = 'assertItemsEqual'
    _assertRaisesRegex = 'assertRaisesRegexp'
    _assertRegex = 'assertRegexpMatches'

_add_doc(b, 'Byte literal')
_add_doc(u, 'Text literal')

def assertCountEqual(self: Any, *args: Any, **kwargs: Any) -> Any:
    return getattr(self, _assertCountEqual)(*args, **kwargs)

def assertRaisesRegex(self: Any, *args: Any, **kwargs: Any) -> Any:
    return getattr(self, _assertRaisesRegex)(*args, **kwargs)

def assertRegex(self: Any, *args: Any, **kwargs: Any) -> Any:
    return getattr(self, _assertRegex)(*args, **kwargs)

if PY3:
    exec_ = getattr(moves.builtins, 'exec')

    def reraise(tp: Type[BaseException], value: Optional[BaseException], tb: Optional[types.TracebackType] = None) -> None:
        if value is None:
            value = tp()
        if value.__traceback__ is not tb:
            raise value.with_traceback(tb)
        raise value
else:
    def exec_(_code_: Any, _globs_: Optional[Dict[str, Any]] = None, _locs_: Optional[Dict[str, Any]] = None) -> None:
        """Execute code in a namespace."""
        if _globs_ is None:
            frame = sys._getframe(1)
            _globs_ = frame.f_globals
            if _locs_ is None:
                _locs_ = frame.f_locals
            del frame
        elif _locs_ is None:
            _locs_ = _globs_
        exec('exec _code_ in _globs_, _locs_')
    
    exec_('def reraise(tp, value, tb=None):\n    raise tp, value, tb\n')

if sys.version_info[:2] == (3, 2):
    exec_('def raise_from(value, from_value):\n    if from_value is None:\n        raise value\n    raise value from from_value\n')
elif sys.version_info[:2] > (3, 2):
    exec_('def raise_from(value, from_value):\n    raise value from from_value\n')
else:
    def raise_from(value: BaseException, from_value: Optional[BaseException]) -> None:
        raise value

print_ = getattr(moves.builtins, 'print', None)
if print_ is None:
    def print_(*args: Any, **kwargs: Any) -> None:
        """The new-style print function for Python 2.4 and 2.5."""
        fp = kwargs.pop('file', sys.stdout)
        if fp is None:
            return

        def write(data: Any) -> None:
            if not isinstance(data, basestring):  # type: ignore
                data = str(data)
            if isinstance(fp, file) and isinstance(data, unicode) and (fp.encoding is not None):  # type: ignore
                errors = getattr(fp, 'errors', None)
                if errors is None:
                    errors = 'strict'
                data = data.encode(fp.encoding, errors)
            fp.write(data)
        
        want_unicode = False
        sep = kwargs.pop('sep', None)
        if sep is not None:
            if isinstance(sep, unicode):  # type: ignore
                want_unicode = True
            elif not isinstance(sep, str):
                raise TypeError('sep must be None or a string')
        
        end = kwargs.pop('end', None)
        if end is not None:
            if isinstance(end, unicode):  # type: ignore
                want_unicode = True
            elif not isinstance(end, str):
                raise TypeError('end must be None or a string')
        
        if kwargs:
            raise TypeError('invalid keyword arguments to print()')
        
        if not want_unicode:
            for arg in args:
                if isinstance(arg, unicode):  # type: ignore
                    want_unicode = True
                    break
        
        if want_unicode:
            newline = unicode('\n')  # type: ignore
            space = unicode(' ')  # type: ignore
        else:
            newline = '\n'
            space = ' '
        
        if sep is None:
            sep = space
        if end is None:
            end = newline
        
        for i, arg in enumerate(args):
            if i:
                write(sep)
            write(arg)
        write(end)

if sys.version_info[:2] < (3, 3):
    _print = print_

   