from __future__ import absolute_import
import functools
import itertools
import operator
import sys
import types
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Type, Union

__author__ = 'Benjamin Peterson <benjamin@python.org>'
__version__ = '1.10.0'
PY2: bool = sys.version_info[0] == 2
PY3: bool = sys.version_info[0] == 3
PY34: bool = sys.version_info[0:2] >= (3, 4)
if PY3:
    string_types: tuple = (str,)
    integer_types: tuple = (int,)
    class_types: tuple = (type,)
    text_type = str
    binary_type = bytes
    MAXSIZE: int = sys.maxsize
else:
    string_types = (basestring,)
    integer_types = (int, long)  # type: ignore
    class_types = (type, types.ClassType)
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
        self.name: str = name

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
            self.mod: str = new
        else:
            self.mod = old

    def _resolve(self) -> types.ModuleType:
        return _import_module(self.mod)

    def __getattr__(self, attr: str) -> Any:
        _module = self._resolve()
        value: Any = getattr(_module, attr)
        setattr(self, attr, value)
        return value

class _LazyModule(types.ModuleType):
    _moved_attributes: List[Any] = []

    def __init__(self, name: str) -> None:
        super(_LazyModule, self).__init__(name)
        self.__doc__ = self.__class__.__doc__

    def __dir__(self) -> List[str]:
        attrs: List[str] = ['__doc__', '__name__']
        attrs += [attr.name for attr in self._moved_attributes]
        return attrs

class MovedAttribute(_LazyDescr):
    def __init__(self, name: str, old_mod: str, new_mod: Optional[str], old_attr: Optional[str] = None, new_attr: Optional[str] = None) -> None:
        super(MovedAttribute, self).__init__(name)
        if PY3:
            if new_mod is None:
                new_mod = name
            self.mod: str = new_mod
            if new_attr is None:
                if old_attr is None:
                    new_attr = name
                else:
                    new_attr = old_attr
            self.attr: str = new_attr
        else:
            self.mod = old_mod
            if old_attr is None:
                old_attr = name
            self.attr = old_attr

    def _resolve(self) -> Any:
        module: types.ModuleType = _import_module(self.mod)
        return getattr(module, self.attr)

class _SixMetaPathImporter(object):
    """
    A meta path importer to import six.moves and its submodules.

    This class implements a PEP302 finder and loader. It should be compatible
    with Python 2.5 and all existing versions of Python3
    """
    def __init__(self, six_module_name: str) -> None:
        self.name: str = six_module_name
        self.known_modules: Dict[str, Any] = {}

    def _add_module(self, mod: Any, *fullnames: str) -> None:
        for fullname in fullnames:
            self.known_modules[self.name + '.' + fullname] = mod

    def _get_module(self, fullname: str) -> Any:
        return self.known_modules[self.name + '.' + fullname]

    def find_module(self, fullname: str, path: Optional[Any] = None) -> Optional['_SixMetaPathImporter']:
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
        mod: Any = self.__get_module(fullname)
        if isinstance(mod, MovedModule):
            mod = mod._resolve()
        else:
            mod.__loader__ = self
        sys.modules[fullname] = mod
        return mod

    def is_package(self, fullname: str) -> bool:
        """
        Return true, if the named module is a package.

        We need this method to get correct spec objects with
        Python 3.4 (see PEP451)
        """
        return hasattr(self.__get_module(fullname), '__path__')

    def get_code(self, fullname: str) -> None:
        self.__get_module(fullname)
        return None
    get_source = get_code

_importer = _SixMetaPathImporter(__name__)

class _MovedItems(_LazyModule):
    """Lazy loading of moved objects"""
    __path__: List[Any] = []
_moved_attributes: List[Union[MovedAttribute, MovedModule]] = [
    MovedAttribute('cStringIO', 'cStringIO', 'io', 'StringIO'),
    MovedAttribute('filter', 'itertools', 'builtins', 'ifilter', 'filter'),
    MovedAttribute('filterfalse', 'itertools', 'itertools', 'ifilterfalse', 'filterfalse'),
    MovedAttribute('input', '__builtin__', 'builtins', 'raw_input', 'input'),
    MovedAttribute('intern', '__builtin__', 'sys'),
    MovedAttribute('map', 'itertools', 'builtins', 'imap', 'map'),
    MovedAttribute('getcwd', 'os', 'os', 'getcwdu', 'getcwd'),
    MovedAttribute('getcwdb', 'os', 'os', 'getcwd', 'getcwdb'),
    MovedAttribute('getstatusoutput', 'commands', 'subprocess'),
    MovedAttribute('getoutput', 'commands', 'subprocess'),
    MovedAttribute('range', '__builtin__', 'builtins', 'xrange', 'range'),
    MovedAttribute('reload_module', '__builtin__', 'importlib' if PY34 else 'imp', 'reload'),
    MovedAttribute('reduce', '__builtin__', 'functools'),
    MovedAttribute('shlex_quote', 'pipes', 'shlex', 'quote'),
    MovedAttribute('StringIO', 'StringIO', 'io'),
    MovedAttribute('UserDict', 'UserDict', 'collections'),
    MovedAttribute('UserList', 'UserList', 'collections'),
    MovedAttribute('UserString', 'UserString', 'collections'),
    MovedAttribute('xrange', '__builtin__', 'builtins', 'xrange', 'range'),
    MovedAttribute('zip', 'itertools', 'builtins', 'izip', 'zip'),
    MovedAttribute('zip_longest', 'itertools', 'itertools', 'izip_longest', 'zip_longest'),
    MovedModule('builtins', '__builtin__'),
    MovedModule('configparser', 'ConfigParser'),
    MovedModule('copyreg', 'copy_reg'),
    MovedModule('dbm_gnu', 'gdbm', 'dbm.gnu'),
    MovedModule('_dummy_thread', 'dummy_thread', '_dummy_thread'),
    MovedModule('http_cookiejar', 'cookielib', 'http.cookiejar'),
    MovedModule('http_cookies', 'Cookie', 'http.cookies'),
    MovedModule('html_entities', 'htmlentitydefs', 'html.entities'),
    MovedModule('html_parser', 'HTMLParser', 'html.parser'),
    MovedModule('http_client', 'httplib', 'http.client'),
    MovedModule('email_mime_base', 'email.MIMEBase', 'email.mime.base'),
    MovedModule('email_mime_image', 'email.MIMEImage', 'email.mime.image'),
    MovedModule('email_mime_multipart', 'email.MIMEMultipart', 'email.mime.multipart'),
    MovedModule('email_mime_nonmultipart', 'email.MIMENonMultipart', 'email.mime.nonmultipart'),
    MovedModule('email_mime_text', 'email.MIMEText', 'email.mime.text'),
    MovedModule('BaseHTTPServer', 'BaseHTTPServer', 'http.server'),
    MovedModule('CGIHTTPServer', 'CGIHTTPServer', 'http.server'),
    MovedModule('SimpleHTTPServer', 'SimpleHTTPServer', 'http.server'),
    MovedModule('cPickle', 'cPickle', 'pickle'),
    MovedModule('queue', 'Queue'),
    MovedModule('reprlib', 'repr'),
    MovedModule('socketserver', 'SocketServer'),
    MovedModule('_thread', 'thread', '_thread'),
    MovedModule('tkinter', 'Tkinter'),
    MovedModule('tkinter_dialog', 'Dialog', 'tkinter.dialog'),
    MovedModule('tkinter_filedialog', 'FileDialog', 'tkinter.filedialog'),
    MovedModule('tkinter_scrolledtext', 'ScrolledText', 'tkinter.scrolledtext'),
    MovedModule('tkinter_simpledialog', 'SimpleDialog', 'tkinter.simpledialog'),
    MovedModule('tkinter_tix', 'Tix', 'tkinter.tix'),
    MovedModule('tkinter_ttk', 'ttk', 'tkinter.ttk'),
    MovedModule('tkinter_constants', 'Tkconstants', 'tkinter.constants'),
    MovedModule('tkinter_dnd', 'Tkdnd', 'tkinter.dnd'),
    MovedModule('tkinter_colorchooser', 'tkColorChooser', 'tkinter.colorchooser'),
    MovedModule('tkinter_commondialog', 'tkCommonDialog', 'tkinter.commondialog'),
    MovedModule('tkinter_tkfiledialog', 'tkFileDialog', 'tkinter.filedialog'),
    MovedModule('tkinter_font', 'tkFont', 'tkinter.font'),
    MovedModule('tkinter_messagebox', 'tkMessageBox', 'tkinter.messagebox'),
    MovedModule('tkinter_tksimpledialog', 'tkSimpleDialog', 'tkinter.simpledialog'),
    MovedModule('urllib_parse', __name__ + '.moves.urllib_parse', 'urllib.parse'),
    MovedModule('urllib_error', __name__ + '.moves.urllib_error', 'urllib.error'),
    MovedModule('urllib', __name__ + '.moves.urllib', __name__ + '.moves.urllib'),
    MovedModule('urllib_robotparser', 'robotparser', 'urllib.robotparser'),
    MovedModule('xmlrpc_client', 'xmlrpclib', 'xmlrpc.client'),
    MovedModule('xmlrpc_server', 'SimpleXMLRPCServer', 'xmlrpc.server')
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

def add_move(move: _LazyDescr) -> None:
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
    advance_iterator: Callable[[Iterator[Any]], Any] = next
except NameError:
    def advance_iterator(it: Iterator[Any]) -> Any:
        return it.next()
next = advance_iterator
try:
    callable  # type: ignore
except NameError:
    def callable(obj: Any) -> bool:
        return any(('__call__' in klass.__dict__ for klass in type(obj).__mro__))

if PY3:
    def get_unbound_function(unbound: Any) -> Any:
        return unbound
    create_bound_method: Callable[..., Any] = types.MethodType

    def create_unbound_method(func: Callable[..., Any], cls: Type[Any]) -> Callable[..., Any]:
        return func
    IteratorBase = object
else:
    def get_unbound_function(unbound: Any) -> Any:
        return unbound.im_func

    def create_bound_method(func: Callable[..., Any], obj: Any) -> Any:
        return types.MethodType(func, obj, obj.__class__)

    def create_unbound_method(func: Callable[..., Any], cls: Type[Any]) -> Any:
        return types.MethodType(func, None, cls)

    class Iterator(object):
        def next(self) -> Any:
            return type(self).__next__(self)
    IteratorBase = Iterator
    callable = callable  # type: ignore

get_method_function: Callable[[Any], Any] = operator.attrgetter(_meth_func)
get_method_self: Callable[[Any], Any] = operator.attrgetter(_meth_self)
get_function_closure: Callable[[Any], Any] = operator.attrgetter(_func_closure)
get_function_code: Callable[[Any], Any] = operator.attrgetter(_func_code)
get_function_defaults: Callable[[Any], Any] = operator.attrgetter(_func_defaults)
get_function_globals: Callable[[Any], Any] = operator.attrgetter(_func_globals)

if PY3:
    def iterkeys(d: Mapping[Any, Any], **kw: Any) -> Iterator[Any]:
        return iter(d.keys(**kw))

    def itervalues(d: Mapping[Any, Any], **kw: Any) -> Iterator[Any]:
        return iter(d.values(**kw))

    def iteritems(d: Mapping[Any, Any], **kw: Any) -> Iterator[Any]:
        return iter(d.items(**kw))

    def iterlists(d: Mapping[Any, Any], **kw: Any) -> Iterator[Any]:
        return iter(d.lists(**kw))
    viewkeys = operator.methodcaller('keys')
    viewvalues = operator.methodcaller('values')
    viewitems = operator.methodcaller('items')
else:
    def iterkeys(d: Any, **kw: Any) -> Any:
        return d.iterkeys(**kw)

    def itervalues(d: Any, **kw: Any) -> Any:
        return d.itervalues(**kw)

    def iteritems(d: Any, **kw: Any) -> Any:
        return d.iteritems(**kw)

    def iterlists(d: Any, **kw: Any) -> Any:
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
    int2byte: Callable[[int], bytes] = struct.Struct('>B').pack
    del struct
    byte2int: Callable[[bytes], int] = operator.itemgetter(0)
    indexbytes: Callable[[Any, int], Any] = operator.getitem
    iterbytes: Callable[[Any], Iterator[Any]] = iter
    import io
    StringIO: Any = io.StringIO
    BytesIO: Any = io.BytesIO
    _assertCountEqual: str = 'assertCountEqual'
    if sys.version_info[1] <= 1:
        _assertRaisesRegex: str = 'assertRaisesRegexp'
        _assertRegex: str = 'assertRegexpMatches'
    else:
        _assertRaisesRegex: str = 'assertRaisesRegex'
        _assertRegex: str = 'assertRegex'
else:
    def b(s: str) -> str:
        return s

    def u(s: str) -> Any:
        return unicode(s.replace('\\\\', '\\\\\\\\'), 'unicode_escape')
    unichr = unichr  # type: ignore
    int2byte: Callable[[int], str] = chr

    def byte2int(bs: bytes) -> int:
        return ord(bs[0])

    def indexbytes(buf: Any, i: int) -> int:
        return ord(buf[i])
    iterbytes = functools.partial(itertools.imap, ord)  # type: ignore
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

    def reraise(tp: Exception, value: Exception, tb: Optional[Any] = None) -> None:
        try:
            if value is None:
                value = tp()  # type: ignore
            if value.__traceback__ is not tb:
                raise value.with_traceback(tb)
            raise value
        finally:
            value = None
            tb = None
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
    exec_('def reraise(tp, value, tb=None):\n    try:\n        raise tp, value, tb\n    finally:\n        tb = None\n')

if sys.version_info[:2] == (3, 2):
    exec_('def raise_from(value, from_value):\n    try:\n        if from_value is None:\n            raise value\n        raise value from from_value\n    finally:\n        value = None\n')
elif sys.version_info[:2] > (3, 2):
    exec_('def raise_from(value, from_value):\n    try:\n        raise value from from_value\n    finally:\n        value = None\n')
else:
    def raise_from(value: Exception, from_value: Any) -> None:
        raise value

print_ = getattr(moves.builtins, 'print', None)
if print_ is None:
    def print_(*args: Any, **kwargs: Any) -> None:
        """The new-style print function for Python 2.4 and 2.5."""
        fp: Any = kwargs.pop('file', sys.stdout)
        if fp is None:
            return

        def write(data: Any) -> None:
            if not isinstance(data, basestring):
                data = str(data)
            if isinstance(fp, file) and isinstance(data, unicode) and (fp.encoding is not None):  # type: ignore
                errors = getattr(fp, 'errors', None)
                if errors is None:
                    errors = 'strict'
                data = data.encode(fp.encoding, errors)
            fp.write(data)
        want_unicode: bool = False
        sep: Any = kwargs.pop('sep', None)
        if sep is not None:
            if isinstance(sep, unicode):  # type: ignore
                want_unicode = True
            elif not isinstance(sep, str):
                raise TypeError('sep must be None or a string')
        end: Any = kwargs.pop('end', None)
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
    def print_(*args: Any, **kwargs: Any) -> None:
        fp: Any = kwargs.get('file', sys.stdout)
        flush: bool = kwargs.pop('flush', False)
        _print(*args, **kwargs)
        if flush and fp is not None:
            fp.flush()
_add_doc(reraise, 'Reraise an exception.')

if sys.version_info[0:2] < (3, 4):
    def wraps(wrapped: Callable[..., Any],
              assigned: Any = functools.WRAPPER_ASSIGNMENTS,
              updated: Any = functools.WRAPPER_UPDATES) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def wrapper(f: Callable[..., Any]) -> Callable[..., Any]:
            f = functools.wraps(wrapped, assigned, updated)(f)
            f.__wrapped__ = wrapped
            return f
        return wrapper
else:
    wraps = functools.wraps

def with_metaclass(meta: Type[Any], *bases: Type[Any]) -> Type[Any]:
    """Create a base class with a metaclass."""
    class metaclass(meta):  # type: ignore
        def __new__(cls, name: str, this_bases: Any, d: Dict[str, Any]) -> Any:
            return meta(name, bases, d)
    return type.__new__(metaclass, 'temporary_class', (), {})

def add_metaclass(metaclass: Type[Any]) -> Callable[[Type[Any]], Type[Any]]:
    """Class decorator for creating a class with a metaclass."""
    def wrapper(cls: Type[Any]) -> Type[Any]:
        orig_vars: Dict[str, Any] = cls.__dict__.copy()
        slots: Optional[Union[str, List[str]]] = orig_vars.get('__slots__')
        if slots is not None:
            if isinstance(slots, str):
                slots = [slots]
            for slots_var in slots:
                orig_vars.pop(slots_var, None)
        orig_vars.pop('__dict__', None)
        orig_vars.pop('__weakref__', None)
        return metaclass(cls.__name__, cls.__bases__, orig_vars)
    return wrapper

def python_2_unicode_compatible(klass: Type[Any]) -> Type[Any]:
    """
    A decorator that defines __unicode__ and __str__ methods under Python 2.
    Under Python 3 it does nothing.

    To support Python 2 and 3 with a single code base, define a __str__ method
    returning text and apply this decorator to the class.
    """
    if PY2:
        if '__str__' not in klass.__dict__:
            raise ValueError("@python_2_unicode_compatible cannot be applied to %s because it doesn't define __str__()." % klass.__name__)
        klass.__unicode__ = klass.__str__
        klass.__str__ = lambda self: self.__unicode__().encode('utf-8')
    return klass

__path__ = []
__package__ = __name__
if globals().get('__spec__') is not None:
    __spec__.submodule_search_locations = []
if sys.meta_path:
    for i, importer in enumerate(sys.meta_path):
        if type(importer).__name__ == '_SixMetaPathImporter' and importer.name == __name__:
            del sys.meta_path[i]
            break
    del i, importer
sys.meta_path.append(_importer)