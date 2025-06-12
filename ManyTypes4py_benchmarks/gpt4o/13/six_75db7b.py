from __future__ import absolute_import
import functools
import itertools
import operator
import sys
import types
from typing import Any, Callable, Dict, Iterator as TypingIterator, List, Optional, Tuple, Type, Union

__author__ = 'Benjamin Peterson <benjamin@python.org>'
__version__ = '1.10.0'
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
    string_types = (basestring,)
    integer_types = (int, long)
    class_types = (type, types.ClassType)
    text_type = unicode
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

def _add_doc(func: Callable, doc: str) -> None:
    func.__doc__ = doc

def _import_module(name: str) -> types.ModuleType:
    __import__(name)
    return sys.modules[name]

class _LazyDescr(object):
    def __init__(self, name: str) -> None:
        self.name = name

    def __get__(self, obj: Any, tp: Type) -> Any:
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
    _moved_attributes: List[Union[MovedModule, 'MovedAttribute']] = []

class MovedAttribute(_LazyDescr):
    def __init__(self, name: str, old_mod: str, new_mod: str, old_attr: Optional[str] = None, new_attr: Optional[str] = None) -> None:
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
        self.known_modules: Dict[str, Union[MovedModule, _LazyModule]] = {}

    def _add_module(self, mod: Union[MovedModule, _LazyModule], *fullnames: str) -> None:
        for fullname in fullnames:
            self.known_modules[self.name + '.' + fullname] = mod

    def _get_module(self, fullname: str) -> Union[MovedModule, _LazyModule]:
        return self.known_modules[self.name + '.' + fullname]

    def find_module(self, fullname: str, path: Optional[str] = None) -> Optional['_SixMetaPathImporter']:
        if fullname in self.known_modules:
            return self
        return None

    def __get_module(self, fullname: str) -> Union[MovedModule, _LazyModule]:
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
    __path__: List[str] = []

_moved_attributes: List[Union[MovedModule, MovedAttribute]] = [
    MovedAttribute('cStringIO', 'cStringIO', 'io', 'StringIO'),
    MovedAttribute('filter', 'itertools', 'builtins', 'ifilter', 'filter'),
    MovedAttribute('filterfalse', 'itertools', 'itertools', 'ifilterfalse', 'filterfalse'),
    MovedAttribute('input', '__builtin__', 'builtins', 'raw_input', 'input'),
    MovedAttribute('intern', '__builtin__', 'sys'),
    MovedAttribute('map', 'itertools', 'builtins', 'imap', 'map'),
    MovedAttribute('getcwd', 'os', 'os', 'getcwdu', 'getcwd'),
    MovedAttribute('getcwdb', 'os', 'os', 'getcwd', 'getcwdb'),
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
    MovedModule('email_mime_multipart', 'email.MIMEMultipart', 'email.mime.multipart'),
    MovedModule('email_mime_nonmultipart', 'email.MIMENonMultipart', 'email.mime.nonmultipart'),
    MovedModule('email_mime_text', 'email.MIMEText', 'email.mime.text'),
    MovedModule('email_mime_base', 'email.MIMEBase', 'email.mime.base'),
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
moves = _MovedItems(__name__ + '.moves')
_importer._add_module(moves, 'moves')

class Module_six_moves_urllib_parse(_LazyModule):
    __path__: List[str] = []

_urllib_parse_moved_attributes: List[MovedAttribute] = [
    MovedAttribute('ParseResult', 'urlparse', 'urllib.parse'),
    MovedAttribute('SplitResult', 'urlparse', 'urllib.parse'),
    MovedAttribute('parse_qs', 'urlparse', 'urllib.parse'),
    MovedAttribute('parse_qsl', 'urlparse', 'urllib.parse'),
    MovedAttribute('urldefrag', 'urlparse', 'urllib.parse'),
    MovedAttribute('urljoin', 'urlparse', 'urllib.parse'),
    MovedAttribute('urlparse', 'urlparse', 'urllib.parse'),
    MovedAttribute('urlsplit', 'urlparse', 'urllib.parse'),
    MovedAttribute('urlunparse', 'urlparse', 'urllib.parse'),
    MovedAttribute('urlunsplit', 'urlparse', 'urllib.parse'),
    MovedAttribute('quote', 'urllib', 'urllib.parse'),
    MovedAttribute('quote_plus', 'urllib', 'urllib.parse'),
    MovedAttribute('unquote', 'urllib', 'urllib.parse'),
    MovedAttribute('unquote_plus', 'urllib', 'urllib.parse'),
    MovedAttribute('urlencode', 'urllib', 'urllib.parse'),
    MovedAttribute('splitquery', 'urllib', 'urllib.parse'),
    MovedAttribute('splittag', 'urllib', 'urllib.parse'),
    MovedAttribute('splituser', 'urllib', 'urllib.parse'),
    MovedAttribute('uses_fragment', 'urlparse', 'urllib.parse'),
    MovedAttribute('uses_netloc', 'urlparse', 'urllib.parse'),
    MovedAttribute('uses_params', 'urlparse', 'urllib.parse'),
    MovedAttribute('uses_query', 'urlparse', 'urllib.parse'),
    MovedAttribute('uses_relative', 'urlparse', 'urllib.parse')
]

for attr in _urllib_parse_moved_attributes:
    setattr(Module_six_moves_urllib_parse, attr.name, attr)

del attr
Module_six_moves_urllib_parse._moved_attributes = _urllib_parse_moved_attributes
_importer._add_module(Module_six_moves_urllib_parse(__name__ + '.moves.urllib_parse'), 'moves.urllib_parse', 'moves.urllib.parse')

class Module_six_moves_urllib_error(_LazyModule):
    __path__: List[str] = []

_urllib_error_moved_attributes: List[MovedAttribute] = [
    MovedAttribute('URLError', 'urllib2', 'urllib.error'),
    MovedAttribute('HTTPError', 'urllib2', 'urllib.error'),
    MovedAttribute('ContentTooShortError', 'urllib', 'urllib.error')
]

for attr in _urllib_error_moved_attributes:
    setattr(Module_six_moves_urllib_error, attr.name, attr)

del attr
Module_six_moves_urllib_error._moved_attributes = _urllib_error_moved_attributes
_importer._add_module(Module_six_moves_urllib_error(__name__ + '.moves.urllib.error'), 'moves.urllib_error', 'moves.urllib.error')

class Module_six_moves_urllib_request(_LazyModule):
    __path__: List[str] = []

_urllib_request_moved_attributes: List[MovedAttribute] = [
    MovedAttribute('urlopen', 'urllib2', 'urllib.request'),
    MovedAttribute('install_opener', 'urllib2', 'urllib.request'),
    MovedAttribute('build_opener', 'urllib2', 'urllib.request'),
    MovedAttribute('pathname2url', 'urllib', 'urllib.request'),
    MovedAttribute('url2pathname', 'urllib', 'urllib.request'),
    MovedAttribute('getproxies', 'urllib', 'urllib.request'),
    MovedAttribute('Request', 'urllib2', 'urllib.request'),
    MovedAttribute('OpenerDirector', 'urllib2', 'urllib.request'),
    MovedAttribute('HTTPDefaultErrorHandler', 'urllib2', 'urllib.request'),
    MovedAttribute('HTTPRedirectHandler', 'urllib2', 'urllib.request'),
    MovedAttribute('HTTPCookieProcessor', 'urllib2', 'urllib.request'),
    MovedAttribute('ProxyHandler', 'urllib2', 'urllib.request'),
    MovedAttribute('BaseHandler', 'urllib2', 'urllib.request'),
    MovedAttribute('HTTPPasswordMgr', 'urllib2', 'urllib.request'),
    MovedAttribute('HTTPPasswordMgrWithDefaultRealm', 'urllib2', 'urllib.request'),
    MovedAttribute('AbstractBasicAuthHandler', 'urllib2', 'urllib.request'),
    MovedAttribute('HTTPBasicAuthHandler', 'urllib2', 'urllib.request'),
    MovedAttribute('ProxyBasicAuthHandler', 'urllib2', 'urllib.request'),
    MovedAttribute('AbstractDigestAuthHandler', 'urllib2', 'urllib.request'),
    MovedAttribute('HTTPDigestAuthHandler', 'urllib2', 'urllib.request'),
    MovedAttribute('ProxyDigestAuthHandler', 'urllib2', 'urllib.request'),
    MovedAttribute('HTTPHandler', 'urllib2', 'urllib.request'),
    MovedAttribute('HTTPSHandler', 'urllib2', 'urllib.request'),
    MovedAttribute('FileHandler', 'urllib2', 'urllib.request'),
    MovedAttribute('FTPHandler', 'urllib2', 'urllib.request'),
    MovedAttribute('CacheFTPHandler', 'urllib2', 'urllib.request'),
    MovedAttribute('UnknownHandler', 'urllib2', 'urllib.request'),
    MovedAttribute('HTTPErrorProcessor', 'urllib2', 'urllib.request'),
    MovedAttribute('urlretrieve', 'urllib', 'urllib.request'),
    MovedAttribute('urlcleanup', 'urllib', 'urllib.request'),
    MovedAttribute('URLopener', 'urllib', 'urllib.request'),
    MovedAttribute('FancyURLopener', 'urllib', 'urllib.request'),
    MovedAttribute('proxy_bypass', 'urllib', 'urllib.request')
]

for attr in _urllib_request_moved_attributes:
    setattr(Module_six_moves_urllib_request, attr.name, attr)

del attr
Module_six_moves_urllib_request._moved_attributes = _urllib_request_moved_attributes
_importer._add_module(Module_six_moves_urllib_request(__name__ + '.moves.urllib.request'), 'moves.urllib_request', 'moves.urllib.request')

class Module_six_moves_urllib_response(_LazyModule):
    __path__: List[str] = []

_urllib_response_moved_attributes: List[MovedAttribute] = [
    MovedAttribute('addbase', 'urllib', 'urllib.response'),
    MovedAttribute('addclosehook', 'urllib', 'urllib.response'),
    MovedAttribute('addinfo', 'urllib', 'urllib.response'),
    MovedAttribute('addinfourl', 'urllib', 'urllib.response')
]

for attr in _urllib_response_moved_attributes:
    setattr(Module_six_moves_urllib_response, attr.name, attr)

del attr
Module_six_moves_urllib_response._moved_attributes = _urllib_response_moved_attributes
_importer._add_module(Module_six_moves_urllib_response(__name__ + '.moves.urllib.response'), 'moves.urllib_response', 'moves.urllib.response')

class Module_six_moves_urllib_robotparser(_LazyModule):
    __path__: List[str] = []

_urllib_robotparser_moved_attributes: List[MovedAttribute] = [
    MovedAttribute('RobotFileParser', 'robotparser', 'urllib.robotparser')
]

for attr in _urllib_robotparser_moved_attributes:
    setattr(Module_six_moves_urllib_robotparser, attr.name, attr)

del attr
Module_six_moves_urllib_robotparser._moved_attributes = _urllib_robotparser_moved_attributes
_importer._add_module(Module_six_moves_urllib_robotparser(__name__ + '.moves.urllib.robotparser'), 'moves.urllib_robotparser', 'moves.urllib.robotparser')

class Module_six_moves_urllib(types.ModuleType):
    __path__: List[str] = []
    parse: Module_six_moves_urllib_parse = _importer._get_module('moves.urllib_parse')
    error: Module_six_moves_urllib_error = _importer._get_module('moves.urllib_error')
    request: Module_six_moves_urllib_request = _importer._get_module('moves.urllib_request')
    response: Module_six_moves_urllib_response = _importer._get_module('moves.urllib_response')
    robotparser: Module_six_moves_urllib_robotparser = _importer._get_module('moves.urllib_robotparser')

    def __dir__(self) -> List[str]:
        return ['parse', 'error', 'request', 'response', 'robotparser']

_importer._add_module(Module_six_moves_urllib(__name__ + '.moves.urllib'), 'moves.urllib')

def add_move(move: Union[MovedModule, MovedAttribute]) -> None:
    setattr(_MovedItems, move.name, move)

def remove_move(name: str) -> None:
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
    advance_iterator = next
except NameError:
    def advance_iterator(it: TypingIterator) -> Any:
        return it.next()

next = advance_iterator

try:
    callable = callable
except NameError:
    def callable(obj: Any) -> bool:
        return any(('__call__' in klass.__dict__ for klass in type(obj).__mro__))

if PY3:
    def get_unbound_function(unbound: Callable) -> Callable:
        return unbound

    create_bound_method = types.MethodType

    def create_unbound_method(func: Callable, cls: Type) -> Callable:
        return func

    Iterator = object
else:
    def get_unbound_function(unbound: Callable) -> Callable:
        return unbound.im_func

    def create_bound_method(func: Callable, obj: Any) -> types.MethodType:
        return types.MethodType(func, obj, obj.__class__)

    def create_unbound_method(func: Callable, cls: Type) -> types.MethodType:
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
    def iterkeys(d: Dict, **kw: Any) -> TypingIterator:
        return iter(d.keys(**kw))

    def itervalues(d: Dict, **kw: Any) -> TypingIterator:
        return iter(d.values(**kw))

    def iteritems(d: Dict, **kw: Any) -> TypingIterator:
        return iter(d.items(**kw))

    def iterlists(d: Dict, **kw: Any) -> TypingIterator:
        return iter(d.lists(**kw))

    viewkeys = operator.methodcaller('keys')
    viewvalues = operator.methodcaller('values')
    viewitems = operator.methodcaller('items')
else:
    def iterkeys(d: Dict, **kw: Any) -> TypingIterator:
        return d.iterkeys(**kw)

    def itervalues(d: Dict, **kw: Any) -> TypingIterator:
        return d.itervalues(**kw)

    def iteritems(d: Dict, **kw: Any) -> TypingIterator:
        return d.iteritems(**kw)

    def iterlists(d: Dict, **kw: Any) -> TypingIterator:
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

    def u(s: str) -> str:
        return unicode(s.replace('\\\\', '\\\\\\\\'), 'unicode_escape')

    unichr = unichr
    int2byte = chr

    def byte2int(bs: bytes) -> int:
        return ord(bs[0])

    def indexbytes(buf: bytes, i: int) -> int:
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
    def exec_(_code_: str, _globs_: Optional[Dict[str, Any]] = None, _locs_: Optional[Dict[str, Any]] = None) -> None:
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

print_: Optional[Callable[..., None]] = getattr(moves.builtins, 'print', None)

if print_ is None:
    def print_(*args: Any, **kwargs: Any) -> None:
        fp = kwargs.pop('file', sys.stdout)
        if fp is None:
            return

        def write(data: Any) -> None:
            if not isinstance(data, basestring):
                data = str(data)
            if isinstance(fp, file) and isinstance(data, unicode) and (fp.encoding is not None):
                errors = getattr(fp, 'errors', None)
                if errors is None:
                    errors = 'strict'
                data = data.encode(fp.encoding, errors)
            fp.write(data)

        want_unicode = False
        sep = kwargs.pop('sep', None)
        if sep is not None:
            if isinstance(sep, unicode):
                want_unicode = True
            elif not isinstance(sep, str):
                raise TypeError('sep must be None or a string')
        end = kwargs.pop('end', None)
        if end is not None:
            if isinstance(end, unicode):
                want_unicode = True
            elif not isinstance(end, str):
                raise TypeError('end must be None or a string')
        if kwargs:
            raise TypeError('invalid keyword arguments to print()')
        if not want_unicode:
            for arg in args:
                if isinstance(arg, unicode):
                    want_unicode = True
                    break
        if want_unicode:
            newline = unicode('\n')
            space = unicode(' ')
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
        fp = kwargs.get('file', sys.stdout)
        flush = kwargs.pop('flush', False)
        _print(*args, **kwargs)
        if flush and fp is not None:
            fp.flush()

_add_doc(reraise, 'Reraise an exception.')

if sys.version_info[0:2] < (3, 4):
    def wraps(wrapped: Callable, assigned: Tuple[str, ...] = functools.WRAPPER_ASSIGNMENTS, updated: Tuple[str, ...] = functools.WRAPPER_UPDATES) -> Callable:
        def wrapper(f: Callable) -> Callable:
            f = functools.wraps(wrapped, assigned, updated)(f)
            f.__wrapped__ = wrapped
            return f
        return wrapper
else:
    wraps = functools.wraps

def with_metaclass(meta: Type, *bases: Type) -> Type:
    class metaclass(meta):
        def __new__(cls: Type, name: str, this_bases: Tuple[Type, ...], d: Dict[str, Any]) -> Type:
            return meta(name, bases, d)
    return type.__new__(metaclass, 'temporary_class', (), {})

def add_metaclass(metaclass: Type) -> Callable[[Type], Type]:
    def wrapper(cls: Type) -> Type:
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

def python_2_unicode_compatible(klass: Type) -> Type:
    if PY2:
        if '__str__' not in klass.__dict__:
            raise ValueError("@python_2_unicode_compatible cannot be applied to %s because it doesn't define __str__()." % klass.__name__)
        klass.__unicode__ = klass.__str__
        klass.__str__ = lambda self: self.__unicode__().encode('utf-8')
    return klass

__path__: List[str] = []
__package__: str = __name__

if globals().get('__spec__') is not None:
    __spec__.submodule_search_locations = []

if sys.meta_path:
    for i, importer in enumerate(sys.meta_path):
        if type(importer).__name__ == '_SixMetaPathImporter' and importer.name == __name__:
            del sys.meta_path[i]
            break
    del i, importer

sys.meta_path.append(_importer)
