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

    def __get__(self, obj: Any, tp: type) -> Any:
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

    def find_module(self, fullname: str, path: Optional[List[str]] = None) -> Optional['_SixMetaPathImporter']:
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
moves = _MovedItems(__name__ + '.moves')
_importer._add_module(moves, 'moves')

class Module_six_moves_urllib_parse(_LazyModule):
    """Lazy loading of moved objects in six.moves.urllib_parse"""

_urllib_parse_moved_attributes = [
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
    MovedAttribute('unquote_to_bytes', 'urllib', 'urllib.parse', 'unquote', 'unquote_to_bytes'),
    MovedAttribute('urlencode', 'urllib', 'urllib.parse'),
    MovedAttribute('splitquery', 'urllib', 'urllib.parse'),
    MovedAttribute('splittag', 'urllib', 'urllib.parse'),
    MovedAttribute('splituser', 'urllib', 'urllib.parse'),
    MovedAttribute('splitvalue', 'urllib', 'urllib.parse'),
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
_importer._add_module(
    Module_six_moves_urllib_parse(__name__ + '.moves.urllib_parse'),
    'moves.urllib_parse', 'moves.urllib.parse'
)

class Module_six_moves_urllib_error(_LazyModule):
    """Lazy loading of moved objects in six.moves.urllib_error"""

_urllib_error_moved_attributes = [
    MovedAttribute('URLError', 'urllib2', 'urllib.error'),
    MovedAttribute('HTTPError', 'urllib2', 'urllib.error'),
    MovedAttribute('ContentTooShortError', 'urllib', 'urllib.error')
]

for attr in _urllib_error_moved_attributes:
    setattr(Module_six_moves_urllib_error, attr.name, attr)
del attr

Module_six_moves_urllib_error._moved_attributes = _urllib_error_moved_attributes
_importer._add_module(
    Module_six_moves_urllib_error(__name__ + '.moves.urllib.error'),
    'moves.urllib_error', 'moves.urllib.error'
)

class Module_six_moves_urllib_request(_LazyModule):
    """Lazy loading of moved objects in six.moves.urllib_request"""

_urllib_request_moved_attributes = [
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
    MovedAttribute('ProxyBasicAuthHandler', 'urllib2', 'urllib.request