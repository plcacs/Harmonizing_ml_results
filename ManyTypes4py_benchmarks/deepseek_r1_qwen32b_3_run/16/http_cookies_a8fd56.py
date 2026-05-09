"""
requests.cookies
~~~~~~~~~~~~~~~~

Compatibility code to be able to use `cookielib.CookieJar` with requests.

requests.utils imports from here, so be careful with imports.
"""
import copy
import time
import calendar
from collections.abc import MutableMapping
from ._internal_utils import to_native_string
from ._basics import cookielib, urlparse, urlunparse, Morsel
try:
    import threading
except ImportError:
    import dummy_threading as threading
from typing import Any, Optional, Union, Dict, List, Tuple, Iterator

class MockRequest(object):
    """Wraps a `requests.Request` to mimic a `urllib2.Request`."""

    def __init__(self, request: object) -> None:
        self._r = request
        self._new_headers = {}
        self.type = urlparse(self._r.url).scheme

    def get_type(self) -> str:
        return self.type

    def get_host(self) -> str:
        return urlparse(self._r.url).netloc

    def get_origin_req_host(self) -> str:
        return self.get_host()

    def get_full_url(self) -> str:
        if not self._r.headers.get('Host'):
            return self._r.url
        host = to_native_string(self._r.headers['Host'], encoding='utf-8')
        parsed = urlparse(self._r.url)
        return urlunparse([parsed.scheme, host, parsed.path, parsed.params, parsed.query, parsed.fragment])

    def is_unverifiable(self) -> bool:
        return True

    def has_header(self, name: str) -> bool:
        return name in self._r.headers or name in self._new_headers

    def get_header(self, name: str, default: Optional[Any] = None) -> Optional[Any]:
        return self._r.headers.get(name, self._new_headers.get(name, default))

    def add_unredirected_header(self, name: str, value: str) -> None:
        self._new_headers[name] = value

    def get_new_headers(self) -> Dict[str, str]:
        return self._new_headers

    @property
    def unverifiable(self) -> bool:
        return self.is_unverifiable()

    @property
    def origin_req_host(self) -> str:
        return self.get_origin_req_host()

    @property
    def host(self) -> str:
        return self.get_host()

class Headers:
    def __init__(self, headers: object) -> None:
        self._headers = headers

    def get_all(self, key: str, default: Optional[Any]) -> List[Any]:
        return self._headers.getlist(key, default)

class MockResponse(object):
    """Wraps a `httplib.HTTPMessage` to mimic a `urllib.addinfourl`."""

    def __init__(self, headers: object) -> None:
        self._headers = headers

    def get_all(self, name: str, default: Optional[Any]) -> List[Any]:
        return self.getheaders(name)

    def info(self) -> Headers:
        return self.headers

    def getheaders(self, name: str) -> List[str]:
        return self.headers.getlist(name)

    @property
    def headers(self) -> Headers:
        return Headers(self._headers)

def extract_cookies_to_jar(jar: cookielib.CookieJar, request: object, response: object) -> None:
    """Extract the cookies from the response into a CookieJar."""
    if not hasattr(response, 'cookies'):
        return None
    req = MockRequest(request)
    res = MockResponse(response.headers)
    if not hasattr(response, 'cookies'):
        return None
    jar.extract_cookies(res, req)

def get_cookie_header(jar: cookielib.CookieJar, request: object) -> Optional[str]:
    """Produce an appropriate Cookie header string to be sent with `request`, or None."""
    r = MockRequest(request)
    jar.add_cookie_header(r)
    return r.get_new_headers().get('Cookie')

def remove_cookie_by_name(cookiejar: cookielib.CookieJar, name: str, domain: Optional[str] = None, path: Optional[str] = None) -> None:
    """Unsets a cookie by name, by default over all domains and paths."""
    clearables = []
    for cookie in cookiejar:
        if cookie.name != name:
            continue
        if domain is not None and domain != cookie.domain:
            continue
        if path is not None and path != cookie.path:
            continue
        clearables.append((cookie.domain, cookie.path, cookie.name))
    for domain, path, name in clearables:
        cookiejar.clear(domain, path, name)

class CookieConflictError(RuntimeError):
    """There are two cookies that meet the criteria specified in the cookie jar."""
    def __init__(self, message: str) -> None:
        super().__init__(message)

class RequestsCookieJar(cookielib.CookieJar, MutableMapping):
    """Compatibility class; is a cookielib.CookieJar, but exposes a dict interface."""

    def get(self, name: str, default: Optional[Any] = None, domain: Optional[str] = None, path: Optional[str] = None) -> Union[Any, None]:
        """Dict-like get() that also supports optional domain and path args."""
        try:
            return self._find_no_duplicates(name, domain, path)
        except KeyError:
            return default

    def set(self, name: str, value: Union[str, Morsel], **kwargs) -> Optional[cookielib.Cookie]:
        """Dict-like set() that also supports optional domain and path args."""
        if value is None:
            remove_cookie_by_name(self, name, domain=kwargs.get('domain'), path=kwargs.get('path'))
            return None
        if isinstance(value, Morsel):
            c = morsel_to_cookie(value)
        else:
            c = create_cookie(name, value, **kwargs)
        self.set_cookie(c)
        return c

    def iterkeys(self) -> Iterator[str]:
        """Dict-like iterkeys() that returns an iterator of names of cookies."""
        for cookie in iter(self):
            yield cookie.name

    def keys(self) -> List[str]:
        """Dict-like keys() that returns a list of names of cookies."""
        return list(self.iterkeys())

    def itervalues(self) -> Iterator[str]:
        """Dict-like itervalues() that returns an iterator of values of cookies."""
        for cookie in iter(self):
            yield cookie.value

    def values(self) -> List[str]:
        """Dict-like values() that returns a list of values of cookies."""
        return list(self.itervalues())

    def iteritems(self) -> Iterator[Tuple[str, str]]:
        """Dict-like iteritems() that returns an iterator of name-value tuples."""
        for cookie in iter(self):
            yield (cookie.name, cookie.value)

    def items(self) -> List[Tuple[str, str]]:
        """Dict-like items() that returns a list of name-value tuples."""
        return list(self.iteritems())

    def list_domains(self) -> List[str]:
        """Utility method to list all the domains in the jar."""
        domains = []
        for cookie in iter(self):
            if cookie.domain not in domains:
                domains.append(cookie.domain)
        return domains

    def list_paths(self) -> List[str]:
        """Utility method to list all the paths in the jar."""
        paths = []
        for cookie in iter(self):
            if cookie.path not in paths:
                paths.append(cookie.path)
        return paths

    def multiple_domains(self) -> bool:
        """Returns True if there are multiple domains in the jar."""
        domains = []
        for cookie in iter(self):
            if cookie.domain is not None and cookie.domain in domains:
                return True
            domains.append(cookie.domain)
        return False

    def get_dict(self, domain: Optional[str] = None, path: Optional[str] = None) -> Dict[str, str]:
        """Takes as an argument an optional domain and path and returns a plain old Python dict."""
        dictionary = {}
        for cookie in iter(self):
            if (domain is None or cookie.domain == domain) and (path is None or cookie.path == path):
                dictionary[cookie.name] = cookie.value
        return dictionary

    def __contains__(self, name: str) -> bool:
        try:
            return super(RequestsCookieJar, self).__contains__(name)
        except CookieConflictError:
            return True

    def __getitem__(self, name: str) -> str:
        """Dict-like __getitem__() for compatibility with client code."""
        return self._find_no_duplicates(name)

    def __setitem__(self, name: str, value: Union[str, Morsel]) -> None:
        """Dict-like __setitem__ for compatibility with client code."""
        self.set(name, value)

    def __delitem__(self, name: str) -> None:
        """Deletes a cookie given a name."""
        remove_cookie_by_name(self, name)

    def set_cookie(self, cookie: cookielib.Cookie, *args: Any, **kwargs: Any) -> None:
        if hasattr(cookie.value, 'startswith') and cookie.value.startswith('"') and cookie.value.endswith('"'):
            cookie.value = cookie.value.replace('\\"', '')
        return super(RequestsCookieJar, self).set_cookie(cookie, *args, **kwargs)

    def update(self, other: Union[cookielib.CookieJar, MutableMapping]) -> None:
        """Updates this jar with cookies from another CookieJar or dict-like"""
        if isinstance(other, cookielib.CookieJar):
            for cookie in other:
                self.set_cookie(copy.copy(cookie))
        else:
            super(RequestsCookieJar, self).update(other)

    def _find(self, name: str, domain: Optional[str] = None, path: Optional[str] = None) -> str:
        """Requests uses this method internally to get cookie values."""
        for cookie in iter(self):
            if cookie.name == name:
                if domain is None or cookie.domain == domain:
                    if path is None or cookie.path == path:
                        return cookie.value
        raise KeyError('name=%r, domain=%r, path=%r' % (name, domain, path))

    def _find_no_duplicates(self, name: str, domain: Optional[str] = None, path: Optional[str] = None) -> str:
        """Both ``__get_item__`` and ``get`` call this function."""
        toReturn = None
        for cookie in iter(self):
            if cookie.name == name:
                if domain is None or cookie.domain == domain:
                    if path is None or cookie.path == path:
                        if toReturn is not None:
                            raise CookieConflictError('There are multiple cookies with name, %r' % name)
                        toReturn = cookie.value
        if toReturn:
            return toReturn
        raise KeyError('name=%r, domain=%r, path=%r' % (name, domain, path))

    def __getstate__(self) -> Dict[str, Any]:
        """Unlike a normal CookieJar, this class is pickleable."""
        state = self.__dict__.copy()
        state.pop('_cookies_lock')
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Unlike a normal CookieJar, this class is pickleable."""
        self.__dict__.update(state)
        if '_cookies_lock' not in self.__dict__:
            self._cookies_lock = threading.RLock()

    def copy(self) -> 'RequestsCookieJar':
        """Return a copy of this RequestsCookieJar."""
        new_cj = RequestsCookieJar(self._policy)
        new_cj.update(self)
        return new_cj

    def get_policy(self) -> 'CookiePolicy':
        """Return the CookiePolicy instance used."""
        return self._policy

def _copy_cookie_jar(jar: Optional[cookielib.CookieJar]) -> Optional[cookielib.CookieJar]:
    if jar is None:
        return None
    if hasattr(jar, 'copy'):
        return jar.copy()
    new_jar = copy.copy(jar)
    new_jar.clear()
    for cookie in jar:
        new_jar.set_cookie(copy.copy(cookie))
    return new_jar

def create_cookie(name: str, value: str, **kwargs: Any) -> cookielib.Cookie:
    """Make a cookie from underspecified parameters."""
    result = {
        'version': 0,
        'name': name,
        'value': value,
        'port': None,
        'domain': '',
        'path': '/',
        'secure': False,
        'expires': None,
        'discard': True,
        'comment': None,
        'comment_url': None,
        'rest': {'HttpOnly': None},
        'rfc2109': False
    }
    badargs = set(kwargs) - set(result)
    if badargs:
        err = 'create_cookie() got unexpected keyword arguments: %s'
        raise TypeError(err % list(badargs))
    result.update(kwargs)
    result['port_specified'] = bool(result['port'])
    result['domain_specified'] = bool(result['domain'])
    result['domain_initial_dot'] = result['domain'].startswith('.')
    result['path_specified'] = bool(result['path'])
    return cookielib.Cookie(**result)

def morsel_to_cookie(morsel: Morsel) -> cookielib.Cookie:
    """Convert a Morsel object into a Cookie containing the one k/v pair."""
    expires = None
    if morsel['max-age']:
        try:
            expires = int(time.time() + int(morsel['max-age']))
        except ValueError:
            raise TypeError('max-age: %s must be integer' % morsel['max-age'])
    elif morsel['expires']:
        time_template = '%a, %d-%b-%Y %H:%M:%S GMT'
        expires = calendar.timegm(time.strptime(morsel['expires'], time_template))
    return create_cookie(
        comment=morsel['comment'],
        comment_url=bool(morsel['comment']),
        discard=False,
        domain=morsel['domain'],
        expires=expires,
        name=morsel.key,
        path=morsel['path'],
        port=None,
        rest={'HttpOnly': morsel['httponly']},
        rfc2109=False,
        secure=bool(morsel['secure']),
        value=morsel.value,
        version=morsel['version'] or 0
    )

def cookiejar_from_dict(cookie_dict: Optional[Dict[str, str]], cookiejar: Optional[cookielib.CookieJar] = None, overwrite: bool = True) -> cookielib.CookieJar:
    """Returns a CookieJar from a key/value dictionary."""
    if cookiejar is None:
        cookiejar = RequestsCookieJar()
    if cookie_dict is not None:
        names_from_jar = [cookie.name for cookie in cookiejar]
        for name in cookie_dict:
            if overwrite or name not in names_from_jar:
                cookiejar.set_cookie(create_cookie(name, cookie_dict[name]))
    return cookiejar

def merge_cookies(cookiejar: cookielib.CookieJar, cookies: Union[Dict[str, str], cookielib.CookieJar]) -> cookielib.CookieJar:
    """Add cookies to cookiejar and returns a merged CookieJar."""
    if not isinstance(cookiejar, cookielib.CookieJar):
        raise ValueError('You can only merge into CookieJar')
    if isinstance(cookies, dict):
        cookiejar = cookiejar_from_dict(cookies, cookiejar=cookiejar, overwrite=False)
    elif isinstance(cookies, cookielib.CookieJar):
        try:
            cookiejar.update(cookies)
        except AttributeError:
            for cookie_in_jar in cookies:
                cookiejar.set_cookie(cookie_in_jar)
    return cookiejar