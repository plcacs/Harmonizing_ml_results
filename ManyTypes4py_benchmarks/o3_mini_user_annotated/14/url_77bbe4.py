from __future__ import absolute_import
from collections import namedtuple
from typing import Optional, Tuple, Iterable, Type

from ..exceptions import LocationParseError

url_attrs = ["scheme", "auth", "host", "port", "path", "query", "fragment"]
# We only want to normalize urls with an HTTP(S) scheme.
# urllib3 infers URLs without a scheme (None) to be http.
NORMALIZABLE_SCHEMES = ("http", "https", None)


class Url(namedtuple("Url", url_attrs)):
    """
    Datastructure for representing an HTTP URL. Used as a return value for
    :func:`parse_url`. Both the scheme and host are normalized as they are
    both case-insensitive according to RFC 3986.
    """

    __slots__ = ()

    def __new__(
        cls: Type["Url"],
        scheme: Optional[str] = None,
        auth: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        path: Optional[str] = None,
        query: Optional[str] = None,
        fragment: Optional[str] = None,
    ) -> "Url":
        if path and not path.startswith("/"):
            path = "/" + path
        if scheme:
            scheme = scheme.lower()
        if host and scheme in NORMALIZABLE_SCHEMES:
            host = host.lower()
        return super(Url, cls).__new__(
            cls, scheme, auth, host, port, path, query, fragment
        )

    @property
    def hostname(self) -> Optional[str]:
        """For backwards-compatibility with urlparse. We're nice like that."""
        return self.host

    @property
    def request_uri(self) -> str:
        """Absolute path including the query string."""
        uri: str = self.path or "/"
        if self.query is not None:
            uri += "?" + self.query
        return uri

    @property
    def netloc(self) -> Optional[str]:
        """Network location including host and port"""
        if self.host is None:
            return None
        if self.port:
            return "%s:%d" % (self.host, self.port)
        return self.host

    @property
    def url(self) -> str:
        """
        Convert self into a url

        This function should more or less round-trip with :func:`.parse_url`. The
        returned url may not be exactly the same as the url inputted to
        :func:`.parse_url`, but it should be equivalent by the RFC (e.g., urls
        with a blank port will have : removed).

        Example: ::

            >>> U = parse_url('http://google.com/mail/')
            >>> U.url
            'http://google.com/mail/'
            >>> Url('http', 'username:password', 'host.com', 80,
            ... '/path', 'query', 'fragment').url
            'http://username:password@host.com:80/path?query#fragment'
        """
        scheme, auth, host, port, path, query, fragment = self
        url_str = ""
        # We use "is not None" we want things to happen with empty strings (or 0 port)
        if scheme is not None:
            url_str += scheme + "://"
        if auth is not None:
            url_str += auth + "@"
        if host is not None:
            url_str += host
        if port is not None:
            url_str += ":" + str(port)
        if path is not None:
            url_str += path
        if query is not None:
            url_str += "?" + query
        if fragment is not None:
            url_str += "#" + fragment
        return url_str

    def __str__(self) -> str:
        return self.url


def split_first(s: str, delims: Iterable[str]) -> Tuple[str, str, Optional[str]]:
    """
    Given a string and an iterable of delimiters, split on the first found
    delimiter. Return two split parts and the matched delimiter.

    If not found, then the first part is the full input string.

    Example::

        >>> split_first('foo/bar?baz', '?/=')
        ('foo', 'bar?baz', '/')
        >>> split_first('foo/bar?baz', '123')
        ('foo/bar?baz', '', None)

    Scales linearly with number of delims. Not ideal for large number of delims.
    """
    min_idx: Optional[int] = None
    min_delim: Optional[str] = None
    for d in delims:
        idx = s.find(d)
        if idx < 0:
            continue

        if min_idx is None or idx < min_idx:
            min_idx = idx
            min_delim = d
    if min_idx is None or min_idx < 0:
        return s, "", None

    return s[:min_idx], s[min_idx + 1 :], min_delim


def parse_url(url: str) -> Url:
    """
    Given a url, return a parsed :class:`.Url` namedtuple. Best-effort is
    performed to parse incomplete urls. Fields not provided will be None.

    Partly backwards-compatible with :mod:`urlparse`.

    Example::

        >>> parse_url('http://google.com/mail/')
        Url(scheme='http', host='google.com', port=None, path='/mail/', ...)
        >>> parse_url('google.com:80')
        Url(scheme=None, host='google.com', port=80, path=None, ...)
        >>> parse_url('/foo?bar')
        Url(scheme=None, host=None, port=None, path='/foo', query='bar', ...)
    """
    if not url:
        # Empty
        return Url()

    scheme: Optional[str] = None
    auth: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    path: Optional[str] = None
    fragment: Optional[str] = None
    query: Optional[str] = None

    # Scheme
    if "://" in url:
        scheme, url = url.split("://", 1)
    # Find the earliest Authority Terminator
    # (http://tools.ietf.org/html/rfc3986#section-3.2)
    url, path_, delim = split_first(url, ["/", "?", "#"])
    if delim:
        # Reassemble the path
        path = delim + path_
    # Auth
    if "@" in url:
        # Last '@' denotes end of auth part
        auth, url = url.rsplit("@", 1)
    # IPv6
    if url and url[0] == "[":
        host, url = url.split("]", 1)
        host += "]"
    # Port
    if ":" in url:
        _host, port_str = url.split(":", 1)
        if not host:
            host = _host
        if port_str:
            # If given, ports must be integers. No whitespace, no plus or
            # minus prefixes, no non-integer digits such as ^2 (superscript).
            if not port_str.isdigit():
                raise LocationParseError(url)
            try:
                port = int(port_str)
            except ValueError:
                raise LocationParseError(url)
        else:
            # Blank ports are cool, too. (rfc3986#section-3.2.3)
            port = None
    elif not host and url:
        host = url
    if not path:
        return Url(scheme, auth, host, port, path, query, fragment)

    # Fragment
    if "#" in path:
        path, fragment = path.split("#", 1)
    # Query
    if "?" in path:
        path, query = path.split("?", 1)
    return Url(scheme, auth, host, port, path, query, fragment)


def get_host(url: str) -> Tuple[str, Optional[str], Optional[int]]:
    """
    Deprecated. Use :func:`parse_url` instead.
    """
    p = parse_url(url)
    return p.scheme or "http", p.hostname, p.port