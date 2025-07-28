#!/usr/bin/env python3
"""
Escaping/unescaping methods for HTML, JSON, URLs, and others.

Also includes a few other miscellaneous string manipulation functions that
have crept in over time.

Many functions in this module have near-equivalents in the standard library
(the differences mainly relate to handling of bytes and unicode strings,
and were more relevant in Python 2). In new code, the standard library
functions are encouraged instead of this module where applicable. See the
docstrings on each function for details.
"""
import html
import json
import re
import urllib.parse
from tornado.util import unicode_type
import typing
from typing import Union, Any, Optional, Dict, List, Callable

def xhtml_escape(value: Union[str, bytes]) -> str:
    """Escapes a string so it is valid within HTML or XML.

    Escapes the characters ``<``, ``>``, ``"``, ``'``, and ``&``.
    When used in attribute values the escaped strings must be enclosed
    in quotes.

    Equivalent to `html.escape` except that this function always returns
    type `str` while `html.escape` returns `bytes` if its input is `bytes`.
    """
    return html.escape(to_unicode(value))

def xhtml_unescape(value: Union[str, bytes]) -> str:
    """Un-escapes an XML-escaped string.

    Equivalent to `html.unescape` except that this function always returns
    type `str` while `html.unescape` returns `bytes` if its input is `bytes`.
    """
    return html.unescape(to_unicode(value))

def json_encode(value: Any) -> str:
    """JSON-encodes the given Python object.

    Equivalent to `json.dumps` with the additional guarantee that the output
    will never contain the character sequence ``</`` which can be problematic
    when JSON is embedded in an HTML ``<script>`` tag.
    """
    return json.dumps(value).replace('</', '<\\/')

def json_decode(value: Union[str, bytes]) -> Any:
    """Returns Python objects for the given JSON string.

    Supports both `str` and `bytes` inputs. Equivalent to `json.loads`.
    """
    return json.loads(value)

def squeeze(value: str) -> str:
    """Replace all sequences of whitespace chars with a single space."""
    return re.sub('[\\x00-\\x20]+', ' ', value).strip()

def url_escape(value: str, plus: bool = True) -> str:
    """Returns a URL-encoded version of the given value.

    Equivalent to either `urllib.parse.quote_plus` or `urllib.parse.quote` depending on the ``plus``
    argument.
    """
    quote = urllib.parse.quote_plus if plus else urllib.parse.quote
    return quote(value)

@typing.overload
def url_unescape(value: Union[str, bytes], encoding: str, plus: bool = True) -> str:
    ...

@typing.overload
def url_unescape(value: Union[str, bytes], encoding: None, plus: bool = True) -> bytes:
    ...

def url_unescape(value: Union[str, bytes], encoding: Optional[str] = 'utf-8', plus: bool = True) -> Union[str, bytes]:
    """Decodes the given value from a URL.

    The argument may be either a byte or unicode string.

    If encoding is None, the result will be a byte string and this function is equivalent to
    `urllib.parse.unquote_to_bytes` if ``plus=False``. Otherwise, the result is a unicode string in
    the specified encoding and this function is equivalent to either `urllib.parse.unquote_plus` or
    `urllib.parse.unquote` except that this function also accepts `bytes` as input.
    """
    if encoding is None:
        if plus:
            value = to_basestring(value)
            value = value.replace('+', ' ')
        return urllib.parse.unquote_to_bytes(value)
    else:
        unquote = urllib.parse.unquote_plus if plus else urllib.parse.unquote
        return unquote(to_basestring(value), encoding=encoding)

def parse_qs_bytes(qs: Union[str, bytes],
                   keep_blank_values: bool = False,
                   strict_parsing: bool = False) -> Dict[str, List[bytes]]:
    """Parses a query string like urlparse.parse_qs,
    but takes bytes and returns the values as byte strings.

    Keys still become type str (interpreted as latin1 in python3!)
    because it's too painful to keep them as byte strings in
    python3 and in practice they're nearly always ascii anyway.
    """
    if isinstance(qs, bytes):
        qs = qs.decode('latin1')
    result: Dict[str, List[str]] = urllib.parse.parse_qs(
        qs,
        keep_blank_values=keep_blank_values,
        strict_parsing=strict_parsing,
        encoding='latin1',
        errors='strict'
    )
    encoded: Dict[str, List[bytes]] = {}
    for k, v in result.items():
        encoded[k] = [i.encode('latin1') for i in v]
    return encoded

_UTF8_TYPES = (bytes, type(None))

@typing.overload
def utf8(value: Union[str, bytes, None]) -> Union[bytes, None]:
    ...

@typing.overload
def utf8(value: Union[str]) -> bytes:
    ...

@typing.overload
def utf8(value: Union[bytes]) -> bytes:
    ...

def utf8(value: Union[str, bytes, None]) -> Union[bytes, None]:
    """Converts a string argument to a byte string.

    If the argument is already a byte string or None, it is returned unchanged.
    Otherwise it must be a unicode string and is encoded as utf8.
    """
    if isinstance(value, _UTF8_TYPES):
        return value  # type: ignore
    if not isinstance(value, unicode_type):
        raise TypeError('Expected bytes, unicode, or None; got %r' % type(value))
    return value.encode('utf-8')

_TO_UNICODE_TYPES = (unicode_type, type(None))

@typing.overload
def to_unicode(value: Union[str, bytes, None]) -> Union[str, None]:
    ...

@typing.overload
def to_unicode(value: str) -> str:
    ...

@typing.overload
def to_unicode(value: bytes) -> str:
    ...

def to_unicode(value: Union[str, bytes, None]) -> Union[str, None]:
    """Converts a string argument to a unicode string.

    If the argument is already a unicode string or None, it is returned
    unchanged. Otherwise it must be a byte string and is decoded as utf8.
    """
    if isinstance(value, _TO_UNICODE_TYPES):
        return value  # type: ignore
    if not isinstance(value, bytes):
        raise TypeError('Expected bytes, unicode, or None; got %r' % type(value))
    return value.decode('utf-8')

_unicode = to_unicode
native_str = to_unicode
to_basestring = to_unicode

def recursive_unicode(obj: Any) -> Any:
    """Walks a simple data structure, converting byte strings to unicode.

    Supports lists, tuples, and dictionaries.
    """
    if isinstance(obj, dict):
        return {recursive_unicode(k): recursive_unicode(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_unicode(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_unicode(i) for i in obj)
    elif isinstance(obj, bytes):
        return to_unicode(obj)
    else:
        return obj

_URL_RE: re.Pattern[str] = re.compile(to_unicode(r'\b((?:([\w-]+):(/{1,3})|www[.])(?:(?:(?:[^\s&()]|&amp;|&quot;)*(?:[^!"#$%&\'()*+,.:;<=>?@\[\\\]^`{|}~\s]))|(?:\((?:[^\s&()]|&amp;|&quot;)*\)))+)'))

def linkify(text: str,
            shorten: bool = False,
            extra_params: Union[str, Callable[[str], str]] = '',
            require_protocol: bool = False,
            permitted_protocols: List[str] = ['http', 'https']) -> str:
    """Converts plain text into HTML with links.

    For example: ``linkify("Hello http://tornadoweb.org!")`` would return
    ``Hello <a href="http://tornadoweb.org">http://tornadoweb.org</a>!``
    """
    if extra_params and (not callable(extra_params)):
        extra_params = ' ' + extra_params.strip()

    def make_link(m: re.Match[str]) -> str:
        url: str = m.group(1)
        proto: Optional[str] = m.group(2)
        if require_protocol and (not proto):
            return url
        if proto and proto not in permitted_protocols:
            return url
        href: str = m.group(1)
        if not proto:
            href = 'http://' + href
        if callable(extra_params):
            params: str = ' ' + extra_params(href).strip()
        else:
            params = extra_params  # type: ignore
        max_len: int = 30
        if shorten and len(url) > max_len:
            before_clip: str = url
            if proto:
                proto_len: int = len(proto) + 1 + len(m.group(3) or '')
            else:
                proto_len = 0
            parts = url[proto_len:].split('/')
            if len(parts) > 1:
                url = url[:proto_len] + parts[0] + '/' + parts[1][:8].split('?')[0].split('.')[0]
            if len(url) > int(max_len * 1.5):
                url = url[:max_len]
            if url != before_clip:
                amp: int = url.rfind('&')
                if amp > max_len - 5:
                    url = url[:amp]
                url += '...'
                if len(url) >= len(before_clip):
                    url = before_clip
                else:
                    params += ' title="%s"' % href
        return f'<a href="{href}"{params}>{url}</a>'
    text = _unicode(xhtml_escape(text))
    return _URL_RE.sub(make_link, text)