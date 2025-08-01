"""Escaping/unescaping methods for HTML, JSON, URLs, and others.

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
from typing import Union, Any, Optional, Dict, List, Callable, TypeVar, Tuple, Set, Pattern, Match

_T = TypeVar('_T')
_StrOrBytes = Union[str, bytes]

def xhtml_escape(value: _StrOrBytes) -> str:
    """Escapes a string so it is valid within HTML or XML.

    Escapes the characters ``<``, ``>``, ``"``, ``'``, and ``&``.
    When used in attribute values the escaped strings must be enclosed
    in quotes.

    Equivalent to `html.escape` except that this function always returns
    type `str` while `html.escape` returns `bytes` if its input is `bytes`.

    .. versionchanged:: 3.2

       Added the single quote to the list of escaped characters.

    .. versionchanged:: 6.4

       Now simply wraps `html.escape`. This is equivalent to the old behavior
       except that single quotes are now escaped as ``&#x27;`` instead of
       ``&#39;`` and performance may be different.
    """
    return html.escape(to_unicode(value))

def xhtml_unescape(value: _StrOrBytes) -> str:
    """Un-escapes an XML-escaped string.

    Equivalent to `html.unescape` except that this function always returns
    type `str` while `html.unescape` returns `bytes` if its input is `bytes`.

    .. versionchanged:: 6.4

       Now simply wraps `html.unescape`. This changes behavior for some inputs
       as required by the HTML 5 specification
       https://html.spec.whatwg.org/multipage/parsing.html#numeric-character-reference-end-state

       Some invalid inputs such as surrogates now raise an error, and numeric
       references to certain ISO-8859-1 characters are now handled correctly.
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

    Supports both `str` and `bytes` inputs. Equvalent to `json.loads`.
    """
    return json.loads(value)

def squeeze(value: str) -> str:
    """Replace all sequences of whitespace chars with a single space."""
    return re.sub('[\\x00-\\x20]+', ' ', value).strip()

def url_escape(value: _StrOrBytes, plus: bool = True) -> str:
    """Returns a URL-encoded version of the given value.

    Equivalent to either `urllib.parse.quote_plus` or `urllib.parse.quote` depending on the ``plus``
    argument.

    If ``plus`` is true (the default), spaces will be represented as ``+`` and slashes will be
    represented as ``%2F``.  This is appropriate for query strings. If ``plus`` is false, spaces
    will be represented as ``%20`` and slashes are left as-is. This is appropriate for the path
    component of a URL. Note that the default of ``plus=True`` is effectively the
    reverse of Python's urllib module.

    .. versionadded:: 3.1
        The ``plus`` argument
    """
    quote = urllib.parse.quote_plus if plus else urllib.parse.quote
    return quote(utf8(value))

@typing.overload
def url_unescape(value: bytes, encoding: None = ..., plus: bool = ...) -> bytes: ...

@typing.overload
def url_unescape(value: _StrOrBytes, encoding: str = ..., plus: bool = ...) -> str: ...

def url_unescape(value: _StrOrBytes, encoding: Optional[str] = 'utf-8', plus: bool = True) -> Union[str, bytes]:
    """Decodes the given value from a URL.

    The argument may be either a byte or unicode string.

    If encoding is None, the result will be a byte string and this function is equivalent to
    `urllib.parse.unquote_to_bytes` if ``plus=False``.  Otherwise, the result is a unicode string in
    the specified encoding and this function is equivalent to either `urllib.parse.unquote_plus` or
    `urllib.parse.unquote` except that this function also accepts `bytes` as input.

    If ``plus`` is true (the default), plus signs will be interpreted as spaces (literal plus signs
    must be represented as "%2B").  This is appropriate for query strings and form-encoded values
    but not for the path component of a URL.  Note that this default is the reverse of Python's
    urllib module.

    .. versionadded:: 3.1
       The ``plus`` argument
    """
    if encoding is None:
        if plus:
            value = to_basestring(value).replace('+', ' ')
        return urllib.parse.unquote_to_bytes(value)
    else:
        unquote = urllib.parse.unquote_plus if plus else urllib.parse.unquote
        return unquote(to_basestring(value), encoding=encoding)

def parse_qs_bytes(qs: Union[str, bytes], keep_blank_values: bool = False, strict_parsing: bool = False) -> Dict[str, List[bytes]]:
    """Parses a query string like urlparse.parse_qs,
    but takes bytes and returns the values as byte strings.

    Keys still become type str (interpreted as latin1 in python3!)
    because it's too painful to keep them as byte strings in
    python3 and in practice they're nearly always ascii anyway.
    """
    if isinstance(qs, bytes):
        qs = qs.decode('latin1')
    result = urllib.parse.parse_qs(qs, keep_blank_values, strict_parsing, encoding='latin1', errors='strict')
    encoded = {}
    for k, v in result.items():
        encoded[k] = [i.encode('latin1') for i in v]
    return encoded

_UTF8_TYPES = (bytes, type(None))

@typing.overload
def utf8(value: None) -> None: ...

@typing.overload
def utf8(value: bytes) -> bytes: ...

@typing.overload
def utf8(value: str) -> bytes: ...

def utf8(value: Union[str, bytes, None]) -> Union[bytes, None]:
    """Converts a string argument to a byte string.

    If the argument is already a byte string or None, it is returned unchanged.
    Otherwise it must be a unicode string and is encoded as utf8.
    """
    if isinstance(value, _UTF8_TYPES):
        return value
    if not isinstance(value, unicode_type):
        raise TypeError('Expected bytes, unicode, or None; got %r' % type(value))
    return value.encode('utf-8')

_TO_UNICODE_TYPES = (unicode_type, type(None))

@typing.overload
def to_unicode(value: None) -> None: ...

@typing.overload
def to_unicode(value: str) -> str: ...

@typing.overload
def to_unicode(value: bytes) -> str: ...

def to_unicode(value: Union[str, bytes, None]) -> Union[str, None]:
    """Converts a string argument to a unicode string.

    If the argument is already a unicode string or None, it is returned
    unchanged.  Otherwise it must be a byte string and is decoded as utf8.
    """
    if isinstance(value, _TO_UNICODE_TYPES):
        return value
    if not isinstance(value, bytes):
        raise TypeError('Expected bytes, unicode, or None; got %r' % type(value))
    return value.decode('utf-8')

_unicode = to_unicode
native_str = to_unicode
to_basestring = to_unicode

def recursive_unicode(obj: _T) -> _T:
    """Walks a simple data structure, converting byte strings to unicode.

    Supports lists, tuples, and dictionaries.
    """
    if isinstance(obj, dict):
        return {recursive_unicode(k): recursive_unicode(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return list((recursive_unicode(i) for i in obj))
    elif isinstance(obj, tuple):
        return tuple((recursive_unicode(i) for i in obj))
    elif isinstance(obj, bytes):
        return to_unicode(obj)
    else:
        return obj

_URL_RE: Pattern[str] = re.compile(to_unicode('\\b((?:([\\w-]+):(/{1,3})|www[.])(?:(?:(?:[^\\s&()]|&amp;|&quot;)*(?:[^!"#$%&\'()*+,.:;<=>?@\\[\\]^`{|}~\\s]))|(?:\\((?:[^\\s&()]|&amp;|&quot;)*\\)))+)'))

_ExtraParamsType = Union[str, Callable[[str], str]]

def linkify(
    text: str,
    shorten: bool = False,
    extra_params: _ExtraParamsType = '',
    require_protocol: bool = False,
    permitted_protocols: Union[List[str], Set[str]] = ['http', 'https']
) -> str:
    """Converts plain text into HTML with links.

    For example: ``linkify("Hello http://tornadoweb.org!")`` would return
    ``Hello <a href="http://tornadoweb.org">http://tornadoweb.org</a>!``

    Parameters:

    * ``shorten``: Long urls will be shortened for display.

    * ``extra_params``: Extra text to include in the link tag, or a callable
      taking the link as an argument and returning the extra text
      e.g. ``linkify(text, extra_params='rel="nofollow" class="external"')``,
      or::

          def extra_params_cb(url):
              if url.startswith("http://example.com"):
                  return 'class="internal"'
              else:
                  return 'class="external" rel="nofollow"'
          linkify(text, extra_params=extra_params_cb)

    * ``require_protocol``: Only linkify urls which include a protocol. If
      this is False, urls such as www.facebook.com will also be linkified.

    * ``permitted_protocols``: List (or set) of protocols which should be
      linkified, e.g. ``linkify(text, permitted_protocols=["http", "ftp",
      "mailto"])``. It is very unsafe to include protocols such as
      ``javascript``.
    """
    if extra_params and (not callable(extra_params)):
        extra_params = ' ' + extra_params.strip()

    def make_link(m: Match[str]) -> str:
        url = m.group(1)
        proto = m.group(2)
        if require_protocol and (not proto):
            return url
        if proto and proto not in permitted_protocols:
            return url
        href = m.group(1)
        if not proto:
            href = 'http://' + href
        if callable(extra_params):
            params = ' ' + extra_params(href).strip()
        else:
            params = extra_params
        max_len = 30
        if shorten and len(url) > max_len:
            before_clip = url
            if proto:
                proto_len = len(proto) + 1 + len(m.group(3) or '')
            else:
                proto_len = 0
            parts = url[proto_len:].split('/')
            if len(parts) > 1:
                url = url[:proto_len] + parts[0] + '/' + parts[1][:8].split('?')[0].split('.')[0]
            if len(url) > max_len * 1.5:
                url = url[:max_len]
            if url != before_clip:
                amp = url.rfind('&')
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
