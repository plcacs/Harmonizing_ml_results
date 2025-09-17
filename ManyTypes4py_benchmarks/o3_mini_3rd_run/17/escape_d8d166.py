import html
import json
import re
import urllib.parse
from typing import Union, Any, Optional, Dict, List, Callable
from tornado.util import unicode_type

def xhtml_escape(value: Any) -> str:
    """Escapes a string so it is valid within HTML or XML.

    Escapes the characters ``<``, ``>``, ``"``, ``'``, and ``&``.
    When used in attribute values the escaped strings must be enclosed
    in quotes.
    """
    return html.escape(to_unicode(value))

def xhtml_unescape(value: Any) -> str:
    """Un-escapes an XML-escaped string.

    Equivalent to `html.unescape` except that this function always returns
    type `str` while `html.unescape` returns `bytes` if its input is `bytes`.
    """
    return html.unescape(to_unicode(value))

def json_encode(value: Any) -> str:
    """JSON-encodes the given Python object.

    Ensures that the output will never contain the character sequence ``</``.
    """
    return json.dumps(value).replace('</', '<\\/')

def json_decode(value: Union[str, bytes]) -> Any:
    """Returns Python objects for the given JSON string.

    Supports both `str` and `bytes` inputs.
    """
    return json.loads(value)

def squeeze(value: str) -> str:
    """Replace all sequences of whitespace chars with a single space."""
    return re.sub('[\\x00-\\x20]+', ' ', value).strip()

def url_escape(value: str, plus: bool = True) -> str:
    """Returns a URL-encoded version of the given value.

    If ``plus`` is true, spaces are represented as ``+``; otherwise as ``%20``.
    """
    quote = urllib.parse.quote_plus if plus else urllib.parse.quote
    return quote(value)

@typing.overload
def url_unescape(value: Union[str, bytes], encoding: str, plus: bool = True) -> str: ...
@typing.overload
def url_unescape(value: Union[str, bytes], encoding: Optional[str] = 'utf-8', plus: bool = True) -> Union[str, bytes]: ...

def url_unescape(value: Union[str, bytes], encoding: Optional[str] = 'utf-8', plus: bool = True) -> Union[str, bytes]:
    """Decodes the given value from a URL.

    If encoding is None, returns bytes; otherwise returns a unicode string.
    """
    if encoding is None:
        if plus:
            value = to_basestring(value).replace('+', ' ')
        return urllib.parse.unquote_to_bytes(value)
    else:
        unquote = urllib.parse.unquote_plus if plus else urllib.parse.unquote
        return unquote(to_basestring(value), encoding=encoding)

def parse_qs_bytes(qs: Union[str, bytes], keep_blank_values: bool = False, strict_parsing: bool = False) -> Dict[str, List[bytes]]:
    """Parses a query string like urlparse.parse_qs, but takes bytes and returns the values as byte strings."""
    if isinstance(qs, bytes):
        qs = qs.decode('latin1')
    result = urllib.parse.parse_qs(qs, keep_blank_values, strict_parsing, encoding='latin1', errors='strict')
    encoded: Dict[str, List[bytes]] = {}
    for k, v in result.items():
        encoded[k] = [i.encode('latin1') for i in v]
    return encoded

_UTF8_TYPES = (bytes, type(None))

@typing.overload
def utf8(value: Optional[Union[bytes, str]]) -> Optional[bytes]: ...
@typing.overload
def utf8(value: Optional[Union[bytes, str]]) -> Optional[bytes]: ...
@typing.overload
def utf8(value: Optional[Union[bytes, str]]) -> Optional[bytes]: ...

def utf8(value: Optional[Union[bytes, str]]) -> Optional[bytes]:
    """Converts a string argument to a byte string using utf-8 encoding.

    If the argument is already a byte string or None, it is returned unchanged.
    """
    if isinstance(value, _UTF8_TYPES):
        return value
    if not isinstance(value, unicode_type):
        raise TypeError('Expected bytes, unicode, or None; got %r' % type(value))
    return value.encode('utf-8')

_TO_UNICODE_TYPES = (unicode_type, type(None))

@typing.overload
def to_unicode(value: Optional[Union[str, bytes]]) -> Optional[str]: ...
@typing.overload
def to_unicode(value: Optional[Union[str, bytes]]) -> Optional[str]: ...
@typing.overload
def to_unicode(value: Optional[Union[str, bytes]]) -> Optional[str]: ...

def to_unicode(value: Optional[Union[str, bytes]]) -> Optional[str]:
    """Converts a string argument to a unicode string using utf-8 decoding.

    If the argument is already a unicode string or None, it is returned unchanged.
    """
    if isinstance(value, _TO_UNICODE_TYPES):
        return value
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

_URL_RE = re.compile(to_unicode('\\b((?:([\\w-]+):(/{1,3})|www[.])(?:(?:(?:[^\\s&()]|&amp;|&quot;)*(?:[^!"#$%&\'()*+,.:;<=>?@\\[\\]^`{|}~\\s]))|(?:\\((?:[^\\s&()]|&amp;|&quot;)*\\)))+)'))

def linkify(text: str, 
            shorten: bool = False, 
            extra_params: Union[str, Callable[[str], str]] = '', 
            require_protocol: bool = False, 
            permitted_protocols: Union[List[str], set] = ['http', 'https']) -> str:
    """Converts plain text into HTML with links.

    Parameters:
      shorten: Long urls will be shortened for display.
      extra_params: Extra text for the link tag or a callable returning extra text.
      require_protocol: Only linkify urls which include a protocol.
      permitted_protocols: List or set of permitted protocols.
    """
    if extra_params and (not callable(extra_params)):
        extra_params = ' ' + extra_params.strip()

    def make_link(m: re.Match) -> str:
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
            if len(url) > int(max_len * 1.5):
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