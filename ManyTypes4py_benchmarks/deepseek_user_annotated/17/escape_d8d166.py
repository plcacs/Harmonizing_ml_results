#
# Copyright 2009 Facebook
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

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
from typing import Union, Any, Optional, Dict, List, Callable, Match, Tuple, TypeVar

_T = TypeVar('_T')

def xhtml_escape(value: Union[str, bytes]) -> str:
    return html.escape(to_unicode(value))

def xhtml_unescape(value: Union[str, bytes]) -> str:
    return html.unescape(to_unicode(value))

def json_encode(value: Any) -> str:
    return json.dumps(value).replace("</", "<\\/")

def json_decode(value: Union[str, bytes]) -> Any:
    return json.loads(value)

def squeeze(value: str) -> str:
    return re.sub(r"[\x00-\x20]+", " ", value).strip()

def url_escape(value: Union[str, bytes], plus: bool = True) -> str:
    quote = urllib.parse.quote_plus if plus else urllib.parse.quote
    return quote(value)

@typing.overload
def url_unescape(value: Union[str, bytes], encoding: None, plus: bool = True) -> bytes:
    ...

@typing.overload
def url_unescape(
    value: Union[str, bytes], encoding: str = "utf-8", plus: bool = True
) -> str:
    ...

def url_unescape(
    value: Union[str, bytes], encoding: Optional[str] = "utf-8", plus: bool = True
) -> Union[str, bytes]:
    if encoding is None:
        if plus:
            value = to_basestring(value).replace("+", " ")
        return urllib.parse.unquote_to_bytes(value)
    else:
        unquote = urllib.parse.unquote_plus if plus else urllib.parse.unquote
        return unquote(to_basestring(value), encoding=encoding)

def parse_qs_bytes(
    qs: Union[str, bytes], keep_blank_values: bool = False, strict_parsing: bool = False
) -> Dict[str, List[bytes]]:
    if isinstance(qs, bytes):
        qs = qs.decode("latin1")
    result = urllib.parse.parse_qs(
        qs, keep_blank_values, strict_parsing, encoding="latin1", errors="strict"
    )
    encoded = {}
    for k, v in result.items():
        encoded[k] = [i.encode("latin1") for i in v]
    return encoded

_UTF8_TYPES = (bytes, type(None))

@typing.overload
def utf8(value: bytes) -> bytes:
    ...

@typing.overload
def utf8(value: str) -> bytes:
    ...

@typing.overload
def utf8(value: None) -> None:
    ...

def utf8(value: Union[None, str, bytes]) -> Optional[bytes]:
    if isinstance(value, _UTF8_TYPES):
        return value
    if not isinstance(value, unicode_type):
        raise TypeError("Expected bytes, unicode, or None; got %r" % type(value))
    return value.encode("utf-8")

_TO_UNICODE_TYPES = (unicode_type, type(None))

@typing.overload
def to_unicode(value: str) -> str:
    ...

@typing.overload
def to_unicode(value: bytes) -> str:
    ...

@typing.overload
def to_unicode(value: None) -> None:
    ...

def to_unicode(value: Union[None, str, bytes]) -> Optional[str]:
    if isinstance(value, _TO_UNICODE_TYPES):
        return value
    if not isinstance(value, bytes):
        raise TypeError("Expected bytes, unicode, or None; got %r" % type(value))
    return value.decode("utf-8")

_unicode = to_unicode
native_str = to_unicode
to_basestring = to_unicode

def recursive_unicode(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {recursive_unicode(k): recursive_unicode(v) for (k, v) in obj.items()}
    elif isinstance(obj, list):
        return list(recursive_unicode(i) for i in obj)
    elif isinstance(obj, tuple):
        return tuple(recursive_unicode(i) for i in obj)
    elif isinstance(obj, bytes):
        return to_unicode(obj)
    else:
        return obj

_URL_RE = re.compile(
    to_unicode(
        r"""\b((?:([\w-]+):(/{1,3})|www[.])(?:(?:(?:[^\s&()]|&amp;|&quot;)*(?:[^!"#$%&'()*+,.:;<=>?@\[\]^`{|}~\s]))|(?:\((?:[^\s&()]|&amp;|&quot;)*\)))+)"""  # noqa: E501
    )
)

def linkify(
    text: Union[str, bytes],
    shorten: bool = False,
    extra_params: Union[str, Callable[[str], str]] = "",
    require_protocol: bool = False,
    permitted_protocols: List[str] = ["http", "https"],
) -> str:
    if extra_params and not callable(extra_params):
        extra_params = " " + extra_params.strip()

    def make_link(m: Match[str]) -> str:
        url = m.group(1)
        proto = m.group(2)
        if require_protocol and not proto:
            return url

        if proto and proto not in permitted_protocols:
            return url

        href = m.group(1)
        if not proto:
            href = "http://" + href

        if callable(extra_params):
            params = " " + extra_params(href).strip()
        else:
            params = extra_params

        max_len = 30
        if shorten and len(url) > max_len:
            before_clip = url
            if proto:
                proto_len = len(proto) + 1 + len(m.group(3) or "")
            else:
                proto_len = 0

            parts = url[proto_len:].split("/")
            if len(parts) > 1:
                url = (
                    url[:proto_len]
                    + parts[0]
                    + "/"
                    + parts[1][:8].split("?")[0].split(".")[0]
                )

            if len(url) > max_len * 1.5:
                url = url[:max_len]

            if url != before_clip:
                amp = url.rfind("&")
                if amp > max_len - 5:
                    url = url[:amp]
                url += "..."

                if len(url) >= len(before_clip):
                    url = before_clip
                else:
                    params += ' title="%s"' % href

        return f'<a href="{href}"{params}>{url}</a>'

    text = _unicode(xhtml_escape(text))
    return _URL_RE.sub(make_link, text)
