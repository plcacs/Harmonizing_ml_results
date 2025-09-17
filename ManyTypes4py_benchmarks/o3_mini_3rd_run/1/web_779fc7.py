#!/usr/bin/env python3
import base64
import binascii
import datetime
import email.utils
import functools
import gzip
import hashlib
import hmac
import http.cookies
from inspect import isclass
from io import BytesIO
import mimetypes
import numbers
import os.path
import re
import socket
import sys
import threading
import time
import warnings
import tornado
import traceback
import types
import urllib.parse
from urllib.parse import urlencode
from tornado.concurrent import Future, future_set_result_unless_cancelled
from tornado import escape
from tornado import gen
from tornado.httpserver import HTTPServer
from tornado import httputil
from tornado import iostream
from tornado import locale
from tornado.log import access_log, app_log, gen_log
from tornado import template
from tornado.escape import utf8, _unicode
from tornado.routing import AnyMatches, DefaultHostMatches, HostMatches, ReversibleRouter, Rule, ReversibleRuleRouter, URLSpec, _RuleList
from tornado.util import ObjectDict, unicode_type, _websocket_mask
url = URLSpec
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union, cast, overload

if __name__ == "__main__":
    # The rest of the code (the full implementation of RequestHandler, Application, etc.)
    # is assumed to be here.
    pass

# -----------------------------------------------------------------------------
# Free functions with type annotations

T = TypeVar("T", bound="RequestHandler")

def stream_request_body(cls: Type[T]) -> Type[T]:
    """
    Apply to RequestHandler subclasses to enable streaming body support.
    """
    if not issubclass(cls, RequestHandler):
        raise TypeError('expected subclass of RequestHandler, got %r' % cls)
    cls._stream_request_body = True
    return cls

def _has_stream_request_body(cls: Type["RequestHandler"]) -> bool:
    if not issubclass(cls, RequestHandler):
        raise TypeError('expected subclass of RequestHandler, got %r' % cls)
    return cls._stream_request_body

def removeslash(method: Callable[..., Any]) -> Callable[..., Any]:
    """Use this decorator to remove trailing slashes from the request path."""
    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if self.request.path.endswith('/'):
            if self.request.method in ('GET', 'HEAD'):
                uri: str = self.request.path.rstrip('/')
                if uri:
                    if self.request.query:
                        uri += '?' + self.request.query
                    self.redirect(uri, permanent=True)
                    return None
            else:
                raise HTTPError(404)
        return method(self, *args, **kwargs)
    return wrapper

def addslash(method: Callable[..., Any]) -> Callable[..., Any]:
    """Use this decorator to add a missing trailing slash to the request path."""
    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not self.request.path.endswith('/'):
            if self.request.method in ('GET', 'HEAD'):
                uri: str = self.request.path + '/'
                if self.request.query:
                    uri += '?' + self.request.query
                self.redirect(uri, permanent=True)
                return None
            raise HTTPError(404)
        return method(self, *args, **kwargs)
    return wrapper

def create_signed_value(
    secret: Union[str, Dict[int, str]],
    name: str,
    value: str,
    version: Optional[int] = None,
    clock: Optional[Callable[[], float]] = None,
    key_version: Optional[int] = None
) -> bytes:
    if version is None:
        version = DEFAULT_SIGNED_VALUE_VERSION
    if clock is None:
        clock = time.time
    timestamp: bytes = utf8(str(int(clock())))
    value = base64.b64encode(utf8(value))
    if version == 1:
        assert not isinstance(secret, dict)
        signature: bytes = _create_signature_v1(secret, name, value, timestamp)
        result: bytes = b'|'.join([value, timestamp, signature])
        return result
    elif version == 2:
        def format_field(s: str) -> bytes:
            return utf8(f'{len(s)}:') + utf8(s)
        to_sign: bytes = b'|'.join([b'2', format_field(str(key_version or 0)), format_field(timestamp.decode("utf8")), format_field(name), format_field(value.decode("utf8")), b''])
        if isinstance(secret, dict):
            assert key_version is not None, 'Key version must be set when sign key dict is used'
            assert version >= 2, 'Version must be at least 2 for key version support'
            secret = secret[key_version]
        signature = _create_signature_v2(secret, to_sign)
        return to_sign + signature
    else:
        raise ValueError('Unsupported version %d' % version)

_signed_value_version_re: re.Pattern = re.compile(b'^([1-9][0-9]*)\\|(.*)$')

def _get_version(value: bytes) -> int:
    m = _signed_value_version_re.match(value)
    if m is None:
        version = 1
    else:
        try:
            version = int(m.group(1))
            if version > 999:
                version = 1
        except ValueError:
            version = 1
    return version

def decode_signed_value(
    secret: Union[str, Dict[int, str]],
    name: str,
    value: Optional[str],
    max_age_days: int = 31,
    clock: Optional[Callable[[], float]] = None,
    min_version: Optional[int] = None
) -> Optional[bytes]:
    if clock is None:
        clock = time.time
    if min_version is None:
        min_version = DEFAULT_SIGNED_VALUE_MIN_VERSION
    if min_version > 2:
        raise ValueError('Unsupported min_version %d' % min_version)
    if not value:
        return None
    value_bytes: bytes = utf8(value)
    version = _get_version(value_bytes)
    if version < min_version:
        return None
    if version == 1:
        assert not isinstance(secret, dict)
        return _decode_signed_value_v1(secret, name, value_bytes, max_age_days, clock)
    elif version == 2:
        return _decode_signed_value_v2(secret, name, value_bytes, max_age_days, clock)
    else:
        return None

def _decode_signed_value_v1(
    secret: str,
    name: str,
    value: bytes,
    max_age_days: int,
    clock: Callable[[], float]
) -> Optional[bytes]:
    parts: List[bytes] = utf8(value).split(b'|')
    if len(parts) != 3:
        return None
    signature: bytes = _create_signature_v1(secret, name, parts[0], parts[1])
    if not hmac.compare_digest(parts[2], signature):
        gen_log.warning('Invalid cookie signature %r', value)
        return None
    timestamp: int = int(parts[1])
    if timestamp < clock() - max_age_days * 86400:
        gen_log.warning('Expired cookie %r', value)
        return None
    if timestamp > clock() + 31 * 86400:
        gen_log.warning('Cookie timestamp in future; possible tampering %r', value)
        return None
    if parts[1].startswith(b'0'):
        gen_log.warning('Tampered cookie %r', value)
        return None
    try:
        return base64.b64decode(parts[0])
    except Exception:
        return None

def _decode_fields_v2(value: bytes) -> Tuple[int, bytes, bytes, bytes, bytes]:
    def _consume_field(s: bytes) -> Tuple[bytes, bytes]:
        length_str, sep, rest = s.partition(b':')
        n: int = int(length_str)
        field_value: bytes = rest[:n]
        if rest[n:n + 1] != b'|':
            raise ValueError('malformed v2 signed value field')
        rest = rest[n + 1:]
        return (field_value, rest)
    rest: bytes = value[2:]
    key_version_field, rest = _consume_field(rest)
    timestamp_field, rest = _consume_field(rest)
    name_field, rest = _consume_field(rest)
    value_field, passed_sig = _consume_field(rest)
    return (int(key_version_field.decode("utf8")), timestamp_field, name_field, value_field, passed_sig)

def _decode_signed_value_v2(
    secret: Union[str, Dict[int, str]],
    name: str,
    value: bytes,
    max_age_days: int,
    clock: Callable[[], float]
) -> Optional[bytes]:
    try:
        key_version, timestamp_bytes, name_field, value_field, passed_sig = _decode_fields_v2(value)
    except ValueError:
        return None
    signed_string: bytes = value[:-len(passed_sig)]
    if isinstance(secret, dict):
        try:
            secret = secret[key_version]
        except KeyError:
            return None
    expected_sig: bytes = _create_signature_v2(secret, signed_string)
    if not hmac.compare_digest(passed_sig, expected_sig):
        return None
    if name_field != utf8(name):
        return None
    timestamp: int = int(timestamp_bytes)
    if timestamp < clock() - max_age_days * 86400:
        return None
    try:
        return base64.b64decode(value_field)
    except Exception:
        return None

def get_signature_key_version(value: str) -> Optional[int]:
    value_bytes: bytes = utf8(value)
    version: int = _get_version(value_bytes)
    if version < 2:
        return None
    try:
        key_version, _, _, _, _ = _decode_fields_v2(value_bytes)
    except ValueError:
        return None
    return key_version

def _create_signature_v1(secret: str, *parts: Any) -> bytes:
    hash_obj = hmac.new(utf8(secret), digestmod=hashlib.sha1)
    for part in parts:
        hash_obj.update(utf8(part))
    return utf8(hash_obj.hexdigest())

def _create_signature_v2(secret: str, s: Any) -> bytes:
    hash_obj = hmac.new(utf8(secret), digestmod=hashlib.sha256)
    hash_obj.update(utf8(s))
    return utf8(hash_obj.hexdigest())

def is_absolute(path: str) -> bool:
    return any((path.startswith(x) for x in ['/', 'http:', 'https:']))
