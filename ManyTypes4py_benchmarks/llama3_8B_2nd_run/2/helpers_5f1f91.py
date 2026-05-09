from typing import TypeVar, Generic, Protocol, Any, Callable, ContextManager, Mapping, Tuple, Union, Type, Dict, List, Optional, NamedTuple
from asyncio import TimeoutError
import re
import base64
import binascii
import inspect
import os
import platform
import time
import warnings
import weakref
from collections import namedtuple
from contextlib import suppress
from email.parser import HeaderParser
from email.utils import parsedate
from http.cookies import SimpleCookie
from math import ceil
from pathlib import Path
from types import TracebackType
from urllib.parse import quote
from multidict import CIMultiDict, MultiDict, MultiDictProxy, MultiMapping
from propcache.api import under_cached_property as reify
from yarl import URL
from . import hdrs
from .log import client_logger
from .typedefs import PathLike
from . import frozen_dataclass_decorator
from . import ETag
from . import CookieMixin

class BasicAuth(namedtuple('BasicAuth', ['login', 'password', 'encoding'])):
    ...

class ProxyInfo:
    pass

class MimeType:
    pass

class TimeoutHandle:
    ...

class HeadersMixin:
    ...

class ChainMapProxy(Mapping[Union[str, AppKey[Any]], Any]):
    ...

class CookieMixin:
    ...

class ETag:
    ...

def validate_etag_value(value: str) -> None:
    ...

def parse_http_date(date_str: str) -> datetime.datetime:
    ...

def must_be_empty_body(method: str, code: int) -> bool:
    ...

def should_remove_content_length(method: str, code: int) -> bool:
    ...
