import asyncio
import datetime
import io
import re
import socket
import string
import sys
import tempfile
import types
from http.cookies import SimpleCookie
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, Final, Iterator, Mapping, MutableMapping, Optional, Pattern, Tuple, Union, cast
from urllib.parse import parse_qsl
from multidict import CIMultiDict, CIMultiDictProxy, MultiDict, MultiDictProxy
from yarl import URL
from . import hdrs
from .abc import AbstractStreamWriter
from .helpers import _SENTINEL, ETAG_ANY, LIST_QUOTED_ETAG_RE, ChainMapProxy, ETag, HeadersMixin, frozen_dataclass_decorator, is_expected_content_type, parse_http_date, reify, sentinel, set_exception
from .http_parser import RawRequestMessage
from .http_writer import HttpVersion
from .multipart import BodyPartReader, MultipartReader
from .streams import EmptyStreamReader, StreamReader
from .typedefs import DEFAULT_JSON_DECODER, JSONDecoder, LooseHeaders, RawHeaders, StrOrURL
from .web_exceptions import HTTPBadRequest, HTTPRequestEntityTooLarge, HTTPUnsupportedMediaType
from .web_response import StreamResponse
if sys.version_info >= (3, 11):
    from typing import Self
else:
    Self = Any

@frozen_dataclass_decorator
class FileField:
    pass

_TCHAR: Final[str] = string.digits + string.ascii_letters + "!#$%&'*+.^_`|~-"
_TOKEN: Final[str] = f'[{_TCHAR}]+'
_QDTEXT: Final[str] = '[{}]'.format(''.join((chr(c) for c in (9, 32, 33) + tuple(range(35, 127))))
_QUOTED_PAIR: Final[str] = '\\\\[\\t !-~]'
_QUOTED_STRING: Final[str] = '"(?:{quoted_pair}|{qdtext})*"'.format(qdtext=_QDTEXT, quoted_pair=_QUOTED_PAIR)
_FORWARDED_PAIR: Final[str] = '({token})=({token}|{quoted_string})(:\\d{{1,4}})?'.format(token=_TOKEN, quoted_string=_QUOTED_STRING)
_QUOTED_PAIR_REPLACE_RE: Final[re.Pattern] = re.compile('\\\\([\\t !-~])')
_FORWARDED_PAIR_RE: Final[re.Pattern] = re.compile(_FORWARDED_PAIR)

class BaseRequest(MutableMapping[str, Any], HeadersMixin):
    POST_METHODS: Final[set[hdrs.Method]] = {hdrs.METH_PATCH, hdrs.METH_POST, hdrs.METH_PUT, hdrs.METH_TRACE, hdrs.METH_DELETE}
    _post: Optional[MultiDictProxy] = None
    _read_bytes: Optional[bytes] = None

    def __init__(self, message: RawRequestMessage, payload: StreamReader, protocol: AbstractStreamWriter, payload_writer: AbstractStreamWriter, task: asyncio.Task, loop: asyncio.BaseEventLoop, *, client_max_size: int = 1024 ** 2, state: Dict[str, Any] = None, scheme: Optional[str] = None, host: Optional[str] = None, remote: Optional[str] = None) -> None:
        # ...

    @property
    def task(self) -> asyncio.Task:
        return self._task

    @property
    def protocol(self) -> AbstractStreamWriter:
        return self._protocol

    @property
    def transport(self) -> socket.socket:
        return self._protocol.transport

    @property
    def writer(self) -> AbstractStreamWriter:
        return self._payload_writer

    @property
    def client_max_size(self) -> int:
        return self._client_max_size

    # ...

class Request(BaseRequest):
    _match_info: Optional[UrlMappingMatchInfo] = None

    def clone(self, *, method: Optional[hdrs.Method] = sentinel, rel_url: Optional[URL] = sentinel, headers: Optional[LooseHeaders] = sentinel, scheme: Optional[str] = sentinel, host: Optional[str] = sentinel, remote: Optional[str] = sentinel, client_max_size: Optional[int] = sentinel) -> Self:
        # ...

    @reify
    def match_info(self) -> UrlMappingMatchInfo:
        match_info = self._match_info
        assert match_info is not None
        return match_info

    @property
    def app(self) -> Application:
        match_info = self._match_info
        assert match_info is not None
        return match_info.current_app

    @property
    def config_dict(self) -> ChainMapProxy:
        match_info = self._match_info
        assert match_info is not None
        lst = match_info.apps
        app = self.app
        idx = lst.index(app)
        sublist = list(reversed(lst[:idx + 1]))
        return ChainMapProxy(sublist)

    async def _prepare_hook(self, response: StreamResponse) -> None:
        match_info = self._match_info
        if match_info is None:
            return
        for app in match_info._apps:
            if (on_response_prepare := app.on_response_prepare):
                await on_response_prepare.send(self, response)
