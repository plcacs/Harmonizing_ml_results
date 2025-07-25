from __future__ import annotations
import io
import itertools
import sys
import typing
from typing import Optional, Iterable, Iterator, List, Tuple, Dict, Any, Callable, TypeVar
from .._models import Request, Response
from .._types import SyncByteStream
from .base import BaseTransport
if typing.TYPE_CHECKING:
    from _typeshed import OptExcInfo
    from _typeshed.wsgi import WSGIApplication

_T = TypeVar('_T')
__all__ = ['WSGITransport']

def _skip_leading_empty_chunks(body: Iterable[bytes]) -> Iterator[bytes]:
    body_iter = iter(body)
    for chunk in body_iter:
        if chunk:
            return itertools.chain([chunk], body_iter)
    return iter([])

class WSGIByteStream(SyncByteStream):
    def __init__(self, result: Iterable[bytes]) -> None:
        self._close: Optional[Callable[[], None]] = getattr(result, 'close', None)
        self._result: Iterator[bytes] = _skip_leading_empty_chunks(result)

    def __iter__(self) -> Iterator[bytes]:
        for part in self._result:
            yield part

    def close(self) -> None:
        if self._close is not None:
            self._close()

class WSGITransport(BaseTransport):
    def __init__(
        self,
        app: WSGIApplication,
        raise_app_exceptions: bool = True,
        script_name: str = '',
        remote_addr: str = '127.0.0.1',
        wsgi_errors: Optional[io.TextIOBase] = None
    ) -> None:
        self.app: WSGIApplication = app
        self.raise_app_exceptions: bool = raise_app_exceptions
        self.script_name: str = script_name
        self.remote_addr: str = remote_addr
        self.wsgi_errors: Optional[io.TextIOBase] = wsgi_errors

    def handle_request(self, request: Request) -> Response:
        request.read()
        wsgi_input = io.BytesIO(request.content)
        port = request.url.port or {'http': 80, 'https': 443}[request.url.scheme]
        environ: Dict[str, Any] = {
            'wsgi.version': (1, 0),
            'wsgi.url_scheme': request.url.scheme,
            'wsgi.input': wsgi_input,
            'wsgi.errors': self.wsgi_errors or sys.stderr,
            'wsgi.multithread': True,
            'wsgi.multiprocess': False,
            'wsgi.run_once': False,
            'REQUEST_METHOD': request.method,
            'SCRIPT_NAME': self.script_name,
            'PATH_INFO': request.url.path,
            'QUERY_STRING': request.url.query.decode('ascii'),
            'SERVER_NAME': request.url.host,
            'SERVER_PORT': str(port),
            'SERVER_PROTOCOL': 'HTTP/1.1',
            'REMOTE_ADDR': self.remote_addr
        }
        for header_key, header_value in request.headers.raw:
            key = header_key.decode('ascii').upper().replace('-', '_')
            if key not in ('CONTENT_TYPE', 'CONTENT_LENGTH'):
                key = 'HTTP_' + key
            environ[key] = header_value.decode('ascii')
        
        seen_status: Optional[str] = None
        seen_response_headers: Optional[List[Tuple[str, str]]] = None
        seen_exc_info: Optional[OptExcInfo] = None

        def start_response(
            status: str,
            response_headers: List[Tuple[str, str]],
            exc_info: Optional[OptExcInfo] = None
        ) -> Callable[[bytes], None]:
            nonlocal seen_status, seen_response_headers, seen_exc_info
            seen_status = status
            seen_response_headers = response_headers
            seen_exc_info = exc_info
            return lambda _: None

        result: Iterable[bytes] = self.app(environ, start_response)
        stream = WSGIByteStream(result)
        assert seen_status is not None
        assert seen_response_headers is not None
        if seen_exc_info and seen_exc_info[0] and self.raise_app_exceptions:
            raise seen_exc_info[1]
        status_code = int(seen_status.split()[0])
        headers = [(key.encode('ascii'), value.encode('ascii')) for key, value in seen_response_headers]
        return Response(status_code, headers=headers, stream=stream)
