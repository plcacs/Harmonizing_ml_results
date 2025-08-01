from __future__ import annotations
import io
import itertools
import sys
import typing
from .._models import Request, Response
from .._types import SyncByteStream
from .base import BaseTransport

if typing.TYPE_CHECKING:
    from _typeshed import OptExcInfo
    from _typeshed.wsgi import WSGIApplication

_T = typing.TypeVar('_T')
__all__ = ['WSGITransport']


def _skip_leading_empty_chunks(body: typing.Iterable[bytes]) -> typing.Iterator[bytes]:
    body = iter(body)
    for chunk in body:
        if chunk:
            return itertools.chain([chunk], body)
    return iter([])


class WSGIByteStream(SyncByteStream):
    def __init__(self, result: typing.Iterable[bytes]) -> None:
        self._close: typing.Optional[typing.Callable[[], None]] = getattr(result, "close", None)
        self._result: typing.Iterator[bytes] = _skip_leading_empty_chunks(result)

    def __iter__(self) -> typing.Iterator[bytes]:
        for part in self._result:
            yield part

    def close(self) -> None:
        if self._close is not None:
            self._close()


class WSGITransport(BaseTransport):
    """
    A custom transport that handles sending requests directly to an WSGI app.
    The simplest way to use this functionality is to use the `app` argument.

    ```
    client = httpx.Client(app=app)
    ```

    Alternatively, you can setup the transport instance explicitly.
    This allows you to include any additional configuration arguments specific
    to the WSGITransport class:

    ```
    transport = httpx.WSGITransport(
        app=app,
        script_name="/submount",
        remote_addr="1.2.3.4"
    )
    client = httpx.Client(transport=transport)
    ```

    Arguments:

    * `app` - The WSGI application.
    * `raise_app_exceptions` - Boolean indicating if exceptions in the application
       should be raised. Default to `True`. Can be set to `False` for use cases
       such as testing the content of a client 500 response.
    * `script_name` - The root path on which the WSGI application should be mounted.
    * `remote_addr` - A string indicating the client IP of incoming requests.
    """
    def __init__(
        self,
        app: WSGIApplication,
        raise_app_exceptions: bool = True,
        script_name: str = "",
        remote_addr: str = "127.0.0.1",
        wsgi_errors: typing.Optional[io.TextIOBase] = None,
    ) -> None:
        self.app = app
        self.raise_app_exceptions = raise_app_exceptions
        self.script_name = script_name
        self.remote_addr = remote_addr
        self.wsgi_errors = wsgi_errors

    def handle_request(self, request: Request) -> Response:
        request.read()
        wsgi_input: io.BytesIO = io.BytesIO(request.content)
        port: int = request.url.port or {"http": 80, "https": 443}[request.url.scheme]
        environ: typing.Dict[str, typing.Any] = {
            "wsgi.version": (1, 0),
            "wsgi.url_scheme": request.url.scheme,
            "wsgi.input": wsgi_input,
            "wsgi.errors": self.wsgi_errors or sys.stderr,
            "wsgi.multithread": True,
            "wsgi.multiprocess": False,
            "wsgi.run_once": False,
            "REQUEST_METHOD": request.method,
            "SCRIPT_NAME": self.script_name,
            "PATH_INFO": request.url.path,
            "QUERY_STRING": request.url.query.decode("ascii"),
            "SERVER_NAME": request.url.host,
            "SERVER_PORT": str(port),
            "SERVER_PROTOCOL": "HTTP/1.1",
            "REMOTE_ADDR": self.remote_addr,
        }
        for header_key, header_value in request.headers.raw:
            key: str = header_key.decode("ascii").upper().replace("-", "_")
            if key not in ("CONTENT_TYPE", "CONTENT_LENGTH"):
                key = "HTTP_" + key
            environ[key] = header_value.decode("ascii")

        seen_status: typing.Optional[str] = None
        seen_response_headers: typing.Optional[list[tuple[str, str]]] = None
        seen_exc_info: typing.Optional[OptExcInfo] = None

        def start_response(
            status: str,
            response_headers: list[tuple[str, str]],
            exc_info: OptExcInfo = None,
        ) -> typing.Callable[[bytes], None]:
            nonlocal seen_status, seen_response_headers, seen_exc_info
            seen_status = status
            seen_response_headers = response_headers
            seen_exc_info = exc_info
            return lambda _: None

        result: typing.Iterable[bytes] = self.app(environ, start_response)
        stream: WSGIByteStream = WSGIByteStream(result)
        assert seen_status is not None
        assert seen_response_headers is not None
        if seen_exc_info and seen_exc_info[0] and self.raise_app_exceptions:
            raise seen_exc_info[1]
        status_code: int = int(seen_status.split()[0])
        headers: list[tuple[bytes, bytes]] = [
            (key.encode("ascii"), value.encode("ascii")) for key, value in seen_response_headers
        ]
        return Response(status_code, headers=headers, stream=stream)