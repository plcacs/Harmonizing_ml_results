from typing import Any, Dict, Optional, Union, Tuple, List
import os.path
import socket
import rfc3986
from . import core
from .http_models import Response, AsyncResponse
from ._basics import urlparse, basestring
from .http_utils import DEFAULT_CA_BUNDLE_PATH, get_encoding_from_headers, prepend_scheme_if_needed, get_auth_from_url, urldefragauth, select_proxy
from ._structures import HTTPHeaderDict
from .http_cookies import extract_cookies_to_jar
from .exceptions import ConnectionError, ConnectTimeout, ReadTimeout, SSLError, ProxyError, RetryError, InvalidScheme
from .http_auth import _basic_auth_str

try:
    from .core._http.contrib.socks import SOCKSProxyManager
except ImportError:

    def SOCKSProxyManager(*args: Any, **kwargs: Any) -> Any:
        raise InvalidScheme('Missing dependencies for SOCKS support.')

DEFAULT_POOLBLOCK: bool = False
DEFAULT_POOLSIZE: int = 10
DEFAULT_RETRIES: int = 0
DEFAULT_POOL_TIMEOUT: Optional[float] = None

def _pool_kwargs(verify: Union[bool, str], cert: Optional[Union[str, Tuple[str, str]]] = None) -> Dict[str, Any]:
    """Create a dictionary of keyword arguments to pass to a
    :class:`PoolManager <urllib3.poolmanager.PoolManager>` with the
    necessary SSL configuration.
    """
    pool_kwargs: Dict[str, Any] = {}
    if verify:
        cert_loc: Optional[str] = None
        if verify is not True:
            cert_loc = verify
        if not cert_loc:
            cert_loc = DEFAULT_CA_BUNDLE_PATH
        if not cert_loc or not os.path.exists(cert_loc):
            raise IOError('Could not find a suitable TLS CA certificate bundle, invalid path: {0}'.format(cert_loc))
        pool_kwargs['cert_reqs'] = 'CERT_REQUIRED'
        if not os.path.isdir(cert_loc):
            pool_kwargs['ca_certs'] = cert_loc
            pool_kwargs['ca_cert_dir'] = None
        else:
            pool_kwargs['ca_cert_dir'] = cert_loc
            pool_kwargs['ca_certs'] = None
    else:
        pool_kwargs['cert_reqs'] = 'CERT_NONE'
        pool_kwargs['ca_certs'] = None
        pool_kwargs['ca_cert_dir'] = None
    if cert:
        if not isinstance(cert, basestring):
            pool_kwargs['cert_file'] = cert[0]
            pool_kwargs['key_file'] = cert[1]
        else:
            pool_kwargs['cert_file'] = cert
            pool_kwargs['key_file'] = None
        cert_file: Optional[str] = pool_kwargs['cert_file']
        key_file: Optional[str] = pool_kwargs['key_file']
        if cert_file and (not os.path.exists(cert_file)):
            raise IOError('Could not find the TLS certificate file, invalid path: {0}'.format(cert_file))
        if key_file and (not os.path.exists(key_file)):
            raise IOError('Could not find the TLS key file, invalid path: {0}'.format(key_file))
    return pool_kwargs

class BaseAdapter(object):
    """The Base Transport Adapter"""

    def __init__(self) -> None:
        super(BaseAdapter, self).__init__()

    def send(
        self,
        request: Any,
        stream: bool = False,
        timeout: Optional[Union[float, Tuple[float, float]]] = None,
        verify: Union[bool, str] = True,
        cert: Optional[Union[str, Tuple[str, str]]] = None,
        proxies: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Sends PreparedRequest object. Returns Response object."""
        raise NotImplementedError

    def close(self) -> None:
        """Cleans up adapter specific items."""
        raise NotImplementedError

class HTTPAdapter(BaseAdapter):
    """The built-in HTTP Adapter for urllib3.
    Provides a general-case interface for Requests sessions to contact HTTP and
    HTTPS urls by implementing the Transport Adapter interface.
    """
    __attrs__ = ['max_retries', 'config', '_pool_connections', '_pool_maxsize', '_pool_block']

    def __init__(self) -> None:
        super(HTTPAdapter, self).__init__()
        self.client: Any = core.http3.Client()

    def __getstate__(self) -> Dict[str, Any]:
        return {attr: getattr(self, attr, None) for attr in self.__attrs__}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.proxy_manager: Dict[str, Any] = {}
        self.config = {}
        for attr, value in state.items():
            setattr(self, attr, value)
        self.init_poolmanager(self._pool_connections, self._pool_maxsize, block=self._pool_block)

    def build_response(self, req: Any, resp: Any) -> Response:
        """Builds a :class:`Response <requests.Response>` object from a urllib3
        response.
        """
        response: Response = Response()
        response.status_code = getattr(resp, 'status_code', None)
        response.headers = HTTPHeaderDict(getattr(resp, 'headers', {}))
        response.encoding = get_encoding_from_headers(response.headers)
        response.protocol = getattr(resp, 'protocol', None)
        response.raw = resp
        response.reason = getattr(resp, 'reason_phrase', None)
        if isinstance(req.url, bytes):
            response.url = req.url.decode('utf-8')
        else:
            response.url = req.url
        extract_cookies_to_jar(response.cookies, req, resp)
        response.request = req
        response.connection = self
        return response

    def request_url(self, request: Any, proxies: Optional[Dict[str, Any]] = None) -> str:
        """Obtain the url to use when making the final request."""
        return request.url

    def add_headers(self, request: Any, **kwargs: Any) -> None:
        """Add any headers needed by the connection."""
        pass

    def proxy_headers(self, proxy: str) -> Dict[str, str]:
        """Returns a dictionary of the headers to add to any request sent through a proxy."""
        headers: Dict[str, str] = {}
        username, password = get_auth_from_url(proxy)
        if username:
            headers['Proxy-Authorization'] = _basic_auth_str(username, password)
        return headers

    def send(
        self,
        request: Any,
        stream: bool = False,
        timeout: Optional[Union[float, Tuple[float, float]]] = None,
        verify: Union[bool, str] = True,
        cert: Optional[Union[str, Tuple[str, str]]] = None,
        proxies: Optional[Dict[str, Any]] = None
    ) -> Response:
        """Sends PreparedRequest object. Returns Response object."""
        url: str = self.request_url(request, proxies)
        self.add_headers(request)
        chunked: bool = not (request.body is None or 'Content-Length' in request.headers)
        if isinstance(timeout, tuple):
            try:
                connect, read = timeout
            except ValueError as e:
                err: str = 'Invalid timeout {0}. Pass a (connect, read) timeout tuple, or a single float to set both timeouts to the same value'.format(timeout)
                raise ValueError(err)
        try:
            resp: Any = core.blocking_request(
                method=request.method,
                url=url,
                data=request.body,
                headers=[(k, request.headers[k]) for k in request.headers],
                allow_redirects=False,
                timeout=timeout,
                client=self.client
            )
        except (core.http3.exceptions.ProtocolError, socket.error) as err:
            raise ConnectionError(err, request=request)
        except core.http3.exceptions.PoolTimeout as e:
            raise ConnectionError(e, request=request)
        except (core.http3.exceptions.HttpError,) as e:
            if isinstance(e, core.http3.exceptions.PoolTimeout.ReadTimeout):
                raise core.http3.exceptions.PoolTimeout.ReadTimeout(e, request=request)
            else:
                raise
        return self.build_response(request, resp)

class AsyncHTTPAdapter(HTTPAdapter):
    """docstring for AsyncHTTPAdapter"""

    def __init__(self, backend: Optional[Any] = None, *args: Any, **kwargs: Any) -> None:
        super(AsyncHTTPAdapter, self).__init__(*args, **kwargs)
        self.client = core.http3.AsyncClient()

    async def build_response(self, req: Any, resp: Any) -> AsyncResponse:
        """Builds a :class:`Response <requests.Response>` object from a urllib3 response."""
        response: AsyncResponse = AsyncResponse()
        response.status_code = getattr(resp, 'status_code', None)
        response.headers = HTTPHeaderDict(getattr(resp, 'headers', {}))
        response.encoding = get_encoding_from_headers(response.headers)
        response.raw = resp
        response.reason = getattr(resp, 'reason_phrase', None)
        response.protocol = getattr(resp, 'protocol', None)
        if isinstance(req.url, bytes):
            response.url = req.url.decode('utf-8')
        else:
            response.url = req.url
        extract_cookies_to_jar(response.cookies, req, resp)
        response.request = req
        response.connection = self
        return response

    def close(self) -> None:
        """Disposes of any internal state."""
        pass

    async def send(
        self,
        request: Any,
        stream: bool = False,
        timeout: Optional[Union[float, Tuple[float, float]]] = None,
        verify: Union[bool, str] = True,
        cert: Optional[Union[str, Tuple[str, str]]] = None,
        proxies: Optional[Dict[str, Any]] = None
    ) -> AsyncResponse:
        """Sends PreparedRequest object. Returns Response object."""
        url: str = self.request_url(request, proxies)
        self.add_headers(request)
        chunked: bool = not (request.body is None or 'Content-Length' in request.headers)
        if isinstance(timeout, tuple):
            try:
                connect, read = timeout
            except ValueError as e:
                err: str = 'Invalid timeout {0}. Pass a (connect, read) timeout tuple, or a single float to set both timeouts to the same value'.format(timeout)
                raise ValueError(err)
        try:
            resp: Any = await core.request(
                method=request.method,
                url=url,
                data=request.body,
                headers=[(k, request.headers[k]) for k in request.headers],
                allow_redirects=False,
                timeout=timeout,
                client=self.client
            )
        except (core.http3.exceptions.ProtocolError, socket.error) as err:
            raise ConnectionError(err, request=request)
        except core.http3.exceptions.PoolTimeout as e:
            raise ConnectionError(e, request=request)
        except (core.http3.exceptions.HttpError,) as e:
            if isinstance(e, core.http3.exceptions.PoolTimeout.ReadTimeout):
                raise core.http3.exceptions.PoolTimeout.ReadTimeout(e, request=request)
            else:
                raise
        return await self.build_response(request, resp)