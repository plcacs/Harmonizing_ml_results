"""
requests.adapters
~~~~~~~~~~~~~~~~~

This module contains the transport adapters that Requests uses to define
and maintain connections.
"""
import os.path
import socket
from typing import Any, Dict, Optional, Tuple, Union, cast
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
    def SOCKSProxyManager(*args: Any, **kwargs: Any) -> None:
        raise InvalidScheme('Missing dependencies for SOCKS support.')

DEFAULT_POOLBLOCK: bool = False
DEFAULT_POOLSIZE: int = 10
DEFAULT_RETRIES: int = 0
DEFAULT_POOL_TIMEOUT: Optional[float] = None

def _pool_kwargs(verify: Union[bool, str], cert: Optional[Union[str, Tuple[str, str]]]) -> Dict[str, Any]:
    """Create a dictionary of keyword arguments to pass to a
    :class:`PoolManager <urllib3.poolmanager.PoolManager>` with the
    necessary SSL configuration.

    :param verify: Whether we should actually verify the certificate;
                   optionally a path to a CA certificate bundle or
                   directory of CA certificates.
    :param cert: The path to the client certificate and key, if any.
                 This can either be the path to the certificate and
                 key concatenated in a single file, or as a tuple of
                 (cert_file, key_file).
    """
    pool_kwargs: Dict[str, Any] = {}
    if verify:
        cert_loc: Optional[str] = None
        if verify is not True:
            cert_loc = cast(str, verify)
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
        cert_file = pool_kwargs['cert_file']
        key_file = pool_kwargs['key_file']
        if cert_file and (not os.path.exists(cert_file)):
            raise IOError('Could not find the TLS certificate file, invalid path: {0}'.format(cert_file))
        if key_file and (not os.path.exists(key_file)):
            raise IOError('Could not find the TLS key file, invalid path: {0}'.format(key_file))
    return pool_kwargs

class BaseAdapter(object):
    """The Base Transport Adapter"""

    def __init__(self) -> None:
        super(BaseAdapter, self).__init__()

    def send(self, request: Any, stream: bool = False, timeout: Optional[Union[float, Tuple[float, float]]] = None, verify: Union[bool, str] = True, cert: Optional[Union[str, Tuple[str, str]]] = None, proxies: Optional[Dict[str, str]] = None) -> Response:
        """Sends PreparedRequest object. Returns Response object.

        :param request: The :class:`PreparedRequest <PreparedRequest>` being sent.
        :param stream: (optional) Whether to stream the request content.
        :param timeout: (optional) How long to wait for the server to send
            data before giving up, as a float, or a :ref:`(connect timeout,
            read timeout) <timeouts>` tuple.
        :type timeout: float or tuple
        :param verify: (optional) Either a boolean, in which case it controls whether we verify
            the server's TLS certificate, or a string, in which case it must be a path
            to a CA bundle to use
        :param cert: (optional) Any user-provided SSL certificate to be trusted.
        :param proxies: (optional) The proxies dictionary to apply to the request.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Cleans up adapter specific items."""
        raise NotImplementedError

class HTTPAdapter(BaseAdapter):
    """The built-in HTTP Adapter for urllib3."""
    __attrs__ = ['max_retries', 'config', '_pool_connections', '_pool_maxsize', '_pool_block']

    def __init__(self) -> None:
        super(HTTPAdapter, self).__init__()
        self.client = core.http3.Client()

    def __getstate__(self) -> Dict[str, Any]:
        return {attr: getattr(self, attr, None) for attr in self.__attrs__}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.proxy_manager = {}
        self.config = {}
        for attr, value in state.items():
            setattr(self, attr, value)
        self.init_poolmanager(self._pool_connections, self._pool_maxsize, block=self._pool_block)

    def build_response(self, req: Any, resp: Any) -> Response:
        """Builds a :class:`Response <requests.Response>` object from a urllib3
        response. This should not be called from user code, and is only exposed
        for use when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`

        :param req: The :class:`PreparedRequest <PreparedRequest>` used to generate the response.
        :param resp: The urllib3 response object.
        :rtype: requests.Response
        """
        response = Response()
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

    def request_url(self, request: Any, proxies: Optional[Dict[str, str]]) -> str:
        """Obtain the url to use when making the final request."""
        return request.url

    def add_headers(self, request: Any, **kwargs: Any) -> None:
        """Add any headers needed by the connection."""
        pass

    def proxy_headers(self, proxy: str) -> Dict[str, str]:
        """Returns a dictionary of the headers to add to any request sent
        through a proxy."""
        headers: Dict[str, str] = {}
        username, password = get_auth_from_url(proxy)
        if username:
            headers['Proxy-Authorization'] = _basic_auth_str(username, password)
        return headers

    def send(self, request: Any, stream: bool = False, timeout: Optional[Union[float, Tuple[float, float]]] = None, verify: Union[bool, str] = True, cert: Optional[Union[str, Tuple[str, str]]] = None, proxies: Optional[Dict[str, str]] = None) -> Response:
        """Sends PreparedRequest object. Returns Response object."""
        url = self.request_url(request, proxies)
        self.add_headers(request)
        chunked = not (request.body is None or 'Content-Length' in request.headers)
        if isinstance(timeout, tuple):
            try:
                connect, read = timeout
            except ValueError as e:
                err = 'Invalid timeout {0}. Pass a (connect, read) timeout tuple, or a single float to set both timeouts to the same value'.format(timeout)
                raise ValueError(err)
        try:
            resp = core.blocking_request(method=request.method, url=url, data=request.body, headers=[(k, request.headers[k]) for k in request.headers], allow_redirects=False, timeout=timeout, client=self.client)
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
        """Builds a :class:`Response <requests.Response>` object from a urllib3
        response."""
        response = AsyncResponse()
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

    async def send(self, request: Any, stream: bool = False, timeout: Optional[Union[float, Tuple[float, float]]] = None, verify: Union[bool, str] = True, cert: Optional[Union[str, Tuple[str, str]]] = None, proxies: Optional[Dict[str, str]] = None) -> AsyncResponse:
        """Sends PreparedRequest object. Returns Response object."""
        url = self.request_url(request, proxies)
        self.add_headers(request)
        chunked = not (request.body is None or 'Content-Length' in request.headers)
        if isinstance(timeout, tuple):
            try:
                connect, read = timeout
            except ValueError as e:
                err = 'Invalid timeout {0}. Pass a (connect, read) timeout tuple, or a single float to set both timeouts to the same value'.format(timeout)
                raise ValueError(err)
        try:
            resp = await core.request(method=request.method, url=url, data=request.body, headers=[(k, request.headers[k]) for k in request.headers], allow_redirects=False, timeout=timeout, client=self.client)
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
