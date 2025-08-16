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
from typing import Union, Tuple, Dict, Any, Optional

try:
    from .core._http.contrib.socks import SOCKSProxyManager
except ImportError:

    def SOCKSProxyManager(*args, **kwargs) -> None:
        raise InvalidScheme('Missing dependencies for SOCKS support.')
DEFAULT_POOLBLOCK: bool = False
DEFAULT_POOLSIZE: int = 10
DEFAULT_RETRIES: int = 0
DEFAULT_POOL_TIMEOUT: Optional[float] = None

def _pool_kwargs(verify: Union[bool, str], cert: Union[str, Tuple[str, str]]) -> Dict[str, Any]:
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
        cert_file = pool_kwargs['cert_file']
        key_file = pool_kwargs['key_file']
        if cert_file and (not os.path.exists(cert_file)):
            raise IOError('Could not find the TLS certificate file, invalid path: {0}'.format(cert_file))
        if key_file and (not os.path.exists(key_file)):
            raise IOError('Could not find the TLS key file, invalid path: {0}'.format(key_file))
    return pool_kwargs

class BaseAdapter(object):
    def __init__(self) -> None:
        super(BaseAdapter, self).__init__()

    def send(self, request: Any, stream: bool = False, timeout: Optional[Union[float, Tuple[float, float]]] = None, verify: Union[bool, str] = True, cert: Union[str, Tuple[str, str], None] = None, proxies: Optional[Dict[str, str]] = None) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

class HTTPAdapter(BaseAdapter):
    def __init__(self) -> None:
        super(HTTPAdapter, self).__init__()
        self.client = core.http3.Client()

    def build_response(self, req: Any, resp: Any) -> Response:
        ...

    def request_url(self, request: Any, proxies: Dict[str, str]) -> str:
        ...

    def add_headers(self, request: Any, **kwargs: Any) -> None:
        ...

    def proxy_headers(self, proxy: str) -> Dict[str, str]:
        ...

    def send(self, request: Any, stream: bool = False, timeout: Optional[Union[float, Tuple[float, float]]] = None, verify: Union[bool, str] = True, cert: Union[str, Tuple[str, str], None] = None, proxies: Optional[Dict[str, str]] = None) -> Response:
        ...

class AsyncHTTPAdapter(HTTPAdapter):
    def __init__(self, backend: Any = None, *args: Any, **kwargs: Any) -> None:
        super(AsyncHTTPAdapter, self).__init__(*args, **kwargs)
        self.client = core.http3.AsyncClient()

    async def build_response(self, req: Any, resp: Any) -> AsyncResponse:
        ...

    def close(self) -> None:
        ...

    async def send(self, request: Any, stream: bool = False, timeout: Optional[Union[float, Tuple[float, float]]] = None, verify: Union[bool, str] = True, cert: Union[str, Tuple[str, str], None] = None, proxies: Optional[Dict[str, str]] = None) -> AsyncResponse:
        ...
