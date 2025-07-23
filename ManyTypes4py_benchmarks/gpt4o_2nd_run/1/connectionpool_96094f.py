from __future__ import absolute_import
import errno
import logging
import sys
import warnings
from socket import error as SocketError, timeout as SocketTimeout
import socket
import h11
from typing import Optional, Dict, Any, Union
from ..base import Request, DEFAULT_PORTS
from ..exceptions import ClosedPoolError, ProtocolError, EmptyPoolError, LocationValueError, MaxRetryError, ProxyError, ReadTimeoutError, SSLError, TimeoutError, InsecureRequestWarning, NewConnectionError
from ..packages.ssl_match_hostname import CertificateError
from ..packages import six
from ..packages.six.moves import queue
from ..request import RequestMethods
from .response import HTTPResponse
from .connection import HTTP1Connection
from ..util.connection import is_connection_dropped
from ..util.request import set_file_position
from ..util.retry import Retry
from ..util.ssl_ import create_urllib3_context, merge_context_settings, resolve_ssl_version, resolve_cert_reqs, BaseSSLError
from ..util.timeout import Timeout
from ..util.url import get_host, Url

try:
    import ssl
except ImportError:
    ssl = None

if six.PY2:
    import Queue as _unused_module_Queue

xrange = six.moves.xrange
log = logging.getLogger(__name__)
_Default = object()

def _add_transport_headers(headers: Dict[str, str]) -> None:
    transfer_headers = ('content-length', 'transfer-encoding')
    for header_name in headers:
        if header_name.lower() in transfer_headers:
            return
    headers['transfer-encoding'] = 'chunked'

def _build_context(context: Optional[ssl.SSLContext], keyfile: Optional[str], certfile: Optional[str], cert_reqs: Optional[str], ca_certs: Optional[str], ca_cert_dir: Optional[str], ssl_version: Optional[str]) -> ssl.SSLContext:
    if context is None:
        context = create_urllib3_context(ssl_version=resolve_ssl_version(ssl_version), cert_reqs=resolve_cert_reqs(cert_reqs))
    context = merge_context_settings(context, keyfile=keyfile, certfile=certfile, cert_reqs=cert_reqs, ca_certs=ca_certs, ca_cert_dir=ca_cert_dir)
    return context

class ConnectionPool(object):
    scheme: Optional[str] = None
    QueueCls = queue.LifoQueue

    def __init__(self, host: str, port: Optional[int] = None) -> None:
        if not host:
            raise LocationValueError('No host specified.')
        self.host = _ipv6_host(host).lower()
        self.port = port

    def __str__(self) -> str:
        return '%s(host=%r, port=%r)' % (type(self).__name__, self.host, self.port)

    def __enter__(self) -> 'ConnectionPool':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def close(self) -> None:
        pass

_blocking_errnos = set([errno.EAGAIN, errno.EWOULDBLOCK])

class HTTPConnectionPool(ConnectionPool, RequestMethods):
    scheme: str = 'http'
    ConnectionCls = HTTP1Connection
    ResponseCls = HTTPResponse

    def __init__(self, host: str, port: Optional[int] = None, timeout: Union[Timeout, float] = Timeout.DEFAULT_TIMEOUT, maxsize: int = 1, block: bool = False, headers: Optional[Dict[str, str]] = None, retries: Optional[Retry] = None, _proxy: Optional[Url] = None, _proxy_headers: Optional[Dict[str, str]] = None, **conn_kw: Any) -> None:
        ConnectionPool.__init__(self, host, port)
        RequestMethods.__init__(self, headers)
        if not isinstance(timeout, Timeout):
            timeout = Timeout.from_float(timeout)
        if retries is None:
            retries = Retry.DEFAULT
        self.timeout = timeout
        self.retries = retries
        self.pool = self.QueueCls(maxsize)
        self.block = block
        self.proxy = _proxy
        self.proxy_headers = _proxy_headers or {}
        for _ in xrange(maxsize):
            self.pool.put(None)
        self.num_connections = 0
        self.num_requests = 0
        self.conn_kw = conn_kw
        if self.proxy:
            self.conn_kw.setdefault('socket_options', [])

    def _new_conn(self) -> HTTP1Connection:
        self.num_connections += 1
        for kw in ('strict',):
            if kw in self.conn_kw:
                self.conn_kw.pop(kw)
        log.debug('Starting new HTTP connection (%d): %s:%s', self.num_connections, self.host, self.port or '80')
        conn = self.ConnectionCls(host=self.host, port=self.port, **self.conn_kw)
        return conn

    def _get_conn(self, timeout: Optional[float] = None) -> HTTP1Connection:
        conn = None
        try:
            conn = self.pool.get(block=self.block, timeout=timeout)
        except AttributeError:
            raise ClosedPoolError(self, 'Pool is closed.')
        except queue.Empty:
            if self.block:
                raise EmptyPoolError(self, 'Pool reached maximum size and no more connections are allowed.')
            pass
        if conn and is_connection_dropped(conn):
            log.debug('Resetting dropped connection: %s', self.host)
            conn.close()
        return conn or self._new_conn()

    def _put_conn(self, conn: Optional[HTTP1Connection]) -> None:
        try:
            self.pool.put(conn, block=False)
            return
        except AttributeError:
            pass
        except queue.Full:
            log.warning('Connection pool is full, discarding connection: %s', self.host)
        if conn:
            conn.close()

    def _start_conn(self, conn: HTTP1Connection, connect_timeout: Optional[float]) -> None:
        conn.connect(connect_timeout=connect_timeout)

    def _get_timeout(self, timeout: Union[Timeout, float, object]) -> Timeout:
        if timeout is _Default:
            return self.timeout.clone()
        if isinstance(timeout, Timeout):
            return timeout.clone()
        else:
            return Timeout.from_float(timeout)

    def _raise_timeout(self, err: Exception, url: str, timeout_value: Optional[float]) -> None:
        if isinstance(err, SocketTimeout):
            raise ReadTimeoutError(self, url, 'Read timed out. (read timeout=%s)' % timeout_value)
        if hasattr(err, 'errno') and err.errno in _blocking_errnos:
            raise ReadTimeoutError(self, url, 'Read timed out. (read timeout=%s)' % timeout_value)
        if 'timed out' in str(err) or 'did not complete (read)' in str(err):
            raise ReadTimeoutError(self, url, 'Read timed out. (read timeout=%s)' % timeout_value)

    def _make_request(self, conn: HTTP1Connection, method: str, url: str, timeout: Union[Timeout, float, object] = _Default, body: Optional[bytes] = None, headers: Optional[Dict[str, str]] = None) -> HTTPResponse:
        self.num_requests += 1
        timeout_obj = self._get_timeout(timeout)
        timeout_obj.start_connect()
        try:
            self._start_conn(conn, timeout_obj.connect_timeout)
        except (SocketTimeout, BaseSSLError) as e:
            self._raise_timeout(err=e, url=url, timeout_value=conn.timeout)
            raise
        request = Request(method=method, target=url, headers=headers, body=body)
        host = self.host
        port = self.port
        scheme = self.scheme
        request.add_host(host, port, scheme)
        read_timeout = timeout_obj.read_timeout
        if read_timeout == 0:
            raise ReadTimeoutError(self, url, 'Read timed out. (read timeout=%s)' % read_timeout)
        if read_timeout is Timeout.DEFAULT_TIMEOUT:
            read_timeout = socket.getdefaulttimeout()
        try:
            response = conn.send_request(request, read_timeout=read_timeout)
        except (SocketTimeout, BaseSSLError, SocketError) as e:
            self._raise_timeout(err=e, url=url, timeout_value=read_timeout)
            raise
        http_version = getattr(conn, '_http_vsn_str', 'HTTP/?')
        log.debug('%s://%s:%s "%s %s %s" %s', self.scheme, self.host, self.port, method, url, http_version, response.status_code)
        return response

    def _absolute_url(self, path: str) -> str:
        return Url(scheme=self.scheme, host=self.host, port=self.port, path=path).url

    def close(self) -> None:
        if self.pool is None:
            return
        old_pool, self.pool = (self.pool, None)
        try:
            while True:
                conn = old_pool.get(block=False)
                if conn:
                    conn.close()
        except queue.Empty:
            pass

    def is_same_host(self, url: str) -> bool:
        if url.startswith('/'):
            return True
        scheme, host, port = get_host(url)
        host = _ipv6_host(host).lower()
        if self.port and (not port):
            port = DEFAULT_PORTS.get(scheme)
        elif not self.port and port == DEFAULT_PORTS.get(scheme):
            port = None
        return (scheme, host, port) == (self.scheme, self.host, self.port)

    def urlopen(self, method: str, url: str, body: Optional[bytes] = None, headers: Optional[Dict[str, str]] = None, retries: Optional[Union[Retry, int, bool]] = None, timeout: Union[Timeout, float, object] = _Default, pool_timeout: Optional[float] = None, body_pos: Optional[int] = None, **response_kw: Any) -> HTTPResponse:
        if headers is None:
            headers = self.headers
        if not isinstance(retries, Retry):
            retries = Retry.from_int(retries, default=self.retries, redirect=False)
        conn = None
        release_this_conn = False
        if self.scheme == 'http':
            headers = headers.copy()
            headers.update(self.proxy_headers)
        err = None
        clean_exit = False
        body_pos = set_file_position(body, body_pos)
        if body is not None:
            _add_transport_headers(headers)
        try:
            timeout_obj = self._get_timeout(timeout)
            conn = self._get_conn(timeout=pool_timeout)
            conn.timeout = timeout_obj.connect_timeout
            base_response = self._make_request(conn, method, url, timeout=timeout_obj, body=body, headers=headers)
            response_kw['request_method'] = method
            response = self.ResponseCls.from_base(base_response, pool=self, retries=retries, **response_kw)
            clean_exit = True
        except queue.Empty:
            raise EmptyPoolError(self, 'No pool connections are available.')
        except (TimeoutError, SocketError, ProtocolError, h11.ProtocolError, BaseSSLError, SSLError, CertificateError) as e:
            clean_exit = False
            if isinstance(e, (BaseSSLError, CertificateError)):
                e = SSLError(e)
            elif isinstance(e, (SocketError, NewConnectionError)) and self.proxy:
                e = ProxyError('Cannot connect to proxy.', e)
            elif isinstance(e, (SocketError, h11.ProtocolError)):
                e = ProtocolError('Connection aborted.', e)
            retries = retries.increment(method, url, error=e, _pool=self, _stacktrace=sys.exc_info()[2])
            retries.sleep()
            err = e
        finally:
            if not clean_exit:
                conn = conn and conn.close()
                release_this_conn = True
            if release_this_conn:
                self._put_conn(conn)
        if not conn:
            log.warning("Retrying (%r) after connection broken by '%r': %s", retries, err, url)
            return self.urlopen(method, url, body, headers, retries, timeout=timeout, pool_timeout=pool_timeout, body_pos=body_pos, **response_kw)

        def drain_and_release_conn(response: HTTPResponse) -> None:
            try:
                response.read()
            except (TimeoutError, SocketError, ProtocolError, BaseSSLError, SSLError) as e:
                pass
        has_retry_after = bool(response.getheader('Retry-After'))
        if retries.is_retry(method, response.status, has_retry_after):
            try:
                retries = retries.increment(method, url, response=response, _pool=self)
            except MaxRetryError:
                if retries.raise_on_status:
                    drain_and_release_conn(response)
                    raise
                return response
            drain_and_release_conn(response)
            retries.sleep(response)
            log.debug('Retry: %s', url)
            return self.urlopen(method, url, body, headers, retries=retries, timeout=timeout, pool_timeout=pool_timeout, body_pos=body_pos, **response_kw)
        return response

class HTTPSConnectionPool(HTTPConnectionPool):
    scheme: str = 'https'

    def __init__(self, host: str, port: Optional[int] = None, timeout: Union[Timeout, float] = Timeout.DEFAULT_TIMEOUT, maxsize: int = 1, block: bool = False, headers: Optional[Dict[str, str]] = None, retries: Optional[Retry] = None, _proxy: Optional[Url] = None, _proxy_headers: Optional[Dict[str, str]] = None, key_file: Optional[str] = None, cert_file: Optional[str] = None, cert_reqs: Optional[str] = None, ca_certs: Optional[str] = None, ssl_version: Optional[str] = None, assert_hostname: Optional[Union[str, bool]] = None, assert_fingerprint: Optional[str] = None, ca_cert_dir: Optional[str] = None, ssl_context: Optional[ssl.SSLContext] = None, **conn_kw: Any) -> None:
        HTTPConnectionPool.__init__(self, host, port, timeout, maxsize, block, headers, retries, _proxy, _proxy_headers, **conn_kw)
        if ssl is None:
            raise SSLError('SSL module is not available')
        if ca_certs and cert_reqs is None:
            cert_reqs = 'CERT_REQUIRED'
        self.ssl_context = _build_context(ssl_context, keyfile=key_file, certfile=cert_file, cert_reqs=cert_reqs, ca_certs=ca_certs, ca_cert_dir=ca_cert_dir, ssl_version=ssl_version)
        self.assert_hostname = assert_hostname
        self.assert_fingerprint = assert_fingerprint

    def _new_conn(self) -> HTTP1Connection:
        self.num_connections += 1
        log.debug('Starting new HTTPS connection (%d): %s:%s', self.num_connections, self.host, self.port or '443')
        actual_host = self.host
        actual_port = self.port
        tunnel_host = None
        tunnel_port = None
        tunnel_headers = None
        if self.proxy is not None:
            actual_host = self.proxy.host
            actual_port = self.proxy.port
            tunnel_host = self.host
            tunnel_port = self.port
            tunnel_headers = self.proxy_headers
        for kw in ('strict', 'redirect'):
            if kw in self.conn_kw:
                self.conn_kw.pop(kw)
        conn = self.ConnectionCls(host=actual_host, port=actual_port, tunnel_host=tunnel_host, tunnel_port=tunnel_port, tunnel_headers=tunnel_headers, **self.conn_kw)
        return conn

    def _start_conn(self, conn: HTTP1Connection, connect_timeout: Optional[float]) -> None:
        conn.connect(ssl_context=self.ssl_context, fingerprint=self.assert_fingerprint, assert_hostname=self.assert_hostname, connect_timeout=connect_timeout)
        if not conn.is_verified:
            warnings.warn('Unverified HTTPS request is being made. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings', InsecureRequestWarning)

def connection_from_url(url: str, **kw: Any) -> ConnectionPool:
    scheme, host, port = get_host(url)
    port = port or DEFAULT_PORTS.get(scheme, 80)
    if scheme == 'https':
        return HTTPSConnectionPool(host, port=port, **kw)
    else:
        return HTTPConnectionPool(host, port=port, **kw)

def _ipv6_host(host: str) -> str:
    if host.startswith('[') and host.endswith(']'):
        host = host.replace('%25', '%').strip('[]')
    return host
