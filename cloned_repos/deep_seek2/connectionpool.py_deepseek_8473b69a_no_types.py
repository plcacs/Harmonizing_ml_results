from __future__ import absolute_import
import errno
import logging
import sys
import warnings
from socket import error as SocketError, timeout as SocketTimeout
import socket
import h11
from typing import Optional, Dict, Any, Union, List, Set, Tuple, Type, Callable, cast
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
_blocking_errnos: Set[int] = {errno.EAGAIN, errno.EWOULDBLOCK}

def _add_transport_headers(headers):
    transfer_headers = ('content-length', 'transfer-encoding')
    for header_name in headers:
        if header_name.lower() in transfer_headers:
            return
    headers['transfer-encoding'] = 'chunked'

def _build_context(context, keyfile, certfile, cert_reqs, ca_certs, ca_cert_dir, ssl_version):
    if context is None:
        context = create_urllib3_context(ssl_version=resolve_ssl_version(ssl_version), cert_reqs=resolve_cert_reqs(cert_reqs))
    context = merge_context_settings(context, keyfile=keyfile, certfile=certfile, cert_reqs=cert_reqs, ca_certs=ca_certs, ca_cert_dir=ca_cert_dir)
    return context

class ConnectionPool:
    scheme: Optional[str] = None
    QueueCls: Type[queue.Queue] = queue.LifoQueue

    def __init__(self, host, port=None):
        if not host:
            raise LocationValueError('No host specified.')
        self.host = _ipv6_host(host).lower()
        self.port = port

    def __str__(self):
        return '%s(host=%r, port=%r)' % (type(self).__name__, self.host, self.port)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        pass

class HTTPConnectionPool(ConnectionPool, RequestMethods):
    scheme: str = 'http'
    ConnectionCls: Type[HTTP1Connection] = HTTP1Connection
    ResponseCls: Type[HTTPResponse] = HTTPResponse

    def __init__(self, host, port=None, timeout=Timeout.DEFAULT_TIMEOUT, maxsize=1, block=False, headers=None, retries=None, _proxy=None, _proxy_headers=None, **conn_kw: Any):
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

    def _new_conn(self):
        self.num_connections += 1
        for kw in ('strict',):
            if kw in self.conn_kw:
                self.conn_kw.pop(kw)
        log.debug('Starting new HTTP connection (%d): %s:%s', self.num_connections, self.host, self.port or '80')
        conn = self.ConnectionCls(host=self.host, port=self.port, **self.conn_kw)
        return conn

    def _get_conn(self, timeout=None):
        conn = None
        try:
            conn = self.pool.get(block=self.block, timeout=timeout)
        except AttributeError:
            raise ClosedPoolError(self, 'Pool is closed.')
        except queue.Empty:
            if self.block:
                raise EmptyPoolError(self, 'Pool reached maximum size and no more connections are allowed.')
        if conn and is_connection_dropped(conn):
            log.debug('Resetting dropped connection: %s', self.host)
            conn.close()
        return conn or self._new_conn()

    def _put_conn(self, conn):
        try:
            self.pool.put(conn, block=False)
            return
        except AttributeError:
            pass
        except queue.Full:
            log.warning('Connection pool is full, discarding connection: %s', self.host)
        if conn:
            conn.close()

    def _start_conn(self, conn, connect_timeout):
        conn.connect(connect_timeout=connect_timeout)

    def _get_timeout(self, timeout):
        if timeout is _Default:
            return self.timeout.clone()
        if isinstance(timeout, Timeout):
            return timeout.clone()
        else:
            return Timeout.from_float(timeout)

    def _raise_timeout(self, err, url, timeout_value):
        if isinstance(err, SocketTimeout):
            raise ReadTimeoutError(self, url, 'Read timed out. (read timeout=%s)' % timeout_value)
        if hasattr(err, 'errno') and err.errno in _blocking_errnos:
            raise ReadTimeoutError(self, url, 'Read timed out. (read timeout=%s)' % timeout_value)
        if 'timed out' in str(err) or 'did not complete (read)' in str(err):
            raise ReadTimeoutError(self, url, 'Read timed out. (read timeout=%s)' % timeout_value)

    def _make_request(self, conn, method, url, timeout=_Default, body=None, headers=None):
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

    def _absolute_url(self, path):
        return Url(scheme=self.scheme, host=self.host, port=self.port, path=path).url

    def close(self):
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

    def is_same_host(self, url):
        if url.startswith('/'):
            return True
        scheme, host, port = get_host(url)
        host = _ipv6_host(host).lower()
        if self.port and (not port):
            port = DEFAULT_PORTS.get(scheme)
        elif not self.port and port == DEFAULT_PORTS.get(scheme):
            port = None
        return (scheme, host, port) == (self.scheme, self.host, self.port)

    def urlopen(self, method, url, body=None, headers=None, retries=None, timeout=_Default, pool_timeout=None, body_pos=None, **response_kw: Any):
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

        def drain_and_release_conn(response):
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

    def __init__(self, host, port=None, timeout=Timeout.DEFAULT_TIMEOUT, maxsize=1, block=False, headers=None, retries=None, _proxy=None, _proxy_headers=None, key_file=None, cert_file=None, cert_reqs=None, ca_certs=None, ssl_version=None, assert_hostname=None, assert_fingerprint=None, ca_cert_dir=None, ssl_context=None, **conn_kw: Any):
        HTTPConnectionPool.__init__(self, host, port, timeout, maxsize, block, headers, retries, _proxy, _proxy_headers, **conn_kw)
        if ssl is None:
            raise SSLError('SSL module is not available')
        if ca_certs and cert_reqs is None:
            cert_reqs = 'CERT_REQUIRED'
        self.ssl_context = _build_context(ssl_context, keyfile=key_file, certfile=cert_file, cert_reqs=cert_reqs, ca_certs=ca_certs, ca_cert_dir=ca_cert_dir, ssl_version=ssl_version)
        self.assert_hostname = assert_hostname
        self.assert_fingerprint = assert_fingerprint

    def _new_conn(self):
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

    def _start_conn(self, conn, connect_timeout):
        conn.connect(ssl_context=self.ssl_context, fingerprint=self.assert_fingerprint, assert_hostname=self.assert_hostname, connect_timeout=connect_timeout)
        if not conn.is_verified:
            warnings.warn('Unverified HTTPS request is being made. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings', InsecureRequestWarning)

def connection_from_url(url, **kw: Any):
    scheme, host, port = get_host(url)
    port = port or DEFAULT_PORTS.get(scheme, 80)
    if scheme == 'https':
        return HTTPSConnectionPool(host, port=port, **kw)
    else:
        return HTTPConnectionPool(host, port=port, **kw)

def _ipv6_host(host):
    if host.startswith('[') and host.endswith(']'):
        host = host.replace('%25', '%').strip('[]')
    return host