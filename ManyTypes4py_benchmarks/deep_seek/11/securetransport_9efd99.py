from __future__ import absolute_import
import contextlib
import ctypes
import errno
import os.path
import shutil
import socket
import ssl
import threading
import weakref
from typing import Any, BinaryIO, Dict, Generator, List, Optional, Tuple, Type, TypeVar, Union, cast
from .. import util
from ._securetransport.bindings import Security, SecurityConst, CoreFoundation
from ._securetransport.low_level import _assert_no_error, _cert_array_from_pem, _temporary_keychain, _load_client_cert_chain

try:
    from socket import _fileobject
except ImportError:
    _fileobject = None
    from ..packages.backports.makefile import backport_makefile

try:
    memoryview(b'')
except NameError:
    raise ImportError('SecureTransport only works on Pythons with memoryview')

__all__ = ['inject_into_urllib3', 'extract_from_urllib3']

HAS_SNI = True
orig_util_HAS_SNI = util.HAS_SNI
orig_util_SSLContext = util.ssl_.SSLContext
_connection_refs = weakref.WeakValueDictionary()  # type: weakref.WeakValueDictionary[int, 'WrappedSocket']
_connection_ref_lock = threading.Lock()
SSL_WRITE_BLOCKSIZE = 16384

CIPHER_SUITES = [
    SecurityConst.TLS_AES_256_GCM_SHA384,
    SecurityConst.TLS_CHACHA20_POLY1305_SHA256,
    SecurityConst.TLS_AES_128_GCM_SHA256,
    SecurityConst.TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
    SecurityConst.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
    SecurityConst.TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
    SecurityConst.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
    SecurityConst.TLS_DHE_DSS_WITH_AES_256_GCM_SHA384,
    SecurityConst.TLS_DHE_RSA_WITH_AES_256_GCM_SHA384,
    SecurityConst.TLS_DHE_DSS_WITH_AES_128_GCM_SHA256,
    SecurityConst.TLS_DHE_RSA_WITH_AES_128_GCM_SHA256,
    SecurityConst.TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA384,
    SecurityConst.TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384,
    SecurityConst.TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA,
    SecurityConst.TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA,
    SecurityConst.TLS_DHE_RSA_WITH_AES_256_CBC_SHA256,
    SecurityConst.TLS_DHE_DSS_WITH_AES_256_CBC_SHA256,
    SecurityConst.TLS_DHE_RSA_WITH_AES_256_CBC_SHA,
    SecurityConst.TLS_DHE_DSS_WITH_AES_256_CBC_SHA,
    SecurityConst.TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA256,
    SecurityConst.TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256,
    SecurityConst.TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA,
    SecurityConst.TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA,
    SecurityConst.TLS_DHE_RSA_WITH_AES_128_CBC_SHA256,
    SecurityConst.TLS_DHE_DSS_WITH_AES_128_CBC_SHA256,
    SecurityConst.TLS_DHE_RSA_WITH_AES_128_CBC_SHA,
    SecurityConst.TLS_DHE_DSS_WITH_AES_128_CBC_SHA,
    SecurityConst.TLS_RSA_WITH_AES_256_GCM_SHA384,
    SecurityConst.TLS_RSA_WITH_AES_128_GCM_SHA256,
    SecurityConst.TLS_RSA_WITH_AES_256_CBC_SHA256,
    SecurityConst.TLS_RSA_WITH_AES_128_CBC_SHA256,
    SecurityConst.TLS_RSA_WITH_AES_256_CBC_SHA,
    SecurityConst.TLS_RSA_WITH_AES_128_CBC_SHA,
]

_protocol_to_min_max = {
    ssl.PROTOCOL_SSLv23: (SecurityConst.kTLSProtocol1, SecurityConst.kTLSProtocol12)
}  # type: Dict[int, Tuple[int, int]]

if hasattr(ssl, 'PROTOCOL_SSLv2'):
    _protocol_to_min_max[ssl.PROTOCOL_SSLv2] = (SecurityConst.kSSLProtocol2, SecurityConst.kSSLProtocol2)
if hasattr(ssl, 'PROTOCOL_SSLv3'):
    _protocol_to_min_max[ssl.PROTOCOL_SSLv3] = (SecurityConst.kSSLProtocol3, SecurityConst.kSSLProtocol3)
if hasattr(ssl, 'PROTOCOL_TLSv1'):
    _protocol_to_min_max[ssl.PROTOCOL_TLSv1] = (SecurityConst.kTLSProtocol1, SecurityConst.kTLSProtocol1)
if hasattr(ssl, 'PROTOCOL_TLSv1_1'):
    _protocol_to_min_max[ssl.PROTOCOL_TLSv1_1] = (SecurityConst.kTLSProtocol11, SecurityConst.kTLSProtocol11)
if hasattr(ssl, 'PROTOCOL_TLSv1_2'):
    _protocol_to_min_max[ssl.PROTOCOL_TLSv1_2] = (SecurityConst.kTLSProtocol12, SecurityConst.kTLSProtocol12)
if hasattr(ssl, 'PROTOCOL_TLS'):
    _protocol_to_min_max[ssl.PROTOCOL_TLS] = _protocol_to_min_max[ssl.PROTOCOL_SSLv23]

def inject_into_urllib3() -> None:
    """Monkey-patch urllib3 with SecureTransport-backed SSL-support."""
    util.ssl_.SSLContext = SecureTransportContext
    util.HAS_SNI = HAS_SNI
    util.ssl_.HAS_SNI = HAS_SNI
    util.IS_SECURETRANSPORT = True
    util.ssl_.IS_SECURETRANSPORT = True

def extract_from_urllib3() -> None:
    """Undo monkey-patching by :func:`inject_into_urllib3`."""
    util.ssl_.SSLContext = orig_util_SSLContext
    util.HAS_SNI = orig_util_HAS_SNI
    util.ssl_.HAS_SNI = orig_util_HAS_SNI
    util.IS_SECURETRANSPORT = False
    util.ssl_.IS_SECURETRANSPORT = False

def _read_callback(connection_id: int, data_buffer: ctypes.c_void_p, data_length_pointer: ctypes.POINTER[ctypes.c_size_t]) -> int:
    """SecureTransport read callback."""
    wrapped_socket = None
    try:
        wrapped_socket = _connection_refs.get(connection_id)
        if wrapped_socket is None:
            return SecurityConst.errSSLInternal
        base_socket = wrapped_socket.socket
        requested_length = data_length_pointer[0]
        timeout = wrapped_socket.gettimeout()
        error = None
        read_count = 0
        buffer = (ctypes.c_char * requested_length).from_address(data_buffer)
        buffer_view = memoryview(buffer)
        try:
            while read_count < requested_length:
                if timeout is None or timeout >= 0:
                    readables = util.wait_for_read([base_socket], timeout)
                    if not readables:
                        raise socket.error(errno.EAGAIN, 'timed out')
                chunk_size = base_socket.recv_into(buffer_view[read_count:requested_length])
                read_count += chunk_size
                if not chunk_size:
                    if not read_count:
                        return SecurityConst.errSSLClosedGraceful
                    break
        except socket.error as e:
            error = e.errno
            if error is not None and error != errno.EAGAIN:
                if error == errno.ECONNRESET:
                    return SecurityConst.errSSLClosedAbort
                raise
        data_length_pointer[0] = read_count
        if read_count != requested_length:
            return SecurityConst.errSSLWouldBlock
        return 0
    except Exception as e:
        if wrapped_socket is not None:
            wrapped_socket._exception = e
        return SecurityConst.errSSLInternal

def _write_callback(connection_id: int, data_buffer: ctypes.c_void_p, data_length_pointer: ctypes.POINTER[ctypes.c_size_t]) -> int:
    """SecureTransport write callback."""
    wrapped_socket = None
    try:
        wrapped_socket = _connection_refs.get(connection_id)
        if wrapped_socket is None:
            return SecurityConst.errSSLInternal
        base_socket = wrapped_socket.socket
        bytes_to_write = data_length_pointer[0]
        data = ctypes.string_at(data_buffer, bytes_to_write)
        timeout = wrapped_socket.gettimeout()
        error = None
        sent = 0
        try:
            while sent < bytes_to_write:
                if timeout is None or timeout >= 0:
                    writables = util.wait_for_write([base_socket], timeout)
                    if not writables:
                        raise socket.error(errno.EAGAIN, 'timed out')
                chunk_sent = base_socket.send(data)
                sent += chunk_sent
                data = data[chunk_sent:]
        except socket.error as e:
            error = e.errno
            if error is not None and error != errno.EAGAIN:
                if error == errno.ECONNRESET:
                    return SecurityConst.errSSLClosedAbort
                raise
        data_length_pointer[0] = sent
        if sent != bytes_to_write:
            return SecurityConst.errSSLWouldBlock
        return 0
    except Exception as e:
        if wrapped_socket is not None:
            wrapped_socket._exception = e
        return SecurityConst.errSSLInternal

_read_callback_pointer = Security.SSLReadFunc(_read_callback)
_write_callback_pointer = Security.SSLWriteFunc(_write_callback)

class WrappedSocket(object):
    """API-compatibility wrapper for Python's OpenSSL wrapped socket object."""

    def __init__(self, socket: socket.socket) -> None:
        self.socket = socket
        self.context = None  # type: Optional[ctypes.c_void_p]
        self._makefile_refs = 0
        self._closed = False
        self._exception = None  # type: Optional[Exception]
        self._keychain = None  # type: Optional[ctypes.c_void_p]
        self._keychain_dir = None  # type: Optional[str]
        self._client_cert_chain = None  # type: Optional[ctypes.c_void_p]
        self._timeout = self.socket.gettimeout()
        self.socket.settimeout(0)

    @contextlib.contextmanager
    def _raise_on_error(self) -> Generator[None, None, None]:
        """Context manager to propagate SecureTransport I/O exceptions."""
        self._exception = None
        yield
        if self._exception is not None:
            exception, self._exception = self._exception, None
            self.close()
            raise exception

    def _set_ciphers(self) -> None:
        """Set up the allowed ciphers."""
        ciphers = (Security.SSLCipherSuite * len(CIPHER_SUITES))(*CIPHER_SUITES)
        result = Security.SSLSetEnabledCiphers(self.context, ciphers, len(CIPHER_SUITES))
        _assert_no_error(result)

    def _custom_validate(self, verify: bool, trust_bundle: Optional[Union[str, bytes]]) -> None:
        """Custom certificate validation."""
        if not verify:
            return
        if isinstance(trust_bundle, str) and os.path.isfile(trust_bundle):
            with open(trust_bundle, 'rb') as f:
                trust_bundle = f.read()
        cert_array = None  # type: Optional[ctypes.c_void_p]
        trust = Security.SecTrustRef()
        try:
            cert_array = _cert_array_from_pem(trust_bundle)
            result = Security.SSLCopyPeerTrust(self.context, ctypes.byref(trust))
            _assert_no_error(result)
            if not trust:
                raise ssl.SSLError('Failed to copy trust reference')
            result = Security.SecTrustSetAnchorCertificates(trust, cert_array)
            _assert_no_error(result)
            result = Security.SecTrustSetAnchorCertificatesOnly(trust, True)
            _assert_no_error(result)
            trust_result = Security.SecTrustResultType()
            result = Security.SecTrustEvaluate(trust, ctypes.byref(trust_result))
            _assert_no_error(result)
        finally:
            if trust:
                CoreFoundation.CFRelease(trust)
            if cert_array is not None:
                CoreFoundation.CFRelease(cert_array)
        successes = (SecurityConst.kSecTrustResultUnspecified, SecurityConst.kSecTrustResultProceed)
        if trust_result.value not in successes:
            raise ssl.SSLError('certificate verify failed, error code: %d' % trust_result.value)

    def handshake(
        self,
        server_hostname: Optional[str],
        verify: bool,
        trust_bundle: Optional[Union[str, bytes]],
        min_version: int,
        max_version: int,
        client_cert: Optional[str],
        client_key: Optional[str],
        client_key_passphrase: Optional[str]
    ) -> None:
        """Perform the TLS handshake."""
        self.context = Security.SSLCreateContext(None, SecurityConst.kSSLClientSide, SecurityConst.kSSLStreamType)
        result = Security.SSLSetIOFuncs(self.context, _read_callback_pointer, _write_callback_pointer)
        _assert_no_error(result)
        with _connection_ref_lock:
            handle = id(self) % 2147483647
            while handle in _connection_refs:
                handle = (handle + 1) % 2147483647
            _connection_refs[handle] = self
        result = Security.SSLSetConnection(self.context, handle)
        _assert_no_error(result)
        if server_hostname:
            if not isinstance(server_hostname, bytes):
                server_hostname = server_hostname.encode('utf-8')
            result = Security.SSLSetPeerDomainName(self.context, server_hostname, len(server_hostname))
            _assert_no_error(result)
        self._set_ciphers()
        result = Security.SSLSetProtocolVersionMin(self.context, min_version)
        _assert_no_error(result)
        result = Security.SSLSetProtocolVersionMax(self.context, max_version)
        _assert_no_error(result)
        if not verify or trust_bundle is not None:
            result = Security.SSLSetSessionOption(self.context, SecurityConst.kSSLSessionOptionBreakOnServerAuth, True)
            _assert_no_error(result)
        if client_cert:
            self._keychain, self._keychain_dir = _temporary_keychain()
            self._client_cert_chain = _load_client_cert_chain(self._keychain, client_cert, client_key)
            result = Security.SSLSetCertificate(self.context, self._client_cert_chain)
            _assert_no_error(result)
        while True:
            with self._raise_on_error():
                result = Security.SSLHandshake(self.context)
                if result == SecurityConst.errSSLWouldBlock:
                    raise socket.timeout('handshake timed out')
                elif result == SecurityConst.errSSLServerAuthCompleted:
                    self._custom_validate(verify, trust_bundle)
                    continue
                else:
                    _assert_no_error(result)
                    break

    def fileno(self) -> int:
        return self.socket.fileno()

    def _decref_socketios(self) -> None:
        if self._makefile_refs > 0:
            self._makefile_refs -= 1
        if self._closed:
            self.close()

    def recv(self, bufsiz: int) -> bytes:
        buffer = ctypes.create_string_buffer(bufsiz)
        bytes_read = self.recv_into(buffer, bufsiz)
        data = buffer[:bytes_read]
        return data

    def recv_into(self, buffer: Union[bytearray, memoryview], nbytes: Optional[int] = None) -> int:
        if self._closed:
            return 0
        if nbytes is None:
            nbytes = len(buffer)
        buffer = (ctypes.c_char * nbytes).from_buffer(buffer)
        processed_bytes = ctypes.c_size_t(0)
        with self._raise_on_error():
            result = Security.SSLRead(self.context, buffer, nbytes, ctypes.byref(processed_bytes))
        if result == SecurityConst.errSSLWouldBlock:
            if processed_bytes.value == 0:
                raise socket.timeout('recv timed out')
        elif result in (SecurityConst.errSSLClosedGraceful, SecurityConst.errSSLClosedNoNotify):
            self.close()
        else:
            _assert_no_error(result)
        return processed_bytes.value

    def settimeout(self, timeout: Optional[float]) -> None:
        self._timeout = timeout

    def gettimeout(self) -> Optional[float]:
        return self._timeout

    def send(self, data: bytes) -> int:
        processed_bytes = ctypes.c_size_t(0)
        with self._raise_on_error():
            result = Security.SSLWrite(self.context, data, len(data), ctypes.byref(processed_bytes))
        if result == SecurityConst.errSSLWouldBlock and processed_bytes.value == 0:
            raise socket.timeout('send timed out')
        else:
            _assert_no_error(result)
        return processed_bytes.value

    def sendall(self, data: bytes) -> None:
        total_sent = 0
        while total_sent < len(data):
            sent = self.send(data[total_sent:total_sent + SSL_WRITE_BLOCKSIZE])
            total_sent += sent

    def shutdown(self) -> None:
        with self._raise_on_error():
            Security.SSLClose(self.context)

    def close(self) -> None:
        if self._makefile_refs <