from __future__ import absolute_import
import OpenSSL.SSL
from cryptography import x509
from cryptography.hazmat.backends.openssl import backend as openssl_backend
from cryptography.hazmat.backends.openssl.x509 import _Certificate
from socket import timeout, error as SocketError
from io import BytesIO
from typing import Optional, Dict, Tuple, List, Union, Any, Callable, TypeVar, cast
try:
    from socket import _fileobject
except ImportError:
    _fileobject = None
    from ..packages.backports.makefile import backport_makefile
import logging
import ssl
from ..packages import six
import sys
from .. import util

__all__ = ['inject_into_urllib3', 'extract_from_urllib3']

HAS_SNI: bool = True
_openssl_versions: Dict[int, int] = {
    ssl.PROTOCOL_SSLv23: OpenSSL.SSL.SSLv23_METHOD,
    ssl.PROTOCOL_TLSv1: OpenSSL.SSL.TLSv1_METHOD
}

if hasattr(ssl, 'PROTOCOL_TLSv1_1') and hasattr(OpenSSL.SSL, 'TLSv1_1_METHOD'):
    _openssl_versions[ssl.PROTOCOL_TLSv1_1] = OpenSSL.SSL.TLSv1_1_METHOD
if hasattr(ssl, 'PROTOCOL_TLSv1_2') and hasattr(OpenSSL.SSL, 'TLSv1_2_METHOD'):
    _openssl_versions[ssl.PROTOCOL_TLSv1_2] = OpenSSL.SSL.TLSv1_2_METHOD
try:
    _openssl_versions.update({ssl.PROTOCOL_SSLv3: OpenSSL.SSL.SSLv3_METHOD})
except AttributeError:
    pass

_stdlib_to_openssl_verify: Dict[int, int] = {
    ssl.CERT_NONE: OpenSSL.SSL.VERIFY_NONE,
    ssl.CERT_OPTIONAL: OpenSSL.SSL.VERIFY_PEER,
    ssl.CERT_REQUIRED: OpenSSL.SSL.VERIFY_PEER + OpenSSL.SSL.VERIFY_FAIL_IF_NO_PEER_CERT
}
_openssl_to_stdlib_verify: Dict[int, int] = dict(((v, k) for k, v in _stdlib_to_openssl_verify.items()))
SSL_WRITE_BLOCKSIZE: int = 16384
orig_util_HAS_SNI: bool = util.HAS_SNI
orig_util_SSLContext: Any = util.ssl_.SSLContext
log: logging.Logger = logging.getLogger(__name__)

def inject_into_urllib3() -> None:
    """Monkey-patch urllib3 with PyOpenSSL-backed SSL-support."""
    _validate_dependencies_met()
    util.ssl_.SSLContext = PyOpenSSLContext
    util.HAS_SNI = HAS_SNI
    util.ssl_.HAS_SNI = HAS_SNI
    util.IS_PYOPENSSL = True
    util.ssl_.IS_PYOPENSSL = True

def extract_from_urllib3() -> None:
    """Undo monkey-patching by :func:`inject_into_urllib3`."""
    util.ssl_.SSLContext = orig_util_SSLContext
    util.HAS_SNI = orig_util_HAS_SNI
    util.ssl_.HAS_SNI = orig_util_HAS_SNI
    util.IS_PYOPENSSL = False
    util.ssl_.IS_PYOPENSSL = False

def _validate_dependencies_met() -> None:
    """
    Verifies that PyOpenSSL's package-level dependencies have been met.
    Throws `ImportError` if they are not met.
    """
    from cryptography.x509.extensions import Extensions
    if getattr(Extensions, 'get_extension_for_class', None) is None:
        raise ImportError("'cryptography' module missing required functionality.  Try upgrading to v1.3.4 or newer.")
    from OpenSSL.crypto import X509
    x509 = X509()
    if getattr(x509, '_x509', None) is None:
        raise ImportError("'pyOpenSSL' module missing required functionality. Try upgrading to v0.14 or newer.")

def _dnsname_to_stdlib(name: str) -> str:
    """
    Converts a dNSName SubjectAlternativeName field to the form used by the
    standard library on the given Python version.
    """
    def idna_encode(name: str) -> bytes:
        """
        Borrowed wholesale from the Python Cryptography Project. It turns out
        that we can't just safely call `idna.encode`: it can explode for
        wildcard names. This avoids that problem.
        """
        import idna
        for prefix in [u'*.', u'.']:
            if name.startswith(prefix):
                name = name[len(prefix):]
                return prefix.encode('ascii') + idna.encode(name)
        return idna.encode(name)
    
    name_bytes = idna_encode(name)
    if sys.version_info >= (3, 0):
        return name_bytes.decode('utf-8')
    return name_bytes

def get_subj_alt_name(peer_cert: Any) -> List[Tuple[str, str]]:
    """
    Given an PyOpenSSL certificate, provides all the subject alternative names.
    """
    if hasattr(peer_cert, 'to_cryptography'):
        cert = peer_cert.to_cryptography()
    else:
        cert = _Certificate(openssl_backend, peer_cert._x509)
    try:
        ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except x509.ExtensionNotFound:
        return []
    except (x509.DuplicateExtension, x509.UnsupportedExtension, x509.UnsupportedGeneralNameType, UnicodeError) as e:
        log.warning('A problem was encountered with the certificate that prevented urllib3 from finding the SubjectAlternativeName field. This can affect certificate validation. The error was %s', e)
        return []
    names = [('DNS', _dnsname_to_stdlib(name)) for name in ext.get_values_for_type(x509.DNSName)]
    names.extend((('IP Address', str(name)) for name in ext.get_values_for_type(x509.IPAddress)))
    return names

class WrappedSocket(object):
    """API-compatibility wrapper for Python OpenSSL's Connection-class."""

    def __init__(self, connection: OpenSSL.SSL.Connection, socket: Any, suppress_ragged_eofs: bool = True) -> None:
        self.connection = connection
        self.socket = socket
        self.suppress_ragged_eofs = suppress_ragged_eofs
        self._makefile_refs = 0
        self._closed = False

    def fileno(self) -> int:
        return self.socket.fileno()

    def _decref_socketios(self) -> None:
        if self._makefile_refs > 0:
            self._makefile_refs -= 1
        if self._closed:
            self.close()

    def recv(self, *args: Any, **kwargs: Any) -> bytes:
        try:
            data = self.connection.recv(*args, **kwargs)
        except OpenSSL.SSL.SysCallError as e:
            if self.suppress_ragged_eofs and e.args == (-1, 'Unexpected EOF'):
                return b''
            else:
                raise SocketError(str(e))
        except OpenSSL.SSL.ZeroReturnError as e:
            if self.connection.get_shutdown() == OpenSSL.SSL.RECEIVED_SHUTDOWN:
                return b''
            else:
                raise
        except OpenSSL.SSL.WantReadError:
            rd = util.wait_for_read(self.socket, self.socket.gettimeout())
            if not rd:
                raise timeout('The read operation timed out')
            else:
                return self.recv(*args, **kwargs)
        else:
            return data

    def recv_into(self, *args: Any, **kwargs: Any) -> int:
        try:
            return self.connection.recv_into(*args, **kwargs)
        except OpenSSL.SSL.SysCallError as e:
            if self.suppress_ragged_eofs and e.args == (-1, 'Unexpected EOF'):
                return 0
            else:
                raise SocketError(str(e))
        except OpenSSL.SSL.ZeroReturnError as e:
            if self.connection.get_shutdown() == OpenSSL.SSL.RECEIVED_SHUTDOWN:
                return 0
            else:
                raise
        except OpenSSL.SSL.WantReadError:
            rd = util.wait_for_read(self.socket, self.socket.gettimeout())
            if not rd:
                raise timeout('The read operation timed out')
            else:
                return self.recv_into(*args, **kwargs)

    def settimeout(self, timeout: float) -> None:
        return self.socket.settimeout(timeout)

    def _send_until_done(self, data: bytes) -> int:
        while True:
            try:
                return self.connection.send(data)
            except OpenSSL.SSL.WantWriteError:
                wr = util.wait_for_write(self.socket, self.socket.gettimeout())
                if not wr:
                    raise timeout()
                continue
            except OpenSSL.SSL.SysCallError as e:
                raise SocketError(str(e))

    def send(self, data: bytes) -> int:
        return self._send_until_done(data)

    def sendall(self, data: bytes) -> None:
        total_sent = 0
        while total_sent < len(data):
            sent = self._send_until_done(data[total_sent:total_sent + SSL_WRITE_BLOCKSIZE])
            total_sent += sent

    def shutdown(self) -> None:
        self.connection.shutdown()

    def close(self) -> None:
        if self._makefile_refs < 1:
            try:
                self._closed = True
                return self.connection.close()
            except OpenSSL.SSL.Error:
                return
        else:
            self._makefile_refs -= 1

    def getpeercert(self, binary_form: bool = False) -> Optional[Union[bytes, Dict[str, Any]]]:
        x509 = self.connection.get_peer_certificate()
        if not x509:
            return x509
        if binary_form:
            return OpenSSL.crypto.dump_certificate(OpenSSL.crypto.FILETYPE_ASN1, x509)
        return {
            'subject': ((('commonName', x509.get_subject().CN),),),
            'subjectAltName': get_subj_alt_name(x509)
        }

    def setblocking(self, flag: bool) -> None:
        return self.connection.setblocking(flag)

    def _reuse(self) -> None:
        self._makefile_refs += 1

    def _drop(self) -> None:
        if self._makefile_refs < 1:
            self.close()
        else:
            self._makefile_refs -= 1

if _fileobject:
    def makefile(self: Any, mode: str, bufsize: int = -1) -> Any:
        self._makefile_refs += 1
        return _fileobject(self, mode, bufsize, close=True)
else:
    makefile = backport_makefile

WrappedSocket.makefile = makefile

class PyOpenSSLContext(object):
    """
    I am a wrapper class for the PyOpenSSL ``Context`` object.
    """

    def __init__(self, protocol: int) -> None:
        self.protocol = _openssl_versions[protocol]
        self._ctx = OpenSSL.SSL.Context(self.protocol)
        self._options = 0
        self.check_hostname = False

    @property
    def options(self) -> int:
        return self._options

    @options.setter
    def options(self, value: int) -> None:
        self._options = value
        self._ctx.set_options(value)

    @property
    def verify_mode(self) -> int:
        return _openssl_to_stdlib_verify[self._ctx.get_verify_mode()]

    @verify_mode.setter
    def verify_mode(self, value: int) -> None:
        self._ctx.set_verify(_stdlib_to_openssl_verify[value], _verify_callback)

    def set_default_verify_paths(self) -> None:
        self._ctx.set_default_verify_paths()

    def set_ciphers(self, ciphers: Union[str, bytes]) -> None:
        if isinstance(ciphers, six.text_type):
            ciphers = ciphers.encode('utf-8')
        self._ctx.set_cipher_list(ciphers)

    def load_verify_locations(self, cafile: Optional[str] = None, capath: Optional[str] = None, cadata: Optional[bytes] = None) -> None:
        if cafile is not None:
            cafile = cafile.encode('utf-8')
        if capath is not None:
            capath = capath.encode('utf-8')
        self._ctx.load_verify_locations(cafile, capath)
        if cadata is not None:
            self._ctx.load_verify_locations(BytesIO(cadata))

    def load_cert_chain(self, certfile: str, keyfile: Optional[str] = None, password: Optional[str] = None) -> None:
        self._ctx.use_certificate_chain_file(certfile)
        if password is not None:
            self._ctx.set_passwd_cb(lambda max_length, prompt_twice, userdata: password)
        self._ctx.use_privatekey_file(keyfile or certfile)

    def wrap_socket(
        self,
        sock: Any,
        server_side: bool = False,
        do_handshake_on_connect: bool = True,
        suppress_ragged_eofs: bool = True,
        server_hostname: Optional[str] = None
    ) -> WrappedSocket:
        cnx = OpenSSL.SSL.Connection(self._ctx, sock)
        if isinstance(server_hostname, six.text_type):
            server_hostname = server_hostname.encode('utf-8')
        if server_hostname is not None:
            cnx.set_tlsext_host_name(server_hostname)
        cnx.set_connect_state()
        while True:
            try:
                cnx.do_handshake()
            except OpenSSL.SSL.WantReadError:
                rd = util.wait_for_read(sock, sock.gettimeout())
                if not rd:
                    raise timeout('select timed out')
                continue
            except OpenSSL.SSL.Error as e:
                raise ssl.SSLError('bad handshake: %r' % e)
            break
        return WrappedSocket(cnx, sock)

def _verify_callback(cnx: Any, x509: Any, err_no: int, err_depth: int, return_code: int) -> bool:
    return err_no == 0
