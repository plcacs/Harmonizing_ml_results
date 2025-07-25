"""
SSL with SNI_-support for Python 2. Follow these instructions if you would
like to verify SSL certificates in Python 2. Note, the default libraries do
*not* do certificate checking; you need to do additional work to validate
certificates yourself.

This needs the following packages installed:

* pyOpenSSL (tested with 16.0.0)
* cryptography (minimum 1.3.4, from pyopenssl)
* idna (minimum 2.0, from cryptography)

However, pyopenssl depends on cryptography, which depends on idna, so while we
use all three directly here we end up having relatively few packages required.

You can install them with the following command:

    pip install pyopenssl cryptography idna

To activate certificate checking, call
:func:`~urllib3.contrib.pyopenssl.inject_into_urllib3` from your Python code
before you begin making HTTP requests. This can be done in a ``sitecustomize``
module, or at any other time before your application begins using ``urllib3``,
like this::

    try:
        import urllib3.contrib.pyopenssl
        urllib3.contrib.pyopenssl.inject_into_urllib3()
    except ImportError:
        pass

Now you can use :mod:`urllib3` as you normally would, and it will support SNI
when the required modules are installed.

Activating this module also has the positive side effect of disabling SSL/TLS
compression in Python 2 (see `CRIME attack`_).

If you want to configure the default list of supported cipher suites, you can
set the ``urllib3.contrib.pyopenssl.DEFAULT_SSL_CIPHER_LIST`` variable.

.. _sni: https://en.wikipedia.org/wiki/Server_Name_Indication
.. _crime attack: https://en.wikipedia.org/wiki/CRIME_(security_exploit)
"""
from __future__ import absolute_import
import logging
import ssl
import sys
from io import BytesIO
from socket import timeout, error as SocketError
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import OpenSSL.SSL
from cryptography import x509
from cryptography.hazmat.backends.openssl import backend as openssl_backend
from cryptography.hazmat.backends.openssl.x509 import _Certificate

from .. import util
from ..packages import six

try:
    from socket import _fileobject
except ImportError:
    _fileobject = None
    from ..packages.backports.makefile import backport_makefile

__all__ = ['inject_into_urllib3', 'extract_from_urllib3']
HAS_SNI: bool = True

_openssl_versions: Dict[int, Any] = {
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
    ssl.CERT_REQUIRED: OpenSSL.SSL.VERIFY_PEER | OpenSSL.SSL.VERIFY_FAIL_IF_NO_PEER_CERT
}

_openssl_to_stdlib_verify: Dict[int, int] = {v: k for k, v in _stdlib_to_openssl_verify.items()}

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
    from cryptography.x509.extensions import Extensions  # type: ignore

    if getattr(Extensions, 'get_extension_for_class', None) is None:
        raise ImportError("'cryptography' module missing required functionality.  Try upgrading to v1.3.4 or newer.")
    
    from OpenSSL.crypto import X509  # type: ignore

    x509_instance = X509()
    if getattr(x509_instance, '_x509', None) is None:
        raise ImportError("'pyOpenSSL' module missing required functionality. Try upgrading to v0.14 or newer.")


def _dnsname_to_stdlib(name: str) -> str:
    """
    Converts a dNSName SubjectAlternativeName field to the form used by the
    standard library on the given Python version.

    Cryptography produces a dNSName as a unicode string that was idna-decoded
    from ASCII bytes. We need to idna-encode that string to get it back, and
    then on Python 3 we also need to convert to unicode via UTF-8 (the stdlib
    uses PyUnicode_FromStringAndSize on it, which decodes via UTF-8).
    """

    def idna_encode(name_inner: str) -> bytes:
        """
        Borrowed wholesale from the Python Cryptography Project. It turns out
        that we can't just safely call `idna.encode`: it can explode for
        wildcard names. This avoids that problem.
        """
        import idna
        for prefix in [u'*.', u'.']:
            if name_inner.startswith(prefix):
                name_inner_trimmed = name_inner[len(prefix):]
                return prefix.encode('ascii') + idna.encode(name_inner_trimmed)
        return idna.encode(name_inner)

    name_encoded: bytes = idna_encode(name)
    if sys.version_info >= (3, 0):
        name_encoded = name_encoded.decode('utf-8')
    return name_encoded


def get_subj_alt_name(peer_cert: Any) -> List[Tuple[str, str]]:
    """
    Given an PyOpenSSL certificate, provides all the subject alternative names.
    """
    if hasattr(peer_cert, 'to_cryptography'):
        cert = peer_cert.to_cryptography()
    else:
        cert = _Certificate(openssl_backend, peer_cert._x509)  # type: ignore

    try:
        ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except x509.ExtensionNotFound:
        return []
    except (
        x509.DuplicateExtension,
        x509.UnsupportedExtension,
        x509.UnsupportedGeneralNameType,
        UnicodeError
    ) as e:
        log.warning(
            'A problem was encountered with the certificate that prevented urllib3 from finding the SubjectAlternativeName field. This can affect certificate validation. The error was %s',
            e
        )
        return []
    
    names: List[Tuple[str, str]] = [
        ('DNS', _dnsname_to_stdlib(name)) for name in ext.get_values_for_type(x509.DNSName)
    ]
    names.extend(
        [('IP Address', str(name)) for name in ext.get_values_for_type(x509.IPAddress)]
    )
    return names


class WrappedSocket:
    """API-compatibility wrapper for Python OpenSSL's Connection-class.

    Note: _makefile_refs, _drop() and _reuse() are needed for the garbage
    collector of pypy.
    """

    def __init__(
        self,
        connection: OpenSSL.SSL.Connection,
        socket_obj: Any,
        suppress_ragged_eofs: bool = True
    ) -> None:
        self.connection: OpenSSL.SSL.Connection = connection
        self.socket: Any = socket_obj
        self.suppress_ragged_eofs: bool = suppress_ragged_eofs
        self._makefile_refs: int = 0
        self._closed: bool = False

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
        except OpenSSL.SSL.ZeroReturnError:
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
        except OpenSSL.SSL.ZeroReturnError:
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

    def settimeout(self, timeout_val: float) -> None:
        return self.socket.settimeout(timeout_val)

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
        total_sent: int = 0
        data_length: int = len(data)
        while total_sent < data_length:
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

    def getpeercert(self, binary_form: bool = False) -> Union[Dict[str, Any], bytes, None]:
        x509_cert = self.connection.get_peer_certificate()
        if not x509_cert:
            return x509_cert
        if binary_form:
            return OpenSSL.crypto.dump_certificate(OpenSSL.crypto.FILETYPE_ASN1, x509_cert)
        return {
            'subject': ((('commonName', x509_cert.get_subject().CN),),),
            'subjectAltName': get_subj_alt_name(x509_cert)
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


if _fileobject is not None:

    def makefile(self, mode: str, bufsize: int = -1) -> Any:
        self._makefile_refs += 1
        return _fileobject(self, mode, bufsize, close=True)
else:
    makefile = backport_makefile

WrappedSocket.makefile = makefile  # type: ignore


class PyOpenSSLContext:
    """
    I am a wrapper class for the PyOpenSSL ``Context`` object. I am responsible
    for translating the interface of the standard library ``SSLContext`` object
    to calls into PyOpenSSL.
    """

    def __init__(self, protocol: int) -> None:
        self.protocol: Any = _openssl_versions[protocol]
        self._ctx: OpenSSL.SSL.Context = OpenSSL.SSL.Context(self.protocol)
        self._options: int = 0
        self.check_hostname: bool = False

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

    def load_verify_locations(
        self,
        cafile: Optional[str] = None,
        capath: Optional[str] = None,
        cadata: Optional[bytes] = None
    ) -> None:
        if cafile is not None:
            cafile = cafile.encode('utf-8')
        if capath is not None:
            capath = capath.encode('utf-8')
        self._ctx.load_verify_locations(cafile, capath)
        if cadata is not None:
            self._ctx.load_verify_locations(BytesIO(cadata))

    def load_cert_chain(
        self,
        certfile: str,
        keyfile: Optional[str] = None,
        password: Optional[str] = None
    ) -> None:
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
        cnx: OpenSSL.SSL.Connection = OpenSSL.SSL.Connection(self._ctx, sock)
        if isinstance(server_hostname, six.text_type):
            server_hostname_encoded: bytes = server_hostname.encode('utf-8')
            server_hostname = server_hostname_encoded  # type: ignore
        if server_hostname is not None:
            cnx.set_tlsext_host_name(server_hostname)  # type: ignore
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
                raise ssl.SSLError(f'bad handshake: {e!r}') from e
            break
        return WrappedSocket(cnx, sock)


def _verify_callback(
    cnx: Any,
    x509: Any,
    err_no: int,
    err_depth: int,
    return_code: int
) -> int:
    return int(err_no == 0)
