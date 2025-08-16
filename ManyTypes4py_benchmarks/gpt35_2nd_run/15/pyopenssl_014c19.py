from __future__ import absolute_import
import OpenSSL.SSL
from cryptography import x509
from cryptography.hazmat.backends.openssl import backend as openssl_backend
from cryptography.hazmat.backends.openssl.x509 import _Certificate
from socket import timeout, error as SocketError
from io import BytesIO
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
_openssl_versions: dict = {ssl.PROTOCOL_SSLv23: OpenSSL.SSL.SSLv23_METHOD, ssl.PROTOCOL_TLSv1: OpenSSL.SSL.TLSv1_METHOD}
if hasattr(ssl, 'PROTOCOL_TLSv1_1') and hasattr(OpenSSL.SSL, 'TLSv1_1_METHOD'):
    _openssl_versions[ssl.PROTOCOL_TLSv1_1] = OpenSSL.SSL.TLSv1_1_METHOD
if hasattr(ssl, 'PROTOCOL_TLSv1_2') and hasattr(OpenSSL.SSL, 'TLSv1_2_METHOD'):
    _openssl_versions[ssl.PROTOCOL_TLSv1_2] = OpenSSL.SSL.TLSv1_2_METHOD
try:
    _openssl_versions.update({ssl.PROTOCOL_SSLv3: OpenSSL.SSL.SSLv3_METHOD})
except AttributeError:
    pass
_stdlib_to_openssl_verify: dict = {ssl.CERT_NONE: OpenSSL.SSL.VERIFY_NONE, ssl.CERT_OPTIONAL: OpenSSL.SSL.VERIFY_PEER, ssl.CERT_REQUIRED: OpenSSL.SSL.VERIFY_PEER + OpenSSL.SSL.VERIFY_FAIL_IF_NO_PEER_CERT}
_openssl_to_stdlib_verify: dict = dict(((v, k) for k, v in _stdlib_to_openssl_verify.items()))
SSL_WRITE_BLOCKSIZE: int = 16384
orig_util_HAS_SNI: bool = util.HAS_SNI
orig_util_SSLContext = util.ssl_.SSLContext
log = logging.getLogger(__name__)

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
    ...

def get_subj_alt_name(peer_cert) -> list:
    ...

class WrappedSocket:
    ...

def _verify_callback(cnx, x509, err_no, err_depth, return_code) -> bool:
    ...

if _fileobject:
    def makefile(self, mode, bufsize=-1) -> _fileobject:
        ...
else:
    makefile = backport_makefile
WrappedSocket.makefile = makefile

class PyOpenSSLContext:
    ...

    def __init__(self, protocol) -> None:
        ...

    @property
    def options(self) -> int:
        ...

    @options.setter
    def options(self, value: int) -> None:
        ...

    @property
    def verify_mode(self) -> int:
        ...

    @verify_mode.setter
    def verify_mode(self, value: int) -> None:
        ...

    def set_default_verify_paths(self) -> None:
        ...

    def set_ciphers(self, ciphers: str) -> None:
        ...

    def load_verify_locations(self, cafile=None, capath=None, cadata=None) -> None:
        ...

    def load_cert_chain(self, certfile, keyfile=None, password=None) -> None:
        ...

    def wrap_socket(self, sock, server_side=False, do_handshake_on_connect=True, suppress_ragged_eofs=True, server_hostname=None) -> WrappedSocket:
        ...
