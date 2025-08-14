from __future__ import absolute_import
import errno
import logging
import warnings
import hmac
import ssl
from binascii import hexlify, unhexlify
from hashlib import md5, sha1, sha256
from typing import Optional, Union, Any

from ..exceptions import SSLError, InsecurePlatformWarning, SNIMissingWarning
from ..packages.ssl_match_hostname import (
    match_hostname as _match_hostname,
    CertificateError,
)

SSLContext: Any = None
HAS_SNI: bool = False
IS_PYOPENSSL: bool = False
IS_SECURETRANSPORT: bool = False
# Maps the length of a digest to a possible hash function producing this digest
HASHFUNC_MAP = {32: md5, 40: sha1, 64: sha256}
log = logging.getLogger(__name__)


def _const_compare_digest_backport(a: bytes, b: bytes) -> bool:
    """
    Compare two digests of equal length in constant time.
    
    The digests must be of type bytes.
    Returns True if the digests match, and False otherwise.
    """
    result = abs(len(a) - len(b))
    for l, r in zip(bytearray(a), bytearray(b)):
        result |= l ^ r
    return result == 0


_const_compare_digest = getattr(hmac, "compare_digest", _const_compare_digest_backport)

try:  # Test for SSL features
    from ssl import wrap_socket, CERT_NONE, PROTOCOL_SSLv23
    from ssl import HAS_SNI  # Has SNI?
    from ssl import SSLError as BaseSSLError
except ImportError:

    class BaseSSLError(Exception):
        pass


try:
    from ssl import OP_NO_SSLv2, OP_NO_SSLv3, OP_NO_COMPRESSION
except ImportError:
    OP_NO_SSLv2, OP_NO_SSLv3 = 0x1000000, 0x2000000
    OP_NO_COMPRESSION = 0x20000
# A secure default.
DEFAULT_CIPHERS: str = ":".join(
    [
        "TLS13-AES-256-GCM-SHA384",
        "TLS13-CHACHA20-POLY1305-SHA256",
        "TLS13-AES-128-GCM-SHA256",
        "ECDH+AESGCM",
        "ECDH+CHACHA20",
        "DH+AESGCM",
        "DH+CHACHA20",
        "ECDH+AES256",
        "DH+AES256",
        "ECDH+AES128",
        "DH+AES",
        "RSA+AESGCM",
        "RSA+AES",
        "!aNULL",
        "!eNULL",
        "!MD5",
    ]
)
try:
    from ssl import SSLContext  # Modern SSL?
except ImportError:

    # TODO: Can we remove this by choosing to support only platforms with
    # actual SSLContext objects?
    class SSLContext(object):  # Platform-specific: Python 2 & 3.1
        def __init__(self, protocol_version: int) -> None:
            self.protocol = protocol_version
            # Use default values from a real SSLContext
            self.check_hostname = False
            self.verify_mode = ssl.CERT_NONE
            self.ca_certs = None
            self.options = 0
            self.certfile = None
            self.keyfile = None
            self.ciphers = None

        def load_cert_chain(self, certfile: str, keyfile: str) -> None:
            self.certfile = certfile
            self.keyfile = keyfile

        def load_verify_locations(self, cafile: Optional[str] = None, capath: Optional[str] = None) -> None:
            self.ca_certs = cafile
            if capath is not None:
                raise SSLError("CA directories not supported in older Pythons")

        def set_ciphers(self, cipher_suite: str) -> None:
            self.ciphers = cipher_suite

        def wrap_socket(self, socket: Any, server_hostname: Optional[str] = None, server_side: bool = False) -> Any:
            warnings.warn(
                "A true SSLContext object is not available. This prevents "
                "urllib3 from configuring SSL appropriately and may cause "
                "certain SSL connections to fail. You can upgrade to a newer "
                "version of Python to solve this. For more information, see "
                "https://urllib3.readthedocs.io/en/latest/advanced-usage.html"
                "#ssl-warnings",
                InsecurePlatformWarning,
            )
            kwargs = {
                "keyfile": self.keyfile,
                "certfile": self.certfile,
                "ca_certs": self.ca_certs,
                "cert_reqs": self.verify_mode,
                "ssl_version": self.protocol,
                "server_side": server_side,
            }
            return wrap_socket(socket, ciphers=self.ciphers, **kwargs)


def assert_fingerprint(cert: bytes, fingerprint: str) -> None:
    """
    Checks if given fingerprint matches the supplied certificate.

    :param cert:
        Certificate as bytes object.
    :param fingerprint:
        Fingerprint as string of hexdigits, can be interspersed by colons.
    """
    fingerprint = fingerprint.replace(":", "").lower()
    digest_length = len(fingerprint)
    hashfunc = HASHFUNC_MAP.get(digest_length)
    if not hashfunc:
        raise SSLError("Fingerprint of invalid length: {0}".format(fingerprint))

    fingerprint_bytes = unhexlify(fingerprint.encode())
    cert_digest = hashfunc(cert).digest()
    if not _const_compare_digest(cert_digest, fingerprint_bytes):
        raise SSLError(
            'Fingerprints did not match. Expected "{0}", got "{1}".'.format(
                fingerprint, hexlify(cert_digest)
            )
        )


def resolve_cert_reqs(candidate: Optional[Union[str, int]]) -> int:
    """
    Resolves the argument to a numeric constant, which can be passed to
    the wrap_socket function/method from the ssl module.
    Defaults to :data:`ssl.CERT_NONE`.
    """
    if candidate is None:
        return ssl.CERT_NONE

    if isinstance(candidate, str):
        res = getattr(ssl, candidate, None)
        if res is None:
            res = getattr(ssl, "CERT_" + candidate)
        return res

    return candidate


def resolve_ssl_version(candidate: Optional[Union[str, int]]) -> int:
    """
    Like resolve_cert_reqs but for SSL versions.
    """
    if candidate is None:
        return ssl.PROTOCOL_SSLv23

    if isinstance(candidate, str):
        res = getattr(ssl, candidate, None)
        if res is None:
            res = getattr(ssl, "PROTOCOL_" + candidate)
        return res

    return candidate


def create_urllib3_context(
    ssl_version: Optional[int] = None,
    cert_reqs: Optional[int] = None,
    options: Optional[int] = None,
    ciphers: Optional[str] = None,
) -> SSLContext:
    """All arguments have the same meaning as ``ssl_wrap_socket``.

    By default, this function does a lot of the same work that
    ``ssl.create_default_context`` does on Python 3.4+. It:

    - Disables SSLv2, SSLv3, and compression
    - Sets a restricted set of server ciphers

    :param ssl_version:
        The desired protocol version to use.
    :param cert_reqs:
        Whether to require certificate verification.
    :param options:
        Specific OpenSSL options.
    :param ciphers:
        Which cipher suites to allow.
    :returns:
        Constructed SSLContext object with specified options.
    """
    context = SSLContext(ssl_version or ssl.PROTOCOL_SSLv23)
    cert_reqs = ssl.CERT_REQUIRED if cert_reqs is None else cert_reqs
    if options is None:
        options = 0
        options |= OP_NO_SSLv2
        options |= OP_NO_SSLv3
        options |= OP_NO_COMPRESSION
    context.options |= options
    context.set_ciphers(ciphers or DEFAULT_CIPHERS)
    context.verify_mode = cert_reqs
    if getattr(context, "check_hostname", None) is not None:
        context.check_hostname = False
    return context


def merge_context_settings(
    context: SSLContext,
    keyfile: Optional[str] = None,
    certfile: Optional[str] = None,
    cert_reqs: Optional[Union[str, int]] = None,
    ca_certs: Optional[str] = None,
    ca_cert_dir: Optional[str] = None,
) -> SSLContext:
    """
    Merges provided settings into an SSL Context.
    """
    if cert_reqs is not None:
        context.verify_mode = resolve_cert_reqs(cert_reqs)
    if ca_certs or ca_cert_dir:
        try:
            context.load_verify_locations(ca_certs, ca_cert_dir)
        except IOError as e:
            raise SSLError(e)
        except OSError as e:
            if e.errno == errno.ENOENT:
                raise SSLError(e)
            raise
    elif getattr(context, "load_default_certs", None) is not None:
        context.load_default_certs()
    if certfile:
        context.load_cert_chain(certfile, keyfile)  # type: ignore
    return context


def ssl_wrap_socket(
    sock: Any,
    keyfile: Optional[str] = None,
    certfile: Optional[str] = None,
    cert_reqs: Optional[Union[str, int]] = None,
    ca_certs: Optional[str] = None,
    server_hostname: Optional[str] = None,
    ssl_version: Optional[int] = None,
    ciphers: Optional[str] = None,
    ssl_context: Optional[SSLContext] = None,
    ca_cert_dir: Optional[str] = None,
) -> Any:
    """
    Wraps a socket using SSL.
    """
    context: SSLContext
    if ssl_context is None:
        context = create_urllib3_context(ssl_version, cert_reqs, ciphers=ciphers)
    else:
        context = ssl_context
    if ca_certs or ca_cert_dir:
        try:
            context.load_verify_locations(ca_certs, ca_cert_dir)
        except IOError as e:
            raise SSLError(e)
        except OSError as e:
            if e.errno == errno.ENOENT:
                raise SSLError(e)
            raise
    elif getattr(context, "load_default_certs", None) is not None:
        context.load_default_certs()
    if certfile:
        context.load_cert_chain(certfile, keyfile)  # type: ignore
    if HAS_SNI:
        return context.wrap_socket(sock, server_hostname=server_hostname)
    warnings.warn(
        "An HTTPS request has been made, but the SNI (Server Name "
        "Indication) extension to TLS is not available on this platform. "
        "This may cause the server to present an incorrect TLS "
        "certificate, which can cause validation failures. You can upgrade to "
        "a newer version of Python to solve this. For more information, see "
        "https://urllib3.readthedocs.io/en/latest/advanced-usage.html"
        "#ssl-warnings",
        SNIMissingWarning,
    )
    return context.wrap_socket(sock)


def match_hostname(cert: Any, asserted_hostname: str) -> None:
    try:
        _match_hostname(cert, asserted_hostname)
    except CertificateError as e:
        log.error(
            "Certificate did not match expected hostname: %s. Certificate: %s",
            asserted_hostname,
            cert,
        )
        e._peer_cert = cert  # type: ignore
        raise