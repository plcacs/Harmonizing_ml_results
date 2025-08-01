from __future__ import absolute_import
import errno
import logging
import warnings
import hmac
from binascii import hexlify, unhexlify
from hashlib import md5, sha1, sha256
from typing import Optional, Callable, Dict, Any, Union
from ..exceptions import SSLError, InsecurePlatformWarning, SNIMissingWarning
from ..packages.ssl_match_hostname import match_hostname as _match_hostname, CertificateError
import ssl
import socket

SSLContext: Optional[Any] = None
HAS_SNI: bool = False
IS_PYOPENSSL: bool = False
IS_SECURETRANSPORT: bool = False
HASHFUNC_MAP: Dict[int, Callable[[bytes], Any]] = {32: md5, 40: sha1, 64: sha256}
log: logging.Logger = logging.getLogger(__name__)

def _const_compare_digest_backport(a: Union[bytes, bytearray], b: Union[bytes, bytearray]) -> bool:
    """
    Compare two digests of equal length in constant time.

    The digests must be of type str/bytes.
    Returns True if the digests match, and False otherwise.
    """
    result: int = abs(len(a) - len(b))
    for l, r in zip(bytearray(a), bytearray(b)):
        result |= l ^ r
    return result == 0

_const_compare_digest: Callable[[Union[bytes, bytearray], Union[bytes, bytearray]], bool] = getattr(
    hmac, 'compare_digest', _const_compare_digest_backport
)

try:
    import ssl
    from ssl import wrap_socket, CERT_NONE, PROTOCOL_SSLv23
    from ssl import HAS_SNI
    from ssl import SSLError as BaseSSLError
except ImportError:

    class BaseSSLError(Exception):
        pass

try:
    from ssl import OP_NO_SSLv2, OP_NO_SSLv3, OP_NO_COMPRESSION
except ImportError:
    OP_NO_SSLv2: int = 16777216
    OP_NO_SSLv3: int = 33554432
    OP_NO_COMPRESSION: int = 131072

DEFAULT_CIPHERS: str = ':'.join([
    'TLS13-AES-256-GCM-SHA384',
    'TLS13-CHACHA20-POLY1305-SHA256',
    'TLS13-AES-128-GCM-SHA256',
    'ECDH+AESGCM',
    'ECDH+CHACHA20',
    'DH+AESGCM',
    'DH+CHACHA20',
    'ECDH+AES256',
    'DH+AES256',
    'ECDH+AES128',
    'DH+AES',
    'RSA+AESGCM',
    'RSA+AES',
    '!aNULL',
    '!eNULL',
    '!MD5'
])

try:
    from ssl import SSLContext
except ImportError:

    class SSLContext:
        protocol: int
        check_hostname: bool
        verify_mode: int
        ca_certs: Optional[str]
        options: int
        certfile: Optional[str]
        keyfile: Optional[str]
        ciphers: Optional[str]

        def __init__(self, protocol_version: int) -> None:
            self.protocol = protocol_version
            self.check_hostname = False
            self.verify_mode = ssl.CERT_NONE
            self.ca_certs = None
            self.options = 0
            self.certfile = None
            self.keyfile = None
            self.ciphers = None

        def load_cert_chain(self, certfile: str, keyfile: Optional[str] = None) -> None:
            self.certfile = certfile
            self.keyfile = keyfile

        def load_verify_locations(self, cafile: Optional[str] = None, capath: Optional[str] = None) -> None:
            self.ca_certs = cafile
            if capath is not None:
                raise SSLError('CA directories not supported in older Pythons')

        def set_ciphers(self, cipher_suite: str) -> None:
            self.ciphers = cipher_suite

        def wrap_socket(
            self,
            socket_: socket.socket,
            server_hostname: Optional[str] = None,
            server_side: bool = False
        ) -> ssl.SSLSocket:
            warnings.warn(
                'A true SSLContext object is not available. This prevents urllib3 from configuring SSL appropriately and may cause certain SSL connections to fail. You can upgrade to a newer version of Python to solve this. For more information, see https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings',
                InsecurePlatformWarning
            )
            kwargs: Dict[str, Any] = {
                'keyfile': self.keyfile,
                'certfile': self.certfile,
                'ca_certs': self.ca_certs,
                'cert_reqs': self.verify_mode,
                'ssl_version': self.protocol,
                'server_side': server_side
            }
            return wrap_socket(socket_, ciphers=self.ciphers, **kwargs)

def assert_fingerprint(cert: bytes, fingerprint: str) -> None:
    """
    Checks if given fingerprint matches the supplied certificate.

    :param cert:
        Certificate as bytes object.
    :param fingerprint:
        Fingerprint as string of hexdigits, can be interspersed by colons.
    """
    fingerprint_clean: str = fingerprint.replace(':', '').lower()
    digest_length: int = len(fingerprint_clean)
    hashfunc: Optional[Callable[[bytes], Any]] = HASHFUNC_MAP.get(digest_length)
    if not hashfunc:
        raise SSLError('Fingerprint of invalid length: {0}'.format(fingerprint))
    fingerprint_bytes: bytes = unhexlify(fingerprint_clean.encode())
    cert_digest: bytes = hashfunc(cert).digest()
    if not _const_compare_digest(cert_digest, fingerprint_bytes):
        raise SSLError('Fingerprints did not match. Expected "{0}", got "{1}".'.format(
            fingerprint, hexlify(cert_digest)
        ))

def resolve_cert_reqs(candidate: Optional[Union[str, int]]) -> int:
    """
    Resolves the argument to a numeric constant, which can be passed to
    the wrap_socket function/method from the ssl module.
    Defaults to :data:`ssl.CERT_NONE`.
    If given a string it is assumed to be the name of the constant in the
    :mod:`ssl` module or its abbrevation.
    (So you can specify `REQUIRED` instead of `CERT_REQUIRED`.
    If it's neither `None` nor a string we assume it is already the numeric
    constant which can directly be passed to wrap_socket.
    """
    if candidate is None:
        return CERT_NONE
    if isinstance(candidate, str):
        res = getattr(ssl, candidate, None)
        if res is None:
            res = getattr(ssl, 'CERT_' + candidate)
        if res is None:
            raise ValueError(f'Invalid cert_reqs value: {candidate}')
        return res
    return candidate

def resolve_ssl_version(candidate: Optional[Union[str, int]]) -> int:
    """
    like resolve_cert_reqs
    """
    if candidate is None:
        return PROTOCOL_SSLv23
    if isinstance(candidate, str):
        res = getattr(ssl, candidate, None)
        if res is None:
            res = getattr(ssl, 'PROTOCOL_' + candidate)
        if res is None:
            raise ValueError(f'Invalid ssl_version value: {candidate}')
        return res
    return candidate

def create_urllib3_context(
    ssl_version: Optional[Union[str, int]] = None,
    cert_reqs: Optional[Union[str, int]] = None,
    options: Optional[int] = None,
    ciphers: Optional[str] = None
) -> SSLContext:
    """All arguments have the same meaning as ``ssl_wrap_socket``.

    By default, this function does a lot of the same work that
    ``ssl.create_default_context`` does on Python 3.4+. It:

    - Disables SSLv2, SSLv3, and compression
    - Sets a restricted set of server ciphers

    If you wish to enable SSLv3, you can do::

        from urllib3.util import ssl_
        context = ssl_.create_urllib3_context()
        context.options &= ~ssl_.OP_NO_SSLv3

    You can do the same to enable compression (substituting ``COMPRESSION``
    for ``SSLv3`` in the last line above).

    :param ssl_version:
        The desired protocol version to use. This will default to
        PROTOCOL_SSLv23 which will negotiate the highest protocol that both
        the server and your installation of OpenSSL support.
    :param cert_reqs:
        Whether to require the certificate verification. This defaults to
        ``ssl.CERT_REQUIRED``.
    :param options:
        Specific OpenSSL options. These default to ``ssl.OP_NO_SSLv2``,
        ``ssl.OP_NO_SSLv3``, ``ssl.OP_NO_COMPRESSION``.
    :param ciphers:
        Which cipher suites to allow the server to select.
    :returns:
        Constructed SSLContext object with specified options
    :rtype: SSLContext
    """
    protocol_resolved: int = resolve_ssl_version(ssl_version) if ssl_version else PROTOCOL_SSLv23
    context: SSLContext = SSLContext(protocol_resolved)
    cert_reqs_resolved: int = resolve_cert_reqs(cert_reqs) if cert_reqs is not None else ssl.CERT_REQUIRED
    context.verify_mode = cert_reqs_resolved
    if options is None:
        options_resolved: int = 0
        options_resolved |= OP_NO_SSLv2
        options_resolved |= OP_NO_SSLv3
        options_resolved |= OP_NO_COMPRESSION
    else:
        options_resolved = options
    context.options |= options_resolved
    context.set_ciphers(ciphers or DEFAULT_CIPHERS)
    if hasattr(context, 'check_hostname') and isinstance(context.check_hostname, bool):
        context.check_hostname = False
    return context

def merge_context_settings(
    context: SSLContext,
    keyfile: Optional[str] = None,
    certfile: Optional[str] = None,
    cert_reqs: Optional[Union[str, int]] = None,
    ca_certs: Optional[str] = None,
    ca_cert_dir: Optional[str] = None
) -> SSLContext:
    """
    Merges provided settings into an SSL Context.
    """
    if cert_reqs is not None:
        context.verify_mode = resolve_cert_reqs(cert_reqs)
    if ca_certs or ca_cert_dir:
        try:
            context.load_verify_locations(cafile=ca_certs, capath=ca_cert_dir)
        except IOError as e:
            raise SSLError(e)
        except OSError as e:
            if e.errno == errno.ENOENT:
                raise SSLError(e)
            raise
    elif hasattr(context, 'load_default_certs'):
        context.load_default_certs()
    if certfile:
        context.load_cert_chain(certfile, keyfile)
    return context

def ssl_wrap_socket(
    sock: socket.socket,
    keyfile: Optional[str] = None,
    certfile: Optional[str] = None,
    cert_reqs: Optional[Union[str, int]] = None,
    ca_certs: Optional[str] = None,
    server_hostname: Optional[str] = None,
    ssl_version: Optional[Union[str, int]] = None,
    ciphers: Optional[str] = None,
    ssl_context: Optional[SSLContext] = None,
    ca_cert_dir: Optional[str] = None
) -> ssl.SSLSocket:
    """
    All arguments except for server_hostname, ssl_context, and ca_cert_dir have
    the same meaning as they do when using :func:`ssl.wrap_socket`.

    :param server_hostname:
        When SNI is supported, the expected hostname of the certificate
    :param ssl_context:
        A pre-made :class:`SSLContext` object. If none is provided, one will
        be created using :func:`create_urllib3_context`.
    :param ciphers:
        A string of ciphers we wish the client to support.
    :param ca_cert_dir:
        A directory containing CA certificates in multiple separate files, as
        supported by OpenSSL's -CApath flag or the capath argument to
        SSLContext.load_verify_locations().
    """
    context: SSLContext = ssl_context
    if context is None:
        context = create_urllib3_context(ssl_version=ssl_version, cert_reqs=cert_reqs, ciphers=ciphers)
    if ca_certs or ca_cert_dir:
        try:
            context.load_verify_locations(cafile=ca_certs, capath=ca_cert_dir)
        except IOError as e:
            raise SSLError(e)
        except OSError as e:
            if e.errno == errno.ENOENT:
                raise SSLError(e)
            raise
    elif hasattr(context, 'load_default_certs'):
        context.load_default_certs()
    if certfile:
        context.load_cert_chain(certfile, keyfile)
    if HAS_SNI:
        return context.wrap_socket(sock, server_hostname=server_hostname)
    warnings.warn(
        'An HTTPS request has been made, but the SNI (Server Name Indication) extension to TLS is not available on this platform. This may cause the server to present an incorrect TLS certificate, which can cause validation failures. You can upgrade to a newer version of Python to solve this. For more information, see https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings',
        SNIMissingWarning
    )
    return context.wrap_socket(sock)

def match_hostname(cert: Dict[str, Any], asserted_hostname: str) -> None:
    try:
        _match_hostname(cert, asserted_hostname)
    except CertificateError as e:
        log.error('Certificate did not match expected hostname: %s. Certificate: %s', asserted_hostname, cert)
        e._peer_cert = cert
        raise
