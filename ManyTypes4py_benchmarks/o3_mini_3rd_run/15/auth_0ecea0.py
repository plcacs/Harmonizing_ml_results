import ssl
from typing import Any, Optional, Union

from faust.types.auth import AuthProtocol, CredentialsT, SASLMechanism

__all__ = ['Credentials', 'SASLCredentials', 'GSSAPICredentials', 'SSLCredentials']


class Credentials(CredentialsT):
    """Base class for authentication credentials."""


class SASLCredentials(Credentials):
    """Describe SASL credentials."""
    protocol: AuthProtocol = AuthProtocol.SASL_PLAINTEXT
    mechanism: SASLMechanism = SASLMechanism.PLAIN

    def __init__(
        self,
        *,
        username: Optional[str] = None,
        password: Optional[str] = None,
        ssl_context: Optional[ssl.SSLContext] = None,
        mechanism: Optional[Union[str, SASLMechanism]] = None
    ) -> None:
        self.username: Optional[str] = username
        self.password: Optional[str] = password
        self.ssl_context: Optional[ssl.SSLContext] = ssl_context
        if ssl_context is not None:
            self.protocol = AuthProtocol.SASL_SSL
        if mechanism is not None:
            self.mechanism = SASLMechanism(mechanism)

    def __repr__(self) -> str:
        return f'<{type(self).__name__}: username={self.username}>'


class GSSAPICredentials(Credentials):
    """Describe GSSAPI credentials over SASL."""
    protocol: AuthProtocol = AuthProtocol.SASL_PLAINTEXT
    mechanism: SASLMechanism = SASLMechanism.GSSAPI

    def __init__(
        self,
        *,
        kerberos_service_name: str = 'kafka',
        kerberos_domain_name: Optional[str] = None,
        ssl_context: Optional[ssl.SSLContext] = None,
        mechanism: Optional[Union[str, SASLMechanism]] = None
    ) -> None:
        self.kerberos_service_name: str = kerberos_service_name
        self.kerberos_domain_name: Optional[str] = kerberos_domain_name
        self.ssl_context: Optional[ssl.SSLContext] = ssl_context
        if ssl_context is not None:
            self.protocol = AuthProtocol.SASL_SSL
        if mechanism is not None:
            self.mechanism = SASLMechanism(mechanism)

    def __repr__(self) -> str:
        return '<{0}: kerberos service={1!r} domain={2!r}>'.format(
            type(self).__name__,
            self.kerberos_service_name,
            self.kerberos_domain_name
        )


class SSLCredentials(Credentials):
    """Describe SSL credentials/settings."""
    protocol: AuthProtocol = AuthProtocol.SSL

    def __init__(
        self,
        context: Optional[ssl.SSLContext] = None,
        *,
        purpose: Optional[ssl.Purpose] = None,
        cafile: Optional[str] = None,
        capath: Optional[str] = None,
        cadata: Optional[Union[str, bytes]] = None
    ) -> None:
        if context is None:
            context = ssl.create_default_context(purpose=purpose, cafile=cafile, capath=capath, cadata=cadata)
        self.context: ssl.SSLContext = context

    def __repr__(self) -> str:
        return f'<{type(self).__name__}: context={self.context}>'