"""Authentication Credentials."""
import ssl
from typing import Any, Optional, Union
from faust.types.auth import AuthProtocol, CredentialsT, SASLMechanism
__all__ = ['Credentials', 'SASLCredentials', 'GSSAPICredentials', 'SSLCredentials']

class Credentials(CredentialsT):
    """Base class for authentication credentials."""

class SASLCredentials(Credentials):
    """Describe SASL credentials."""
    protocol = AuthProtocol.SASL_PLAINTEXT
    mechanism = SASLMechanism.PLAIN

    def __init__(self, *, username=None, password=None, ssl_context=None, mechanism=None) -> None:
        self.username = username
        self.password = password
        self.ssl_context = ssl_context
        if ssl_context is not None:
            self.protocol = AuthProtocol.SASL_SSL
        if mechanism is not None:
            self.mechanism = SASLMechanism(mechanism)

    def __repr__(self) -> typing.Text:
        return f'<{type(self).__name__}: username={self.username}>'

class GSSAPICredentials(Credentials):
    """Describe GSSAPI credentials over SASL."""
    protocol = AuthProtocol.SASL_PLAINTEXT
    mechanism = SASLMechanism.GSSAPI

    def __init__(self, *, kerberos_service_name='kafka', kerberos_domain_name=None, ssl_context=None, mechanism=None) -> None:
        self.kerberos_service_name = kerberos_service_name
        self.kerberos_domain_name = kerberos_domain_name
        self.ssl_context = ssl_context
        if ssl_context is not None:
            self.protocol = AuthProtocol.SASL_SSL
        if mechanism is not None:
            self.mechanism = SASLMechanism(mechanism)

    def __repr__(self) -> typing.Text:
        return '<{0}: kerberos service={1!r} domain={2!r}'.format(type(self).__name__, self.kerberos_service_name, self.kerberos_domain_name)

class SSLCredentials(Credentials):
    """Describe SSL credentials/settings."""
    protocol = AuthProtocol.SSL

    def __init__(self, context: Union[None, dict[str, typing.Any], str]=None, *, purpose: Union[None, ssl.SSLContext, typing.Sequence[typing.Any], str]=None, cafile: Union[None, ssl.SSLContext, typing.Sequence[typing.Any], str]=None, capath: Union[None, ssl.SSLContext, typing.Sequence[typing.Any], str]=None, cadata: Union[None, ssl.SSLContext, typing.Sequence[typing.Any], str]=None) -> None:
        if context is None:
            context = ssl.create_default_context(purpose=purpose, cafile=cafile, capath=capath, cadata=cadata)
        self.context = context

    def __repr__(self) -> typing.Text:
        return f'<{type(self).__name__}: context={self.context}>'