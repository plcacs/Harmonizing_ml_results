from faust.types.auth import AuthProtocol, CredentialsT, SASLMechanism

class Credentials(CredentialsT):
    def __init__(self) -> None:
        ...

class SASLCredentials(Credentials):
    def __init__(self, *, username: Optional[str] = None, password: Optional[str] = None, ssl_context: Optional[ssl.SSLContext] = None, mechanism: Optional[Union[str, SASLMechanism]] = None) -> None:
        ...

class GSSAPICredentials(Credentials):
    def __init__(self, *, kerberos_service_name: str = 'kafka', kerberos_domain_name: Optional[str] = None, ssl_context: Optional[ssl.SSLContext] = None, mechanism: Optional[Union[str, SASLMechanism]] = None) -> None:
        ...

class SSLCredentials(Credentials):
    def __init__(self, context: Optional[ssl.SSLContext] = None, *, purpose: Optional[int] = None, cafile: Optional[str] = None, capath: Optional[str] = None, cadata: Optional[bytes] = None) -> None:
        ...
