from typing import Union, Tuple, Dict, Any, Generator

def url_regex() -> 'Pattern':
    ...

def multi_host_url_regex() -> 'Pattern':
    ...

def ascii_domain_regex() -> 'Pattern':
    ...

def int_domain_regex() -> 'Pattern':
    ...

def host_regex() -> 'Pattern':
    ...

class AnyUrl(str):
    strip_whitespace: bool = True
    min_length: int = 1
    max_length: int = 2 ** 16
    allowed_schemes: Union[None, set] = None
    tld_required: bool = False
    user_required: bool = False
    host_required: bool = True
    hidden_parts: set = set()
    __slots__: Tuple[str] = ('scheme', 'user', 'password', 'host', 'tld', 'host_type', 'port', 'path', 'query', 'fragment')

    @no_type_check
    def __new__(cls, url: str, **kwargs: Any) -> 'AnyUrl':
        ...

    def __init__(self, url: str, *, scheme: str, user: str = None, password: str = None, host: str = None, tld: str = None, host_type: str = 'domain', port: str = None, path: str = None, query: str = None, fragment: str = None) -> None:
        ...

    @classmethod
    def build(cls, *, scheme: str, user: str = None, password: str = None, host: str, port: str = None, path: str = None, query: str = None, fragment: str = None, **_kwargs: Any) -> str:
        ...

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        ...

    @classmethod
    def __get_validators__(cls) -> Generator:
        ...

    @classmethod
    def validate(cls, value: Any, field: Any, config: Any) -> 'AnyUrl':
        ...

    @classmethod
    def _build_url(cls, m: 'Match', url: str, parts: 'Parts') -> 'AnyUrl':
        ...

    @staticmethod
    def _match_url(url: str) -> 'Match':
        ...

    @staticmethod
    def _validate_port(port: str) -> None:
        ...

    @classmethod
    def validate_parts(cls, parts: 'Parts', validate_port: bool = True) -> 'Parts':
        ...

    @classmethod
    def validate_host(cls, parts: 'Parts') -> Tuple[str, str, str, bool]:
        ...

    @staticmethod
    def get_default_parts(parts: 'Parts') -> Dict[str, Any]:
        ...

    @classmethod
    def apply_default_parts(cls, parts: 'Parts') -> 'Parts':
        ...

    def __repr__(self) -> str:
        ...

class AnyHttpUrl(AnyUrl):
    allowed_schemes: set = {'http', 'https'}
    __slots__: Tuple = ()

class HttpUrl(AnyHttpUrl):
    tld_required: bool = True
    max_length: int = 2083
    hidden_parts: set = {'port'}

    @staticmethod
    def get_default_parts(parts: 'Parts') -> Dict[str, Any]:
        ...

class FileUrl(AnyUrl):
    allowed_schemes: set = {'file'}
    host_required: bool = False
    __slots__: Tuple = ()

class MultiHostDsn(AnyUrl):
    __slots__: Tuple = AnyUrl.__slots__ + ('hosts',)

    def __init__(self, *args: Any, hosts: Any = None, **kwargs: Any) -> None:
        ...

    @staticmethod
    def _match_url(url: str) -> 'Match':
        ...

    @classmethod
    def validate_parts(cls, parts: 'Parts', validate_port: bool = True) -> 'Parts':
        ...

    @classmethod
    def _build_url(cls, m: 'Match', url: str, parts: 'Parts') -> 'MultiHostDsn':
        ...

class PostgresDsn(MultiHostDsn):
    allowed_schemes: set = {'postgres', 'postgresql', 'postgresql+asyncpg', 'postgresql+pg8000', 'postgresql+psycopg', 'postgresql+psycopg2', 'postgresql+psycopg2cffi', 'postgresql+py-postgresql', 'postgresql+pygresql'}
    user_required: bool = True
    __slots__: Tuple = ()

class CockroachDsn(AnyUrl):
    allowed_schemes: set = {'cockroachdb', 'cockroachdb+psycopg2', 'cockroachdb+asyncpg'}
    user_required: bool = True

class AmqpDsn(AnyUrl):
    allowed_schemes: set = {'amqp', 'amqps'}
    host_required: bool = False

class RedisDsn(AnyUrl):
    __slots__: Tuple = ()
    allowed_schemes: set = {'redis', 'rediss'}
    host_required: bool = False

    @staticmethod
    def get_default_parts(parts: 'Parts') -> Dict[str, Any]:
        ...

class MongoDsn(AnyUrl):
    allowed_schemes: set = {'mongodb'}

    @staticmethod
    def get_default_parts(parts: 'Parts') -> Dict[str, Any]:
        ...

class KafkaDsn(AnyUrl):
    allowed_schemes: set = {'kafka'}

    @staticmethod
    def get_default_parts(parts: 'Parts') -> Dict[str, Any]:
        ...

def stricturl(*, strip_whitespace: bool = True, min_length: int = 1, max_length: int = 2 ** 16, tld_required: bool = True, host_required: bool = True, allowed_schemes: Union[None, set] = None) -> Type['AnyUrl']:
    ...

def import_email_validator() -> None:
    ...

class EmailStr(str):
    ...

class NameEmail(Representation):
    __slots__: Tuple = ('name', 'email')

    def __init__(self, name: str, email: str) -> None:
        ...

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        ...

    @classmethod
    def __get_validators__(cls) -> Generator:
        ...

    @classmethod
    def validate(cls, value: str) -> 'NameEmail':
        ...

    def __str__(self) -> str:
        ...

class IPvAnyAddress(_BaseAddress):
    __slots__: Tuple = ()

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        ...

    @classmethod
    def __get_validators__(cls) -> Generator:
        ...

    @classmethod
    def validate(cls, value: str) -> Union[IPv4Address, IPv6Address]:
        ...

class IPvAnyInterface(_BaseAddress):
    __slots__: Tuple = ()

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        ...

    @classmethod
    def __get_validators__(cls) -> Generator:
        ...

    @classmethod
    def validate(cls, value: str) -> Union[IPv4Interface, IPv6Interface]:
        ...

class IPvAnyNetwork(_BaseNetwork):
    ...

def validate_email(value: str) -> Tuple[str, str]:
    ...
