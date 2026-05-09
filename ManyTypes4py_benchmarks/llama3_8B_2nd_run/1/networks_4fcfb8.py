import re
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network, _BaseAddress, _BaseNetwork
from typing import TYPE_CHECKING, Any, Collection, Dict, Generator, List, Match, Optional, Pattern, Set, Tuple, Type, Union, cast, no_type_check
from pydantic.v1 import errors
from pydantic.v1.utils import Representation, update_not_none
from pydantic.v1.validators import constr_length_validator, str_validator
if TYPE_CHECKING:
    import email_validator
    from typing_extensions import TypedDict
    from pydantic.v1.config import BaseConfig
    from pydantic.v1.fields import ModelField
    from pydantic.v1.typing import AnyCallable
    CallableGenerator = Generator[AnyCallable, None, None]

    class Parts(TypedDict, total=False):
        pass

    class HostParts(TypedDict, total=False):
        pass
else:
    email_validator = None

    class Parts(dict):
        pass

NetworkType = Union[str, bytes, int, Tuple[Union[str, bytes, int], Union[str, int]]]

__all__ = ['AnyUrl', 'AnyHttpUrl', 'FileUrl', 'HttpUrl', 'stricturl', 'EmailStr', 'NameEmail', 'IPvAnyAddress', 'IPvAnyInterface', 'IPvAnyNetwork', 'PostgresDsn', 'CockroachDsn', 'AmqpDsn', 'RedisDsn', 'MongoDsn', 'KafkaDsn', 'validate_email']

_url_regex_cache: Optional[re.Pattern] = None
_multi_host_url_regex_cache: Optional[re.Pattern] = None
_ascii_domain_regex_cache: Optional[re.Pattern] = None
_int_domain_regex_cache: Optional[re.Pattern] = None
_host_regex_cache: Optional[re.Pattern] = None

class AnyUrl(str):
    strip_whitespace: bool
    min_length: int
    max_length: int
    allowed_schemes: Optional[Collection[str]]
    tld_required: bool
    user_required: bool
    host_required: bool
    hidden_parts: Set[str]
    __slots__: Tuple[str, ...]

    @no_type_check
    def __new__(cls, url: str, **kwargs: Any) -> str:
        return str.__new__(cls, cls.build(**kwargs) if url is None else url)

    def __init__(self, url: str, *, scheme: str, user: Optional[str], password: Optional[str], host: str, tld: Optional[str], host_type: str, port: Optional[int], path: Optional[str], query: Optional[str], fragment: Optional[str]) -> None:
        str.__init__(url)
        self.scheme = scheme
        self.user = user
        self.password = password
        self.host = host
        self.tld = tld
        self.host_type = host_type
        self.port = port
        self.path = path
        self.query = query
        self.fragment = fragment

    @classmethod
    def build(cls, *, scheme: str, user: Optional[str], password: Optional[str], host: str, port: Optional[int], path: Optional[str], query: Optional[str], fragment: Optional[str], **_kwargs: Any) -> str:
        # ...

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(field_schema, minLength=cls.min_length, maxLength=cls.max_length, format='uri')

    @classmethod
    def __get_validators__(cls) -> Generator[Callable[[str], str], None, None]:
        yield cls.validate

    @classmethod
    def validate(cls, value: str, field: ModelField, config: BaseConfig) -> str:
        # ...

    @classmethod
    def _build_url(cls, m: Match, url: str, parts: Parts) -> str:
        # ...

    @classmethod
    def _match_url(cls) -> re.Pattern:
        return url_regex()

    @classmethod
    def _validate_port(cls, port: Optional[int]) -> None:
        if port is not None and port > 65535:
            raise errors.UrlPortError()

    @classmethod
    def validate_parts(cls, parts: Parts, validate_port: bool = True) -> Parts:
        # ...

    @classmethod
    def get_default_parts(cls, parts: Parts) -> Parts:
        # ...

    def __repr__(self) -> str:
        extra = ', '.join((f'{n}={getattr(self, n)!r}' for n in self.__slots__ if getattr(self, n) is not None))
        return f'{self.__class__.__name__}({super().__repr__()}, {extra})'
