"""The networks module contains types for common network-related fields."""
from __future__ import annotations as _annotations
import dataclasses as _dataclasses
import re
from dataclasses import fields
from functools import lru_cache
from importlib.metadata import version
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Dict, List, Optional, Tuple, Union
from pydantic_core import MultiHostHost, PydanticCustomError, PydanticSerializationUnexpectedValue, SchemaSerializer, core_schema
from pydantic_core import MultiHostUrl as _CoreMultiHostUrl
from pydantic_core import Url as _CoreUrl
from typing_extensions import Self, TypeAlias
from pydantic.errors import PydanticUserError
from ._internal import _repr, _schema_generation_shared
from ._migration import getattr_migration
from .annotated_handlers import GetCoreSchemaHandler
from .json_schema import JsonSchemaValue
from .type_adapter import TypeAdapter

if TYPE_CHECKING:
    import email_validator
    NetworkType = Union[str, bytes, int, Tuple[Union[str, bytes, int], Union[str, int]]
else:
    email_validator = None

__all__ = [
    'AnyUrl', 'AnyHttpUrl', 'FileUrl', 'FtpUrl', 'HttpUrl', 'WebsocketUrl', 'AnyWebsocketUrl', 
    'UrlConstraints', 'EmailStr', 'NameEmail', 'IPvAnyAddress', 'IPvAnyInterface', 'IPvAnyNetwork', 
    'PostgresDsn', 'CockroachDsn', 'AmqpDsn', 'RedisDsn', 'MongoDsn', 'KafkaDsn', 'NatsDsn', 
    'validate_email', 'MySQLDsn', 'MariaDBDsn', 'ClickHouseDsn', 'SnowflakeDsn'
]

@_dataclasses.dataclass
class UrlConstraints:
    """Url constraints."""
    max_length: Optional[int] = None
    allowed_schemes: Optional[List[str]] = None
    host_required: Optional[bool] = None
    default_host: Optional[str] = None
    default_port: Optional[int] = None
    default_path: Optional[str] = None

    def __hash__(self) -> int:
        return hash((
            self.max_length, 
            tuple(self.allowed_schemes) if self.allowed_schemes is not None else None, 
            self.host_required, 
            self.default_host, 
            self.default_port, 
            self.default_path
        ))

    @property
    def defined_constraints(self) -> Dict[str, Any]:
        """Fetch a key / value mapping of constraints to values that are not None."""
        return {field.name: value for field in fields(self) if (value := getattr(self, field.name)) is not None}

    def __get_pydantic_core_schema__(self, source: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        schema = handler(source)
        schema_to_mutate = schema['schema'] if schema['type'] == 'function-wrap' else schema
        if (annotated_type := (schema_to_mutate['type'] not in ('url', 'multi-host-url'))):
            raise PydanticUserError(f"'UrlConstraints' cannot annotate '{annotated_type}'.", code='invalid-annotated-type')
        for constraint_key, constraint_value in self.defined_constraints.items():
            schema_to_mutate[constraint_key] = constraint_value
        return schema

class _BaseUrl:
    _constraints: ClassVar[UrlConstraints] = UrlConstraints()

    def __init__(self, url: Union[str, _CoreUrl]) -> None:
        self._url: _CoreUrl = _build_type_adapter(self.__class__).validate_python(url)._url

    @property
    def scheme(self) -> str:
        """The scheme part of the URL."""
        return self._url.scheme

    @property
    def username(self) -> Optional[str]:
        """The username part of the URL, or `None`."""
        return self._url.username

    @property
    def password(self) -> Optional[str]:
        """The password part of the URL, or `None`."""
        return self._url.password

    @property
    def host(self) -> Optional[str]:
        """The host part of the URL, or `None`."""
        return self._url.host

    def unicode_host(self) -> Optional[str]:
        """The host part of the URL as a unicode string, or `None`."""
        return self._url.unicode_host()

    @property
    def port(self) -> Optional[int]:
        """The port part of the URL, or `None`."""
        return self._url.port

    @property
    def path(self) -> Optional[str]:
        """The path part of the URL, or `None`."""
        return self._url.path

    @property
    def query(self) -> Optional[str]:
        """The query part of the URL, or `None`."""
        return self._url.query

    def query_params(self) -> List[Tuple[str, str]]:
        """The query part of the URL as a list of key-value pairs."""
        return self._url.query_params()

    @property
    def fragment(self) -> Optional[str]:
        """The fragment part of the URL, or `None`."""
        return self._url.fragment

    def unicode_string(self) -> str:
        """The URL as a unicode string."""
        return self._url.unicode_string()

    def __str__(self) -> str:
        """The URL as a string."""
        return str(self._url)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({str(self._url)!r})'

    def __deepcopy__(self, memo: Dict[int, Any]) -> Self:
        return self.__class__(self._url)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and self._url == other._url

    def __lt__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and self._url < other._url

    def __gt__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and self._url > other._url

    def __le__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and self._url <= other._url

    def __ge__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and self._url >= other._url

    def __hash__(self) -> int:
        return hash(self._url)

    def __len__(self) -> int:
        return len(str(self._url))

    @classmethod
    def build(
        cls,
        *,
        scheme: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        host: str,
        port: Optional[int] = None,
        path: Optional[str] = None,
        query: Optional[str] = None,
        fragment: Optional[str] = None
    ) -> Self:
        """Build a new `Url` instance from its component parts."""
        return cls(_CoreUrl.build(
            scheme=scheme,
            username=username,
            password=password,
            host=host,
            port=port,
            path=path,
            query=query,
            fragment=fragment
        ))

    @classmethod
    def serialize_url(cls, url: Any, info: Any) -> Union[str, Any]:
        if not isinstance(url, cls):
            raise PydanticSerializationUnexpectedValue(f"Expected `{cls}` but got `{type(url)}` with value `'{url}'` - serialized value may not be as expected.")
        if info.mode == 'json':
            return str(url)
        return url

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        def wrap_val(v: Any, h: Any) -> Any:
            if isinstance(v, source):
                return v
            if isinstance(v, _BaseUrl):
                v = str(v)
            core_url = h(v)
            instance = source.__new__(source)
            instance._url = core_url
            return instance
        return core_schema.no_info_wrap_validator_function(
            wrap_val,
            schema=core_schema.url_schema(**cls._constraints.defined_constraints),
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls.serialize_url, info_arg=True, when_used='always'
            )
        )

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: core_schema.CoreSchema, handler: Any) -> JsonSchemaValue:
        inner_schema = core_schema['schema'] if core_schema['type'] == 'function-wrap' else core_schema
        return handler(inner_schema)

    __pydantic_serializer__ = SchemaSerializer(core_schema.any_schema(serialization=core_schema.to_string_ser_schema()))

class _BaseMultiHostUrl:
    _constraints: ClassVar[UrlConstraints] = UrlConstraints()

    def __init__(self, url: Union[str, _CoreMultiHostUrl]) -> None:
        self._url: _CoreMultiHostUrl = _build_type_adapter(self.__class__).validate_python(url)._url

    @property
    def scheme(self) -> str:
        """The scheme part of the URL."""
        return self._url.scheme

    @property
    def path(self) -> Optional[str]:
        """The path part of the URL, or `None`."""
        return self._url.path

    @property
    def query(self) -> Optional[str]:
        """The query part of the URL, or `None`."""
        return self._url.query

    def query_params(self) -> List[Tuple[str, str]]:
        """The query part of the URL as a list of key-value pairs."""
        return self._url.query_params()

    @property
    def fragment(self) -> Optional[str]:
        """The fragment part of the URL, or `None`."""
        return self._url.fragment

    def hosts(self) -> List[MultiHostHost]:
        """The hosts of the `MultiHostUrl` as `MultiHostHost` typed dicts."""
        return self._url.hosts()

    def unicode_string(self) -> str:
        """The URL as a unicode string."""
        return self._url.unicode_string()

    def __str__(self) -> str:
        """The URL as a string."""
        return str(self._url)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({str(self._url)!r})'

    def __deepcopy__(self, memo: Dict[int, Any]) -> Self:
        return self.__class__(self._url)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and self._url == other._url

    def __hash__(self) -> int:
        return hash(self._url)

    def __len__(self) -> int:
        return len(str(self._url))

    @classmethod
    def build(
        cls,
        *,
        scheme: str,
        hosts: Optional[List[MultiHostHost]] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        path: Optional[str] = None,
        query: Optional[str] = None,
        fragment: Optional[str] = None
    ) -> Self:
        """Build a new `MultiHostUrl` instance from its component parts."""
        return cls(_CoreMultiHostUrl.build(
            scheme=scheme,
            hosts=hosts,
            username=username,
            password=password,
            host=host,
            port=port,
            path=path,
            query=query,
            fragment=fragment
        ))

    @classmethod
    def serialize_url(cls, url: Any, info: Any) -> Union[str, Any]:
        if not isinstance(url, cls):
            raise PydanticSerializationUnexpectedValue(f"Expected `{cls}` but got `{type(url)}` with value `'{url}'` - serialized value may not be as expected.")
        if info.mode == 'json':
            return str(url)
        return url

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        def wrap_val(v: Any, h: Any) -> Any:
            if isinstance(v, source):
                return v
            if isinstance(v, _BaseMultiHostUrl):
                v = str(v)
            core_url = h(v)
            instance = source.__new__(source)
            instance._url = core_url
            return instance
        return core_schema.no_info_wrap_validator_function(
            wrap_val,
            schema=core_schema.multi_host_url_schema(**cls._constraints.defined_constraints),
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls.serialize_url, info_arg=True, when_used='always'
            )
        )

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: core_schema.CoreSchema, handler: Any) -> JsonSchemaValue:
        inner_schema = core_schema['schema'] if core_schema['type'] == 'function-wrap' else core_schema
        return handler(inner_schema)

    __pydantic_serializer__ = SchemaSerializer(core_schema.any_schema(serialization=core_schema.to_string_ser_schema()))

@lru_cache
def _build_type_adapter(cls: Any) -> TypeAdapter:
    return TypeAdapter(cls)

class AnyUrl(_BaseUrl):
    """Base type for all URLs."""

class AnyHttpUrl(AnyUrl):
    """A type that will accept any http or https URL."""
    _constraints: ClassVar[UrlConstraints] = UrlConstraints(allowed_schemes=['http', 'https'])

class HttpUrl(AnyUrl):
    """A type that will accept any http or https URL."""
    _constraints: ClassVar[UrlConstraints] = UrlConstraints(max_length=2083, allowed_schemes=['http', 'https'])

class AnyWebsocketUrl(AnyUrl):
    """A type that will accept any ws or wss URL."""
    _constraints: ClassVar[UrlConstraints] = UrlConstraints(allowed_schemes=['ws', 'wss'])

class WebsocketUrl(AnyUrl):
    """A type that will accept any ws or wss URL."""
    _constraints: ClassVar[UrlConstraints] = UrlConstraints(max_length=2083, allowed_schemes=['ws', 'wss'])

class FileUrl(AnyUrl):
    """A type that will accept any file URL."""
    _constraints: ClassVar[UrlConstraints] = UrlConstraints(allowed_schemes=['file'])

class FtpUrl(AnyUrl):
    """A type that will accept ftp URL."""
    _constraints: ClassVar[UrlConstraints] = UrlConstraints(allowed_schemes=['ftp'])

class PostgresDsn(_BaseMultiHostUrl):
    """A type that will accept any Postgres DSN."""
    _constraints: ClassVar[UrlConstraints] = UrlConstraints(
        host_required=True,
        allowed_schemes=[
            'postgres', 'postgresql', 'postgresql+asyncpg', 'postgresql+pg8000',
            'postgresql+psycopg', 'postgresql+psycopg2', 'postgresql+psycopg2cffi',
            'postgresql+py-postgresql', 'postgresql+pygresql'
        ]
    )

    @property
    def host(self) -> str:
        """The required URL host."""
        return self._url.host

class CockroachDsn(AnyUrl):
    """A type that will accept any Cockroach DSN."""
    _constraints: ClassVar[UrlConstraints] = UrlConstraints(
        host_required=True,
        allowed_schemes=['cockroachdb', 'cockroachdb+psycopg2', 'cockroachdb+asyncpg']
    )

    @property
    def host(self) -> str:
        """The required URL host."""
        return self._url.host

class AmqpDsn(AnyUrl):
    """A type that will accept any AMQP DSN."""
    _constraints: ClassVar[UrlConstraints] = UrlConstraints(allowed_schemes=['amqp', 'amqps'])

class RedisDsn(AnyUrl):
    """A type that will accept any Redis DSN."""
    _constraints: ClassVar[UrlConstraints] = UrlConstraints(
        allowed_schemes=['redis', 'rediss'],
        default_host='localhost',
        default_port=6379,
        default_path='/0',
        host_required=True
    )

    @property
    def host(self) -> str:
        """The required URL host."""
        return self._url.host

class MongoDsn(_BaseMultiHostUrl):
    """A type that will accept any MongoDB DSN."""
    _constraints: ClassVar[UrlConstraints] = UrlConstraints(
        allowed_schemes=['mongodb', 'mongodb+srv'],
        default_port=27017
    )

class KafkaDsn(AnyUrl):
    """A type that will accept any Kafka DSN."""
    _constraints: ClassVar[UrlConstraints] = UrlConstraints(
        allowed_schemes=['kafka'],
        default_host='localhost',
        default_port=9092
    )

class NatsDsn(_BaseMultiHostUrl):
    """A type that will accept any NATS DSN."""
    _constraints: ClassVar[UrlConstraints] = UrlConstraints(
        allowed_schemes=['nats', 'tls', 'ws', 'wss'],
        default_host='localhost',
        default_port=4222
    )

class MySQLDsn(AnyUrl):
    """A type that will accept any MySQL DSN."""
    _constraints: ClassVar[UrlConstraints] = UrlConstraints(
        allowed_schemes=[
            'mysql', 'mysql+mysql