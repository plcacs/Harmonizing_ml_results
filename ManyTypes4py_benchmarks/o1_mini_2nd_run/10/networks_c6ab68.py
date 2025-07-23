"""The networks module contains types for common network-related fields."""
from __future__ import annotations as _annotations
import dataclasses as _dataclasses
import re
from dataclasses import fields
from functools import lru_cache
from importlib.metadata import version
from ipaddress import (
    IPv4Address,
    IPv4Interface,
    IPv4Network,
    IPv6Address,
    IPv6Interface,
    IPv6Network,
)
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Callable, Dict, List, Optional, Tuple, Union
from pydantic_core import (
    MultiHostHost,
    PydanticCustomError,
    PydanticSerializationUnexpectedValue,
    SchemaSerializer,
    core_schema,
)
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
    NetworkType = 'str | bytes | int | tuple[str | bytes | int, str | int]'
else:
    email_validator = None

__all__ = [
    'AnyUrl',
    'AnyHttpUrl',
    'FileUrl',
    'FtpUrl',
    'HttpUrl',
    'WebsocketUrl',
    'AnyWebsocketUrl',
    'UrlConstraints',
    'EmailStr',
    'NameEmail',
    'IPvAnyAddress',
    'IPvAnyInterface',
    'IPvAnyNetwork',
    'PostgresDsn',
    'CockroachDsn',
    'AmqpDsn',
    'RedisDsn',
    'MongoDsn',
    'KafkaDsn',
    'NatsDsn',
    'validate_email',
    'MySQLDsn',
    'MariaDBDsn',
    'ClickHouseDsn',
    'SnowflakeDsn',
]

@_dataclasses.dataclass
class UrlConstraints:
    """Url constraints.

    Attributes:
        max_length: The maximum length of the url. Defaults to `None`.
        allowed_schemes: The allowed schemes. Defaults to `None`.
        host_required: Whether the host is required. Defaults to `None`.
        default_host: The default host. Defaults to `None`.
        default_port: The default port. Defaults to `None`.
        default_path: The default path. Defaults to `None`.
    """
    max_length: Optional[int] = None
    allowed_schemes: Optional[List[str]] = None
    host_required: Optional[bool] = None
    default_host: Optional[str] = None
    default_port: Optional[int] = None
    default_path: Optional[str] = None

    def __hash__(self) -> int:
        return hash(
            (
                self.max_length,
                tuple(self.allowed_schemes) if self.allowed_schemes is not None else None,
                self.host_required,
                self.default_host,
                self.default_port,
                self.default_path,
            )
        )

    @property
    def defined_constraints(self) -> Dict[str, Any]:
        """Fetch a key / value mapping of constraints to values that are not None. Used for core schema updates."""
        return {field.name: value for field in fields(self) if (value := getattr(self, field.name)) is not None}

    def __get_pydantic_core_schema__(self, source: Any, handler: GetCoreSchemaHandler) -> Dict[str, Any]:
        schema = handler(source)
        schema_to_mutate = schema['schema'] if schema['type'] == 'function-wrap' else schema
        if (annotated_type := (schema_to_mutate['type'] not in ('url', 'multi-host-url'))):
            raise PydanticUserError(f"'UrlConstraints' cannot annotate '{annotated_type}'.", code='invalid-annotated-type')
        for constraint_key, constraint_value in self.defined_constraints.items():
            schema_to_mutate[constraint_key] = constraint_value
        return schema

class _BaseUrl:
    _constraints: UrlConstraints = UrlConstraints()

    def __init__(self, url: Any):
        self._url = _build_type_adapter(self.__class__).validate_python(url)._url

    @property
    def scheme(self) -> str | None:
        """The scheme part of the URL.

        e.g. `https` in `https://user:pass@host:port/path?query#fragment`
        """
        return self._url.scheme

    @property
    def username(self) -> str | None:
        """The username part of the URL, or `None`.

        e.g. `user` in `https://user:pass@host:port/path?query#fragment`
        """
        return self._url.username

    @property
    def password(self) -> str | None:
        """The password part of the URL, or `None`.

        e.g. `pass` in `https://user:pass@host:port/path?query#fragment`
        """
        return self._url.password

    @property
    def host(self) -> str | None:
        """The host part of the URL, or `None`.

        If the URL must be punycode encoded, this is the encoded host, e.g if the input URL is `https://£££.com`,
        `host` will be `xn--9aaa.com`
        """
        return self._url.host

    def unicode_host(self) -> str | None:
        """The host part of the URL as a unicode string, or `None`.

        e.g. `host` in `https://user:pass@host:port/path?query#fragment`

        If the URL must be punycode encoded, this is the decoded host, e.g if the input URL is `https://£££.com`,
        `unicode_host()` will be `£££.com`
        """
        return self._url.unicode_host()

    @property
    def port(self) -> int | None:
        """The port part of the URL, or `None`.

        e.g. `port` in `https://user:pass@host:port/path?query#fragment`
        """
        return self._url.port

    @property
    def path(self) -> str | None:
        """The path part of the URL, or `None`.

        e.g. `/path` in `https://user:pass@host:port/path?query#fragment`
        """
        return self._url.path

    @property
    def query(self) -> str | None:
        """The query part of the URL, or `None`.

        e.g. `query` in `https://user:pass@host:port/path?query#fragment`
        """
        return self._url.query

    def query_params(self) -> List[Tuple[str, str]]:
        """The query part of the URL as a list of key-value pairs.

        e.g. `[('foo', 'bar')]` in `https://user:pass@host:port/path?foo=bar#fragment`
        """
        return self._url.query_params()

    @property
    def fragment(self) -> str | None:
        """The fragment part of the URL, or `None`.

        e.g. `fragment` in `https://user:pass@host:port/path?query#fragment`
        """
        return self._url.fragment

    def unicode_string(self) -> str:
        """The URL as a unicode string, unlike `__str__()` this will not punycode encode the host.

        If the URL must be punycode encoded, this is the decoded string, e.g if the input URL is `https://£££.com`,
        `unicode_string()` will be `https://£££.com`
        """
        return self._url.unicode_string()

    def __str__(self) -> str:
        """The URL as a string, this will punycode encode the host if required."""
        return str(self._url)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({str(self._url)!r})'

    def __deepcopy__(self, memo: Any) -> Self:
        return self.__class__(self._url)

    def __eq__(self, other: Any) -> bool:
        return self.__class__ is other.__class__ and self._url == other._url

    def __lt__(self, other: Any) -> bool:
        return self.__class__ is other.__class__ and self._url < other._url

    def __gt__(self, other: Any) -> bool:
        return self.__class__ is other.__class__ and self._url > other._url

    def __le__(self, other: Any) -> bool:
        return self.__class__ is other.__class__ and self._url <= other._url

    def __ge__(self, other: Any) -> bool:
        return self.__class__ is other.__class__ and self._url >= other._url

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
        fragment: Optional[str] = None,
    ) -> Self:
        """Build a new `Url` instance from its component parts.

        Args:
            scheme: The scheme part of the URL.
            username: The username part of the URL, or omit for no username.
            password: The password part of the URL, or omit for no password.
            host: The host part of the URL.
            port: The port part of the URL, or omit for no port.
            path: The path part of the URL, or omit for no path.
            query: The query part of the URL, or omit for no query.
            fragment: The fragment part of the URL, or omit for no fragment.

        Returns:
            An instance of URL
        """
        return cls(
            _CoreUrl.build(
                scheme=scheme,
                username=username,
                password=password,
                host=host,
                port=port,
                path=path,
                query=query,
                fragment=fragment,
            )
        )

    @classmethod
    def serialize_url(cls, url: Self, info: Any) -> Union[str, Self]:
        if not isinstance(url, cls):
            raise PydanticSerializationUnexpectedValue(
                f"Expected `{cls}` but got `{type(url)}` with value `'{url}'` - serialized value may not be as expected."
            )
        if info.mode == 'json':
            return str(url)
        return url

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> Dict[str, Any]:

        def wrap_val(v: Any, h: Callable[[Any], Any]) -> Self:
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
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Dict[str, Any], handler: Callable[[Any], Any]) -> Dict[str, Any]:
        inner_schema = core_schema['schema'] if core_schema['type'] == 'function-wrap' else core_schema
        return handler(inner_schema)

    __pydantic_serializer__: SchemaSerializer = SchemaSerializer(
        core_schema.any_schema(serialization=core_schema.to_string_ser_schema())
    )

class _BaseMultiHostUrl:
    _constraints: UrlConstraints = UrlConstraints()

    def __init__(self, url: Any):
        self._url = _build_type_adapter(self.__class__).validate_python(url)._url

    @property
    def scheme(self) -> str | None:
        """The scheme part of the URL.

        e.g. `https` in `https://foo.com,bar.com/path?query#fragment`
        """
        return self._url.scheme

    @property
    def path(self) -> str | None:
        """The path part of the URL, or `None`.

        e.g. `/path` in `https://foo.com,bar.com/path?query#fragment`
        """
        return self._url.path

    @property
    def query(self) -> str | None:
        """The query part of the URL, or `None`.

        e.g. `query` in `https://foo.com,bar.com/path?query#fragment`
        """
        return self._url.query

    def query_params(self) -> List[Tuple[str, str]]:
        """The query part of the URL as a list of key-value pairs.

        e.g. `[('foo', 'bar')]` in `https://foo.com,bar.com/path?foo=bar#fragment`
        """
        return self._url.query_params()

    @property
    def fragment(self) -> str | None:
        """The fragment part of the URL, or `None`.

        e.g. `fragment` in `https://foo.com,bar.com/path?query#fragment`
        """
        return self._url.fragment

    def hosts(self) -> List[MultiHostHost]:
        '''The hosts of the `MultiHostUrl` as [`MultiHostHost`][pydantic_core.MultiHostHost] typed dicts.

        