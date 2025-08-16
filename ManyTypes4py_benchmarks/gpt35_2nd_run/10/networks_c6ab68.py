from __future__ import annotations
import dataclasses
import re
from dataclasses import fields
from functools import lru_cache
from importlib.metadata import version
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from typing import TYPE_CHECKING, Annotated, Any, ClassVar
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
    NetworkType = 'str | bytes | int | tuple[str | bytes | int, str | int]'
else:
    email_validator = None

__all__ = ['AnyUrl', 'AnyHttpUrl', 'FileUrl', 'FtpUrl', 'HttpUrl', 'WebsocketUrl', 'AnyWebsocketUrl', 'UrlConstraints', 'EmailStr', 'NameEmail', 'IPvAnyAddress', 'IPvAnyInterface', 'IPvAnyNetwork', 'PostgresDsn', 'CockroachDsn', 'AmqpDsn', 'RedisDsn', 'MongoDsn', 'KafkaDsn', 'NatsDsn', 'validate_email', 'MySQLDsn', 'MariaDBDsn', 'ClickHouseDsn', 'SnowflakeDsn']

@dataclasses.dataclass
class UrlConstraints:
    max_length: Any = None
    allowed_schemes: Any = None
    host_required: Any = None
    default_host: Any = None
    default_port: Any = None
    default_path: Any = None

    def __hash__(self) -> Any:
        return hash((self.max_length, tuple(self.allowed_schemes) if self.allowed_schemes is not None else None, self.host_required, self.default_host, self.default_port, self.default_path))

    @property
    def defined_constraints(self) -> Any:
        return {field.name: value for field in fields(self) if (value := getattr(self, field.name)) is not None}

    def __get_pydantic_core_schema__(self, source: Any, handler: Any) -> Any:
        schema = handler(source)
        schema_to_mutate = schema['schema'] if schema['type'] == 'function-wrap' else schema
        if (annotated_type := (schema_to_mutate['type'] not in ('url', 'multi-host-url'))):
            raise PydanticUserError(f"'UrlConstraints' cannot annotate '{annotated_type}'.", code='invalid-annotated-type')
        for constraint_key, constraint_value in self.defined_constraints.items():
            schema_to_mutate[constraint_key] = constraint_value
        return schema

class _BaseUrl:
    _constraints: UrlConstraints = UrlConstraints()

    def __init__(self, url: Any) -> None:
        self._url = _build_type_adapter(self.__class__).validate_python(url)._url

    @property
    def scheme(self) -> Any:
        return self._url.scheme

    @property
    def username(self) -> Any:
        return self._url.username

    @property
    def password(self) -> Any:
        return self._url.password

    @property
    def host(self) -> Any:
        return self._url.host

    def unicode_host(self) -> Any:
        return self._url.unicode_host()

    @property
    def port(self) -> Any:
        return self._url.port

    @property
    def path(self) -> Any:
        return self._url.path

    @property
    def query(self) -> Any:
        return self._url.query

    def query_params(self) -> Any:
        return self._url.query_params()

    @property
    def fragment(self) -> Any:
        return self._url.fragment

    def unicode_string(self) -> Any:
        return self._url.unicode_string()

    def __str__(self) -> Any:
        return str(self._url)

    def __repr__(self) -> Any:
        return f'{self.__class__.__name__}({str(self._url)!r})'

    def __deepcopy__(self, memo: Any) -> Any:
        return self.__class__(self._url)

    def __eq__(self, other: Any) -> Any:
        return self.__class__ is other.__class__ and self._url == other._url

    def __lt__(self, other: Any) -> Any:
        return self.__class__ is other.__class__ and self._url < other._url

    def __gt__(self, other: Any) -> Any:
        return self.__class__ is other.__class__ and self._url > other._url

    def __le__(self, other: Any) -> Any:
        return self.__class__ is other.__class__ and self._url <= other._url

    def __ge__(self, other: Any) -> Any:
        return self.__class__ is other.__class__ and self._url >= other._url

    def __hash__(self) -> Any:
        return hash(self._url)

    def __len__(self) -> Any:
        return len(str(self._url))

    @classmethod
    def build(cls, *, scheme: Any, username: Any = None, password: Any = None, host: Any, port: Any = None, path: Any = None, query: Any = None, fragment: Any = None) -> Any:
        return cls(_CoreUrl.build(scheme=scheme, username=username, password=password, host=host, port=port, path=path, query=query, fragment=fragment))

    @classmethod
    def serialize_url(cls, url: Any, info: Any) -> Any:
        if not isinstance(url, cls):
            raise PydanticSerializationUnexpectedValue(f"Expected `{cls}` but got `{type(url)}` with value `'{url}'` - serialized value may not be as expected.")
        if info.mode == 'json':
            return str(url)
        return url

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:

        def wrap_val(v: Any, h: Any) -> Any:
            if isinstance(v, source):
                return v
            if isinstance(v, _BaseUrl):
                v = str(v)
            core_url = h(v)
            instance = source.__new__(source)
            instance._url = core_url
            return instance
        return core_schema.no_info_wrap_validator_function(wrap_val, schema=core_schema.url_schema(**cls._constraints.defined_constraints), serialization=core_schema.plain_serializer_function_ser_schema(cls.serialize_url, info_arg=True, when_used='always'))

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Any) -> Any:
        inner_schema = core_schema['schema'] if core_schema['type'] == 'function-wrap' else core_schema
        return handler(inner_schema)
    __pydantic_serializer__ = SchemaSerializer(core_schema.any_schema(serialization=core_schema.to_string_ser_schema()))

class _BaseMultiHostUrl:
    _constraints: UrlConstraints = UrlConstraints()

    def __init__(self, url: Any) -> None:
        self._url = _build_type_adapter(self.__class__).validate_python(url)._url

    @property
    def scheme(self) -> Any:
        return self._url.scheme

    @property
    def path(self) -> Any:
        return self._url.path

    @property
    def query(self) -> Any:
        return self._url.query

    def query_params(self) -> Any:
        return self._url.query_params()

    @property
    def fragment(self) -> Any:
        return self._url.fragment

    def hosts(self) -> Any:
        return self._url.hosts()

    def unicode_string(self) -> Any:
        return self._url.unicode_string()

    def __str__(self) -> Any:
        return str(self._url)

    def __repr__(self) -> Any:
        return f'{self.__class__.__name__}({str(self._url)!r})'

    def __deepcopy__(self, memo: Any) -> Any:
        return self.__class__(self._url)

    def __eq__(self, other: Any) -> Any:
        return self.__class__ is other.__class__ and self._url == other._url

    def __hash__(self) -> Any:
        return hash(self._url)

    def __len__(self) -> Any:
        return len(str(self._url))

    @classmethod
    def build(cls, *, scheme: Any, hosts: Any = None, username: Any = None, password: Any = None, host: Any = None, port: Any = None, path: Any = None, query: Any = None, fragment: Any = None) -> Any:
        return cls(_CoreMultiHostUrl.build(scheme=scheme, hosts=hosts, username=username, password=password, host=host, port=port, path=path, query=query, fragment=fragment))

    @classmethod
    def serialize_url(cls, url: Any, info: Any) -> Any:
        if not isinstance(url, cls):
            raise PydanticSerializationUnexpectedValue(f"Expected `{cls}` but got `{type(url)}` with value `'{url}'` - serialized value may not be as expected.")
        if info.mode == 'json':
            return str(url)
        return url

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:

        def wrap_val(v: Any, h: Any) -> Any:
            if isinstance(v, source):
                return v
            if isinstance(v, _BaseMultiHostUrl):
                v = str(v)
            core_url = h(v)
            instance = source.__new__(source)
            instance._url = core_url
            return instance
        return core_schema.no_info_wrap_validator_function(wrap_val, schema=core_schema.multi_host_url_schema(**cls._constraints.defined_constraints), serialization=core_schema.plain_serializer_function_ser_schema(cls.serialize_url, info_arg=True, when_used='always'))

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Any) -> Any:
        inner_schema = core_schema['schema'] if core_schema['type'] == 'function-wrap' else core_schema
        return handler(inner_schema)
    __pydantic_serializer__ = SchemaSerializer(core_schema.any_schema(serialization=core_schema.to_string_ser_schema()))

@lru_cache
def _build_type_adapter(cls: Any) -> Any:
    return TypeAdapter(cls)

class AnyUrl(_BaseUrl):
    pass

class AnyHttpUrl(AnyUrl):
    _constraints: UrlConstraints = UrlConstraints(allowed_schemes=['http', 'https'])

class HttpUrl(AnyUrl):
    _constraints: UrlConstraints = UrlConstraints(max_length=2083, allowed_schemes=['http', 'https'])

class AnyWebsocketUrl(AnyUrl):
    _constraints: UrlConstraints = UrlConstraints(allowed_schemes=['ws', 'wss'])

class WebsocketUrl(AnyUrl):
    _constraints: UrlConstraints = UrlConstraints(max_length=2083, allowed_schemes=['ws', 'wss'])

class FileUrl(AnyUrl):
    _constraints: UrlConstraints = UrlConstraints(allowed_schemes=['file'])

class FtpUrl(AnyUrl):
    _constraints: UrlConstraints = UrlConstraints(allowed_schemes=['ftp'])

class PostgresDsn(_BaseMultiHostUrl):
    _constraints: UrlConstraints = UrlConstraints(host_required=True, allowed_schemes=['postgres', 'postgresql', 'postgresql+asyncpg', 'postgresql+pg8000', 'postgresql+psycopg', 'postgresql+psycopg2', 'postgresql+psycopg2cffi', 'postgresql+py-postgresql', 'postgresql+pygresql'])

    @property
    def host(self) -> Any:
        return self._url.host

class CockroachDsn(AnyUrl):
    _constraints: UrlConstraints = UrlConstraints(host_required=True, allowed_schemes=['cockroachdb', 'cockroachdb+psycopg2', 'cockroachdb+asyncpg'])

    @property
    def host(self) -> Any:
        return self._url.host

class AmqpDsn(AnyUrl):
    _constraints: UrlConstraints = UrlConstraints(allowed_schemes=['amqp', 'amqps'])

class RedisDsn(AnyUrl):
    _constraints: UrlConstraints = UrlConstraints(allowed_schemes=['redis', 'rediss'], default_host='localhost', default_port=6379, default_path='/0', host_required=True)

    @property
    def host(self) -> Any:
        return self._url.host

class MongoDsn(_BaseMultiHostUrl):
    _constraints: UrlConstraints = UrlConstraints(allowed_schemes=['mongodb', 'mongodb+srv'], default_port=27017)

class KafkaDsn(AnyUrl):
    _constraints: UrlConstraints = UrlConstraints(allowed_schemes=['kafka'], default_host='localhost', default_port=9092)

class NatsDsn(_BaseMultiHostUrl):
    _constraints: UrlConstraints = UrlConstraints(allowed_schemes=['nats', 'tls', 'ws', 'wss'], default_host='localhost', default_port=4222)

class MySQLDsn(AnyUrl):
    _constraints: UrlConstraints = UrlConstraints(allowed_schemes=['mysql', 'mysql+mysqlconnector', 'mysql+aiomysql', 'mysql+asyncmy', 'mysql+mysqldb', 'mysql+pymysql', 'mysql+cymysql', 'mysql+pyodbc'], default_port=3306, host_required=True)

class MariaDBDsn(AnyUrl):
    _constraints: UrlConstraints = UrlConstraints(allowed_schemes=['mariadb', 'mariadb+mariadbconnector', 'mariadb+pymysql'], default_port=3306)

class ClickHouseDsn(AnyUrl):
    _constraints: UrlConstraints = UrlConstraints(allowed_schemes=['clickhouse+native', 'clickhouse+asynch', 'clickhouse+http', 'clickhouse', 'clickhouses', 'clickhousedb'], default_host='localhost', default_port=9000)

class SnowflakeDsn(AnyUrl):
    _constraints: UrlConstraints = UrlConstraints(allowed_schemes=['snowflake'], host_required=True)

    @property
    def host(self) -> Any:
        return self._url.host

def import_email_validator() -> Any:
    global email_validator
    try:
        import email_validator
    except ImportError as e:
        raise ImportError('email-validator is not installed, run `pip install pydantic[email]`') from e
    if not version('email-validator').partition('.')[0] == '2':
        raise ImportError('email-validator version >= 2.0 required, run pip install -U email-validator')

class EmailStr:
    @classmethod
    def __get_pydantic_core_schema__(cls, _source: Any, _handler: Any) -> Any:
        import_email_validator()
        return core_schema.no_info_after_validator_function(cls._validate, core_schema.str_schema())

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Any) -> Any:
        field_schema = handler(core_schema)
        field_schema.update(type='string', format='email')
        return field_schema

    @classmethod
    def _validate(cls, input_value: Any, /) -> Any:
        return validate_email(input_value)[1]

class NameEmail:
    __slots__ = ('name', 'email')

    def __init__(self, name: Any, email: Any) -> None:
        self.name = name
        self.email = email

    def __eq__(self, other: Any) -> Any:
        return isinstance(other, NameEmail) and (self.name, self.email) == (other.name, other.email)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Any) -> Any:
        field_schema = handler(core_schema)
        field_schema.update(type='string', format='name-email')
        return field_schema

    @classmethod
    def __get_pydantic_core_schema__(cls, _source: Any, _handler: Any) -> Any:
        import_email_validator()
        return core_schema.no_info_after_validator_function(cls._validate, core_schema.json_or_python_schema(json_schema=core_schema.str_schema(), python_schema=core_schema.union_schema([core_schema.is_instance_schema(cls), core_schema.str_schema()], custom_error_type='name_email_type', custom_error_message='Input is not a valid NameEmail'), serialization=core_schema.to_string_ser_schema())

    @classmethod
    def _validate(cls, input_value: Any, /) -> Any:
        if isinstance(input_value, str):
            name, email = validate_email(input_value)
            return cls(name, email)
        else:
            return input_value

    def __str__(self) -> Any:
        if '@' in self.name:
            return f'"{self.name}" <{self.email}>'
        return f'{self.name} <{self.email}>'

class IPvAnyAddress:
    __slots__ = ()

    def __new__(cls, value: Any) -> Any:
        try:
            return IPv4Address(value)
        except ValueError:
            pass
        try:
            return IPv6Address(value)
        except ValueError:
            raise PydanticCustomError('ip_any_address', 'value is not a valid IPv4 or IPv6 address')

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Any) -> Any:
        field_schema = {}
        field_schema.update(type='string', format='ipvanyaddress')
        return field_schema

    @classmethod
    def __get_pydantic_core_schema__(cls, _source: Any, _handler: Any) -> Any:
        return core_schema.no_info_plain_validator_function(cls._validate, serialization=core_schema.to_string_ser_schema())

    @classmethod
    def _validate(cls, input_value: Any, /) -> Any:
        return cls(input_value)

class IPvAnyInterface:
    __slots__ = ()

    def __new__(cls, value: Any) -> Any:
        try:
            return IPv4Interface(value)
        except ValueError:
            pass
        try:
            return IPv6Interface(value)
        except ValueError:
            raise PydanticCustomError('ip_any_interface', 'value is not a valid IPv4 or IPv6 interface')

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Any) -> Any:
        field_schema = {}
        field_schema.update(type='string', format='ipvanyinterface')
        return field_schema

    @classmethod
    def __get_pydantic_core_schema__(cls,