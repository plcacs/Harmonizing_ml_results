from __future__ import annotations
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
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)
from pydantic_core import (
    MultiHostHost,
    PydanticCustomError,
    PydanticSerializationUnexpectedValue,
    SchemaSerializer,
    core_schema,
)
from pydantic_core import MultiHostUrl as _CoreMultiHostUrl
from pydantic_core import Url as _CoreUrl
from typing_extensions import Self
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
            self.default_path,
        ))

    @property
    def defined_constraints(self) -> Dict[str, Any]:
        return {field.name: value for field in fields(self) if (value := getattr(self, field.name)) is not None}

    def __get_pydantic_core_schema__(self, source: Any, handler: Any) -> Any:
        schema: Dict[str, Any] = handler(source)
        schema_to_mutate: Dict[str, Any]
        if schema.get('type') == 'function-wrap':
            schema_to_mutate = schema['schema']
        else:
            schema_to_mutate = schema
        annotated_type: Any = schema_to_mutate.get('type')
        if annotated_type not in ('url', 'multi-host-url'):
            raise PydanticUserError(f"'UrlConstraints' cannot annotate '{annotated_type}'.", code='invalid-annotated-type')
        for constraint_key, constraint_value in self.defined_constraints.items():
            schema_to_mutate[constraint_key] = constraint_value
        return schema


class _BaseUrl:
    _constraints: ClassVar[UrlConstraints] = UrlConstraints()

    def __init__(self, url: Any) -> None:
        self._url: Any = _build_type_adapter(self.__class__).validate_python(url)._url

    @property
    def scheme(self) -> str:
        return self._url.scheme

    @property
    def username(self) -> Optional[str]:
        return self._url.username

    @property
    def password(self) -> Optional[str]:
        return self._url.password

    @property
    def host(self) -> Optional[str]:
        return self._url.host

    def unicode_host(self) -> Optional[str]:
        return self._url.unicode_host()

    @property
    def port(self) -> Optional[int]:
        return self._url.port

    @property
    def path(self) -> Optional[str]:
        return self._url.path

    @property
    def query(self) -> Optional[str]:
        return self._url.query

    def query_params(self) -> List[Tuple[str, str]]:
        return self._url.query_params()

    @property
    def fragment(self) -> Optional[str]:
        return self._url.fragment

    def unicode_string(self) -> str:
        return self._url.unicode_string()

    def __str__(self) -> str:
        return str(self._url)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({str(self._url)!r})'

    def __deepcopy__(self, memo: Dict[int, Any]) -> Self:
        return self.__class__(self._url)

    def __eq__(self, other: Any) -> bool:
        return self.__class__ is getattr(other, "__class__", None) and self._url == other._url

    def __lt__(self, other: Any) -> bool:
        return self.__class__ is getattr(other, "__class__", None) and self._url < other._url

    def __gt__(self, other: Any) -> bool:
        return self.__class__ is getattr(other, "__class__", None) and self._url > other._url

    def __le__(self, other: Any) -> bool:
        return self.__class__ is getattr(other, "__class__", None) and self._url <= other._url

    def __ge__(self, other: Any) -> bool:
        return self.__class__ is getattr(other, "__class__", None) and self._url >= other._url

    def __hash__(self) -> int:
        return hash(self._url)

    def __len__(self) -> int:
        return len(str(self._url))

    @classmethod
    def build(
        cls: Type[Self],
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
    def serialize_url(cls, url: Any, info: Any) -> Any:
        if not isinstance(url, cls):
            raise PydanticSerializationUnexpectedValue(
                f"Expected `{cls}` but got `{type(url)}` with value `'{url}'` - serialized value may not be as expected."
            )
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
            core_url: Any = h(v)
            instance: Any = source.__new__(source)
            instance._url = core_url
            return instance

        return core_schema.no_info_wrap_validator_function(
            wrap_val,
            schema=core_schema.url_schema(**cls._constraints.defined_constraints),
            serialization=core_schema.plain_serializer_function_ser_schema(cls.serialize_url, info_arg=True, when_used='always'),
        )

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema_value: Any, handler: Any) -> Any:
        inner_schema = core_schema_value['schema'] if core_schema_value.get('type') == 'function-wrap' else core_schema_value
        return handler(inner_schema)

    __pydantic_serializer__ = SchemaSerializer(core_schema.any_schema(serialization=core_schema.to_string_ser_schema()))


class _BaseMultiHostUrl:
    _constraints: ClassVar[UrlConstraints] = UrlConstraints()

    def __init__(self, url: Any) -> None:
        self._url: Any = _build_type_adapter(self.__class__).validate_python(url)._url

    @property
    def scheme(self) -> str:
        return self._url.scheme

    @property
    def path(self) -> Optional[str]:
        return self._url.path

    @property
    def query(self) -> Optional[str]:
        return self._url.query

    def query_params(self) -> List[Tuple[str, str]]:
        return self._url.query_params()

    @property
    def fragment(self) -> Optional[str]:
        return self._url.fragment

    def hosts(self) -> List[MultiHostHost]:
        return self._url.hosts()

    def unicode_string(self) -> str:
        return self._url.unicode_string()

    def __str__(self) -> str:
        return str(self._url)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({str(self._url)!r})'

    def __deepcopy__(self, memo: Dict[int, Any]) -> Self:
        return self.__class__(self._url)

    def __eq__(self, other: Any) -> bool:
        return self.__class__ is getattr(other, "__class__", None) and self._url == other._url

    def __hash__(self) -> int:
        return hash(self._url)

    def __len__(self) -> int:
        return len(str(self._url))

    @classmethod
    def build(
        cls: Type[Self],
        *,
        scheme: str,
        hosts: Optional[List[MultiHostHost]] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        path: Optional[str] = None,
        query: Optional[str] = None,
        fragment: Optional[str] = None,
    ) -> Self:
        return cls(
            _CoreMultiHostUrl.build(
                scheme=scheme,
                hosts=hosts,
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
    def serialize_url(cls, url: Any, info: Any) -> Any:
        if not isinstance(url, cls):
            raise PydanticSerializationUnexpectedValue(
                f"Expected `{cls}` but got `{type(url)}` with value `'{url}'` - serialized value may not be as expected."
            )
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
            core_url: Any = h(v)
            instance: Any = source.__new__(source)
            instance._url = core_url
            return instance

        return core_schema.no_info_wrap_validator_function(
            wrap_val,
            schema=core_schema.multi_host_url_schema(**cls._constraints.defined_constraints),
            serialization=core_schema.plain_serializer_function_ser_schema(cls.serialize_url, info_arg=True, when_used='always'),
        )

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema_value: Any, handler: Any) -> Any:
        inner_schema = core_schema_value['schema'] if core_schema_value.get('type') == 'function-wrap' else core_schema_value
        return handler(inner_schema)

    __pydantic_serializer__ = SchemaSerializer(core_schema.any_schema(serialization=core_schema.to_string_ser_schema()))


@lru_cache
def _build_type_adapter(cls: Type[Any]) -> TypeAdapter:
    return TypeAdapter(cls)


class AnyUrl(_BaseUrl):
    """
    Base type for all URLs.
    """


class AnyHttpUrl(AnyUrl):
    _constraints: ClassVar[UrlConstraints] = UrlConstraints(allowed_schemes=['http', 'https'])


class HttpUrl(AnyUrl):
    _constraints: ClassVar[UrlConstraints] = UrlConstraints(max_length=2083, allowed_schemes=['http', 'https'])


class AnyWebsocketUrl(AnyUrl):
    _constraints: ClassVar[UrlConstraints] = UrlConstraints(allowed_schemes=['ws', 'wss'])


class WebsocketUrl(AnyUrl):
    _constraints: ClassVar[UrlConstraints] = UrlConstraints(max_length=2083, allowed_schemes=['ws', 'wss'])


class FileUrl(AnyUrl):
    _constraints: ClassVar[UrlConstraints] = UrlConstraints(allowed_schemes=['file'])


class FtpUrl(AnyUrl):
    _constraints: ClassVar[UrlConstraints] = UrlConstraints(allowed_schemes=['ftp'])


class PostgresDsn(_BaseMultiHostUrl):
    _constraints: ClassVar[UrlConstraints] = UrlConstraints(
        host_required=True,
        allowed_schemes=[
            'postgres',
            'postgresql',
            'postgresql+asyncpg',
            'postgresql+pg8000',
            'postgresql+psycopg',
            'postgresql+psycopg2',
            'postgresql+psycopg2cffi',
            'postgresql+py-postgresql',
            'postgresql+pygresql',
        ]
    )

    @property
    def host(self) -> str:
        return self._url.host


class CockroachDsn(AnyUrl):
    _constraints: ClassVar[UrlConstraints] = UrlConstraints(
        host_required=True,
        allowed_schemes=['cockroachdb', 'cockroachdb+psycopg2', 'cockroachdb+asyncpg'],
    )

    @property
    def host(self) -> str:
        return self._url.host


class AmqpDsn(AnyUrl):
    _constraints: ClassVar[UrlConstraints] = UrlConstraints(allowed_schemes=['amqp', 'amqps'])


class RedisDsn(AnyUrl):
    _constraints: ClassVar[UrlConstraints] = UrlConstraints(
        allowed_schemes=['redis', 'rediss'],
        default_host='localhost',
        default_port=6379,
        default_path='/0',
        host_required=True,
    )

    @property
    def host(self) -> str:
        return self._url.host


class MongoDsn(_BaseMultiHostUrl):
    _constraints: ClassVar[UrlConstraints] = UrlConstraints(allowed_schemes=['mongodb', 'mongodb+srv'], default_port=27017)


class KafkaDsn(AnyUrl):
    _constraints: ClassVar[UrlConstraints] = UrlConstraints(allowed_schemes=['kafka'], default_host='localhost', default_port=9092)


class NatsDsn(_BaseMultiHostUrl):
    _constraints: ClassVar[UrlConstraints] = UrlConstraints(
        allowed_schemes=['nats', 'tls', 'ws', 'wss'],
        default_host='localhost',
        default_port=4222,
    )


class MySQLDsn(AnyUrl):
    _constraints: ClassVar[UrlConstraints] = UrlConstraints(
        allowed_schemes=[
            'mysql',
            'mysql+mysqlconnector',
            'mysql+aiomysql',
            'mysql+asyncmy',
            'mysql+mysqldb',
            'mysql+pymysql',
            'mysql+cymysql',
            'mysql+pyodbc',
        ],
        default_port=3306,
        host_required=True,
    )


class MariaDBDsn(AnyUrl):
    _constraints: ClassVar[UrlConstraints] = UrlConstraints(
        allowed_schemes=['mariadb', 'mariadb+mariadbconnector', 'mariadb+pymysql'],
        default_port=3306,
    )


class ClickHouseDsn(AnyUrl):
    _constraints: ClassVar[UrlConstraints] = UrlConstraints(
        allowed_schemes=['clickhouse+native', 'clickhouse+asynch', 'clickhouse+http', 'clickhouse', 'clickhouses', 'clickhousedb'],
        default_host='localhost',
        default_port=9000,
    )


class SnowflakeDsn(AnyUrl):
    _constraints: ClassVar[UrlConstraints] = UrlConstraints(allowed_schemes=['snowflake'], host_required=True)

    @property
    def host(self) -> str:
        return self._url.host


def import_email_validator() -> None:
    global email_validator
    try:
        import email_validator  # type: ignore
    except ImportError as e:
        raise ImportError('email-validator is not installed, run `pip install pydantic[email]`') from e
    if not version('email-validator').partition('.')[0] == '2':
        raise ImportError('email-validator version >= 2.0 required, run pip install -U email-validator')


if TYPE_CHECKING:
    EmailStr: type = str  # type: ignore
else:

    class EmailStr:
        """
        Validate email addresses.
        """

        @classmethod
        def __get_pydantic_core_schema__(cls, _source: Any, _handler: Any) -> Any:
            import_email_validator()
            return core_schema.no_info_after_validator_function(cls._validate, core_schema.str_schema())

        @classmethod
        def __get_pydantic_json_schema__(cls, core_schema_value: Any, handler: Any) -> Any:
            field_schema: Dict[str, Any] = handler(core_schema_value)
            field_schema.update(type='string', format='email')
            return field_schema

        @classmethod
        def _validate(cls, input_value: Any, /) -> Any:
            return validate_email(input_value)[1]


class NameEmail(_repr.Representation):
    __slots__ = ('name', 'email')

    def __init__(self, name: str, email: str) -> None:
        self.name: str = name
        self.email: str = email

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, NameEmail) and (self.name, self.email) == (other.name, other.email)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema_value: Any, handler: Any) -> Any:
        field_schema: Dict[str, Any] = handler(core_schema_value)
        field_schema.update(type='string', format='name-email')
        return field_schema

    @classmethod
    def __get_pydantic_core_schema__(cls, _source: Any, _handler: Any) -> Any:
        import_email_validator()
        return core_schema.no_info_after_validator_function(
            cls._validate,
            core_schema.json_or_python_schema(
                json_schema=core_schema.str_schema(),
                python_schema=core_schema.union_schema(
                    [core_schema.is_instance_schema(cls), core_schema.str_schema()],
                    custom_error_type='name_email_type',
                    custom_error_message='Input is not a valid NameEmail',
                ),
                serialization=core_schema.to_string_ser_schema(),
            ),
        )

    @classmethod
    def _validate(cls, input_value: Any, /) -> Any:
        if isinstance(input_value, str):
            name, email = validate_email(input_value)
            return cls(name, email)
        else:
            return input_value

    def __str__(self) -> str:
        if '@' in self.name:
            return f'"{self.name}" <{self.email}>'
        return f'{self.name} <{self.email}>'


IPvAnyAddressType = Union[IPv4Address, IPv6Address]
IPvAnyInterfaceType = Union[IPv4Interface, IPv6Interface]
IPvAnyNetworkType = Union[IPv4Network, IPv6Network]

if TYPE_CHECKING:
    IPvAnyAddress: type = IPvAnyAddressType  # type: ignore
    IPvAnyInterface: type = IPvAnyInterfaceType  # type: ignore
    IPvAnyNetwork: type = IPvAnyNetworkType  # type: ignore
else:

    class IPvAnyAddress:
        __slots__ = ()

        def __new__(cls, value: Any) -> IPvAnyAddressType:
            try:
                return IPv4Address(value)
            except ValueError:
                pass
            try:
                return IPv6Address(value)
            except ValueError:
                raise PydanticCustomError('ip_any_address', 'value is not a valid IPv4 or IPv6 address')

        @classmethod
        def __get_pydantic_json_schema__(cls, core_schema_value: Any, handler: Any) -> Any:
            field_schema: Dict[str, Any] = {}
            field_schema.update(type='string', format='ipvanyaddress')
            return field_schema

        @classmethod
        def __get_pydantic_core_schema__(cls, _source: Any, _handler: Any) -> Any:
            return core_schema.no_info_plain_validator_function(cls._validate, serialization=core_schema.to_string_ser_schema())

        @classmethod
        def _validate(cls, input_value: Any, /) -> IPvAnyAddressType:
            return cls(input_value)

    class IPvAnyInterface:
        __slots__ = ()

        def __new__(cls, value: Any) -> Union[IPv4Interface, IPv6Interface]:
            try:
                return IPv4Interface(value)
            except ValueError:
                pass
            try:
                return IPv6Interface(value)
            except ValueError:
                raise PydanticCustomError('ip_any_interface', 'value is not a valid IPv4 or IPv6 interface')

        @classmethod
        def __get_pydantic_json_schema__(cls, core_schema_value: Any, handler: Any) -> Any:
            field_schema: Dict[str, Any] = {}
            field_schema.update(type='string', format='ipvanyinterface')
            return field_schema

        @classmethod
        def __get_pydantic_core_schema__(cls, _source: Any, _handler: Any) -> Any:
            return core_schema.no_info_plain_validator_function(cls._validate, serialization=core_schema.to_string_ser_schema())

        @classmethod
        def _validate(cls, input_value: Any, /) -> Union[IPv4Interface, IPv6Interface]:
            return cls(input_value)

    class IPvAnyNetwork:
        __slots__ = ()

        def __new__(cls, value: Any) -> Union[IPv4Network, IPv6Network]:
            try:
                return IPv4Network(value)
            except ValueError:
                pass
            try:
                return IPv6Network(value)
            except ValueError:
                raise PydanticCustomError('ip_any_network', 'value is not a valid IPv4 or IPv6 network')

        @classmethod
        def __get_pydantic_json_schema__(cls, core_schema_value: Any, handler: Any) -> Any:
            field_schema: Dict[str, Any] = {}
            field_schema.update(type='string', format='ipvanynetwork')
            return field_schema

        @classmethod
        def __get_pydantic_core_schema__(cls, _source: Any, _handler: Any) -> Any:
            return core_schema.no_info_plain_validator_function(cls._validate, serialization=core_schema.to_string_ser_schema())

        @classmethod
        def _validate(cls, input_value: Any, /) -> Union[IPv4Network, IPv6Network]:
            return cls(input_value)


def _build_pretty_email_regex() -> re.Pattern:
    name_chars: str = r"[\w!#$%&\\'*+\-/=?^_`{|}~]"
    unquoted_name_group: str = f'((?:{name_chars}+\\s+)*{name_chars}+)'
    quoted_name_group: str = r'"((?:[^"]|\\")+)"'
    email_group: str = r'<(.+)>'
    return re.compile(rf'\s*(?:{unquoted_name_group}|{quoted_name_group})?\s*{email_group}\s*')


pretty_email_regex: re.Pattern = _build_pretty_email_regex()
MAX_EMAIL_LENGTH: int = 2048


def validate_email(value: str) -> Tuple[str, str]:
    if email_validator is None:
        import_email_validator()
    if len(value) > MAX_EMAIL_LENGTH:
        raise PydanticCustomError('value_error', 'value is not a valid email address: {reason}', {'reason': f'Length must not exceed {MAX_EMAIL_LENGTH} characters'})
    m: Optional[re.Match[str]] = pretty_email_regex.fullmatch(value)
    name: Optional[str] = None
    if m:
        unquoted_name, quoted_name, value = m.groups()
        name = unquoted_name or quoted_name
    email: str = value.strip()
    try:
        parts = email_validator.validate_email(email, check_deliverability=False)  # type: ignore
    except email_validator.EmailNotValidError as e:  # type: ignore
        raise PydanticCustomError('value_error', 'value is not a valid email address: {reason}', {'reason': str(e.args[0])}) from e
    email = parts.normalized  # type: ignore
    assert email is not None
    name = name or parts.local_part  # type: ignore
    return (name, email)


__getattr__ = getattr_migration(__name__)