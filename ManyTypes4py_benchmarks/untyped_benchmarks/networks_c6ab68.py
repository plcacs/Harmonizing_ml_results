"""The networks module contains types for common network-related fields."""
from __future__ import annotations as _annotations
import dataclasses as _dataclasses
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
    max_length = None
    allowed_schemes = None
    host_required = None
    default_host = None
    default_port = None
    default_path = None

    def __hash__(self):
        return hash((self.max_length, tuple(self.allowed_schemes) if self.allowed_schemes is not None else None, self.host_required, self.default_host, self.default_port, self.default_path))

    @property
    def defined_constraints(self):
        """Fetch a key / value mapping of constraints to values that are not None. Used for core schema updates."""
        return {field.name: value for field in fields(self) if (value := getattr(self, field.name)) is not None}

    def __get_pydantic_core_schema__(self, source, handler):
        schema = handler(source)
        schema_to_mutate = schema['schema'] if schema['type'] == 'function-wrap' else schema
        if (annotated_type := (schema_to_mutate['type'] not in ('url', 'multi-host-url'))):
            raise PydanticUserError(f"'UrlConstraints' cannot annotate '{annotated_type}'.", code='invalid-annotated-type')
        for constraint_key, constraint_value in self.defined_constraints.items():
            schema_to_mutate[constraint_key] = constraint_value
        return schema

class _BaseUrl:
    _constraints = UrlConstraints()

    def __init__(self, url):
        self._url = _build_type_adapter(self.__class__).validate_python(url)._url

    @property
    def scheme(self):
        """The scheme part of the URL.

        e.g. `https` in `https://user:pass@host:port/path?query#fragment`
        """
        return self._url.scheme

    @property
    def username(self):
        """The username part of the URL, or `None`.

        e.g. `user` in `https://user:pass@host:port/path?query#fragment`
        """
        return self._url.username

    @property
    def password(self):
        """The password part of the URL, or `None`.

        e.g. `pass` in `https://user:pass@host:port/path?query#fragment`
        """
        return self._url.password

    @property
    def host(self):
        """The host part of the URL, or `None`.

        If the URL must be punycode encoded, this is the encoded host, e.g if the input URL is `https://£££.com`,
        `host` will be `xn--9aaa.com`
        """
        return self._url.host

    def unicode_host(self):
        """The host part of the URL as a unicode string, or `None`.

        e.g. `host` in `https://user:pass@host:port/path?query#fragment`

        If the URL must be punycode encoded, this is the decoded host, e.g if the input URL is `https://£££.com`,
        `unicode_host()` will be `£££.com`
        """
        return self._url.unicode_host()

    @property
    def port(self):
        """The port part of the URL, or `None`.

        e.g. `port` in `https://user:pass@host:port/path?query#fragment`
        """
        return self._url.port

    @property
    def path(self):
        """The path part of the URL, or `None`.

        e.g. `/path` in `https://user:pass@host:port/path?query#fragment`
        """
        return self._url.path

    @property
    def query(self):
        """The query part of the URL, or `None`.

        e.g. `query` in `https://user:pass@host:port/path?query#fragment`
        """
        return self._url.query

    def query_params(self):
        """The query part of the URL as a list of key-value pairs.

        e.g. `[('foo', 'bar')]` in `https://user:pass@host:port/path?foo=bar#fragment`
        """
        return self._url.query_params()

    @property
    def fragment(self):
        """The fragment part of the URL, or `None`.

        e.g. `fragment` in `https://user:pass@host:port/path?query#fragment`
        """
        return self._url.fragment

    def unicode_string(self):
        """The URL as a unicode string, unlike `__str__()` this will not punycode encode the host.

        If the URL must be punycode encoded, this is the decoded string, e.g if the input URL is `https://£££.com`,
        `unicode_string()` will be `https://£££.com`
        """
        return self._url.unicode_string()

    def __str__(self):
        """The URL as a string, this will punycode encode the host if required."""
        return str(self._url)

    def __repr__(self):
        return f'{self.__class__.__name__}({str(self._url)!r})'

    def __deepcopy__(self, memo):
        return self.__class__(self._url)

    def __eq__(self, other):
        return self.__class__ is other.__class__ and self._url == other._url

    def __lt__(self, other):
        return self.__class__ is other.__class__ and self._url < other._url

    def __gt__(self, other):
        return self.__class__ is other.__class__ and self._url > other._url

    def __le__(self, other):
        return self.__class__ is other.__class__ and self._url <= other._url

    def __ge__(self, other):
        return self.__class__ is other.__class__ and self._url >= other._url

    def __hash__(self):
        return hash(self._url)

    def __len__(self):
        return len(str(self._url))

    @classmethod
    def build(cls, *, scheme, username=None, password=None, host, port=None, path=None, query=None, fragment=None):
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
        return cls(_CoreUrl.build(scheme=scheme, username=username, password=password, host=host, port=port, path=path, query=query, fragment=fragment))

    @classmethod
    def serialize_url(cls, url, info):
        if not isinstance(url, cls):
            raise PydanticSerializationUnexpectedValue(f"Expected `{cls}` but got `{type(url)}` with value `'{url}'` - serialized value may not be as expected.")
        if info.mode == 'json':
            return str(url)
        return url

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):

        def wrap_val(v, h):
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
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        inner_schema = core_schema['schema'] if core_schema['type'] == 'function-wrap' else core_schema
        return handler(inner_schema)
    __pydantic_serializer__ = SchemaSerializer(core_schema.any_schema(serialization=core_schema.to_string_ser_schema()))

class _BaseMultiHostUrl:
    _constraints = UrlConstraints()

    def __init__(self, url):
        self._url = _build_type_adapter(self.__class__).validate_python(url)._url

    @property
    def scheme(self):
        """The scheme part of the URL.

        e.g. `https` in `https://foo.com,bar.com/path?query#fragment`
        """
        return self._url.scheme

    @property
    def path(self):
        """The path part of the URL, or `None`.

        e.g. `/path` in `https://foo.com,bar.com/path?query#fragment`
        """
        return self._url.path

    @property
    def query(self):
        """The query part of the URL, or `None`.

        e.g. `query` in `https://foo.com,bar.com/path?query#fragment`
        """
        return self._url.query

    def query_params(self):
        """The query part of the URL as a list of key-value pairs.

        e.g. `[('foo', 'bar')]` in `https://foo.com,bar.com/path?query#fragment`
        """
        return self._url.query_params()

    @property
    def fragment(self):
        """The fragment part of the URL, or `None`.

        e.g. `fragment` in `https://foo.com,bar.com/path?query#fragment`
        """
        return self._url.fragment

    def hosts(self):
        '''The hosts of the `MultiHostUrl` as [`MultiHostHost`][pydantic_core.MultiHostHost] typed dicts.

        ```python
        from pydantic_core import MultiHostUrl

        mhu = MultiHostUrl('https://foo.com:123,foo:bar@bar.com/path')
        print(mhu.hosts())
        """
        [
            {'username': None, 'password': None, 'host': 'foo.com', 'port': 123},
            {'username': 'foo', 'password': 'bar', 'host': 'bar.com', 'port': 443}
        ]
        ```
        Returns:
            A list of dicts, each representing a host.
        '''
        return self._url.hosts()

    def unicode_string(self):
        """The URL as a unicode string, unlike `__str__()` this will not punycode encode the hosts."""
        return self._url.unicode_string()

    def __str__(self):
        """The URL as a string, this will punycode encode the host if required."""
        return str(self._url)

    def __repr__(self):
        return f'{self.__class__.__name__}({str(self._url)!r})'

    def __deepcopy__(self, memo):
        return self.__class__(self._url)

    def __eq__(self, other):
        return self.__class__ is other.__class__ and self._url == other._url

    def __hash__(self):
        return hash(self._url)

    def __len__(self):
        return len(str(self._url))

    @classmethod
    def build(cls, *, scheme, hosts=None, username=None, password=None, host=None, port=None, path=None, query=None, fragment=None):
        """Build a new `MultiHostUrl` instance from its component parts.

        This method takes either `hosts` - a list of `MultiHostHost` typed dicts, or the individual components
        `username`, `password`, `host` and `port`.

        Args:
            scheme: The scheme part of the URL.
            hosts: Multiple hosts to build the URL from.
            username: The username part of the URL.
            password: The password part of the URL.
            host: The host part of the URL.
            port: The port part of the URL.
            path: The path part of the URL.
            query: The query part of the URL, or omit for no query.
            fragment: The fragment part of the URL, or omit for no fragment.

        Returns:
            An instance of `MultiHostUrl`
        """
        return cls(_CoreMultiHostUrl.build(scheme=scheme, hosts=hosts, username=username, password=password, host=host, port=port, path=path, query=query, fragment=fragment))

    @classmethod
    def serialize_url(cls, url, info):
        if not isinstance(url, cls):
            raise PydanticSerializationUnexpectedValue(f"Expected `{cls}` but got `{type(url)}` with value `'{url}'` - serialized value may not be as expected.")
        if info.mode == 'json':
            return str(url)
        return url

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):

        def wrap_val(v, h):
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
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        inner_schema = core_schema['schema'] if core_schema['type'] == 'function-wrap' else core_schema
        return handler(inner_schema)
    __pydantic_serializer__ = SchemaSerializer(core_schema.any_schema(serialization=core_schema.to_string_ser_schema()))

@lru_cache
def _build_type_adapter(cls):
    return TypeAdapter(cls)

class AnyUrl(_BaseUrl):
    """Base type for all URLs.

    * Any scheme allowed
    * Top-level domain (TLD) not required
    * Host not required

    Assuming an input URL of `http://samuel:pass@example.com:8000/the/path/?query=here#fragment=is;this=bit`,
    the types export the following properties:

    - `scheme`: the URL scheme (`http`), always set.
    - `host`: the URL host (`example.com`).
    - `username`: optional username if included (`samuel`).
    - `password`: optional password if included (`pass`).
    - `port`: optional port (`8000`).
    - `path`: optional path (`/the/path/`).
    - `query`: optional URL query (for example, `GET` arguments or "search string", such as `query=here`).
    - `fragment`: optional fragment (`fragment=is;this=bit`).
    """

class AnyHttpUrl(AnyUrl):
    """A type that will accept any http or https URL.

    * TLD not required
    * Host not required
    """
    _constraints = UrlConstraints(allowed_schemes=['http', 'https'])

class HttpUrl(AnyUrl):
    """A type that will accept any http or https URL.

    * TLD not required
    * Host not required
    * Max length 2083

    ```python
    from pydantic import BaseModel, HttpUrl, ValidationError

    class MyModel(BaseModel):
        url: HttpUrl

    m = MyModel(url='http://www.example.com')  # (1)!
    print(m.url)
    #> http://www.example.com/

    try:
        MyModel(url='ftp://invalid.url')
    except ValidationError as e:
        print(e)
        '''
        1 validation error for MyModel
        url
          URL scheme should be 'http' or 'https' [type=url_scheme, input_value='ftp://invalid.url', input_type=str]
        '''

    try:
        MyModel(url='not a url')
    except ValidationError as e:
        print(e)
        '''
        1 validation error for MyModel
        url
          Input should be a valid URL, relative URL without a base [type=url_parsing, input_value='not a url', input_type=str]
        '''
    ```

    1. Note: mypy would prefer `m = MyModel(url=HttpUrl('http://www.example.com'))`, but Pydantic will convert the string to an HttpUrl instance anyway.

    "International domains" (e.g. a URL where the host or TLD includes non-ascii characters) will be encoded via
    [punycode](https://en.wikipedia.org/wiki/Punycode) (see
    [this article](https://www.xudongz.com/blog/2017/idn-phishing/) for a good description of why this is important):

    ```python
    from pydantic import BaseModel, HttpUrl

    class MyModel(BaseModel):
        url: HttpUrl

    m1 = MyModel(url='http://puny£code.com')
    print(m1.url)
    #> http://xn--punycode-eja.com/
    m2 = MyModel(url='https://www.аррӏе.com/')
    print(m2.url)
    #> https://www.xn--80ak6aa92e.com/
    m3 = MyModel(url='https://www.example.珠宝/')
    print(m3.url)
    #> https://www.example.xn--pbt977c/
    ```


    !!! warning "Underscores in Hostnames"
        In Pydantic, underscores are allowed in all parts of a domain except the TLD.
        Technically this might be wrong - in theory the hostname cannot have underscores, but subdomains can.

        To explain this; consider the following two cases:

        - `exam_ple.co.uk`: the hostname is `exam_ple`, which should not be allowed since it contains an underscore.
        - `foo_bar.example.com` the hostname is `example`, which should be allowed since the underscore is in the subdomain.

        Without having an exhaustive list of TLDs, it would be impossible to differentiate between these two. Therefore
        underscores are allowed, but you can always do further validation in a validator if desired.

        Also, Chrome, Firefox, and Safari all currently accept `http://exam_ple.com` as a URL, so we're in good
        (or at least big) company.
    """
    _constraints = UrlConstraints(max_length=2083, allowed_schemes=['http', 'https'])

class AnyWebsocketUrl(AnyUrl):
    """A type that will accept any ws or wss URL.

    * TLD not required
    * Host not required
    """
    _constraints = UrlConstraints(allowed_schemes=['ws', 'wss'])

class WebsocketUrl(AnyUrl):
    """A type that will accept any ws or wss URL.

    * TLD not required
    * Host not required
    * Max length 2083
    """
    _constraints = UrlConstraints(max_length=2083, allowed_schemes=['ws', 'wss'])

class FileUrl(AnyUrl):
    """A type that will accept any file URL.

    * Host not required
    """
    _constraints = UrlConstraints(allowed_schemes=['file'])

class FtpUrl(AnyUrl):
    """A type that will accept ftp URL.

    * TLD not required
    * Host not required
    """
    _constraints = UrlConstraints(allowed_schemes=['ftp'])

class PostgresDsn(_BaseMultiHostUrl):
    """A type that will accept any Postgres DSN.

    * User info required
    * TLD not required
    * Host required
    * Supports multiple hosts

    If further validation is required, these properties can be used by validators to enforce specific behaviour:

    ```python
    from pydantic import (
        BaseModel,
        HttpUrl,
        PostgresDsn,
        ValidationError,
        field_validator,
    )

    class MyModel(BaseModel):
        url: HttpUrl

    m = MyModel(url='http://www.example.com')

    # the repr() method for a url will display all properties of the url
    print(repr(m.url))
    #> HttpUrl('http://www.example.com/')
    print(m.url.scheme)
    #> http
    print(m.url.host)
    #> www.example.com
    print(m.url.port)
    #> 80

    class MyDatabaseModel(BaseModel):
        db: PostgresDsn

        @field_validator('db')
        def check_db_name(cls, v):
            assert v.path and len(v.path) > 1, 'database must be provided'
            return v

    m = MyDatabaseModel(db='postgres://user:pass@localhost:5432/foobar')
    print(m.db)
    #> postgres://user:pass@localhost:5432/foobar

    try:
        MyDatabaseModel(db='postgres://user:pass@localhost:5432')
    except ValidationError as e:
        print(e)
        '''
        1 validation error for MyDatabaseModel
        db
          Assertion failed, database must be provided
        assert (None)
         +  where None = PostgresDsn('postgres://user:pass@localhost:5432').path [type=assertion_error, input_value='postgres://user:pass@localhost:5432', input_type=str]
        '''
    ```
    """
    _constraints = UrlConstraints(host_required=True, allowed_schemes=['postgres', 'postgresql', 'postgresql+asyncpg', 'postgresql+pg8000', 'postgresql+psycopg', 'postgresql+psycopg2', 'postgresql+psycopg2cffi', 'postgresql+py-postgresql', 'postgresql+pygresql'])

    @property
    def host(self):
        """The required URL host."""
        return self._url.host

class CockroachDsn(AnyUrl):
    """A type that will accept any Cockroach DSN.

    * User info required
    * TLD not required
    * Host required
    """
    _constraints = UrlConstraints(host_required=True, allowed_schemes=['cockroachdb', 'cockroachdb+psycopg2', 'cockroachdb+asyncpg'])

    @property
    def host(self):
        """The required URL host."""
        return self._url.host

class AmqpDsn(AnyUrl):
    """A type that will accept any AMQP DSN.

    * User info required
    * TLD not required
    * Host not required
    """
    _constraints = UrlConstraints(allowed_schemes=['amqp', 'amqps'])

class RedisDsn(AnyUrl):
    """A type that will accept any Redis DSN.

    * User info required
    * TLD not required
    * Host required (e.g., `rediss://:pass@localhost`)
    """
    _constraints = UrlConstraints(allowed_schemes=['redis', 'rediss'], default_host='localhost', default_port=6379, default_path='/0', host_required=True)

    @property
    def host(self):
        """The required URL host."""
        return self._url.host

class MongoDsn(_BaseMultiHostUrl):
    """A type that will accept any MongoDB DSN.

    * User info not required
    * Database name not required
    * Port not required
    * User info may be passed without user part (e.g., `mongodb://mongodb0.example.com:27017`).
    """
    _constraints = UrlConstraints(allowed_schemes=['mongodb', 'mongodb+srv'], default_port=27017)

class KafkaDsn(AnyUrl):
    """A type that will accept any Kafka DSN.

    * User info required
    * TLD not required
    * Host not required
    """
    _constraints = UrlConstraints(allowed_schemes=['kafka'], default_host='localhost', default_port=9092)

class NatsDsn(_BaseMultiHostUrl):
    """A type that will accept any NATS DSN.

    NATS is a connective technology built for the ever increasingly hyper-connected world.
    It is a single technology that enables applications to securely communicate across
    any combination of cloud vendors, on-premise, edge, web and mobile, and devices.
    More: https://nats.io
    """
    _constraints = UrlConstraints(allowed_schemes=['nats', 'tls', 'ws', 'wss'], default_host='localhost', default_port=4222)

class MySQLDsn(AnyUrl):
    """A type that will accept any MySQL DSN.

    * User info required
    * TLD not required
    * Host not required
    """
    _constraints = UrlConstraints(allowed_schemes=['mysql', 'mysql+mysqlconnector', 'mysql+aiomysql', 'mysql+asyncmy', 'mysql+mysqldb', 'mysql+pymysql', 'mysql+cymysql', 'mysql+pyodbc'], default_port=3306, host_required=True)

class MariaDBDsn(AnyUrl):
    """A type that will accept any MariaDB DSN.

    * User info required
    * TLD not required
    * Host not required
    """
    _constraints = UrlConstraints(allowed_schemes=['mariadb', 'mariadb+mariadbconnector', 'mariadb+pymysql'], default_port=3306)

class ClickHouseDsn(AnyUrl):
    """A type that will accept any ClickHouse DSN.

    * User info required
    * TLD not required
    * Host not required
    """
    _constraints = UrlConstraints(allowed_schemes=['clickhouse+native', 'clickhouse+asynch', 'clickhouse+http', 'clickhouse', 'clickhouses', 'clickhousedb'], default_host='localhost', default_port=9000)

class SnowflakeDsn(AnyUrl):
    """A type that will accept any Snowflake DSN.

    * User info required
    * TLD not required
    * Host required
    """
    _constraints = UrlConstraints(allowed_schemes=['snowflake'], host_required=True)

    @property
    def host(self):
        """The required URL host."""
        return self._url.host

def import_email_validator():
    global email_validator
    try:
        import email_validator
    except ImportError as e:
        raise ImportError('email-validator is not installed, run `pip install pydantic[email]`') from e
    if not version('email-validator').partition('.')[0] == '2':
        raise ImportError('email-validator version >= 2.0 required, run pip install -U email-validator')
if TYPE_CHECKING:
    EmailStr = Annotated[str, ...]
else:

    class EmailStr:
        """
        Info:
            To use this type, you need to install the optional
            [`email-validator`](https://github.com/JoshData/python-email-validator) package:

            ```bash
            pip install email-validator
            ```

        Validate email addresses.

        ```python
        from pydantic import BaseModel, EmailStr

        class Model(BaseModel):
            email: EmailStr

        print(Model(email='contact@mail.com'))
        #> email='contact@mail.com'
        ```
        """

        @classmethod
        def __get_pydantic_core_schema__(cls, _source, _handler):
            import_email_validator()
            return core_schema.no_info_after_validator_function(cls._validate, core_schema.str_schema())

        @classmethod
        def __get_pydantic_json_schema__(cls, core_schema, handler):
            field_schema = handler(core_schema)
            field_schema.update(type='string', format='email')
            return field_schema

        @classmethod
        def _validate(cls, input_value, /):
            return validate_email(input_value)[1]

class NameEmail(_repr.Representation):
    """
    Info:
        To use this type, you need to install the optional
        [`email-validator`](https://github.com/JoshData/python-email-validator) package:

        ```bash
        pip install email-validator
        ```

    Validate a name and email address combination, as specified by
    [RFC 5322](https://datatracker.ietf.org/doc/html/rfc5322#section-3.4).

    The `NameEmail` has two properties: `name` and `email`.
    In case the `name` is not provided, it's inferred from the email address.

    ```python
    from pydantic import BaseModel, NameEmail

    class User(BaseModel):
        email: NameEmail

    user = User(email='Fred Bloggs <fred.bloggs@example.com>')
    print(user.email)
    #> Fred Bloggs <fred.bloggs@example.com>
    print(user.email.name)
    #> Fred Bloggs

    user = User(email='fred.bloggs@example.com')
    print(user.email)
    #> fred.bloggs <fred.bloggs@example.com>
    print(user.email.name)
    #> fred.bloggs
    ```
    """
    __slots__ = ('name', 'email')

    def __init__(self, name, email):
        self.name = name
        self.email = email

    def __eq__(self, other):
        return isinstance(other, NameEmail) and (self.name, self.email) == (other.name, other.email)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        field_schema = handler(core_schema)
        field_schema.update(type='string', format='name-email')
        return field_schema

    @classmethod
    def __get_pydantic_core_schema__(cls, _source, _handler):
        import_email_validator()
        return core_schema.no_info_after_validator_function(cls._validate, core_schema.json_or_python_schema(json_schema=core_schema.str_schema(), python_schema=core_schema.union_schema([core_schema.is_instance_schema(cls), core_schema.str_schema()], custom_error_type='name_email_type', custom_error_message='Input is not a valid NameEmail'), serialization=core_schema.to_string_ser_schema()))

    @classmethod
    def _validate(cls, input_value, /):
        if isinstance(input_value, str):
            name, email = validate_email(input_value)
            return cls(name, email)
        else:
            return input_value

    def __str__(self):
        if '@' in self.name:
            return f'"{self.name}" <{self.email}>'
        return f'{self.name} <{self.email}>'
IPvAnyAddressType = 'IPv4Address | IPv6Address'
IPvAnyInterfaceType = 'IPv4Interface | IPv6Interface'
IPvAnyNetworkType = 'IPv4Network | IPv6Network'
if TYPE_CHECKING:
    IPvAnyAddress = IPvAnyAddressType
    IPvAnyInterface = IPvAnyInterfaceType
    IPvAnyNetwork = IPvAnyNetworkType
else:

    class IPvAnyAddress:
        """Validate an IPv4 or IPv6 address.

        ```python
        from pydantic import BaseModel
        from pydantic.networks import IPvAnyAddress

        class IpModel(BaseModel):
            ip: IPvAnyAddress

        print(IpModel(ip='127.0.0.1'))
        #> ip=IPv4Address('127.0.0.1')

        try:
            IpModel(ip='http://www.example.com')
        except ValueError as e:
            print(e.errors())
            '''
            [
                {
                    'type': 'ip_any_address',
                    'loc': ('ip',),
                    'msg': 'value is not a valid IPv4 or IPv6 address',
                    'input': 'http://www.example.com',
                }
            ]
            '''
        ```
        """
        __slots__ = ()

        def __new__(cls, value):
            """Validate an IPv4 or IPv6 address."""
            try:
                return IPv4Address(value)
            except ValueError:
                pass
            try:
                return IPv6Address(value)
            except ValueError:
                raise PydanticCustomError('ip_any_address', 'value is not a valid IPv4 or IPv6 address')

        @classmethod
        def __get_pydantic_json_schema__(cls, core_schema, handler):
            field_schema = {}
            field_schema.update(type='string', format='ipvanyaddress')
            return field_schema

        @classmethod
        def __get_pydantic_core_schema__(cls, _source, _handler):
            return core_schema.no_info_plain_validator_function(cls._validate, serialization=core_schema.to_string_ser_schema())

        @classmethod
        def _validate(cls, input_value, /):
            return cls(input_value)

    class IPvAnyInterface:
        """Validate an IPv4 or IPv6 interface."""
        __slots__ = ()

        def __new__(cls, value):
            """Validate an IPv4 or IPv6 interface."""
            try:
                return IPv4Interface(value)
            except ValueError:
                pass
            try:
                return IPv6Interface(value)
            except ValueError:
                raise PydanticCustomError('ip_any_interface', 'value is not a valid IPv4 or IPv6 interface')

        @classmethod
        def __get_pydantic_json_schema__(cls, core_schema, handler):
            field_schema = {}
            field_schema.update(type='string', format='ipvanyinterface')
            return field_schema

        @classmethod
        def __get_pydantic_core_schema__(cls, _source, _handler):
            return core_schema.no_info_plain_validator_function(cls._validate, serialization=core_schema.to_string_ser_schema())

        @classmethod
        def _validate(cls, input_value, /):
            return cls(input_value)

    class IPvAnyNetwork:
        """Validate an IPv4 or IPv6 network."""
        __slots__ = ()

        def __new__(cls, value):
            """Validate an IPv4 or IPv6 network."""
            try:
                return IPv4Network(value)
            except ValueError:
                pass
            try:
                return IPv6Network(value)
            except ValueError:
                raise PydanticCustomError('ip_any_network', 'value is not a valid IPv4 or IPv6 network')

        @classmethod
        def __get_pydantic_json_schema__(cls, core_schema, handler):
            field_schema = {}
            field_schema.update(type='string', format='ipvanynetwork')
            return field_schema

        @classmethod
        def __get_pydantic_core_schema__(cls, _source, _handler):
            return core_schema.no_info_plain_validator_function(cls._validate, serialization=core_schema.to_string_ser_schema())

        @classmethod
        def _validate(cls, input_value, /):
            return cls(input_value)

def _build_pretty_email_regex():
    name_chars = "[\\w!#$%&\\'*+\\-/=?^_`{|}~]"
    unquoted_name_group = f'((?:{name_chars}+\\s+)*{name_chars}+)'
    quoted_name_group = '"((?:[^"]|\\")+)"'
    email_group = '<(.+)>'
    return re.compile(f'\\s*(?:{unquoted_name_group}|{quoted_name_group})?\\s*{email_group}\\s*')
pretty_email_regex = _build_pretty_email_regex()
MAX_EMAIL_LENGTH = 2048
'Maximum length for an email.\nA somewhat arbitrary but very generous number compared to what is allowed by most implementations.\n'

def validate_email(value):
    """Email address validation using [email-validator](https://pypi.org/project/email-validator/).

    Returns:
        A tuple containing the local part of the email (or the name for "pretty" email addresses)
            and the normalized email.

    Raises:
        PydanticCustomError: If the email is invalid.

    Note:
        Note that:

        * Raw IP address (literal) domain parts are not allowed.
        * `"John Doe <local_part@domain.com>"` style "pretty" email addresses are processed.
        * Spaces are striped from the beginning and end of addresses, but no error is raised.
    """
    if email_validator is None:
        import_email_validator()
    if len(value) > MAX_EMAIL_LENGTH:
        raise PydanticCustomError('value_error', 'value is not a valid email address: {reason}', {'reason': f'Length must not exceed {MAX_EMAIL_LENGTH} characters'})
    m = pretty_email_regex.fullmatch(value)
    name = None
    if m:
        unquoted_name, quoted_name, value = m.groups()
        name = unquoted_name or quoted_name
    email = value.strip()
    try:
        parts = email_validator.validate_email(email, check_deliverability=False)
    except email_validator.EmailNotValidError as e:
        raise PydanticCustomError('value_error', 'value is not a valid email address: {reason}', {'reason': str(e.args[0])}) from e
    email = parts.normalized
    assert email is not None
    name = name or parts.local_part
    return (name, email)
__getattr__ = getattr_migration(__name__)