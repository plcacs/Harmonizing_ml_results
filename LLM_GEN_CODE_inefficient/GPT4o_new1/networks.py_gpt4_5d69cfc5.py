import re
from ipaddress import (
    IPv4Address,
    IPv4Interface,
    IPv4Network,
    IPv6Address,
    IPv6Interface,
    IPv6Network,
    _BaseAddress,
    _BaseNetwork,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Collection,
    Dict,
    Generator,
    List,
    Match,
    Optional,
    Pattern,
    Set,
    Tuple,
    Type,
    Union,
    cast,
    no_type_check,
)

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
        scheme: str
        user: Optional[str]
        password: Optional[str]
        ipv4: Optional[str]
        ipv6: Optional[str]
        domain: Optional[str]
        port: Optional[str]
        path: Optional[str]
        query: Optional[str]
        fragment: Optional[str]

    class HostParts(TypedDict, total=False):
        host: str
        tld: Optional[str]
        host_type: Optional[str]
        port: Optional[str]
        rebuild: bool

else:
    email_validator = None

    class Parts(dict):
        pass


NetworkType = Union[str, bytes, int, Tuple[Union[str, bytes, int], Union[str, int]]]

__all__ = [
    'AnyUrl',
    'AnyHttpUrl',
    'FileUrl',
    'HttpUrl',
    'stricturl',
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
    'validate_email',
]

_url_regex_cache: Optional[Pattern[str]] = None
_multi_host_url_regex_cache: Optional[Pattern[str]] = None
_ascii_domain_regex_cache: Optional[Pattern[str]] = None
_int_domain_regex_cache: Optional[Pattern[str]] = None
_host_regex_cache: Optional[Pattern[str]] = None

_host_regex: str = (
    r'(?:'
    r'(?P<ipv4>(?:\d{1,3}\.){3}\d{1,3})(?=$|[/:#?])|'  # ipv4
    r'(?P<ipv6>\[[A-F0-9]*:[A-F0-9:]+\])(?=$|[/:#?])|'  # ipv6
    r'(?P<domain>[^\s/:?#]+)'  # domain, validation occurs later
    r')?'
    r'(?::(?P<port>\d+))?'  # port
)
_scheme_regex: str = r'(?:(?P<scheme>[a-z][a-z0-9+\-.]+)://)?'  # scheme https://tools.ietf.org/html/rfc3986#appendix-A
_user_info_regex: str = r'(?:(?P<user>[^\s:/]*)(?::(?P<password>[^\s/]*))?@)?'
_path_regex: str = r'(?P<path>/[^\s?#]*)?'
_query_regex: str = r'(?:\?(?P<query>[^\s#]*))?'
_fragment_regex: str = r'(?:#(?P<fragment>[^\s#]*))?'


def url_regex() -> Pattern[str]:
    global _url_regex_cache
    if _url_regex_cache is None:
        _url_regex_cache = re.compile(
            rf'{_scheme_regex}{_user_info_regex}{_host_regex}{_path_regex}{_query_regex}{_fragment_regex}',
            re.IGNORECASE,
        )
    return _url_regex_cache


def multi_host_url_regex() -> Pattern[str]:
    global _multi_host_url_regex_cache
    if _multi_host_url_regex_cache is None:
        _multi_host_url_regex_cache = re.compile(
            rf'{_scheme_regex}{_user_info_regex}'
            r'(?P<hosts>([^/]*))'  # validation occurs later
            rf'{_path_regex}{_query_regex}{_fragment_regex}',
            re.IGNORECASE,
        )
    return _multi_host_url_regex_cache


def ascii_domain_regex() -> Pattern[str]:
    global _ascii_domain_regex_cache
    if _ascii_domain_regex_cache is None:
        ascii_chunk = r'[_0-9a-z](?:[-_0-9a-z]{0,61}[_0-9a-z])?'
        ascii_domain_ending = r'(?P<tld>\.[a-z]{2,63})?\.?'
        _ascii_domain_regex_cache = re.compile(
            fr'(?:{ascii_chunk}\.)*?{ascii_chunk}{ascii_domain_ending}', re.IGNORECASE
        )
    return _ascii_domain_regex_cache


def int_domain_regex() -> Pattern[str]:
    global _int_domain_regex_cache
    if _int_domain_regex_cache is None:
        int_chunk = r'[_0-9a-\U00040000](?:[-_0-9a-\U00040000]{0,61}[_0-9a-\U00040000])?'
        int_domain_ending = r'(?P<tld>(\.[^\W\d_]{2,63})|(\.(?:xn--)[_0-9a-z-]{2,63}))?\.?'
        _int_domain_regex_cache = re.compile(fr'(?:{int_chunk}\.)*?{int_chunk}{int_domain_ending}', re.IGNORECASE)
    return _int_domain_regex_cache


def host_regex() -> Pattern[str]:
    global _host_regex_cache
    if _host_regex_cache is None:
        _host_regex_cache = re.compile(
            _host_regex,
            re.IGNORECASE,
        )
    return _host_regex_cache


class AnyUrl(str):
    strip_whitespace: bool = True
    min_length: int = 1
    max_length: int = 2**16
    allowed_schemes: Optional[Collection[str]] = None
    tld_required: bool = False
    user_required: bool = False
    host_required: bool = True
    hidden_parts: Set[str] = set()

    __slots__ = ('scheme', 'user', 'password', 'host', 'tld', 'host_type', 'port', 'path', 'query', 'fragment')

    @no_type_check
    def __new__(cls, url: Optional[str], **kwargs: Any) -> object:
        return str.__new__(cls, cls.build(**kwargs) if url is None else url)

    def __init__(
        self,
        url: str,
        *,
        scheme: str,
        user: Optional[str] = None,
        password: Optional[str] = None,
        host: Optional[str] = None,
        tld: Optional[str] = None,
        host_type: str = 'domain',
        port: Optional[str] = None,
        path: Optional[str] = None,
        query: Optional[str] = None,
        fragment: Optional[str] = None,
    ) -> None:
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
    def build(
        cls,
        *,
        scheme: str,
        user: Optional[str] = None,
        password: Optional[str] = None,
        host: str,
        port: Optional[str] = None,
        path: Optional[str] = None,
        query: Optional[str] = None,
        fragment: Optional[str] = None,
        **_kwargs: str,
    ) -> str:
        parts: Parts = Parts(
            scheme=scheme,
            user=user,
            password=password,
            host=host,
            port=port,
            path=path,
            query=query,
            fragment=fragment,
            **_kwargs,  # type: ignore[misc]
        )

        url = scheme + '://'
        if user:
            url += user
        if password:
            url += ':' + password
        if user or password:
            url += '@'
        url += host
        if port and ('port' not in cls.hidden_parts or cls.get_default_parts(parts).get('port') != port):
            url += ':' + port
        if path:
            url += path
        if query:
            url += '?' + query
        if fragment:
            url += '#' + fragment
        return url

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(field_schema, minLength=cls.min_length, maxLength=cls.max_length, format='uri')

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield cls.validate

    @classmethod
    def validate(cls, value: Any, field: 'ModelField', config: 'BaseConfig') -> 'AnyUrl':
        if value.__class__ == cls:
            return value
        value = str_validator(value)
        if cls.strip_whitespace:
            value = value.strip()
        url: str = cast(str, constr_length_validator(value, field, config))

        m: Optional[Match[str]] = cls._match_url(url)
        assert m, 'URL regex failed unexpectedly'

        original_parts: Parts = cast('Parts', m.groupdict())
        parts: Parts = cls.apply_default_parts(original_parts)
        parts = cls.validate_parts(parts)

        if m.end() != len(url):
            raise errors.UrlExtraError(extra=url[m.end() :])

        return cls._build_url(m, url, parts)

    @classmethod
    def _build_url(cls, m: Match[str], url: str, parts: 'Parts') -> 'AnyUrl':
        host, tld, host_type, rebuild = cls.validate_host(parts)

        return cls(
            None if rebuild else url,
            scheme=parts['scheme'],
            user=parts['user'],
            password=parts['password'],
            host=host,
            tld=tld,
            host_type=host_type,
            port=parts['port'],
            path=parts['path'],
            query=parts['query'],
            fragment=parts['fragment'],
        )

    @staticmethod
    def _match_url(url: str) -> Optional[Match[str]]:
        return url_regex().match(url)

    @staticmethod
    def _validate_port(port: Optional[str]) -> None:
        if port is not None and int(port) > 65_535:
            raise errors.UrlPortError()

    @classmethod
    def validate_parts(cls, parts: 'Parts', validate_port: bool = True) -> 'Parts':
        scheme = parts['scheme']
        if scheme is None:
            raise errors.UrlSchemeError()

        if cls.allowed_schemes and scheme.lower() not in cls.allowed_schemes:
            raise errors.UrlSchemePermittedError(set(cls.allowed_schemes))

        if validate_port:
            cls._validate_port(parts['port'])

        user = parts['user']
        if cls.user_required and user is None:
            raise errors.UrlUserInfoError()

        return parts

    @classmethod
    def validate_host(cls, parts: 'Parts') -> Tuple[str, Optional[str], str, bool]:
        tld: Optional[str] = None
        host_type: Optional[str] = None
        rebuild: bool = False
        host: Optional[str] = None
        for f in ('domain', 'ipv4', 'ipv6'):
            host = parts[f]  # type: ignore[literal-required]
            if host:
                host_type = f
                break

        if host is None:
            if cls.host_required:
                raise errors.UrlHostError()
        elif host_type == 'domain':
            is_international: bool = False
            d: Optional[Match[str]] = ascii_domain_regex().fullmatch(host)
            if d is None:
                d = int_domain_regex().fullmatch(host)
                if d is None:
                    raise errors.UrlHostError()
                is_international = True

            tld = d.group('tld')
            if tld is None and not is_international:
                d = int_domain_regex().fullmatch(host)
                assert d is not None
                tld = d.group('tld')
                is_international = True

            if tld is not None:
                tld = tld[1:]
            elif cls.tld_required:
                raise errors.UrlHostTldError()

            if is_international:
                host_type = 'int_domain'
                rebuild = True
                host = host.encode('idna').decode('ascii')
                if tld is not None:
                    tld = tld.encode('idna').decode('ascii')

        return host, tld, host_type, rebuild  # type: ignore

    @staticmethod
    def get_default_parts(parts: 'Parts') -> 'Parts':
        return {}

    @classmethod
    def apply_default_parts(cls, parts: 'Parts') -> 'Parts':
        for key, value in cls.get_default_parts(parts).items():
            if not parts[key]:  # type: ignore[literal-required]
                parts[key] = value  # type: ignore[literal-required]
        return parts

    def __repr__(self) -> str:
        extra: str = ', '.join(f'{n}={getattr(self, n)!r}' for n in self.__slots__ if getattr(self, n) is not None)
        return f'{self.__class__.__name__}({super().__repr__()}, {extra})'


class AnyHttpUrl(AnyUrl):
    allowed_schemes: Set[str] = {'http', 'https'}

    __slots__: Tuple = ()


class HttpUrl(AnyHttpUrl):
    tld_required: bool = True
    max_length: int = 2083
    hidden_parts: Set[str] = {'port'}

    @staticmethod
    def get_default_parts(parts: 'Parts') -> 'Parts':
        return {'port': '80' if parts['scheme'] == 'http' else '443'}


class FileUrl(AnyUrl):
    allowed_schemes: Set[str] = {'file'}
    host_required: bool = False

    __slots__: Tuple = ()


class MultiHostDsn(AnyUrl):
    __slots__: Tuple = AnyUrl.__slots__ + ('hosts',)

    def __init__(self, *args: Any, hosts: Optional[List['HostParts']] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.hosts = hosts

    @staticmethod
    def _match_url(url: str) -> Optional[Match[str]]:
        return multi_host_url_regex().match(url)

    @classmethod
    def validate_parts(cls, parts: 'Parts', validate_port: bool = True) -> 'Parts':
        return super().validate_parts(parts, validate_port=False)

    @classmethod
    def _build_url(cls, m: Match[str], url: str, parts: 'Parts') -> 'MultiHostDsn':
        hosts_parts: List['HostParts'] = []
        host_re: Pattern[str] = host_regex()
        for host in m.groupdict()['hosts'].split(','):
            d: Parts = host_re.match(host).groupdict()  # type: ignore
            host, tld, host_type, rebuild = cls.validate_host(d)
            port = d.get('port')
            cls._validate_port(port)
            hosts_parts.append(
                {
                    'host': host,
                    'host_type': host_type,
                    'tld': tld,
                    'rebuild': rebuild,
                    'port': port,
                }
            )

        if len(hosts_parts) > 1:
            return cls(
                None if any([hp['rebuild'] for hp in hosts_parts]) else url,
                scheme=parts['scheme'],
                user=parts['user'],
                password=parts['password'],
                path=parts['path'],
                query=parts['query'],
                fragment=parts['fragment'],
                host_type=None,
                hosts=hosts_parts,
            )
        else:
            host_part: HostParts = hosts_parts[0]
            return cls(
                None if host_part['rebuild'] else url,
                scheme=parts['scheme'],
                user=parts['user'],
                password=parts['password'],
                host=host_part['host'],
                tld=host_part['tld'],
                host_type=host_part['host_type'],
                port=host_part.get('port'),
                path=parts['path'],
                query=parts['query'],
                fragment=parts['fragment'],
            )


class PostgresDsn(MultiHostDsn):
    allowed_schemes: Set[str] = {
        'postgres',
        'postgresql',
        'postgresql+asyncpg',
        'postgresql+pg8000',
        'postgresql+psycopg',
        'postgresql+psycopg2',
        'postgresql+psycopg2cffi',
        'postgresql+py-postgresql',
        'postgresql+pygresql',
    }
    user_required: bool = True

    __slots__: Tuple = ()


class CockroachDsn(AnyUrl):
    allowed_schemes: Set[str] = {
        'cockroachdb',
        'cockroachdb+psycopg2',
        'cockroachdb+asyncpg',
    }
    user_required: bool = True


class AmqpDsn(AnyUrl):
    allowed_schemes: Set[str] = {'amqp', 'amqps'}
    host_required: bool = False


class RedisDsn(AnyUrl):
    __slots__: Tuple = ()
    allowed_schemes: Set[str] = {'redis', 'rediss'}
    host_required: bool = False

    @staticmethod
    def get_default_parts(parts: 'Parts') -> 'Parts':
        return {
            'domain': 'localhost' if not (parts['ipv4'] or parts['ipv6']) else '',
            'port': '6379',
            'path': '/0',
        }


class MongoDsn(AnyUrl):
    allowed_schemes: Set[str] = {'mongodb'}

    @staticmethod
    def get_default_parts(parts: 'Parts') -> 'Parts':
        return {
            'port': '27017',
        }


class KafkaDsn(AnyUrl):
    allowed_schemes: Set[str] = {'kafka'}

    @staticmethod
    def get_default_parts(parts: 'Parts') -> 'Parts':
        return {
            'domain': 'localhost',
            'port': '9092',
        }


def stricturl(
    *,
    strip_whitespace: bool = True,
    min_length: int = 1,
    max_length: int = 2**16,
    tld_required: bool = True,
    host_required: bool = True,
    allowed_schemes: Optional[Collection[str]] = None,
) -> Type[AnyUrl]:
    namespace: Dict[str, Any] = dict(
        strip_whitespace=strip_whitespace,
        min_length=min_length,
        max_length=max_length,
        tld_required=tld_required,
        host_required=host_required,
        allowed_schemes=allowed_schemes,
    )
    return type('UrlValue', (AnyUrl,), namespace)


def import_email_validator() -> None:
    global email_validator
    try:
        import email_validator
    except ImportError as e:
        raise ImportError('email-validator is not installed, run `pip install pydantic[email]`') from e


class EmailStr(str):
    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        field_schema.update(type='string', format='email')

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        import_email_validator()

        yield str_validator
        yield cls.validate

    @classmethod
    def validate(cls, value: Union[str]) -> str:
        return validate_email(value)[1]


class NameEmail(Representation):
    __slots__: Tuple = ('name', 'email')

    def __init__(self, name: str, email: str) -> None:
        self.name = name
        self.email = email

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, NameEmail) and (self.name, self.email) == (other.name, other.email)

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        field_schema.update(type='string', format='name-email')

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        import_email_validator()

        yield cls.validate

    @classmethod
    def validate(cls, value: Any) -> 'NameEmail':
        if value.__class__ == cls:
            return value
        value = str_validator(value)
        return cls(*validate_email(value))

    def __str__(self) -> str:
        return f'{self.name} <{self.email}>'


class IPvAnyAddress(_BaseAddress):
    __slots__: Tuple = ()

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        field_schema.update(type='string', format='ipvanyaddress')

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield cls.validate

    @classmethod
    def validate(cls, value: Union[str, bytes, int]) -> Union[IPv4Address, IPv6Address]:
        try:
            return IPv4Address(value)
        except ValueError:
            pass

        try:
            return IPv6Address(value)
        except ValueError:
            raise errors.IPvAnyAddressError()


class IPvAnyInterface(_BaseAddress):
    __slots__: Tuple = ()

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        field_schema.update(type='string', format='ipvanyinterface')

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield cls.validate

    @classmethod
    def validate(cls, value: NetworkType) -> Union[IPv4Interface, IPv6Interface]:
        try:
            return IPv4Interface(value)
        except ValueError:
            pass

        try:
            return IPv6Interface(value)
        except ValueError:
            raise errors.IPvAnyInterfaceError()


class IPvAnyNetwork(_BaseNetwork):  # type: ignore
    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        field_schema.update(type='string', format='ipvanynetwork')

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield cls.validate

    @classmethod
    def validate(cls, value: NetworkType) -> Union[IPv4Network, IPv6Network]:
        try:
            return IPv4Network(value)
        except ValueError:
            pass

        try:
            return IPv6Network(value)
        except ValueError:
            raise errors.IPvAnyNetworkError()


pretty_email_regex: Pattern[str] = re.compile(r'([\w ]*?) *<(.*)> *')
MAX_EMAIL_LENGTH: int = 2048


def validate_email(value: Union[str]) -> Tuple[str, str]:
    if email_validator is None:
        import_email_validator()

    if len(value) > MAX_EMAIL_LENGTH:
        raise errors.EmailError()

    m: Optional[Match[str]] = pretty_email_regex.fullmatch(value)
    name: Union[str, None] = None
    if m:
        name, value = m.groups()
    email: str = value.strip()
    try:
        parts = email_validator.validate_email(email, check_deliverability=False)
    except email_validator.EmailNotValidError as e:
        raise errors.EmailError from e

    if hasattr(parts, 'normalized'):
        email = parts.normalized
        assert email is not None
        name = name or parts.local_part
        return name, email
    else:
        at_index: int = email.index('@')
        local_part: str = email[:at_index]
        global_part: str = email[at_index:].lower()

        return name or local_part, local_part + global_part
