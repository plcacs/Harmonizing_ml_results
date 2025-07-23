import re
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network, _BaseAddress, _BaseNetwork
from typing import TYPE_CHECKING, Any, Collection, Dict, Generator, List, Match, Optional, Pattern, Set, Tuple, Type, Union, cast, no_type_check, overload
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
        scheme: Optional[str]
        user: Optional[str]
        password: Optional[str]
        domain: Optional[str]
        ipv4: Optional[str]
        ipv6: Optional[str]
        port: Optional[str]
        path: Optional[str]
        query: Optional[str]
        fragment: Optional[str]

    class HostParts(TypedDict, total=False):
        host: Optional[str]
        host_type: Optional[str]
        tld: Optional[str]
        rebuild: bool
        port: Optional[str]
else:
    email_validator = None

    class Parts(dict):
        pass

NetworkType = Union[str, bytes, int, Tuple[Union[str, bytes, int], Union[str, int]]]
__all__ = ['AnyUrl', 'AnyHttpUrl', 'FileUrl', 'HttpUrl', 'stricturl', 'EmailStr', 'NameEmail', 'IPvAnyAddress', 'IPvAnyInterface', 'IPvAnyNetwork', 'PostgresDsn', 'CockroachDsn', 'AmqpDsn', 'RedisDsn', 'MongoDsn', 'KafkaDsn', 'validate_email']

_url_regex_cache: Optional[Pattern[str]] = None
_multi_host_url_regex_cache: Optional[Pattern[str]] = None
_ascii_domain_regex_cache: Optional[Pattern[str]] = None
_int_domain_regex_cache: Optional[Pattern[str]] = None
_host_regex_cache: Optional[Pattern[str]] = None
_host_regex: str = '(?:(?P<ipv4>(?:\\d{1,3}\\.){3}\\d{1,3})(?=$|[/:#?])|(?P<ipv6>\\[[A-F0-9]*:[A-F0-9:]+\\])(?=$|[/:#?])|(?P<domain>[^\\s/:?#]+))?(?::(?P<port>\\d+))?'
_scheme_regex: str = '(?:(?P<scheme>[a-z][a-z0-9+\\-.]+)://)?'
_user_info_regex: str = '(?:(?P<user>[^\\s:/]*)(?::(?P<password>[^\\s/]*))?@)?'
_path_regex: str = '(?P<path>/[^\\s?#]*)?'
_query_regex: str = '(?:\\?(?P<query>[^\\s#]*))?'
_fragment_regex: str = '(?:#(?P<fragment>[^\\s#]*))?'

def url_regex() -> Pattern[str]:
    global _url_regex_cache
    if _url_regex_cache is None:
        _url_regex_cache = re.compile(f'{_scheme_regex}{_user_info_regex}{_host_regex}{_path_regex}{_query_regex}{_fragment_regex}', re.IGNORECASE)
    return _url_regex_cache

def multi_host_url_regex() -> Pattern[str]:
    global _multi_host_url_regex_cache
    if _multi_host_url_regex_cache is None:
        _multi_host_url_regex_cache = re.compile(f'{_scheme_regex}{_user_info_regex}(?P<hosts>([^/]*)){_path_regex}{_query_regex}{_fragment_regex}', re.IGNORECASE)
    return _multi_host_url_regex_cache

def ascii_domain_regex() -> Pattern[str]:
    global _ascii_domain_regex_cache
    if _ascii_domain_regex_cache is None:
        ascii_chunk = '[_0-9a-z](?:[-_0-9a-z]{0,61}[_0-9a-z])?'
        ascii_domain_ending = '(?P<tld>\\.[a-z]{2,63})?\\.?'
        _ascii_domain_regex_cache = re.compile(f'(?:{ascii_chunk}\\.)*?{ascii_chunk}{ascii_domain_ending}', re.IGNORECASE)
    return _ascii_domain_regex_cache

def int_domain_regex() -> Pattern[str]:
    global _int_domain_regex_cache
    if _int_domain_regex_cache is None:
        int_chunk = '[_0-9a-\\U00040000](?:[-_0-9a-\\U00040000]{0,61}[_0-9a-\\U00040000])?'
        int_domain_ending = '(?P<tld>(\\.[^\\W\\d_]{2,63})|(\\.(?:xn--)[_0-9a-z-]{2,63}))?\\.?'
        _int_domain_regex_cache = re.compile(f'(?:{int_chunk}\\.)*?{int_chunk}{int_domain_ending}', re.IGNORECASE)
    return _int_domain_regex_cache

def host_regex() -> Pattern[str]:
    global _host_regex_cache
    if _host_regex_cache is None:
        _host_regex_cache = re.compile(_host_regex, re.IGNORECASE)
    return _host_regex_cache

class AnyUrl(str):
    strip_whitespace: bool = True
    min_length: int = 1
    max_length: int = 2 ** 16
    allowed_schemes: Optional[Collection[str]] = None
    tld_required: bool = False
    user_required: bool = False
    host_required: bool = True
    hidden_parts: Set[str] = set()
    __slots__ = ('scheme', 'user', 'password', 'host', 'tld', 'host_type', 'port', 'path', 'query', 'fragment')

    @no_type_check
    def __new__(cls, url: Optional[str], **kwargs: Any) -> 'AnyUrl':
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
        fragment: Optional[str] = None
    ) -> None:
        str.__init__(url)
        self.scheme: str = scheme
        self.user: Optional[str] = user
        self.password: Optional[str] = password
        self.host: Optional[str] = host
        self.tld: Optional[str] = tld
        self.host_type: str = host_type
        self.port: Optional[str] = port
        self.path: Optional[str] = path
        self.query: Optional[str] = query
        self.fragment: Optional[str] = fragment

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
        **kwargs: Any
    ) -> str:
        parts = Parts(scheme=scheme, user=user, password=password, host=host, port=port, path=path, query=query, fragment=fragment, **kwargs)
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
        url = cast(str, constr_length_validator(value, field, config))
        m = cls._match_url(url)
        assert m, 'URL regex failed unexpectedly'
        original_parts = cast('Parts', m.groupdict())
        parts = cls.apply_default_parts(original_parts)
        parts = cls.validate_parts(parts)
        if m.end() != len(url):
            raise errors.UrlExtraError(extra=url[m.end():])
        return cls._build_url(m, url, parts)

    @classmethod
    def _build_url(cls, m: Match[str], url: str, parts: 'Parts') -> 'AnyUrl':
        host, tld, host_type, rebuild = cls.validate_host(parts)
        return cls(None if rebuild else url, scheme=parts['scheme'], user=parts['user'], password=parts['password'], host=host, tld=tld, host_type=host_type, port=parts['port'], path=parts['path'], query=parts['query'], fragment=parts['fragment'])

    @staticmethod
    def _match_url(url: str) -> Optional[Match[str]]:
        return url_regex().match(url)

    @staticmethod
    def _validate_port(port: Optional[str]) -> None:
        if port is not None and int(port) > 65535:
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
    def validate_host(cls, parts: 'Parts') -> Tuple[Optional[str], Optional[str], str, bool]:
        tld: Optional[str] = None
        host_type: str = 'domain'
        rebuild: bool = False
        host: Optional[str] = None
        
        for f in ('domain', 'ipv4', 'ipv6'):
            host = parts[f]
            if host:
                host_type = f
                break
                
        if host is None:
            if cls.host_required:
                raise errors.UrlHostError()
        elif host_type == 'domain':
            is_international = False
            d = ascii_domain_regex().fullmatch(host)
            if d is None:
                d = int_domain_regex().fullmatch(host)
                if d is None:
                    raise errors.UrlHostError()
                is_international = True
            tld = d.group('tld')
            if tld is None and (not is_international):
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
        return (host, tld, host_type, rebuild)

    @staticmethod
    def get_default_parts(parts: 'Parts') -> Dict[str, str]:
        return {}

    @classmethod
    def apply_default_parts(cls, parts: 'Parts') -> 'Parts':
        for key, value in cls.get_default_parts(parts).items():
            if not parts[key]:
                parts[key] = value
        return parts

    def __repr__(self) -> str:
        extra = ', '.join((f'{n}={getattr(self, n)!r}' for n in self.__slots__ if getattr(self, n) is not None))
        return f'{self.__class__.__name__}({super().__repr__()}, {extra})'

class AnyHttpUrl(AnyUrl):
    allowed_schemes: Set[str] = {'http', 'https'}
    __slots__ = ()

class HttpUrl(AnyHttpUrl):
    tld_required: bool = True
    max_length: int = 2083
    hidden_parts: Set[str] = {'port'}

    @staticmethod
    def get_default_parts(parts: 'Parts') -> Dict[str, str]:
        return {'port': '80' if parts['scheme'] == 'http' else '443'}

class FileUrl(AnyUrl):
    allowed_schemes: Set[str] = {'file'}
    host_required: bool = False
    __slots__ = ()

class MultiHostDsn(AnyUrl):
    __slots__ = AnyUrl.__slots__ + ('hosts',)

    def __init__(
        self,
        url: str,
        *,
        scheme: str,
        user: Optional[str] = None,
        password: Optional[str] = None,
        host: Optional[str] = None,
        tld: Optional[str] = None,
        host_type: Optional[str] = None,
        port: Optional[str] = None,
        path: Optional[str] = None,
        query: Optional[str] = None,
        fragment: Optional[str] = None,
        hosts: Optional[List['HostParts']] = None
    ) -> None:
        super().__init__(url, scheme=scheme, user=user, password=password, host=host, tld=tld, host_type=host_type, port=port, path=path, query=query, fragment=fragment)
        self.hosts: Optional[List['HostParts']] = hosts

    @staticmethod
    def _match_url(url: str) -> Optional[Match[str]]:
        return multi_host_url_regex().match(url)

    @classmethod
    def validate_parts(cls, parts: 'Parts', validate_port: bool = True) -> 'Parts':
        return super().validate_parts(parts, validate_port=False)

    @classmethod
    def _build_url(cls, m: Match[str], url: str, parts: 'Parts') -> 'MultiHostDsn':
        hosts_parts: List['HostParts'] = []
        host_re = host_regex()
        for host in m.groupdict()['hosts'].split(','):
            d = host_re.match(host).groupdict()
            host, tld, host_type, rebuild = cls.validate_host(d)
            port = d.get('port')
            cls._validate_port(port)
            hosts_parts.append({'host': host, 'host_type': host_type, 'tld': tld, 'rebuild': rebuild, 'port': port})
        if len(hosts_parts) > 1:
            return cls(None if any([hp['rebuild'] for hp in hosts_parts]) else url, scheme=parts['scheme'], user=parts['user'], password=parts['password'], path=parts['path'], query=parts['query'], fragment=parts['fragment'], host_type=None, hosts=hosts_parts)
        else:
            host_part = hosts_parts[0]
            return cls(None if host_part['rebuild'] else url, scheme=parts['scheme'], user=parts['user'], password=parts['password'], host=host_part['host'], tld=host_part['tld'], host_type=host_part['host_type'], port=host_part.get('port'), path=parts['path'], query=parts['query'], fragment=parts['fragment'])

class PostgresDsn(MultiHostDsn):
    allowed_schemes: Set[str] = {'postgres', 'postgresql', 'postgresql+asyncpg', 'postgresql+pg8000', 'postgresql+psycopg', 'postgresql+psycopg2', 'postgresql+psycopg2cffi', 'postgresql+py-postgresql', 'postgresql+pygresql'}
    user_required: bool = True
    __slots__ = ()

class CockroachDsn(AnyUrl):
    allowed_schemes: Set[str] = {'cockroachdb', 'cockroachdb+psycopg2', 'cockroachdb+asyncpg'}
    user_required: bool = True

class AmqpDsn(AnyUrl):
    allowed_schemes: Set[str] = {'amqp', 'amqps'}
    host_required: bool = False

class RedisDsn(Any