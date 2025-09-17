from __future__ import annotations
import typing as t
import urllib.error
import urllib.parse
import urllib.request
from base64 import b64encode
from ipaddress import IPv4Address, IPv6Address
from mimesis.datasets import CONTENT_ENCODING_DIRECTIVES, CORS_OPENER_POLICIES, CORS_RESOURCE_POLICIES, HTTP_METHODS, HTTP_SERVERS, HTTP_STATUS_CODES, HTTP_STATUS_MSGS, PUBLIC_DNS, TLD, USER_AGENTS, USERNAMES
from mimesis.enums import DSNType, IPv4Purpose, Locale, MimeType, PortRange, TLDType, URLScheme
from mimesis.providers.base import BaseProvider
from mimesis.providers.code import Code
from mimesis.providers.date import Datetime
from mimesis.providers.file import File
from mimesis.providers.text import Text
from mimesis.types import Keywords


__all__ = ['Internet']


class Internet(BaseProvider):
    _MAX_IPV4: int = 2 ** 32 - 1
    _MAX_IPV6: int = 2 ** 128 - 1

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        super().__init__(*args, **kwargs)
        self._file: File = File(seed=self.seed, random=self.random)
        self._code: Code = Code(seed=self.seed, random=self.random)
        self._text: Text = Text(locale=Locale.EN, seed=self.seed, random=self.random)
        self._datetime: Datetime = Datetime(locale=Locale.EN, seed=self.seed, random=self.random)

    class Meta:
        name: str = 'internet'

    def content_type(self, mime_type: t.Optional[MimeType] = None) -> str:
        return self._file.mime_type(type_=mime_type)

    def dsn(self, dsn_type: t.Optional[DSNType] = None, **kwargs: t.Any) -> str:
        hostname: str = self.hostname(**kwargs)
        scheme, port = self.validate_enum(dsn_type, DSNType)
        return f'{scheme}://{hostname}:{port}'

    def http_status_message(self) -> str:
        return self.random.choice(HTTP_STATUS_MSGS)

    def http_status_code(self) -> int:
        return self.random.choice(HTTP_STATUS_CODES)

    def http_method(self) -> str:
        return self.random.choice(HTTP_METHODS)

    def ip_v4_object(self) -> IPv4Address:
        return IPv4Address(self.random.randint(0, self._MAX_IPV4))

    def ip_v4_with_port(self, port_range: PortRange = PortRange.ALL) -> str:
        addr: str = self.ip_v4()
        port: int = self.port(port_range)
        return f'{addr}:{port}'

    def ip_v4(self) -> str:
        return str(self.ip_v4_object())

    def ip_v6_object(self) -> IPv6Address:
        return IPv6Address(self.random.randint(0, self._MAX_IPV6))

    def ip_v6(self) -> str:
        return str(self.ip_v6_object())

    def asn(self) -> str:
        ranges: t.Tuple[int, int] = (1, 4199999999)
        number: int = self.random.randint(*ranges)
        return f'AS{number}'

    def mac_address(self) -> str:
        mac_hex: t.List[int] = [
            0,
            22,
            62,
            self.random.randint(0, 127),
            self.random.randint(0, 255),
            self.random.randint(0, 255)
        ]
        mac: t.List[str] = [f'{x:02x}' for x in mac_hex]
        return ':'.join(mac)

    @staticmethod
    def stock_image_url(width: int = 1920, height: int = 1080, keywords: t.Optional[t.Sequence[str]] = None) -> str:
        if keywords is not None:
            keywords_str = ','.join(keywords)
        else:
            keywords_str = ''
        return f'https://source.unsplash.com/{width}x{height}?{keywords_str}'

    def hostname(self, tld_type: t.Optional[TLDType] = None, subdomains: t.Optional[t.List[str]] = None) -> str:
        tld: str = self.tld(tld_type=tld_type)
        host: str = self.random.choice(USERNAMES)
        if subdomains:
            subdomain: str = self.random.choice(subdomains)
            host = f'{subdomain}.{host}'
        return f'{host}{tld}'

    def url(self, scheme: URLScheme = URLScheme.HTTPS, port_range: t.Optional[PortRange] = None,
            tld_type: t.Optional[TLDType] = None, subdomains: t.Optional[t.List[str]] = None) -> str:
        host: str = self.hostname(tld_type, subdomains)
        url_scheme: str = self.validate_enum(scheme, URLScheme)
        url: str = f'{url_scheme}://{host}'
        if port_range is not None:
            url = f'{url}:{self.port(port_range)}'
        return f'{url}/'

    def uri(self, scheme: URLScheme = URLScheme.HTTPS, tld_type: t.Optional[TLDType] = None,
            subdomains: t.Optional[t.List[str]] = None, query_params_count: t.Optional[int] = None) -> str:
        directory: str = self._datetime.date(start=2010, end=self._datetime._CURRENT_YEAR).strftime('%Y-%m-%d').replace('-', '/')
        url: str = self.url(scheme, None, tld_type, subdomains)
        uri: str = f'{url}{directory}/{self.slug()}'
        if query_params_count:
            uri += f'?{self.query_string(query_params_count)}'
        return uri

    def query_string(self, length: t.Optional[int] = None) -> str:
        return urllib.parse.urlencode(self.query_parameters(length))

    def query_parameters(self, length: t.Optional[int] = None) -> t.Dict[str, str]:
        def pick_unique_words(quantity: int = 5) -> t.List[str]:
            words: t.Set[str] = set()
            while len(words) != quantity:
                words.add(self._text.word())
            return list(words)

        if not length:
            length = self.random.randint(1, 10)
        if length > 32:
            raise ValueError('Maximum allowed length of query parameters is 32.')
        keys: t.List[str] = pick_unique_words(length)
        values: t.List[str] = self._text.words(length)
        return dict(zip(keys, values))

    def top_level_domain(self, tld_type: TLDType = TLDType.CCTLD) -> str:
        key: TLDType = self.validate_enum(item=tld_type, enum=TLDType)
        return self.random.choice(TLD[key])

    def tld(self, *args: t.Any, **kwargs: t.Any) -> str:
        return self.top_level_domain(*args, **kwargs)

    def user_agent(self) -> str:
        return self.random.choice(USER_AGENTS)

    def port(self, port_range: PortRange = PortRange.ALL) -> int:
        rng: t.Tuple[int, int] = self.validate_enum(port_range, PortRange)
        return self.random.randint(*rng)

    def path(self, *args: t.Any, **kwargs: t.Any) -> str:
        return self.slug(*args, **kwargs).replace('-', '/')

    def slug(self, parts_count: t.Optional[int] = None) -> str:
        if not parts_count:
            parts_count = self.random.randint(2, 12)
        if parts_count > 12:
            raise ValueError("Slug's parts count must be <= 12")
        if parts_count < 2:
            raise ValueError('Slug must contain more than 2 parts')
        return '-'.join(self._text.words(parts_count))

    def public_dns(self) -> str:
        return self.random.choice(PUBLIC_DNS)

    def http_response_headers(self) -> t.Dict[str, t.Union[str, int]]:
        max_age: int = self.random.randint(0, 60 * 60 * 15)
        cookie_attributes: t.List[str] = [
            'Secure',
            'HttpOnly',
            'SameSite=Lax',
            'SameSite=Strict',
            f'Max-Age={max_age}',
            f'Domain={self.hostname()}'
        ]
        k, v = self._text.words(quantity=2)
        cookie_attr: str = self.random.choice(cookie_attributes)
        csrf_token: str = b64encode(self.random.randbytes(n=32)).decode()
        cookie_value: str = f'csrftoken={csrf_token}; {k}={v}; {cookie_attr}'
        headers: t.Dict[str, t.Union[str, int]] = {
            'Allow': '*',
            'Age': max_age,
            'Server': self.random.choice(HTTP_SERVERS),
            'Content-Type': self._file.mime_type(),
            'X-Request-ID': self.random.randbytes(16).hex(),
            'Content-Language': self._code.locale_code(),
            'Content-Location': self.path(parts_count=4),
            'Set-Cookie': cookie_value,
            'Upgrade-Insecure-Requests': 1,
            'X-Content-Type-Options': 'nosniff',
            'X-XSS-Protection': 1,
            'Connection': self.random.choice(['close', 'keep-alive']),
            'X-Frame-Options': self.random.choice(['DENY', 'SAMEORIGIN']),
            'Content-Encoding': self.random.choice(CONTENT_ENCODING_DIRECTIVES),
            'Cross-Origin-Opener-Policy': self.random.choice(CORS_OPENER_POLICIES),
            'Cross-Origin-Resource-Policy': self.random.choice(CORS_RESOURCE_POLICIES),
            'Strict-Transport-Security': f'max-age={max_age}'
        }
        return headers

    def http_request_headers(self) -> t.Dict[str, t.Union[str, int]]:
        k, v = self._text.words(quantity=2)
        max_age: int = self.random.randint(0, 60 * 60 * 15)
        token: str = b64encode(self.random.randbytes(64)).hex()
        csrf_token: str = b64encode(self.random.randbytes(n=32)).decode()
        headers: t.Dict[str, t.Union[str, int]] = {
            'Referer': self.uri(),
            'Authorization': f'Bearer {token}',
            'Cookie': f'csrftoken={csrf_token}; {k}={v}',
            'User-Agent': self.user_agent(),
            'X-CSRF-Token': b64encode(self.random.randbytes(32)).hex(),
            'Content-Type': self._file.mime_type(),
            'Content-Length': self.random.randint(0, 10000),
            'Connection': self.random.choice(['close', 'keep-alive']),
            'Cache-Control': self.random.choice(['no-cache', 'no-store', 'must-revalidate', 'public', 'private', f'max-age={max_age}']),
            'Accept': self.random.choice(['*/*', self._file.mime_type()]),
            'Host': self.hostname(),
            'Accept-Language': self.random.choice(['*', self._code.locale_code()])
        }
        return headers

    def special_ip_v4_object(self, purpose: t.Optional[IPv4Purpose] = None) -> IPv4Address:
        ranges: t.Tuple[int, int] = self.validate_enum(purpose, IPv4Purpose)
        number: int = self.random.randint(*ranges)
        return IPv4Address(number)

    def special_ip_v4(self, purpose: t.Optional[IPv4Purpose] = None) -> str:
        return str(self.special_ip_v4_object(purpose))