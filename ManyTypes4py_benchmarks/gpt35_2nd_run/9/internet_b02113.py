import typing as t
from ipaddress import IPv4Address, IPv6Address
from mimesis.enums import DSNType, IPv4Purpose, Locale, MimeType, PortRange, TLDType, URLScheme
from mimesis.providers.base import BaseProvider
from mimesis.providers.code import Code
from mimesis.providers.date import Datetime
from mimesis.providers.file import File
from mimesis.providers.text import Text
from mimesis.types import Keywords

class Internet(BaseProvider):
    _MAX_IPV4: int = 2 ** 32 - 1
    _MAX_IPV6: int = 2 ** 128 - 1

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        super().__init__(*args, **kwargs)
        self._file: File = File(seed=self.seed, random=self.random)
        self._code: Code = Code(seed=self.seed, random=self.random)
        self._text: Text = Text(locale=Locale.EN, seed=self.seed, random=self.random)
        self._datetime: Datetime = Datetime(locale=Locale.EN, seed=self.seed, random=self.random)

    def content_type(self, mime_type: t.Optional[MimeType] = None) -> str:
    def dsn(self, dsn_type: t.Optional[DSNType] = None, **kwargs: t.Any) -> str:
    def http_status_message(self) -> str:
    def http_status_code(self) -> int:
    def http_method(self) -> str:
    def ip_v4_object(self) -> IPv4Address:
    def ip_v4_with_port(self, port_range: PortRange = PortRange.ALL) -> str:
    def ip_v4(self) -> str:
    def ip_v6_object(self) -> IPv6Address:
    def ip_v6(self) -> str:
    def asn(self) -> str:
    def mac_address(self) -> str:
    @staticmethod
    def stock_image_url(width: int = 1920, height: int = 1080, keywords: t.Optional[t.List[str]] = None) -> str:
    def hostname(self, tld_type: t.Optional[TLDType] = None, subdomains: t.Optional[t.List[str]] = None) -> str:
    def url(self, scheme: URLScheme = URLScheme.HTTPS, port_range: t.Optional[PortRange] = None, tld_type: t.Optional[TLDType] = None, subdomains: t.Optional[t.List[str]] = None) -> str:
    def uri(self, scheme: URLScheme = URLScheme.HTTPS, tld_type: t.Optional[TLDType] = None, subdomains: t.Optional[t.List[str]] = None, query_params_count: t.Optional[int] = None) -> str:
    def query_string(self, length: t.Optional[int] = None) -> str:
    def query_parameters(self, length: t.Optional[int] = None) -> t.Dict[str, str]:
    def top_level_domain(self, tld_type: TLDType = TLDType.CCTLD) -> str:
    def tld(self, *args: t.Any, **kwargs: t.Any) -> str:
    def user_agent(self) -> str:
    def port(self, port_range: PortRange = PortRange.ALL) -> int:
    def path(self, *args: t.Any, **kwargs: t.Any) -> str:
    def slug(self, parts_count: t.Optional[int] = None) -> str:
    def public_dns(self) -> str:
    def http_response_headers(self) -> t.Dict[str, t.Union[str, int]]:
    def http_request_headers(self) -> t.Dict[str, t.Union[str, int]]:
    def special_ip_v4_object(self, purpose: t.Optional[IPv4Purpose] = None) -> IPv4Address:
    def special_ip_v4(self, purpose: t.Optional[IPv4Purpose] = None) -> str:
