"""Provides data related to internet."""
import typing as t
import urllib.error
import urllib.parse
import urllib.request
from base64 import b64encode
from ipaddress import IPv4Address, IPv6Address
from mimesis.enums import DSNType, IPv4Purpose, Locale, MimeType, PortRange, TLDType, URLScheme

__all__: list[str] = ['Internet']

class Internet:
    """Class for generating data related to the internet."""
    _MAX_IPV4: int
    _MAX_IPV6: int

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        ...

    class Meta:
        name: str

    def content_type(self, mime_type: t.Optional[MimeType] = None) -> str:
        ...

    def dsn(self, dsn_type: t.Optional[DSNType] = None, **kwargs: t.Any) -> str:
        ...

    def http_status_message(self) -> str:
        ...

    def http_status_code(self) -> int:
        ...

    def http_method(self) -> str:
        ...

    def ip_v4_object(self) -> IPv4Address:
        ...

    def ip_v4_with_port(self, port_range: t.Optional[PortRange] = None) -> str:
        ...

    def ip_v4(self) -> str:
        ...

    def ip_v6_object(self) -> IPv6Address:
        ...

    def ip_v6(self) -> str:
        ...

    def asn(self) -> str:
        ...

    def mac_address(self) -> str:
        ...

    @staticmethod
    def stock_image_url(width: int = 1920, height: int = 1080, keywords: t.Optional[t.List[str]] = None) -> str:
        ...

    def hostname(self, tld_type: t.Optional[TLDType] = None, subdomains: t.Optional[t.List[str]] = None) -> str:
        ...

    def url(self, scheme: t.Optional[URLScheme] = None, port_range: t.Optional[PortRange] = None, tld_type: t.Optional[TLDType] = None, subdomains: t.Optional[t.List[str]] = None) -> str:
        ...

    def uri(self, scheme: t.Optional[URLScheme] = None, tld_type: t.Optional[TLDType] = None, subdomains: t.Optional[t.List[str]] = None, query_params_count: t.Optional[int] = None) -> str:
        ...

    def query_string(self, length: t.Optional[int] = None) -> str:
        ...

    def query_parameters(self, length: t.Optional[int] = None) -> t.Dict[str, str]:
        ...

    def top_level_domain(self, tld_type: TLDType = TLDType.CCTLD) -> str:
        ...

    def tld(self, *args: t.Any, **kwargs: t.Any) -> str:
        ...

    def user_agent(self) -> str:
        ...

    def port(self, port_range: t.Optional[PortRange] = None) -> int:
        ...

    def path(self, *args: t.Any, **kwargs: t.Any) -> str:
        ...

    def slug(self, parts_count: t.Optional[int] = None) -> str:
        ...

    def public_dns(self) -> str:
        ...

    def http_response_headers(self) -> t.Dict[str, t.Any]:
        ...

    def http_request_headers(self) -> t.Dict[str, t.Any]:
        ...

    def special_ip_v4_object(self, purpose: t.Optional[IPv4Purpose] = None) -> IPv4Address:
        ...

    def special_ip_v4(self, purpose: t.Optional[IPv4Purpose] = None) -> str:
        ...