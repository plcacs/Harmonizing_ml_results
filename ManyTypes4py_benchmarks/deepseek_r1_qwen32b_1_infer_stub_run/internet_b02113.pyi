"""Provides data related to internet."""

import typing as t
from ipaddress import IPv4Address, IPv6Address
from mimesis.enums import DSNType, IPv4Purpose, Locale, MimeType, PortRange, TLDType, URLScheme
from typing import Any, Dict, List, Optional, Set, Tuple, Union

__all__: List[str] = ['Internet']

class Internet:
    """Class for generating data related to the internet."""
    _MAX_IPV4: int
    _MAX_IPV6: int

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...

    class Meta:
        name: str

    def content_type(self, mime_type: Optional[MimeType] = None) -> str:
        ...

    def dsn(self, dsn_type: Optional[DSNType] = None, **kwargs: Any) -> str:
        ...

    def http_status_message(self) -> str:
        ...

    def http_status_code(self) -> int:
        ...

    def http_method(self) -> str:
        ...

    def ip_v4_object(self) -> IPv4Address:
        ...

    def ip_v4_with_port(self, port_range: Optional[PortRange] = PortRange.ALL) -> str:
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
    def stock_image_url(width: int = 1920, height: int = 1080, keywords: Optional[List[str]] = None) -> str:
        ...

    def hostname(self, tld_type: Optional[TLDType] = None, subdomains: Optional[List[str]] = None) -> str:
        ...

    def url(self, scheme: URLScheme = URLScheme.HTTPS, port_range: Optional[PortRange] = None, tld_type: Optional[TLDType] = None, subdomains: Optional[List[str]] = None) -> str:
        ...

    def uri(self, scheme: URLScheme = URLScheme.HTTPS, tld_type: Optional[TLDType] = None, subdomains: Optional[List[str]] = None, query_params_count: Optional[int] = None) -> str:
        ...

    def query_string(self, length: Optional[int] = None) -> str:
        ...

    def query_parameters(self, length: Optional[int] = None) -> Dict[str, str]:
        ...

    def top_level_domain(self, tld_type: TLDType = TLDType.CCTLD) -> str:
        ...

    def tld(self, *args: Any, **kwargs: Any) -> str:
        ...

    def user_agent(self) -> str:
        ...

    def port(self, port_range: PortRange = PortRange.ALL) -> int:
        ...

    def path(self, *args: Any, **kwargs: Any) -> str:
        ...

    def slug(self, parts_count: Optional[int] = None) -> str:
        ...

    def public_dns(self) -> str:
        ...

    def http_response_headers(self) -> Dict[str, Any]:
        ...

    def http_request_headers(self) -> Dict[str, Any]:
        ...

    def special_ip_v4_object(self, purpose: Optional[IPv4Purpose] = None) -> IPv4Address:
        ...

    def special_ip_v4(self, purpose: Optional[IPv4Purpose] = None) -> str:
        ...