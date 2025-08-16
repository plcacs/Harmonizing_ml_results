from __future__ import annotations
import logging
import ssl
import httpx
import xmltodict
from homeassistant.core import HomeAssistant
from homeassistant.helpers import template
from homeassistant.helpers.httpx_client import create_async_httpx_client
from homeassistant.helpers.json import json_dumps
from homeassistant.util.ssl import SSLCipherList
from .const import XML_MIME_TYPES
DEFAULT_TIMEOUT: int = 10
_LOGGER: logging.Logger = logging.getLogger(__name__)

class RestData:
    def __init__(self, hass: HomeAssistant, method: str, resource: str, encoding: str, auth: tuple[str, str], headers: dict[str, str], params: dict[str, str], data: bytes, verify_ssl: bool, ssl_cipher_list: str, timeout: int = DEFAULT_TIMEOUT) -> None:
        self._hass: HomeAssistant = hass
        self._method: str = method
        self._resource: str = resource
        self._encoding: str = encoding
        self._auth: tuple[str, str] = auth
        self._headers: dict[str, str] = headers
        self._params: dict[str, str] = params
        self._request_data: bytes = data
        self._timeout: int = timeout
        self._verify_ssl: bool = verify_ssl
        self._ssl_cipher_list: SSLCipherList = SSLCipherList(ssl_cipher_list)
        self._async_client: httpx.AsyncClient | None = None
        self.data: bytes | None = None
        self.last_exception: Exception | None = None
        self.headers: dict[str, str] | None = None

    def set_payload(self, payload: bytes) -> None:
        self._request_data = payload

    @property
    def url(self) -> str:
        return self._resource

    def set_url(self, url: str) -> None:
        self._resource = url

    def data_without_xml(self) -> bytes | None:
        if (value := self.data) is not None and (headers := self.headers) is not None and (content_type := headers.get('content-type')) and content_type.startswith(XML_MIME_TYPES):
            value = json_dumps(xmltodict.parse(value))
        return value

    async def async_update(self, log_errors: bool = True) -> None:
        if not self._async_client:
            self._async_client = create_async_httpx_client(self._hass, verify_ssl=self._verify_ssl, default_encoding=self._encoding, ssl_cipher_list=self._ssl_cipher_list)
        rendered_headers = template.render_complex(self._headers, parse_result=False)
        rendered_params = template.render_complex(self._params)
        try:
            response = await self._async_client.request(self._method, self._resource, headers=rendered_headers, params=rendered_params, auth=self._auth, content=self._request_data, timeout=self._timeout, follow_redirects=True)
            self.data = response.text
            self.headers = response.headers
        except httpx.TimeoutException as ex:
            if log_errors:
                _LOGGER.error('Timeout while fetching data: %s', self._resource)
            self.last_exception = ex
            self.data = None
            self.headers = None
        except httpx.RequestError as ex:
            if log_errors:
                _LOGGER.error('Error fetching data: %s failed with %s', self._resource, ex)
            self.last_exception = ex
            self.data = None
            self.headers = None
        except ssl.SSLError as ex:
            if log_errors:
                _LOGGER.error('Error connecting to %s failed with %s', self._resource, ex)
            self.last_exception = ex
            self.data = None
            self.headers = None
