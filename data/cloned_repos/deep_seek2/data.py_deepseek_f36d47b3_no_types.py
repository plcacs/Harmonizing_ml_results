"""Support for RESTful API."""
from __future__ import annotations
import logging
import ssl
from typing import Any, Dict, Optional, Tuple, Union
import httpx
import xmltodict
from homeassistant.core import HomeAssistant
from homeassistant.helpers import template
from homeassistant.helpers.httpx_client import create_async_httpx_client
from homeassistant.helpers.json import json_dumps
from homeassistant.util.ssl import SSLCipherList
from .const import XML_MIME_TYPES
DEFAULT_TIMEOUT = 10
_LOGGER = logging.getLogger(__name__)

class RestData:
    """Class for handling the data retrieval."""

    def __init__(self, hass, method, resource, encoding, auth, headers, params, data, verify_ssl, ssl_cipher_list, timeout=DEFAULT_TIMEOUT):
        """Initialize the data object."""
        self._hass: HomeAssistant = hass
        self._method: str = method
        self._resource: str = resource
        self._encoding: str = encoding
        self._auth: Optional[Union[httpx.DigestAuth, Tuple[str, str]]] = auth
        self._headers: Optional[Dict[str, str]] = headers
        self._params: Optional[Dict[str, str]] = params
        self._request_data: Optional[str] = data
        self._timeout: int = timeout
        self._verify_ssl: bool = verify_ssl
        self._ssl_cipher_list: SSLCipherList = SSLCipherList(ssl_cipher_list)
        self._async_client: Optional[httpx.AsyncClient] = None
        self.data: Optional[str] = None
        self.last_exception: Optional[Exception] = None
        self.headers: Optional[httpx.Headers] = None

    def set_payload(self, payload):
        """Set request data."""
        self._request_data = payload

    @property
    def url(self):
        """Get url."""
        return self._resource

    def set_url(self, url):
        """Set url."""
        self._resource = url

    def data_without_xml(self):
        """If the data is an XML string, convert it to a JSON string."""
        _LOGGER.debug('Data fetched from resource: %s', self.data)
        if (value := self.data) is not None and (headers := self.headers) is not None and (content_type := headers.get('content-type')) and content_type.startswith(XML_MIME_TYPES):
            value = json_dumps(xmltodict.parse(value))
            _LOGGER.debug('JSON converted from XML: %s', value)
        return value

    async def async_update(self, log_errors: bool=True) -> None:
        """Get the latest data from REST service with provided method."""
        if not self._async_client:
            self._async_client = create_async_httpx_client(self._hass, verify_ssl=self._verify_ssl, default_encoding=self._encoding, ssl_cipher_list=self._ssl_cipher_list)
        rendered_headers: Optional[Dict[str, Any]] = template.render_complex(self._headers, parse_result=False)
        rendered_params: Optional[Dict[str, Any]] = template.render_complex(self._params)
        _LOGGER.debug('Updating from %s', self._resource)
        try:
            response: httpx.Response = await self._async_client.request(self._method, self._resource, headers=rendered_headers, params=rendered_params, auth=self._auth, content=self._request_data, timeout=self._timeout, follow_redirects=True)
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