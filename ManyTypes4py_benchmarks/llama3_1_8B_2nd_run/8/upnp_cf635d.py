"""Support UPNP discovery method that mimics Hue hubs."""
from __future__ import annotations
import asyncio
from contextlib import suppress
import logging
import socket
from typing import cast
from aiohttp import web
from homeassistant import core
from homeassistant.components.http import HomeAssistantView
from .config import Config
from .const import HUE_SERIAL_NUMBER, HUE_UUID
_LOGGER = logging.getLogger(__name__)
BROADCAST_PORT: int = 1900
BROADCAST_ADDR: str = '239.255.255.250'

class DescriptionXmlView(HomeAssistantView):
    """Handles requests for the description.xml file."""
    url: str = '/description.xml'
    name: str = 'description:xml'
    requires_auth: bool = False

    def __init__(self, config: Config) -> None:
        """Initialize the instance of the view."""
        self.config: Config = config

    @core.callback
    def get(self, request: web.Request) -> web.Response:
        """Handle a GET request."""
        resp_text: str = f'<?xml version="1.0" encoding="UTF-8" ?>\n<root xmlns="urn:schemas-upnp-org:device-1-0">\n<specVersion>\n<major>1</major>\n<minor>0</minor>\n</specVersion>\n<URLBase>http://{self.config.advertise_ip}:{self.config.advertise_port}/</URLBase>\n<device>\n<deviceType>urn:schemas-upnp-org:device:Basic:1</deviceType>\n<friendlyName>Home Assistant Bridge ({self.config.advertise_ip})</friendlyName>\n<manufacturer>Royal Philips Electronics</manufacturer>\n<manufacturerURL>http://www.philips.com</manufacturerURL>\n<modelDescription>Philips hue Personal Wireless Lighting</modelDescription>\n<modelName>Philips hue bridge 2015</modelName>\n<modelNumber>BSB002</modelNumber>\n<modelURL>http://www.meethue.com</modelURL>\n<serialNumber>{HUE_SERIAL_NUMBER}</serialNumber>\n<UDN>uuid:{HUE_UUID}</UDN>\n</device>\n</root>\n'
        return web.Response(text=resp_text, content_type='text/xml')

class UPNPResponderProtocol(asyncio.Protocol):
    """Handle responding to UPNP/SSDP discovery requests."""

    def __init__(self, loop: asyncio.BaseEventLoop, ssdp_socket: socket.socket, advertise_ip: str, advertise_port: int) -> None:
        """Initialize the class."""
        self.transport: asyncio.BaseTransport | None = None
        self._loop: asyncio.BaseEventLoop = loop
        self._sock: socket.socket = ssdp_socket
        self.advertise_ip: str = advertise_ip
        self.advertise_port: int = advertise_port
        self._upnp_root_response: bytes = self._prepare_response('upnp:rootdevice', f'uuid:{HUE_UUID}::upnp:rootdevice')
        self._upnp_device_response: bytes = self._prepare_response('urn:schemas-upnp-org:device:basic:1', f'uuid:{HUE_UUID}')

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        """Set the transport."""
        self.transport = cast(asyncio.DatagramTransport, transport)

    def connection_lost(self, exc: Exception | None) -> None:
        """Handle connection lost."""

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        """Respond to msearch packets."""
        decoded_data: str = data.decode('utf-8', errors='ignore')
        if 'M-SEARCH' not in decoded_data:
            return
        _LOGGER.debug('UPNP Responder M-SEARCH method received: %s', data)
        response: bytes = self._handle_request(decoded_data)
        _LOGGER.debug('UPNP Responder responding with: %s', response)
        assert self.transport is not None
        self.transport.sendto(response, addr)

    def error_received(self, exc: Exception) -> None:
        """Log UPNP errors."""
        _LOGGER.error('UPNP Error received: %s', exc)

    def close(self) -> None:
        """Stop the server."""
        _LOGGER.info('UPNP responder shutting down')
        if self.transport:
            self.transport.close()
        self._loop.remove_writer(self._sock.fileno())
        self._loop.remove_reader(self._sock.fileno())
        self._sock.close()

    def _handle_request(self, decoded_data: str) -> bytes:
        if 'upnp:rootdevice' in decoded_data:
            return self._upnp_root_response
        return self._upnp_device_response

    def _prepare_response(self, search_target: str, unique_service_name: str) -> bytes:
        response: str = f'HTTP/1.1 200 OK\nCACHE-CONTROL: max-age=60\nEXT:\nLOCATION: http://{self.advertise_ip}:{self.advertise_port}/description.xml\nSERVER: FreeRTOS/6.0.5, UPnP/1.0, IpBridge/1.16.0\nhue-bridgeid: {HUE_SERIAL_NUMBER}\nST: {search_target}\nUSN: {unique_service_name}\n\n'
        return response.replace('\n', '\r\n').encode('utf-8')

async def async_create_upnp_datagram_endpoint(host_ip_addr: str, upnp_bind_multicast: bool, advertise_ip: str, advertise_port: int) -> asyncio.Transport:
    """Create the UPNP socket and protocol."""
    ssdp_socket: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ssdp_socket.setblocking(False)
    ssdp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    with suppress(AttributeError):
        ssdp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    ssdp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    ssdp_socket.setsockopt(socket.SOL_IP, socket.IP_MULTICAST_IF, socket.inet_aton(host_ip_addr))
    ssdp_socket.setsockopt(socket.SOL_IP, socket.IP_ADD_MEMBERSHIP, socket.inet_aton(BROADCAST_ADDR) + socket.inet_aton(host_ip_addr))
    ssdp_socket.bind(('' if upnp_bind_multicast else host_ip_addr, BROADCAST_PORT))
    loop: asyncio.BaseEventLoop = asyncio.get_event_loop()
    transport_protocol: asyncio.BaseTransport = await loop.create_datagram_endpoint(lambda: UPNPResponderProtocol(loop, ssdp_socket, advertise_ip, advertise_port), sock=ssdp_socket)
    return transport_protocol
