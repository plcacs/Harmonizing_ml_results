from __future__ import annotations
import asyncio
from contextlib import suppress
import logging
import socket
from typing import cast, Tuple
from aiohttp import web
from homeassistant import core
from homeassistant.components.http import HomeAssistantView
from .config import Config
from .const import HUE_SERIAL_NUMBER, HUE_UUID

_LOGGER: logging.Logger = logging.getLogger(__name__)
BROADCAST_PORT: int = 1900
BROADCAST_ADDR: str = '239.255.255.250'


class DescriptionXmlView(HomeAssistantView):
    url: str = '/description.xml'
    name: str = 'description:xml'
    requires_auth: bool = False

    def __init__(self, config: Config) -> None:
        self.config: Config = config

    @core.callback
    def get(self, request: web.Request) -> web.Response:
        resp_text: str = (
            f'<?xml version="1.0" encoding="UTF-8" ?>\n'
            f'<root xmlns="urn:schemas-upnp-org:device-1-0">\n'
            f'<specVersion>\n'
            f'<major>1</major>\n'
            f'<minor>0</minor>\n'
            f'</specVersion>\n'
            f'<URLBase>http://{self.config.advertise_ip}:{self.config.advertise_port}/</URLBase>\n'
            f'<device>\n'
            f'<deviceType>urn:schemas-upnp-org:device:Basic:1</deviceType>\n'
            f'<friendlyName>Home Assistant Bridge ({self.config.advertise_ip})</friendlyName>\n'
            f'<manufacturer>Royal Philips Electronics</manufacturer>\n'
            f'<manufacturerURL>http://www.philips.com</manufacturerURL>\n'
            f'<modelDescription>Philips hue Personal Wireless Lighting</modelDescription>\n'
            f'<modelName>Philips hue bridge 2015</modelName>\n'
            f'<modelNumber>BSB002</modelNumber>\n'
            f'<modelURL>http://www.meethue.com</modelURL>\n'
            f'<serialNumber>{HUE_SERIAL_NUMBER}</serialNumber>\n'
            f'<UDN>uuid:{HUE_UUID}</UDN>\n'
            f'</device>\n'
            f'</root>\n'
        )
        return web.Response(text=resp_text, content_type='text/xml')


class UPNPResponderProtocol(asyncio.Protocol):
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        ssdp_socket: socket.socket,
        advertise_ip: str,
        advertise_port: int,
    ) -> None:
        self.transport: asyncio.DatagramTransport | None = None
        self._loop: asyncio.AbstractEventLoop = loop
        self._sock: socket.socket = ssdp_socket
        self.advertise_ip: str = advertise_ip
        self.advertise_port: int = advertise_port
        self._upnp_root_response: bytes = self._prepare_response(
            'upnp:rootdevice', f'uuid:{HUE_UUID}::upnp:rootdevice'
        )
        self._upnp_device_response: bytes = self._prepare_response(
            'urn:schemas-upnp-org:device:basic:1', f'uuid:{HUE_UUID}'
        )

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:
        self.transport = cast(asyncio.DatagramTransport, transport)

    def connection_lost(self, exc: Exception | None) -> None:
        pass

    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
        decoded_data: str = data.decode('utf-8', errors='ignore')
        if 'M-SEARCH' not in decoded_data:
            return
        _LOGGER.debug('UPNP Responder M-SEARCH method received: %s', data)
        response: bytes = self._handle_request(decoded_data)
        _LOGGER.debug('UPNP Responder responding with: %s', response)
        assert self.transport is not None
        self.transport.sendto(response, addr)

    def error_received(self, exc: Exception) -> None:
        _LOGGER.error('UPNP Error received: %s', exc)

    def close(self) -> None:
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
        response: str = (
            f'HTTP/1.1 200 OK\n'
            f'CACHE-CONTROL: max-age=60\n'
            f'EXT:\n'
            f'LOCATION: http://{self.advertise_ip}:{self.advertise_port}/description.xml\n'
            f'SERVER: FreeRTOS/6.0.5, UPnP/1.0, IpBridge/1.16.0\n'
            f'hue-bridgeid: {HUE_SERIAL_NUMBER}\n'
            f'ST: {search_target}\n'
            f'USN: {unique_service_name}\n'
            f'\n'
        )
        return response.replace('\n', '\r\n').encode('utf-8')


async def async_create_upnp_datagram_endpoint(
    host_ip_addr: str,
    upnp_bind_multicast: bool,
    advertise_ip: str,
    advertise_port: int,
) -> asyncio.DatagramProtocol:
    ssdp_socket: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ssdp_socket.setblocking(False)
    ssdp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    with suppress(AttributeError):
        ssdp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    ssdp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    ssdp_socket.setsockopt(
        socket.SOL_IP, socket.IP_MULTICAST_IF, socket.inet_aton(host_ip_addr)
    )
    ssdp_socket.setsockopt(
        socket.SOL_IP,
        socket.IP_ADD_MEMBERSHIP,
        socket.inet_aton(BROADCAST_ADDR) + socket.inet_aton(host_ip_addr),
    )
    ssdp_socket.bind(("" if upnp_bind_multicast else host_ip_addr, BROADCAST_PORT))
    loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
    transport_protocol: Tuple[asyncio.DatagramTransport, asyncio.DatagramProtocol] = await loop.create_datagram_endpoint(
        lambda: UPNPResponderProtocol(loop, ssdp_socket, advertise_ip, advertise_port),
        sock=ssdp_socket,
    )
    return transport_protocol[1]