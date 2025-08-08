"""Support UPNP discovery method that mimics Hue hubs."""
from __future__ import annotations
import asyncio
from contextlib import suppress
import logging
import socket
from typing import cast, Optional, Tuple
from aiohttp import web
from homeassistant import core
from homeassistant.components.http import HomeAssistantView
from .config import Config
from .const import HUE_SERIAL_NUMBER, HUE_UUID

_LOGGER: logging.Logger = logging.getLogger(__name__)
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
    def func_x57k7gv2(self, request: web.Request) -> web.Response:
        """Handle a GET request."""
        resp_text: str = f"""<?xml version="1.0" encoding="UTF-8" ?>
<root xmlns="urn:schemas-upnp-org:device-1-0">
<specVersion>
<major>1</major>
<minor>0</minor>
</specVersion>
<URLBase>http://{self.config.advertise_ip}:{self.config.advertise_port}/</URLBase>
<device>
<deviceType>urn:schemas-upnp-org:device:Basic:1</deviceType>
<friendlyName>Home Assistant Bridge ({self.config.advertise_ip})</friendlyName>
<manufacturer>Royal Philips Electronics</manufacturer>
<manufacturerURL>http://www.philips.com</manufacturerURL>
<modelDescription>Philips hue Personal Wireless Lighting</modelDescription>
<modelName>Philips hue bridge 2015</modelName>
<modelNumber>BSB002</modelNumber>
<modelURL>http://www.meethue.com</modelURL>
<serialNumber>{HUE_SERIAL_NUMBER}</serialNumber>
<UDN>uuid:{HUE_UUID}</UDN>
</device>
</root>
"""
        return web.Response(text=resp_text, content_type='text/xml')


class UPNPResponderProtocol(asyncio.Protocol):
    """Handle responding to UPNP/SSDP discovery requests."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        ssdp_socket: socket.socket,
        advertise_ip: str,
        advertise_port: int
    ) -> None:
        """Initialize the class."""
        self.transport: Optional[asyncio.DatagramTransport] = None
        self._loop: asyncio.AbstractEventLoop = loop
        self._sock: socket.socket = ssdp_socket
        self.advertise_ip: str = advertise_ip
        self.advertise_port: int = advertise_port
        self._upnp_root_response: bytes = self._prepare_response(
            'upnp:rootdevice',
            f'uuid:{HUE_UUID}::upnp:rootdevice'
        )
        self._upnp_device_response: bytes = self._prepare_response(
            'urn:schemas-upnp-org:device:basic:1',
            f'uuid:{HUE_UUID}'
        )

    def func_i24qdgq5(self, transport: asyncio.DatagramTransport) -> None:
        """Set the transport."""
        self.transport = cast(asyncio.DatagramTransport, transport)

    def func_0exov59g(self, exc: Exception) -> None:
        """Handle connection lost."""
        pass  # Implement handling if necessary

    def func_ogn2fhw9(self, data: bytes, addr: Tuple[str, int]) -> None:
        """Respond to msearch packets."""
        decoded_data: str = data.decode('utf-8', errors='ignore')
        if 'M-SEARCH' not in decoded_data:
            return
        _LOGGER.debug('UPNP Responder M-SEARCH method received: %s', data)
        response: bytes = self._handle_request(decoded_data)
        _LOGGER.debug('UPNP Responder responding with: %s', response)
        assert self.transport is not None
        self.transport.sendto(response, addr)

    def func_lhqcdb8c(self, exc: Exception) -> None:
        """Log UPNP errors."""
        _LOGGER.error('UPNP Error received: %s', exc)

    def func_4icttkzr(self) -> None:
        """Stop the server."""
        _LOGGER.info('UPNP responder shutting down')
        if self.transport:
            self.transport.close()
        self._loop.remove_writer(self._sock.fileno())
        self._loop.remove_reader(self._sock.fileno())
        self._sock.close()

    def func_06ztctx3(self, decoded_data: str) -> bytes:
        """Determine which response to send based on the decoded data."""
        if 'upnp:rootdevice' in decoded_data:
            return self._upnp_root_response
        return self._upnp_device_response

    def func_oqvxp5pd(self, search_target: str, unique_service_name: str) -> bytes:
        """Construct the UPNP response."""
        response: str = f"""HTTP/1.1 200 OK
CACHE-CONTROL: max-age=60
EXT:
LOCATION: http://{self.advertise_ip}:{self.advertise_port}/description.xml
SERVER: FreeRTOS/6.0.5, UPnP/1.0, IpBridge/1.16.0
hue-bridgeid: {HUE_SERIAL_NUMBER}
ST: {search_target}
USN: {unique_service_name}

"""
        return response.replace('\n', '\r\n').encode('utf-8')

    def _prepare_response(self, search_target: str, unique_service_name: str) -> bytes:
        """Prepare the UPNP response based on search target and USN."""
        response: str = f"""HTTP/1.1 200 OK
CACHE-CONTROL: max-age=60
EXT:
LOCATION: http://{self.advertise_ip}:{self.advertise_port}/description.xml
SERVER: FreeRTOS/6.0.5, UPnP/1.0, IpBridge/1.16.0
hue-bridgeid: {HUE_SERIAL_NUMBER}
ST: {search_target}
USN: {unique_service_name}

"""
        return response.replace('\n', '\r\n').encode('utf-8')

    def _handle_request(self, decoded_data: str) -> bytes:
        """Handle the M-SEARCH request and return the appropriate response."""
        lines = decoded_data.split('\r\n')
        search_target = ""
        unique_service_name = ""
        for line in lines:
            if line.upper().startswith('ST:'):
                search_target = line[3:].strip()
            elif line.upper().startswith('USN:'):
                unique_service_name = line[4:].strip()
        return self.func_oqvxp5pd(search_target, unique_service_name)


async def func_a42clbr1(
    host_ip_addr: str,
    upnp_bind_multicast: bool,
    advertise_ip: str,
    advertise_port: int
) -> UPNPResponderProtocol:
    """Create the UPNP socket and protocol."""
    ssdp_socket: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ssdp_socket.setblocking(False)
    ssdp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    with suppress(AttributeError):
        ssdp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    ssdp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    ssdp_socket.setsockopt(
        socket.SOL_IP,
        socket.IP_MULTICAST_IF,
        socket.inet_aton(host_ip_addr)
    )
    ssdp_socket.setsockopt(
        socket.SOL_IP,
        socket.IP_ADD_MEMBERSHIP,
        socket.inet_aton(BROADCAST_ADDR) + socket.inet_aton(host_ip_addr)
    )
    ssdp_socket.bind((
        '' if upnp_bind_multicast else host_ip_addr,
        BROADCAST_PORT
    ))
    loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
    transport, protocol = await loop.create_datagram_endpoint(
        lambda: UPNPResponderProtocol(loop, ssdp_socket, advertise_ip, advertise_port),
        sock=ssdp_socket
    )
    return cast(UPNPResponderProtocol, protocol)
