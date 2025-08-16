from __future__ import annotations
import asyncio
import logging
import socket
from typing import cast
from aiohttp import web
from homeassistant import core
from homeassistant.components.http import HomeAssistantView
from .config import Config
from .const import HUE_SERIAL_NUMBER, HUE_UUID

_LOGGER: logging.Logger
BROADCAST_PORT: int
BROADCAST_ADDR: str

class DescriptionXmlView(HomeAssistantView):
    url: str
    name: str
    requires_auth: bool

    def __init__(self, config: Config):
        self.config: Config

    def get(self, request: web.Request) -> web.Response:

class UPNPResponderProtocol(asyncio.Protocol):

    def __init__(self, loop: asyncio.AbstractEventLoop, ssdp_socket: socket.socket, advertise_ip: str, advertise_port: int):

    def connection_made(self, transport: asyncio.transports.DatagramTransport):

    def connection_lost(self, exc):

    def datagram_received(self, data: bytes, addr):

    def error_received(self, exc):

    def close(self):

    def _handle_request(self, decoded_data: str) -> bytes:

    def _prepare_response(self, search_target: str, unique_service_name: str) -> bytes:

async def async_create_upnp_datagram_endpoint(host_ip_addr: str, upnp_bind_multicast: bool, advertise_ip: str, advertise_port: int) -> asyncio.transports.DatagramTransport:
