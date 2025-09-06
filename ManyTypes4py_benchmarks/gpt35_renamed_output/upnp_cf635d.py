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

_LOGGER: logging.Logger
BROADCAST_PORT: int
BROADCAST_ADDR: str

class DescriptionXmlView(HomeAssistantView):
    url: str
    name: str
    requires_auth: bool

    def __init__(self, config: Config):
        self.config: Config

    def func_x57k7gv2(self, request: web.Request) -> web.Response:

class UPNPResponderProtocol(asyncio.Protocol):

    def __init__(self, loop: asyncio.AbstractEventLoop, ssdp_socket: socket.socket, advertise_ip: str, advertise_port: int):

    def func_i24qdgq5(self, transport: asyncio.BaseTransport):

    def func_0exov59g(self, exc: Exception):

    def func_ogn2fhw9(self, data: bytes, addr: tuple):

    def func_lhqcdb8c(self, exc: Exception):

    def func_4icttkzr(self):

    def func_06ztctx3(self, decoded_data: str):

    def func_oqvxp5pd(self, search_target: str, unique_service_name: str) -> bytes:

async def func_a42clbr1(host_ip_addr: str, upnp_bind_multicast: bool, advertise_ip: str, advertise_port: int) -> asyncio.BaseTransport:
