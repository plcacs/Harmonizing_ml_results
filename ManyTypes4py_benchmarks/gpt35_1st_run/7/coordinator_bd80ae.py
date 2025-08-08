from __future__ import annotations
import asyncio
from collections.abc import Mapping
from datetime import datetime, timedelta
import email
from email.header import decode_header, make_header
from email.message import Message
from email.utils import parseaddr, parsedate_to_datetime
import logging
from typing import TYPE_CHECKING, Any
from aioimaplib import AUTH, IMAP4_SSL, NONAUTH, SELECTED, AioImapException
from homeassistant.const import CONF_PASSWORD, CONF_PORT, CONF_USERNAME, CONF_VERIFY_SSL, CONTENT_TYPE_TEXT_PLAIN
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryError, TemplateError
from homeassistant.helpers.json import json_bytes
from homeassistant.helpers.template import Template
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util
from homeassistant.util.ssl import SSLCipherList, client_context, create_no_verify_ssl_context
from .const import CONF_CHARSET, CONF_CUSTOM_EVENT_DATA_TEMPLATE, CONF_EVENT_MESSAGE_DATA, CONF_FOLDER, CONF_MAX_MESSAGE_SIZE, CONF_SEARCH, CONF_SERVER, CONF_SSL_CIPHER_LIST, DEFAULT_MAX_MESSAGE_SIZE, DOMAIN, MESSAGE_DATA_OPTIONS
from .errors import InvalidAuth, InvalidFolder

if TYPE_CHECKING:
    from . import ImapConfigEntry

_LOGGER: logging.Logger = logging.getLogger(__name__)
BACKOFF_TIME: int = 10
EVENT_IMAP: str = 'imap_content'
MAX_ERRORS: int = 3
MAX_EVENT_DATA_BYTES: int = 32168
DIAGNOSTICS_ATTRIBUTES: list[str] = ['date', 'initial']

async def connect_to_server(data: Mapping[str, Any]) -> IMAP4_SSL:
    ...

class ImapMessage:
    def __init__(self, raw_message: bytes) -> None:
        ...

    @staticmethod
    def _decode_payload(part: email.message.Message) -> str:
        ...

    @property
    def headers(self) -> dict[str, list[str]]:
        ...

    @property
    def message_id(self) -> str | None:
        ...

    @property
    def date(self) -> datetime | None:
        ...

    @property
    def sender(self) -> str:
        ...

    @property
    def subject(self) -> str:
        ...

    @property
    def text(self) -> str:
        ...

class ImapDataUpdateCoordinator(DataUpdateCoordinator[int | None]):
    def __init__(self, hass: HomeAssistant, imap_client: IMAP4_SSL, entry: ImapConfigEntry, update_interval: timedelta) -> None:
        ...

    async def async_start(self) -> None:
        ...

    async def _async_reconnect_if_needed(self) -> None:
        ...

    async def _async_process_event(self, last_message_uid: str) -> None:
        ...

    async def _async_fetch_number_of_messages(self) -> int:
        ...

    async def _cleanup(self, log_error: bool = False) -> None:
        ...

    async def shutdown(self, *args: Any) -> None:
        ...

    def _update_diagnostics(self, data: dict[str, Any]) -> None:
        ...

    @property
    def diagnostics_data(self) -> dict[str, Any]:
        ...

class ImapPollingDataUpdateCoordinator(ImapDataUpdateCoordinator):
    def __init__(self, hass: HomeAssistant, imap_client: IMAP4_SSL, entry: ImapConfigEntry) -> None:
        ...

    async def _async_update_data(self) -> int:
        ...

class ImapPushDataUpdateCoordinator(ImapDataUpdateCoordinator):
    def __init__(self, hass: HomeAssistant, imap_client: IMAP4_SSL, entry: ImapConfigEntry) -> None:
        ...

    async def _async_update_data(self) -> int:
        ...

    async def async_start(self) -> None:
        ...

    async def _async_wait_push_loop(self) -> None:
        ...

    async def shutdown(self, *args: Any) -> None:
        ...
