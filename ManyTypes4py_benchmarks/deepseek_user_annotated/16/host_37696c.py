"""Module which encapsulates the NVR/camera API and subscription."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Mapping
import logging
from time import time
from typing import Any, Literal, Optional, Dict, List, Set, Callable, Awaitable, Union

import aiohttp
from aiohttp.web import Request
from reolink_aio.api import ALLOWED_SPECIAL_CHARS, Host
from reolink_aio.enums import SubType
from reolink_aio.exceptions import NotSupportedError, ReolinkError, SubscriptionError

from homeassistant.components import webhook
from homeassistant.const import (
    CONF_HOST,
    CONF_PASSWORD,
    CONF_PORT,
    CONF_PROTOCOL,
    CONF_USERNAME,
)
from homeassistant.core import CALLBACK_TYPE, HassJob, HomeAssistant, callback
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.device_registry import format_mac
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.event import async_call_later
from homeassistant.helpers.network import NoURLAvailableError, get_url
from homeassistant.helpers.storage import Store
from homeassistant.util.ssl import SSLCipherList

from .const import CONF_SUPPORTS_PRIVACY_MODE, CONF_USE_HTTPS, DOMAIN
from .exceptions import (
    PasswordIncompatible,
    ReolinkSetupException,
    ReolinkWebhookException,
    UserNotAdmin,
)
from .util import get_store

DEFAULT_TIMEOUT: int = 30
FIRST_TCP_PUSH_TIMEOUT: int = 10
FIRST_ONVIF_TIMEOUT: int = 10
FIRST_ONVIF_LONG_POLL_TIMEOUT: int = 90
SUBSCRIPTION_RENEW_THRESHOLD: int = 300
POLL_INTERVAL_NO_PUSH: int = 5
LONG_POLL_COOLDOWN: float = 0.75
LONG_POLL_ERROR_COOLDOWN: int = 30
BATTERY_WAKE_UPDATE_INTERVAL: int = 3600

_LOGGER: logging.Logger = logging.getLogger(__name__)


class ReolinkHost:
    """The implementation of the Reolink Host class."""

    def __init__(
        self,
        hass: HomeAssistant,
        config: Mapping[str, Any],
        options: Mapping[str, Any],
        config_entry_id: str | None = None,
    ) -> None:
        """Initialize Reolink Host. Could be either NVR, or Camera."""
        self._hass: HomeAssistant = hass
        self._config_entry_id: str | None = config_entry_id
        self._config: Mapping[str, Any] = config
        self._unique_id: str = ""

        def get_aiohttp_session() -> aiohttp.ClientSession:
            """Return the HA aiohttp session."""
            return async_get_clientsession(
                hass,
                verify_ssl=False,
                ssl_cipher=SSLCipherList.INSECURE,
            )

        self._api: Host = Host(
            config[CONF_HOST],
            config[CONF_USERNAME],
            config[CONF_PASSWORD],
            port=config.get(CONF_PORT),
            use_https=config.get(CONF_USE_HTTPS),
            protocol=options[CONF_PROTOCOL],
            timeout=DEFAULT_TIMEOUT,
            aiohttp_get_session_callback=get_aiohttp_session,
        )

        self.last_wake: float = 0
        self.update_cmd: defaultdict[str, defaultdict[int | None, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.firmware_ch_list: list[int | None] = []

        self.starting: bool = True
        self.privacy_mode: bool | None = None
        self.credential_errors: int = 0

        self.webhook_id: str | None = None
        self._onvif_push_supported: bool = True
        self._onvif_long_poll_supported: bool = True
        self._base_url: str = ""
        self._webhook_url: str = ""
        self._webhook_reachable: bool = False
        self._long_poll_received: bool = False
        self._long_poll_error: bool = False
        self._cancel_poll: CALLBACK_TYPE | None = None
        self._cancel_tcp_push_check: CALLBACK_TYPE | None = None
        self._cancel_onvif_check: CALLBACK_TYPE | None = None
        self._cancel_long_poll_check: CALLBACK_TYPE | None = None
        self._poll_job: HassJob = HassJob(self._async_poll_all_motion, cancel_on_shutdown=True)
        self._fast_poll_error: bool = False
        self._long_poll_task: asyncio.Task | None = None
        self._lost_subscription_start: bool = False
        self._lost_subscription: bool = False
        self.cancel_refresh_privacy_mode: CALLBACK_TYPE | None = None

    @callback
    def async_register_update_cmd(self, cmd: str, channel: int | None = None) -> None:
        """Register the command to update the state."""
        self.update_cmd[cmd][channel] += 1

    @callback
    def async_unregister_update_cmd(self, cmd: str, channel: int | None = None) -> None:
        """Unregister the command to update the state."""
        self.update_cmd[cmd][channel] -= 1
        if not self.update_cmd[cmd][channel]:
            del self.update_cmd[cmd][channel]
        if not self.update_cmd[cmd]:
            del self.update_cmd[cmd]

    @property
    def unique_id(self) -> str:
        """Create the unique ID, base for all entities."""
        return self._unique_id

    @property
    def api(self) -> Host:
        """Return the API object."""
        return self._api

    async def async_init(self) -> None:
        """Connect to Reolink host."""
        if not self._api.valid_password():
            raise PasswordIncompatible(
                "Reolink password contains incompatible special character, "
                "please change the password to only contain characters: "
                f"a-z, A-Z, 0-9 or {ALLOWED_SPECIAL_CHARS}"
            )

        store: Store[str] | None = None
        if self._config_entry_id is not None:
            store = get_store(self._hass, self._config_entry_id)
            if self._config.get(CONF_SUPPORTS_PRIVACY_MODE) and (
                data := await store.async_load()
            ):
                self._api.set_raw_host_data(data)

        await self._api.get_host_data()

        if self._api.mac_address is None:
            raise ReolinkSetupException("Could not get mac address")

        if not self._api.is_admin:
            raise UserNotAdmin(
                f"User '{self._api.username}' has authorization level "
                f"'{self._api.user_level}', only admin users can change camera settings"
            )

        self.privacy_mode = self._api.baichuan.privacy_mode()

        if (
            store
            and self._api.supported(None, "privacy_mode")
            and not self.privacy_mode
        ):
            _LOGGER.debug(
                "Saving raw host data for next reload in case privacy mode is enabled"
            )
            data = self._api.get_raw_host_data()
            await store.async_save(data)

        onvif_supported: bool = self._api.supported(None, "ONVIF")
        self._onvif_push_supported = onvif_supported
        self._onvif_long_poll_supported = onvif_supported

        enable_rtsp: bool | None = None
        enable_onvif: bool | None = None
        enable_rtmp: bool | None = None

        if not self._api.rtsp_enabled:
            _LOGGER.debug(
                "RTSP is disabled on %s, trying to enable it", self._api.nvr_name
            )
            enable_rtsp = True

        if not self._api.onvif_enabled and onvif_supported:
            _LOGGER.debug(
                "ONVIF is disabled on %s, trying to enable it", self._api.nvr_name
            )
            enable_onvif = True

        if not self._api.rtmp_enabled and self._api.protocol == "rtmp":
            _LOGGER.debug(
                "RTMP is disabled on %s, trying to enable it", self._api.nvr_name
            )
            enable_rtmp = True

        if enable_onvif or enable_rtmp or enable_rtsp:
            try:
                await self._api.set_net_port(
                    enable_onvif=enable_onvif,
                    enable_rtmp=enable_rtmp,
                    enable_rtsp=enable_rtsp,
                )
            except ReolinkError:
                ports: str = ""
                if enable_rtsp:
                    ports += "RTSP "

                if enable_onvif:
                    ports += "ONVIF "

                if enable_rtmp:
                    ports += "RTMP "

                ir.async_create_issue(
                    self._hass,
                    DOMAIN,
                    "enable_port",
                    is_fixable=False,
                    severity=ir.IssueSeverity.WARNING,
                    translation_key="enable_port",
                    translation_placeholders={
                        "name": self._api.nvr_name,
                        "ports": ports,
                        "info_link": "https://support.reolink.com/hc/en-us/articles/900004435763-How-to-Set-up-Reolink-Ports-Settings-via-Reolink-Client-New-Client-",
                    },
                )
        else:
            ir.async_delete_issue(self._hass, DOMAIN, "enable_port")

        if self._api.supported(None, "UID"):
            self._unique_id = self._api.uid
        else:
            self._unique_id = format_mac(self._api.mac_address)

        try:
            await self._api.baichuan.subscribe_events()
        except ReolinkError:
            await self._async_check_tcp_push()
        else:
            self._cancel_tcp_push_check = async_call_later(
                self._hass, FIRST_TCP_PUSH_TIMEOUT, self._async_check_tcp_push
            )

        ch_list: list[int | None] = [None]
        if self._api.is_nvr:
            ch_list.extend(self._api.channels)
        for ch in ch_list:
            if not self._api.supported(ch, "firmware"):
                continue

            key: str | int = ch if ch is not None else "host"
            if self._api.camera_sw_version_update_required(ch):
                ir.async_create_issue(
                    self._hass,
                    DOMAIN,
                    f"firmware_update_{key}",
                    is_fixable=False,
                    severity=ir.IssueSeverity.WARNING,
                    translation_key="firmware_update",
                    translation_placeholders={
                        "required_firmware": self._api.camera_sw_version_required(
                            ch
                        ).version_string,
                        "current_firmware": self._api.camera_sw_version(ch),
                        "model": self._api.camera_model(ch),
                        "hw_version": self._api.camera_hardware_version(ch),
                        "name": self._api.camera_name(ch),
                        "download_link": "https://reolink.com/download-center/",
                    },
                )
            else:
                ir.async_delete_issue(self._hass, DOMAIN, f"firmware_update_{key}")

    async def _async_check_tcp_push(self, *_: Any) -> None:
        """Check the TCP push subscription."""
        if self._api.baichuan.events_active:
            ir.async_delete_issue(self._hass, DOMAIN, "webhook_url")
            self._cancel_tcp_push_check = None
            return

        _LOGGER.debug(
            "Reolink %s, did not receive initial TCP push event after %i seconds",
            self._api.nvr_name,
            FIRST_TCP_PUSH_TIMEOUT,
        )

        if self._onvif_push_supported:
            try:
                await self.subscribe()
            except ReolinkError:
                self._onvif_push_supported = False
                self.unregister_webhook()
                await self._api.unsubscribe()
            else:
                if self._api.supported(None, "initial_ONVIF_state"):
                    _LOGGER.debug(
                        "Waiting for initial ONVIF state on webhook '%s'",
                        self._webhook_url,
                    )
                else:
                    _LOGGER.debug(
                        "Camera model %s most likely does not push its initial state"
                        " upon ONVIF subscription, do not check",
                        self._api.model,
                    )
                self._cancel_onvif_check = async_call_later(
                    self._hass, FIRST_ONVIF_TIMEOUT, self._async_check_onvif
                )

        # start long polling if ONVIF push failed immediately
        if not self._onvif_push_supported and not self._api.baichuan.privacy_mode():
            _LOGGER.debug(
                "Camera model %s does not support ONVIF push, using ONVIF long polling instead",
                self._api.model,
            )
            try:
                await self._async_start_long_polling(initial=True)
            except NotSupportedError:
                _LOGGER.debug(
                    "Camera model %s does not support ONVIF long polling, using fast polling instead",
                    self._api.model,
                )
                self._onvif_long_poll_supported = False
                await self._api.unsubscribe()
                await self._async_poll_all_motion()
            else:
                self._cancel_long_poll_check = async_call_later(
                    self._hass,
                    FIRST_ONVIF_LONG_POLL_TIMEOUT,
                    self._async_check_onvif_long_poll,
                )

        self._cancel_tcp_push_check = None

    async def _async_check_onvif(self, *_: Any) -> None:
        """Check the ONVIF subscription."""
        if self._webhook_reachable:
            ir.async_delete_issue(self._hass, DOMAIN, "webhook_url")
            self._cancel_onvif_check = None
            return
        if self._api.supported(None, "initial_ONVIF_state"):
            _LOGGER.debug(
                "Did not receive initial ONVIF state on webhook '%s' after %i seconds",
                self._webhook_url,
                FIRST_ONVIF_TIMEOUT,
            )

        # ONVIF push is not received, start long polling and schedule check
        await self._async_start_long_polling()
        self._cancel_long_poll_check = async_call_later(
            self._hass, FIRST_ONVIF_LONG_POLL_TIMEOUT, self._async_check_onvif_long_poll
        )

        self._cancel_onvif_check = None

    async def _async_check_onvif_long_poll(self, *_: Any) -> None:
        """Check if ONVIF long polling is working."""
        if not self._long_poll_received:
            _LOGGER.debug(
                "Did not receive state through ONVIF long polling after %i seconds",
                FIRST_ONVIF_LONG_POLL_TIMEOUT,
            )
            ir.async_create_issue(
                self._hass,
                DOMAIN,
                "webhook_url",
                is_fixable=False,
                severity=ir.IssueSeverity.WARNING,
                translation_key="webhook_url",
                translation_placeholders={
                    "name": self._api.nvr_name,
                    "base_url": self._base_url,
                    "network_link": "https://my.home-assistant.io/redirect/network/",
                },
            )

            if self._base_url.startswith("https"):
                ir.async_create_issue(
                    self._hass,
                    DOMAIN,
                    "https_webhook",
                    is_fixable=False,
                    severity=ir.IssueSeverity.WARNING,
                    translation_key="https_webhook",
                    translation_placeholders={
                        "base_url": self._base_url,
                        "network_link": "https://my.home-assistant.io/redirect/network/",
                    },
                )
            else:
                ir.async_delete_issue(self._hass, DOMAIN, "https_webhook")

            if self._hass.config.api is not None and self._hass.config.api.use_ssl:
                ir.async_create_issue(
                    self._hass,
                    DOMAIN,
                    "ssl",
                    is_fixable=False,
                    severity=ir.IssueSeverity.WARNING,
                    translation_key="ssl",
                    translation_placeholders={
                        "ssl_link": "https://www.home-assistant.io/integrations/http/#ssl_certificate",
                        "base_url": self._base_url,
                        "network_link": "https://my.home-assistant.io/redirect/network/",
                        "nginx_link": "https://github.com/home-assistant/addons/tree/master/nginx_proxy",
                    },
                )
            else:
                ir.async_delete_issue(self._hass, DOMAIN, "ssl")
        else:
            ir.async_delete_issue(self._hass, DOMAIN, "webhook_url")
            ir.async_delete_issue(self._hass, DOMAIN, "https_webhook")
            ir.async_delete_issue(self._hass, DOMAIN, "ssl")

        # If no ONVIF push or long polling state is received, start fast polling
        await self._async_poll_all_motion()

        self._cancel_long_poll_check = None

    async def update_states(self) -> None:
        """Call the API of the camera device to update the internal states."""
        wake: bool = False
        if time() - self.last_wake > BATTERY_WAKE_UPDATE_INTERVAL:
            # wake the battery cameras for a complete update
            wake = True
            self.last_wake = time()

        if self._api.baichuan.privacy_mode():
            await self._api.baichuan