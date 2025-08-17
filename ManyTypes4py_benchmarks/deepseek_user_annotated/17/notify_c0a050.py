"""Support for the Mailgun mail notifications."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from pymailgunner import (
    Client,
    MailgunCredentialsError,
    MailgunDomainError,
    MailgunError,
)
import voluptuous as vol

from homeassistant.components.notify import (
    ATTR_DATA,
    ATTR_TITLE,
    ATTR_TITLE_DEFAULT,
    PLATFORM_SCHEMA as NOTIFY_PLATFORM_SCHEMA,
    BaseNotificationService,
)
from homeassistant.const import CONF_API_KEY, CONF_DOMAIN, CONF_RECIPIENT, CONF_SENDER
from homeassistant.core import HomeAssistant
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

from . import CONF_SANDBOX, DOMAIN as MAILGUN_DOMAIN

_LOGGER = logging.getLogger(__name__)

# Images to attach to notification
ATTR_IMAGES = "images"

DEFAULT_SANDBOX = False

PLATFORM_SCHEMA = NOTIFY_PLATFORM_SCHEMA.extend(
    {vol.Required(CONF_RECIPIENT): vol.Email(), vol.Optional(CONF_SENDER): vol.Email()}
)


def get_service(
    hass: HomeAssistant,
    config: ConfigType,
    discovery_info: DiscoveryInfoType | None = None,
) -> MailgunNotificationService | None:
    """Get the Mailgun notification service."""
    data = hass.data[MAILGUN_DOMAIN]
    mailgun_service = MailgunNotificationService(
        data.get(CONF_DOMAIN),
        data.get(CONF_SANDBOX),
        data.get(CONF_API_KEY),
        config.get(CONF_SENDER),
        config.get(CONF_RECIPIENT),
    )
    if mailgun_service.connection_is_valid():
        return mailgun_service

    return None


class MailgunNotificationService(BaseNotificationService):
    """Implement a notification service for the Mailgun mail service."""

    def __init__(
        self,
        domain: str,
        sandbox: bool,
        api_key: str,
        sender: str | None,
        recipient: str,
    ) -> None:
        """Initialize the service."""
        self._client: Optional[Client] = None  # Mailgun API client
        self._domain: str = domain
        self._sandbox: bool = sandbox
        self._api_key: str = api_key
        self._sender: Optional[str] = sender
        self._recipient: str = recipient

    def initialize_client(self) -> None:
        """Initialize the connection to Mailgun."""

        self._client = Client(self._api_key, self._domain, self._sandbox)
        _LOGGER.debug("Mailgun domain: %s", self._client.domain)
        self._domain = self._client.domain
        if not self._sender:
            self._sender = f"hass@{self._domain}"

    def connection_is_valid(self) -> bool:
        """Check whether the provided credentials are valid."""

        try:
            self.initialize_client()
        except MailgunCredentialsError:
            _LOGGER.exception("Invalid credentials")
            return False
        except MailgunDomainError:
            _LOGGER.exception("Unexpected exception")
            return False
        return True

    def send_message(self, message: str = "", **kwargs: Any) -> None:
        """Send a mail to the recipient."""

        subject: str = kwargs.get(ATTR_TITLE, ATTR_TITLE_DEFAULT)
        data: Optional[Dict[str, Any]] = kwargs.get(ATTR_DATA)
        files: Optional[Any] = data.get(ATTR_IMAGES) if data else None

        try:
            # Initialize the client in case it was not.
            if self._client is None:
                self.initialize_client()
            resp: Any = self._client.send_mail(
                sender=self._sender,
                to=self._recipient,
                subject=subject,
                text=message,
                files=files,
            )
            _LOGGER.debug("Message sent: %s", resp)
        except MailgunError:
            _LOGGER.exception("Failed to send message")
