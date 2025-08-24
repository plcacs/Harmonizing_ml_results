"""Support for the Mailgun mail notifications."""
from __future__ import annotations

import logging
from typing import Any, Mapping, Optional, Sequence, cast

from pymailgunner import Client, MailgunCredentialsError, MailgunDomainError, MailgunError
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

_LOGGER: logging.Logger = logging.getLogger(__name__)

ATTR_IMAGES: str = 'images'
DEFAULT_SANDBOX: bool = False

PLATFORM_SCHEMA: vol.Schema = NOTIFY_PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_RECIPIENT): vol.Email(),
        vol.Optional(CONF_SENDER): vol.Email(),
    }
)


def get_service(
    hass: HomeAssistant,
    config: ConfigType,
    discovery_info: DiscoveryInfoType | None = None,
) -> MailgunNotificationService | None:
    """Get the Mailgun notification service."""
    data: Mapping[str, Any] = cast(Mapping[str, Any], hass.data[MAILGUN_DOMAIN])
    mailgun_service = MailgunNotificationService(
        domain=cast(str, data.get(CONF_DOMAIN)),
        sandbox=cast(bool, data.get(CONF_SANDBOX, DEFAULT_SANDBOX)),
        api_key=cast(str, data.get(CONF_API_KEY)),
        sender=cast(Optional[str], config.get(CONF_SENDER)),
        recipient=cast(str, config.get(CONF_RECIPIENT)),
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
        sender: Optional[str],
        recipient: str,
    ) -> None:
        """Initialize the service."""
        self._client: Client | None = None
        self._domain: str = domain
        self._sandbox: bool = sandbox
        self._api_key: str = api_key
        self._sender: Optional[str] = sender
        self._recipient: str = recipient

    def initialize_client(self) -> None:
        """Initialize the connection to Mailgun."""
        self._client = Client(self._api_key, self._domain, self._sandbox)
        _LOGGER.debug('Mailgun domain: %s', self._client.domain)
        self._domain = self._client.domain
        if not self._sender:
            self._sender = f'hass@{self._domain}'

    def connection_is_valid(self) -> bool:
        """Check whether the provided credentials are valid."""
        try:
            self.initialize_client()
        except MailgunCredentialsError:
            _LOGGER.exception('Invalid credentials')
            return False
        except MailgunDomainError:
            _LOGGER.exception('Unexpected exception')
            return False
        return True

    def send_message(self, message: str = '', **kwargs: Any) -> None:
        """Send a mail to the recipient."""
        subject: str = kwargs.get(ATTR_TITLE, ATTR_TITLE_DEFAULT)
        data: Optional[dict[str, Any]] = kwargs.get(ATTR_DATA)
        files: Optional[Sequence[str]] = data.get(ATTR_IMAGES) if data else None
        try:
            if self._client is None:
                self.initialize_client()
            resp: Any = self._client.send_mail(
                sender=cast(str, self._sender),
                to=self._recipient,
                subject=subject,
                text=message,
                files=files,
            )
            _LOGGER.debug('Message sent: %s', resp)
        except MailgunError:
            _LOGGER.exception('Failed to send message')