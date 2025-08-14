#!/usr/bin/env python3
"""Mail (SMTP) notification service."""

from __future__ import annotations

import email.utils
import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Union

import smtplib
from email.message import Message
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import voluptuous as vol

from homeassistant.components.notify import (
    ATTR_DATA,
    ATTR_TARGET,
    ATTR_TITLE,
    ATTR_TITLE_DEFAULT,
    PLATFORM_SCHEMA as NOTIFY_PLATFORM_SCHEMA,
    BaseNotificationService,
)
from homeassistant.const import (
    CONF_PASSWORD,
    CONF_PORT,
    CONF_RECIPIENT,
    CONF_SENDER,
    CONF_TIMEOUT,
    CONF_USERNAME,
    CONF_VERIFY_SSL,
    Platform,
)
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.reload import setup_reload_service
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import dt as dt_util
from homeassistant.util.ssl import client_context

from .const import (
    ATTR_HTML,
    ATTR_IMAGES,
    CONF_DEBUG,
    CONF_ENCRYPTION,
    CONF_SENDER_NAME,
    CONF_SERVER,
    DEFAULT_DEBUG,
    DEFAULT_ENCRYPTION,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_TIMEOUT,
    DOMAIN,
    ENCRYPTION_OPTIONS,
)

PLATFORMS = [Platform.NOTIFY]

_LOGGER = logging.getLogger(__name__)

PLATFORM_SCHEMA = NOTIFY_PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_RECIPIENT): vol.All(cv.ensure_list, [vol.Email()]),
        vol.Required(CONF_SENDER): vol.Email(),
        vol.Optional(CONF_SERVER, default=DEFAULT_HOST): cv.string,
        vol.Optional(CONF_PORT, default=DEFAULT_PORT): cv.port,
        vol.Optional(CONF_TIMEOUT, default=DEFAULT_TIMEOUT): cv.positive_int,
        vol.Optional(CONF_ENCRYPTION, default=DEFAULT_ENCRYPTION): vol.In(ENCRYPTION_OPTIONS),
        vol.Optional(CONF_USERNAME): cv.string,
        vol.Optional(CONF_PASSWORD): cv.string,
        vol.Optional(CONF_SENDER_NAME): cv.string,
        vol.Optional(CONF_DEBUG, default=DEFAULT_DEBUG): cv.boolean,
        vol.Optional(CONF_VERIFY_SSL, default=True): cv.boolean,
    }
)


def get_service(
    hass: HomeAssistant,
    config: ConfigType,
    discovery_info: DiscoveryInfoType | None = None,
) -> Optional[MailNotificationService]:
    """Get the mail notification service."""
    setup_reload_service(hass, DOMAIN, PLATFORMS)
    mail_service = MailNotificationService(
        server=config[CONF_SERVER],
        port=config[CONF_PORT],
        timeout=config[CONF_TIMEOUT],
        sender=config[CONF_SENDER],
        encryption=config[CONF_ENCRYPTION],
        username=config.get(CONF_USERNAME),
        password=config.get(CONF_PASSWORD),
        recipients=config[CONF_RECIPIENT],
        sender_name=config.get(CONF_SENDER_NAME),
        debug=config[CONF_DEBUG],
        verify_ssl=config[CONF_VERIFY_SSL],
    )

    if mail_service.connection_is_valid():
        return mail_service

    return None


class MailNotificationService(BaseNotificationService):
    """Implement the notification service for E-mail messages."""

    def __init__(
        self,
        server: str,
        port: int,
        timeout: int,
        sender: str,
        encryption: str,
        username: Optional[str],
        password: Optional[str],
        recipients: List[str],
        sender_name: Optional[str],
        debug: bool,
        verify_ssl: bool,
    ) -> None:
        """Initialize the SMTP service."""
        self._server: str = server
        self._port: int = port
        self._timeout: int = timeout
        self._sender: str = sender
        self.encryption: str = encryption
        self.username: Optional[str] = username
        self.password: Optional[str] = password
        self.recipients: List[str] = recipients
        self._sender_name: Optional[str] = sender_name
        self.debug: bool = debug
        self._verify_ssl: bool = verify_ssl
        self.tries: int = 2

    def connect(self) -> smtplib.SMTP:
        """Connect/authenticate to SMTP Server."""
        ssl_context = client_context() if self._verify_ssl else None
        if self.encryption == "tls":
            mail: smtplib.SMTP = smtplib.SMTP_SSL(
                self._server,
                self._port,
                timeout=self._timeout,
                context=ssl_context,
            )
        else:
            mail = smtplib.SMTP(self._server, self._port, timeout=self._timeout)
        mail.set_debuglevel(self.debug)
        mail.ehlo_or_helo_if_needed()
        if self.encryption == "starttls":
            mail.starttls(context=ssl_context)
            mail.ehlo()
        if self.username and self.password:
            mail.login(self.username, self.password)
        return mail

    def connection_is_valid(self) -> bool:
        """Check for valid config, verify connectivity."""
        server: Optional[smtplib.SMTP] = None
        try:
            server = self.connect()
        except (smtplib.socket.gaierror, ConnectionRefusedError):
            _LOGGER.exception(
                (
                    "SMTP server not found or refused connection (%s:%s). Please check"
                    " the IP address, hostname, and availability of your SMTP server"
                ),
                self._server,
                self._port,
            )

        except smtplib.SMTPAuthenticationError:
            _LOGGER.exception(
                "Login not possible. Please check your setting and/or your credentials"
            )
            return False

        finally:
            if server:
                server.quit()

        return True

    def send_message(self, message: str = "", **kwargs: Any) -> None:
        """Build and send a message to a user.

        Will send plain text normally, with pictures as attachments if images config is
        defined, or will build a multipart HTML if html config is defined.
        """
        subject: str = kwargs.get(ATTR_TITLE, ATTR_TITLE_DEFAULT)

        if data := kwargs.get(ATTR_DATA):
            if ATTR_HTML in data:
                msg: Message = _build_html_msg(
                    self.hass,
                    message,
                    data[ATTR_HTML],
                    images=data.get(ATTR_IMAGES, []),
                )
            else:
                msg = _build_multipart_msg(
                    self.hass, message, images=data.get(ATTR_IMAGES, [])
                )
        else:
            msg = _build_text_msg(message)

        msg["Subject"] = subject

        recipients: Union[str, List[str]] = kwargs.get(ATTR_TARGET)
        if not recipients:
            recipients = self.recipients
        msg["To"] = recipients if isinstance(recipients, str) else ",".join(recipients)
        if self._sender_name:
            msg["From"] = f"{self._sender_name} <{self._sender}>"
        else:
            msg["From"] = self._sender
        msg["X-Mailer"] = "Home Assistant"
        msg["Date"] = email.utils.format_datetime(dt_util.now())
        msg["Message-Id"] = email.utils.make_msgid()

        self._send_email(msg, recipients)

    def _send_email(self, msg: Message, recipients: Union[str, List[str]]) -> None:
        """Send the message."""
        mail: smtplib.SMTP = self.connect()
        for _ in range(self.tries):
            try:
                mail.sendmail(self._sender, recipients, msg.as_string())
                break
            except smtplib.SMTPServerDisconnected:
                _LOGGER.warning(
                    "SMTPServerDisconnected sending mail: retrying connection"
                )
                mail.quit()
                mail = self.connect()
            except smtplib.SMTPException:
                _LOGGER.warning("SMTPException sending mail: retrying connection")
                mail.quit()
                mail = self.connect()
        mail.quit()


def _build_text_msg(message: str) -> MIMEText:
    """Build plaintext email."""
    _LOGGER.debug("Building plain text email")
    return MIMEText(message)


def _attach_file(
    hass: HomeAssistant, atch_name: str, content_id: str = ""
) -> Optional[Message]:
    """Create a message attachment.

    If MIMEImage is successful and content_id is passed (HTML), add images in-line.
    Otherwise add them as attachments.
    """
    try:
        file_path: Path = Path(atch_name).parent
        if os.path.exists(file_path) and not hass.config.is_allowed_path(str(file_path)):
            allow_list: str = "allowlist_external_dirs"
            file_name: str = os.path.basename(atch_name)
            url: str = "https://www.home-assistant.io/docs/configuration/basic/"
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="remote_path_not_allowed",
                translation_placeholders={
                    "allow_list": allow_list,
                    "file_path": file_path,
                    "file_name": file_name,
                    "url": url,
                },
            )
        with open(atch_name, "rb") as attachment_file:
            file_bytes: bytes = attachment_file.read()
    except FileNotFoundError:
        _LOGGER.warning("Attachment %s not found. Skipping", atch_name)
        return None

    try:
        attachment: Message = MIMEImage(file_bytes)
    except TypeError:
        _LOGGER.warning(
            "Attachment %s has an unknown MIME type. Falling back to file",
            atch_name,
        )
        attachment = MIMEApplication(file_bytes, Name=os.path.basename(atch_name))
        attachment["Content-Disposition"] = (
            f'attachment; filename="{os.path.basename(atch_name)}"'
        )
    else:
        if content_id:
            attachment.add_header("Content-ID", f"<{content_id}>")
        else:
            attachment.add_header(
                "Content-Disposition",
                f"attachment; filename={os.path.basename(atch_name)}",
            )

    return attachment


def _build_multipart_msg(hass: HomeAssistant, message: str, images: List[str]) -> MIMEMultipart:
    """Build Multipart message with images as attachments."""
    _LOGGER.debug("Building multipart email with image attachments")
    msg: MIMEMultipart = MIMEMultipart()
    body_txt: MIMEText = MIMEText(message)
    msg.attach(body_txt)

    for atch_name in images:
        attachment: Optional[Message] = _attach_file(hass, atch_name)
        if attachment:
            msg.attach(attachment)

    return msg


def _build_html_msg(hass: HomeAssistant, text: str, html: str, images: List[str]) -> MIMEMultipart:
    """Build Multipart message with in-line images and rich HTML (UTF-8)."""
    _LOGGER.debug("Building HTML rich email")
    msg: MIMEMultipart = MIMEMultipart("related")
    alternative: MIMEMultipart = MIMEMultipart("alternative")
    alternative.attach(MIMEText(text, _charset="utf-8"))
    alternative.attach(MIMEText(html, ATTR_HTML, _charset="utf-8"))
    msg.attach(alternative)

    for atch_name in images:
        name: str = os.path.basename(atch_name)
        attachment: Optional[Message] = _attach_file(hass, atch_name, name)
        if attachment:
            msg.attach(attachment)
    return msg
