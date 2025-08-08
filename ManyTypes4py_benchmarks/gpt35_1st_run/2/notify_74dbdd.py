from __future__ import annotations
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import email.utils
import logging
import os
from pathlib import Path
import smtplib
import voluptuous as vol
from homeassistant.components.notify import ATTR_DATA, ATTR_TARGET, ATTR_TITLE, ATTR_TITLE_DEFAULT, PLATFORM_SCHEMA as NOTIFY_PLATFORM_SCHEMA, BaseNotificationService
from homeassistant.const import CONF_PASSWORD, CONF_PORT, CONF_RECIPIENT, CONF_SENDER, CONF_TIMEOUT, CONF_USERNAME, CONF_VERIFY_SSL, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.reload import setup_reload_service
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import dt as dt_util
from homeassistant.util.ssl import client_context
from .const import ATTR_HTML, ATTR_IMAGES, CONF_DEBUG, CONF_ENCRYPTION, CONF_SENDER_NAME, CONF_SERVER, DEFAULT_DEBUG, DEFAULT_ENCRYPTION, DEFAULT_HOST, DEFAULT_PORT, DEFAULT_TIMEOUT, DOMAIN, ENCRYPTION_OPTIONS

PLATFORM_SCHEMA: vol.Schema = NOTIFY_PLATFORM_SCHEMA.extend({vol.Required(CONF_RECIPIENT): vol.All(cv.ensure_list, [vol.Email()]), vol.Required(CONF_SENDER): vol.Email(), vol.Optional(CONF_SERVER, default=DEFAULT_HOST): cv.string, vol.Optional(CONF_PORT, default=DEFAULT_PORT): cv.port, vol.Optional(CONF_TIMEOUT, default=DEFAULT_TIMEOUT): cv.positive_int, vol.Optional(CONF_ENCRYPTION, default=DEFAULT_ENCRYPTION): vol.In(ENCRYPTION_OPTIONS), vol.Optional(CONF_USERNAME): cv.string, vol.Optional(CONF_PASSWORD): cv.string, vol.Optional(CONF_SENDER_NAME): cv.string, vol.Optional(CONF_DEBUG, default=DEFAULT_DEBUG): cv.boolean, vol.Optional(CONF_VERIFY_SSL, default=True): cv.boolean}

def get_service(hass: HomeAssistant, config: ConfigType, discovery_info: DiscoveryInfoType = None) -> MailNotificationService:
    ...

class MailNotificationService(BaseNotificationService):
    def __init__(self, server: str, port: int, timeout: int, sender: str, encryption: str, username: str, password: str, recipients: List[str], sender_name: str, debug: bool, verify_ssl: bool) -> None:
        ...

    def connect(self) -> smtplib.SMTP:
        ...

    def connection_is_valid(self) -> bool:
        ...

    def send_message(self, message: str = '', **kwargs: Any) -> bool:
        ...

    def _send_email(self, msg: MIMEMultipart, recipients: Union[str, List[str]]) -> None:
        ...

def _build_text_msg(message: str) -> MIMEText:
    ...

def _attach_file(hass: HomeAssistant, atch_name: str, content_id: str = '') -> Optional[MIMEImage]:
    ...

def _build_multipart_msg(hass: HomeAssistant, message: str, images: List[str]) -> MIMEMultipart:
    ...

def _build_html_msg(hass: HomeAssistant, text: str, html: str, images: List[str]) -> MIMEMultipart:
    ...
