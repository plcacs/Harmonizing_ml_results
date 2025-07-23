"""Pushsafer platform for notify component."""
from __future__ import annotations
import base64
from http import HTTPStatus
import logging
import mimetypes
from typing import Any, Callable, Dict, List, Optional, Union
import requests
from requests.auth import HTTPBasicAuth
import voluptuous as vol
from homeassistant.components.notify import (
    ATTR_DATA,
    ATTR_TARGET,
    ATTR_TITLE,
    ATTR_TITLE_DEFAULT,
    PLATFORM_SCHEMA as NOTIFY_PLATFORM_SCHEMA,
    BaseNotificationService,
)
from homeassistant.const import ATTR_ICON
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

_LOGGER: logging.Logger = logging.getLogger(__name__)
_RESOURCE: str = 'https://www.pushsafer.com/api'
_ALLOWED_IMAGES: List[str] = ['image/gif', 'image/jpeg', 'image/png']
CONF_DEVICE_KEY: str = 'private_key'
CONF_TIMEOUT: int = 15
ATTR_SOUND: str = 'sound'
ATTR_VIBRATION: str = 'vibration'
ATTR_ICONCOLOR: str = 'iconcolor'
ATTR_URL: str = 'url'
ATTR_URLTITLE: str = 'urltitle'
ATTR_TIME2LIVE: str = 'time2live'
ATTR_PRIORITY: str = 'priority'
ATTR_RETRY: str = 'retry'
ATTR_EXPIRE: str = 'expire'
ATTR_CONFIRM: str = 'confirm'
ATTR_ANSWER: str = 'answer'
ATTR_ANSWEROPTIONS: str = 'answeroptions'
ATTR_ANSWERFORCE: str = 'answerforce'
ATTR_PICTURE1: str = 'picture1'
ATTR_PICTURE1_URL: str = 'url'
ATTR_PICTURE1_PATH: str = 'path'
ATTR_PICTURE1_USERNAME: str = 'username'
ATTR_PICTURE1_PASSWORD: str = 'password'
ATTR_PICTURE1_AUTH: str = 'auth'

PLATFORM_SCHEMA = NOTIFY_PLATFORM_SCHEMA.extend({vol.Required(CONF_DEVICE_KEY): cv.string})


def get_service(
    hass: HomeAssistant, config: ConfigType, discovery_info: Optional[DiscoveryInfoType] = None
) -> PushsaferNotificationService:
    """Get the Pushsafer.com notification service."""
    return PushsaferNotificationService(config.get(CONF_DEVICE_KEY), hass.config.is_allowed_path)


class PushsaferNotificationService(BaseNotificationService):
    """Implementation of the notification service for Pushsafer.com."""

    def __init__(self, private_key: str, is_allowed_path: Callable[[str], bool]) -> None:
        """Initialize the service."""
        self._private_key: str = private_key
        self.is_allowed_path: Callable[[str], bool] = is_allowed_path

    def send_message(
        self, message: str = '', **kwargs: Any
    ) -> None:
        """Send a message to specified target."""
        if kwargs.get(ATTR_TARGET) is None:
            targets: List[str] = ['a']
            _LOGGER.debug('No target specified. Sending push to all')
        else:
            targets = kwargs.get(ATTR_TARGET)
            _LOGGER.debug('%s target(s) specified', len(targets))
        title: str = kwargs.get(ATTR_TITLE, ATTR_TITLE_DEFAULT)
        data: Dict[str, Any] = kwargs.get(ATTR_DATA, {})
        picture1: Optional[Dict[str, Any]] = data.get(ATTR_PICTURE1)
        picture1_encoded: str = ''
        if picture1 is not None:
            _LOGGER.debug('picture1 is available')
            url: Optional[str] = picture1.get(ATTR_PICTURE1_URL)
            local_path: Optional[str] = picture1.get(ATTR_PICTURE1_PATH)
            username: Optional[str] = picture1.get(ATTR_PICTURE1_USERNAME)
            password: Optional[str] = picture1.get(ATTR_PICTURE1_PASSWORD)
            auth: Optional[str] = picture1.get(ATTR_PICTURE1_AUTH)
            if url is not None:
                _LOGGER.debug('Loading image from url %s', url)
                picture1_encoded = self.load_from_url(url, username, password, auth)
            elif local_path is not None:
                _LOGGER.debug('Loading image from file %s', local_path)
                picture1_encoded = self.load_from_file(local_path)
            else:
                _LOGGER.warning('Missing url or local_path for picture1')
        else:
            _LOGGER.debug('picture1 is not specified')
        payload: Dict[str, Union[str, int]] = {
            'k': self._private_key,
            't': title,
            'm': message,
            's': data.get(ATTR_SOUND, ''),
            'v': data.get(ATTR_VIBRATION, ''),
            'i': data.get(ATTR_ICON, ''),
            'c': data.get(ATTR_ICONCOLOR, ''),
            'u': data.get(ATTR_URL, ''),
            'ut': data.get(ATTR_URLTITLE, ''),
            'l': data.get(ATTR_TIME2LIVE, ''),
            'pr': data.get(ATTR_PRIORITY, ''),
            're': data.get(ATTR_RETRY, ''),
            'ex': data.get(ATTR_EXPIRE, ''),
            'cr': data.get(ATTR_CONFIRM, ''),
            'a': data.get(ATTR_ANSWER, ''),
            'ao': data.get(ATTR_ANSWEROPTIONS, ''),
            'af': data.get(ATTR_ANSWERFORCE, ''),
            'p': picture1_encoded,
        }
        for target in targets:
            payload['d'] = target
            try:
                response: requests.Response = requests.post(
                    _RESOURCE, data=payload, timeout=CONF_TIMEOUT
                )
                if response.status_code != HTTPStatus.OK:
                    _LOGGER.error('Pushsafer failed with: %s', response.text)
                else:
                    _LOGGER.debug('Push send: %s', response.json())
            except requests.RequestException as error:
                _LOGGER.error("Error sending Pushsafer notification: %s", error)

    @classmethod
    def get_base64(cls, filebyte: bytes, mimetype: str) -> Optional[str]:
        """Convert the image to the expected base64 string of pushsafer."""
        if mimetype not in _ALLOWED_IMAGES:
            _LOGGER.warning('%s is a not supported mimetype for images', mimetype)
            return None
        base64_image: str = base64.b64encode(filebyte).decode('utf8')
        return f'data:{mimetype};base64,{base64_image}'

    def load_from_url(
        self,
        url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        auth: Optional[str] = None
    ) -> Optional[str]:
        """Load image/document/etc from URL."""
        if url is not None:
            _LOGGER.debug('Downloading image from %s', url)
            try:
                if username is not None and password is not None:
                    auth_ = HTTPBasicAuth(username, password)
                    response: requests.Response = requests.get(url, auth=auth_, timeout=CONF_TIMEOUT)
                else:
                    response = requests.get(url, timeout=CONF_TIMEOUT)
                return self.get_base64(response.content, response.headers.get('content-type', ''))
            except requests.RequestException as error:
                _LOGGER.error("Error loading from URL: %s", error)
                return None
        _LOGGER.warning('No url was found in param')
        return None

    def load_from_file(self, local_path: Optional[str] = None) -> Optional[str]:
        """Load image/document/etc from a local path."""
        try:
            if local_path is not None:
                _LOGGER.debug('Loading image from local path')
                if self.is_allowed_path(local_path):
                    file_mimetype: Optional[str] = mimetypes.guess_type(local_path)[0]
                    _LOGGER.debug('Detected mimetype %s', file_mimetype)
                    if file_mimetype is not None:
                        with open(local_path, 'rb') as binary_file:
                            data: bytes = binary_file.read()
                        return self.get_base64(data, file_mimetype)
                    else:
                        _LOGGER.warning('Could not determine mimetype for %s', local_path)
            else:
                _LOGGER.warning('Local path not found in params!')
        except OSError as error:
            _LOGGER.error("Can't load from local path: %s", error)
        return None
