"""Jabber (XMPP) notification service."""
from __future__ import annotations
from concurrent.futures import TimeoutError as FutTimeoutError
from http import HTTPStatus
import logging
import mimetypes
import pathlib
import random
import string
from typing import Any, Dict, List, Optional, Union, cast
import requests
import slixmpp
from slixmpp.exceptions import IqError, IqTimeout, XMPPError
from slixmpp.plugins.xep_0363.http_upload import FileTooBig, FileUploadError, UploadServiceNotFound
from slixmpp.xmlstream.xmlstream import NotConnectedError
import voluptuous as vol
from homeassistant.components.notify import ATTR_TITLE, ATTR_TITLE_DEFAULT, PLATFORM_SCHEMA as NOTIFY_PLATFORM_SCHEMA, BaseNotificationService
from homeassistant.const import CONF_PASSWORD, CONF_RECIPIENT, CONF_RESOURCE, CONF_ROOM, CONF_SENDER
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv, template as template_helper
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
_LOGGER = logging.getLogger(__name__)
ATTR_DATA = 'data'
ATTR_PATH = 'path'
ATTR_PATH_TEMPLATE = 'path_template'
ATTR_TIMEOUT = 'timeout'
ATTR_URL = 'url'
ATTR_URL_TEMPLATE = 'url_template'
ATTR_VERIFY = 'verify'
CONF_TLS = 'tls'
CONF_VERIFY = 'verify'
DEFAULT_CONTENT_TYPE = 'application/octet-stream'
DEFAULT_RESOURCE = 'home-assistant'
XEP_0363_TIMEOUT = 10
PLATFORM_SCHEMA = NOTIFY_PLATFORM_SCHEMA.extend({vol.Required(CONF_SENDER): cv.string, vol.Required(CONF_PASSWORD): cv.string, vol.Required(CONF_RECIPIENT): vol.All(cv.ensure_list, [cv.string]), vol.Optional(CONF_RESOURCE, default=DEFAULT_RESOURCE): cv.string, vol.Optional(CONF_ROOM, default=''): cv.string, vol.Optional(CONF_TLS, default=True): cv.boolean, vol.Optional(CONF_VERIFY, default=True): cv.boolean})

async def async_get_service(hass: HomeAssistant, config: ConfigType, discovery_info: Optional[DiscoveryInfoType] = None) -> XmppNotificationService:
    """Get the Jabber (XMPP) notification service."""
    return XmppNotificationService(config.get(CONF_SENDER), config.get(CONF_RESOURCE), config.get(CONF_PASSWORD), config.get(CONF_RECIPIENT), config.get(CONF_TLS), config.get(CONF_VERIFY), config.get(CONF_ROOM), hass)

class XmppNotificationService(BaseNotificationService):
    """Implement the notification service for Jabber (XMPP)."""

    def __init__(self, sender: str, resource: str, password: str, recipient: List[str], tls: bool, verify: bool, room: str, hass: HomeAssistant) -> None:
        """Initialize the service."""
        self._hass = hass
        self._sender = sender
        self._resource = resource
        self._password = password
        self._recipients = recipient
        self._tls = tls
        self._verify = verify
        self._room = room

    async def async_send_message(self, message: str = '', **kwargs: Any) -> None:
        """Send a message to a user."""
        title = kwargs.get(ATTR_TITLE, ATTR_TITLE_DEFAULT)
        text = f'{title}: {message}' if title else message
        data = kwargs.get(ATTR_DATA)
        timeout = data.get(ATTR_TIMEOUT, XEP_0363_TIMEOUT) if data else None
        await async_send_message(f'{self._sender}/{self._resource}', self._password, self._recipients, self._tls, self._verify, self._room, self._hass, text, timeout, data)

async def async_send_message(sender: str, password: str, recipients: List[str], use_tls: bool, verify_certificate: bool, room: str, hass: HomeAssistant, message: str, timeout: Optional[int] = None, data: Optional[Dict[str, Any]] = None) -> None:
    """Send a message over XMPP."""

    class SendNotificationBot(slixmpp.ClientXMPP):
        """Service for sending Jabber (XMPP) messages."""

        def __init__(self) -> None:
            """Initialize the Jabber Bot."""
            super().__init__(sender, password)
            self.loop = hass.loop
            self.force_starttls = use_tls
            self.use_ipv6 = False
            self.add_event_handler('failed_all_auth', self.disconnect_on_login_fail)
            self.add_event_handler('session_start', self.start)
            if room:
                self.register_plugin('xep_0045')
            if not verify_certificate:
                self.add_event_handler('ssl_invalid_cert', self.discard_ssl_invalid_cert)
            if data:
                self.register_plugin('xep_0030')
                self.register_plugin('xep_0066')
                self.register_plugin('xep_0071')
                self.register_plugin('xep_0128')
                self.register_plugin('xep_0363')
            self.connect(force_starttls=self.force_starttls, use_ssl=False)

        async def start(self, event: Any) -> None:
            """Start the communication and sends the message."""
            if room:
                _LOGGER.debug('Joining room %s', room)
                await self.plugin['xep_0045'].join_muc_wait(room, sender, seconds=0)
            if data:
                await self.send_file(timeout=timeout)
            if message:
                self.send_text_message()
            self.disconnect()

        async def send_file(self, timeout: Optional[int] = None) -> None:
            """Send file via XMPP.

            Send XMPP file message using OOB (XEP_0066) and
            HTTP Upload (XEP_0363)
            """
            try:
                _LOGGER.debug('Timeout set to %ss', timeout)
                url = await self.upload_file(timeout=timeout)
                _LOGGER.debug('Upload success')
                for recipient in recipients:
                    if room:
                        _LOGGER.debug('Sending file to %s', room)
                        message = self.Message(sto=room, stype='groupchat')
                    else:
                        _LOGGER.debug('Sending file to %s', recipient)
                        message = self.Message(sto=recipient, stype='chat')
                    message['body'] = url
                    message['oob']['url'] = url
                    try:
                        message.send()
                    except (IqError, IqTimeout, XMPPError) as ex:
                        _LOGGER.error('Could not send image message %s', ex)
                    if room:
                        break
            except (IqError, IqTimeout, XMPPError) as ex:
                _LOGGER.error('Upload error, could not send message %s', ex)
            except NotConnectedError as ex:
                _LOGGER.error('Connection error %s', ex)
            except FileTooBig as ex:
                _LOGGER.error('File too big for server, could not upload file %s', ex)
            except UploadServiceNotFound as ex:
                _LOGGER.error('UploadServiceNotFound, could not upload file %s', ex)
            except FileUploadError as ex:
                _LOGGER.error('FileUploadError, could not upload file %s', ex)
            except requests.exceptions.SSLError as ex:
                _LOGGER.error('Cannot establish SSL connection %s', ex)
            except requests.exceptions.ConnectionError as ex:
                _LOGGER.error('Cannot connect to server %s', ex)
            except (FileNotFoundError, PermissionError, IsADirectoryError, TimeoutError) as ex:
                _LOGGER.error('Error reading file %s', ex)
            except FutTimeoutError as ex:
                _LOGGER.error('The server did not respond in time, %s', ex)

        async def upload_file(self, timeout: Optional[int] = None) -> str:
            """Upload file to Jabber server and return new URL.

            upload a file with Jabber XEP_0363 from a remote URL or a local
            file path and return a URL of that file.
            """
            if data is None:
                raise FileUploadError('Could not upload file')
                
            url: Optional[str] = None
            if data.get(ATTR_URL_TEMPLATE):
                _LOGGER.debug('Got url template: %s', data[ATTR_URL_TEMPLATE])
                templ = template_helper.Template(data[ATTR_URL_TEMPLATE], hass)
                get_url = template_helper.render_complex(templ, None)
                url = await self.upload_file_from_url(get_url, timeout=timeout)
            elif data.get(ATTR_URL):
                url = await self.upload_file_from_url(data[ATTR_URL], timeout=timeout)
            elif data.get(ATTR_PATH_TEMPLATE):
                _LOGGER.debug('Got path template: %s', data[ATTR_PATH_TEMPLATE])
                templ = template_helper.Template(data[ATTR_PATH_TEMPLATE], hass)
                get_path = template_helper.render_complex(templ, None)
                url = await self.upload_file_from_path(get_path, timeout=timeout)
            elif data.get(ATTR_PATH):
                url = await self.upload_file_from_path(data[ATTR_PATH], timeout=timeout)
            
            if url is None:
                _LOGGER.error('No path or URL found for file')
                raise FileUploadError('Could not upload file')
            return url

        async def upload_file_from_url(self, url: str, timeout: Optional[int] = None) -> Optional[str]:
            """Upload a file from a URL. Returns a URL.

            uploaded via XEP_0363 and HTTP and returns the resulting URL
            """
            _LOGGER.debug('Getting file from %s', url)

            def get_url(url: str) -> requests.Response:
                """Return result for GET request to url."""
                if data is None:
                    verify = True
                else:
                    verify = data.get(ATTR_VERIFY, True)
                return requests.get(url, verify=verify, timeout=timeout)
                
            result = await hass.async_add_executor_job(get_url, url)
            if result.status_code >= HTTPStatus.BAD_REQUEST:
                _LOGGER.error('Could not load file from %s', url)
                return None
            filesize = len(result.content)
            if data and data.get(ATTR_PATH):
                filename = self.get_random_filename(data.get(ATTR_PATH))
            else:
                extension = mimetypes.guess_extension(result.headers['Content-Type']) or '.unknown'
                _LOGGER.debug('Got %s extension', extension)
                filename = self.get_random_filename(None, extension=extension)
            _LOGGER.debug('Uploading file from URL, %s', filename)
            return await self['xep_0363'].upload_file(filename, size=filesize, input_file=result.content, content_type=result.headers['Content-Type'], timeout=timeout)

        def _read_upload_file(self, path: str) -> bytes:
            """Read file from path."""
            with open(path, 'rb') as upfile:
                _LOGGER.debug('Reading file %s', path)
                return upfile.read()

        async def upload_file_from_path(self, path: str, timeout: Optional[int] = None) -> str:
            """Upload a file from a local file path via XEP_0363."""
            _LOGGER.debug('Uploading file from path, %s', path)
            if not hass.config.is_allowed_path(path):
                raise PermissionError('Could not access file. Path not allowed')
            input_file = await hass.async_add_executor_job(self._read_upload_file, path)
            filesize = len(input_file)
            _LOGGER.debug('Filesize is %s bytes', filesize)
            content_type = mimetypes.guess_type(path)[0]
            if content_type is None:
                content_type = DEFAULT_CONTENT_TYPE
            _LOGGER.debug('Content type is %s', content_type)
            if data is None:
                attr_path = None
            else:
                attr_path = data.get(ATTR_PATH)
            filename = self.get_random_filename(attr_path)
            _LOGGER.debug('Uploading file with random filename %s', filename)
            return await self['xep_0363'].upload_file(filename, size=filesize, input_file=input_file, content_type=content_type, timeout=timeout)

        def send_text_message(self) -> None:
            """Send a text only message to a room or a recipient."""
            try:
                if room:
                    _LOGGER.debug('Sending message to room %s', room)
                    self.send_message(mto=room, mbody=message, mtype='groupchat')
                else:
                    for recipient in recipients:
                        _LOGGER.debug('Sending message to %s', recipient)
                        self.send_message(mto=recipient, mbody=message, mtype='chat')
            except (IqError, IqTimeout, XMPPError) as ex:
                _LOGGER.error('Could not send text message %s', ex)
            except NotConnectedError as ex:
                _LOGGER.error('Connection error %s', ex)

        def get_random_filename(self, filename: Optional[str], extension: Optional[str] = None) -> str:
            """Return a random filename, leaving the extension intact."""
            if extension is None:
                if filename is None:
                    extension = '.txt'
                else:
                    path = pathlib.Path(filename)
                    if path.suffix:
                        extension = ''.join(path.suffixes)
                    else:
                        extension = '.txt'
            return ''.join((random.choice(string.ascii_letters) for i in range(10))) + extension

        def disconnect_on_login_fail(self, event: Any) -> None:
            """Disconnect from the server if credentials are invalid."""
            _LOGGER.warning('Login failed')
            self.disconnect()

        @staticmethod
        def discard_ssl_invalid_cert(event: Any) -> None:
            """Do nothing if ssl certificate is invalid."""
            _LOGGER.debug('Ignoring invalid SSL certificate as requested')
    SendNotificationBot()
