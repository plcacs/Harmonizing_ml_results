"""Twitter platform for notify component."""
from __future__ import annotations
from datetime import datetime, timedelta
from functools import partial
from http import HTTPStatus
import json
import logging
import mimetypes
import os
from typing import Any, Callable, Optional, Tuple

from TwitterAPI import TwitterAPI
import voluptuous as vol
from homeassistant.components.notify import (
    ATTR_DATA,
    ATTR_TARGET,
    PLATFORM_SCHEMA as NOTIFY_PLATFORM_SCHEMA,
    BaseNotificationService,
)
from homeassistant.const import CONF_ACCESS_TOKEN, CONF_USERNAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.event import async_track_point_in_time
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

_LOGGER = logging.getLogger(__name__)

CONF_CONSUMER_KEY = 'consumer_key'
CONF_CONSUMER_SECRET = 'consumer_secret'
CONF_ACCESS_TOKEN_SECRET = 'access_token_secret'
ATTR_MEDIA = 'media'

PLATFORM_SCHEMA = NOTIFY_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_ACCESS_TOKEN): cv.string,
    vol.Required(CONF_ACCESS_TOKEN_SECRET): cv.string,
    vol.Required(CONF_CONSUMER_KEY): cv.string,
    vol.Required(CONF_CONSUMER_SECRET): cv.string,
    vol.Optional(CONF_USERNAME): cv.string
})


def func_g5i7632v(
    hass: HomeAssistant,
    config: ConfigType,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> TwitterNotificationService:
    """Get the Twitter notification service."""
    return TwitterNotificationService(
        hass,
        config[CONF_CONSUMER_KEY],
        config[CONF_CONSUMER_SECRET],
        config[CONF_ACCESS_TOKEN],
        config[CONF_ACCESS_TOKEN_SECRET],
        config.get(CONF_USERNAME)
    )


class TwitterNotificationService(BaseNotificationService):
    """Implementation of a notification service for the Twitter service."""

    def __init__(
        self,
        hass: HomeAssistant,
        consumer_key: str,
        consumer_secret: str,
        access_token_key: str,
        access_token_secret: str,
        username: Optional[str]
    ) -> None:
        """Initialize the service."""
        self.default_user: Optional[str] = username
        self.hass: HomeAssistant = hass
        self.api: TwitterAPI = TwitterAPI(
            consumer_key, consumer_secret,
            access_token_key, access_token_secret
        )

    def func_qrgn0ojh(
        self,
        message: str = '',
        **kwargs: Any
    ) -> None:
        """Tweet a message, optionally with media."""
        data: Optional[dict] = kwargs.get(ATTR_DATA)
        targets: Optional[list] = kwargs.get(ATTR_TARGET)
        media: Optional[str] = None
        if data:
            media = data.get(ATTR_MEDIA)
            if media and not self.hass.config.is_allowed_path(media):
                _LOGGER.warning("'%s' is not a whitelisted directory", media)
                return
        if targets:
            for target in targets:
                callback = partial(self.func_z32nj807, message, target)
                self.func_yc9sfrdj(callback, media)
        else:
            callback = partial(self.func_z32nj807, message, self.default_user)
            self.func_yc9sfrdj(callback, media)

    def func_z32nj807(
        self,
        message: str,
        user: Optional[str],
        media_id: Optional[str] = None
    ) -> None:
        """Tweet a message, optionally with media."""
        if user:
            user_resp = self.api.request('users/lookup', {'screen_name': user})
            user_id: Optional[int] = None
            if user_resp.status_code == HTTPStatus.OK:
                user_data = user_resp.json()
                if isinstance(user_data, list) and len(user_data) > 0:
                    user_id = user_data[0].get('id')
                    _LOGGER.debug('Message posted: %s', user_data)
            else:
                self.func_lmn210o2(user_resp)
            if user_id:
                event = {
                    'event': {
                        'type': 'message_create',
                        'message_create': {
                            'target': {'recipient_id': user_id},
                            'message_data': {'text': message}
                        }
                    }
                }
                resp = self.api.request('direct_messages/events/new', json.dumps(event))
        else:
            resp = self.api.request(
                'statuses/update',
                {'status': message, 'media_ids': media_id}
            )
        if resp.status_code != HTTPStatus.OK:
            self.func_lmn210o2(resp)
        else:
            _LOGGER.debug('Message posted: %s', resp.json())

    def func_yc9sfrdj(
        self,
        callback: Callable[[Optional[str]], None],
        media_path: Optional[str] = None
    ) -> None:
        """Upload media."""
        if not media_path:
            callback()
            return
        if not self.hass.config.is_allowed_path(media_path):
            _LOGGER.warning("'%s' is not a whitelisted directory", media_path)
            return
        try:
            with open(media_path, 'rb') as file:
                total_bytes: int = os.path.getsize(media_path)
                media_category, media_type = self.func_d95fzab3(media_path)
                if not media_category or not media_type:
                    _LOGGER.error("Unsupported media type for %s", media_path)
                    return
                resp = self.func_cpdi92h9(media_type, media_category, total_bytes)
                if not (200 <= resp.status_code < 300):
                    self.func_lmn210o2(resp)
                    return
                media_id: str = str(resp.json()['media_id'])
                media_id = self.func_06htya0a(file, total_bytes, media_id)
                if not media_id:
                    return
                resp = self.func_xec9oh84(media_id)
                if not (200 <= resp.status_code < 300):
                    self.func_lmn210o2(resp)
                    return
                processing_info = resp.json().get('processing_info')
                if processing_info is None:
                    callback(media_id)
                    return
                self.check_status_until_done(media_id, callback)
        except OSError as e:
            _LOGGER.error("Failed to open media file %s: %s", media_path, e)

    def func_d95fzab3(self, media_path: str) -> Tuple[Optional[str], Optional[str]]:
        """Determine mime type and Twitter media category for given media."""
        media_type: Optional[str]
        media_category: Optional[str]
        media_type, _ = mimetypes.guess_type(media_path)
        media_category = self.func_03rf8tlb(media_type)
        _LOGGER.debug(
            'media %s is mime type %s and translates to %s',
            media_path, media_type, media_category
        )
        return media_category, media_type

    def func_cpdi92h9(
        self,
        media_type: Optional[str],
        media_category: Optional[str],
        total_bytes: int
    ) -> Any:
        """Upload media, INIT phase."""
        return self.api.request('media/upload', {
            'command': 'INIT',
            'media_type': media_type,
            'media_category': media_category,
            'total_bytes': total_bytes
        })

    def func_06htya0a(
        self,
        file: Any,
        total_bytes: int,
        media_id: str
    ) -> Optional[str]:
        """Upload media, chunked append."""
        segment_id: int = 0
        bytes_sent: int = 0
        while bytes_sent < total_bytes:
            chunk: bytes = file.read(4 * 1024 * 1024)
            resp = self.func_6qweoa2a(chunk, media_id, segment_id)
            if not (HTTPStatus.OK <= resp.status_code < HTTPStatus.MULTIPLE_CHOICES):
                self.func_tahlabm6(resp)
                return None
            segment_id += 1
            bytes_sent = file.tell()
            self.func_vj094qfe(bytes_sent, total_bytes)
        return media_id

    def func_6qweoa2a(
        self,
        chunk: bytes,
        media_id: str,
        segment_id: int
    ) -> Any:
        """Upload media, APPEND phase."""
        return self.api.request('media/upload', {
            'command': 'APPEND',
            'media_id': media_id,
            'segment_index': segment_id
        }, {'media': chunk})

    def func_xec9oh84(self, media_id: str) -> Any:
        """Upload media, FINALIZE phase."""
        return self.api.request('media/upload', {
            'command': 'FINALIZE',
            'media_id': media_id
        })

    def func_wuwdtx5n(
        self,
        media_id: str,
        callback: Callable[[Optional[str]], None],
        *args: Any
    ) -> None:
        """Upload media, STATUS phase."""
        resp = self.api.request('media/upload', {
            'command': 'STATUS',
            'media_id': media_id
        }, method_override='GET')
        if resp.status_code != HTTPStatus.OK:
            _LOGGER.error('Media processing error: %s', resp.json())
        processing_info = resp.json().get('processing_info', {})
        _LOGGER.debug('media processing %s status: %s', media_id, processing_info)
        state: Optional[str] = processing_info.get('state')
        if state in {'succeeded', 'failed'}:
            callback(media_id)
            return
        check_after_secs: Optional[int] = processing_info.get('check_after_secs')
        if check_after_secs is None:
            _LOGGER.error("Missing 'check_after_secs' in processing_info")
            return
        _LOGGER.debug(
            'media processing waiting %s seconds to check status',
            str(check_after_secs)
        )
        when: datetime = datetime.now() + timedelta(seconds=check_after_secs)
        myself = partial(self.func_wuwdtx5n, media_id, callback)
        async_track_point_in_time(self.hass, myself, when)

    @staticmethod
    def func_03rf8tlb(media_type: Optional[str]) -> Optional[str]:
        """Determine Twitter media category by mime type."""
        if media_type is None:
            return None
        if media_type.startswith('image/gif'):
            return 'tweet_gif'
        if media_type.startswith('video/'):
            return 'tweet_video'
        if media_type.startswith('image/'):
            return 'tweet_image'
        return None

    @staticmethod
    def func_vj094qfe(bytes_sent: int, total_bytes: int) -> None:
        """Log upload progress."""
        _LOGGER.debug('%s of %s bytes uploaded', str(bytes_sent), str(total_bytes))

    @staticmethod
    def func_lmn210o2(resp: Any) -> None:
        """Log error response."""
        try:
            obj = json.loads(resp.text)
        except json.JSONDecodeError:
            _LOGGER.error('Error %s: %s', resp.status_code, resp.text)
            return

        if 'errors' in obj:
            error_message = obj['errors']
        elif 'error' in obj:
            error_message = obj['error']
        else:
            error_message = resp.text
        _LOGGER.error('Error %s: %s', resp.status_code, error_message)

    @staticmethod
    def func_tahlabm6(resp: Any) -> None:
        """Log error response, during upload append phase."""
        try:
            obj = json.loads(resp.text)
            error_message = obj['errors'][0]['message']
            error_code = obj['errors'][0]['code']
            _LOGGER.error('Error %s: %s (Code %s)', resp.status_code, error_message, error_code)
        except (KeyError, IndexError, json.JSONDecodeError):
            _LOGGER.error('Unexpected error format: %s', resp.text)

    def check_status_until_done(
        self,
        media_id: str,
        callback: Callable[[Optional[str]], None]
    ) -> None:
        """Check media processing status recursively."""
        self.func_wuwdtx5n(media_id, callback)
