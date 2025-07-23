"""Provide functionality for TTS."""
from __future__ import annotations
import asyncio
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime
from functools import partial
import hashlib
from http import HTTPStatus
import io
import logging
import mimetypes
import os
import re
import secrets
import subprocess
import tempfile
from typing import Any, Final, TypedDict, final, Optional, Union, Dict, List, Set, Tuple, cast

from aiohttp import web
import mutagen
from mutagen.id3 import ID3, TextFrame as ID3Text
from propcache.api import cached_property
import voluptuous as vol
from homeassistant.components import ffmpeg, websocket_api
from homeassistant.components.http import HomeAssistantView
from homeassistant.components.media_player import ATTR_MEDIA_ANNOUNCE, ATTR_MEDIA_CONTENT_ID, ATTR_MEDIA_CONTENT_TYPE, DOMAIN as DOMAIN_MP, SERVICE_PLAY_MEDIA, MediaType
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_ENTITY_ID, PLATFORM_FORMAT, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HassJob, HomeAssistant, ServiceCall, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_component import EntityComponent
from homeassistant.helpers.event import async_call_later
from homeassistant.helpers.network import get_url
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.typing import UNDEFINED, ConfigType, UndefinedType
from homeassistant.util import dt as dt_util, language as language_util

from .const import ATTR_CACHE, ATTR_LANGUAGE, ATTR_MESSAGE, ATTR_OPTIONS, CONF_CACHE, CONF_CACHE_DIR, CONF_TIME_MEMORY, DATA_COMPONENT, DATA_TTS_MANAGER, DEFAULT_CACHE, DEFAULT_CACHE_DIR, DEFAULT_TIME_MEMORY, DOMAIN, TtsAudioType
from .helper import get_engine_instance
from .legacy import PLATFORM_SCHEMA, PLATFORM_SCHEMA_BASE, Provider, async_setup_legacy
from .media_source import generate_media_source_id, media_source_id_to_kwargs
from .models import Voice

__all__ = ['ATTR_AUDIO_OUTPUT', 'ATTR_PREFERRED_FORMAT', 'ATTR_PREFERRED_SAMPLE_BYTES', 'ATTR_PREFERRED_SAMPLE_CHANNELS', 'ATTR_PREFERRED_SAMPLE_RATE', 'CONF_LANG', 'DEFAULT_CACHE_DIR', 'PLATFORM_SCHEMA', 'PLATFORM_SCHEMA_BASE', 'Provider', 'SampleFormat', 'TtsAudioType', 'Voice', 'async_default_engine', 'async_get_media_source_audio', 'async_support_options', 'generate_media_source_id']

_LOGGER: Final = logging.getLogger(__name__)
ATTR_PLATFORM: Final = 'platform'
ATTR_AUDIO_OUTPUT: Final = 'audio_output'
ATTR_PREFERRED_FORMAT: Final = 'preferred_format'
ATTR_PREFERRED_SAMPLE_RATE: Final = 'preferred_sample_rate'
ATTR_PREFERRED_SAMPLE_CHANNELS: Final = 'preferred_sample_channels'
ATTR_PREFERRED_SAMPLE_BYTES: Final = 'preferred_sample_bytes'
ATTR_MEDIA_PLAYER_ENTITY_ID: Final = 'media_player_entity_id'
ATTR_VOICE: Final = 'voice'
_DEFAULT_FORMAT: Final = 'mp3'
_PREFFERED_FORMAT_OPTIONS: Final = {ATTR_PREFERRED_FORMAT, ATTR_PREFERRED_SAMPLE_RATE, ATTR_PREFERRED_SAMPLE_CHANNELS, ATTR_PREFERRED_SAMPLE_BYTES}
CONF_LANG: Final = 'language'
SERVICE_CLEAR_CACHE: Final = 'clear_cache'
_RE_LEGACY_VOICE_FILE: Final = re.compile('([a-f0-9]{40})_([^_]+)_([^_]+)_([a-z_]+)\\.[a-z0-9]{3,4}')
_RE_VOICE_FILE: Final = re.compile('([a-f0-9]{40})_([^_]+)_([^_]+)_(tts\\.[a-z0-9_]+)\\.[a-z0-9]{3,4}')
KEY_PATTERN: Final = '{0}_{1}_{2}_{3}'
SCHEMA_SERVICE_CLEAR_CACHE: Final = vol.Schema({})

class TTSCache(TypedDict):
    """Cached TTS file."""
    filename: str
    voice: bytes
    pending: Optional[asyncio.Task[Any]]

@callback
def async_default_engine(hass: HomeAssistant) -> Optional[str]:
    """Return the domain or entity id of the default engine."""
    default_entity_id: Optional[str] = None
    for entity in hass.data[DATA_COMPONENT].entities:
        if entity.platform and entity.platform.platform_name == 'cloud':
            return entity.entity_id
        if default_entity_id is None:
            default_entity_id = entity.entity_id
    return default_entity_id or next(iter(hass.data[DATA_TTS_MANAGER].providers), None)

@callback
def async_resolve_engine(hass: HomeAssistant, engine: Optional[str]) -> Optional[str]:
    """Resolve engine."""
    if engine is not None:
        if not hass.data[DATA_COMPONENT].get_entity(engine) and engine not in hass.data[DATA_TTS_MANAGER].providers:
            return None
        return engine
    return async_default_engine(hass)

async def async_support_options(hass: HomeAssistant, engine: str, language: Optional[str] = None, options: Optional[Dict[str, Any]] = None) -> bool:
    """Return if an engine supports options."""
    if (engine_instance := get_engine_instance(hass, engine)) is None:
        raise HomeAssistantError(f'Provider {engine} not found')
    try:
        hass.data[DATA_TTS_MANAGER].process_options(engine_instance, language, options)
    except HomeAssistantError:
        return False
    return True

async def async_get_media_source_audio(hass: HomeAssistant, media_source_id: str) -> TtsAudioType:
    """Get TTS audio as extension, data."""
    return await hass.data[DATA_TTS_MANAGER].async_get_tts_audio(**media_source_id_to_kwargs(media_source_id))

@callback
def async_get_text_to_speech_languages(hass: HomeAssistant) -> Set[str]:
    """Return a set with the union of languages supported by tts engines."""
    languages: Set[str] = set()
    for entity in hass.data[DATA_COMPONENT].entities:
        for language_tag in entity.supported_languages:
            languages.add(language_tag)
    for tts_engine in hass.data[DATA_TTS_MANAGER].providers.values():
        for language_tag in tts_engine.supported_languages:
            languages.add(language_tag)
    return languages

async def async_convert_audio(
    hass: HomeAssistant,
    from_extension: str,
    audio_bytes: bytes,
    to_extension: str,
    to_sample_rate: Optional[int] = None,
    to_sample_channels: Optional[int] = None,
    to_sample_bytes: Optional[int] = None
) -> bytes:
    """Convert audio to a preferred format using ffmpeg."""
    ffmpeg_manager = ffmpeg.get_ffmpeg_manager(hass)
    return await hass.async_add_executor_job(
        lambda: _convert_audio(
            ffmpeg_manager.binary,
            from_extension,
            audio_bytes,
            to_extension,
            to_sample_rate=to_sample_rate,
            to_sample_channels=to_sample_channels,
            to_sample_bytes=to_sample_bytes
        )
    )

def _convert_audio(
    ffmpeg_binary: str,
    from_extension: str,
    audio_bytes: bytes,
    to_extension: str,
    to_sample_rate: Optional[int] = None,
    to_sample_channels: Optional[int] = None,
    to_sample_bytes: Optional[int] = None
) -> bytes:
    """Convert audio to a preferred format using ffmpeg."""
    with tempfile.NamedTemporaryFile(mode='wb+', suffix=f'.{to_extension}') as output_file:
        command = [ffmpeg_binary, '-y', '-f', from_extension, '-i', 'pipe:']
        command.extend(['-f', to_extension])
        if to_sample_rate is not None:
            command.extend(['-ar', str(to_sample_rate)])
        if to_sample_channels is not None:
            command.extend(['-ac', str(to_sample_channels)])
        if to_extension == 'mp3':
            command.extend(['-q:a', '0'])
        if to_sample_bytes == 2:
            command.extend(['-sample_fmt', 's16'])
        command.append(output_file.name)
        with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
            _stdout, stderr = proc.communicate(input=audio_bytes)
            if proc.returncode != 0:
                _LOGGER.error(stderr.decode())
                raise RuntimeError(f'Unexpected error while running ffmpeg with arguments: {command}.See log for details.')
        output_file.seek(0)
        return output_file.read()

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up TTS."""
    websocket_api.async_register_command(hass, websocket_list_engines)
    websocket_api.async_register_command(hass, websocket_get_engine)
    websocket_api.async_register_command(hass, websocket_list_engine_voices)
    conf = config[DOMAIN][0] if config.get(DOMAIN) else {}
    use_cache = conf.get(CONF_CACHE, DEFAULT_CACHE)
    cache_dir = conf.get(CONF_CACHE_DIR, DEFAULT_CACHE_DIR)
    time_memory = conf.get(CONF_TIME_MEMORY, DEFAULT_TIME_MEMORY)
    tts = SpeechManager(hass, use_cache, cache_dir, time_memory)
    try:
        await tts.async_init_cache()
    except (HomeAssistantError, KeyError):
        _LOGGER.exception('Error on cache init')
        return False
    hass.data[DATA_TTS_MANAGER] = tts
    component = hass.data[DATA_COMPONENT] = EntityComponent[TextToSpeechEntity](_LOGGER, DOMAIN, hass)
    component.register_shutdown()
    hass.http.register_view(TextToSpeechView(tts))
    hass.http.register_view(TextToSpeechUrlView(tts))
    platform_setups = await async_setup_legacy(hass, config)
    component.async_register_entity_service('speak', {
        vol.Required(ATTR_MEDIA_PLAYER_ENTITY_ID): cv.comp_entity_ids,
        vol.Required(ATTR_MESSAGE): cv.string,
        vol.Optional(ATTR_CACHE, default=DEFAULT_CACHE): cv.boolean,
        vol.Optional(ATTR_LANGUAGE): cv.string,
        vol.Optional(ATTR_OPTIONS): dict
    }, 'async_speak')

    async def async_clear_cache_handle(service: ServiceCall) -> None:
        """Handle clear cache service call."""
        await tts.async_clear_cache()
    hass.services.async_register(DOMAIN, SERVICE_CLEAR_CACHE, async_clear_cache_handle, schema=SCHEMA_SERVICE_CLEAR_CACHE)
    for setup in platform_setups:
        hass.async_create_task(setup, eager_start=True)
    return True

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up a config entry."""
    return await hass.data[DATA_COMPONENT].async_setup_entry(entry)

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.data[DATA_COMPONENT].async_unload_entry(entry)

CACHED_PROPERTIES_WITH_ATTR_: Final = {'default_language', 'default_options', 'supported_languages', 'supported_options'}

class TextToSpeechEntity(RestoreEntity, cached_properties=CACHED_PROPERTIES_WITH_ATTR_):
    """Represent a single TTS engine."""
    _attr_should_poll: Final = False
    __last_tts_loaded: Optional[str] = None
    _attr_default_options: Optional[Dict[str, Any]] = None
    _attr_supported_options: Optional[List[str]] = None

    @property
    @final
    def state(self) -> Optional[str]:
        """Return the state of the entity."""
        return self.__last_tts_loaded

    @cached_property
    def supported_languages(self) -> List[str]:
        """Return a list of supported languages."""
        return cast(List[str], self._attr_supported_languages)

    @cached_property
    def default_language(self) -> Optional[str]:
        """Return the default language."""
        return cast(Optional[str], self._attr_default_language)

    @cached_property
    def supported_options(self) -> Optional[List[str]]:
        """Return a list of supported options like voice, emotions."""
        return cast(Optional[List[str]], self._attr_supported_options)

    @cached_property
    def default_options(self) -> Optional[Dict[str, Any]]:
        """Return a mapping with the default options."""
        return cast(Optional[Dict[str, Any]], self._attr_default_options)

    @callback
    def async_get_supported_voices(self, language: str) -> Optional[List[Voice]]:
        """Return a list of supported voices for a language."""
        return None

    async def async_internal_added_to_hass(self) -> None:
        """Call when the entity is added to hass."""
        await super().async_internal_added_to_hass()
        try:
            _ = self.default_language
        except AttributeError as err:
            raise AttributeError("TTS entities must either set the '_attr_default_language' attribute or override the 'default_language' property") from err
        try:
            _ = self.supported_languages
        except AttributeError as err:
            raise AttributeError("TTS entities must either set the '_attr_supported_languages' attribute or override the 'supported_languages' property") from err
        state = await self.async_get_last_state()
        if state is not None and state.state is not None and (state.state not in (STATE_UNAVAILABLE, STATE_UNKNOWN)):
            self.__last_tts_loaded = state.state

    async def async_speak(
        self,
        media_player_entity_id: str,
        message: str,
        cache: bool,
        language: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> None:
        """Speak via a Media Player."""
        await self.hass.services.async_call(
            DOMAIN_MP,
            SERVICE_PLAY_MEDIA,
            {
                ATTR_ENTITY_ID: media_player_entity_id,
                ATTR_MEDIA_CONTENT_ID: generate_media_source_id(
                    self.hass,
                    message=message,
                    engine=self.entity_id,
                    language=language,
                    options=options,
                    cache=cache
                ),
                ATTR_MEDIA_CONTENT_TYPE: MediaType.MUSIC,
                ATTR_MEDIA_ANNOUNCE: True
            },
            blocking=True,
            context=self._context
        )

    @final
    async def internal_async_get_tts_audio(
        self,
        message: str,
        language: str,
        options: Dict[str, Any]
    ) -> TtsAudioType:
        """Process an audio stream to TTS service."""
        self.__last_tts_loaded = dt_util.utcnow().isoformat()
        self.async_write_ha_state()
        return await self.async_get_tts_audio(message=message, language=language, options=options)

    def get_tts_audio(self, message: str, language: str, options: Dict[str, Any]) -> TtsAudioType:
        """Load tts audio file from the engine."""
        raise NotImplementedError

    async def async_get_tts_audio(
        self,
        message: str,
        language: str,
        options: Dict[str, Any]
    ) -> TtsAudioType:
        """Load tts audio file from the engine."""
        return await self.hass.async_add_executor_job(
            partial(self.get_tts_audio, message, language, options=options)
        )

def _hash_options(options: Dict[str, Any]) -> str:
    """Hashes an options dictionary."""
    opts_hash = hashlib.blake2s(digest_size=5)
    for key, value in sorted(options.items()):
        opts_hash.update(str(key).encode())
        opts_hash.update(str(value).encode())
    return opts_hash.hexdigest()

class SpeechManager:
    """Representation of a speech store."""

    def __init__(
        self,
        hass: HomeAssistant,
        use_cache: bool,
        cache_dir: str,
        time_memory: int
    ) -> None:
        """Initialize a speech store."""
        self.hass = hass
        self.providers: Dict[str, Union[Provider, TextToSpeechEntity]] = {}
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.time_memory = time_memory
        self.file_cache: Dict[str, str] = {}
        self.mem_cache: Dict[str, TTSCache] = {}
        self.filename_to_token: Dict[str, str] = {}
        self.token_to_filename: Dict[str, str] = {}

    def _init_cache(self) -> Dict[str, str]:
        """Init cache folder and fetch files."""
        try:
            self.cache_dir = _init_tts_cache_dir(self.hass, self.cache_dir)
        except OSError as err:
            raise HomeAssistantError