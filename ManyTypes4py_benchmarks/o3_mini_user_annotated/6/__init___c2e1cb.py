from __future__ import annotations

import asyncio
from collections.abc import Mapping
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
from typing import Any, Final, TypedDict, final, Optional, Union, Dict, List, Callable, Tuple

from aiohttp import web
import mutagen
from mutagen.id3 import ID3, TextFrame as ID3Text
from propcache.api import cached_property
import voluptuous as vol

from homeassistant.components import ffmpeg, websocket_api
from homeassistant.components.http import HomeAssistantView
from homeassistant.components.media_player import (
    ATTR_MEDIA_ANNOUNCE,
    ATTR_MEDIA_CONTENT_ID,
    ATTR_MEDIA_CONTENT_TYPE,
    DOMAIN as DOMAIN_MP,
    SERVICE_PLAY_MEDIA,
    MediaType,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    ATTR_ENTITY_ID,
    PLATFORM_FORMAT,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
)
from homeassistant.core import HassJob, HomeAssistant, ServiceCall, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_component import EntityComponent
from homeassistant.helpers.event import async_call_later
from homeassistant.helpers.network import get_url
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.typing import UNDEFINED, ConfigType
from homeassistant.util import dt as dt_util, language as language_util

from .const import (
    ATTR_CACHE,
    ATTR_LANGUAGE,
    ATTR_MESSAGE,
    ATTR_OPTIONS,
    CONF_CACHE,
    CONF_CACHE_DIR,
    CONF_TIME_MEMORY,
    DATA_COMPONENT,
    DATA_TTS_MANAGER,
    DEFAULT_CACHE,
    DEFAULT_CACHE_DIR,
    DEFAULT_TIME_MEMORY,
    DOMAIN,
    TtsAudioType,
)
from .helper import get_engine_instance
from .legacy import PLATFORM_SCHEMA, PLATFORM_SCHEMA_BASE, Provider, async_setup_legacy
from .media_source import generate_media_source_id, media_source_id_to_kwargs
from .models import Voice

__all__ = [
    "ATTR_AUDIO_OUTPUT",
    "ATTR_PREFERRED_FORMAT",
    "ATTR_PREFERRED_SAMPLE_BYTES",
    "ATTR_PREFERRED_SAMPLE_CHANNELS",
    "ATTR_PREFERRED_SAMPLE_RATE",
    "CONF_LANG",
    "DEFAULT_CACHE_DIR",
    "PLATFORM_SCHEMA",
    "PLATFORM_SCHEMA_BASE",
    "Provider",
    "SampleFormat",
    "TtsAudioType",
    "Voice",
    "async_default_engine",
    "async_get_media_source_audio",
    "async_support_options",
    "generate_media_source_id",
]

_LOGGER: Final = logging.getLogger(__name__)

ATTR_PLATFORM = "platform"
ATTR_AUDIO_OUTPUT = "audio_output"
ATTR_PREFERRED_FORMAT = "preferred_format"
ATTR_PREFERRED_SAMPLE_RATE = "preferred_sample_rate"
ATTR_PREFERRED_SAMPLE_CHANNELS = "preferred_sample_channels"
ATTR_PREFERRED_SAMPLE_BYTES = "preferred_sample_bytes"
ATTR_MEDIA_PLAYER_ENTITY_ID = "media_player_entity_id"
ATTR_VOICE = "voice"

_DEFAULT_FORMAT: Final = "mp3"
_PREFFERED_FORMAT_OPTIONS: Final[set[str]] = {
    ATTR_PREFERRED_FORMAT,
    ATTR_PREFERRED_SAMPLE_RATE,
    ATTR_PREFERRED_SAMPLE_CHANNELS,
    ATTR_PREFERRED_SAMPLE_BYTES,
}

CONF_LANG = "language"

SERVICE_CLEAR_CACHE = "clear_cache"

_RE_LEGACY_VOICE_FILE: Final = re.compile(
    r"([a-f0-9]{40})_([^_]+)_([^_]+)_([a-z_]+)\.[a-z0-9]{3,4}"
)
_RE_VOICE_FILE: Final = re.compile(
    r"([a-f0-9]{40})_([^_]+)_([^_]+)_(tts\.[a-z0-9_]+)\.[a-z0-9]{3,4}"
)
KEY_PATTERN: Final = "{0}_{1}_{2}_{3}"


SCHEMA_SERVICE_CLEAR_CACHE: Final = vol.Schema({})


class TTSCache(TypedDict):
    filename: str
    voice: bytes
    pending: Optional[asyncio.Task[Any]]


@callback
def async_default_engine(hass: HomeAssistant) -> Optional[str]:
    default_entity_id: Optional[str] = None

    for entity in hass.data[DATA_COMPONENT].entities:
        if entity.platform and entity.platform.platform_name == "cloud":
            return entity.entity_id

        if default_entity_id is None:
            default_entity_id = entity.entity_id

    return default_entity_id or next(iter(hass.data[DATA_TTS_MANAGER].providers), None)


@callback
def async_resolve_engine(hass: HomeAssistant, engine: Optional[str]) -> Optional[str]:
    if engine is not None:
        if (
            not hass.data[DATA_COMPONENT].get_entity(engine)
            and engine not in hass.data[DATA_TTS_MANAGER].providers
        ):
            return None
        return engine

    return async_default_engine(hass)


async def async_support_options(
    hass: HomeAssistant,
    engine: str,
    language: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> bool:
    if (engine_instance := get_engine_instance(hass, engine)) is None:
        raise HomeAssistantError(f"Provider {engine} not found")

    try:
        hass.data[DATA_TTS_MANAGER].process_options(engine_instance, language, options)
    except HomeAssistantError:
        return False

    return True


async def async_get_media_source_audio(
    hass: HomeAssistant,
    media_source_id: str,
) -> Tuple[str, bytes]:
    return await hass.data[DATA_TTS_MANAGER].async_get_tts_audio(
        **media_source_id_to_kwargs(media_source_id),
    )


@callback
def async_get_text_to_speech_languages(hass: HomeAssistant) -> set[str]:
    languages: set[str] = set()

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
    to_sample_bytes: Optional[int] = None,
) -> bytes:
    ffmpeg_manager = ffmpeg.get_ffmpeg_manager(hass)
    return await hass.async_add_executor_job(
        lambda: _convert_audio(
            ffmpeg_manager.binary,
            from_extension,
            audio_bytes,
            to_extension,
            to_sample_rate=to_sample_rate,
            to_sample_channels=to_sample_channels,
            to_sample_bytes=to_sample_bytes,
        )
    )


def _convert_audio(
    ffmpeg_binary: str,
    from_extension: str,
    audio_bytes: bytes,
    to_extension: str,
    to_sample_rate: Optional[int] = None,
    to_sample_channels: Optional[int] = None,
    to_sample_bytes: Optional[int] = None,
) -> bytes:
    with tempfile.NamedTemporaryFile(
        mode="wb+", suffix=f".{to_extension}"
    ) as output_file:
        command: List[str] = [
            ffmpeg_binary,
            "-y",
            "-f",
            from_extension,
            "-i",
            "pipe:",
        ]
        command.extend(["-f", to_extension])

        if to_sample_rate is not None:
            command.extend(["-ar", str(to_sample_rate)])

        if to_sample_channels is not None:
            command.extend(["-ac", str(to_sample_channels)])

        if to_extension == "mp3":
            command.extend(["-q:a", "0"])

        if to_sample_bytes == 2:
            command.extend(["-sample_fmt", "s16"])

        command.append(output_file.name)

        with subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE
        ) as proc:
            _stdout, stderr = proc.communicate(input=audio_bytes)
            if proc.returncode != 0:
                _LOGGER.error(stderr.decode())
                raise RuntimeError(
                    f"Unexpected error while running ffmpeg with arguments: {command}."
                    "See log for details."
                )

        output_file.seek(0)
        return output_file.read()


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    websocket_api.async_register_command(hass, websocket_list_engines)
    websocket_api.async_register_command(hass, websocket_get_engine)
    websocket_api.async_register_command(hass, websocket_list_engine_voices)

    conf: Dict[str, Any] = config[DOMAIN][0] if config.get(DOMAIN) else {}
    use_cache: bool = conf.get(CONF_CACHE, DEFAULT_CACHE)
    cache_dir: str = conf.get(CONF_CACHE_DIR, DEFAULT_CACHE_DIR)
    time_memory: int = conf.get(CONF_TIME_MEMORY, DEFAULT_TIME_MEMORY)

    tts = SpeechManager(hass, use_cache, cache_dir, time_memory)

    try:
        await tts.async_init_cache()
    except (HomeAssistantError, KeyError):
        _LOGGER.exception("Error on cache init")
        return False

    hass.data[DATA_TTS_MANAGER] = tts
    component: EntityComponent[TextToSpeechEntity] = hass.data[DATA_COMPONENT] = EntityComponent(
        _LOGGER, DOMAIN, hass
    )

    component.register_shutdown()

    hass.http.register_view(TextToSpeechView(tts))
    hass.http.register_view(TextToSpeechUrlView(tts))

    platform_setups: List[Any] = await async_setup_legacy(hass, config)

    component.async_register_entity_service(
        "speak",
        {
            vol.Required(ATTR_MEDIA_PLAYER_ENTITY_ID): cv.comp_entity_ids,
            vol.Required(ATTR_MESSAGE): cv.string,
            vol.Optional(ATTR_CACHE, default=DEFAULT_CACHE): cv.boolean,
            vol.Optional(ATTR_LANGUAGE): cv.string,
            vol.Optional(ATTR_OPTIONS): dict,
        },
        "async_speak",
    )

    async def async_clear_cache_handle(service: ServiceCall) -> None:
        await tts.async_clear_cache()

    hass.services.async_register(
        DOMAIN,
        SERVICE_CLEAR_CACHE,
        async_clear_cache_handle,
        schema=SCHEMA_SERVICE_CLEAR_CACHE,
    )

    for setup in platform_setups:
        hass.async_create_task(setup, eager_start=True)

    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    return await hass.data[DATA_COMPONENT].async_setup_entry(entry)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    return await hass.data[DATA_COMPONENT].async_unload_entry(entry)


CACHED_PROPERTIES_WITH_ATTR_: Final[set[str]] = {
    "default_language",
    "default_options",
    "supported_languages",
    "supported_options",
}


class TextToSpeechEntity(RestoreEntity):
    _attr_should_poll: Final[bool] = False
    __last_tts_loaded: Optional[str] = None

    _attr_default_language: str
    _attr_default_options: Optional[Mapping[str, Any]] = None
    _attr_supported_languages: List[str]
    _attr_supported_options: Optional[List[str]] = None

    @property
    @final
    def state(self) -> Optional[str]:
        if self.__last_tts_loaded is None:
            return None
        return self.__last_tts_loaded

    @cached_property
    def supported_languages(self) -> List[str]:
        return self._attr_supported_languages

    @cached_property
    def default_language(self) -> str:
        return self._attr_default_language

    @cached_property
    def supported_options(self) -> Optional[List[str]]:
        return self._attr_supported_options

    @cached_property
    def default_options(self) -> Optional[Mapping[str, Any]]:
        return self._attr_default_options

    @callback
    def async_get_supported_voices(self, language: str) -> Optional[List[Voice]]:
        return None

    async def async_internal_added_to_hass(self) -> None:
        await super().async_internal_added_to_hass()
        try:
            _ = self.default_language
        except AttributeError as err:
            raise AttributeError(
                "TTS entities must either set the '_attr_default_language' attribute or override the 'default_language' property"
            ) from err
        try:
            _ = self.supported_languages
        except AttributeError as err:
            raise AttributeError(
                "TTS entities must either set the '_attr_supported_languages' attribute or override the 'supported_languages' property"
            ) from err
        state = await self.async_get_last_state()
        if (
            state is not None
            and state.state is not None
            and state.state not in (STATE_UNAVAILABLE, STATE_UNKNOWN)
        ):
            self.__last_tts_loaded = state.state

    async def async_speak(
        self,
        media_player_entity_id: List[str],
        message: str,
        cache: bool,
        language: Optional[str],
        options: Optional[Dict[str, Any]],
    ) -> None:
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
                    cache=cache,
                ),
                ATTR_MEDIA_CONTENT_TYPE: MediaType.MUSIC,
                ATTR_MEDIA_ANNOUNCE: True,
            },
            blocking=True,
            context=self._context,
        )

    @final
    async def internal_async_get_tts_audio(
        self, message: str, language: str, options: Dict[str, Any]
    ) -> TtsAudioType:
        self.__last_tts_loaded = dt_util.utcnow().isoformat()
        self.async_write_ha_state()
        return await self.async_get_tts_audio(message=message, language=language, options=options)

    def get_tts_audio(self, message: str, language: str, options: Dict[str, Any]) -> TtsAudioType:
        raise NotImplementedError

    async def async_get_tts_audio(
        self, message: str, language: str, options: Dict[str, Any]
    ) -> TtsAudioType:
        return await self.hass.async_add_executor_job(
            partial(self.get_tts_audio, message, language, options=options)
        )


def _hash_options(options: Dict[str, Any]) -> str:
    opts_hash = hashlib.blake2s(digest_size=5)
    for key, value in sorted(options.items()):
        opts_hash.update(str(key).encode())
        opts_hash.update(str(value).encode())
    return opts_hash.hexdigest()


class SpeechManager:
    def __init__(
        self,
        hass: HomeAssistant,
        use_cache: bool,
        cache_dir: str,
        time_memory: int,
    ) -> None:
        self.hass: HomeAssistant = hass
        self.providers: Dict[str, Provider] = {}

        self.use_cache: bool = use_cache
        self.cache_dir: str = cache_dir
        self.time_memory: int = time_memory
        self.file_cache: Dict[str, str] = {}
        self.mem_cache: Dict[str, TTSCache] = {}

        self.filename_to_token: Dict[str, str] = {}
        self.token_to_filename: Dict[str, str] = {}

    def _init_cache(self) -> Dict[str, str]:
        try:
            self.cache_dir = _init_tts_cache_dir(self.hass, self.cache_dir)
        except OSError as err:
            raise HomeAssistantError(f"Can't init cache dir {err}") from err

        try:
            return _get_cache_files(self.cache_dir)
        except OSError as err:
            raise HomeAssistantError(f"Can't read cache dir {err}") from err

    async def async_init_cache(self) -> None:
        self.file_cache.update(await self.hass.async_add_executor_job(self._init_cache))

    async def async_clear_cache(self) -> None:
        self.mem_cache = {}

        def remove_files() -> None:
            for filename in self.file_cache.values():
                try:
                    os.remove(os.path.join(self.cache_dir, filename))
                except OSError as err:
                    _LOGGER.warning("Can't remove cache file '%s': %s", filename, err)

        await self.hass.async_add_executor_job(remove_files)
        self.file_cache = {}

    @callback
    def async_register_legacy_engine(
        self, engine: str, provider: Provider, config: ConfigType
    ) -> None:
        provider.hass = self.hass
        if provider.name is None:
            provider.name = engine
        self.providers[engine] = provider

        self.hass.config.components.add(
            PLATFORM_FORMAT.format(domain=DOMAIN, platform=engine)
        )

    @callback
    def process_options(
        self,
        engine_instance: Union[TextToSpeechEntity, Provider],
        language: Optional[str],
        options: Optional[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Any]]:
        language_final: Optional[str] = language or engine_instance.default_language  # type: ignore[attr-defined]
        if (
            language_final is None
            or engine_instance.supported_languages is None  # type: ignore[attr-defined]
            or language_final not in engine_instance.supported_languages  # type: ignore[attr-defined]
        ):
            raise HomeAssistantError(f"Language '{language_final}' not supported")

        options = options or {}
        supported_options: List[str] = engine_instance.supported_options or []  # type: ignore[attr-defined]

        invalid_opts: List[str] = []
        merged_options: Dict[str, Any] = dict(engine_instance.default_options or {})  # type: ignore[attr-defined]
        for option_name, option_value in options.items():
            if (option_name in supported_options) or (option_name in _PREFFERED_FORMAT_OPTIONS):
                merged_options[option_name] = option_value
            else:
                invalid_opts.append(option_name)

        if invalid_opts:
            raise HomeAssistantError(f"Invalid options found: {invalid_opts}")

        return language_final, merged_options

    async def async_get_url_path(
        self,
        engine: str,
        message: str,
        cache: Optional[bool] = None,
        language: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        if (engine_instance := get_engine_instance(self.hass, engine)) is None:
            raise HomeAssistantError(f"Provider {engine} not found")

        language, options = self.process_options(engine_instance, language, options)
        cache_key: str = self._generate_cache_key(message, language, options, engine)
        use_cache: bool = cache if cache is not None else self.use_cache

        if cache_key in self.mem_cache:
            filename: str = self.mem_cache[cache_key]["filename"]
        elif use_cache and cache_key in self.file_cache:
            filename = self.file_cache[cache_key]
            self.hass.async_create_task(self._async_file_to_mem(cache_key))
        else:
            filename = await self._async_get_tts_audio(
                engine_instance, cache_key, message, use_cache, language, options
            )

        token: Optional[str] = self.filename_to_token.get(filename)
        if not token:
            token = secrets.token_urlsafe(16) + os.path.splitext(filename)[1]
            self.filename_to_token[filename] = token
            self.token_to_filename[token] = filename

        return f"/api/tts_proxy/{token}"

    async def async_get_tts_audio(
        self,
        engine: str,
        message: str,
        cache: Optional[bool] = None,
        language: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, bytes]:
        if (engine_instance := get_engine_instance(self.hass, engine)) is None:
            raise HomeAssistantError(f"Provider {engine} not found")

        language, options = self.process_options(engine_instance, language, options)
        cache_key: str = self._generate_cache_key(message, language, options, engine)
        use_cache: bool = cache if cache is not None else self.use_cache

        if cache_key not in self.mem_cache:
            if use_cache and cache_key in self.file_cache:
                await self._async_file_to_mem(cache_key)
            else:
                await self._async_get_tts_audio(
                    engine_instance, cache_key, message, use_cache, language, options
                )

        extension: str = os.path.splitext(self.mem_cache[cache_key]["filename"])[1][1:]
        cached: TTSCache = self.mem_cache[cache_key]
        if pending := cached.get("pending"):
            await pending
            cached = self.mem_cache[cache_key]
        return extension, cached["voice"]

    @callback
    def _generate_cache_key(
        self,
        message: str,
        language: str,
        options: Optional[Dict[str, Any]],
        engine: str,
    ) -> str:
        options_key: str = _hash_options(options) if options else "-"
        msg_hash: str = hashlib.sha1(bytes(message, "utf-8")).hexdigest()
        return KEY_PATTERN.format(
            msg_hash, language.replace("_", "-"), options_key, engine
        ).lower()

    async def _async_get_tts_audio(
        self,
        engine_instance: Union[TextToSpeechEntity, Provider],
        cache_key: str,
        message: str,
        cache: bool,
        language: str,
        options: Dict[str, Any],
    ) -> str:
        if ATTR_PREFERRED_FORMAT in (engine_instance.supported_options or []):  # type: ignore[attr-defined]
            final_extension: str = options.get(ATTR_PREFERRED_FORMAT, _DEFAULT_FORMAT)  # type: ignore
        else:
            final_extension = options.pop(ATTR_PREFERRED_FORMAT, _DEFAULT_FORMAT)

        if ATTR_PREFERRED_SAMPLE_RATE in (engine_instance.supported_options or []):  # type: ignore[attr-defined]
            sample_rate: Optional[Union[int, str]] = options.get(ATTR_PREFERRED_SAMPLE_RATE)
        else:
            sample_rate = options.pop(ATTR_PREFERRED_SAMPLE_RATE, None)

        if sample_rate is not None:
            sample_rate = int(sample_rate)

        if ATTR_PREFERRED_SAMPLE_CHANNELS in (engine_instance.supported_options or []):  # type: ignore[attr-defined]
            sample_channels: Optional[Union[int, str]] = options.get(ATTR_PREFERRED_SAMPLE_CHANNELS)
        else:
            sample_channels = options.pop(ATTR_PREFERRED_SAMPLE_CHANNELS, None)

        if sample_channels is not None:
            sample_channels = int(sample_channels)

        if ATTR_PREFERRED_SAMPLE_BYTES in (engine_instance.supported_options or []):  # type: ignore[attr-defined]
            sample_bytes: Optional[Union[int, str]] = options.get(ATTR_PREFERRED_SAMPLE_BYTES)
        else:
            sample_bytes = options.pop(ATTR_PREFERRED_SAMPLE_BYTES, None)

        if sample_bytes is not None:
            sample_bytes = int(sample_bytes)

        async def get_tts_data() -> str:
            if engine_instance.name is None or engine_instance.name is UNDEFINED:  # type: ignore[attr-defined]
                raise HomeAssistantError("TTS engine name is not set.")

            if isinstance(engine_instance, Provider):
                extension, data = await engine_instance.async_get_tts_audio(
                    message, language, options
                )
            else:
                extension, data = await engine_instance.internal_async_get_tts_audio(
                    message, language, options
                )

            if data is None or extension is None:
                raise HomeAssistantError(
                    f"No TTS from {engine_instance.name} for '{message}'"
                )

            needs_conversion: bool = (
                (final_extension != extension)
                or (sample_rate is not None)
                or (sample_channels is not None)
                or (sample_bytes is not None)
            )

            if needs_conversion:
                data = await async_convert_audio(
                    self.hass,
                    extension,
                    data,
                    to_extension=final_extension,
                    to_sample_rate=sample_rate,
                    to_sample_channels=sample_channels,
                    to_sample_bytes=sample_bytes,
                )

            filename: str = f"{cache_key}.{final_extension}".lower()

            if not _RE_VOICE_FILE.match(filename) and not _RE_LEGACY_VOICE_FILE.match(filename):
                raise HomeAssistantError(
                    f"TTS filename '{filename}' from {engine_instance.name} is invalid!"
                )

            if final_extension == "mp3":
                data = self.write_tags(
                    filename, data, engine_instance.name, message, language, options
                )

            self._async_store_to_memcache(cache_key, filename, data)

            if cache:
                self.hass.async_create_task(
                    self._async_save_tts_audio(cache_key, filename, data)
                )

            return filename

        audio_task: asyncio.Task[str] = self.hass.async_create_task(get_tts_data(), eager_start=False)

        def handle_error(_future: asyncio.Future[Any]) -> None:
            if audio_task.exception():
                self.mem_cache.pop(cache_key, None)

        audio_task.add_done_callback(handle_error)

        filename: str = f"{cache_key}.{final_extension}".lower()
        self.mem_cache[cache_key] = {
            "filename": filename,
            "voice": b"",
            "pending": audio_task,
        }
        return filename

    async def _async_save_tts_audio(
        self, cache_key: str, filename: str, data: bytes
    ) -> None:
        voice_file: str = os.path.join(self.cache_dir, filename)

        def save_speech() -> None:
            with open(voice_file, "wb") as speech:
                speech.write(data)

        try:
            await self.hass.async_add_executor_job(save_speech)
            self.file_cache[cache_key] = filename
        except OSError as err:
            _LOGGER.error("Can't write %s: %s", filename, err)

    async def _async_file_to_mem(self, cache_key: str) -> None:
        filename: Optional[str] = self.file_cache.get(cache_key)
        if not filename:
            raise HomeAssistantError(f"Key {cache_key} not in file cache!")

        voice_file: str = os.path.join(self.cache_dir, filename)

        def load_speech() -> bytes:
            with open(voice_file, "rb") as speech:
                return speech.read()

        try:
            data: bytes = await self.hass.async_add_executor_job(load_speech)
        except OSError as err:
            del self.file_cache[cache_key]
            raise HomeAssistantError(f"Can't read {voice_file}") from err

        self._async_store_to_memcache(cache_key, filename, data)

    @callback
    def _async_store_to_memcache(
        self, cache_key: str, filename: str, data: bytes
    ) -> None:
        self.mem_cache[cache_key] = {
            "filename": filename,
            "voice": data,
            "pending": None,
        }

        @callback
        def async_remove_from_mem(_: datetime) -> None:
            self.mem_cache.pop(cache_key, None)

        async_call_later(
            self.hass,
            self.time_memory,
            HassJob(
                async_remove_from_mem,
                name="tts remove_from_mem",
                cancel_on_shutdown=True,
            ),
        )

    async def async_read_tts(self, token: str) -> Tuple[Optional[str], bytes]:
        filename: Optional[str] = self.token_to_filename.get(token)
        if not filename:
            raise HomeAssistantError(f"{token} was not recognized!")

        record = _RE_VOICE_FILE.match(filename.lower()) or _RE_LEGACY_VOICE_FILE.match(filename.lower())
        if not record:
            raise HomeAssistantError("Wrong tts file format!")

        cache_key: str = KEY_PATTERN.format(
            record.group(1), record.group(2), record.group(3), record.group(4)
        )
        if cache_key not in self.mem_cache:
            if cache_key not in self.file_cache:
                raise HomeAssistantError(f"{cache_key} not in cache!")
            await self._async_file_to_mem(cache_key)

        cached: TTSCache = self.mem_cache[cache_key]
        if pending := cached.get("pending"):
            await pending
            cached = self.mem_cache[cache_key]

        content, _ = mimetypes.guess_type(filename)
        return content, cached["voice"]

    @staticmethod
    def write_tags(
        filename: str,
        data: bytes,
        engine_name: str,
        message: str,
        language: str,
        options: Optional[Dict[str, Any]],
    ) -> bytes:
        data_bytes: io.BytesIO = io.BytesIO(data)
        data_bytes.name = filename
        data_bytes.seek(0)

        album: str = engine_name
        artist: str = language

        if options is not None and (voice := options.get("voice")) is not None:
            artist = voice

        try:
            tts_file = mutagen.File(data_bytes)
            if tts_file is not None:
                if not tts_file.tags:
                    tts_file.add_tags()
                if isinstance(tts_file.tags, ID3):
                    tts_file["artist"] = ID3Text(
                        encoding=3,
                        text=artist,
                    )
                    tts_file["album"] = ID3Text(
                        encoding=3,
                        text=album,
                    )
                    tts_file["title"] = ID3Text(
                        encoding=3,
                        text=message,
                    )
                else:
                    tts_file["artist"] = artist
                    tts_file["album"] = album
                    tts_file["title"] = message
                tts_file.save(data_bytes)
        except mutagen.MutagenError as err:
            _LOGGER.error("ID3 tag error: %s", err)

        return data_bytes.getvalue()


def _init_tts_cache_dir(hass: HomeAssistant, cache_dir: str) -> str:
    if not os.path.isabs(cache_dir):
        cache_dir = hass.config.path(cache_dir)
    if not os.path.isdir(cache_dir):
        _LOGGER.info("Create cache dir %s", cache_dir)
        os.mkdir(cache_dir)
    return cache_dir


def _get_cache_files(cache_dir: str) -> Dict[str, str]:
    cache: Dict[str, str] = {}
    folder_data: List[str] = os.listdir(cache_dir)
    for file_data in folder_data:
        record = _RE_VOICE_FILE.match(file_data) or _RE_LEGACY_VOICE_FILE.match(file_data)
        if record:
            key: str = KEY_PATTERN.format(
                record.group(1), record.group(2), record.group(3), record.group(4)
            )
            cache[key.lower()] = file_data.lower()
    return cache


class TextToSpeechUrlView(HomeAssistantView):
    requires_auth: Final[bool] = True
    url: Final[str] = "/api/tts_get_url"
    name: Final[str] = "api:tts:geturl"

    def __init__(self, tts: SpeechManager) -> None:
        self.tts: SpeechManager = tts

    async def post(self, request: web.Request) -> web.Response:
        try:
            data: Dict[str, Any] = await request.json()
        except ValueError:
            return self.json_message("Invalid JSON specified", HTTPStatus.BAD_REQUEST)
        if (not data.get("engine_id") and not data.get(ATTR_PLATFORM)) or not data.get(ATTR_MESSAGE):
            return self.json_message(
                "Must specify platform and message", HTTPStatus.BAD_REQUEST
            )

        engine: str = data.get("engine_id") or data[ATTR_PLATFORM]
        message: str = data[ATTR_MESSAGE]
        cache: Optional[bool] = data.get(ATTR_CACHE)
        language: Optional[str] = data.get(ATTR_LANGUAGE)
        options: Optional[Dict[str, Any]] = data.get(ATTR_OPTIONS)

        try:
            path: str = await self.tts.async_get_url_path(
                engine, message, cache=cache, language=language, options=options
            )
        except HomeAssistantError as err:
            _LOGGER.error("Error on init tts: %s", err)
            return self.json({"error": str(err)}, HTTPStatus.BAD_REQUEST)

        base: str = get_url(self.tts.hass)
        url: str = base + path

        return self.json({"url": url, "path": path})


class TextToSpeechView(HomeAssistantView):
    requires_auth: Final[bool] = False
    url: Final[str] = "/api/tts_proxy/{filename}"
    name: Final[str] = "api:tts_speech"

    def __init__(self, tts: SpeechManager) -> None:
        self.tts: SpeechManager = tts

    async def get(self, request: web.Request, filename: str) -> web.Response:
        try:
            content: Optional[str]
            data: bytes
            content, data = await self.tts.async_read_tts(filename)
        except HomeAssistantError as err:
            _LOGGER.error("Error on load tts: %s", err)
            return web.Response(status=HTTPStatus.NOT_FOUND)

        return web.Response(body=data, content_type=content)


@websocket_api.websocket_command(
    {
        "type": "tts/engine/list",
        vol.Optional("country"): str,
        vol.Optional("language"): str,
    }
)
@callback
def websocket_list_engines(
    hass: HomeAssistant, connection: websocket_api.ActiveConnection, msg: Dict[str, Any]
) -> None:
    country: Optional[str] = msg.get("country")
    language: Optional[str] = msg.get("language")
    providers: List[Dict[str, Any]] = []
    provider_info: Dict[str, Any]
    entity_domains: set[str] = set()

    for entity in hass.data[DATA_COMPONENT].entities:
        provider_info = {
            "engine_id": entity.entity_id,
            "supported_languages": entity.supported_languages,
        }
        if language:
            provider_info["supported_languages"] = language_util.matches(
                language, entity.supported_languages, country
            )
        providers.append(provider_info)
        if entity.platform:
            entity_domains.add(entity.platform.platform_name)
    for engine_id, provider in hass.data[DATA_TTS_MANAGER].providers.items():
        provider_info = {
            "engine_id": engine_id,
            "name": provider.name,
            "supported_languages": provider.supported_languages,
        }
        if language:
            provider_info["supported_languages"] = language_util.matches(
                language, provider.supported_languages, country
            )
        if engine_id in entity_domains:
            provider_info["deprecated"] = True
        providers.append(provider_info)

    connection.send_message(
        websocket_api.result_message(msg["id"], {"providers": providers})
    )


@websocket_api.websocket_command(
    {
        "type": "tts/engine/get",
        vol.Required("engine_id"): str,
    }
)
@callback
def websocket_get_engine(
    hass: HomeAssistant, connection: websocket_api.ActiveConnection, msg: Dict[str, Any]
) -> None:
    engine_id: str = msg["engine_id"]
    provider: Optional[Union[TextToSpeechEntity, Provider]] = next(
        (
            entity
            for entity in hass.data[DATA_COMPONENT].entities
            if entity.entity_id == engine_id
        ),
        None,
    )
    if not provider:
        provider = hass.data[DATA_TTS_MANAGER].providers.get(engine_id)

    if not provider:
        connection.send_error(
            msg["id"],
            websocket_api.ERR_NOT_FOUND,
            f"tts engine {engine_id} not found",
        )
        return

    provider_info: Dict[str, Any] = {
        "engine_id": engine_id,
        "supported_languages": provider.supported_languages,
    }
    if isinstance(provider, Provider):
        provider_info["name"] = provider.name

    connection.send_message(
        websocket_api.result_message(msg["id"], {"provider": provider_info})
    )


@websocket_api.websocket_command(
    {
        "type": "tts/engine/voices",
        vol.Required("engine_id"): str,
        vol.Required("language"): str,
    }
)
@callback
def websocket_list_engine_voices(
    hass: HomeAssistant, connection: websocket_api.ActiveConnection, msg: Dict[str, Any]
) -> None:
    engine_id: str = msg["engine_id"]
    language: str = msg["language"]

    engine_instance: Optional[Union[TextToSpeechEntity, Provider]] = get_engine_instance(hass, engine_id)

    if not engine_instance:
        connection.send_error(
            msg["id"],
            websocket_api.ERR_NOT_FOUND,
            f"tts engine {engine_id} not found",
        )
        return

    voices: Dict[str, Any] = {"voices": engine_instance.async_get_supported_voices(language)}  # type: ignore
    connection.send_message(websocket_api.result_message(msg["id"], voices))
