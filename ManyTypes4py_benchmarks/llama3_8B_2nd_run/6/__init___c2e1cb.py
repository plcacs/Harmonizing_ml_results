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
from typing import Any, Final, TypedDict, final
from aiohttp import web
import mutagen
from mutagen.id3 import ID3, TextFrame as ID3Text
from propcache.api import cached_property
import voluptuous as vol
from homeassistant.components import ffmpeg, websocket_api
from homeassistant.components.http import HomeAssistantView
from homeassistant.components.media_player import ATTR_MEDIA_ANNOUNCE, ATTR_MEDIA_CONTENT_ID, ATTR_MEDIA_CONTENT_TYPE, MediaType
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_ENTITY_ID, PLATFORM_FORMAT, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HassJob, HomeAssistant, ServiceCall, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_component import EntityComponent
from homeassistant.helpers.event import async_call_later
from homeassistant.helpers.network import get_url
from homeassistant.util import dt as dt_util, language as language_util
from .const import ATTR_AUDIO_OUTPUT, ATTR_PREFERRED_FORMAT, ATTR_PREFERRED_SAMPLE_BYTES, ATTR_PREFERRED_SAMPLE_CHANNELS, ATTR_PREFERRED_SAMPLE_RATE, CONF_CACHE, CONF_CACHE_DIR, CONF_TIME_MEMORY, DATA_COMPONENT, DATA_TTS_MANAGER, DEFAULT_CACHE, DEFAULT_CACHE_DIR, DEFAULT_TIME_MEMORY, DOMAIN, TtsAudioType
from .helper import get_engine_instance
from .legacy import PLATFORM_SCHEMA, PLATFORM_SCHEMA_BASE, Provider, async_setup_legacy
from .media_source import generate_media_source_id, media_source_id_to_kwargs
from .models import Voice

class TTSCache(TypedDict):
    """Cached TTS file."""

@callback
def async_default_engine(hass: HomeAssistant) -> str | None:
    """Return the domain or entity id of the default engine.

    Returns None if no engines found.
    """
    default_entity_id: str | None = None
    for entity in hass.data[DATA_COMPONENT].entities:
        if entity.platform and entity.platform.platform_name == 'cloud':
            return entity.entity_id
        if default_entity_id is None:
            default_entity_id = entity.entity_id
    return default_entity_id or next(iter(hass.data[DATA_TTS_MANAGER].providers), None)

@callback
def async_resolve_engine(hass: HomeAssistant, engine: str) -> str | None:
    """Resolve engine.

    Returns None if no engines found or invalid engine passed in.
    """
    if engine is not None:
        if not hass.data[DATA_COMPONENT].get_entity(engine) and engine not in hass.data[DATA_TTS_MANAGER].providers:
            return None
        return engine
    return async_default_engine(hass)

async def async_support_options(hass: HomeAssistant, engine: str, language: str | None, options: dict[str, Any]) -> bool:
    """Return if an engine supports options."""
    if (engine_instance := get_engine_instance(hass, engine)) is None:
        raise HomeAssistantError(f'Provider {engine} not found')
    try:
        hass.data[DATA_TTS_MANAGER].process_options(engine_instance, language, options)
    except HomeAssistantError:
        return False
    return True

async def async_get_media_source_audio(hass: HomeAssistant, media_source_id: str) -> tuple[str, bytes]:
    """Get TTS audio as extension, data."""
    return await hass.data[DATA_TTS_MANAGER].async_get_tts_audio(**media_source_id_to_kwargs(media_source_id))

class TextToSpeechEntity(RestoreEntity, cached_properties=CACHED_PROPERTIES_WITH_ATTR_):
    """Represent a single TTS engine."""

    _attr_should_poll: bool
    __last_tts_loaded: datetime | None
    _attr_default_options: dict[str, Any] | None
    _attr_supported_options: list[str] | None

    @property
    @final
    def state(self) -> str | None:
        """Return the state of the entity."""
        if self.__last_tts_loaded is None:
            return None
        return self.__last_tts_loaded.isoformat()

    @cached_property
    def supported_languages(self) -> set[str]:
        """Return a list of supported languages."""
        return self._attr_supported_languages

    @cached_property
    def default_language(self) -> str | None:
        """Return the default language."""
        return self._attr_default_language

    @cached_property
    def supported_options(self) -> list[str]:
        """Return a list of supported options like voice, emotions."""
        return self._attr_supported_options

    @cached_property
    def default_options(self) -> dict[str, Any]:
        """Return a mapping with the default options."""
        return self._attr_default_options

    @callback
    def async_get_supported_voices(self, language: str) -> list[str]:
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

    async def async_speak(self, media_player