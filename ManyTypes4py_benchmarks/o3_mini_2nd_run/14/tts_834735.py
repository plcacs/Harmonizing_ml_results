"""Support for the Google Cloud TTS service."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, cast, Optional, Dict, List, Tuple
from google.api_core.exceptions import GoogleAPIError, Unauthenticated
from google.cloud import texttospeech
import voluptuous as vol
from homeassistant.components.tts import (
    CONF_LANG,
    PLATFORM_SCHEMA as TTS_PLATFORM_SCHEMA,
    Provider,
    TextToSpeechEntity,
    TtsAudioType,
    Voice,
)
from homeassistant.config_entries import SOURCE_IMPORT, ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from .const import (
    CONF_ENCODING,
    CONF_GAIN,
    CONF_GENDER,
    CONF_KEY_FILE,
    CONF_PITCH,
    CONF_PROFILES,
    CONF_SERVICE_ACCOUNT_INFO,
    CONF_SPEED,
    CONF_TEXT_TYPE,
    CONF_VOICE,
    DEFAULT_GAIN,
    DEFAULT_LANG,
    DEFAULT_PITCH,
    DEFAULT_SPEED,
    DOMAIN,
)
from .helpers import async_tts_voices, tts_options_schema, tts_platform_schema

_LOGGER = logging.getLogger(__name__)
PLATFORM_SCHEMA = TTS_PLATFORM_SCHEMA.extend(tts_platform_schema().schema)


async def async_get_engine(
    hass: HomeAssistant, config: ConfigType, discovery_info: Optional[DiscoveryInfoType] = None
) -> Optional[GoogleCloudTTSProvider]:
    """Set up Google Cloud TTS component."""
    key_file: Optional[str] = config.get(CONF_KEY_FILE)
    if key_file:
        key_file = hass.config.path(key_file)
        if not Path(key_file).is_file():
            _LOGGER.error("File %s doesn't exist", key_file)
            return None
    if key_file:
        client: texttospeech.TextToSpeechAsyncClient = texttospeech.TextToSpeechAsyncClient.from_service_account_file(key_file)
        if not hass.config_entries.async_entries(DOMAIN):
            _LOGGER.debug("Creating config entry by importing: %s", config)
            hass.async_create_task(
                hass.config_entries.flow.async_init(DOMAIN, context={"source": SOURCE_IMPORT}, data=config)
            )
    else:
        client = texttospeech.TextToSpeechAsyncClient()
    try:
        voices: Dict[str, List[Any]] = await async_tts_voices(client)
    except GoogleAPIError as err:
        _LOGGER.error("Error from calling list_voices: %s", err)
        return None
    return GoogleCloudTTSProvider(
        client,
        voices,
        config.get(CONF_LANG, DEFAULT_LANG),
        tts_options_schema(config, voices),
    )


async def async_setup_entry(
    hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    """Set up Google Cloud text-to-speech."""
    service_account_info: Any = config_entry.data[CONF_SERVICE_ACCOUNT_INFO]
    client: texttospeech.TextToSpeechAsyncClient = texttospeech.TextToSpeechAsyncClient.from_service_account_info(service_account_info)
    try:
        voices: Dict[str, List[Any]] = await async_tts_voices(client)
    except GoogleAPIError as err:
        _LOGGER.error("Error from calling list_voices: %s", err)
        if isinstance(err, Unauthenticated):
            config_entry.async_start_reauth(hass)
        return
    options_schema = tts_options_schema(dict(config_entry.options), voices)
    language: str = config_entry.options.get(CONF_LANG, DEFAULT_LANG)
    async_add_entities([GoogleCloudTTSEntity(config_entry, client, voices, language, options_schema)])


class BaseGoogleCloudProvider:
    """The Google Cloud TTS base provider."""

    def __init__(
        self,
        client: texttospeech.TextToSpeechAsyncClient,
        voices: Dict[str, List[Any]],
        language: str,
        options_schema: vol.Schema,
    ) -> None:
        """Init Google Cloud TTS base provider."""
        self._client: texttospeech.TextToSpeechAsyncClient = client
        self._voices: Dict[str, List[Any]] = voices
        self._language: str = language
        self._options_schema: vol.Schema = options_schema

    @property
    def supported_languages(self) -> List[str]:
        """Return a list of supported languages."""
        return list(self._voices)

    @property
    def default_language(self) -> str:
        """Return the default language."""
        return self._language

    @property
    def supported_options(self) -> List[Any]:
        """Return a list of supported options."""
        return [option.schema for option in self._options_schema.schema]

    @property
    def default_options(self) -> Dict[str, Any]:
        """Return a dict including default options."""
        return cast(dict[str, Any], self._options_schema({}))

    @callback
    def async_get_supported_voices(self, language: str) -> Optional[List[Voice]]:
        """Return a list of supported voices for a language."""
        voices = self._voices.get(language)
        if not voices:
            return None
        return [Voice(voice, voice) for voice in voices]

    async def _async_get_tts_audio(
        self, message: str, language: str, options: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[bytes]]:
        """Load TTS from Google Cloud."""
        try:
            options = self._options_schema(options)
        except vol.Invalid as err:
            _LOGGER.error("Error: %s when validating options: %s", err, options)
            return (None, None)
        encoding = texttospeech.AudioEncoding[options[CONF_ENCODING]]
        gender = texttospeech.SsmlVoiceGender[options[CONF_GENDER]]
        voice = options[CONF_VOICE]
        if voice:
            gender = None  # type: ignore
            if not voice.startswith(language):
                language = voice[:5]
        request = texttospeech.SynthesizeSpeechRequest(
            input=texttospeech.SynthesisInput(**{options[CONF_TEXT_TYPE]: message}),
            voice=texttospeech.VoiceSelectionParams(
                language_code=language, ssml_gender=gender, name=voice
            ),
            audio_config=texttospeech.AudioConfig(
                audio_encoding=encoding,
                speaking_rate=options[CONF_SPEED] if options[CONF_SPEED] != DEFAULT_SPEED else None,
                pitch=options[CONF_PITCH] if options[CONF_PITCH] != DEFAULT_PITCH else None,
                volume_gain_db=options[CONF_GAIN] if options[CONF_GAIN] != DEFAULT_GAIN else None,
                effects_profile_id=options[CONF_PROFILES],
            ),
        )
        response = await self._client.synthesize_speech(request, timeout=10)
        if encoding == texttospeech.AudioEncoding.MP3:
            extension = "mp3"
        elif encoding == texttospeech.AudioEncoding.OGG_OPUS:
            extension = "ogg"
        else:
            extension = "wav"
        return (extension, response.audio_content)


class GoogleCloudTTSEntity(BaseGoogleCloudProvider, TextToSpeechEntity):
    """The Google Cloud TTS entity."""

    def __init__(
        self,
        entry: ConfigEntry,
        client: texttospeech.TextToSpeechAsyncClient,
        voices: Dict[str, List[Any]],
        language: str,
        options_schema: vol.Schema,
    ) -> None:
        """Init Google Cloud TTS entity."""
        super().__init__(client, voices, language, options_schema)
        self._attr_unique_id: str = f"{entry.entry_id}"
        self._attr_name: str = entry.title
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            manufacturer="Google",
            model="Cloud",
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        self._entry: ConfigEntry = entry

    async def async_get_tts_audio(
        self, message: str, language: str, options: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[bytes]]:
        """Load TTS from Google Cloud."""
        try:
            return await self._async_get_tts_audio(message, language, options)
        except GoogleAPIError as err:
            _LOGGER.error("Error occurred during Google Cloud TTS call: %s", err)
            if isinstance(err, Unauthenticated):
                self._entry.async_start_reauth(self.hass)
            return (None, None)


class GoogleCloudTTSProvider(BaseGoogleCloudProvider, Provider):
    """The Google Cloud TTS API provider."""

    def __init__(
        self,
        client: texttospeech.TextToSpeechAsyncClient,
        voices: Dict[str, List[Any]],
        language: str,
        options_schema: vol.Schema,
    ) -> None:
        """Init Google Cloud TTS service."""
        super().__init__(client, voices, language, options_schema)
        self.name: str = "Google Cloud TTS"

    async def async_get_tts_audio(
        self, message: str, language: str, options: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[bytes]]:
        """Load TTS from Google Cloud."""
        try:
            return await self._async_get_tts_audio(message, language, options)
        except GoogleAPIError as err:
            _LOGGER.error("Error occurred during Google Cloud TTS call: %s", err)
            return (None, None)