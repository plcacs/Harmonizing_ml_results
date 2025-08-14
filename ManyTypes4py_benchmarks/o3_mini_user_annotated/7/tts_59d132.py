from typing import Any, Dict, List, Optional, Tuple

import logging

from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import TextToSpeechV1
import voluptuous as vol

from homeassistant.components.tts import (
    PLATFORM_SCHEMA as TTS_PLATFORM_SCHEMA,
    Provider,
)
from homeassistant.helpers import config_validation as cv

_LOGGER = logging.getLogger(__name__)

CONF_URL = "watson_url"
CONF_APIKEY = "watson_apikey"

DEFAULT_URL = "https://api.us-south.text-to-speech.watson.cloud.ibm.com"

CONF_VOICE = "voice"
CONF_OUTPUT_FORMAT = "output_format"
CONF_TEXT_TYPE = "text"

# List from https://tinyurl.com/watson-tts-docs
SUPPORTED_VOICES: List[str] = [
    "de-DE_BirgitV2Voice",
    "de-DE_BirgitV3Voice",
    "de-DE_BirgitVoice",
    "de-DE_DieterV2Voice",
    "de-DE_DieterV3Voice",
    "de-DE_DieterVoice",
    "de-DE_ErikaV3Voice",
    "en-AU_HeidiExpressive",
    "en-AU_JackExpressive",
    "en-GB_CharlotteV3Voice",
    "en-GB_JamesV3Voice",
    "en-GB_KateV3Voice",
    "en-GB_KateVoice",
    "en-US_AllisonExpressive",
    "en-US_AllisonV2Voice",
    "en-US_AllisonV3Voice",
    "en-US_AllisonVoice",
    "en-US_EmilyV3Voice",
    "en-US_EmmaExpressive",
    "en-US_HenryV3Voice",
    "en-US_KevinV3Voice",
    "en-US_LisaExpressive",
    "en-US_LisaV2Voice",
    "en-US_LisaV3Voice",
    "en-US_LisaVoice",
    "en-US_MichaelExpressive",
    "en-US_MichaelV2Voice",
    "en-US_MichaelV3Voice",
    "en-US_MichaelVoice",
    "en-US_OliviaV3Voice",
    "es-ES_EnriqueV3Voice",
    "es-ES_EnriqueVoice",
    "es-ES_LauraV3Voice",
    "es-ES_LauraVoice",
    "es-LA_SofiaV3Voice",
    "es-LA_SofiaVoice",
    "es-US_SofiaV3Voice",
    "es-US_SofiaVoice",
    "fr-CA_LouiseV3Voice",
    "fr-FR_NicolasV3Voice",
    "fr-FR_ReneeV3Voice",
    "fr-FR_ReneeVoice",
    "it-IT_FrancescaV2Voice",
    "it-IT_FrancescaV3Voice",
    "it-IT_FrancescaVoice",
    "ja-JP_EmiV3Voice",
    "ja-JP_EmiVoice",
    "ko-KR_JinV3Voice",
    "nl-NL_MerelV3Voice",
    "pt-BR_IsabelaV3Voice",
    "pt-BR_IsabelaVoice",
]

DEPRECATED_VOICES: List[str] = [
    "de-DE_BirgitVoice",
    "de-DE_DieterVoice",
    "en-US_AllisonVoice",
    "en-US_LisaVoice",
    "en-US_MichaelVoice",
    "es-ES_EnriqueVoice",
    "es-ES_LauraVoice",
    "es-LA_SofiaVoice",
    "es-US_SofiaVoice",
    "fr-FR_ReneeVoice",
    "it-IT_FrancescaVoice",
    "ja-JP_EmiVoice",
    "pt-BR_IsabelaVoice",
]

SUPPORTED_OUTPUT_FORMATS: List[str] = [
    "audio/flac",
    "audio/mp3",
    "audio/mpeg",
    "audio/ogg",
    "audio/ogg;codecs=opus",
    "audio/ogg;codecs=vorbis",
    "audio/wav",
]

CONTENT_TYPE_EXTENSIONS: Dict[str, str] = {
    "audio/flac": "flac",
    "audio/mp3": "mp3",
    "audio/mpeg": "mp3",
    "audio/ogg": "ogg",
    "audio/ogg;codecs=opus": "ogg",
    "audio/ogg;codecs=vorbis": "ogg",
    "audio/wav": "wav",
}

DEFAULT_VOICE = "en-US_AllisonV3Voice"
DEFAULT_OUTPUT_FORMAT = "audio/mp3"

PLATFORM_SCHEMA = TTS_PLATFORM_SCHEMA.extend(
    {
        vol.Optional(CONF_URL, default=DEFAULT_URL): cv.string,
        vol.Required(CONF_APIKEY): cv.string,
        vol.Optional(CONF_VOICE, default=DEFAULT_VOICE): vol.In(SUPPORTED_VOICES),
        vol.Optional(CONF_OUTPUT_FORMAT, default=DEFAULT_OUTPUT_FORMAT): vol.In(
            SUPPORTED_OUTPUT_FORMATS
        ),
    }
)


def get_engine(hass: Any, config: Dict[str, Any], discovery_info: Optional[Any] = None) -> "WatsonTTSProvider":
    authenticator = IAMAuthenticator(config[CONF_APIKEY])
    service: TextToSpeechV1 = TextToSpeechV1(authenticator)
    service.set_service_url(config[CONF_URL])

    supported_languages: List[str] = list({s[:5] for s in SUPPORTED_VOICES})
    default_voice: str = config[CONF_VOICE]
    output_format: str = config[CONF_OUTPUT_FORMAT]
    service.set_default_headers({"x-watson-learning-opt-out": "true"})

    if default_voice in DEPRECATED_VOICES:
        _LOGGER.warning(
            "Watson TTS voice %s is deprecated, it may be removed in the future",
            default_voice,
        )

    return WatsonTTSProvider(service, supported_languages, default_voice, output_format)


class WatsonTTSProvider(Provider):
    def __init__(
        self,
        service: TextToSpeechV1,
        supported_languages: List[str],
        default_voice: str,
        output_format: str,
    ) -> None:
        self.service: TextToSpeechV1 = service
        self.supported_langs: List[str] = supported_languages
        self.default_lang: str = default_voice[:5]
        self.default_voice: str = default_voice
        self.output_format: str = output_format
        self.name: str = "Watson TTS"

    @property
    def supported_languages(self) -> List[str]:
        return self.supported_langs

    @property
    def default_language(self) -> str:
        return self.default_lang

    @property
    def default_options(self) -> Dict[str, str]:
        return {CONF_VOICE: self.default_voice}

    @property
    def supported_options(self) -> List[str]:
        return [CONF_VOICE]

    def get_tts_audio(
        self, message: str, language: str, options: Dict[str, Any]
    ) -> Tuple[str, bytes]:
        response = self.service.synthesize(
            text=message, accept=self.output_format, voice=options[CONF_VOICE]
        ).get_result()

        return CONTENT_TYPE_EXTENSIONS[self.output_format], response.content