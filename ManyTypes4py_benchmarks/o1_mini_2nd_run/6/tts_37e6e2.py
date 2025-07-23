"""Support for the Microsoft Cognitive Services text-to-speech service."""
import logging
from typing import Optional, Dict, Any, Tuple, List
from pycsspeechtts import pycsspeechtts
from requests.exceptions import HTTPError
import voluptuous as vol
from homeassistant.components.tts import CONF_LANG, PLATFORM_SCHEMA as TTS_PLATFORM_SCHEMA, Provider
from homeassistant.const import CONF_API_KEY, CONF_REGION, CONF_TYPE, PERCENTAGE
from homeassistant.generated.microsoft_tts import SUPPORTED_LANGUAGES
from homeassistant.helpers import config_validation as cv

CONF_GENDER = 'gender'
CONF_OUTPUT = 'output'
CONF_RATE = 'rate'
CONF_VOLUME = 'volume'
CONF_PITCH = 'pitch'
CONF_CONTOUR = 'contour'
_LOGGER = logging.getLogger(__name__)

GENDERS: List[str] = ['Female', 'Male']
DEFAULT_LANG: str = 'en-us'
DEFAULT_GENDER: str = 'Female'
DEFAULT_TYPE: str = 'JennyNeural'
DEFAULT_OUTPUT: str = 'audio-24khz-96kbitrate-mono-mp3'
DEFAULT_RATE: int = 0
DEFAULT_VOLUME: int = 0
DEFAULT_PITCH: str = 'default'
DEFAULT_CONTOUR: str = ''
DEFAULT_REGION: str = 'eastus'

PLATFORM_SCHEMA = TTS_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_API_KEY): cv.string,
    vol.Optional(CONF_LANG, default=DEFAULT_LANG): vol.In(SUPPORTED_LANGUAGES),
    vol.Optional(CONF_GENDER, default=DEFAULT_GENDER): vol.In(GENDERS),
    vol.Optional(CONF_TYPE, default=DEFAULT_TYPE): cv.string,
    vol.Optional(CONF_RATE, default=DEFAULT_RATE): vol.All(vol.Coerce(int), vol.Range(-100, 100)),
    vol.Optional(CONF_VOLUME, default=DEFAULT_VOLUME): vol.All(vol.Coerce(int), vol.Range(-100, 100)),
    vol.Optional(CONF_PITCH, default=DEFAULT_PITCH): cv.string,
    vol.Optional(CONF_CONTOUR, default=DEFAULT_CONTOUR): cv.string,
    vol.Optional(CONF_REGION, default=DEFAULT_REGION): cv.string
})

def get_engine(hass: Any, config: Dict[str, Any], discovery_info: Optional[Any] = None) -> Provider:
    """Set up Microsoft speech component."""
    return MicrosoftProvider(
        config[CONF_API_KEY],
        config[CONF_LANG],
        config[CONF_GENDER],
        config[CONF_TYPE],
        config[CONF_RATE],
        config[CONF_VOLUME],
        config[CONF_PITCH],
        config[CONF_CONTOUR],
        config[CONF_REGION]
    )

class MicrosoftProvider(Provider):
    """The Microsoft speech API provider."""

    def __init__(
        self, 
        apikey: str, 
        lang: str, 
        gender: str, 
        ttype: str, 
        rate: int, 
        volume: int, 
        pitch: str, 
        contour: str, 
        region: str
    ) -> None:
        """Init Microsoft TTS service."""
        self._apikey: str = apikey
        self._lang: str = lang
        self._gender: str = gender
        self._type: str = ttype
        self._output: str = DEFAULT_OUTPUT
        self._rate: str = f'{rate}{PERCENTAGE}'
        self._volume: str = f'{volume}{PERCENTAGE}'
        self._pitch: str = pitch
        self._contour: str = contour
        self._region: str = region
        self.name: str = 'Microsoft'

    @property
    def default_language(self) -> str:
        """Return the default language."""
        return self._lang

    @property
    def supported_languages(self) -> List[str]:
        """Return list of supported languages."""
        return SUPPORTED_LANGUAGES

    @property
    def supported_options(self) -> List[str]:
        """Return list of supported options like voice, emotion."""
        return [CONF_GENDER, CONF_TYPE]

    @property
    def default_options(self) -> Dict[str, str]:
        """Return a dict include default options."""
        return {CONF_GENDER: self._gender, CONF_TYPE: self._type}

    def get_tts_audio(
        self, 
        message: str, 
        language: Optional[str], 
        options: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[bytes]]:
        """Load TTS from Microsoft."""
        if language is None:
            language = self._lang
        try:
            trans = pycsspeechtts.TTSTranslator(self._apikey, self._region)
            data: bytes = trans.speak(
                language=language,
                gender=options[CONF_GENDER],
                voiceType=options[CONF_TYPE],
                output=self._output,
                rate=self._rate,
                volume=self._volume,
                pitch=self._pitch,
                contour=self._contour,
                text=message
            )
        except HTTPError as ex:
            _LOGGER.error('Error occurred for Microsoft TTS: %s', ex)
            return (None, None)
        return ('mp3', data)
