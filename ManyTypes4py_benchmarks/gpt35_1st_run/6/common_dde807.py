from __future__ import annotations
from collections.abc import AsyncIterable
from typing import Any, Callable, Coroutine
from homeassistant.components.stt import AudioBitRates, AudioChannels, AudioCodecs, AudioFormats, AudioSampleRates, Provider, SpeechMetadata, SpeechResult, SpeechResultState, SpeechToTextEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

TEST_DOMAIN: str = 'test'

class BaseProvider:
    fail_process_audio: bool = False

    def __init__(self, *, supported_languages: list[str] = None, text: str = 'test_result') -> None:
        self._supported_languages: list[str] = supported_languages or ['de', 'de-CH', 'en']
        self.calls: list[tuple[SpeechMetadata, AsyncIterable[Any]]] = []
        self.received: list[Any] = []
        self.text: str = text

    @property
    def supported_languages(self) -> list[str]:
        return self._supported_languages

    @property
    def supported_formats(self) -> list[AudioFormats]:
        return [AudioFormats.WAV, AudioFormats.OGG]

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        return [AudioCodecs.PCM, AudioCodecs.OPUS]

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        return [AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        return [AudioSampleRates.SAMPLERATE_16000]

    @property
    def supported_channels(self) -> list[AudioChannels]:
        return [AudioChannels.CHANNEL_MONO]

    async def async_process_audio_stream(self, metadata: SpeechMetadata, stream: AsyncIterable[Any]) -> SpeechResult:
        self.calls.append((metadata, stream))
        async for data in stream:
            if not data:
                break
            self.received.append(data)
        if self.fail_process_audio:
            return SpeechResult(None, SpeechResultState.ERROR)
        return SpeechResult(self.text, SpeechResultState.SUCCESS)

class MockSTTProvider(BaseProvider, Provider):
    url_path: str = TEST_DOMAIN

class MockSTTProviderEntity(BaseProvider, SpeechToTextEntity):
    url_path: str = 'stt.test'
    _attr_name: str = 'test'

class MockSTTPlatform(MockPlatform):
    def __init__(self, async_get_engine: Callable = None, get_engine: Callable = None) -> None:
        super().__init__()
        if get_engine:
            self.get_engine = get_engine
        if async_get_engine:
            self.async_get_engine = async_get_engine

def mock_stt_platform(hass: HomeAssistant, tmp_path: Path, integration: str = 'stt', async_get_engine: Callable = None, get_engine: Callable = None) -> MockSTTPlatform:
    loaded_platform = MockSTTPlatform(async_get_engine, get_engine)
    mock_platform(hass, f'{integration}.stt', loaded_platform)
    return loaded_platform

def mock_stt_entity_platform(hass: HomeAssistant, tmp_path: Path, integration: str, async_setup_entry: Callable = None) -> MockPlatform:
    loaded_platform = MockPlatform(async_setup_entry=async_setup_entry)
    mock_platform(hass, f'{integration}.stt', loaded_platform)
    return loaded_platform
