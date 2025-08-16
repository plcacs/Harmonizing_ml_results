from ipaddress import IPv4Address
from typing import Any, List, Dict, Coroutine, Generator, Callable

class SonosMockEventListener:
    def __init__(self, ip_address: str):
        self.address: List[str] = [ip_address, '8080']

class SonosMockSubscribe:
    def __init__(self, ip_address: str, *args: Any, **kwargs: Any):
        self.event_listener: SonosMockEventListener = SonosMockEventListener(ip_address)
        self.service: Any = Mock()
        self.callback_future: Any = None
        self._callback: Any = None

    @property
    def callback(self) -> Any:
        return self._callback

    @callback.setter
    def callback(self, callback: Any) -> None:
        self._callback = callback
        future = self._get_callback_future()
        if not future.done():
            future.set_result(callback)

    def _get_callback_future(self) -> Any:
        if not self.callback_future:
            self.callback_future = asyncio.get_running_loop().create_future()
        return self.callback_future

    async def wait_for_callback_to_be_set(self) -> Any:
        return await self._get_callback_future()

    async def unsubscribe(self) -> None:
        pass

class SonosMockService:
    def __init__(self, service_type: str, ip_address: str = '192.168.42.2'):
        self.service_type: str = service_type
        self.subscribe: Any = AsyncMock(return_value=SonosMockSubscribe(ip_address))

class SonosMockEvent:
    def __init__(self, soco: Any, service: Any, variables: Dict[str, str]):
        self.sid: str = f'{soco.uid}_sub0000000001'
        self.seq: str = '0'
        self.timestamp: float = 1621000000.0
        self.service: Any = service
        self.variables: Dict[str, str] = variables

    def increment_variable(self, var_name: str) -> str:
        self.variables = copy(self.variables)
        base, count = self.variables[var_name].split(':')
        newcount = int(count) + 1
        self.variables[var_name] = ':'.join([base, str(newcount)])
        return self.variables[var_name]

def zeroconf_payload() -> ZeroconfServiceInfo:
    return ZeroconfServiceInfo(ip_address=IPv4Address('192.168.4.2'), ip_addresses=[IPv4Address('192.168.4.2')], hostname='Sonos-aaa', name='Sonos-aaa@Living Room._sonos._tcp.local.', port=None, properties={'bootseq': '1234'}, type='mock_type')

async def async_autosetup_sonos(async_setup_sonos: Coroutine) -> None:
    await async_setup_sonos()

def async_setup_sonos(hass: HomeAssistant, config_entry: MockConfigEntry, fire_zgs_event: Callable) -> Coroutine:
    async def _wrapper() -> None:
        config_entry.add_to_hass(hass)
        sonos_alarms = Alarms()
        sonos_alarms.last_alarm_list_version = 'RINCON_test:0'
        assert await hass.config_entries.async_setup(config_entry.entry_id)
        await hass.async_block_till_done(wait_background_tasks=True)
        await fire_zgs_event()
        await hass.async_block_till_done(wait_background_tasks=True)
    return _wrapper

def config_entry_fixture() -> MockConfigEntry:
    return MockConfigEntry(domain=DOMAIN, title='Sonos')

class MockSoCo(MagicMock):
    uid: str = 'RINCON_test'
    play_mode: str = 'NORMAL'
    mute: bool = False
    night_mode: bool = True
    dialog_level: bool = True
    loudness: bool = True
    volume: int = 19
    audio_delay: int = 2
    balance: Tuple[int, int] = (61, 100)
    bass: int = 1
    treble: int = -1
    mic_enabled: bool = False
    sub_crossover: Any = None
    sub_enabled: bool = False
    sub_gain: int = 5
    surround_enabled: bool = True
    surround_mode: bool = True
    surround_level: int = 3
    music_surround_level: int = 4
    soundbar_audio_input_format: str = 'Dolby 5.1'

    @property
    def visible_zones(self) -> Dict[str, Any]:
        return {self}

class SoCoMockFactory:
    def __init__(self, music_library: Any, speaker_info: Dict[str, str], current_track_info_empty: Dict[str, str], battery_info: Dict[str, Any], alarm_clock: Any, sonos_playlists: Any, sonos_queue: Any):
        self.mock_list: Dict[str, Any] = {}
        self.music_library: Any = music_library
        self.speaker_info: Dict[str, str] = speaker_info
        self.current_track_info: Dict[str, str] = current_track_info_empty
        self.battery_info: Dict[str, Any] = battery_info
        self.alarm_clock: Any = alarm_clock
        self.sonos_playlists: Any = sonos_playlists
        self.sonos_queue: Any = sonos_queue

    def cache_mock(self, mock_soco: MagicMock, ip_address: str, name: str = 'Zone A') -> MagicMock:
        mock_soco.mock_add_spec(SoCo)
        mock_soco.ip_address: str = ip_address
        if ip_address != '192.168.42.2':
            mock_soco.uid += f'_{ip_address}'
        mock_soco.music_library: Any = self.music_library
        mock_soco.get_current_track_info.return_value = self.current_track_info
        mock_soco.music_source_from_uri = SoCo.music_source_from_uri
        mock_soco.get_sonos_playlists.return_value = self.sonos_playlists
        mock_soco.get_queue.return_value = self.sonos_queue
        my_speaker_info: Dict[str, str] = self.speaker_info.copy()
        my_speaker_info['zone_name'] = name
        my_speaker_info['uid'] = mock_soco.uid
        mock_soco.get_speaker_info = Mock(return_value=my_speaker_info)
        mock_soco.add_to_queue = Mock(return_value=10)
        mock_soco.add_uri_to_queue = Mock(return_value=10)
        mock_soco.avTransport = SonosMockService('AVTransport', ip_address)
        mock_soco.renderingControl = SonosMockService('RenderingControl', ip_address)
        mock_soco.zoneGroupTopology = SonosMockService('ZoneGroupTopology', ip_address)
        mock_soco.contentDirectory = SonosMockService('ContentDirectory', ip_address)
        mock_soco.deviceProperties = SonosMockService('DeviceProperties', ip_address)
        mock_soco.alarmClock = self.alarm_clock
        mock_soco.get_battery_info.return_value = self.battery_info
        mock_soco.all_zones = {mock_soco}
        mock_soco.group.coordinator = mock_soco
        self.mock_list[ip_address] = mock_soco
        return mock_soco

    def get_mock(self, *args: Any) -> MagicMock:
        if len(args) > 0:
            ip_address: str = args[0]
        else:
            ip_address: str = '192.168.42.2'
        if ip_address in self.mock_list:
            return self.mock_list[ip_address]
        mock_soco: MagicMock = MockSoCo(name=f'Soco Mock {ip_address}')
        self.cache_mock(mock_soco, ip_address)
        return mock_soco

def patch_gethostbyname(host: str) -> str:
    return host
