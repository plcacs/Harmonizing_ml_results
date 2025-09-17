#!/usr/bin/env python3
"""Configuration for Sonos tests."""
import asyncio
from collections.abc import Callable, Coroutine, Generator
from copy import copy
from ipaddress import ip_address, IPv4Address
from typing import Any, Awaitable, Dict, List, Optional, Union

import pytest
from soco import SoCo
from soco.alarms import Alarms
from soco.data_structures import DidlFavorite, DidlMusicTrack, DidlPlaylistContainer, SearchResult
from soco.events_base import Event as SonosEvent
from homeassistant.components import ssdp
from homeassistant.components.media_player import DOMAIN as MP_DOMAIN
from homeassistant.components.sonos import DOMAIN
from homeassistant.const import CONF_HOSTS
from homeassistant.core import HomeAssistant
from homeassistant.helpers.service_info.ssdp import ATTR_UPNP_UDN, SsdpServiceInfo
from homeassistant.helpers.service_info.zeroconf import ZeroconfServiceInfo
from homeassistant.setup import async_setup_component
from tests.common import MockConfigEntry, load_fixture, load_json_value_fixture
from unittest.mock import AsyncMock, MagicMock, Mock, patch


class SonosMockEventListener:
    """Mock the event listener."""

    def __init__(self, ip_address: str) -> None:
        """Initialize the mock event listener."""
        self.address: List[str] = [ip_address, '8080']


class SonosMockSubscribe:
    """Mock the subscription."""

    def __init__(self, ip_address: str, *args: Any, **kwargs: Any) -> None:
        """Initialize the mock subscriber."""
        self.event_listener: SonosMockEventListener = SonosMockEventListener(ip_address)
        self.service: MagicMock = Mock()
        self.callback_future: Optional[asyncio.Future[Callable[..., Any]]] = None
        self._callback: Optional[Callable[..., Any]] = None

    @property
    def callback(self) -> Optional[Callable[..., Any]]:
        """Return the callback."""
        return self._callback

    @callback.setter
    def callback(self, callback: Callable[..., Any]) -> None:
        """Set the callback."""
        self._callback = callback
        future: asyncio.Future[Callable[..., Any]] = self._get_callback_future()
        if not future.done():
            future.set_result(callback)

    def _get_callback_future(self) -> asyncio.Future[Callable[..., Any]]:
        """Get the callback future."""
        if not self.callback_future:
            self.callback_future = asyncio.get_running_loop().create_future()
        return self.callback_future

    async def wait_for_callback_to_be_set(self) -> Callable[..., Any]:
        """Wait for the callback to be set."""
        return await self._get_callback_future()

    async def unsubscribe(self) -> None:
        """Unsubscribe mock."""
        pass


class SonosMockService:
    """Mock a Sonos Service used in callbacks."""

    def __init__(self, service_type: str, ip_address: str = '192.168.42.2') -> None:
        """Initialize the instance."""
        self.service_type: str = service_type
        self.subscribe: AsyncMock = AsyncMock(return_value=SonosMockSubscribe(ip_address))


class SonosMockEvent:
    """Mock a Sonos Event used in callbacks."""

    def __init__(self, soco: SoCo, service: Any, variables: Dict[str, Any]) -> None:
        """Initialize the instance."""
        self.sid: str = f'{soco.uid}_sub0000000001'
        self.seq: str = '0'
        self.timestamp: float = 1621000000.0
        self.service: Any = service
        self.variables: Dict[str, Any] = variables

    def increment_variable(self, var_name: str) -> str:
        """Increment the value of the var_name key in variables dict attribute.

        Assumes value has a format of <str>:<int>.
        """
        self.variables = copy(self.variables)
        base, count = self.variables[var_name].split(':')
        newcount: int = int(count) + 1
        self.variables[var_name] = ':'.join([base, str(newcount)])
        return self.variables[var_name]


@pytest.fixture
def zeroconf_payload() -> ZeroconfServiceInfo:
    """Return a default zeroconf payload."""
    return ZeroconfServiceInfo(
        ip_address=ip_address('192.168.4.2'),
        ip_addresses=[ip_address('192.168.4.2')],
        hostname='Sonos-aaa',
        name='Sonos-aaa@Living Room._sonos._tcp.local.',
        port=None,
        properties={'bootseq': '1234'},
        type='mock_type',
    )


@pytest.fixture
async def async_autosetup_sonos(async_setup_sonos: Callable[[], Awaitable[None]]) -> None:
    """Set up a Sonos integration instance on test run."""
    await async_setup_sonos()


@pytest.fixture
def async_setup_sonos(hass: HomeAssistant, config_entry: MockConfigEntry, fire_zgs_event: Callable[[], Awaitable[None]]
                     ) -> Callable[[], Awaitable[None]]:
    """Return a coroutine to set up a Sonos integration instance on demand."""

    async def _wrapper() -> None:
        config_entry.add_to_hass(hass)
        sonos_alarms: Alarms = Alarms()
        sonos_alarms.last_alarm_list_version = 'RINCON_test:0'
        assert await hass.config_entries.async_setup(config_entry.entry_id)  # type: ignore
        await hass.async_block_till_done(wait_background_tasks=True)
        await fire_zgs_event()
        await hass.async_block_till_done(wait_background_tasks=True)

    return _wrapper


@pytest.fixture(name='config_entry')
def config_entry_fixture() -> MockConfigEntry:
    """Create a mock Sonos config entry."""
    return MockConfigEntry(domain=DOMAIN, title='Sonos')


class MockSoCo(MagicMock):
    """Mock the Soco Object."""
    uid: str = 'RINCON_test'
    play_mode: str = 'NORMAL'
    mute: bool = False
    night_mode: bool = True
    dialog_level: bool = True
    loudness: bool = True
    volume: int = 19
    audio_delay: int = 2
    balance: tuple[int, int] = (61, 100)
    bass: int = 1
    treble: int = -1
    mic_enabled: bool = False
    sub_crossover: Optional[Any] = None
    sub_enabled: bool = False
    sub_gain: int = 5
    surround_enabled: bool = True
    surround_mode: bool = True
    surround_level: int = 3
    music_surround_level: int = 4
    soundbar_audio_input_format: str = 'Dolby 5.1'

    @property
    def visible_zones(self) -> set:
        """Return visible zones and allow property to be overridden by device classes."""
        return {self}


class SoCoMockFactory:
    """Factory for creating SoCo Mocks."""

    def __init__(
        self,
        music_library: Any,
        speaker_info: Dict[str, Any],
        current_track_info_empty: Dict[str, Any],
        battery_info: Dict[str, Any],
        alarm_clock: Any,
        sonos_playlists: Any,
        sonos_queue: List[Any],
    ) -> None:
        """Initialize the mock factory."""
        self.mock_list: Dict[str, Any] = {}
        self.music_library: Any = music_library
        self.speaker_info: Dict[str, Any] = speaker_info
        self.current_track_info: Dict[str, Any] = current_track_info_empty
        self.battery_info: Dict[str, Any] = battery_info
        self.alarm_clock: Any = alarm_clock
        self.sonos_playlists: Any = sonos_playlists
        self.sonos_queue: List[Any] = sonos_queue

    def cache_mock(self, mock_soco: MagicMock, ip_address: str, name: str = 'Zone A') -> MagicMock:
        """Put a user created mock into the cache."""
        mock_soco.mock_add_spec(SoCo)
        mock_soco.ip_address = ip_address
        if ip_address != '192.168.42.2':
            mock_soco.uid += f'_{ip_address}'
        mock_soco.music_library = self.music_library
        mock_soco.get_current_track_info.return_value = self.current_track_info
        mock_soco.music_source_from_uri = SoCo.music_source_from_uri
        mock_soco.get_sonos_playlists.return_value = self.sonos_playlists
        mock_soco.get_queue.return_value = self.sonos_queue
        my_speaker_info: Dict[str, Any] = self.speaker_info.copy()
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
        """Return a mock."""
        if len(args) > 0:
            ip_address: str = args[0]
        else:
            ip_address = '192.168.42.2'
        if ip_address in self.mock_list:
            return self.mock_list[ip_address]
        mock_soco: MagicMock = MockSoCo(name=f'Soco Mock {ip_address}')
        self.cache_mock(mock_soco, ip_address)
        return mock_soco


def patch_gethostbyname(host: str) -> str:
    """Mock to return host name as ip address for testing."""
    return host


@pytest.fixture(name='soco_sharelink')
def soco_sharelink() -> MagicMock:
    """Fixture to mock soco.plugins.sharelink.ShareLinkPlugin."""
    with patch('homeassistant.components.sonos.speaker.ShareLinkPlugin') as mock_share:
        mock_instance: MagicMock = MagicMock()
        mock_instance.is_share_link.return_value = True
        mock_instance.add_share_link_to_queue.return_value = 10
        mock_share.return_value = mock_instance
        yield mock_instance


@pytest.fixture(name='sonos_websocket')
def sonos_websocket() -> AsyncMock:
    """Fixture to mock SonosWebSocket."""
    with patch('homeassistant.components.sonos.speaker.SonosWebsocket') as mock_sonos_ws:
        mock_instance: AsyncMock = AsyncMock()
        mock_instance.play_clip = AsyncMock()
        mock_instance.play_clip.return_value = [{'success': 1}, {}]
        mock_sonos_ws.return_value = mock_instance
        yield mock_instance


@pytest.fixture(name='soco_factory')
def soco_factory(
    music_library: Any,
    speaker_info: Dict[str, Any],
    current_track_info_empty: Dict[str, Any],
    battery_info: Dict[str, Any],
    alarm_clock: Any,
    sonos_playlists: Any,
    sonos_websocket: AsyncMock,
    sonos_queue: List[Any],
) -> Generator[SoCoMockFactory, None, None]:
    """Create factory for instantiating SoCo mocks."""
    factory: SoCoMockFactory = SoCoMockFactory(
        music_library, speaker_info, current_track_info_empty, battery_info, alarm_clock, sonos_playlists, sonos_queue=sonos_queue
    )
    with patch('homeassistant.components.sonos.SoCo', new=factory.get_mock), patch('socket.gethostbyname', side_effect=patch_gethostbyname), patch('homeassistant.components.sonos.ZGS_SUBSCRIPTION_TIMEOUT', 0):
        yield factory


@pytest.fixture(name='soco')
def soco_fixture(soco_factory: SoCoMockFactory) -> MagicMock:
    """Create a default mock soco SoCo fixture."""
    return soco_factory.get_mock()


@pytest.fixture(autouse=True)
def silent_ssdp_scanner() -> Generator[None, None, None]:
    """Start SSDP component and get Scanner, prevent actual SSDP traffic."""
    with patch('homeassistant.components.ssdp.Scanner._async_start_ssdp_listeners'), patch('homeassistant.components.ssdp.Scanner._async_stop_ssdp_listeners'), patch('homeassistant.components.ssdp.Scanner.async_scan'), patch('homeassistant.components.ssdp.Server._async_start_upnp_servers'), patch('homeassistant.components.ssdp.Server._async_stop_upnp_servers'):
        yield


@pytest.fixture(name='discover', autouse=True)
def discover_fixture(soco: MagicMock) -> Generator[MagicMock, None, None]:
    """Create a mock soco discover fixture."""

    def do_callback(hass: HomeAssistant, callback: Callable[[SsdpServiceInfo, Any], None], match_dict: Optional[Dict[str, Any]] = None) -> MagicMock:
        callback(
            SsdpServiceInfo(
                ssdp_location=f'http://{soco.ip_address}/',
                ssdp_st='urn:schemas-upnp-org:device:ZonePlayer:1',
                ssdp_usn=f'uuid:{soco.uid}_MR::urn:schemas-upnp-org:service:GroupRenderingControl:1',
                upnp={ATTR_UPNP_UDN: f'uuid:{soco.uid}'},
            ),
            ssdp.SsdpChange.ALIVE,
        )
        return MagicMock()

    with patch('homeassistant.components.ssdp.async_register_callback', side_effect=do_callback) as mock_:
        yield mock_


@pytest.fixture(name='config')
def config_fixture() -> Dict[str, Any]:
    """Create hass config fixture."""
    return {DOMAIN: {MP_DOMAIN: {CONF_HOSTS: ['192.168.42.2']}}}


@pytest.fixture(name='sonos_favorites')
def sonos_favorites_fixture() -> SearchResult:
    """Create sonos favorites fixture."""
    favorites: List[Dict[str, Any]] = load_json_value_fixture('sonos_favorites.json', 'sonos')
    favorite_list: List[DidlFavorite] = [DidlFavorite.from_dict(fav) for fav in favorites]
    return SearchResult(favorite_list, 'favorites', 3, 3, 1)


@pytest.fixture(name='sonos_playlists')
def sonos_playlists_fixture() -> SearchResult:
    """Create sonos playlist fixture."""
    playlists: List[Dict[str, Any]] = load_json_value_fixture('sonos_playlists.json', 'sonos')
    playlists_list: List[DidlPlaylistContainer] = [DidlPlaylistContainer.from_dict(pl) for pl in playlists]
    return SearchResult(playlists_list, 'sonos_playlists', 1, 1, 0)


@pytest.fixture(name='sonos_queue')
def sonos_queue() -> List[DidlMusicTrack]:
    """Create sonos queue fixture."""
    queue: List[Dict[str, Any]] = load_json_value_fixture('sonos_queue.json', 'sonos')
    return [DidlMusicTrack.from_dict(track) for track in queue]


class MockMusicServiceItem:
    """Mocks a Soco MusicServiceItem."""

    def __init__(self, title: str, item_id: str, parent_id: str, item_class: str, album_art_uri: Optional[str] = None) -> None:
        """Initialize the mock item."""
        self.title: str = title
        self.item_id: str = item_id
        self.item_class: str = item_class
        self.parent_id: str = parent_id
        self.album_art_uri: Optional[str] = album_art_uri


def list_from_json_fixture(file_name: str) -> List[MockMusicServiceItem]:
    """Create a list of music service items from a json fixture file."""
    item_list: List[Dict[str, Any]] = load_json_value_fixture(file_name, 'sonos')
    return [
        MockMusicServiceItem(
            item.get('title'),
            item.get('item_id'),
            item.get('parent_id'),
            item.get('item_class'),
            item.get('album_art_uri'),
        )
        for item in item_list
    ]


def mock_browse_by_idstring(
    search_type: str,
    idstring: str,
    start: int = 0,
    max_items: int = 100,
    full_album_art_uri: bool = False,
) -> List[MockMusicServiceItem]:
    """Mock the call to browse_by_id_string."""
    if search_type == 'album_artists' and idstring == 'A:ALBUMARTIST/Beatles':
        return [
            MockMusicServiceItem('All', idstring + '/', idstring, 'object.container.playlistContainer.sameArtist'),
            MockMusicServiceItem("A Hard Day's Night", "A:ALBUMARTIST/Beatles/A%20Hard%20Day's%20Night", idstring, 'object.container.album.musicAlbum'),
            MockMusicServiceItem('Abbey Road', 'A:ALBUMARTIST/Beatles/Abbey%20Road', idstring, 'object.container.album.musicAlbum')
        ]
    if search_type == 'genres' and idstring in ('A:GENRE/Classic%20Rock', 'A:GENRE/Classic Rock'):
        return [
            MockMusicServiceItem('All', 'A:GENRE/Classic%20Rock/', 'A:GENRE/Classic%20Rock', 'object.container.albumlist'),
            MockMusicServiceItem('Bruce Springsteen', 'A:GENRE/Classic%20Rock/Bruce%20Springsteen', 'A:GENRE/Classic%20Rock', 'object.container.person.musicArtist'),
            MockMusicServiceItem('Cream', 'A:GENRE/Classic%20Rock/Cream', 'A:GENRE/Classic%20Rock', 'object.container.person.musicArtist')
        ]
    if search_type == 'composers' and idstring in ('A:COMPOSER/Carlos%20Santana', 'A:COMPOSER/Carlos Santana'):
        return [
            MockMusicServiceItem('All', 'A:COMPOSER/Carlos%20Santana/', 'A:COMPOSER/Carlos%20Santana', 'object.container.playlistContainer.sameArtist'),
            MockMusicServiceItem('Between Good And Evil', 'A:COMPOSER/Carlos%20Santana/Between%20Good%20And%20Evil', 'A:COMPOSER/Carlos%20Santana', 'object.container.album.musicAlbum'),
            MockMusicServiceItem('Sacred Fire', 'A:COMPOSER/Carlos%20Santana/Sacred%20Fire', 'A:COMPOSER/Carlos%20Santana', 'object.container.album.musicAlbum')
        ]
    if search_type == 'tracks':
        return list_from_json_fixture('music_library_tracks.json')
    if search_type == 'albums' and idstring == 'A:ALBUM':
        return list_from_json_fixture('music_library_albums.json')
    return []


def mock_get_music_library_information(
    search_type: str, search_term: str, full_album_art_uri: bool = True
) -> List[MockMusicServiceItem]:
    """Mock the call to get music library information."""
    if search_type == 'albums' and search_term == 'Abbey Road':
        return [MockMusicServiceItem('Abbey Road', 'A:ALBUM/Abbey%20Road', 'A:ALBUM', 'object.container.album.musicAlbum')]
    return []


@pytest.fixture(name='music_library_browse_categories')
def music_library_browse_categories() -> List[MockMusicServiceItem]:
    """Create fixture for top-level music library categories."""
    return list_from_json_fixture('music_library_categories.json')


@pytest.fixture(name='music_library')
def music_library_fixture(sonos_favorites: SearchResult, music_library_browse_categories: List[MockMusicServiceItem]) -> Any:
    """Create music_library fixture."""
    music_library: MagicMock = MagicMock()
    music_library.get_sonos_favorites.return_value = sonos_favorites
    music_library.browse_by_idstring = Mock(side_effect=mock_browse_by_idstring)
    music_library.get_music_library_information = mock_get_music_library_information
    music_library.browse = Mock(return_value=music_library_browse_categories)
    return music_library


@pytest.fixture(name='alarm_clock')
def alarm_clock_fixture() -> SonosMockService:
    """Create alarmClock fixture."""
    alarm_clock: SonosMockService = SonosMockService('AlarmClock')
    alarm_clock.ListAlarms = Mock()
    alarm_clock.ListAlarms.return_value = {
        'CurrentAlarmListVersion': 'RINCON_test:14',
        'CurrentAlarmList': '<Alarms><Alarm ID="14" StartTime="07:00:00" Duration="02:00:00" Recurrence="DAILY" Enabled="1" RoomUUID="RINCON_test" ProgramURI="x-rincon-buzzer:0" ProgramMetaData="" PlayMode="SHUFFLE_NOREPEAT" Volume="25" IncludeLinkedZones="0"/></Alarms>'
    }
    return alarm_clock


@pytest.fixture(name='alarm_clock_extended')
def alarm_clock_fixture_extended() -> SonosMockService:
    """Create alarmClock fixture."""
    alarm_clock: SonosMockService = SonosMockService('AlarmClock')
    alarm_clock.ListAlarms = Mock()
    alarm_clock.ListAlarms.return_value = {
        'CurrentAlarmListVersion': 'RINCON_test:15',
        'CurrentAlarmList': '<Alarms><Alarm ID="14" StartTime="07:00:00" Duration="02:00:00" Recurrence="DAILY" Enabled="1" RoomUUID="RINCON_test" ProgramURI="x-rincon-buzzer:0" ProgramMetaData="" PlayMode="SHUFFLE_NOREPEAT" Volume="25" IncludeLinkedZones="0"/><Alarm ID="15" StartTime="07:00:00" Duration="02:00:00" Recurrence="DAILY" Enabled="1" RoomUUID="RINCON_test" ProgramURI="x-rincon-buzzer:0" ProgramMetaData="" PlayMode="SHUFFLE_NOREPEAT" Volume="25" IncludeLinkedZones="0"/></Alarms>'
    }
    return alarm_clock


@pytest.fixture(name='speaker_info')
def speaker_info_fixture() -> Dict[str, str]:
    """Create speaker_info fixture."""
    return {
        'zone_name': 'Zone A',
        'uid': 'RINCON_test',
        'model_name': 'Model Name',
        'model_number': 'S12',
        'hardware_version': '1.20.1.6-1.1',
        'software_version': '49.2-64250',
        'mac_address': '00-11-22-33-44-55',
        'display_version': '13.1',
    }


@pytest.fixture(name='current_track_info_empty')
def current_track_info_empty_fixture() -> Dict[str, Any]:
    """Create current_track_info_empty fixture."""
    return {
        'title': '',
        'artist': '',
        'album': '',
        'album_art': '',
        'position': 'NOT_IMPLEMENTED',
        'playlist_position': '1',
        'duration': 'NOT_IMPLEMENTED',
        'uri': '',
        'metadata': 'NOT_IMPLEMENTED',
    }


@pytest.fixture(name='battery_info')
def battery_info_fixture() -> Dict[str, Any]:
    """Create battery_info fixture."""
    return {
        'Health': 'GREEN',
        'Level': 100,
        'Temperature': 'NORMAL',
        'PowerSource': 'SONOS_CHARGING_RING',
    }


@pytest.fixture(name='device_properties_event')
def device_properties_event_fixture(soco: MagicMock) -> SonosMockEvent:
    """Create device_properties_event fixture."""
    variables: Dict[str, Any] = {
        'zone_name': 'Zone A',
        'mic_enabled': '1',
        'more_info': 'BattChg:NOT_CHARGING,RawBattPct:100,BattPct:100,BattTmp:25',
    }
    return SonosMockEvent(soco, soco.deviceProperties, variables)


@pytest.fixture(name='alarm_event')
def alarm_event_fixture(soco: MagicMock) -> SonosMockEvent:
    """Create alarm_event fixture."""
    variables: Dict[str, Any] = {
        'time_zone': 'ffc40a000503000003000502ffc4',
        'time_server': '0.sonostime.pool.ntp.org,1.sonostime.pool.ntp.org,2.sonostime.pool.ntp.org,3.sonostime.pool.ntp.org',
        'time_generation': '20000001',
        'alarm_list_version': 'RINCON_test:1',
        'time_format': 'INV',
        'date_format': 'INV',
        'daily_index_refresh_time': None,
    }
    return SonosMockEvent(soco, soco.alarmClock, variables)


@pytest.fixture(name='no_media_event')
def no_media_event_fixture(soco: MagicMock) -> SonosMockEvent:
    """Create no_media_event_fixture."""
    variables: Dict[str, Any] = {
        'current_crossfade_mode': '0',
        'current_play_mode': 'NORMAL',
        'current_section': '0',
        'current_track_meta_data': '',
        'current_track_uri': '',
        'enqueued_transport_uri': '',
        'enqueued_transport_uri_meta_data': '',
        'number_of_tracks': '0',
        'transport_state': 'STOPPED',
    }
    return SonosMockEvent(soco, soco.avTransport, variables)


@pytest.fixture(name='tv_event')
def tv_event_fixture(soco: MagicMock) -> SonosMockEvent:
    """Create tv_event fixture."""
    variables: Dict[str, Any] = {
        'transport_state': 'PLAYING',
        'current_play_mode': 'NORMAL',
        'current_crossfade_mode': '0',
        'number_of_tracks': '1',
        'current_track': '1',
        'current_section': '0',
        'current_track_uri': f'x-sonos-htastream:{soco.uid}:spdif',
        'current_track_duration': '',
        'current_track_meta_data': {
            'title': ' ',
            'parent_id': '-1',
            'item_id': '-1',
            'restricted': True,
            'resources': [],
            'desc': None,
        },
        'next_track_uri': '',
        'next_track_meta_data': '',
        'enqueued_transport_uri': '',
        'enqueued_transport_uri_meta_data': '',
        'playback_storage_medium': 'NETWORK',
        'av_transport_uri': f'x-sonos-htastream:{soco.uid}:spdif',
        'av_transport_uri_meta_data': {
            'title': soco.uid,
            'parent_id': '0',
            'item_id': 'spdif-input',
            'restricted': False,
            'resources': [],
            'desc': None,
        },
        'current_transport_actions': 'Set, Play',
        'current_valid_play_modes': '',
    }
    return SonosMockEvent(soco, soco.avTransport, variables)


@pytest.fixture(name='zgs_discovery', scope='package')
def zgs_discovery_fixture() -> str:
    """Load ZoneGroupState discovery payload and return it."""
    return load_fixture('sonos/zgs_discovery.xml')


@pytest.fixture(name='fire_zgs_event')
def zgs_event_fixture(hass: HomeAssistant, soco: MagicMock, zgs_discovery: str) -> Callable[[], Awaitable[None]]:
    """Create fire_zgs_event fixture."""
    variables: Dict[str, Any] = {'ZoneGroupState': zgs_discovery}

    async def _wrapper() -> None:
        event: SonosMockEvent = SonosMockEvent(soco, soco.zoneGroupTopology, variables)
        subscription: Any = soco.zoneGroupTopology.subscribe.return_value
        sub_callback: Callable[..., Any] = await subscription.wait_for_callback_to_be_set()
        sub_callback(event)
        await hass.async_block_till_done(wait_background_tasks=True)

    return _wrapper


@pytest.fixture(name='sonos_setup_two_speakers')
async def sonos_setup_two_speakers(hass: HomeAssistant, soco_factory: SoCoMockFactory) -> List[MagicMock]:
    """Set up home assistant with two Sonos Speakers."""
    soco_lr: MagicMock = soco_factory.cache_mock(MockSoCo(), '10.10.10.1', 'Living Room')
    soco_br: MagicMock = soco_factory.cache_mock(MockSoCo(), '10.10.10.2', 'Bedroom')
    await async_setup_component(
        hass,
        DOMAIN,
        {DOMAIN: {'media_player': {'interface_addr': '127.0.0.1', 'hosts': ['10.10.10.1', '10.10.10.2']}}},
    )
    await hass.async_block_till_done()
    return [soco_lr, soco_br]