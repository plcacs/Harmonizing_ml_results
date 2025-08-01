"""Helper for HomematicIP Cloud Tests."""
import json
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from unittest.mock import Mock, patch
from homematicip.aio.class_maps import TYPE_CLASS_MAP, TYPE_GROUP_MAP, TYPE_SECURITY_EVENT_MAP
from homematicip.aio.device import AsyncDevice
from homematicip.aio.group import AsyncGroup
from homematicip.aio.home import AsyncHome
from homematicip.base.homematicip_object import HomeMaticIPObject
from homematicip.base.functionalChannels import FunctionalChannel
from homematicip.home import Home
from homeassistant.components.homematicip_cloud import DOMAIN as HMIPC_DOMAIN
from homeassistant.components.homematicip_cloud.entity import ATTR_IS_GROUP, ATTR_MODEL_TYPE
from homeassistant.components.homematicip_cloud.hap import HomematicipHAP
from homeassistant.core import HomeAssistant, State
from homeassistant.setup import async_setup_component
from tests.common import MockConfigEntry, load_fixture

HAPID: str = '3014F7110000000000000001'
HAPPIN: str = '5678'
AUTH_TOKEN: str = '1234'
FIXTURE_DATA: str = load_fixture('homematicip_cloud.json', 'homematicip_cloud')

def get_and_check_entity_basics(
    hass: HomeAssistant,
    mock_hap: HomematicipHAP,
    entity_id: str,
    entity_name: str,
    device_model: Optional[str]
) -> Tuple[State, Union[AsyncDevice, AsyncGroup, None]]:
    """Get and test basic device."""
    ha_state = hass.states.get(entity_id)
    assert ha_state is not None
    if device_model:
        assert ha_state.attributes[ATTR_MODEL_TYPE] == device_model
    assert ha_state.name == entity_name
    hmip_device = mock_hap.hmip_device_by_entity_id.get(entity_id)
    if hmip_device:
        if isinstance(hmip_device, AsyncDevice):
            assert ha_state.attributes[ATTR_IS_GROUP] is False
        elif isinstance(hmip_device, AsyncGroup):
            assert ha_state.attributes[ATTR_IS_GROUP]
    return (ha_state, hmip_device)

async def async_manipulate_test_data(
    hass: HomeAssistant,
    hmip_device: Union[AsyncDevice, AsyncGroup, FunctionalChannel],
    attribute: str,
    new_value: Any,
    channel: int = 1,
    fire_device: Optional[Union[AsyncDevice, AsyncGroup, AsyncHome]] = None
) -> None:
    """Set new value on hmip device."""
    if channel == 1:
        setattr(hmip_device, attribute, new_value)
    if hasattr(hmip_device, 'functionalChannels'):
        functional_channel = hmip_device.functionalChannels[channel]
        setattr(functional_channel, attribute, new_value)
    fire_target = hmip_device if fire_device is None else fire_device
    if isinstance(fire_target, AsyncHome):
        fire_target.fire_update_event(fire_target._rawJSONData)
    else:
        fire_target.fire_update_event()
    await hass.async_block_till_done()

class HomeFactory:
    """Factory to create a HomematicIP Cloud Home."""

    def __init__(
        self,
        hass: HomeAssistant,
        mock_connection: Mock,
        hmip_config_entry: MockConfigEntry
    ) -> None:
        """Initialize the Factory."""
        self.hass = hass
        self.mock_connection = mock_connection
        self.hmip_config_entry = hmip_config_entry

    async def async_get_mock_hap(
        self,
        test_devices: Optional[List[str]] = None,
        test_groups: Optional[List[str]] = None
    ) -> HomematicipHAP:
        """Create a mocked homematic access point."""
        home_name = self.hmip_config_entry.data['name']
        mock_home = HomeTemplate(
            connection=self.mock_connection,
            home_name=home_name,
            test_devices=test_devices,
            test_groups=test_groups
        ).init_home().get_async_home_mock()
        self.hmip_config_entry.add_to_hass(self.hass)
        with patch('homeassistant.components.homematicip_cloud.hap.HomematicipHAP.get_hap', return_value=mock_home):
            assert await async_setup_component(self.hass, HMIPC_DOMAIN, {})
        await self.hass.async_block_till_done()
        hap = self.hass.data[HMIPC_DOMAIN][HAPID]
        mock_home.on_update(hap.async_update)
        mock_home.on_create(hap.async_create_entity)
        return hap

class HomeTemplate(Home):
    """Home template as builder for home mock."""

    _typeClassMap: Dict[str, Any] = TYPE_CLASS_MAP
    _typeGroupMap: Dict[str, Any] = TYPE_GROUP_MAP
    _typeSecurityEventMap: Dict[str, Any] = TYPE_SECURITY_EVENT_MAP

    def __init__(
        self,
        connection: Optional[Mock] = None,
        home_name: str = '',
        test_devices: Optional[List[str]] = None,
        test_groups: Optional[List[str]] = None
    ) -> None:
        """Init template with connection."""
        super().__init__(connection=connection)
        self.name = home_name
        self.label = 'Home'
        self.model_type = 'HomematicIP Home'
        self.init_json_state: Optional[Dict[str, Any]] = None
        self.test_devices = test_devices
        self.test_groups = test_groups

    def _cleanup_json(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        if self.test_devices is not None:
            new_devices = {}
            for json_device in json_data['devices'].items():
                if json_device[1]['label'] in self.test_devices:
                    new_devices.update([json_device])
            json_data['devices'] = new_devices
        if self.test_groups is not None:
            new_groups = {}
            for json_group in json_data['groups'].items():
                if json_group[1]['label'] in self.test_groups:
                    new_groups.update([json_group])
            json_data['groups'] = new_groups
        return json_data

    def init_home(self) -> 'HomeTemplate':
        """Init template with json."""
        self.init_json_state = self._cleanup_json(json.loads(FIXTURE_DATA))
        self.update_home(json_state=self.init_json_state, clearConfig=True)
        return self

    def update_home(self, json_state: Dict[str, Any], clearConfig: bool = False) -> bool:
        """Update home and ensure that mocks are created."""
        result = super().update_home(json_state, clearConfig)
        self._generate_mocks()
        return result

    def _generate_mocks(self) -> None:
        """Generate mocks for groups and devices."""
        self.devices = [_get_mock(device) for device in self.devices]
        for device in self.devices:
            device.functionalChannels = [_get_mock(ch) for ch in device.functionalChannels]
        self.groups = [_get_mock(group) for group in self.groups]

    def download_configuration(self) -> Dict[str, Any]:
        """Return the initial json config."""
        return cast(Dict[str, Any], self.init_json_state)

    def get_async_home_mock(self) -> Mock:
        """Create Mock for Async_Home."""
        mock_home = Mock(spec=AsyncHome, wraps=self, label='Home', modelType='HomematicIP Home')
        mock_home.__dict__.update(self.__dict__)
        return mock_home

def _get_mock(instance: Union[HomeMaticIPObject, Mock]) -> Mock:
    """Create a mock and copy instance attributes over mock."""
    if isinstance(instance, Mock):
        instance.__dict__.update(instance._mock_wraps.__dict__)
        return instance
    mock = Mock(spec=instance, wraps=instance)
    mock.__dict__.update(instance.__dict__)
    return mock
