"""Define test fixtures for RainMachine."""
from collections.abc import AsyncGenerator
import json
from typing import Any
from unittest.mock import AsyncMock, patch
import pytest
from homeassistant.components.rainmachine import DOMAIN
from homeassistant.const import CONF_IP_ADDRESS, CONF_PASSWORD, CONF_PORT, CONF_SSL
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
from tests.common import MockConfigEntry, load_fixture

@pytest.fixture(name='client')
def client_fixture(controller: Any, controller_mac: Any) -> AsyncMock:
    """Define a regenmaschine client."""
    return AsyncMock(load_local=AsyncMock(), controllers={controller_mac: controller})

@pytest.fixture(name='config')
def config_fixture() -> dict[, typing.Union[typing.Text,int]]:
    """Define a config entry data fixture."""
    return {CONF_IP_ADDRESS: '192.168.1.100', CONF_PASSWORD: 'password', CONF_PORT: 8080, CONF_SSL: True}

@pytest.fixture(name='config_entry')
def config_entry_fixture(hass: homeassistancore.HomeAssistant, config: Union[homeassistancore.HomeAssistant, dict], controller_mac: Union[homeassistancore.HomeAssistant, dict]) -> MockConfigEntry:
    """Define a config entry fixture."""
    entry = MockConfigEntry(domain=DOMAIN, unique_id=controller_mac, data=config, entry_id='81bd010ed0a63b705f6da8407cb26d4b')
    entry.add_to_hass(hass)
    return entry

@pytest.fixture(name='controller')
def controller_fixture(controller_mac: Union[dict, dict[str, typing.Any], str], data_api_versions: Union[dict, str, dict[str, typing.Any]], data_diagnostics_current: Union[dict, dict[str, typing.Any], bool], data_machine_firmare_update_status: Union[dict, list[str], str], data_programs: Union[dict, dict[str, typing.Any], bool], data_provision_settings: Union[str, homeassistancore.HomeAssistant], data_restrictions_current: Union[dict, dict[str, typing.Any], dict[str, int], None], data_restrictions_universal: Union[typing.Type, dict], data_zones: Union[dict[str, typing.Any], str, dict[str, str]]) -> AsyncMock:
    """Define a regenmaschine controller."""
    controller = AsyncMock()
    controller.api_version = '4.5.0'
    controller.hardware_version = '3'
    controller.name = '12345'
    controller.mac = controller_mac
    controller.software_version = '4.0.925'
    controller.api.versions.return_value = data_api_versions
    controller.diagnostics.current.return_value = data_diagnostics_current
    controller.machine.get_firmware_update_status.return_value = data_machine_firmare_update_status
    controller.programs.all.return_value = data_programs
    controller.provisioning.settings.return_value = data_provision_settings
    controller.restrictions.current.return_value = data_restrictions_current
    controller.restrictions.universal.return_value = data_restrictions_universal
    controller.zones.all.return_value = data_zones
    return controller

@pytest.fixture(name='controller_mac')
def controller_mac_fixture() -> typing.Text:
    """Define a controller MAC address."""
    return 'aa:bb:cc:dd:ee:ff'

@pytest.fixture(name='data_api_versions', scope='package')
def data_api_versions_fixture() -> Union[dict, str, None]:
    """Define API version data."""
    return json.loads(load_fixture('api_versions_data.json', 'rainmachine'))

@pytest.fixture(name='data_diagnostics_current', scope='package')
def data_diagnostics_current_fixture() -> Union[dict, str, None]:
    """Define current diagnostics data."""
    return json.loads(load_fixture('diagnostics_current_data.json', 'rainmachine'))

@pytest.fixture(name='data_machine_firmare_update_status', scope='package')
def data_machine_firmare_update_status_fixture() -> Union[dict, dict[str, typing.Any], None]:
    """Define machine firmware update status data."""
    return json.loads(load_fixture('machine_firmware_update_status_data.json', 'rainmachine'))

@pytest.fixture(name='data_programs', scope='package')
def data_programs_fixture() -> dict[typing.Text, typing.Text]:
    """Define program data."""
    raw_data = json.loads(load_fixture('programs_data.json', 'rainmachine'))
    return {program['uid']: program for program in raw_data}

@pytest.fixture(name='data_provision_settings', scope='package')
def data_provision_settings_fixture() -> Union[dict, str, None]:
    """Define provisioning settings data."""
    return json.loads(load_fixture('provision_settings_data.json', 'rainmachine'))

@pytest.fixture(name='data_restrictions_current', scope='package')
def data_restrictions_current_fixture() -> Union[dict, str, None]:
    """Define current restrictions settings data."""
    return json.loads(load_fixture('restrictions_current_data.json', 'rainmachine'))

@pytest.fixture(name='data_restrictions_universal', scope='package')
def data_restrictions_universal_fixture() -> Union[dict, str, None]:
    """Define universal restrictions settings data."""
    return json.loads(load_fixture('restrictions_universal_data.json', 'rainmachine'))

@pytest.fixture(name='data_zones', scope='package')
def data_zones_fixture() -> dict[typing.Text, dict[typing.Text, ]]:
    """Define zone data."""
    raw_data = json.loads(load_fixture('zones_data.json', 'rainmachine'))
    zone_details = json.loads(load_fixture('zones_details.json', 'rainmachine'))
    zones = {}
    for zone in raw_data:
        [extra] = [z for z in zone_details if z['uid'] == zone['uid']]
        zones[zone['uid']] = {**zone, **extra}
    return zones

@pytest.fixture(name='setup_rainmachine')
async def setup_rainmachine_fixture(hass, client, config):
    """Define a fixture to set up RainMachine."""
    with patch('homeassistant.components.rainmachine.Client', return_value=client), patch('homeassistant.components.rainmachine.config_flow.Client', return_value=client), patch('homeassistant.components.rainmachine.PLATFORMS', []):
        assert await async_setup_component(hass, DOMAIN, config)
        await hass.async_block_till_done()
        yield