"""Test configuration and mocks for the SmartThings component."""
import secrets
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import Mock, patch
from uuid import uuid4

from pysmartthings import (
    CLASSIFICATION_AUTOMATION,
    AppEntity,
    AppOAuthClient,
    AppSettings,
    DeviceEntity,
    DeviceStatus,
    InstalledApp,
    InstalledAppStatus,
    InstalledAppType,
    Location,
    SceneEntity,
    SmartThings,
    Subscription,
)
from pysmartthings.api import Api
import pytest
from homeassistant.components import webhook
from homeassistant.components.smartthings import DeviceBroker
from homeassistant.components.smartthings.const import (
    APP_NAME_PREFIX,
    CONF_APP_ID,
    CONF_INSTALLED_APP_ID,
    CONF_INSTANCE_ID,
    CONF_LOCATION_ID,
    CONF_REFRESH_TOKEN,
    DATA_BROKERS,
    DOMAIN,
    SETTINGS_INSTANCE_ID,
    STORAGE_KEY,
    STORAGE_VERSION,
)
from homeassistant.config_entries import SOURCE_USER, ConfigEntryState
from homeassistant.const import CONF_ACCESS_TOKEN, CONF_CLIENT_ID, CONF_CLIENT_SECRET, CONF_WEBHOOK_ID
from homeassistant.core import HomeAssistant
from homeassistant.core_config import async_process_ha_core_config
from homeassistant.setup import async_setup_component
from tests.common import MockConfigEntry
from tests.components.light.conftest import mock_light_profiles

COMPONENT_PREFIX: str = 'homeassistant.components.smartthings.'


async def setup_platform(
    hass: HomeAssistant,
    platform: str,
    *,
    devices: Optional[List[Any]] = None,
    scenes: Optional[List[Any]] = None,
) -> MockConfigEntry:
    """Set up the SmartThings platform and prerequisites."""
    hass.config.components.add(DOMAIN)
    config_entry: MockConfigEntry = MockConfigEntry(
        version=2, domain=DOMAIN, title='Test', data={CONF_INSTALLED_APP_ID: str(uuid4())}
    )
    config_entry.add_to_hass(hass)
    broker = DeviceBroker(hass, config_entry, Mock(), Mock(), devices or [], scenes or [])
    hass.data[DOMAIN] = {DATA_BROKERS: {config_entry.entry_id: broker}}
    config_entry.mock_state(hass, ConfigEntryState.LOADED)
    await hass.config_entries.async_forward_entry_setups(config_entry, [platform])
    await hass.async_block_till_done()
    return config_entry


@pytest.fixture(autouse=True)
async def setup_component(hass: HomeAssistant, config_file: Dict[str, Any], hass_storage: Dict[str, Any]) -> None:
    """Load the SmartThing component."""
    hass_storage[STORAGE_KEY] = {'data': config_file, 'version': STORAGE_VERSION}
    await async_process_ha_core_config(hass, {'external_url': 'https://test.local'})
    await async_setup_component(hass, 'smartthings', {})


def _create_location() -> Location:
    loc: Location = Mock(spec=Location)
    loc.name = 'Test Location'
    loc.location_id = str(uuid4())
    return loc


@pytest.fixture(name='location')
def location_fixture() -> Location:
    """Fixture for a single location."""
    return _create_location()


@pytest.fixture(name='locations')
def locations_fixture(location: Location) -> List[Location]:
    """Fixture for 2 locations."""
    return [location, _create_location()]


@pytest.fixture(name='app')
async def app_fixture(hass: HomeAssistant, config_file: Dict[str, Any]) -> AppEntity:
    """Fixture for a single app."""
    app: AppEntity = Mock(spec=AppEntity)
    app.app_name = APP_NAME_PREFIX + str(uuid4())
    app.app_id = str(uuid4())
    app.app_type = 'WEBHOOK_SMART_APP'
    app.classifications = [CLASSIFICATION_AUTOMATION]
    app.display_name = 'Home Assistant'
    app.description = f'{hass.config.location_name} at https://test.local'
    app.single_instance = True
    app.webhook_target_url = webhook.async_generate_url(hass, hass.data[DOMAIN][CONF_WEBHOOK_ID])
    settings: AppSettings = Mock(spec=AppSettings)
    settings.app_id = app.app_id
    settings.settings = {SETTINGS_INSTANCE_ID: config_file[CONF_INSTANCE_ID]}
    app.settings.return_value = settings
    return app


@pytest.fixture(name='app_oauth_client')
def app_oauth_client_fixture() -> AppOAuthClient:
    """Fixture for a single app's oauth."""
    client: AppOAuthClient = Mock(spec=AppOAuthClient)
    client.client_id = str(uuid4())
    client.client_secret = str(uuid4())
    return client


@pytest.fixture(name='app_settings')
def app_settings_fixture(app: AppEntity, config_file: Dict[str, Any]) -> AppSettings:
    """Fixture for an app settings."""
    settings: AppSettings = Mock(spec=AppSettings)
    settings.app_id = app.app_id
    settings.settings = {SETTINGS_INSTANCE_ID: config_file[CONF_INSTANCE_ID]}
    return settings


def _create_installed_app(location_id: str, app_id: str) -> InstalledApp:
    item: InstalledApp = Mock(spec=InstalledApp)
    item.installed_app_id = str(uuid4())
    item.installed_app_status = InstalledAppStatus.AUTHORIZED
    item.installed_app_type = InstalledAppType.WEBHOOK_SMART_APP
    item.app_id = app_id
    item.location_id = location_id
    return item


@pytest.fixture(name='installed_app')
def installed_app_fixture(location: Location, app: AppEntity) -> InstalledApp:
    """Fixture for a single installed app."""
    return _create_installed_app(location.location_id, app.app_id)


@pytest.fixture(name='installed_apps')
def installed_apps_fixture(installed_app: InstalledApp, locations: List[Location], app: AppEntity) -> List[InstalledApp]:
    """Fixture for 2 installed apps."""
    return [installed_app, _create_installed_app(locations[1].location_id, app.app_id)]


@pytest.fixture(name='config_file')
def config_file_fixture() -> Dict[str, str]:
    """Fixture representing the local config file contents."""
    return {CONF_INSTANCE_ID: str(uuid4()), CONF_WEBHOOK_ID: secrets.token_hex()}


@pytest.fixture(name='smartthings_mock')
def smartthings_mock_fixture(locations: List[Location]) -> Any:
    """Fixture to mock smartthings API calls."""

    async def _location(location_id: str) -> Optional[Location]:
        return next((location for location in locations if location.location_id == location_id), None)

    smartthings_mock: SmartThings = Mock(spec=SmartThings)
    smartthings_mock.location.side_effect = _location
    mock = Mock(return_value=smartthings_mock)
    with patch(COMPONENT_PREFIX + 'SmartThings', new=mock), patch(
        COMPONENT_PREFIX + 'config_flow.SmartThings', new=mock
    ), patch(COMPONENT_PREFIX + 'smartapp.SmartThings', new=mock):
        yield smartthings_mock


@pytest.fixture(name='device')
def device_fixture(location: Location) -> DeviceEntity:
    """Fixture representing devices loaded."""
    item: DeviceEntity = Mock(spec=DeviceEntity)
    item.device_id = '743de49f-036f-4e9c-839a-2f89d57607db'
    item.name = 'GE In-Wall Smart Dimmer'
    item.label = 'Front Porch Lights'
    item.location_id = location.location_id
    item.capabilities = [
        'switch',
        'switchLevel',
        'refresh',
        'indicator',
        'sensor',
        'actuator',
        'healthCheck',
        'light',
    ]
    item.components = {'main': item.capabilities}
    item.status = Mock(spec=DeviceStatus)
    return item


@pytest.fixture(name='config_entry')
def config_entry_fixture(installed_app: InstalledApp, location: Location) -> MockConfigEntry:
    """Fixture representing a config entry."""
    data: Dict[str, Any] = {
        CONF_ACCESS_TOKEN: str(uuid4()),
        CONF_INSTALLED_APP_ID: installed_app.installed_app_id,
        CONF_APP_ID: installed_app.app_id,
        CONF_LOCATION_ID: location.location_id,
        CONF_REFRESH_TOKEN: str(uuid4()),
        CONF_CLIENT_ID: str(uuid4()),
        CONF_CLIENT_SECRET: str(uuid4()),
    }
    return MockConfigEntry(domain=DOMAIN, data=data, title=location.name, version=2, source=SOURCE_USER)


@pytest.fixture(name='subscription_factory')
def subscription_factory_fixture() -> Callable[[str], Subscription]:
    """Fixture for creating mock subscriptions."""

    def _factory(capability: str) -> Subscription:
        sub: Subscription = Subscription()
        sub.capability = capability
        return sub

    return _factory


@pytest.fixture(name='device_factory')
def device_factory_fixture() -> Callable[[str, List[str], Optional[Dict[str, Any]]], DeviceEntity]:
    """Fixture for creating mock devices."""
    api: Api = Mock(spec=Api)
    api.post_device_command.return_value = {'results': [{'status': 'ACCEPTED'}]}

    def _factory(label: str, capabilities: List[str], status: Optional[Dict[str, Any]] = None) -> DeviceEntity:
        device_data: Dict[str, Any] = {
            'deviceId': str(uuid4()),
            'name': 'Device Type Handler Name',
            'label': label,
            'deviceManufacturerCode': '9135fc86-0929-4436-bf73-5d75f523d9db',
            'locationId': 'fcd829e9-82f4-45b9-acfd-62fda029af80',
            'components': [
                {
                    'id': 'main',
                    'capabilities': [{'id': capability, 'version': 1} for capability in capabilities],
                }
            ],
            'dth': {
                'deviceTypeId': 'b678b29d-2726-4e4f-9c3f-7aa05bd08964',
                'deviceTypeName': 'Switch',
                'deviceNetworkType': 'ZWAVE',
            },
            'type': 'DTH',
        }
        device: DeviceEntity = DeviceEntity(api, data=device_data)
        if status:
            for attribute, value in status.items():
                device.status.apply_attribute_update('main', '', attribute, value)
        return device

    return _factory


@pytest.fixture(name='scene_factory')
def scene_factory_fixture(location: Location) -> Callable[[str], SceneEntity]:
    """Fixture for creating mock scenes."""

    def _factory(name: str) -> SceneEntity:
        scene: SceneEntity = Mock(spec=SceneEntity)
        scene.scene_id = str(uuid4())
        scene.name = name
        scene.icon = None
        scene.color = None
        scene.location_id = location.location_id
        return scene

    return _factory


@pytest.fixture(name='scene')
def scene_fixture(scene_factory: Callable[[str], SceneEntity]) -> SceneEntity:
    """Fixture for an individual scene."""
    return scene_factory('Test Scene')


@pytest.fixture(name='event_factory')
def event_factory_fixture() -> Callable[
    [str, Optional[str], Optional[str], Optional[str], Optional[str], Optional[Any]], Any
]:
    """Fixture for creating mock events."""

    def _factory(
        device_id: str,
        event_type: str = 'DEVICE_EVENT',
        capability: str = '',
        attribute: str = 'Updated',
        value: str = 'Value',
        data: Optional[Any] = None,
    ) -> Any:
        event: Any = Mock()
        event.event_type = event_type
        event.device_id = device_id
        event.component_id = 'main'
        event.capability = capability
        event.attribute = attribute
        event.value = value
        event.data = data
        event.location_id = str(uuid4())
        return event

    return _factory


@pytest.fixture(name='event_request_factory')
def event_request_factory_fixture(
    event_factory: Callable[
        [str, Optional[str], Optional[str], Optional[str], Optional[str], Optional[Any]], Any
    ]
) -> Callable[[Optional[List[str]], Optional[List[Any]]], Any]:
    """Fixture for creating mock smartapp event requests."""

    def _factory(
        device_ids: Optional[List[str]] = None, events: Optional[List[Any]] = None
    ) -> Any:
        request: Any = Mock()
        request.installed_app_id = uuid4()
        if events is None:
            events = []
        if device_ids:
            events.extend([event_factory(device_id) for device_id in device_ids])
            events.append(event_factory(str(uuid4())))
            events.append(event_factory(device_ids[0], event_type='OTHER'))
        request.events = events
        return request

    return _factory
