from collections import namedtuple
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any

ColorTempRange = namedtuple('ColorTempRange', ['min', 'max'])

def _load_feature_fixtures() -> dict:
    fixtures = load_json_value_fixture('features.json', DOMAIN)
    for fixture in fixtures.values():
        if isinstance(fixture['value'], str):
            try:
                time = datetime.strptime(fixture['value'], '%Y-%m-%d %H:%M:%S.%f%z')
                fixture['value'] = time
            except ValueError:
                pass
    return fixtures

FEATURES_FIXTURE: dict = _load_feature_fixtures()
FIXTURE_ENUM_TYPES: dict = {'CleanErrorCode': ErrorCode, 'CleanAreaUnit': AreaUnit}

async def setup_platform_for_device(hass: HomeAssistant, config_entry: ConfigEntry, platform: str, device: Device) -> None:
    """Set up a single tplink platform with a device."""
    config_entry.add_to_hass(hass)
    with patch('homeassistant.components.tplink.PLATFORMS', [platform]), _patch_discovery(device=device), _patch_connect(device=device):
        await hass.config_entries.async_setup(config_entry.entry_id)
        await hass.async_block_till_done(wait_background_tasks=True)

async def snapshot_platform(hass: HomeAssistant, entity_registry: er.EntityRegistry, device_registry: dr.DeviceRegistry, snapshot: Any, config_entry_id: str) -> None:
    """Snapshot a platform."""
    # ... rest of the function ...

async def setup_automation(hass: HomeAssistant, alias: str, entity_id: str) -> None:
    """Set up an automation for tests."""
    # ... rest of the function ...

def _mock_protocol() -> BaseProtocol:
    protocol = MagicMock(spec=BaseProtocol)
    protocol.close = AsyncMock()
    return protocol

def _mocked_device(device_config: DEVICE_CONFIG_LEGACY = DEVICE_CONFIG_LEGACY, credentials_hash: str = CREDENTIALS_HASH_LEGACY, mac: str = MAC_ADDRESS, device_id: str = DEVICE_ID, alias: str = ALIAS, model: str = MODEL, ip_address: str = None, modules: list = None, children: list = None, features: list = None, device_type: DeviceType = None, spec: type = Device) -> Device:
    # ... rest of the function ...

def _mocked_feature(id: str, *, require_fixture: bool = False, value: Any = UNDEFINED, name: str = None, type_: type = None, category: str = None, precision_hint: int = None, choices: list = None, unit: str = None, minimum_value: int = None, maximum_value: int = None, expected_module_key: str = None) -> Feature:
    # ... rest of the function ...

def _mocked_light_module(device: Device) -> Light:
    # ... rest of the function ...

def _mocked_light_effect_module(device: Device) -> LightEffect:
    # ... rest of the function ...

def _mocked_fan_module(effect: LightEffect) -> Fan:
    # ... rest of the function ...

def _mocked_alarm_module(device: Device) -> Alarm:
    # ... rest of the function ...

def _mocked_camera_module(device: Device) -> Camera:
    # ... rest of the function ...

def _mocked_thermostat_module(device: Device) -> Thermostat:
    # ... rest of the function ...

def _mocked_clean_module(device: Device) -> Clean:
    # ... rest of the function ...

def _mocked_speaker_module(device: Device) -> Speaker:
    # ... rest of the function ...

def _mocked_strip_children(features: list = None, alias: str = None) -> list:
    # ... rest of the function ...

def _mocked_energy_features(power: float = None, total: float = None, voltage: float = None, current: float = None, today: float = None) -> list:
    # ... rest of the function ...

MODULE_TO_MOCK_GEN: dict = {Module.Light: _mocked_light_module, Module.LightEffect: _mocked_light_effect_module, Module.Fan: _mocked_fan_module, Module.Alarm: _mocked_alarm_module, Module.Camera: _mocked_camera_module, Module.Thermostat: _mocked_thermostat_module, Module.Clean: _mocked_clean_module, Module.Speaker: _mocked_speaker_module}

def _patch_discovery(device: Device = None, no_device: bool = False, ip_address: str = IP_ADDRESS) -> patch:
    # ... rest of the function ...

def _patch_single_discovery(device: Device = None, no_device: bool = False) -> patch:
    # ... rest of the function ...

def _patch_connect(device: Device = None, no_device: bool = False) -> patch:
    # ... rest of the function ...

async def initialize_config_entry_for_device(hass: HomeAssistant, dev: Device) -> ConfigEntry:
    """Create a mocked configuration entry for the given device."""
    config_entry = MockConfigEntry(title='TP-Link', domain=DOMAIN, unique_id=dev.mac, data={CONF_HOST: dev.host})
    config_entry.add_to_hass(hass)
    with _patch_discovery(device=dev), _patch_single_discovery(device=dev), _patch_connect(device=dev):
        await hass.config_entries.async_setup(config_entry.entry_id)
        await hass.async_block_till_done()
    return config_entry
