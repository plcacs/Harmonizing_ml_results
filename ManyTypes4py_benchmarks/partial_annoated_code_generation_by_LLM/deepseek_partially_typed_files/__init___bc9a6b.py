"""Tests for the TP-Link component."""
from collections import namedtuple
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, cast
from unittest.mock import AsyncMock, MagicMock, patch
from kasa import BaseProtocol, Device, DeviceType, Feature, KasaException, Module, ThermostatState
from kasa.interfaces import Fan, Light, LightEffect, LightState, Thermostat
from kasa.smart.modules import Speaker
from kasa.smart.modules.alarm import Alarm
from kasa.smart.modules.clean import AreaUnit, Clean, ErrorCode, Status
from kasa.smartcam.modules.camera import LOCAL_STREAMING_PORT, Camera
from syrupy import SnapshotAssertion
from homeassistant.components.automation import DOMAIN as AUTOMATION_DOMAIN
from homeassistant.components.tplink.const import DOMAIN
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_HOST, Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.translation import async_get_translations
from homeassistant.helpers.typing import UNDEFINED
from homeassistant.setup import async_setup_component
from .const import ALIAS, CREDENTIALS_HASH_LEGACY, DEVICE_CONFIG_LEGACY, DEVICE_ID, IP_ADDRESS, MAC_ADDRESS, MODEL
from tests.common import MockConfigEntry, load_json_value_fixture

ColorTempRange = namedtuple('ColorTempRange', ['min', 'max'])

def _load_feature_fixtures() -> Dict[str, Any]:
    fixtures = load_json_value_fixture('features.json', DOMAIN)
    for fixture in fixtures.values():
        if isinstance(fixture['value'], str):
            try:
                time = datetime.strptime(fixture['value'], '%Y-%m-%d %H:%M:%S.%f%z')
                fixture['value'] = time
            except ValueError:
                pass
    return fixtures

FEATURES_FIXTURE: Dict[str, Any] = _load_feature_fixtures()
FIXTURE_ENUM_TYPES: Dict[str, Any] = {'CleanErrorCode': ErrorCode, 'CleanAreaUnit': AreaUnit}

async def setup_platform_for_device(hass: HomeAssistant, config_entry: ConfigEntry, platform: Platform, device: Device) -> None:
    """Set up a single tplink platform with a device."""
    config_entry.add_to_hass(hass)
    with patch('homeassistant.components.tplink.PLATFORMS', [platform]), _patch_discovery(device=device), _patch_connect(device=device):
        await hass.config_entries.async_setup(config_entry.entry_id)
        await hass.async_block_till_done(wait_background_tasks=True)

async def snapshot_platform(hass: HomeAssistant, entity_registry: er.EntityRegistry, device_registry: dr.DeviceRegistry, snapshot: SnapshotAssertion, config_entry_id: str) -> None:
    """Snapshot a platform."""
    device_entries = dr.async_entries_for_config_entry(device_registry, config_entry_id)
    assert device_entries
    for device_entry in device_entries:
        assert device_entry == snapshot(name=f'{device_entry.name}-entry'), f'device entry snapshot failed for {device_entry.name}'
    entity_entries = er.async_entries_for_config_entry(entity_registry, config_entry_id)
    assert entity_entries
    assert len({entity_entry.domain for entity_entry in entity_entries}) == 1, 'Please limit the loaded platforms to 1 platform.'
    translations = await async_get_translations(hass, 'en', 'entity', [DOMAIN])
    unique_device_classes: List[Optional[str]] = []
    for entity_entry in entity_entries:
        if entity_entry.translation_key:
            key = f'component.{DOMAIN}.entity.{entity_entry.domain}.{entity_entry.translation_key}.name'
            single_device_class_translation = False
            if key not in translations:
                if entity_entry.original_device_class not in unique_device_classes:
                    single_device_class_translation = True
                    unique_device_classes.append(entity_entry.original_device_class)
            assert key in translations or single_device_class_translation, f'No translation or non unique device_class for entity {entity_entry.unique_id}, expected {key}'
        assert entity_entry == snapshot(name=f'{entity_entry.entity_id}-entry'), f'entity entry snapshot failed for {entity_entry.entity_id}'
        if entity_entry.disabled_by is None:
            state = hass.states.get(entity_entry.entity_id)
            assert state, f'State not found for {entity_entry.entity_id}'
            assert state == snapshot(name=f'{entity_entry.entity_id}-state'), f'state snapshot failed for {entity_entry.entity_id}'

async def setup_automation(hass: HomeAssistant, alias: str, entity_id: str) -> None:
    """Set up an automation for tests."""
    assert await async_setup_component(hass, AUTOMATION_DOMAIN, {AUTOMATION_DOMAIN: {'alias': alias, 'trigger': {'platform': 'state', 'entity_id': entity_id, 'to': 'on'}, 'action': {'action': 'notify.notify', 'metadata': {}, 'data': {}}}})

def _mock_protocol() -> MagicMock:
    protocol = MagicMock(spec=BaseProtocol)
    protocol.close = AsyncMock()
    return protocol

def _mocked_device(device_config: Any = DEVICE_CONFIG_LEGACY, credentials_hash: str = CREDENTIALS_HASH_LEGACY, mac: str = MAC_ADDRESS, device_id: str = DEVICE_ID, alias: str = ALIAS, model: str = MODEL, ip_address: Optional[str] = None, modules: Optional[List[str]] = None, children: Optional[List[Device]] = None, features: Optional[List[Union[str, Feature]]] = None, device_type: Optional[DeviceType] = None, spec: type = Device) -> Device:
    device = MagicMock(spec=spec, name='Mocked device')
    device.update = AsyncMock()
    device.turn_off = AsyncMock()
    device.turn_on = AsyncMock()
    device.mac = mac
    device.alias = alias
    device.model = model
    device.device_id = device_id
    device.hw_info = {'sw_ver': '1.0.0', 'hw_ver': '1.0.0'}
    device.modules = {}
    device.features = {}
    device_config = replace(device_config)
    if not ip_address:
        ip_address = IP_ADDRESS
    else:
        device_config.host = ip_address
    device.host = ip_address
    device_features: Dict[str, Feature] = {}
    if features:
        device_features = {feature_id: _mocked_feature(cast(str, feature_id), require_fixture=True) for feature_id in features if isinstance(feature_id, str)}
        device_features.update({feature.id: feature for feature in features if isinstance(feature, Feature)})
    device.features = device_features
    if modules:
        device.modules = {module_name: MODULE_TO_MOCK_GEN[module_name](device) for module_name in modules}
    module_features = {mod_key if (mod_key := v.expected_module_key) else k: v for (k, v) in device_features.items()}
    for mod in device.modules.values():
        mod.get_feature.side_effect = lambda mod_id: mod_feat if (mod_feat := module_features.get(mod_id)) and mod_feat.id in device_features else None
        mod.has_feature.side_effect = lambda mod_id: (mod_feat := module_features.get(mod_id)) and mod_feat.id in device_features
    device.parent = None
    device.children = []
    if children:
        for child in children:
            child.mac = mac
            child.parent = device
        device.children = children
    device.device_type = device_type if device_type else DeviceType.Unknown
    if not device_type and device.children and all((child.device_type is DeviceType.StripSocket for child in device.children)):
        device.device_type = DeviceType.Strip
    device.protocol = _mock_protocol()
    device.config = device_config
    device.credentials_hash = credentials_hash
    return device

def _mocked_feature(id: str, *, require_fixture: bool = False, value: Any = UNDEFINED, name: Optional[str] = None, type_: Optional[Feature.Type] = None, category: Optional[Feature.Category] = None, precision_hint: Optional[int] = None, choices: Optional[List[str]] = None, unit: Optional[str] = None, minimum_value: Optional[float] = None, maximum_value: Optional[float] = None, expected_module_key: Optional[str] = None) -> Feature:
    """Get a mocked feature.

    If kwargs are provided they will override the attributes for any features defined in fixtures.json
    """
    feature = MagicMock(spec=Feature, name=f'Mocked {id} feature')
    feature.id = id
    feature.name = name or id.upper()
    feature.set_value = AsyncMock()
    fixture: Dict[str, Any] = {}
    if (fixture := FEATURES_FIXTURE.get(id)):
        fixture = dict(fixture)
        if (enum_type := fixture.get('enum_type')):
            val = FIXTURE_ENUM_TYPES[enum_type](fixture['value'])
            fixture['value'] = val
        if (timedelta_type := fixture.get('timedelta_type')):
            fixture['value'] = timedelta(**{timedelta_type: fixture['value']})
        if (unit_enum_type := fixture.get('unit_enum_type')):
            val = FIXTURE_ENUM_TYPES[unit_enum_type](fixture['unit'])
            fixture['unit'] = val
    else:
        assert require_fixture is False, f'No fixture defined for feature {id} and require_fixture is True'
        assert value is not UNDEFINED, f'Value must be provided if feature {id} not defined in features.json'
        fixture = {'value': value, 'category': 'Primary', 'type': 'Sensor'}
    if value is not UNDEFINED:
        fixture['value'] = value
    feature.value = fixture['value']
    feature.type = type_ or Feature.Type[fixture['type']]
    feature.category = category or Feature.Category[fixture['category']]
    feature.precision_hint = precision_hint or fixture.get('precision_hint')
    feature.unit = unit or fixture.get('unit')
    min_val = minimum_value or fixture.get('minimum_value')
    feature.minimum_value = 0 if min_val is None else min_val
    max_val = maximum_value or fixture.get('maximum_value')
    feature.maximum_value = 2 ** 16 if max_val is None else max_val
    feature.choices = choices or fixture.get('choices')
    feature.expected_module_key = mod_key if (mod_key := fixture.get('expected_module_key', expected_module_key)) else None
    return feature

def _mocked_light_module(device: Device) -> Light:
    light = MagicMock(spec=Light, name='Mocked light module')
    light.update = AsyncMock()
    light.brightness = 50
    light.color_temp = 4000
    light.state = LightState(light_on=True, brightness=light.brightness, color_temp=light.color_temp)
    light.hsv = (10, 30, 5)
    light.hw_info = {'sw_ver': '1.0.0', 'hw_ver': '1.0.0'}

    async def _set_state(state: LightState, *_: Any, **__: Any) -> None:
        light.state = state
    light.set_state = AsyncMock(wraps=_set_state)

    async def _set_brightness(brightness: int, *_: Any, **__: Any) -> None:
        light.state.brightness = brightness
        light.state.light_on = brightness > 0
    light.set_brightness = AsyncMock(wraps=_set_brightness)

    async def _set_hsv(h: int, s: int, v: int, *_: Any, **__: Any) -> None:
        light.state.hue = h
        light.state.saturation = s
        light.state.brightness = v
        light.state.light_on = True
    light.set_hsv = AsyncMock(wraps=_set_hsv)

    async def _set_color_temp(temp: int, *_: Any, **__: Any) -> None:
        light.state.color_temp = temp
        light.state.light_on = True
    light.set_color_temp = AsyncMock(wraps=_set_color_temp)
    light.protocol = _mock_protocol()
    return light

def _mocked_light_effect_module(device: Device) -> LightEffect:
    effect = MagicMock(spec=LightEffect, name='Mocked light effect')
    effect.has_custom_effects = True
    effect.effect = 'Effect1'
    effect.effect_list = ['Off', 'Effect1', 'Effect2']

    async def _set_effect(effect_name: str, *_: Any, **__: Any) -> None:
        assert effect_name in effect.effect_list, f"set_effect '{effect_name}' not in {effect.effect_list}"
        assert device.modules[Module.Light], 'Need a light module to test set_effect method'
        device.modules[Module.Light].state.light_on = True
        effect.effect = effect_name
    effect.set_effect = AsyncMock(wraps=_set_effect)
    effect.set_custom_effect = AsyncMock()
    return effect

def _mocked_fan_module(effect: Any) -> Fan:
    fan = MagicMock(auto_spec=Fan, name='Mocked fan')
    fan.fan_speed_level = 0
    fan.set_fan_speed_level = AsyncMock()
    return fan

def _mocked_alarm_module(device: Device) -> Alarm:
    alarm = MagicMock(auto_spec=Alarm, name='Mocked alarm')
    alarm.active = False
    alarm.alarm_sounds = ('Foo', 'Bar')
    alarm.play = AsyncMock()
    alarm.stop = AsyncMock()
    device.features['alarm_volume'] = _mocked_feature('alarm_volume', minimum_value=0, maximum_value=3, value=None)
    device.features['alarm_duration'] = _mocked_feature('alarm_duration', minimum_value=0, maximum_value=300, value=None)
    return alarm

def _mocked_camera_module(device: Device) -> Camera:
    camera = MagicMock(auto_spec=Camera, name='Mocked camera')
    camera.is_on = True
    camera.set_state = AsyncMock()
    camera.stream_rtsp_url.return_value = f'rtsp://user:pass@{device.host}:{LOCAL_STREAMING_PORT}/stream1'
    return camera

def _mocked_thermostat_module(device: Device) -> Thermostat:
    therm = MagicMock(auto_spec=Thermostat, name='Mocked thermostat')
    therm.state = True
    therm.temperature = 20.2
    therm.target_temperature = 22.2
    therm.mode = ThermostatState.Heating
    therm.set_state = AsyncMock()
    therm.set_target_temperature = AsyncMock()
    return therm

def _mocked_clean_module(device: Device) -> Clean:
    clean = MagicMock(auto_spec=Clean, name='Mocked clean')
    clean.start = AsyncMock()
    clean.pause = AsyncMock()
    clean.resume = AsyncMock()
    clean.return_home = AsyncMock()
    clean.set_fan_speed_preset = AsyncMock()
    clean.fan_speed_preset = 'Max'
    clean.error = ErrorCode.Ok
    clean.battery = 100
    clean.status = Status.Charged
    device.features['vacuum_fan_speed'] = _mocked_feature('vacuum_fan_speed', type_=Feature.Type.Choice, category=Feature.Category.Config, choices=['Quiet', 'Max'], value='Max', expected_module_key='fan_speed_preset')
    return clean

def _mocked_speaker_module(device: Device) -> Speaker:
    speaker = MagicMock(auto_spec=Speaker, name='Mocked speaker')
    speaker.locate = AsyncMock()
    return speaker

def _mocked_strip_children(features: Optional[List[Union[str, Feature]]] = None, alias: Optional[str] = None) -> List[Device]:
    plug0 = _mocked_device(alias='Plug0' if alias is None else alias, device_id='bb:bb:cc:dd:ee:ff_PLUG0DEVICEID', mac='bb:bb:cc:dd:ee:ff', device_type=DeviceType.StripSocket, features=features)
    plug1 = _mocked_device(alias='Plug1' if alias is None else alias, device_id='cc:bb:cc:dd:ee:ff_PLUG1DEVICEID', mac='cc:bb:cc:dd:ee:ff', device_type=DeviceType.StripSocket, features=features)
    plug0.is_on = True
    plug1.is_on = False
    return [plug0, plug1]

def _mocked_energy_features(power: Optional[float] = None, total: Optional[float] = None, voltage: Optional[float] = None, current: Optional[float] = None, today: Optional[float] = None) -> List[Feature]:
    feats: List[Feature] = []
    if power is not None:
        feats.append(_mocked_feature('current_consumption', value=power))
    if total is not None:
        feats.append(_mocked_feature('consumption_total', value=total))
    if voltage is not None:
        feats.append(_mocked_feature('voltage', value=voltage))
    if current is not None:
        feats.append(_mocked_feature('current', value=current))
    feats.append(_mocked_feature('consumption_today', value=today if today is not None else 0.0))
    return feats

MOD