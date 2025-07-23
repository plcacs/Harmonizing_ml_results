"""Test selectors."""
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import pytest
import voluptuous as vol
from homeassistant.helpers import selector
from homeassistant.util import yaml as yaml_util

FAKE_UUID: str = 'a266a680b608c32770e6c45bfe6b8411'

@pytest.mark.parametrize('schema', [{'device': None}, {'entity': None}])
def test_valid_base_schema(schema: Dict[str, Any]) -> None:
    """Test base schema validation."""
    selector.validate_selector(schema)

@pytest.mark.parametrize('schema', [None, 'not_a_dict', {}, {'non_existing': {}}, {'device': {}, 'entity': {}}])
def test_invalid_base_schema(schema: Any) -> None:
    """Test base schema validation."""
    with pytest.raises(vol.Invalid):
        selector.validate_selector(schema)

def _test_selector(
    selector_type: str,
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...],
    converter: Optional[Any] = None
) -> None:
    """Help test a selector."""

    def default_converter(x: Any) -> Any:
        return x
    if converter is None:
        converter = default_converter
    config: Dict[str, Any] = {selector_type: schema}
    selector.validate_selector(config)
    selector_instance = selector.selector(config)
    assert selector_instance == selector.selector(config)
    assert selector_instance != 5
    assert not any((isinstance(val, Enum) for val in selector_instance.config.values()))
    vol_schema = vol.Schema({'selection': selector_instance})
    for selection in valid_selections:
        assert vol_schema({'selection': selection}) == {'selection': converter(selection)}
    for selection in invalid_selections:
        with pytest.raises(vol.Invalid):
            vol_schema({'selection': selection})
    selector_instance = selector.selector({selector_type: schema})
    assert selector_instance.serialize() == {'selector': {selector_type: selector_instance.config}}
    yaml_util.dump(selector_instance.serialize())

@pytest.mark.parametrize(('schema', 'valid_selections', 'invalid_selections'), [(None, ('abc123',), (None,)), ({}, ('abc123',), (None,)), ({'integration': 'zha'}, ('abc123',), (None,)), ({'manufacturer': 'mock-manuf'}, ('abc123',), (None,)), ({'model': 'mock-model'}, ('abc123',), (None,)), ({'manufacturer': 'mock-manuf', 'model': 'mock-model'}, ('abc123',), (None,)), ({'integration': 'zha', 'manufacturer': 'mock-manuf', 'model': 'mock-model'}, ('abc123',), (None,)), ({'entity': {'device_class': 'motion'}}, ('abc123',), (None,)), ({'entity': {'device_class': ['motion', 'temperature']}}, ('abc123',), (None,)), ({'entity': [{'domain': 'light'}, {'domain': 'binary_sensor', 'device_class': 'motion'}]}, ('abc123',), (None,)), ({'integration': 'zha', 'manufacturer': 'mock-manuf', 'model': 'mock-model', 'entity': {'domain': 'binary_sensor', 'device_class': 'motion'}}, ('abc123',), (None,)), ({'multiple': True}, (['abc123', 'def456'],), ('abc123', None, ['abc123', None])), ({'filter': {'integration': 'zha', 'manufacturer': 'mock-manuf', 'model': 'mock-model'}}, ('abc123',), (None,)), ({'filter': [{'integration': 'zha', 'manufacturer': 'mock-manuf', 'model': 'mock-model'}, {'integration': 'matter', 'manufacturer': 'other-mock-manuf', 'model': 'other-mock-model'}]}, ('abc123',), (None,))])
def test_device_selector_schema(schema: Any, valid_selections: Tuple[Any, ...], invalid_selections: Tuple[Any, ...]) -> None:
    """Test device selector."""
    _test_selector('device', schema, valid_selections, invalid_selections)

@pytest.mark.parametrize(('schema', 'valid_selections', 'invalid_selections'), [({}, ('sensor.abc123', FAKE_UUID), (None, 'abc123')), ({'integration': 'zha'}, ('sensor.abc123', FAKE_UUID), (None, 'abc123')), ({'domain': 'light'}, ('light.abc123', FAKE_UUID), (None, 'sensor.abc123')), ({'domain': ['light', 'sensor']}, ('light.abc123', 'sensor.abc123', FAKE_UUID), (None, 'dog.abc123')), ({'device_class': 'motion'}, ('sensor.abc123', FAKE_UUID), (None, 'abc123')), ({'device_class': ['motion', 'temperature']}, ('sensor.abc123', FAKE_UUID), (None, 'abc123')), ({'integration': 'zha', 'domain': 'light'}, ('light.abc123', FAKE_UUID), (None, 'sensor.abc123')), ({'integration': 'zha', 'domain': 'binary_sensor', 'device_class': 'motion'}, ('binary_sensor.abc123', FAKE_UUID), (None, 'sensor.abc123')), ({'multiple': True, 'domain': 'sensor'}, (['sensor.abc123', 'sensor.def456'], ['sensor.abc123', FAKE_UUID]), ('sensor.abc123', FAKE_UUID, None, 'abc123', ['sensor.abc123', 'light.def456'])), ({'include_entities': ['sensor.abc123', 'sensor.def456', 'sensor.ghi789'], 'exclude_entities': ['sensor.ghi789', 'sensor.jkl123']}, ('sensor.abc123', FAKE_UUID), ('sensor.ghi789', 'sensor.jkl123')), ({'multiple': True, 'include_entities': ['sensor.abc123', 'sensor.def456', 'sensor.ghi789'], 'exclude_entities': ['sensor.ghi789', 'sensor.jkl123']}, (['sensor.abc123', 'sensor.def456'], ['sensor.abc123', FAKE_UUID]), (['sensor.abc123', 'sensor.jkl123'], ['sensor.abc123', 'sensor.ghi789'])), ({'filter': {'domain': 'light'}}, ('light.abc123', FAKE_UUID), (None,)), ({'filter': [{'domain': 'light'}, {'domain': 'binary_sensor', 'device_class': 'motion'}]}, ('light.abc123', 'binary_sensor.abc123', FAKE_UUID), (None,)), ({'filter': [{'supported_features': ['light.LightEntityFeature.EFFECT']}]}, ('light.abc123', 'blah.blah', FAKE_UUID), (None,)), ({'filter': [{'supported_features': [['light.LightEntityFeature.EFFECT', 'light.LightEntityFeature.TRANSITION']]}]}, ('light.abc123', 'blah.blah', FAKE_UUID), (None,)), ({'filter': [{'supported_features': ['light.LightEntityFeature.EFFECT', 'light.LightEntityFeature.TRANSITION']}]}, ('light.abc123', 'blah.blah', FAKE_UUID), (None,))])
def test_entity_selector_schema(schema: Any, valid_selections: Tuple[Any, ...], invalid_selections: Tuple[Any, ...]) -> None:
    """Test entity selector."""
    _test_selector('entity', schema, valid_selections, invalid_selections)

@pytest.mark.parametrize('schema', [{'filter': [{'supported_features': [1]}]}, {'filter': [{'supported_features': ['blah']}]}, {'filter': [{'supported_features': ['blah.FooEntityFeature.blah']}]}, {'filter': [{'supported_features': ['light.FooEntityFeature.blah']}]}, {'filter': [{'supported_features': ['light.LightEntityFeature.blah']}]}])
def test_entity_selector_schema_error(schema: Any) -> None:
    """Test number selector."""
    with pytest.raises(vol.Invalid):
        selector.validate_selector({'entity': schema})

@pytest.mark.parametrize(('schema', 'valid_selections', 'invalid_selections'), [({}, ('abc123',), (None,)), ({'entity': {}}, ('abc123',), (None,)), ({'entity': {'domain': 'light'}}, ('abc123',), (None,)), ({'entity': {'domain': 'binary_sensor', 'device_class': 'motion'}}, ('abc123',), (None,)), ({'entity': {'domain': 'binary_sensor', 'device_class': 'motion', 'integration': 'demo'}}, ('abc123',), (None,)), ({'entity': [{'domain': 'light'}, {'domain': 'binary_sensor', 'device_class': 'motion'}]}, ('abc123',), (None,)), ({'device': {'integration': 'demo', 'model': 'mock-model'}}, ('abc123',), (None,)), ({'device': [{'integration': 'demo', 'model': 'mock-model'}, {'integration': 'other-demo', 'model': 'other-mock-model'}]}, ('abc123',), (None,)), ({'entity': {'domain': 'binary_sensor', 'device_class': 'motion'}, 'device': {'integration': 'demo', 'model': 'mock-model'}}, ('abc123',), (None,)), ({'multiple': True}, (['abc123', 'def456'],), (None, 'abc123', ['abc123', None]))])
def test_area_selector_schema(schema: Any, valid_selections: Tuple[Any, ...], invalid_selections: Tuple[Any, ...]) -> None:
    """Test area selector."""
    _test_selector('area', schema, valid_selections, invalid_selections)

@pytest.mark.parametrize(('schema', 'valid_selections', 'invalid_selections'), [({}, ('23ouih2iu23ou2', '2j4hp3uy4p87wyrpiuhk34'), (None, True, 1))])
def test_assist_pipeline_selector_schema(schema: Any, valid_selections: Tuple[Any, ...], invalid_selections: Tuple[Any, ...]) -> None:
    """Test assist pipeline selector."""
    _test_selector('assist_pipeline', schema, valid_selections, invalid_selections)

@pytest.mark.parametrize(('schema', 'valid_selections', 'invalid_selections'), [({'min': 10, 'max': 50}, (10, 50), (9, 51)), ({'min': -100, 'max': 100, 'step': 5}, (), ()), ({'min': -20, 'max': -10, 'mode': 'box'}, (), ()), ({'min': 0, 'max': 100, 'unit_of_measurement': 'seconds', 'mode': 'slider'}, (), ()), ({'min': 10, 'max': 1000, 'mode': 'slider', 'step': 0.5}, (), ()), ({'mode': 'box'}, (10,), ()), ({'mode': 'box', 'step': 'any'}, (), ()), ({'mode': 'slider', 'min': 0, 'max': 1, 'step': 'any'}, (), ())])
def test_number_selector_schema(schema: Any, valid_selections: Tuple[Any, ...], invalid_selections: Tuple[Any, ...]) -> None:
    """Test number selector."""
    _test_selector('number', schema, valid_selections, invalid_selections)

@pytest.mark.parametrize('schema', [{}, {'mode': 'slider'}])
def test_number_selector_schema_error(schema: Any) -> None:
    """Test number selector."""
    with pytest.raises(vol.Invalid):
        selector.validate_selector({'number': schema})

@pytest.mark.parametrize(('schema', 'valid_selections', 'invalid_selections'), [({}, ('abc123',), (None,))])
def test_addon_selector_schema(schema: Any, valid_selections: Tuple[Any, ...], invalid_selections: Tuple[Any, ...]) -> None:
    """Test add-on selector."""
    _test_selector('addon', schema, valid_selections, invalid_selections)

@pytest.mark.parametrize(('schema', 'valid_selections', 'invalid_selections'), [({}, ('abc123', '/backup'), (None, 'abc@123', 'abc 123', ''))])
def test_backup_location_selector_schema(schema: Any, valid_selections: Tuple[Any, ...], invalid_selections: Tuple[Any, ...]) -> None:
    """Test backup location selector."""
    _test_selector('backup_location', schema, valid_selections, invalid_selections)

@pytest.mark.parametrize(('schema', 'valid_selections', 'invalid_selections'), [({}, (1, 'one', None), ())])
def test_boolean_selector_schema(schema: Any, valid_selections: Tuple[Any, ...], invalid_selections: Tuple[Any, ...]) -> None:
    """Test boolean selector."""
    _test_selector('boolean', schema, valid_selections, invalid_selections, bool)

@pytest.mark.parametrize(('schema', 'valid_selections', 'invalid_selections'), [({}, ('6b68b250388cbe0d620c92dd3acc93ec', '76f2e8f9a6491a1b580b3a8967c27ddd'), (None, True, 1)), ({'integration': 'adguard'}, ('6b68b250388cbe0d620c92dd3acc93ec', '76f2e8f9a6491a1b580b3a8967c27ddd'), (None, True, 1))])
def test_config_entry_selector_schema(schema: Any, valid_selections: Tuple[Any, ...], invalid_selections: Tuple[Any, ...]) -> None:
    """Test config entry selector."""
    _test_selector('config_entry', schema, valid_selections, invalid_selections)

@pytest.mark.parametrize(('schema', 'valid_selections', 'invalid_selections'), [({}, ('NL', 'DE'), (None, True, 1)), ({'countries': ['NL', 'DE']}, ('NL', 'DE'), (None, True, 1, 'sv', 'en'))])
def test_country_selector_schema(schema: Any, valid_selections: Tuple[Any, ...], invalid_selections: Tuple[Any, ...]) -> None:
    """Test country selector."""
    _test_selector('country', schema, valid_selections, invalid_selections)

@pytest.mark.parametrize(('schema', 'valid_selections', 'invalid_selections'), [({}, ('00:00:00',), ('blah', None))])
def test_time_selector_schema(schema: Any, valid_selections: Tuple[Any, ...], invalid_selections: Tuple[Any, ...]) -> None:
    """Test time selector."""
    _test_selector('time', schema, valid_selections, invalid_selections)

@pytest.mark.parametrize(('schema', 'valid_selections', 'invalid_selections'), [({'entity_id': 'sensor.abc'}, ('on', 'armed'), (None, True, 1))])
def test_state_selector_schema(schema: Any, valid_selections: Tuple[Any, ...], invalid_selections: Tuple[Any, ...]) -> None:
    """Test state selector."""
    _test_selector('state', schema, valid_selections, invalid_selections)

@pytest.mark.parametrize(('schema', 'valid_selections', 'invalid_selections'), [({}, ({'entity_id': ['sensor.abc123']},), ('abc123', None)), ({'entity': {}}, (), ()), ({'entity': {'domain': 'light'}}, (), ()), ({'entity': {'domain': 'binary_sensor', 'device_class': 'motion'}}, (), ()), ({'entity': {'domain': 'binary_sensor', 'device_class': 'motion', 'integration': 'demo'}}, (), ()), ({'entity': [{'domain': 'light'}, {'domain': 'binary_sensor', 'device_class': 'motion'}]}, (), ()), ({'device': {'integration': 'demo', 'model': 'mock-model'}}, (), ()), ({'device': [{'integration': 'demo', 'model': 'mock-model'}, {'integration': 'other-demo', 'model': 'other-mock-model'}]}, (), ()), ({'entity': {'domain': 'binary_sensor', 'device_class': 'motion'}, 'device': {'integration': 'demo', 'model': 'mock-model'}}, (), ())])
def test_target_selector_schema(schema: Any, valid_selections: Tuple[Any, ...], invalid_selections: Tuple[Any, ...]) -> None:
    """Test target selector."""
    _test_selector('target', schema, valid_selections, invalid_selections)

@pytest.mark.parametrize(('schema', 'valid_selections', 'invalid_selections'), [({}, ('abc123',), ())])
def test_action_selector_schema(schema: Any, valid_selections: Tuple[Any, ...], invalid_selections: Tuple[Any, ...]) -> None:
    """Test action sequence selector."""
    _test_selector('action', schema, valid_selections, invalid_selections)

@pytest.mark.parametrize(('schema', 'valid_selections', 'invalid_selections'), [({}, ('abc123',), ())])
def test_object_selector