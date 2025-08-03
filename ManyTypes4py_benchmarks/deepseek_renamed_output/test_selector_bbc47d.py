"""Test selectors."""
from enum import Enum
from typing import Any, List, Dict, Union, Optional, Callable, Tuple, Sequence
import pytest
import voluptuous as vol
from homeassistant.helpers import selector
from homeassistant.util import yaml as yaml_util

FAKE_UUID: str = 'a266a680b608c32770e6c45bfe6b8411'

def default_converter(x: Any) -> Any:
    return x

def drop_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    """Drop metadata key from the input."""
    data.pop('metadata', None)
    return data

def _custom_trigger_serializer(triggers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    res: List[Dict[str, Any]] = []
    for trigger in triggers:
        if 'trigger' in trigger:
            trigger['platform'] = trigger.pop('trigger')
        res.append(trigger)
    return res

@pytest.mark.parametrize('schema', [{'device': None}, {'entity': None}])
def func_pwjdfbv8(schema: Dict[str, Any]) -> None:
    """Test base schema validation."""
    selector.validate_selector(schema)


@pytest.mark.parametrize('schema', [None, 'not_a_dict', {}, {'non_existing':
    {}}, {'device': {}, 'entity': {}}])
def func_yyqo47tm(schema: Any) -> None:
    """Test base schema validation."""
    with pytest.raises(vol.Invalid):
        selector.validate_selector(schema)


def func_wd6ni76z(
    selector_type: str,
    schema: Optional[Dict[str, Any]],
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...],
    converter: Optional[Callable[[Any], Any]] = None
) -> None:
    """Help test a selector."""
    if converter is None:
        converter = default_converter
    config: Dict[str, Any] = {selector_type: schema}
    selector.validate_selector(config)
    selector_instance = selector.selector(config)
    assert selector_instance == selector.selector(config)
    assert selector_instance != 5
    assert not any(isinstance(val, Enum) for val in selector_instance.config.values())
    vol_schema = vol.Schema({'selection': selector_instance})
    for selection in valid_selections:
        assert vol_schema({'selection': selection}) == {'selection': converter(selection)}
    for selection in invalid_selections:
        with pytest.raises(vol.Invalid):
            vol_schema({'selection': selection})
    selector_instance = selector.selector({selector_type: schema})
    assert selector_instance.serialize() == {'selector': {selector_type: selector_instance.config}}
    yaml_util.dump(selector_instance.serialize())


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [(None, ('abc123',), (None,)), ({}, ('abc123',),
    (None,)), ({'integration': 'zha'}, ('abc123',), (None,)), ({
    'manufacturer': 'mock-manuf'}, ('abc123',), (None,)), ({'model':
    'mock-model'}, ('abc123',), (None,)), ({'manufacturer': 'mock-manuf',
    'model': 'mock-model'}, ('abc123',), (None,)), ({'integration': 'zha',
    'manufacturer': 'mock-manuf', 'model': 'mock-model'}, ('abc123',), (
    None,)), ({'entity': {'device_class': 'motion'}}, ('abc123',), (None,)),
    ({'entity': {'device_class': ['motion', 'temperature']}}, ('abc123',),
    (None,)), ({'entity': [{'domain': 'light'}, {'domain': 'binary_sensor',
    'device_class': 'motion'}]}, ('abc123',), (None,)), ({'integration':
    'zha', 'manufacturer': 'mock-manuf', 'model': 'mock-model', 'entity': {
    'domain': 'binary_sensor', 'device_class': 'motion'}}, ('abc123',), (
    None,)), ({'multiple': True}, (['abc123', 'def456'],), ('abc123', None,
    ['abc123', None])), ({'filter': {'integration': 'zha', 'manufacturer':
    'mock-manuf', 'model': 'mock-model'}}, ('abc123',), (None,)), ({
    'filter': [{'integration': 'zha', 'manufacturer': 'mock-manuf', 'model':
    'mock-model'}, {'integration': 'matter', 'manufacturer':
    'other-mock-manuf', 'model': 'other-mock-model'}]}, ('abc123',), (None,))])
def func_4o9o874y(
    schema: Optional[Dict[str, Any]],
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None:
    """Test device selector."""
    func_wd6ni76z('device', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({}, ('sensor.abc123', FAKE_UUID), (None,
    'abc123')), ({'integration': 'zha'}, ('sensor.abc123', FAKE_UUID), (
    None, 'abc123')), ({'domain': 'light'}, ('light.abc123', FAKE_UUID), (
    None, 'sensor.abc123')), ({'domain': ['light', 'sensor']}, (
    'light.abc123', 'sensor.abc123', FAKE_UUID), (None, 'dog.abc123')), ({
    'device_class': 'motion'}, ('sensor.abc123', FAKE_UUID), (None,
    'abc123')), ({'device_class': ['motion', 'temperature']}, (
    'sensor.abc123', FAKE_UUID), (None, 'abc123')), ({'integration': 'zha',
    'domain': 'light'}, ('light.abc123', FAKE_UUID), (None, 'sensor.abc123'
    )), ({'integration': 'zha', 'domain': 'binary_sensor', 'device_class':
    'motion'}, ('binary_sensor.abc123', FAKE_UUID), (None, 'sensor.abc123')
    ), ({'multiple': True, 'domain': 'sensor'}, (['sensor.abc123',
    'sensor.def456'], ['sensor.abc123', FAKE_UUID]), ('sensor.abc123',
    FAKE_UUID, None, 'abc123', ['sensor.abc123', 'light.def456'])), ({
    'include_entities': ['sensor.abc123', 'sensor.def456', 'sensor.ghi789'],
    'exclude_entities': ['sensor.ghi789', 'sensor.jkl123']}, (
    'sensor.abc123', FAKE_UUID), ('sensor.ghi789', 'sensor.jkl123')), ({
    'multiple': True, 'include_entities': ['sensor.abc123', 'sensor.def456',
    'sensor.ghi789'], 'exclude_entities': ['sensor.ghi789', 'sensor.jkl123'
    ]}, (['sensor.abc123', 'sensor.def456'], ['sensor.abc123', FAKE_UUID]),
    (['sensor.abc123', 'sensor.jkl123'], ['sensor.abc123', 'sensor.ghi789']
    )), ({'filter': {'domain': 'light'}}, ('light.abc123', FAKE_UUID), (
    None,)), ({'filter': [{'domain': 'light'}, {'domain': 'binary_sensor',
    'device_class': 'motion'}]}, ('light.abc123', 'binary_sensor.abc123',
    FAKE_UUID), (None,)), ({'filter': [{'supported_features': [
    'light.LightEntityFeature.EFFECT']}]}, ('light.abc123', 'blah.blah',
    FAKE_UUID), (None,)), ({'filter': [{'supported_features': [[
    'light.LightEntityFeature.EFFECT',
    'light.LightEntityFeature.TRANSITION']]}]}, ('light.abc123',
    'blah.blah', FAKE_UUID), (None,)), ({'filter': [{'supported_features':
    ['light.LightEntityFeature.EFFECT',
    'light.LightEntityFeature.TRANSITION']}]}, ('light.abc123', 'blah.blah',
    FAKE_UUID), (None,))])
def func_fv8szmmz(
    schema: Dict[str, Any],
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None:
    """Test entity selector."""
    func_wd6ni76z('entity', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize('schema', [{'filter': [{'supported_features': [1]}
    ]}, {'filter': [{'supported_features': ['blah']}]}, {'filter': [{
    'supported_features': ['blah.FooEntityFeature.blah']}]}, {'filter': [{
    'supported_features': ['light.FooEntityFeature.blah']}]}, {'filter': [{
    'supported_features': ['light.LightEntityFeature.blah']}]}])
def func_1pitmloa(schema: Dict[str, Any]) -> None:
    """Test number selector."""
    with pytest.raises(vol.Invalid):
        selector.validate_selector({'entity': schema})


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({}, ('abc123',), (None,)), ({'entity': {}}, (
    'abc123',), (None,)), ({'entity': {'domain': 'light'}}, ('abc123',), (
    None,)), ({'entity': {'domain': 'binary_sensor', 'device_class':
    'motion'}}, ('abc123',), (None,)), ({'entity': {'domain':
    'binary_sensor', 'device_class': 'motion', 'integration': 'demo'}}, (
    'abc123',), (None,)), ({'entity': [{'domain': 'light'}, {'domain':
    'binary_sensor', 'device_class': 'motion'}]}, ('abc123',), (None,)), ({
    'device': {'integration': 'demo', 'model': 'mock-model'}}, ('abc123',),
    (None,)), ({'device': [{'integration': 'demo', 'model': 'mock-model'},
    {'integration': 'other-demo', 'model': 'other-mock-model'}]}, ('abc123'
    ,), (None,)), ({'entity': {'domain': 'binary_sensor', 'device_class':
    'motion'}, 'device': {'integration': 'demo', 'model': 'mock-model'}}, (
    'abc123',), (None,)), ({'multiple': True}, (['abc123', 'def456'],), (
    None, 'abc123', ['abc123', None]))])
def func_6cv9njnj(
    schema: Dict[str, Any],
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None:
    """Test area selector."""
    func_wd6ni76z('area', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({}, ('23ouih2iu23ou2',
    '2j4hp3uy4p87wyrpiuhk34'), (None, True, 1))])
def func_pjtntd00(
    schema: Dict[str, Any],
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None:
    """Test assist pipeline selector."""
    func_wd6ni76z('assist_pipeline', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({'min': 10, 'max': 50}, (10, 50), (9, 51)), ({
    'min': -100, 'max': 100, 'step': 5}, (), ()), ({'min': -20, 'max': -10,
    'mode': 'box'}, (), ()), ({'min': 0, 'max': 100, 'unit_of_measurement':
    'seconds', 'mode': 'slider'}, (), ()), ({'min': 10, 'max': 1000, 'mode':
    'slider', 'step': 0.5}, (), ()), ({'mode': 'box'}, (10,), ()), ({'mode':
    'box', 'step': 'any'}, (), ()), ({'mode': 'slider', 'min': 0, 'max': 1,
    'step': 'any'}, (), ())])
def func_dn31y2qz(
    schema: Dict[str, Any],
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None:
    """Test number selector."""
    func_wd6ni76z('number', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize('schema', [{}, {'mode': 'slider'}])
def func_qhxiv548(schema: Dict[str, Any]) -> None:
    """Test number selector."""
    with pytest.raises(vol.Invalid):
        selector.validate_selector({'number': schema})


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({}, ('abc123',), (None,))])
def func_5oe36abi(
    schema: Dict[str, Any],
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None:
    """Test add-on selector."""
    func_wd6ni76z('addon', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({}, ('abc123', '/backup'), (None, 'abc@123',
    'abc 123', ''))])
def func_4g6k5wnj(
    schema: Dict[str, Any],
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None:
    """Test backup location selector."""
    func_wd6ni76z('backup_location', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({}, (1, 'one', None), ())])
def func_3kfjyumn(
    schema: Dict[str, Any],
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None:
    """Test boolean selector."""
    func_wd6ni76z('boolean', schema, valid_selections, invalid_selections, bool)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({}, ('6b68b250388cbe0d620c92dd3acc93ec',
    '76f2e8f9a6491a1b580b3a8967c27ddd'), (None, True, 1)), ({'integration':
    'adguard'}, ('6b68b250388cbe0d620c92dd3acc93ec',
    '76f2e8f9a6491a1b580b3a8967c27ddd'), (None, True, 1))])
def func_qzd7y37w(
    schema: Dict[str, Any],
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None:
    """Test config entry selector."""
    func_wd6ni76z('config_entry', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({}, ('NL', 'DE'), (None, True, 1)), ({
    'countries': ['NL', 'DE']}, ('NL', 'DE'), (None, True, 1, 'sv', 'en'))])
def func_1a9t8pzv(
    schema: Dict[str, Any],
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None:
    """Test country selector."""
    func_wd6ni76z('country', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({}, ('00:00:00',), ('blah', None))])
def func_g7pj14aw(
    schema: Dict[str, Any],
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None:
    """Test time selector."""
    func_wd6ni76z('time', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({'entity_id': 'sensor.abc'}, ('on', 'armed'),
    (None, True, 1))])
def func_yau9dfkz(
    schema: Dict[str, Any],
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None:
    """Test state selector."""
    func_wd6ni76z('state', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({}, ({'entity_id': ['sensor.abc123']},), (
    'abc123', None)), ({'entity': {}}, (), ()), ({'entity': {'domain':
    'light'}}, (), ()), ({'entity':