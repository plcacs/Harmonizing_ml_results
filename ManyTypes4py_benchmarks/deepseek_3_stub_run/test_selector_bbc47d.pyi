from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import pytest
import voluptuous as vol
from homeassistant.helpers import selector
from homeassistant.util import yaml as yaml_util

FAKE_UUID: str = ...

@pytest.mark.parametrize('schema', [{'device': None}, {'entity': None}])
def test_valid_base_schema(schema: Dict[str, Optional[Any]]) -> None: ...

@pytest.mark.parametrize('schema', [None, 'not_a_dict', {}, {'non_existing': {}}, {'device': {}, 'entity': {}}])
def test_invalid_base_schema(schema: Optional[Union[str, Dict[str, Any]]]) -> None: ...

def _test_selector(
    selector_type: str,
    schema: Optional[Dict[str, Any]],
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...],
    converter: Optional[Callable[[Any], Any]] = None
) -> None: ...

@pytest.mark.parametrize(
    ('schema', 'valid_selections', 'invalid_selections'),
    [
        (None, ('abc123',), (None,)),
        ({}, ('abc123',), (None,)),
        ({'integration': 'zha'}, ('abc123',), (None,)),
        ({'manufacturer': 'mock-manuf'}, ('abc123',), (None,)),
        ({'model': 'mock-model'}, ('abc123',), (None,)),
        ({'manufacturer': 'mock-manuf', 'model': 'mock-model'}, ('abc123',), (None,)),
        ({'integration': 'zha', 'manufacturer': 'mock-manuf', 'model': 'mock-model'}, ('abc123',), (None,)),
        ({'entity': {'device_class': 'motion'}}, ('abc123',), (None,)),
        ({'entity': {'device_class': ['motion', 'temperature']}}, ('abc123',), (None,)),
        ({'entity': [{'domain': 'light'}, {'domain': 'binary_sensor', 'device_class': 'motion'}]}, ('abc123',), (None,)),
        ({'integration': 'zha', 'manufacturer': 'mock-manuf', 'model': 'mock-model', 'entity': {'domain': 'binary_sensor', 'device_class': 'motion'}}, ('abc123',), (None,)),
        ({'multiple': True}, (['abc123', 'def456'],), ('abc123', None, ['abc123', None])),
        ({'filter': {'integration': 'zha', 'manufacturer': 'mock-manuf', 'model': 'mock-model'}}, ('abc123',), (None,)),
        ({'filter': [{'integration': 'zha', 'manufacturer': 'mock-manuf', 'model': 'mock-model'}, {'integration': 'matter', 'manufacturer': 'other-mock-manuf', 'model': 'other-mock-model'}]}, ('abc123',), (None,))
    ]
)
def test_device_selector_schema(
    schema: Optional[Dict[str, Any]],
    valid_selections: Tuple[Union[str, List[str]], ...],
    invalid_selections: Tuple[Optional[Union[str, List[Optional[str]]]], ...]
) -> None: ...

@pytest.mark.parametrize(
    ('schema', 'valid_selections', 'invalid_selections'),
    [
        ({}, ('sensor.abc123', FAKE_UUID), (None, 'abc123')),
        ({'integration': 'zha'}, ('sensor.abc123', FAKE_UUID), (None, 'abc123')),
        ({'domain': 'light'}, ('light.abc123', FAKE_UUID), (None, 'sensor.abc123')),
        ({'domain': ['light', 'sensor']}, ('light.abc123', 'sensor.abc123', FAKE_UUID), (None, 'dog.abc123')),
        ({'device_class': 'motion'}, ('sensor.abc123', FAKE_UUID), (None, 'abc123')),
        ({'device_class': ['motion', 'temperature']}, ('sensor.abc123', FAKE_UUID), (None, 'abc123')),
        ({'integration': 'zha', 'domain': 'light'}, ('light.abc123', FAKE_UUID), (None, 'sensor.abc123')),
        ({'integration': 'zha', 'domain': 'binary_sensor', 'device_class': 'motion'}, ('binary_sensor.abc123', FAKE_UUID), (None, 'sensor.abc123')),
        ({'multiple': True, 'domain': 'sensor'}, (['sensor.abc123', 'sensor.def456'], ['sensor.abc123', FAKE_UUID]), ('sensor.abc123', FAKE_UUID, None, 'abc123', ['sensor.abc123', 'light.def456'])),
        ({'include_entities': ['sensor.abc123', 'sensor.def456', 'sensor.ghi789'], 'exclude_entities': ['sensor.ghi789', 'sensor.jkl123']}, ('sensor.abc123', FAKE_UUID), ('sensor.ghi789', 'sensor.jkl123')),
        ({'multiple': True, 'include_entities': ['sensor.abc123', 'sensor.def456', 'sensor.ghi789'], 'exclude_entities': ['sensor.ghi789', 'sensor.jkl123']}, (['sensor.abc123', 'sensor.def456'], ['sensor.abc123', FAKE_UUID]), (['sensor.abc123', 'sensor.jkl123'], ['sensor.abc123', 'sensor.ghi789'])),
        ({'filter': {'domain': 'light'}}, ('light.abc123', FAKE_UUID), (None,)),
        ({'filter': [{'domain': 'light'}, {'domain': 'binary_sensor', 'device_class': 'motion'}]}, ('light.abc123', 'binary_sensor.abc123', FAKE_UUID), (None,)),
        ({'filter': [{'supported_features': ['light.LightEntityFeature.EFFECT']}]}, ('light.abc123', 'blah.blah', FAKE_UUID), (None,)),
        ({'filter': [{'supported_features': [['light.LightEntityFeature.EFFECT', 'light.LightEntityFeature.TRANSITION']]}]}, ('light.abc123', 'blah.blah', FAKE_UUID), (None,)),
        ({'filter': [{'supported_features': ['light.LightEntityFeature.EFFECT', 'light.LightEntityFeature.TRANSITION']}]}, ('light.abc123', 'blah.blah', FAKE_UUID), (None,))
    ]
)
def test_entity_selector_schema(
    schema: Dict[str, Any],
    valid_selections: Tuple[Union[str, List[str]], ...],
    invalid_selections: Tuple[Optional[Union[str, List[str]]], ...]
) -> None: ...

@pytest.mark.parametrize(
    'schema',
    [
        {'filter': [{'supported_features': [1]}]},
        {'filter': [{'supported_features': ['blah']}]},
        {'filter': [{'supported_features': ['blah.FooEntityFeature.blah']}]},
        {'filter': [{'supported_features': ['light.FooEntityFeature.blah']}]},
        {'filter': [{'supported_features': ['light.LightEntityFeature.blah']}]}
    ]
)
def test_entity_selector_schema_error(schema: Dict[str, Any]) -> None: ...

@pytest.mark.parametrize(
    ('schema', 'valid_selections', 'invalid_selections'),
    [
        ({}, ('abc123',), (None,)),
        ({'entity': {}}, ('abc123',), (None,)),
        ({'entity': {'domain': 'light'}}, ('abc123',), (None,)),
        ({'entity': {'domain': 'binary_sensor', 'device_class': 'motion'}}, ('abc123',), (None,)),
        ({'entity': {'domain': 'binary_sensor', 'device_class': 'motion', 'integration': 'demo'}}, ('abc123',), (None,)),
        ({'entity': [{'domain': 'light'}, {'domain': 'binary_sensor', 'device_class': 'motion'}]}, ('abc123',), (None,)),
        ({'device': {'integration': 'demo', 'model': 'mock-model'}}, ('abc123',), (None,)),
        ({'device': [{'integration': 'demo', 'model': 'mock-model'}, {'integration': 'other-demo', 'model': 'other-mock-model'}]}, ('abc123',), (None,)),
        ({'entity': {'domain': 'binary_sensor', 'device_class': 'motion'}, 'device': {'integration': 'demo', 'model': 'mock-model'}}, ('abc123',), (None,)),
        ({'multiple': True}, (['abc123', 'def456'],), (None, 'abc123', ['abc123', None]))
    ]
)
def test_area_selector_schema(
    schema: Dict[str, Any],
    valid_selections: Tuple[Union[str, List[str]], ...],
    invalid_selections: Tuple[Optional[Union[str, List[Optional[str]]]], ...]
) -> None: ...

@pytest.mark.parametrize(
    ('schema', 'valid_selections', 'invalid_selections'),
    [({}, ('23ouih2iu23ou2', '2j4hp3uy4p87wyrpiuhk34'), (None, True, 1))]
)
def test_assist_pipeline_selector_schema(
    schema: Dict[str, Any],
    valid_selections: Tuple[str, ...],
    invalid_selections: Tuple[Optional[Union[bool, int]], ...]
) -> None: ...

@pytest.mark.parametrize(
    ('schema', 'valid_selections', 'invalid_selections'),
    [
        ({'min': 10, 'max': 50}, (10, 50), (9, 51)),
        ({'min': -100, 'max': 100, 'step': 5}, (), ()),
        ({'min': -20, 'max': -10, 'mode': 'box'}, (), ()),
        ({'min': 0, 'max': 100, 'unit_of_measurement': 'seconds', 'mode': 'slider'}, (), ()),
        ({'min': 10, 'max': 1000, 'mode': 'slider', 'step': 0.5}, (), ()),
        ({'mode': 'box'}, (10,), ()),
        ({'mode': 'box', 'step': 'any'}, (), ()),
        ({'mode': 'slider', 'min': 0, 'max': 1, 'step': 'any'}, (), ())
    ]
)
def test_number_selector_schema(
    schema: Dict[str, Any],
    valid_selections: Tuple[int, ...],
    invalid_selections: Tuple[int, ...]
) -> None: ...

@pytest.mark.parametrize('schema', [{}, {'mode': 'slider'}])
def test_number_selector_schema_error(schema: Dict[str, Any]) -> None: ...

@pytest.mark.parametrize(
    ('schema', 'valid_selections', 'invalid_selections'),
    [({}, ('abc123',), (None,))]
)
def test_addon_selector_schema(
    schema: Dict[str, Any],
    valid_selections: Tuple[str, ...],
    invalid_selections: Tuple[None, ...]
) -> None: ...

@pytest.mark.parametrize(
    ('schema', 'valid_selections', 'invalid_selections'),
    [({}, ('abc123', '/backup'), (None, 'abc@123', 'abc 123', ''))]
)
def test_backup_location_selector_schema(
    schema: Dict[str, Any],
    valid_selections: Tuple[str, ...],
    invalid_selections: Tuple[Optional[str], ...]
) -> None: ...

@pytest.mark.parametrize(
    ('schema', 'valid_selections', 'invalid_selections'),
    [({}, (1, 'one', None), ())]
)
def test_boolean_selector_schema(
    schema: Dict[str, Any],
    valid_selections: Tuple[Union[int, str, None], ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

@pytest.mark.parametrize(
    ('schema', 'valid_selections', 'invalid_selections'),
    [
        ({}, ('6b68b250388cbe0d620c92dd3acc93ec', '76f2e8f9a6491a1b580b3a8967c27ddd'), (None, True, 1)),
        ({'integration': 'adguard'}, ('6b68b250388cbe0d620c92dd3acc93ec', '76f2e8f9a6491a1b580b3a8967c27ddd'), (None, True, 1))
    ]
)
def test_config_entry_selector_schema(
    schema: Dict[str, Any],
    valid_selections: Tuple[str, ...],
    invalid_selections: Tuple[Optional[Union[bool, int]], ...]
) -> None: ...

@pytest.mark.parametrize(
    ('schema', 'valid_selections', 'invalid_selections'),
    [
        ({}, ('NL', 'DE'), (None, True, 1)),
        ({'countries': ['NL', 'DE']}, ('NL', 'DE'), (None, True, 1, 'sv', 'en'))
    ]
)
def test_country_selector_schema(
    schema: Dict[str, Any],
    valid_selections: Tuple[str, ...],
    invalid_selections: Tuple[Optional[Union[bool, int, str]], ...]
) -> None: ...

@pytest.mark.parametrize(
    ('schema', 'valid_selections', 'invalid_selections'),
    [({}, ('00:00:00',), ('blah', None))]
)
def test_time_selector_schema(
    schema: Dict[str, Any],
    valid_selections: Tuple[str, ...],
    invalid_selections: Tuple[Optional[str], ...]
) -> None: ...

@pytest.mark.parametrize(
    ('schema', 'valid_selections', 'invalid_selections'),
    [({'entity_id': 'sensor.abc'}, ('on', 'armed'), (None, True, 1))]
)
def test_state_selector_schema(
    schema: Dict[str, str],
    valid_selections: Tuple[str, ...],
    invalid_selections: Tuple[Optional[Union[bool, int]], ...]
) -> None: ...

@pytest.mark.parametrize(
    ('schema', 'valid_selections', 'invalid_selections'),
    [
        ({}, ({'entity_id': ['sensor.abc123']},), ('abc123', None)),
        ({'entity': {}}, (), ()),
        ({'entity': {'domain': 'light'}}, (), ()),
        ({'entity': {'domain': 'binary_sensor', 'device_class': 'motion'}}, (), ()),
        ({'entity': [{'domain': 'light'}, {'domain': 'binary_sensor', 'device_class': 'motion'}]}, (), ()),
        ({'entity': {'domain': 'binary_sensor', 'device_class': 'motion', 'integration': 'demo'}}, (), ()),
        ({'device': {'integration': 'demo', 'model': 'mock-model'}}, (), ()),
        ({'device': [{'integration': 'demo', 'model': 'mock-model'}, {'integration': 'other-demo', 'model': 'other-mock-model'}]}, (), ()),
        ({'entity': {'domain': 'binary_sensor', 'device_class': 'motion'}, 'device': {'integration': 'demo', 'model': 'mock-model'}}, (), ())
    ]
)
def test_target_selector_schema(
    schema: Dict[str, Any],
    valid_selections: Tuple[Dict[str, List[str]], ...],
    invalid_selections: Tuple[Optional[str], ...]
) -> None: ...

@pytest.mark.parametrize(
    ('schema', 'valid_selections', 'invalid_selections'),
    [({}, ('abc123',), ())]
)
def test_action_selector_schema(
    schema: Dict[str, Any],
    valid_selections: Tuple[str, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

@pytest.mark.parametrize(
    ('schema', 'valid_selections', 'invalid_selections'),
    [({}, ('abc123',), ())]
)
def test_object_selector_schema(
    schema: Dict[str, Any],
    valid_selections: Tuple[str, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

@pytest.mark.parametrize(
    ('schema', 'valid_selections', 'invalid_selections'),
    [
        ({}, ('abc123',), (None,)),
        ({'multiline': True}, (), ()),
        ({'multiline': False, 'type': 'email'}, (), ()),
        ({'prefix': 'before', 'suffix': 'after'}, (), ()),
        ({'multiple': True}, (['abc123', 'def456'],), ('abc123', None, ['abc123', None]))
    ]
)
def test_text_selector_schema(
    schema: Dict[str, Any],
    valid_selections: Tuple[Union[str, List[str]], ...],
    invalid_selections: Tuple[Optional[Union[str, List[Optional[str]]]], ...]
) -> None: ...

@pytest.mark.parametrize(
    ('schema', 'valid_selections', 'invalid_selections'),
    [
        ({'options': ['red', 'green', 'blue']}, ('red', 'green', 'blue'), ('cat', 0, None, ['red'])),
        ({'options': [{'value': 'red', 'label': 'Ruby Red'}, {'value': 'green', 'label': 'Emerald Green'}]}, ('red', 'green'), ('cat', 0, None, ['red'])),
        ({'options': ['red', 'green', 'blue'], 'translation_key': 'color'}, ('red', 'green', 'blue'), ('cat', 0, None,