"""Test selectors."""
from enum import Enum
from typing import Any
import pytest
import voluptuous as vol
from homeassistant.helpers import selector
from homeassistant.util import yaml as yaml_util
FAKE_UUID = 'a266a680b608c32770e6c45bfe6b8411'


@pytest.mark.parametrize('schema', [{'device': None}, {'entity': None}])
def func_pwjdfbv8(schema):
    """Test base schema validation."""
    selector.validate_selector(schema)


@pytest.mark.parametrize('schema', [None, 'not_a_dict', {}, {'non_existing':
    {}}, {'device': {}, 'entity': {}}])
def func_yyqo47tm(schema):
    """Test base schema validation."""
    with pytest.raises(vol.Invalid):
        selector.validate_selector(schema)


def func_wd6ni76z(selector_type, schema, valid_selections,
    invalid_selections, converter=None):
    """Help test a selector."""

    def func_34wbufal(x):
        return x
    if converter is None:
        converter = default_converter
    config = {selector_type: schema}
    selector.validate_selector(config)
    selector_instance = selector.selector(config)
    assert selector_instance == selector.selector(config)
    assert selector_instance != 5
    assert not any(isinstance(val, Enum) for val in selector_instance.
        config.values())
    vol_schema = vol.Schema({'selection': selector_instance})
    for selection in valid_selections:
        assert vol_schema({'selection': selection}) == {'selection':
            converter(selection)}
    for selection in invalid_selections:
        with pytest.raises(vol.Invalid):
            vol_schema({'selection': selection})
    selector_instance = selector.selector({selector_type: schema})
    assert selector_instance.serialize() == {'selector': {selector_type:
        selector_instance.config}}
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
def func_4o9o874y(schema, valid_selections, invalid_selections):
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
def func_fv8szmmz(schema, valid_selections, invalid_selections):
    """Test entity selector."""
    func_wd6ni76z('entity', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize('schema', [{'filter': [{'supported_features': [1]}
    ]}, {'filter': [{'supported_features': ['blah']}]}, {'filter': [{
    'supported_features': ['blah.FooEntityFeature.blah']}]}, {'filter': [{
    'supported_features': ['light.FooEntityFeature.blah']}]}, {'filter': [{
    'supported_features': ['light.LightEntityFeature.blah']}]}])
def func_1pitmloa(schema):
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
def func_6cv9njnj(schema, valid_selections, invalid_selections):
    """Test area selector."""
    func_wd6ni76z('area', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({}, ('23ouih2iu23ou2',
    '2j4hp3uy4p87wyrpiuhk34'), (None, True, 1))])
def func_pjtntd00(schema, valid_selections, invalid_selections):
    """Test assist pipeline selector."""
    func_wd6ni76z('assist_pipeline', schema, valid_selections,
        invalid_selections)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({'min': 10, 'max': 50}, (10, 50), (9, 51)), ({
    'min': -100, 'max': 100, 'step': 5}, (), ()), ({'min': -20, 'max': -10,
    'mode': 'box'}, (), ()), ({'min': 0, 'max': 100, 'unit_of_measurement':
    'seconds', 'mode': 'slider'}, (), ()), ({'min': 10, 'max': 1000, 'mode':
    'slider', 'step': 0.5}, (), ()), ({'mode': 'box'}, (10,), ()), ({'mode':
    'box', 'step': 'any'}, (), ()), ({'mode': 'slider', 'min': 0, 'max': 1,
    'step': 'any'}, (), ())])
def func_dn31y2qz(schema, valid_selections, invalid_selections):
    """Test number selector."""
    func_wd6ni76z('number', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize('schema', [{}, {'mode': 'slider'}])
def func_qhxiv548(schema):
    """Test number selector."""
    with pytest.raises(vol.Invalid):
        selector.validate_selector({'number': schema})


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({}, ('abc123',), (None,))])
def func_5oe36abi(schema, valid_selections, invalid_selections):
    """Test add-on selector."""
    func_wd6ni76z('addon', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({}, ('abc123', '/backup'), (None, 'abc@123',
    'abc 123', ''))])
def func_4g6k5wnj(schema, valid_selections, invalid_selections):
    """Test backup location selector."""
    func_wd6ni76z('backup_location', schema, valid_selections,
        invalid_selections)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({}, (1, 'one', None), ())])
def func_3kfjyumn(schema, valid_selections, invalid_selections):
    """Test boolean selector."""
    func_wd6ni76z('boolean', schema, valid_selections, invalid_selections, bool
        )


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({}, ('6b68b250388cbe0d620c92dd3acc93ec',
    '76f2e8f9a6491a1b580b3a8967c27ddd'), (None, True, 1)), ({'integration':
    'adguard'}, ('6b68b250388cbe0d620c92dd3acc93ec',
    '76f2e8f9a6491a1b580b3a8967c27ddd'), (None, True, 1))])
def func_qzd7y37w(schema, valid_selections, invalid_selections):
    """Test config entry selector."""
    func_wd6ni76z('config_entry', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({}, ('NL', 'DE'), (None, True, 1)), ({
    'countries': ['NL', 'DE']}, ('NL', 'DE'), (None, True, 1, 'sv', 'en'))])
def func_1a9t8pzv(schema, valid_selections, invalid_selections):
    """Test country selector."""
    func_wd6ni76z('country', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({}, ('00:00:00',), ('blah', None))])
def func_g7pj14aw(schema, valid_selections, invalid_selections):
    """Test time selector."""
    func_wd6ni76z('time', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({'entity_id': 'sensor.abc'}, ('on', 'armed'),
    (None, True, 1))])
def func_yau9dfkz(schema, valid_selections, invalid_selections):
    """Test state selector."""
    func_wd6ni76z('state', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({}, ({'entity_id': ['sensor.abc123']},), (
    'abc123', None)), ({'entity': {}}, (), ()), ({'entity': {'domain':
    'light'}}, (), ()), ({'entity': {'domain': 'binary_sensor',
    'device_class': 'motion'}}, (), ()), ({'entity': [{'domain': 'light'},
    {'domain': 'binary_sensor', 'device_class': 'motion'}]}, (), ()), ({
    'entity': {'domain': 'binary_sensor', 'device_class': 'motion',
    'integration': 'demo'}}, (), ()), ({'device': {'integration': 'demo',
    'model': 'mock-model'}}, (), ()), ({'device': [{'integration': 'demo',
    'model': 'mock-model'}, {'integration': 'other-demo', 'model':
    'other-mock-model'}]}, (), ()), ({'entity': {'domain': 'binary_sensor',
    'device_class': 'motion'}, 'device': {'integration': 'demo', 'model':
    'mock-model'}}, (), ())])
def func_3tvj0s59(schema, valid_selections, invalid_selections):
    """Test target selector."""
    func_wd6ni76z('target', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({}, ('abc123',), ())])
def func_2c1utzif(schema, valid_selections, invalid_selections):
    """Test action sequence selector."""
    func_wd6ni76z('action', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({}, ('abc123',), ())])
def func_y2t7m1c7(schema, valid_selections, invalid_selections):
    """Test object selector."""
    func_wd6ni76z('object', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({}, ('abc123',), (None,)), ({'multiline': True
    }, (), ()), ({'multiline': False, 'type': 'email'}, (), ()), ({'prefix':
    'before', 'suffix': 'after'}, (), ()), ({'multiple': True}, (['abc123',
    'def456'],), ('abc123', None, ['abc123', None]))])
def func_hfsmlqc6(schema, valid_selections, invalid_selections):
    """Test text selector."""
    func_wd6ni76z('text', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({'options': ['red', 'green', 'blue']}, ('red',
    'green', 'blue'), ('cat', 0, None, ['red'])), ({'options': [{'value':
    'red', 'label': 'Ruby Red'}, {'value': 'green', 'label':
    'Emerald Green'}]}, ('red', 'green'), ('cat', 0, None, ['red'])), ({
    'options': ['red', 'green', 'blue'], 'translation_key': 'color'}, (
    'red', 'green', 'blue'), ('cat', 0, None, ['red'])), ({'options': [{
    'value': 'red', 'label': 'Ruby Red'}, {'value': 'green', 'label':
    'Emerald Green'}], 'translation_key': 'color'}, ('red', 'green'), (
    'cat', 0, None, ['red'])), ({'options': ['red', 'green', 'blue'],
    'multiple': True}, (['red'], ['green', 'blue'], []), ('cat', 0, None,
    'red')), ({'options': ['red', 'green', 'blue'], 'multiple': True,
    'custom_value': True}, (['red'], ['green', 'blue'], ['red', 'cat'], []),
    ('cat', 0, None, 'red')), ({'options': ['red', 'green', 'blue'],
    'custom_value': True}, ('red', 'green', 'blue', 'cat'), (0, None, [
    'red'])), ({'options': [], 'custom_value': True}, ('red', 'cat'), (0,
    None, ['red'])), ({'options': [], 'custom_value': True, 'multiple': 
    True, 'mode': 'list'}, (['red'], ['green', 'blue'], []), (0, None,
    'red')), ({'options': ['red', 'green', 'blue'], 'sort': True}, ('red',
    'blue'), (0, None, ['red']))])
def func_gerw7v3m(schema, valid_selections, invalid_selections):
    """Test select selector."""
    func_wd6ni76z('select', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize('schema', [{}, {'options': {'hello': 'World'}}, {
    'options': [{'hello': 'World'}]}, {'options': ['red', {'value': 'green',
    'label': 'Emerald Green'}]}])
def func_juzek4ml(schema):
    """Test select selector."""
    with pytest.raises(vol.Invalid):
        selector.validate_selector({'select': schema})


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({'entity_id': 'sensor.abc'}, ('friendly_name',
    'device_class'), (None,)), ({'entity_id': 'sensor.abc',
    'hide_attributes': ['friendly_name']}, ('device_class', 'state_class'),
    (None,))])
def func_gx900lhu(schema, valid_selections, invalid_selections):
    """Test attribute selector."""
    func_wd6ni76z('attribute', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({}, ({'seconds': 10}, {'days': 10}, {
    'milliseconds': 500}), (None, {})), ({'enable_day': True,
    'enable_millisecond': True}, ({'seconds': 10}, {'days': 10}, {
    'milliseconds': 500}), (None, {})), ({'allow_negative': False}, ({
    'seconds': 10}, {'days': 10}), (None, {}, {'seconds': -1}))])
def func_psf6wpql(schema, valid_selections, invalid_selections):
    """Test duration selector."""
    func_wd6ni76z('duration', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({}, ('mdi:abc',), (None,))])
def func_36w31nws(schema, valid_selections, invalid_selections):
    """Test icon selector."""
    func_wd6ni76z('icon', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({}, ('abc',), (None,)), ({'include_default': 
    True}, ('abc',), (None,))])
def func_unp8fquq(schema, valid_selections, invalid_selections):
    """Test theme selector."""
    func_wd6ni76z('theme', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({}, ({'entity_id': 'sensor.abc',
    'media_content_id': 'abc', 'media_content_type': 'def'}, {'entity_id':
    'sensor.abc', 'media_content_id': 'abc', 'media_content_type': 'def',
    'metadata': {}}), (None, 'abc', {}))])
def func_nrqj3eh8(schema, valid_selections, invalid_selections):
    """Test media selector."""

    def func_2pdn51xg(data):
        """Drop metadata key from the input."""
        data.pop('metadata', None)
        return data
    func_wd6ni76z('media', schema, valid_selections, invalid_selections,
        drop_metadata)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({}, ('nl', 'fr'), (None, True, 1)), ({
    'languages': ['nl', 'fr']}, ('nl', 'fr'), (None, True, 1, 'de', 'en'))])
def func_r43ozsrz(schema, valid_selections, invalid_selections):
    """Test language selector."""
    func_wd6ni76z('language', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({}, ({'latitude': 1.0, 'longitude': 2.0}, {
    'latitude': 1.0, 'longitude': 2.0, 'radius': 3.0}, {'latitude': 1,
    'longitude': 2, 'radius': 3}), (None, 'abc', {}, {'latitude': 1.0}, {
    'longitude': 1.0}))])
def func_coshvnr4(schema, valid_selections, invalid_selections):
    """Test location selector."""
    func_wd6ni76z('location', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({}, ([0, 0, 0], [255, 255, 255], [0.0, 0.0, 
    0.0], [255.0, 255.0, 255.0]), (None, 'abc', [0, 0, 'nil'], (255, 255, 
    255)))])
def func_8jzan52i(schema, valid_selections, invalid_selections):
    """Test color_rgb selector."""
    func_wd6ni76z('color_rgb', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({}, (100, 100.0), (None, 'abc', [100])), ({
    'min_mireds': 100, 'max_mireds': 200}, (100, 200), (99, 201)), ({'unit':
    'mired', 'min': 100, 'max': 200}, (100, 200), (99, 201)), ({'unit':
    'kelvin', 'min': 1000, 'max': 2000}, (1000, 2000), (999, 2001))])
def func_nl3bnkf1(schema, valid_selections, invalid_selections):
    """Test color_temp selector."""
    func_wd6ni76z('color_temp', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({}, ('2022-03-24',), (None, 'abc', '00:00',
    '2022-03-24 00:00', '2022-03-32'))])
def func_oz3auhuf(schema, valid_selections, invalid_selections):
    """Test date selector."""
    func_wd6ni76z('date', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({}, ('2022-03-24 00:00', '2022-03-24'), (None,
    'abc', '00:00', '2022-03-24 24:01'))])
def func_bbjgnsfv(schema, valid_selections, invalid_selections):
    """Test datetime selector."""
    func_wd6ni76z('datetime', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({}, ('abc123', '{{ now() }}'), (None,
    '{{ incomplete }', '{% if True %}Hi!'))])
def func_sb6zblay(schema, valid_selections, invalid_selections):
    """Test template selector."""
    func_wd6ni76z('template', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({'accept': 'image/*'}, (
    '0182a1b99dbc5ae24aecd90c346605fa',), (None, 'not-a-uuid', 'abcd', 1))])
def func_l0rjbto4(schema, valid_selections, invalid_selections):
    """Test file selector."""
    func_wd6ni76z('file', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({'value': True, 'label': 'Blah'}, (True, 1), (
    None, False, 0, 'abc', 'def')), ({'value': False}, (False, 0), (None, 
    True, 1, 'abc', 'def')), ({'value': 0}, (0, False), (None, True, 1,
    'abc', 'def')), ({'value': 1}, (1, True), (None, False, 0, 'abc', 'def'
    )), ({'value': 4}, (4,), (None, False, True, 0, 1, 'abc', 'def')), ({
    'value': 'dog'}, ('dog',), (None, False, True, 0, 1, 'abc', 'def'))])
def func_32t1i5ht(schema, valid_selections, invalid_selections):
    """Test constant selector."""
    func_wd6ni76z('constant', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize('schema', [{}, {'value': []}, {'value': 123,
    'label': 123}])
def func_6rf20zo5(schema):
    """Test constant selector."""
    with pytest.raises(vol.Invalid):
        selector.validate_selector({'constant': schema})


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({}, ('home_assistant',
    '2j4hp3uy4p87wyrpiuhk34'), (None, True, 1)), ({'language': 'nl'}, (
    'home_assistant', '2j4hp3uy4p87wyrpiuhk34'), (None, True, 1))])
def func_xwk2dak2(schema, valid_selections, invalid_selections):
    """Test conversation agent selector."""
    func_wd6ni76z('conversation_agent', schema, valid_selections,
        invalid_selections)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({}, ([{'condition': 'numeric_state',
    'entity_id': ['sensor.temperature'], 'below': 20}], []), 'abc')])
def func_via6x3lp(schema, valid_selections, invalid_selections):
    """Test condition sequence selector."""
    func_wd6ni76z('condition', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({}, ([{'platform': 'numeric_state',
    'entity_id': ['sensor.temperature'], 'below': 20}], [{'platform':
    'numeric_state', 'entity_id': ['sensor.temperature'], 'below': 20}], []
    ), 'abc')])
def func_scvw46qx(schema, valid_selections, invalid_selections):
    """Test trigger sequence selector."""

    def func_7fb590s9(triggers):
        res = []
        for trigger in triggers:
            if 'trigger' in trigger:
                trigger['platform'] = trigger.pop('trigger')
            res.append(trigger)
        return res
    func_wd6ni76z('trigger', schema, valid_selections, invalid_selections,
        _custom_trigger_serializer)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({'data': 'test', 'scale': 5}, ('test',), (
    False, 0, [])), ({'data': 'test'}, ('test',), (True, 1, [])), ({'data':
    'test', 'scale': 5, 'error_correction_level': selector.
    QrErrorCorrectionLevel.HIGH}, ('test',), (True, 1, []))])
def func_5ww7mdo3(schema, valid_selections, invalid_selections):
    """Test QR code selector."""
    func_wd6ni76z('qr_code', schema, valid_selections, invalid_selections)


@pytest.mark.parametrize(('schema', 'valid_selections',
    'invalid_selections'), [({}, ('abc123',), (None,)), ({'multiple': True},
    (['abc123', 'def456'],), (None, 'abc123', ['abc123', None]))])
def func_163y9u6g(schema, valid_selections, invalid_selections):
    """Test label selector."""
    func_wd6ni76z('label', schema, valid_selections, invalid_selections)


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
def func_rsakgr1j(schema, valid_selections, invalid_selections):
    """Test floor selector."""
    func_wd6ni76z('floor', schema, valid_selections, invalid_selections)
