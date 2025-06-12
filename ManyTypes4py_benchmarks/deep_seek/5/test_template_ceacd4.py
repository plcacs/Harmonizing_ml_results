"""Test Home Assistant template helper methods."""
from __future__ import annotations
from collections.abc import Iterable
from datetime import datetime, timedelta
import json
import logging
import math
import random
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast
from unittest.mock import patch
from freezegun import freeze_time
import orjson
import pytest
from syrupy import SnapshotAssertion
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.components import group
from homeassistant.const import ATTR_UNIT_OF_MEASUREMENT, STATE_ON, STATE_UNAVAILABLE, UnitOfArea, UnitOfLength, UnitOfMass, UnitOfPrecipitationDepth, UnitOfPressure, UnitOfSpeed, UnitOfTemperature, UnitOfVolume
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import TemplateError
from homeassistant.helpers import area_registry as ar, device_registry as dr, entity, entity_registry as er, floor_registry as fr, issue_registry as ir, label_registry as lr, template, translation
from homeassistant.helpers.entity_platform import EntityPlatform
from homeassistant.helpers.json import json_dumps
from homeassistant.helpers.typing import TemplateVarsType
from homeassistant.setup import async_setup_component
from homeassistant.util import dt as dt_util
from homeassistant.util.read_only_dict import ReadOnlyDict
from homeassistant.util.unit_system import UnitSystem
from tests.common import MockConfigEntry, async_fire_time_changed

def _set_up_units(hass: HomeAssistant) -> None:
    """Set up the tests."""
    hass.config.units = UnitSystem('custom', accumulated_precipitation=UnitOfPrecipitationDepth.MILLIMETERS, area=UnitOfArea.SQUARE_METERS, conversions={}, length=UnitOfLength.METERS, mass=UnitOfMass.GRAMS, pressure=UnitOfPressure.PA, temperature=UnitOfTemperature.CELSIUS, volume=UnitOfVolume.LITERS, wind_speed=UnitOfSpeed.KILOMETERS_PER_HOUR)

def render(hass: HomeAssistant, template_str: str, variables: Optional[Dict[str, Any]] = None) -> Any:
    """Create render info from template."""
    tmp = template.Template(template_str, hass)
    return tmp.async_render(variables)

def render_to_info(hass: HomeAssistant, template_str: str, variables: Optional[Dict[str, Any]] = None) -> template.RenderInfo:
    """Create render info from template."""
    tmp = template.Template(template_str, hass)
    return tmp.async_render_to_info(variables)

def extract_entities(hass: HomeAssistant, template_str: str, variables: Optional[Dict[str, Any]] = None) -> Set[str]:
    """Extract entities from a template."""
    info = render_to_info(hass, template_str, variables)
    return info.entities

def assert_result_info(info: template.RenderInfo, result: Any, entities: Optional[List[str]] = None, domains: Optional[List[str]] = None, all_states: bool = False) -> None:
    """Check result info."""
    assert info.result() == result
    assert info.all_states == all_states
    assert info.filter('invalid_entity_name.somewhere') == all_states
    if entities is not None:
        assert info.entities == frozenset(entities)
        assert all((info.filter(entity) for entity in entities))
        if not all_states:
            assert not info.filter('invalid_entity_name.somewhere')
    else:
        assert not info.entities
    if domains is not None:
        assert info.domains == frozenset(domains)
        assert all((info.filter(domain + '.entity') for domain in domains))
    else:
        assert not hasattr(info, '_domains')

async def test_template_render_missing_hass(hass: HomeAssistant) -> None:
    """Test template render when hass is not set."""
    hass.states.async_set('sensor.test', '23')
    template_str = "{{ states('sensor.test') }}"
    template_obj = template.Template(template_str, None)
    template._render_info.set(template.RenderInfo(template_obj))
    with pytest.raises(RuntimeError, match='hass not set while rendering'):
        template_obj.async_render_to_info()

async def test_template_render_info_collision(hass: HomeAssistant) -> None:
    """Test template render info collision.

    This usually means the template is being rendered
    in the wrong thread.
    """
    hass.states.async_set('sensor.test', '23')
    template_str = "{{ states('sensor.test') }}"
    template_obj = template.Template(template_str, None)
    template_obj.hass = hass
    template._render_info.set(template.RenderInfo(template_obj))
    with pytest.raises(RuntimeError, match='RenderInfo already set while rendering'):
        template_obj.async_render_to_info()

def test_template_equality() -> None:
    """Test template comparison and hashing."""
    template_one = template.Template('{{ template_one }}')
    template_one_1 = template.Template('{{ template_one }}')
    template_two = template.Template('{{ template_two }}')
    assert template_one == template_one_1
    assert template_one != template_two
    assert hash(template_one) == hash(template_one_1)
    assert hash(template_one) != hash(template_two)
    assert str(template_one_1) == 'Template<template=({{ template_one }}) renders=0>'
    with pytest.raises(TypeError):
        template.Template(['{{ template_one }}'])

def test_invalid_template(hass: HomeAssistant) -> None:
    """Invalid template raises error."""
    tmpl = template.Template('{{', hass)
    with pytest.raises(TemplateError):
        tmpl.ensure_valid()
    with pytest.raises(TemplateError):
        tmpl.async_render()
    info = tmpl.async_render_to_info()
    with pytest.raises(TemplateError):
        assert info.result() == 'impossible'
    tmpl = template.Template('{{states(keyword)}}', hass)
    tmpl.ensure_valid()
    with pytest.raises(TemplateError):
        tmpl.async_render()

def test_referring_states_by_entity_id(hass: HomeAssistant) -> None:
    """Test referring states by entity id."""
    hass.states.async_set('test.object', 'happy')
    assert template.Template('{{ states.test.object.state }}', hass).async_render() == 'happy'
    assert template.Template('{{ states["test.object"].state }}', hass).async_render() == 'happy'
    assert template.Template('{{ states("test.object") }}', hass).async_render() == 'happy'

def test_invalid_entity_id(hass: HomeAssistant) -> None:
    """Test referring states by entity id."""
    with pytest.raises(TemplateError):
        template.Template('{{ states["big.fat..."] }}', hass).async_render()
    with pytest.raises(TemplateError):
        template.Template('{{ states.test["big.fat..."] }}', hass).async_render()
    with pytest.raises(TemplateError):
        template.Template('{{ states["invalid/domain"] }}', hass).async_render()

def test_raise_exception_on_error(hass: HomeAssistant) -> None:
    """Test raising an exception on error."""
    with pytest.raises(TemplateError):
        template.Template('{{ invalid_syntax').ensure_valid()

def test_iterating_all_states(hass: HomeAssistant) -> None:
    """Test iterating all states."""
    tmpl_str = "{% for state in states | sort(attribute='entity_id') %}{{ state.state }}{% endfor %}"
    info = render_to_info(hass, tmpl_str)
    assert_result_info(info, '', all_states=True)
    assert info.rate_limit == template.ALL_STATES_RATE_LIMIT
    hass.states.async_set('test.object', 'happy')
    hass.states.async_set('sensor.temperature', 10)
    info = render_to_info(hass, tmpl_str)
    assert_result_info(info, '10happy', entities=[], all_states=True)

def test_iterating_all_states_unavailable(hass: HomeAssistant) -> None:
    """Test iterating all states unavailable."""
    hass.states.async_set('test.object', 'on')
    tmpl_str = "{{  states  | selectattr('state', 'in', ['unavailable', 'unknown', 'none'])  | list  | count}}"
    info = render_to_info(hass, tmpl_str)
    assert info.all_states is True
    assert info.rate_limit == template.ALL_STATES_RATE_LIMIT
    hass.states.async_set('test.object', 'unknown')
    hass.states.async_set('sensor.temperature', 10)
    info = render_to_info(hass, tmpl_str)
    assert_result_info(info, 1, entities=[], all_states=True)

def test_iterating_domain_states(hass: HomeAssistant) -> None:
    """Test iterating domain states."""
    tmpl_str = '{% for state in states.sensor %}{{ state.state }}{% endfor %}'
    info = render_to_info(hass, tmpl_str)
    assert_result_info(info, '', domains=['sensor'])
    assert info.rate_limit == template.DOMAIN_STATES_RATE_LIMIT
    hass.states.async_set('test.object', 'happy')
    hass.states.async_set('sensor.back_door', 'open')
    hass.states.async_set('sensor.temperature', 10)
    info = render_to_info(hass, tmpl_str)
    assert_result_info(info, 'open10', entities=[], domains=['sensor'])

async def test_import(hass: HomeAssistant) -> None:
    """Test that imports work from the config/custom_templates folder."""
    await template.async_load_custom_templates(hass)
    assert 'test.jinja' in template._get_hass_loader(hass).sources
    assert 'inner/inner_test.jinja' in template._get_hass_loader(hass).sources
    assert template.Template("\n            {% import 'test.jinja' as t %}\n            {{ t.test_macro() }} {{ t.test_variable }}\n            ", hass).async_render() == 'macro variable'
    assert template.Template("\n            {% import 'inner/inner_test.jinja' as t %}\n            {{ t.test_macro() }} {{ t.test_variable }}\n            ", hass).async_render() == 'inner macro inner variable'
    with pytest.raises(TemplateError):
        template.Template("\n            {% import 'notfound.jinja' as t %}\n            {{ t.test_macro() }} {{ t.test_variable }}\n            ", hass).async_render()

async def test_import_change(hass: HomeAssistant) -> None:
    """Test that a change in HassLoader results in updated imports."""
    await template.async_load_custom_templates(hass)
    to_test = template.Template("\n        {% import 'test.jinja' as t %}\n        {{ t.test_macro() }} {{ t.test_variable }}\n        ", hass)
    assert to_test.async_render() == 'macro variable'
    template._get_hass_loader(hass).sources = {'test.jinja': '\n            {% macro test_macro() -%}\n            macro2\n            {%- endmacro %}\n\n            {% set test_variable = "variable2" %}\n            '}
    assert to_test.async_render() == 'macro2 variable2'

def test_loop_controls(hass: HomeAssistant) -> None:
    """Test that loop controls are enabled."""
    assert template.Template('\n            {%- for v in range(10) %}\n                {%- if v == 1 -%}\n                    {%- continue -%}\n                {%- elif v == 3 -%}\n                    {%- break -%}\n                {%- endif -%}\n                {{ v }}\n            {%- endfor -%}\n            ', hass).async_render() == '02'

def test_float_function(hass: HomeAssistant) -> None:
    """Test float function."""
    hass.states.async_set('sensor.temperature', '12')
    assert template.Template('{{ float(states.sensor.temperature.state) }}', hass).async_render() == 12.0
    assert template.Template('{{ float(states.sensor.temperature.state) > 11 }}', hass).async_render() is True
    with pytest.raises(TemplateError):
        template.Template("{{ float('forgiving') }}", hass).async_render()
    assert render(hass, "{{ float('bad', 1) }}") == 1
    assert render(hass, "{{ float('bad', default=1) }}") == 1

def test_float_filter(hass: HomeAssistant) -> None:
    """Test float filter."""
    hass.states.async_set('sensor.temperature', '12')
    assert render(hass, '{{ states.sensor.temperature.state | float }}') == 12.0
    assert render(hass, '{{ states.sensor.temperature.state | float > 11 }}') is True
    with pytest.raises(TemplateError):
        render(hass, "{{ 'bad' | float }}")
    assert render(hass, "{{ 'bad' | float(1) }}") == 1
    assert render(hass, "{{ 'bad' | float(default=1) }}") == 1

def test_int_filter(hass: HomeAssistant) -> None:
    """Test int filter."""
    hass.states.async_set('sensor.temperature', '12.2')
    assert render(hass, '{{ states.sensor.temperature.state | int }}') == 12
    assert render(hass, '{{ states.sensor.temperature.state | int > 11 }}') is True
    hass.states.async_set('sensor.temperature', '0x10')
    assert render(hass, '{{ states.sensor.temperature.state | int(base=16) }}') == 16
    with pytest.raises(TemplateError):
        render(hass, "{{ 'bad' | int }}")
    assert render(hass, "{{ 'bad' | int(1) }}") == 1
    assert render(hass, "{{ 'bad' | int(default=1) }}") == 1

def test_int_function(hass: HomeAssistant) -> None:
    """Test int filter."""
    hass.states.async_set('sensor.temperature', '12.2')
    assert render(hass, '{{ int(states.sensor.temperature.state) }}') == 12
    assert render(hass, '{{ int(states.sensor.temperature.state) > 11 }}') is True
    hass.states.async_set('sensor.temperature', '0x10')
    assert render(hass, '{{ int(states.sensor.temperature.state, base=16) }}') == 16
    with pytest.raises(TemplateError):
        render(hass, "{{ int('bad') }}")
    assert render(hass, "{{ int('bad', 1) }}") == 1
    assert render(hass, "{{ int('bad', default=1) }}") == 1

def test_bool_function(hass: HomeAssistant) -> None:
    """Test bool function."""
    assert render(hass, '{{ bool(true) }}') is True
    assert render(hass, '{{ bool(false) }}') is False
    assert render(hass, "{{ bool('on') }}") is True
    assert render(hass, "{{ bool('off') }}") is False
    with pytest.raises(TemplateError):
        render(hass, "{{ bool('unknown') }}")
    with pytest.raises(TemplateError):
        render(hass, '{{ bool(none) }}')
    assert render(hass, "{{ bool('unavailable', none) }}") is None
    assert render(hass, "{{ bool('unavailable', default=none) }}") is None

def test_bool_filter(hass: HomeAssistant) -> None:
    """Test bool filter."""
    assert render(hass, '{{ true | bool }}') is True
    assert render(hass, '{{ false | bool }}') is False
    assert render(hass, "{{ 'on' | bool }}") is True
    assert render(hass, "{{ 'off' | bool }}") is False
    with pytest.raises(TemplateError):
        render(hass, "{{ 'unknown' | bool }}")
    with pytest.raises(TemplateError):
        render(hass, '{{ none | bool }}')
    assert render(hass, "{{ 'unavailable' | bool(none) }}") is None
    assert render(hass, "{{ 'unavailable' | bool(default=none) }}") is None

@pytest.mark.parametrize(('value', 'expected'), [(0, True), (0.0, True), ('0', True), ('0.0', True), (True, True), (False, True), ('True', False), ('False', False), (None, False), ('None', False), ('horse', False), (math.pi, True), (math.nan, False), (math.inf, False), ('nan', False), ('inf', False)])
def test_isnumber(hass: HomeAssistant, value: Any, expected: bool) -> None:
    """Test is_number."""
    assert template.Template('{{ is_number(value) }}', hass).async_render({'value': value}) == expected
    assert template.Template('{{ value | is_number }}', hass).async_render({'value': value}) == expected
    assert template.Template('{{ value is is_number }}', hass).async_render({'value': value}) == expected

@pytest.mark.parametrize(('value', 'expected'), [([1, 2], True), ({1, 2}, False), ({'a': 1, 'b': 2}, False), (ReadOnlyDict({'a': 1, 'b': 2}), False), (MappingProxyType({'a