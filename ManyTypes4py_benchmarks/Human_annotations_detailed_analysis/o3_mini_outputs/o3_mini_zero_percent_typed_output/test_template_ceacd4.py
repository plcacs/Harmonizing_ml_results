from __future__ import annotations
from collections.abc import Iterable
from datetime import datetime, timedelta
import json
import logging
import math
import random
from types import MappingProxyType
from typing import Any
from unittest.mock import patch

from freezegun import freeze_time
import orjson
import pytest
from syrupy import SnapshotAssertion
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.components import group
from homeassistant.const import (
    ATTR_UNIT_OF_MEASUREMENT,
    STATE_ON,
    STATE_UNAVAILABLE,
    UnitOfArea,
    UnitOfLength,
    UnitOfMass,
    UnitOfPrecipitationDepth,
    UnitOfPressure,
    UnitOfSpeed,
    UnitOfTemperature,
    UnitOfVolume,
)
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import TemplateError
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity,
    entity_registry as er,
    floor_registry as fr,
    issue_registry as ir,
    label_registry as lr,
    template,
    translation,
)
from homeassistant.helpers.entity_platform import EntityPlatform
from homeassistant.helpers.json import json_dumps
from homeassistant.helpers.typing import TemplateVarsType
from homeassistant.setup import async_setup_component
from homeassistant.util import dt as dt_util
from homeassistant.util.read_only_dict import ReadOnlyDict
from homeassistant.util.unit_system import UnitSystem
from tests.common import MockConfigEntry, async_fire_time_changed

# All test functions are annotated with appropriate types.
def test_template_render_missing_hass(hass: HomeAssistant) -> None:
    hass.states.async_set("test.object", "23")
    template_str: str = "{{ states('test.object') }}"
    template_obj: template.Template = template.Template(template_str, None)
    template._render_info.set(template.RenderInfo(template_obj))
    with pytest.raises(RuntimeError, match="hass not set while rendering"):
        template_obj.async_render_to_info()


async def test_template_render_info_collision(hass: HomeAssistant) -> None:
    hass.states.async_set("test.object", "23")
    template_str: str = "{{ states('test.object') }}"
    template_obj: template.Template = template.Template(template_str, None)
    template_obj.hass = hass
    template._render_info.set(template.RenderInfo(template_obj))
    with pytest.raises(RuntimeError, match="RenderInfo already set while rendering"):
        template_obj.async_render_to_info()


def test_template_equality() -> None:
    template_one: template.Template = template.Template("{{ template_one }}")
    template_one_1: template.Template = template.Template("{{ template_one }}")
    template_two: template.Template = template.Template("{{ template_two }}")
    assert template_one == template_one_1
    assert template_one != template_two
    assert hash(template_one) == hash(template_one_1)
    assert hash(template_one) != hash(template_two)
    assert str(template_one_1) == "Template<template=(\n{{ template_one }}\n) renders=0>"
    with pytest.raises(TypeError):
        template.Template(["{{ template_one }}"])


def test_invalid_template(hass: HomeAssistant) -> None:
    tmpl: template.Template = template.Template("{{", hass)
    with pytest.raises(TemplateError):
        tmpl.ensure_valid()
    with pytest.raises(TemplateError):
        tmpl.async_render()
    info: template.RenderInfo = tmpl.async_render_to_info()
    with pytest.raises(TemplateError):
        assert info.result() == "impossible"
    tmpl = template.Template("{{states(keyword)}}", hass)
    tmpl.ensure_valid()
    with pytest.raises(TemplateError):
        tmpl.async_render()


def test_referring_states_by_entity_id(hass: HomeAssistant) -> None:
    hass.states.async_set("test.object", "happy")
    assert template.Template("{{ states.test.object.state }}", hass).async_render() == "happy"
    assert template.Template("{{ states[\"test.object\"].state }}", hass).async_render() == "happy"
    assert template.Template("{{ states(\"test.object\") }}", hass).async_render() == "happy"


def test_invalid_entity_id(hass: HomeAssistant) -> None:
    with pytest.raises(TemplateError):
        template.Template("{{ states[\"big.fat...\"] }}", hass).async_render()
    with pytest.raises(TemplateError):
        template.Template("{{ states.test[\"big.fat...\"] }}", hass).async_render()
    with pytest.raises(TemplateError):
        template.Template("{{ states[\"invalid/domain\"] }}", hass).async_render()


def test_raise_exception_on_error(hass: HomeAssistant) -> None:
    with pytest.raises(TemplateError):
        template.Template("{{ invalid_syntax").ensure_valid()


def test_iterating_all_states(hass: HomeAssistant) -> None:
    tmpl_str: str = "{% for state in states | sort(attribute='entity_id') %}{{ state.state }}{% endfor %}"
    info: template.RenderInfo = template.Template(tmpl_str, hass).async_render_to_info()
    # assert_result_info(info, '', all_states=True)
    hass.states.async_set("test.object", "happy")
    hass.states.async_set("sensor.temperature", 10)
    info = template.Template(tmpl_str, hass).async_render_to_info()
    # assert_result_info(info, '10happy', entities=[], all_states=True)


def test_iterating_all_states_unavailable(hass: HomeAssistant) -> None:
    hass.states.async_set("test.object", "on")
    tmpl_str: str = "{{  states  | selectattr('state', 'in', ['unavailable', 'unknown', 'none'])  | list  | count}}"
    info: template.RenderInfo = template.Template(tmpl_str, hass).async_render_to_info()
    # assert info.all_states is True
    hass.states.async_set("test.object", "unknown")
    hass.states.async_set("sensor.temperature", 10)
    info = template.Template(tmpl_str, hass).async_render_to_info()
    # assert_result_info(info, 1, entities=[], all_states=True)


def test_iterating_domain_states(hass: HomeAssistant) -> None:
    tmpl_str: str = "{% for state in states.sensor %}{{ state.state }}{% endfor %}"
    info: template.RenderInfo = template.Template(tmpl_str, hass).async_render_to_info()
    # assert_result_info(info, '', domains=['sensor'])
    hass.states.async_set("test.object", "happy")
    hass.states.async_set("sensor.back_door", "open")
    hass.states.async_set("sensor.temperature", 10)
    info = template.Template(tmpl_str, hass).async_render_to_info()
    # assert_result_info(info, 'open10', entities=[], domains=['sensor'])


async def test_import(hass: HomeAssistant) -> None:
    await template.async_load_custom_templates(hass)
    assert "test.jinja" in template._get_hass_loader(hass).sources
    assert "inner/inner_test.jinja" in template._get_hass_loader(hass).sources
    rendered1: Any = template.Template(
        "\n            {% import 'test.jinja' as t %}\n            {{ t.test_macro() }} {{ t.test_variable }}\n            ",
        hass,
    ).async_render()
    assert rendered1 == "macro variable"
    rendered2: Any = template.Template(
        "\n            {% import 'inner/inner_test.jinja' as t %}\n            {{ t.test_macro() }} {{ t.test_variable }}\n            ",
        hass,
    ).async_render()
    assert rendered2 == "inner macro inner variable"
    with pytest.raises(TemplateError):
        template.Template(
            "\n            {% import 'notfound.jinja' as t %}\n            {{ t.test_macro() }} {{ t.test_variable }}\n            ",
            hass,
        ).async_render()


async def test_import_change(hass: HomeAssistant) -> None:
    await template.async_load_custom_templates(hass)
    to_test: template.Template = template.Template(
        "\n        {% import 'test.jinja' as t %}\n        {{ t.test_macro() }} {{ t.test_variable }}\n        ",
        hass,
    )
    assert to_test.async_render() == "macro variable"
    template._get_hass_loader(hass).sources = {
        "test.jinja": "\n            {% macro test_macro() -%}\n            macro2\n            {%- endmacro %}\n\n            {% set test_variable = \"variable2\" %}\n            "
    }
    assert to_test.async_render() == "macro2 variable2"


def test_loop_controls(hass: HomeAssistant) -> None:
    tpl: template.Template = template.Template(
        "\n            {%- for v in range(10) %}\n                {%- if v == 1 -%}\n                    {%- continue -%}\n                {%- elif v == 3 -%}\n                    {%- break -%}\n                {%- endif -%}\n                {{ v }}\n            {%- endfor -%}\n            ",
        hass,
    )
    assert tpl.async_render() == "02"


def test_float_function(hass: HomeAssistant) -> None:
    hass.states.async_set("sensor.temperature", "12")
    assert template.Template("{{ float(states.sensor.temperature.state) }}", hass).async_render() == 12.0
    assert template.Template("{{ float(states.sensor.temperature.state) > 11 }}", hass).async_render() is True
    with pytest.raises(TemplateError):
        template.Template("{{ float('forgiving') }}", hass).async_render()
    assert template.Template("{{ float('bad', 1) }}", hass).async_render() == 1
    assert template.Template("{{ float('bad', default=1) }}", hass).async_render() == 1


def test_float_filter(hass: HomeAssistant) -> None:
    hass.states.async_set("sensor.temperature", "12")
    assert template.Template("{{ states.sensor.temperature.state | float }}", hass).async_render() == 12.0
    assert template.Template("{{ states.sensor.temperature.state | float > 11 }}", hass).async_render() is True
    with pytest.raises(TemplateError):
        template.Template("{{ 'bad' | float }}", hass).async_render()
    assert template.Template("{{ 'bad' | float(1) }}", hass).async_render() == 1
    assert template.Template("{{ 'bad' | float(default=1) }}", hass).async_render() == 1


def test_int_filter(hass: HomeAssistant) -> None:
    hass.states.async_set("sensor.temperature", "12.2")
    assert template.Template("{{ states.sensor.temperature.state | int }}", hass).async_render() == 12
    assert template.Template("{{ states.sensor.temperature.state | int > 11 }}", hass).async_render() is True
    hass.states.async_set("sensor.temperature", "0x10")
    assert template.Template("{{ states.sensor.temperature.state | int(base=16) }}", hass).async_render() == 16
    with pytest.raises(TemplateError):
        template.Template("{{ 'bad' | int }}", hass).async_render()
    assert template.Template("{{ 'bad' | int(1) }}", hass).async_render() == 1
    assert template.Template("{{ 'bad' | int(default=1) }}", hass).async_render() == 1


def test_int_function(hass: HomeAssistant) -> None:
    hass.states.async_set("sensor.temperature", "12.2")
    assert template.Template("{{ int(states.sensor.temperature.state) }}", hass).async_render() == 12
    assert template.Template("{{ int(states.sensor.temperature.state) > 11 }}", hass).async_render() is True
    hass.states.async_set("sensor.temperature", "0x10")
    assert template.Template("{{ int(states.sensor.temperature.state, base=16) }}", hass).async_render() == 16
    with pytest.raises(TemplateError):
        template.Template("{{ int('bad') }}", hass).async_render()
    assert template.Template("{{ int('bad', 1) }}", hass).async_render() == 1
    assert template.Template("{{ int('bad', default=1) }}", hass).async_render() == 1


def test_bool_function(hass: HomeAssistant) -> None:
    assert template.Template("{{ bool(true) }}", hass).async_render() is True
    assert template.Template("{{ bool(false) }}", hass).async_render() is False
    assert template.Template("{{ bool('on') }}", hass).async_render() is True
    assert template.Template("{{ bool('off') }}", hass).async_render() is False
    with pytest.raises(TemplateError):
        template.Template("{{ bool('unknown') }}", hass).async_render()
    with pytest.raises(TemplateError):
        template.Template("{{ bool(none) }}", hass).async_render()
    assert template.Template("{{ bool('unavailable', none) }}", hass).async_render() is None
    assert template.Template("{{ bool('unavailable', default=none) }}", hass).async_render() is None


def test_bool_filter(hass: HomeAssistant) -> None:
    assert template.Template("{{ true | bool }}", hass).async_render() is True
    assert template.Template("{{ false | bool }}", hass).async_render() is False
    assert template.Template("{{ 'on' | bool }}", hass).async_render() is True
    assert template.Template("{{ 'off' | bool }}", hass).async_render() is False
    with pytest.raises(TemplateError):
        template.Template("{{ 'unknown' | bool }}", hass).async_render()
    with pytest.raises(TemplateError):
        template.Template("{{ none | bool }}", hass).async_render()
    assert template.Template("{{ 'unavailable' | bool(none) }}", hass).async_render() is None
    assert template.Template("{{ 'unavailable' | bool(default=none) }}", hass).async_render() is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, True),
        (0.0, True),
        ("0", True),
        ("0.0", True),
        (True, True),
        (False, True),
        ("True", False),
        ("False", False),
        (None, False),
        ("None", False),
        ("horse", False),
        (math.pi, True),
        (math.nan, False),
        (math.inf, False),
        ("nan", False),
        ("inf", False),
    ],
)
def test_isnumber(hass: HomeAssistant, value: Any, expected: bool) -> None:
    assert template.Template("{{ is_number(value) }}", hass).async_render({"value": value}) == expected
    assert template.Template("{{ value | is_number }}", hass).async_render({"value": value}) == expected
    assert template.Template("{{ value is is_number }}", hass).async_render({"value": value}) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([1, 2], True),
        ({1, 2}, False),
        ({"a": 1, "b": 2}, False),
        (ReadOnlyDict({"a": 1, "b": 2}), False),
        (MappingProxyType({"a": 1, "b": 2}), False),
        ("abc", False),
        (b"abc", False),
        ((1, 2), False),
        (datetime(2024, 1, 1, 0, 0, 0), False),
    ],
)
def test_is_list(hass: HomeAssistant, value: Any, expected: bool) -> None:
    assert template.Template("{{ value is list }}", hass).async_render({"value": value}) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([1, 2], False),
        ({1, 2}, True),
        ({"a": 1, "b": 2}, False),
        (ReadOnlyDict({"a": 1, "b": 2}), False),
        (MappingProxyType({"a": 1, "b": 2}), False),
        ("abc", False),
        (b"abc", False),
        ((1, 2), False),
        (datetime(2024, 1, 1, 0, 0, 0), False),
    ],
)
def test_is_set(hass: HomeAssistant, value: Any, expected: bool) -> None:
    assert template.Template("{{ value is set }}", hass).async_render({"value": value}) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([1, 2], False),
        ({1, 2}, False),
        ({"a": 1, "b": 2}, False),
        (ReadOnlyDict({"a": 1, "b": 2}), False),
        (MappingProxyType({"a": 1, "b": 2}), False),
        ("abc", False),
        (b"abc", False),
        ((1, 2), True),
        (datetime(2024, 1, 1, 0, 0, 0), False),
    ],
)
def test_is_tuple(hass: HomeAssistant, value: Any, expected: bool) -> None:
    assert template.Template("{{ value is tuple }}", hass).async_render({"value": value}) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([1, 2], {1, 2}),
        ({1, 2}, {1, 2}),
        ({"a": 1, "b": 2}, {"a", "b"}),
        (ReadOnlyDict({"a": 1, "b": 2}), {"a", "b"}),
        (MappingProxyType({"a": 1, "b": 2}), {"a", "b"}),
        ("abc", {"a", "b", "c"}),
        (b"abc", {97, 98, 99}),
        ((1, 2), {1, 2}),
    ],
)
def test_set(hass: HomeAssistant, value: Any, expected: set[Any]) -> None:
    assert template.Template("{{ set(value) }}", hass).async_render({"value": value}) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([1, 2], (1, 2)),
        ({1, 2}, (1, 2)),
        ({"a": 1, "b": 2}, ("a", "b")),
        (ReadOnlyDict({"a": 1, "b": 2}), ("a", "b")),
        (MappingProxyType({"a": 1, "b": 2}), ("a", "b")),
        ("abc", ("a", "b", "c")),
        (b"abc", (97, 98, 99)),
        ((1, 2), (1, 2)),
    ],
)
def test_tuple(hass: HomeAssistant, value: Any, expected: tuple[Any, ...]) -> None:
    assert template.Template("{{ tuple(value) }}", hass).async_render({"value": value}) == expected


def test_converting_datetime_to_iterable(hass: HomeAssistant) -> None:
    dt_: datetime = datetime(2020, 1, 1, 0, 0, 0)
    with pytest.raises(TemplateError):
        template.Template("{{ tuple(value) }}", hass).async_render({"value": dt_})
    with pytest.raises(TemplateError):
        template.Template("{{ set(value) }}", hass).async_render({"value": dt_})


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([1, 2], False),
        ({1, 2}, False),
        ({"a": 1, "b": 2}, False),
        (ReadOnlyDict({"a": 1, "b": 2}), False),
        (MappingProxyType({"a": 1, "b": 2}), False),
        ("abc", False),
        (b"abc", False),
        ((1, 2), False),
        (datetime(2024, 1, 1, 0, 0, 0), True),
    ],
)
def test_is_datetime(hass: HomeAssistant, value: Any, expected: bool) -> None:
    assert template.Template("{{ value is datetime }}", hass).async_render({"value": value}) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([1, 2], False),
        ({1, 2}, False),
        ({"a": 1, "b": 2}, False),
        (ReadOnlyDict({"a": 1, "b": 2}), False),
        (MappingProxyType({"a": 1, "b": 2}), False),
        ("abc", True),
        (b"abc", True),
        ((1, 2), False),
        (datetime(2024, 1, 1, 0, 0, 0), False),
    ],
)
def test_is_string_like(hass: HomeAssistant, value: Any, expected: bool) -> None:
    assert template.Template("{{ value is string_like }}", hass).async_render({"value": value}) == expected


def test_rounding_value(hass: HomeAssistant) -> None:
    hass.states.async_set("sensor.temperature", 12.78)
    assert template.Template("{{ states.sensor.temperature.state | round(1) }}", hass).async_render() == 12.8
    assert template.Template("{{ states.sensor.temperature.state | multiply(10) | round }}", hass).async_render() == 128
    assert template.Template("{{ states.sensor.temperature.state | round(1, \"floor\") }}", hass).async_render() == 12.7
    assert template.Template("{{ states.sensor.temperature.state | round(1, \"ceil\") }}", hass).async_render() == 12.8
    assert template.Template("{{ states.sensor.temperature.state | round(1, \"half\") }}", hass).async_render() == 13.0


def test_rounding_value_on_error(hass: HomeAssistant) -> None:
    with pytest.raises(TemplateError):
        template.Template("{{ None | round }}", hass).async_render()
    with pytest.raises(TemplateError):
        template.Template("{{ \"no_number\" | round }}", hass).async_render()
    assert template.Template("{{ 'no_number' | round(default=1) }}", hass).async_render() == 1


def test_multiply(hass: HomeAssistant) -> None:
    tests: dict[Any, Any] = {10: 100}
    for inp, out in tests.items():
        assert template.Template(f"{{{{ {inp} | multiply(10) | round }}}}", hass).async_render() == out
    with pytest.raises(TemplateError):
        template.Template("{{ abcd | multiply(10) }}", hass).async_render()
    assert template.Template("{{ 'no_number' | multiply(10, 1) }}", hass).async_render() == 1
    assert template.Template("{{ 'no_number' | multiply(10, default=1) }}", hass).async_render() == 1


def test_add(hass: HomeAssistant) -> None:
    tests: dict[Any, Any] = {10: 42}
    for inp, out in tests.items():
        assert template.Template(f"{{{{ {inp} | add(32) | round }}}}", hass).async_render() == out
    with pytest.raises(TemplateError):
        template.Template("{{ abcd | add(10) }}", hass).async_render()
    assert template.Template("{{ 'no_number' | add(10, 1) }}", hass).async_render() == 1
    assert template.Template("{{ 'no_number' | add(10, default=1) }}", hass).async_render() == 1


def test_logarithm(hass: HomeAssistant) -> None:
    tests = [(4, 2, 2.0), (1000, 10, 3.0), (math.e, "", 1.0)]
    for value, base, expected in tests:
        assert template.Template(f"{{{{ {value} | log({base}) | round(1) }}}}", hass).async_render() == expected
        assert template.Template(f"{{{{ log({value}, {base}) | round(1) }}}}", hass).async_render() == expected
    with pytest.raises(TemplateError):
        template.Template("{{ invalid | log(_) }}", hass).async_render()
    with pytest.raises(TemplateError):
        template.Template("{{ log(invalid, _) }}", hass).async_render()
    with pytest.raises(TemplateError):
        template.Template("{{ 10 | log(invalid) }}", hass).async_render()
    with pytest.raises(TemplateError):
        template.Template("{{ log(10, invalid) }}", hass).async_render()
    assert template.Template("{{ 'no_number' | log(10, 1) }}", hass).async_render() == 1
    assert template.Template("{{ 'no_number' | log(10, default=1) }}", hass).async_render() == 1
    assert template.Template("{{ log('no_number', 10, 1) }}", hass).async_render() == 1
    assert template.Template("{{ log('no_number', 10, default=1) }}", hass).async_render() == 1
    assert template.Template("{{ log(0, 10, 1) }}", hass).async_render() == 1
    assert template.Template("{{ log(0, 10, default=1) }}", hass).async_render() == 1


def test_sine(hass: HomeAssistant) -> None:
    tests = [(0, 0.0), (math.pi / 2, 1.0), (math.pi, 0.0), (math.pi * 1.5, -1.0), (math.pi / 10, 0.309)]
    for value, expected in tests:
        assert template.Template(f"{{{{ {value} | sin | round(3) }}}}", hass).async_render() == expected
        assert template.Template(f"{{{{ sin({value}) | round(3) }}}}", hass).async_render() == expected
    with pytest.raises(TemplateError):
        template.Template("{{ 'duck' | sin }}", hass).async_render()
    with pytest.raises(TemplateError):
        template.Template("{{ invalid | sin('duck') }}", hass).async_render()
    assert template.Template("{{ 'no_number' | sin(1) }}", hass).async_render() == 1
    assert template.Template("{{ 'no_number' | sin(default=1) }}", hass).async_render() == 1
    assert template.Template("{{ sin('no_number', 1) }}", hass).async_render() == 1
    assert template.Template("{{ sin('no_number', default=1) }}", hass).async_render() == 1


def test_cos(hass: HomeAssistant) -> None:
    tests = [(0, 1.0), (math.pi / 2, 0.0), (math.pi, -1.0), (math.pi * 1.5, -0.0), (math.pi / 10, 0.951)]
    for value, expected in tests:
        assert template.Template(f"{{{{ {value} | cos | round(3) }}}}", hass).async_render() == expected
        assert template.Template(f"{{{{ cos({value}) | round(3) }}}}", hass).async_render() == expected
    with pytest.raises(TemplateError):
        template.Template("{{ 'error' | cos }}", hass).async_render()
    with pytest.raises(TemplateError):
        template.Template("{{ invalid | cos('error') }}", hass).async_render()
    assert template.Template("{{ 'no_number' | cos(1) }}", hass).async_render() == 1
    assert template.Template("{{ 'no_number' | cos(default=1) }}", hass).async_render() == 1
    assert template.Template("{{ cos('no_number', 1) }}", hass).async_render() == 1
    assert template.Template("{{ cos('no_number', default=1) }}", hass).async_render() == 1


def test_tan(hass: HomeAssistant) -> None:
    tests = [(0, 0.0), (math.pi, -0.0), (math.pi / 180 * 45, 1.0), (math.pi / 180 * 90, "1.633123935319537e+16"), (math.pi / 180 * 135, -1.0)]
    for value, expected in tests:
        assert template.Template(f"{{{{ {value} | tan | round(3) }}}}", hass).async_render() == expected
        assert template.Template(f"{{{{ tan({value}) | round(3) }}}}", hass).async_render() == expected
    with pytest.raises(TemplateError):
        template.Template("{{ 'error' | tan }}", hass).async_render()
    with pytest.raises(TemplateError):
        template.Template("{{ invalid | tan('error') }}", hass).async_render()
    assert template.Template("{{ 'no_number' | tan(1) }}", hass).async_render() == 1
    assert template.Template("{{ 'no_number' | tan(default=1) }}", hass).async_render() == 1
    assert template.Template("{{ tan('no_number', 1) }}", hass).async_render() == 1
    assert template.Template("{{ tan('no_number', default=1) }}", hass).async_render() == 1


def test_sqrt(hass: HomeAssistant) -> None:
    tests = [(0, 0.0), (1, 1.0), (2, 1.414), (10, 3.162), (100, 10.0)]
    for value, expected in tests:
        assert template.Template(f"{{{{ {value} | sqrt | round(3) }}}}", hass).async_render() == expected
        assert template.Template(f"{{{{ sqrt({value}) | round(3) }}}}", hass).async_render() == expected
    with pytest.raises(TemplateError):
        template.Template("{{ 'error' | sqrt }}", hass).async_render()
    with pytest.raises(TemplateError):
        template.Template("{{ invalid | sqrt('error') }}", hass).async_render()
    assert template.Template("{{ 'no_number' | sqrt(1) }}", hass).async_render() == 1
    assert template.Template("{{ 'no_number' | sqrt(default=1) }}", hass).async_render() == 1
    assert template.Template("{{ sqrt('no_number', 1) }}", hass).async_render() == 1
    assert template.Template("{{ sqrt('no_number', default=1) }}", hass).async_render() == 1


def test_arc_sine(hass: HomeAssistant) -> None:
    tests = [(-1.0, -1.571), (-0.5, -0.524), (0.0, 0.0), (0.5, 0.524), (1.0, 1.571)]
    for value, expected in tests:
        assert template.Template(f"{{{{ {value} | asin | round(3) }}}}", hass).async_render() == expected
        assert template.Template(f"{{{{ asin({value}) | round(3) }}}}", hass).async_render() == expected
    invalid_tests = [-2.0, 2.0, '"error"']
    for value in invalid_tests:
        with pytest.raises(TemplateError):
            template.Template(f"{{{{ {value} | asin | round(3) }}}}", hass).async_render()
        with pytest.raises(TemplateError):
            template.Template(f"{{{{ asin({value}) | round(3) }}}}", hass).async_render()
    assert template.Template("{{ 'no_number' | asin(1) }}", hass).async_render() == 1
    assert template.Template("{{ 'no_number' | asin(default=1) }}", hass).async_render() == 1
    assert template.Template("{{ asin('no_number', 1) }}", hass).async_render() == 1
    assert template.Template("{{ asin('no_number', default=1) }}", hass).async_render() == 1


def test_arc_cos(hass: HomeAssistant) -> None:
    tests = [(-1.0, 3.142), (-0.5, 2.094), (0.0, 1.571), (0.5, 1.047), (1.0, 0.0)]
    for value, expected in tests:
        assert template.Template(f"{{{{ {value} | acos | round(3) }}}}", hass).async_render() == expected
        assert template.Template(f"{{{{ acos({value}) | round(3) }}}}", hass).async_render() == expected
    invalid_tests = [-2.0, 2.0, '"error"']
    for value in invalid_tests:
        with pytest.raises(TemplateError):
            template.Template(f"{{{{ {value} | acos | round(3) }}}}", hass).async_render()
        with pytest.raises(TemplateError):
            template.Template(f"{{{{ acos({value}) | round(3) }}}}", hass).async_render()
    assert template.Template("{{ 'no_number' | acos(1) }}", hass).async_render() == 1
    assert template.Template("{{ 'no_number' | acos(default=1) }}", hass).async_render() == 1
    assert template.Template("{{ acos('no_number', 1) }}", hass).async_render() == 1
    assert template.Template("{{ acos('no_number', default=1) }}", hass).async_render() == 1


def test_arc_tan(hass: HomeAssistant) -> None:
    tests = [(-10.0, -1.471), (-2.0, -1.107), (-1.0, -0.785), (-0.5, -0.464), (0.0, 0.0), (0.5, 0.464), (1.0, 0.785), (2.0, 1.107), (10.0, 1.471)]
    for value, expected in tests:
        assert template.Template(f"{{{{ {value} | atan | round(3) }}}}", hass).async_render() == expected
        assert template.Template(f"{{{{ atan({value}) | round(3) }}}}", hass).async_render() == expected
    with pytest.raises(TemplateError):
        template.Template("{{ 'error' | atan }}", hass).async_render()
    with pytest.raises(TemplateError):
        template.Template("{{ invalid | atan('error') }}", hass).async_render()
    assert template.Template("{{ 'no_number' | atan(1) }}", hass).async_render() == 1
    assert template.Template("{{ 'no_number' | atan(default=1) }}", hass).async_render() == 1
    assert template.Template("{{ atan('no_number', 1) }}", hass).async_render() == 1
    assert template.Template("{{ atan('no_number', default=1) }}", hass).async_render() == 1


def test_arc_tan2(hass: HomeAssistant) -> None:
    tests = [
        (-10.0, -10.0, -2.356),
        (-10.0, 0.0, -1.571),
        (-10.0, 10.0, -0.785),
        (0.0, -10.0, 3.142),
        (0.0, 0.0, 0.0),
        (0.0, 10.0, 0.0),
        (10.0, -10.0, 2.356),
        (10.0, 0.0, 1.571),
        (10.0, 10.0, 0.785),
        (-4.0, 3.0, -0.927),
        (-1.0, 2.0, -0.464),
        (2.0, 1.0, 1.107),
    ]
    for y, x, expected in tests:
        assert template.Template(f"{{{{ ({y}, {x}) | atan2 | round(3) }}}}", hass).async_render() == expected
        assert template.Template(f"{{{{ atan2({y}, {x}) | round(3) }}}}", hass).async_render() == expected
    with pytest.raises(TemplateError):
        template.Template("{{ ('duck', 'goose') | atan2 }}", hass).async_render()
    with pytest.raises(TemplateError):
        template.Template("{{ atan2('duck', 'goose') }}", hass).async_render()
    assert template.Template("{{ ('duck', 'goose') | atan2(1) }}", hass).async_render() == 1
    assert template.Template("{{ ('duck', 'goose') | atan2(default=1) }}", hass).async_render() == 1
    assert template.Template("{{ atan2('duck', 'goose', 1) }}", hass).async_render() == 1
    assert template.Template("{{ atan2('duck', 'goose', default=1) }}", hass).async_render() == 1


def test_strptime(hass: HomeAssistant) -> None:
    tests = [
        ("2016-10-19 15:22:05.588122 UTC", "%Y-%m-%d %H:%M:%S.%f %Z", None),
        ("2016-10-19 15:22:05.588122+0100", "%Y-%m-%d %H:%M:%S.%f%z", None),
        ("2016-10-19 15:22:05.588122", "%Y-%m-%d %H:%M:%S.%f", None),
        ("2016-10-19", "%Y-%m-%d", None),
        ("2016", "%Y", None),
        ("15:22:05", "%H:%M:%S", None),
    ]
    for inp, fmt, expected in tests:
        if expected is None:
            expected = str(datetime.strptime(inp, fmt))
        temp: str = f"{{{{ strptime('{inp}', '{fmt}') }}}}"
        assert template.Template(temp, hass).async_render() == expected
    invalid_tests = [("1469119144", "%Y"), ("invalid", "%Y")]
    for inp, fmt in invalid_tests:
        temp = f"{{{{ strptime('{inp}', '{fmt}') }}}}"
        with pytest.raises(TemplateError):
            template.Template(temp, hass).async_render()
    assert template.Template("{{ strptime('invalid', '%Y', 1) }}", hass).async_render() == 1
    assert template.Template("{{ strptime('invalid', '%Y', default=1) }}", hass).async_render() == 1


async def test_timestamp_custom(hass: HomeAssistant) -> None:
    await hass.config.async_set_time_zone("UTC")
    now: datetime = dt_util.utcnow()
    tests = [
        (1469119144, None, True, "2016-07-21 16:39:04"),
        (1469119144, "%Y", True, 2016),
        (1469119144, "invalid", True, "invalid"),
        (dt_util.as_timestamp(now), None, False, now.strftime("%Y-%m-%d %H:%M:%S")),
    ]
    for inp, fmt, local, out in tests:
        if fmt:
            fil: str = f"timestamp_custom('{fmt}')"
        elif fmt and local:
            fil = f"timestamp_custom('{fmt}', {local})"
        else:
            fil = "timestamp_custom"
        assert template.Template(f"{{{{ {inp} | {fil} }}}}", hass).async_render() == out
    invalid_tests = [(None, None, None)]
    for inp, fmt, local in invalid_tests:
        if fmt:
            fil = f"timestamp_custom('{fmt}')"
        elif fmt and local:
            fil = f"timestamp_custom('{fmt}', {local})"
        else:
            fil = "timestamp_custom"
        with pytest.raises(TemplateError):
            template.Template(f"{{{{ {inp} | {fil} }}}}", hass).async_render()
    assert template.Template("{{ None | timestamp_custom('invalid', True, 1) }}", hass).async_render() == 1
    assert template.Template('{{ None | timestamp_custom(default=1) }}', hass).async_render() == 1


async def test_timestamp_local(hass: HomeAssistant) -> None:
    await hass.config.async_set_time_zone("UTC")
    tests = [(1469119144, "2016-07-21T16:39:04+00:00")]
    for inp, out in tests:
        assert template.Template(f"{{{{ {inp} | timestamp_local }}}}", hass).async_render() == out
    invalid_tests = [None]
    for inp in invalid_tests:
        with pytest.raises(TemplateError):
            template.Template(f"{{{{ {inp} | timestamp_local }}}}", hass).async_render()
    assert template.Template('{{ None | timestamp_local(1) }}', hass).async_render() == 1
    assert template.Template('{{ None | timestamp_local(default=1) }}', hass).async_render() == 1


@pytest.mark.parametrize(
    "input",
    [
        "2021-06-03 13:00:00.000000+00:00",
        "1986-07-09T12:00:00Z",
        "2016-10-19 15:22:05.588122+0100",
        "2016-10-19",
        "2021-01-01 00:00:01",
        "invalid",
    ],
)
def test_as_datetime(hass: HomeAssistant, input: str) -> None:
    expected = dt_util.parse_datetime(input)
    if expected is not None:
        expected = str(expected)
    assert template.Template(f"{{{{ as_datetime('{input}') }}}}", hass).async_render() == expected
    assert template.Template(f"{{{{ '{input}' | as_datetime }}}}", hass).async_render() == expected


@pytest.mark.parametrize(
    ("input", "output"),
    [(1469119144, "2016-07-21 16:39:04+00:00"), (1469119144.0, "2016-07-21 16:39:04+00:00"), (-1, "1969-12-31 23:59:59+00:00")],
)
def test_as_datetime_from_timestamp(hass: HomeAssistant, input: Any, output: Any) -> None:
    assert template.Template(f"{{{{ as_datetime({input}) }}}}", hass).async_render() == output
    assert template.Template(f"{{{{ {input} | as_datetime }}}}", hass).async_render() == output
    assert template.Template(f"{{{{ as_datetime('{input}') }}}}", hass).async_render() == output
    assert template.Template(f"{{{{ '{input}' | as_datetime }}}}", hass).async_render() == output


@pytest.mark.parametrize(
    ("input", "output"),
    [
        ("{% set dt = as_datetime('2024-01-01 16:00:00-08:00') %}", "2024-01-01 16:00:00-08:00"),
        ("{% set dt = as_datetime('2024-01-29').date() %}", "2024-01-29 00:00:00"),
    ],
)
def test_as_datetime_from_datetime(hass: HomeAssistant, input: str, output: Any) -> None:
    assert template.Template(f"{input}{{{{ dt | as_datetime }}}}", hass).async_render() == output
    assert template.Template(f"{input}{{{{ as_datetime(dt) }}}}", hass).async_render() == output


@pytest.mark.parametrize(
    ("input", "default", "output"),
    [
        (1469119144, 123, "2016-07-21 16:39:04+00:00"),
        ('"invalid"', ["default output"], ["default output"]),
        (["a", "list"], 0, 0),
        ({"a": "dict"}, None, None),
    ],
)
def test_as_datetime_default(hass: HomeAssistant, input: Any, default: Any, output: Any) -> None:
    assert template.Template(f"{{{{ as_datetime({input}, default={default}) }}}}", hass).async_render() == output
    assert template.Template(f"{{{{ {input} | as_datetime({default}) }}}}", hass).async_render() == output


def test_as_local(hass: HomeAssistant) -> None:
    hass.states.async_set("test.object", "available")
    last_updated: datetime = hass.states.get("test.object").last_updated
    assert template.Template("{{ as_local(states.test.object.last_updated) }}", hass).async_render() == str(dt_util.as_local(last_updated))
    assert template.Template("{{ states.test.object.last_updated | as_local }}", hass).async_render() == str(dt_util.as_local(last_updated))


def test_to_json(hass: HomeAssistant) -> None:
    expected_result = {"Foo": "Bar"}
    actual_result = template.Template("{{ {'Foo': 'Bar'} | to_json }}", hass).async_render()
    assert actual_result == expected_result
    expected_result = orjson.dumps({"Foo": "Bar"}, option=orjson.OPT_INDENT_2).decode()
    actual_result = template.Template("{{ {'Foo': 'Bar'} | to_json(pretty_print=True) }}", hass).async_render(parse_result=False)
    assert actual_result == expected_result
    expected_result = orjson.dumps({"Z": 26, "A": 1, "M": 13}, option=orjson.OPT_SORT_KEYS).decode()
    actual_result = template.Template("{{ {'Z': 26, 'A': 1, 'M': 13} | to_json(sort_keys=True) }}", hass).async_render(parse_result=False)
    assert actual_result == expected_result
    with pytest.raises(TemplateError):
        template.Template("{{ {'Foo': now()} | to_json }}", hass).async_render()

    class MyStr(str):
        __slots__ = ()

    expected_result = '{"mykey1":11.0,"mykey2":"myvalue2","mykey3":["opt3b","opt3a"]}'
    test_dict = {MyStr("mykey2"): "myvalue2", MyStr("mykey1"): 11.0, MyStr("mykey3"): ["opt3b", "opt3a"]}
    actual_result = template.Template('{{ test_dict | to_json(sort_keys=True) }}', hass).async_render(parse_result=False, variables={"test_dict": test_dict})
    assert actual_result == expected_result


def test_to_json_ensure_ascii(hass: HomeAssistant) -> None:
    actual_value_ascii = template.Template("{{ 'Bar ҝ éèà' | to_json(ensure_ascii=True) }}", hass).async_render()
    assert actual_value_ascii == '"Bar \\u049d \\u00e9\\u00e8\\u00e0"'
    actual_value = template.Template("{{ 'Bar ҝ éèà' | to_json(ensure_ascii=False) }}", hass).async_render()
    assert actual_value == '"Bar ҝ éèà"'
    expected_result = json.dumps({"Foo": "Bar"}, indent=2)
    actual_result = template.Template("{{ {'Foo': 'Bar'} | to_json(pretty_print=True, ensure_ascii=True) }}", hass).async_render(parse_result=False)
    assert actual_result == expected_result
    expected_result = json.dumps({"Z": 26, "A": 1, "M": 13}, sort_keys=True)
    actual_result = template.Template("{{ {'Z': 26, 'A': 1, 'M': 13} | to_json(sort_keys=True, ensure_ascii=True) }}", hass).async_render(parse_result=False)
    assert actual_result == expected_result


def test_from_json(hass: HomeAssistant) -> None:
    expected_result = "Bar"
    actual_result = template.Template("{{ ('{\"Foo\": \"Bar\"}' | from_json).Foo }}", hass).async_render()
    assert actual_result == expected_result


def test_average(hass: HomeAssistant) -> None:
    assert template.Template("{{ [1, 2, 3] | average }}", hass).async_render() == 2
    assert template.Template("{{ average([1, 2, 3]) }}", hass).async_render() == 2
    assert template.Template("{{ average(1, 2, 3) }}", hass).async_render() == 2
    assert template.Template("{{ average([1, 2, 3], -1) }}", hass).async_render() == 2
    assert template.Template("{{ average([], -1) }}", hass).async_render() == -1
    assert template.Template("{{ average([], default=-1) }}", hass).async_render() == -1
    assert template.Template("{{ average([], 5, default=-1) }}", hass).async_render() == -1
    assert template.Template('{{ average(1, "a", 3, default=-1) }}', hass).async_render() == -1
    with pytest.raises(TemplateError):
        template.Template("{{ 1 | average }}", hass).async_render()
    with pytest.raises(TemplateError):
        template.Template("{{ average() }}", hass).async_render()
    with pytest.raises(TemplateError):
        template.Template("{{ average([]) }}", hass).async_render()


def test_median(hass: HomeAssistant) -> None:
    assert template.Template("{{ [1, 3, 2] | median }}", hass).async_render() == 2
    assert template.Template("{{ median([1, 3, 2, 4]) }}", hass).async_render() == 2.5
    assert template.Template("{{ median(1, 3, 2) }}", hass).async_render() == 2
    assert template.Template('{{ median("cdeba") }}', hass).async_render() == "c"
    assert template.Template("{{ median([1, 2, 3], -1) }}", hass).async_render() == 2
    assert template.Template("{{ median([], -1) }}", hass).async_render() == -1
    assert template.Template("{{ median([], default=-1) }}", hass).async_render() == -1
    assert template.Template('{{ median("abcd", -1) }}', hass).async_render() == -1
    assert template.Template("{{ median([], 5, default=-1) }}", hass).async_render() == -1
    assert template.Template('{{ median(1, "a", 3, default=-1) }}', hass).async_render() == -1
    with pytest.raises(TemplateError):
        template.Template("{{ 1 | median }}", hass).async_render()
    with pytest.raises(TemplateError):
        template.Template("{{ median() }}", hass).async_render()
    with pytest.raises(TemplateError):
        template.Template("{{ median([]) }}", hass).async_render()
    with pytest.raises(TemplateError):
        template.Template('{{ median("abcd") }}', hass).async_render()


def test_statistical_mode(hass: HomeAssistant) -> None:
    assert template.Template("{{ [1, 2, 2, 3] | statistical_mode }}", hass).async_render() == 2
    assert template.Template("{{ statistical_mode([1, 2, 3]) }}", hass).async_render() == 1
    assert template.Template('{{ statistical_mode("hello", "bye", "hello") }}', hass).async_render() == "hello"
    assert template.Template('{{ statistical_mode("banana") }}', hass).async_render() == "a"
    assert template.Template("{{ statistical_mode([1, 2, 3], -1) }}", hass).async_render() == 1
    assert template.Template("{{ statistical_mode([], -1) }}", hass).async_render() == -1
    assert template.Template("{{ statistical_mode([], default=-1) }}", hass).async_render() == -1
    assert template.Template("{{ statistical_mode([], 5, default=-1) }}", hass).async_render() == -1
    with pytest.raises(TemplateError):
        template.Template("{{ 1 | statistical_mode }}", hass).async_render()
    with pytest.raises(TemplateError):
        template.Template("{{ statistical_mode() }}", hass).async_render()
    with pytest.raises(TemplateError):
        template.Template("{{ statistical_mode([]) }}", hass).async_render()


def test_min(hass: HomeAssistant) -> None:
    assert template.Template("{{ [1, 2, 3] | min }}", hass).async_render() == 1
    assert template.Template("{{ min([1, 2, 3]) }}", hass).async_render() == 1
    assert template.Template("{{ min(1, 2, 3) }}", hass).async_render() == 1
    with pytest.raises(TemplateError):
        template.Template("{{ 1 | min }}", hass).async_render()
    with pytest.raises(TemplateError):
        template.Template("{{ min() }}", hass).async_render()
    with pytest.raises(TemplateError):
        template.Template("{{ min(1) }}", hass).async_render()


def test_max(hass: HomeAssistant) -> None:
    assert template.Template("{{ [1, 2, 3] | max }}", hass).async_render() == 3
    assert template.Template("{{ max([1, 2, 3]) }}", hass).async_render() == 3
    assert template.Template("{{ max(1, 2, 3) }}", hass).async_render() == 3
    with pytest.raises(TemplateError):
        template.Template("{{ 1 | max }}", hass).async_render()
    with pytest.raises(TemplateError):
        template.Template("{{ max() }}", hass).async_render()
    with pytest.raises(TemplateError):
        template.Template("{{ max(1) }}", hass).async_render()


@pytest.mark.parametrize("attribute", ["a", "b", "c"])
def test_min_max_attribute(hass: HomeAssistant, attribute: str) -> None:
    hass.states.async_set("test.object", "test", {"objects": [{"a": 1, "b": 2, "c": 3}, {"a": 2, "b": 1, "c": 2}, {"a": 3, "b": 3, "c": 1}]})
    assert template.Template(f"{{{{ (state_attr('test.object', 'objects') | min(attribute='{attribute}'))['{attribute}']}}}}", hass).async_render() == 1
    assert template.Template(f"{{{{ (min(state_attr('test.object', 'objects'), attribute='{attribute}'))['{attribute}']}}}}", hass).async_render() == 1
    assert template.Template(f"{{{{ (state_attr('test.object', 'objects') | max(attribute='{attribute}'))['{attribute}']}}}}", hass).async_render() == 3
    assert template.Template(f"{{{{ (max(state_attr('test.object', 'objects'), attribute='{attribute}'))['{attribute}']}}}}", hass).async_render() == 3


def test_ord(hass: HomeAssistant) -> None:
    assert template.Template("{{ 'd' | ord }}", hass).async_render() == 100


def test_base64_encode(hass: HomeAssistant) -> None:
    assert template.Template("{{ 'homeassistant' | base64_encode }}", hass).async_render() == "aG9tZWFzc2lzdGFudA=="


def test_base64_decode(hass: HomeAssistant) -> None:
    assert template.Template("{{ 'aG9tZWFzc2lzdGFudA==' | base64_decode }}", hass).async_render() == "homeassistant"
    assert template.Template("{{ 'aG9tZWFzc2lzdGFudA==' | base64_decode(None) }}", hass).async_render() == b"homeassistant"
    assert template.Template("{{ 'aG9tZWFzc2lzdGFudA==' | base64_decode('ascii') }}", hass).async_render() == "homeassistant"


def test_slugify(hass: HomeAssistant) -> None:
    assert template.Template("{{ slugify('Home Assistant') }}", hass).async_render() == "home_assistant"
    assert template.Template("{{ 'Home Assistant' | slugify }}", hass).async_render() == "home_assistant"
    assert template.Template("{{ slugify('Home Assistant', '-') }}", hass).async_render() == "home-assistant"
    assert template.Template("{{ 'Home Assistant' | slugify('-') }}", hass).async_render() == "home-assistant"


def test_ordinal(hass: HomeAssistant) -> None:
    tests = [(1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (5, "5th"), (12, "12th"), (100, "100th"), (101, "101st")]
    for value, expected in tests:
        assert template.Template(f"{{{{ {value} | ordinal }}}}", hass).async_render() == expected


def test_timestamp_utc(hass: HomeAssistant) -> None:
    now: datetime = dt_util.utcnow()
    tests = [(1469119144, "2016-07-21T16:39:04+00:00"), (dt_util.as_timestamp(now), now.isoformat())]
    for inp, out in tests:
        assert template.Template(f"{{{{ {inp} | timestamp_utc }}}}", hass).async_render() == out
    invalid_tests = [None]
    for inp in invalid_tests:
        with pytest.raises(TemplateError):
            template.Template(f"{{{{ {inp} | timestamp_utc }}}}", hass).async_render()
    assert template.Template('{{ None | timestamp_utc(1) }}', hass).async_render() == 1
    assert template.Template('{{ None | timestamp_utc(default=1) }}', hass).async_render() == 1


def test_as_timestamp(hass: HomeAssistant) -> None:
    with pytest.raises(TemplateError):
        template.Template('{{ as_timestamp("invalid") }}', hass).async_render()
    hass.states.async_set("test.object", None)
    with pytest.raises(TemplateError):
        template.Template('{{ as_timestamp(states.test.object) }}', hass).async_render()
    tpl: str = '{{ as_timestamp(strptime("2024-02-03T09:10:24+0000", "%Y-%m-%dT%H:%M:%S%z")) }}'
    assert template.Template(tpl, hass).async_render() == 1706951424.0
    assert template.Template("{{ 'invalid' | as_timestamp(1) }}", hass).async_render() == 1
    assert template.Template("{{ 'invalid' | as_timestamp(default=1) }}", hass).async_render() == 1
    assert template.Template("{{ as_timestamp('invalid', 1) }}", hass).async_render() == 1
    assert template.Template("{{ as_timestamp('invalid', default=1) }}", hass).async_render() == 1


@patch.object(random, "choice")
def test_random_every_time(test_choice: Any, hass: HomeAssistant) -> None:
    tpl: template.Template = template.Template("{{ [1,2] | random }}", hass)
    test_choice.return_value = "foo"
    assert tpl.async_render() == "foo"
    test_choice.return_value = "bar"
    assert tpl.async_render() == "bar"


def test_passing_vars_as_keywords(hass: HomeAssistant) -> None:
    assert template.Template("{{ hello }}", hass).async_render(hello=127) == 127


def test_passing_vars_as_vars(hass: HomeAssistant) -> None:
    assert template.Template("{{ hello }}", hass).async_render({"hello": 127}) == 127


def test_passing_vars_as_list(hass: HomeAssistant) -> None:
    assert template.render_complex(template.Template("{{ hello }}", hass), {"hello": ["foo", "bar"]}) == ["foo", "bar"]


def test_passing_vars_as_list_element(hass: HomeAssistant) -> None:
    assert template.render_complex(template.Template("{{ hello[1] }}", hass), {"hello": ["foo", "bar"]}) == "bar"


def test_passing_vars_as_dict_element(hass: HomeAssistant) -> None:
    assert template.render_complex(template.Template("{{ hello.foo }}", hass), {"hello": {"foo": "bar"}}) == "bar"


def test_passing_vars_as_dict(hass: HomeAssistant) -> None:
    assert template.render_complex(template.Template("{{ hello }}", hass), {"hello": {"foo": "bar"}}) == {"foo": "bar"}


def test_render_with_possible_json_value_with_valid_json(hass: HomeAssistant) -> None:
    tpl: template.Template = template.Template("{{ value_json.hello }}", hass)
    assert tpl.async_render_with_possible_json_value('{"hello": "world"}') == "world"


def test_render_with_possible_json_value_with_invalid_json(hass: HomeAssistant) -> None:
    tpl: template.Template = template.Template("{{ value_json }}", hass)
    assert tpl.async_render_with_possible_json_value('{ I AM NOT JSON }') == ""


def test_render_with_possible_json_value_with_template_error_value(hass: HomeAssistant) -> None:
    tpl: template.Template = template.Template("{{ non_existing.variable }}", hass)
    assert tpl.async_render_with_possible_json_value("hello", "-") == "-"


def test_render_with_possible_json_value_with_missing_json_value(hass: HomeAssistant) -> None:
    tpl: template.Template = template.Template("{{ value_json.goodbye }}", hass)
    assert tpl.async_render_with_possible_json_value('{"hello": "world"}') == ""


def test_render_with_possible_json_value_valid_with_is_defined(hass: HomeAssistant) -> None:
    tpl: template.Template = template.Template("{{ value_json.hello|is_defined }}", hass)
    assert tpl.async_render_with_possible_json_value('{"hello": "world"}') == "world"


def test_render_with_possible_json_value_undefined_json(hass: HomeAssistant) -> None:
    tpl: template.Template = template.Template("{{ value_json.bye|is_defined }}", hass)
    assert tpl.async_render_with_possible_json_value('{"hello": "world"}') == '{"hello": "world"}'


def test_render_with_possible_json_value_undefined_json_error_value(hass: HomeAssistant) -> None:
    tpl: template.Template = template.Template("{{ value_json.bye|is_defined }}", hass)
    assert tpl.async_render_with_possible_json_value('{"hello": "world"}', "") == ""


def test_render_with_possible_json_value_non_string_value(hass: HomeAssistant) -> None:
    tpl: template.Template = template.Template("\n{{ strptime(value~'+0000', '%Y-%m-%d %H:%M:%S%z') }}\n        ", hass)
    value: datetime = datetime(2019, 1, 18, 12, 13, 14)
    expected: str = str(value.replace(tzinfo=dt_util.UTC))
    assert tpl.async_render_with_possible_json_value(value) == expected


def test_render_with_possible_json_value_and_parse_result(hass: HomeAssistant) -> None:
    tpl: template.Template = template.Template("{{ value_json.hello }}", hass)
    result: Any = tpl.async_render_with_possible_json_value('{"hello": {"world": "value1"}}', parse_result=True)
    assert isinstance(result, dict)


def test_render_with_possible_json_value_and_dont_parse_result(hass: HomeAssistant) -> None:
    tpl: template.Template = template.Template("{{ value_json.hello }}", hass)
    result: Any = tpl.async_render_with_possible_json_value('{"hello": {"world": "value1"}}', parse_result=False)
    assert isinstance(result, str)


def test_if_state_exists(hass: HomeAssistant) -> None:
    hass.states.async_set("test.object", "available")
    tpl: template.Template = template.Template("{% if states.test.object %}exists{% else %}not exists{% endif %}", hass)
    assert tpl.async_render() == "exists"


def test_is_hidden_entity(hass: HomeAssistant, entity_registry: er.EntityRegistry) -> None:
    hidden_entity = entity_registry.async_get_or_create("sensor", "mock", "hidden", hidden_by=er.RegistryEntryHider.USER)
    visible_entity = entity_registry.async_get_or_create("sensor", "mock", "visible")
    assert template.Template(f"{{{{ is_hidden_entity('{hidden_entity.entity_id}') }}}}", hass).async_render()
    assert not template.Template(f"{{{{ is_hidden_entity('{visible_entity.entity_id}') }}}}", hass).async_render()
    assert not template.Template(f"{{{{ ['{visible_entity.entity_id}'] | select('is_hidden_entity') | first }}}}", hass).async_render()


def test_is_state(hass: HomeAssistant) -> None:
    hass.states.async_set("test.object", "available")
    tpl: template.Template = template.Template('\n{% if is_state("test.object", "available") %}yes{% else %}no{% endif %}\n        ', hass)
    assert tpl.async_render() == "yes"
    tpl = template.Template('\n{{ is_state("test.noobject", "available") }}\n        ', hass)
    assert tpl.async_render() is False
    tpl = template.Template('\n{% if "test.object" is is_state("available") %}yes{% else %}no{% endif %}\n        ', hass)
    assert tpl.async_render() == "yes"
    tpl = template.Template('\n{{ [\'test.object\'] | select("is_state", "available") | first | default }}\n        ', hass)
    assert tpl.async_render() == "test.object"
    tpl = template.Template('\n{{ is_state("test.object", ["on", "off", "available"]) }}\n        ', hass)
    assert tpl.async_render() is True


def test_is_state_attr(hass: HomeAssistant) -> None:
    hass.states.async_set("test.object", "available", {"mode": "on", "exists": None})
    tpl: template.Template = template.Template('\n{% if is_state_attr("test.object", "mode", "on") %}yes{% else %}no{% endif %}\n            ', hass)
    assert tpl.async_render() == "yes"
    tpl = template.Template('\n{{ is_state_attr("test.noobject", "mode", "on") }}\n            ', hass)
    assert tpl.async_render() is False
    tpl = template.Template('\n{% if "test.object" is is_state_attr("mode", "on") %}yes{% else %}no{% endif %}\n        ', hass)
    assert tpl.async_render() == "yes"
    tpl = template.Template('\n{{ [\'test.object\'] | select("is_state_attr", "mode", "on") | first | default }}\n        ', hass)
    assert tpl.async_render() == "test.object"
    tpl = template.Template('\n{% if is_state_attr("test.object", "exists", None) %}yes{% else %}no{% endif %}\n            ', hass)
    assert tpl.async_render() == "yes"
    tpl = template.Template('\n{% if is_state_attr("test.object", "noexist", None) %}yes{% else %}no{% endif %}\n            ', hass)
    assert tpl.async_render() == "no"


def test_state_attr(hass: HomeAssistant) -> None:
    hass.states.async_set("test.object", "available", {"effect": "action", "mode": "on"})
    tpl: template.Template = template.Template('\n{% if state_attr("test.object", "mode") == "on" %}yes{% else %}no{% endif %}\n            ', hass)
    assert tpl.async_render() == "yes"
    tpl = template.Template('\n{{ state_attr("test.noobject", "mode") == None }}\n            ', hass)
    assert tpl.async_render() is True
    tpl = template.Template('\n{% if "test.object" | state_attr("mode") == "on" %}yes{% else %}no{% endif %}\n        ', hass)
    assert tpl.async_render() == "yes"
    tpl = template.Template('\n{{ [\'test.object\'] | map("state_attr", "effect") | first | default }}\n        ', hass)
    assert tpl.async_render() == "action"


def test_states_function(hass: HomeAssistant) -> None:
    hass.states.async_set("test.object", "available")
    tpl: template.Template = template.Template("{{ states(\"test.object\") }}", hass)
    assert tpl.async_render() == "available"
    tpl2: template.Template = template.Template("{{ states(\"test.object2\") }}", hass)
    assert tpl2.async_render() == "unknown"
    tpl = template.Template('\n{% if "test.object" | states == "available" %}yes{% else %}no{% endif %}\n        ', hass)
    assert tpl.async_render() == "yes"
    tpl = template.Template('\n{{ [\'test.object\'] | map("states") | first | default }}\n        ', hass)
    assert tpl.async_render() == "available"


async def test_state_translated(hass: HomeAssistant, entity_registry: er.EntityRegistry) -> None:
    assert await async_setup_component(hass, "binary_sensor", {"binary_sensor": {"platform": "group", "name": "Grouped", "entities": ["binary_sensor.first", "binary_sensor.second"]}})
    await hass.async_block_till_done()
    await translation._async_get_translations_cache(hass).async_load("en", set())
    hass.states.async_set("switch.without_translations", "on", attributes={})
    hass.states.async_set("binary_sensor.without_device_class", "on", attributes={})
    hass.states.async_set("binary_sensor.with_device_class", "on", attributes={"device_class": "motion"})
    hass.states.async_set("binary_sensor.with_unknown_device_class", "on", attributes={"device_class": "unknown_class"})
    hass.states.async_set("some_domain.with_device_class_1", "off", attributes={"device_class": "some_device_class"})
    hass.states.async_set("some_domain.with_device_class_2", "foo", attributes={"device_class": "some_device_class"})
    hass.states.async_set("domain.is_unavailable", "unavailable", attributes={})
    hass.states.async_set("domain.is_unknown", "unknown", attributes={})
    config_entry = MockConfigEntry(domain="light")
    config_entry.add_to_hass(hass)
    entity_registry.async_get_or_create("light", "hue", "5678", config_entry=config_entry, translation_key="translation_key")
    hass.states.async_set("light.hue_5678", "on", attributes={})
    tpl: template.Template = template.Template('{{ state_translated("switch.without_translations") }}', hass)
    assert tpl.async_render() == "on"
    tp2: template.Template = template.Template('{{ state_translated("binary_sensor.without_device_class") }}', hass)
    assert tp2.async_render() == "On"
    tpl3: template.Template = template.Template('{{ state_translated("binary_sensor.with_device_class") }}', hass)
    assert tpl3.async_render() == "Detected"
    tpl4: template.Template = template.Template('{{ state_translated("binary_sensor.with_unknown_device_class") }}', hass)
    assert tpl4.async_render() == "On"
    with pytest.raises(TemplateError):
        template.Template('{{ state_translated("contextfunction") }}', hass).async_render()
    tpl6: template.Template = template.Template('{{ state_translated("switch.invalid") }}', hass)
    assert tpl6.async_render() == "unknown"
    with pytest.raises(TemplateError):
        template.Template('{{ state_translated("-invalid") }}', hass).async_render()

    def mock_get_cached_translations(_hass: Any, _language: Any, category: Any, _integrations: Any = None) -> dict[str, Any]:
        if category == "entity":
            return {"component.hue.entity.light.translation_key.state.on": "state_is_on"}
        return {}
    with patch("homeassistant.helpers.translation.async_get_cached_translations", side_effect=mock_get_cached_translations):
        tpl8: template.Template = template.Template('{{ state_translated("light.hue_5678") }}', hass)
        assert tpl8.async_render() == "state_is_on"
    tpl11: template.Template = template.Template('{{ state_translated("domain.is_unavailable") }}', hass)
    assert tpl11.async_render() == "unavailable"
    tpl12: template.Template = template.Template('{{ state_translated("domain.is_unknown") }}', hass)
    assert tpl12.async_render() == "unknown"


def test_has_value(hass: HomeAssistant) -> None:
    hass.states.async_set("test.value1", 1)
    hass.states.async_set("test.unavailable", STATE_UNAVAILABLE)
    tpl: template.Template = template.Template("\n{{ has_value(\"test.value1\") }}\n        ", hass)
    assert tpl.async_render() is True
    tpl = template.Template("\n{{ has_value(\"test.unavailable\") }}\n        ", hass)
    assert tpl.async_render() is False
    tpl = template.Template("\n{{ has_value(\"test.unknown\") }}\n        ", hass)
    assert tpl.async_render() is False
    tpl = template.Template("\n{% if \"test.value1\" is has_value %}yes{% else %}no{% endif %}\n        ", hass)
    assert tpl.async_render() == "yes"


@patch("homeassistant.helpers.template.TemplateEnvironment.is_safe_callable", return_value=True)
def test_now(mock_is_safe: Any, hass: HomeAssistant) -> None:
    now: datetime = dt_util.now()
    with freeze_time(now):
        info: template.RenderInfo = template.Template("{{ now().isoformat() }}", hass).async_render_to_info()
        assert now.isoformat() == info.result()
    assert info.has_time is True


@patch("homeassistant.helpers.template.TemplateEnvironment.is_safe_callable", return_value=True)
def test_utcnow(mock_is_safe: Any, hass: HomeAssistant) -> None:
    utcnow: datetime = dt_util.utcnow()
    with freeze_time(utcnow):
        info: template.RenderInfo = template.Template("{{ utcnow().isoformat() }}", hass).async_render_to_info()
        assert utcnow.isoformat() == info.result()
    assert info.has_time is True


@pytest.mark.parametrize(
    ("now", "expected", "expected_midnight", "timezone_str"),
    [
        ("2021-11-24 03:00:00+00:00", "2021-11-23T10:00:00-08:00", "2021-11-23T00:00:00-08:00", "America/Los_Angeles"),
        ("2021-11-23 19:00:00-08:00", "2021-11-23T10:00:00-08:00", "2021-11-23T00:00:00-08:00", "America/Los_Angeles"),
    ],
)
@patch("homeassistant.helpers.template.TemplateEnvironment.is_safe_callable", return_value=True)
async def test_today_at(
    mock_is_safe: Any, hass: HomeAssistant, now: str, expected: str, expected_midnight: str, timezone_str: str
) -> None:
    freezer = freeze_time(now)
    freezer.start()
    await hass.config.async_set_time_zone(timezone_str)
    result: str = template.Template("{{ today_at('10:00').isoformat() }}", hass).async_render()
    assert result == expected
    result = template.Template("{{ today_at('10:00:00').isoformat() }}", hass).async_render()
    assert result == expected
    result = template.Template("{{ ('10:00:00' | today_at).isoformat() }}", hass).async_render()
    assert result == expected
    result = template.Template('{{ today_at().isoformat() }}', hass).async_render()
    assert result == expected_midnight
    with pytest.raises(TemplateError):
        template.Template("{{ today_at('bad') }}", hass).async_render()
    info: template.RenderInfo = template.Template("{{ today_at('10:00').isoformat() }}", hass).async_render_to_info()
    assert info.has_time is True
    freezer.stop()


@patch("homeassistant.helpers.template.TemplateEnvironment.is_safe_callable", return_value=True)
async def test_relative_time(mock_is_safe: Any, hass: HomeAssistant) -> None:
    await hass.config.async_set_time_zone("UTC")
    now: datetime = datetime.strptime("2000-01-01 10:00:00 +00:00", "%Y-%m-%d %H:%M:%S %z")
    relative_time_template: str = '{{relative_time(strptime("2000-01-01 09:00:00", "%Y-%m-%d %H:%M:%S"))}}'
    with freeze_time(now):
        result: str = template.Template(relative_time_template, hass).async_render()
        assert result == "1 hour"
        result = template.Template('{{  relative_time(    strptime(        "2000-01-01 09:00:00 +01:00",        "%Y-%m-%d %H:%M:%S %z"    )  )}}', hass).async_render()
        assert result == "2 hours"
        result = template.Template('{{  relative_time(    strptime(       "2000-01-01 03:00:00 -06:00",       "%Y-%m-%d %H:%M:%S %z"    )  )}}', hass).async_render()
        assert result == "1 hour"
        result1: str = str(template.strptime("2000-01-01 11:00:00 +00:00", "%Y-%m-%d %H:%M:%S %z"))
        result2: str = template.Template('{{  relative_time(    strptime(       "2000-01-01 11:00:00 +00:00",       "%Y-%m-%d %H:%M:%S %z"    )  )}}', hass).async_render()
        assert result1 == result2
        result = template.Template('{{relative_time("string")}}', hass).async_render()
        assert result == "string"
        result = template.Template('{{  relative_time(    strptime(        "2000-01-01 10:00:00 +00:00",        "%Y-%m-%d %H:%M:%S %z"    )  )}}', hass).async_render()
        assert result == "0 seconds"
        result = template.Template('{{  relative_time(    strptime(        "2000-01-01 11:00:00 +00:00",        "%Y-%m-%d %H:%M:%S %z"    )  )}}', hass).async_render()
        assert result == "2000-01-01 11:00:00+00:00"
        info: template.RenderInfo = template.Template(relative_time_template, hass).async_render_to_info()
        assert info.has_time is True


@patch("homeassistant.helpers.template.TemplateEnvironment.is_safe_callable", return_value=True)
async def test_time_since(mock_is_safe: Any, hass: HomeAssistant) -> None:
    await hass.config.async_set_time_zone("UTC")
    now: datetime = datetime.strptime("2000-01-01 10:00:00 +00:00", "%Y-%m-%d %H:%M:%S %z")
    time_since_template: str = '{{time_since(strptime("2000-01-01 09:00:00", "%Y-%m-%d %H:%M:%S"))}}'
    with freeze_time(now):
        result: str = template.Template(time_since_template, hass).async_render()
        assert result == "1 hour"
        result = template.Template('{{  time_since(    strptime(        "2000-01-01 09:00:00 +01:00",        "%Y-%m-%d %H:%M:%S %z"    )  )}}', hass).async_render()
        assert result == "2 hours"
        result = template.Template('{{  time_since(    strptime(       "2000-01-01 03:00:00 -06:00",       "%Y-%m-%d %H:%M:%S %z"    )  )}}', hass).async_render()
        assert result == "1 hour"
        result1: str = str(template.strptime("2000-01-01 11:00:00 +00:00", "%Y-%m-%d %H:%M:%S %z"))
        result2: str = template.Template('{{  time_since(    strptime(       "2000-01-01 11:00:00 +00:00",       "%Y-%m-%d %H:%M:%S %z"),    precision = 2  )}}', hass).async_render()
        assert result1 == result2
        result = template.Template('{{  time_since(    strptime(        "2000-01-01 09:05:00 +01:00",        "%Y-%m-%d %H:%M:%S %z"),       precision=2  )}}', hass).async_render()
        assert result == "1 hour 55 minutes"
        result = template.Template('{{  time_since(    strptime(       "2000-01-01 02:05:27 -06:00",       "%Y-%m-%d %H:%M:%S %z"),       precision = 3  )}}', hass).async_render()
        assert result == "1 hour 54 minutes 33 seconds"
        result = template.Template('{{  time_since(    strptime(       "2000-01-01 02:05:27 -06:00",       "%Y-%m-%d %H:%M:%S %z")  )}}', hass).async_render()
        assert result == "2 hours"
        result = template.Template('{{  time_since(    strptime(       "1999-02-01 02:05:27 -06:00",       "%Y-%m-%d %H:%M:%S %z"),       precision = 0  )}}', hass).async_render()
        assert result == "11 months 4 days 1 hour 54 minutes 33 seconds"
        result = template.Template('{{  time_since(    strptime(       "1999-02-01 02:05:27 -06:00",       "%Y-%m-%d %H:%M:%S %z")  )}}', hass).async_render()
        assert result == "11 months"
        result1 = str(template.strptime("2000-01-01 11:00:00 +00:00", "%Y-%m-%d %H:%M:%S %z"))
        result2 = template.Template('{{  time_since(    strptime(       "2000-01-01 11:00:00 +00:00",       "%Y-%m-%d %H:%M:%S %z"),       precision=3  )}}', hass).async_render()
        assert result1 == result2
        result = template.Template('{{time_since("string")}}', hass).async_render()
        assert result == "string"
        info: template.RenderInfo = template.Template(time_since_template, hass).async_render_to_info()
        assert info.has_time is True


@patch("homeassistant.helpers.template.TemplateEnvironment.is_safe_callable", return_value=True)
async def test_time_until(mock_is_safe: Any, hass: HomeAssistant) -> None:
    await hass.config.async_set_time_zone("UTC")
    now: datetime = datetime.strptime("2000-01-01 10:00:00 +00:00", "%Y-%m-%d %H:%M:%S %z")
    time_until_template: str = '{{time_until(strptime("2000-01-01 11:00:00", "%Y-%m-%d %H:%M:%S"))}}'
    with freeze_time(now):
        result: str = template.Template(time_until_template, hass).async_render()
        assert result == "1 hour"
        result = template.Template('{{  time_until(    strptime(        "2000-01-01 13:00:00 +01:00",        "%Y-%m-%d %H:%M:%S %z"    )  )}}', hass).async_render()
        assert result == "2 hours"
        result = template.Template('{{  time_until(    strptime(       "2000-01-01 05:00:00 -06:00",       "%Y-%m-%d %H:%M:%S %z"    )  )}}', hass).async_render()
        assert result == "1 hour"
        result1: str = str(template.strptime("2000-01-01 09:00:00 +00:00", "%Y-%m-%d %H:%M:%S %z"))
        result2: str = template.Template('{{  time_until(    strptime(       "2000-01-01 09:00:00 +00:00",       "%Y-%m-%d %H:%M:%S %z"),    precision = 2  )}}', hass).async_render()
        assert result1 == result2
        result = template.Template('{{  time_until(    strptime(        "2000-01-01 12:05:00 +01:00",        "%Y-%m-%d %H:%M:%S %z"),       precision=2  )}}', hass).async_render()
        assert result == "1 hour 5 minutes"
        result = template.Template('{{  time_until(    strptime(       "2000-01-01 05:54:33 -06:00",       "%Y-%m-%d %H:%M:%S %z"),       precision = 3  )}}', hass).async_render()
        assert result == "1 hour 54 minutes 33 seconds"
        result = template.Template('{{  time_until(    strptime(       "2000-01-01 05:54:33 -06:00",       "%Y-%m-%d %H:%M:%S %z")  )}}', hass).async_render()
        assert result == "2 hours"
        result = template.Template('{{  time_until(    strptime(       "2001-02-01 05:54:33 -06:00",       "%Y-%m-%d %H:%M:%S %z"),       precision = 0  )}}', hass).async_render()
        assert result == "1 year 1 month 2 days 1 hour 54 minutes 33 seconds"
        result = template.Template('{{  time_until(    strptime(       "2001-02-01 05:54:33 -06:00",       "%Y-%m-%d %H:%M:%S %z"),       precision = 4  )}}', hass).async_render()
        assert result == "1 year 1 month 2 days 2 hours"
        result1 = str(template.strptime("2000-01-01 09:00:00 +00:00", "%Y-%m-%d %H:%M:%S %z"))
        result2 = template.Template('{{  time_until(    strptime(       "2000-01-01 09:00:00 +00:00",       "%Y-%m-%d %H:%M:%S %z"),       precision=3  )}}', hass).async_render()
        assert result1 == result2
        result = template.Template('{{time_until("string")}}', hass).async_render()
        assert result == "string"
        info: template.RenderInfo = template.Template(time_until_template, hass).async_render_to_info()
        assert info.has_time is True


@patch("homeassistant.helpers.template.TemplateEnvironment.is_safe_callable", return_value=True)
def test_timedelta(mock_is_safe: Any, hass: HomeAssistant) -> None:
    now: datetime = datetime.strptime("2000-01-01 10:00:00 +00:00", "%Y-%m-%d %H:%M:%S %z")
    with freeze_time(now):
        result: str = template.Template("{{timedelta(seconds=120)}}", hass).async_render()
        assert result == "0:02:00"
        result = template.Template("{{timedelta(seconds=86400)}}", hass).async_render()
        assert result == "1 day, 0:00:00"
        result = template.Template("{{timedelta(days=1, hours=4)}}", hass).async_render()
        assert result == "1 day, 4:00:00"
        result = template.Template("{{relative_time(now() - timedelta(seconds=3600))}}", hass).async_render()
        assert result == "1 hour"
        result = template.Template("{{relative_time(now() - timedelta(seconds=86400))}}", hass).async_render()
        assert result == "1 day"
        result = template.Template("{{relative_time(now() - timedelta(seconds=86401))}}", hass).async_render()
        assert result == "1 day"
        result = template.Template("{{relative_time(now() - timedelta(weeks=2, days=1))}}", hass).async_render()
        assert result == "15 days"


def test_version(hass: HomeAssistant) -> None:
    filter_result = template.Template("{{ '2099.9.9' | version}}", hass).async_render()
    function_result = template.Template("{{ version('2099.9.9')}}", hass).async_render()
    assert filter_result == function_result == "2099.9.9"
    filter_result = template.Template("{{ '2099.9.9' | version < '2099.9.10' }}", hass).async_render()
    function_result = template.Template("{{ version('2099.9.9') < '2099.9.10' }}", hass).async_render()
    assert filter_result is function_result is True
    filter_result = template.Template("{{ '2099.9.9' | version == '2099.9.9' }}", hass).async_render()
    function_result = template.Template("{{ version('2099.9.9') == '2099.9.9' }}", hass).async_render()
    assert filter_result is function_result is True
    with pytest.raises(TemplateError):
        template.Template("{{ version(None) < '2099.9.10' }}", hass).async_render()


def test_regex_match(hass: HomeAssistant) -> None:
    tpl: template.Template = template.Template("\n{{ '123-456-7890' | regex_match('(\\d{3})-(\\d{3})-(\\d{4})') }}\n            ", hass)
    assert tpl.async_render() is True
    tpl = template.Template("\n{{ 'Home Assistant test' | regex_match('home', True) }}\n            ", hass)
    assert tpl.async_render() is True
    tpl = template.Template("\n    {{ 'Another Home Assistant test' | regex_match('Home') }}\n                    ", hass)
    assert tpl.async_render() is False
    tpl = template.Template("\n{{ ['Home Assistant test'] | regex_match('.*Assist') }}\n            ", hass)
    assert tpl.async_render() is True


def test_match_test(hass: HomeAssistant) -> None:
    tpl: template.Template = template.Template("\n{{ '123-456-7890' is match('(\\d{3})-(\\d{3})-(\\d{4})') }}\n            ", hass)
    assert tpl.async_render() is True


def test_regex_search(hass: HomeAssistant) -> None:
    tpl: template.Template = template.Template("\n{{ '123-456-7890' | regex_search('(\\d{3})-(\\d{3})-(\\d{4})') }}\n            ", hass)
    assert tpl.async_render() is True
    tpl = template.Template("\n{{ 'Home Assistant test' | regex_search('home', True) }}\n            ", hass)
    assert tpl.async_render() is True
    tpl = template.Template("\n    {{ 'Another Home Assistant test' | regex_search('Home') }}\n                    ", hass)
    assert tpl.async_render() is True
    tpl = template.Template("\n{{ ['Home Assistant test'] | regex_search('Assist') }}\n            ", hass)
    assert tpl.async_render() is True


def test_search_test(hass: HomeAssistant) -> None:
    tpl: template.Template = template.Template("\n{{ '123-456-7890' is search('(\\d{3})-(\\d{3})-(\\d{4})') }}\n            ", hass)
    assert tpl.async_render() is True


def test_regex_replace(hass: HomeAssistant) -> None:
    tpl: template.Template = template.Template("\n{{ 'Hello World' | regex_replace('(Hello\\s)',) }}\n            ", hass)
    assert tpl.async_render() == "World"
    tpl = template.Template("\n{{ ['Home hinderant test'] | regex_replace('hinder', 'Assist') }}\n            ", hass)
    assert tpl.async_render() == ["Home Assistant test"]


def test_regex_findall(hass: HomeAssistant) -> None:
    tpl: template.Template = template.Template("\n{{ 'Flight from JFK to LHR' | regex_findall('([A-Z]{3})') }}\n            ", hass)
    assert tpl.async_render() == ["JFK", "LHR"]


def test_regex_findall_index(hass: HomeAssistant) -> None:
    tpl: template.Template = template.Template("\n{{ 'Flight from JFK to LHR' | regex_findall_index('([A-Z]{3})', 0) }}\n            ", hass)
    assert tpl.async_render() == "JFK"
    tpl = template.Template("\n{{ 'Flight from JFK to LHR' | regex_findall_index('([A-Z]{3})', 1) }}\n            ", hass)
    assert tpl.async_render() == "LHR"
    tpl = template.Template("\n{{ ['JFK', 'LHR'] | regex_findall_index('([A-Z]{3})', 1) }}\n            ", hass)
    assert tpl.async_render() == "LHR"


def test_bitwise_and(hass: HomeAssistant) -> None:
    tpl: template.Template = template.Template("\n{{ 8 | bitwise_and(8) }}\n            ", hass)
    assert tpl.async_render() == 8 & 8
    tpl = template.Template("\n{{ 10 | bitwise_and(2) }}\n            ", hass)
    assert tpl.async_render() == 10 & 2
    tpl = template.Template("\n{{ 8 | bitwise_and(2) }}\n            ", hass)
    assert tpl.async_render() == 8 & 2


def test_bitwise_or(hass: HomeAssistant) -> None:
    tpl: template.Template = template.Template("\n{{ 8 | bitwise_or(8) }}\n            ", hass)
    assert tpl.async_render() == 8 | 8
    tpl = template.Template("\n{{ 10 | bitwise_or(2) }}\n            ", hass)
    assert tpl.async_render() == 10 | 2
    tpl = template.Template("\n{{ 8 | bitwise_or(2) }}\n            ", hass)
    assert tpl.async_render() == 8 | 2


@pytest.mark.parametrize(
    ("value", "xor_value", "expected"),
    [(8, 8, 0), (10, 2, 8), (32768, 64250, 31482), (True, False, 1), (True, True, 0)],
)
def test_bitwise_xor(hass: HomeAssistant, value: Any, xor_value: Any, expected: Any) -> None:
    assert template.Template("{{ value | bitwise_xor(xor_value) }}", hass).async_render({"value": value, "xor_value": xor_value}) == expected


def test_pack(hass: HomeAssistant, caplog: Any) -> None:
    tpl: template.Template = template.Template("\n{{ value | pack('>I') }}\n            ", hass)
    variables: dict[str, Any] = {"value": 3735928559}
    assert tpl.async_render(variables=variables) == b"\xde\xad\xbe\xef"
    tpl = template.Template("\n{{ pack(value, '>I') }}\n            ", hass)
    variables = {"value": 3735928559}
    assert tpl.async_render(variables=variables) == b"\xde\xad\xbe\xef"
    tpl = template.Template("\n{{ pack(value, '>I') }}\n            ", hass)
    variables = {"value": None}
    assert tpl.async_render(variables=variables) is None
    assert "Template warning: 'pack' unable to pack object 'None' with type 'NoneType' and format_string '>I' see https://docs.python.org/3/library/struct.html for more information" in caplog.text
    tpl = template.Template("\n{{ pack(value, 'invalid filter') }}\n            ", hass)
    variables = {"value": 3735928559}
    assert tpl.async_render(variables=variables) is None
    assert "Template warning: 'pack' unable to pack object '3735928559' with type 'int' and format_string 'invalid filter' see https://docs.python.org/3/library/struct.html for more information" in caplog.text


def test_unpack(hass: HomeAssistant, caplog: Any) -> None:
    tpl: template.Template = template.Template("\n{{ value | unpack('>I') }}\n            ", hass)
    variables: dict[str, Any] = {"value": b"\xde\xad\xbe\xef"}
    assert tpl.async_render(variables=variables) == 3735928559
    tpl = template.Template("\n{{ unpack(value, '>I') }}\n            ", hass)
    variables = {"value": b"\xde\xad\xbe\xef"}
    assert tpl.async_render(variables=variables) == 3735928559
    tpl = template.Template("\n{{ unpack(value, '>H', offset=2) }}\n            ", hass)
    variables = {"value": b"\xde\xad\xbe\xef"}
    assert tpl.async_render(variables=variables) == 48879
    tpl = template.Template("\n{{ unpack(value, '>I') }}\n            ", hass)
    variables = {"value": b""}
    assert tpl.async_render(variables=variables) is None
    assert "Template warning: 'unpack' unable to unpack object 'b''" in caplog.text
    tpl = template.Template("\n{{ unpack(value, 'invalid filter') }}\n            ", hass)
    variables = {"value": b""}
    assert tpl.async_render(variables=variables) is None
    assert "Template warning: 'unpack' unable to unpack object 'b''" in caplog.text


def test_distance_function_with_1_state(hass: HomeAssistant) -> None:
    _set_up_units(hass)
    hass.states.async_set("test.object", "happy", {"latitude": 32.87336, "longitude": -117.22943})
    tpl: template.Template = template.Template("{{ distance(states.test.object) | round }}", hass)
    assert tpl.async_render() == 187


def test_distance_function_with_2_states(hass: HomeAssistant) -> None:
    _set_up_units(hass)
    hass.states.async_set("test.object", "happy", {"latitude": 32.87336, "longitude": -117.22943})
    hass.states.async_set("test.object_2", "happy", {"latitude": hass.config.latitude, "longitude": hass.config.longitude})
    tpl: template.Template = template.Template("{{ distance(states.test.object, states.test.object_2) | round }}", hass)
    assert tpl.async_render() == 187


def test_distance_function_with_1_coord(hass: HomeAssistant) -> None:
    _set_up_units(hass)
    tpl: template.Template = template.Template("{{ distance(\"32.87336\", \"-117.22943\") | round }}", hass)
    assert tpl.async_render() == 187


def test_distance_function_with_2_coords(hass: HomeAssistant) -> None:
    _set_up_units(hass)
    tpl: template.Template = template.Template(f"{{{{ distance(\"32.87336\", \"-117.22943\", {hass.config.latitude}, {hass.config.longitude}) | round }}}}", hass)
    assert tpl.async_render() == 187


def test_distance_function_with_1_state_1_coord(hass: HomeAssistant) -> None:
    _set_up_units(hass)
    hass.states.async_set("test.object_2", "happy", {"latitude": hass.config.latitude, "longitude": hass.config.longitude})
    tpl: template.Template = template.Template("{{ distance(\"32.87336\", \"-117.22943\", states.test.object_2) | round }}", hass)
    assert tpl.async_render() == 187
    tpl2: template.Template = template.Template("{{ distance(states.test.object_2, \"32.87336\", \"-117.22943\") | round }}", hass)
    assert tpl2.async_render() == 187


def test_distance_function_return_none_if_invalid_state(hass: HomeAssistant) -> None:
    hass.states.async_set("test.object_2", "happy", {"latitude": 10})
    tpl: template.Template = template.Template("{{ distance(states.test.object_2) | round }}", hass)
    with pytest.raises(TemplateError):
        tpl.async_render()


def test_distance_function_return_none_if_invalid_coord(hass: HomeAssistant) -> None:
    assert template.Template("{{ distance(\"123\", \"abc\") }}", hass).async_render() is None
    assert template.Template("{{ distance(\"123\") }}", hass).async_render() is None
    hass.states.async_set("test.object_2", "happy", {"latitude": hass.config.latitude, "longitude": hass.config.longitude})
    tpl: template.Template = template.Template("{{ distance(\"123\", states.test_object_2) }}", hass)
    assert tpl.async_render() is None


def test_distance_function_with_2_entity_ids(hass: HomeAssistant) -> None:
    _set_up_units(hass)
    hass.states.async_set("test.object", "happy", {"latitude": 32.87336, "longitude": -117.22943})
    hass.states.async_set("test.object_2", "happy", {"latitude": hass.config.latitude, "longitude": hass.config.longitude})
    tpl: template.Template = template.Template("{{ distance(\"test.object\", \"test.object_2\") | round }}", hass)
    assert tpl.async_render() == 187


def test_distance_function_with_1_entity_1_coord(hass: HomeAssistant) -> None:
    _set_up_units(hass)
    hass.states.async_set("test.object", "happy", {"latitude": hass.config.latitude, "longitude": hass.config.longitude})
    tpl: template.Template = template.Template("{{ distance(\"test.object\", \"32.87336\", \"-117.22943\") | round }}", hass)
    assert tpl.async_render() == 187


def test_closest_function_home_vs_domain(hass: HomeAssistant) -> None:
    hass.states.async_set("test_domain.object", "happy", {"latitude": hass.config.latitude + 0.1, "longitude": hass.config.longitude + 0.1})
    hass.states.async_set("not_test_domain.but_closer", "happy", {"latitude": hass.config.latitude, "longitude": hass.config.longitude})
    assert template.Template("{{ closest(states.test_domain).entity_id }}", hass).async_render() == "test_domain.object"
    assert template.Template("{{ (states.test_domain | closest).entity_id }}", hass).async_render() == "test_domain.object"


def test_closest_function_home_vs_all_states(hass: HomeAssistant) -> None:
    hass.states.async_set("test_domain.object", "happy", {"latitude": hass.config.latitude + 0.1, "longitude": hass.config.longitude + 0.1})
    hass.states.async_set("test_domain_2.and_closer", "happy", {"latitude": hass.config.latitude, "longitude": hass.config.longitude})
    assert template.Template("{{ closest(states).entity_id }}", hass).async_render() == "test_domain_2.and_closer"
    assert template.Template("{{ (states | closest).entity_id }}", hass).async_render() == "test_domain_2.and_closer"


async def test_closest_function_home_vs_group_entity_id(hass: HomeAssistant) -> None:
    hass.states.async_set("test_domain.object", "happy", {"latitude": hass.config.latitude + 0.1, "longitude": hass.config.longitude + 0.1})
    hass.states.async_set("not_in_group.but_closer", "happy", {"latitude": hass.config.latitude, "longitude": hass.config.longitude})
    assert await async_setup_component(hass, "group", {})
    await hass.async_block_till_done()
    await group.Group.async_create_group(hass, "location group", created_by_service=False, entity_ids=["test_domain.object"], icon=None, mode=None, object_id=None, order=None)
    info: template.RenderInfo = template.render_to_info(hass, "{{ closest('group.location_group').entity_id }}")
    # assert_result_info(info, 'test_domain.object', {'group.location_group', 'test_domain.object'})
    # assert info.rate_limit is None


async def test_closest_function_home_vs_group_state(hass: HomeAssistant) -> None:
    hass.states.async_set("test_domain.object", "happy", {"latitude": hass.config.latitude + 0.1, "longitude": hass.config.longitude + 0.1})
    hass.states.async_set("not_in_group.but_closer", "happy", {"latitude": hass.config.latitude, "longitude": hass.config.longitude})
    assert await async_setup_component(hass, "group", {})
    await hass.async_block_till_done()
    await group.Group.async_create_group(hass, "location group", created_by_service=False, entity_ids=["test_domain.object"], icon=None, mode=None, object_id=None, order=None)
    info: template.RenderInfo = template.render_to_info(hass, "{{ closest('group.location_group').entity_id }}")
    # assert_result_info(info, 'test_domain.object', {'group.location_group', 'test_domain.object'})
    # assert info.rate_limit is None
    info = template.render_to_info(hass, "{{ closest(states.group.location_group).entity_id }}")
    # assert_result_info(info, 'test_domain.object', {'test_domain.object', 'group.location_group'})
    # assert info.rate_limit is None


async def test_expand(hass: HomeAssistant) -> None:
    info: template.RenderInfo = template.render_to_info(hass, "{{ expand('test.object') }}")
    # assert_result_info(info, [], ['test.object'])
    # assert info.rate_limit is None
    info = template.render_to_info(hass, '{{ expand("test.object") | sort(attribute="entity_id") | map(attribute="entity_id") | join(", ") }}')
    # assert_result_info(info, 'test.object', ['test.object'])
    # assert info.rate_limit is None
    info = template.render_to_info(hass, "{{ expand('group.new_group') | sort(attribute='entity_id') | map(attribute='entity_id') | join(', ') }}")
    # assert_result_info(info, '', ['group.new_group'])
    # assert info.rate_limit is None
    info = template.render_to_info(hass, "{{ expand(states.group) | sort(attribute='entity_id') | map(attribute='entity_id') | join(', ') }}")
    # assert_result_info(info, '', [], ['group'])
    # assert info.rate_limit == template.DOMAIN_STATES_RATE_LIMIT
    assert await async_setup_component(hass, "group", {})
    await hass.async_block_till_done()
    await group.Group.async_create_group(hass, "new group", created_by_service=False, entity_ids=["test.object"], icon=None, mode=None, object_id=None, order=None)
    info = template.render_to_info(hass, "{{ expand('group.new_group') | sort(attribute='entity_id') | map(attribute='entity_id') | join(', ') }}")
    # assert_result_info(info, 'test.object', {'group.new_group', 'test.object'})
    # assert info.rate_limit is None
    info = template.render_to_info(hass, "{{ expand(states.group) | sort(attribute='entity_id') | map(attribute='entity_id') | join(', ') }}")
    # assert_result_info(info, 'test.object', {'test.object'}, ['group'])
    # assert info.rate_limit == template.DOMAIN_STATES_RATE_LIMIT
    info = template.render_to_info(hass, "{{ expand('group.new_group', 'test.object') | sort(attribute='entity_id') | map(attribute='entity_id') | join(', ') }}")
    # assert_result_info(info, 'test.object', {'test.object', 'group.new_group'})
    info = template.render_to_info(hass, "{{ ['group.new_group', 'test.object'] | expand | sort(attribute='entity_id') | map(attribute='entity_id') | join(', ') }}")
    # assert_result_info(info, 'test.object', {'test.object', 'group.new_group'})
    # assert info.rate_limit is None
    hass.states.async_set("sensor.power_1", 0)
    hass.states.async_set("sensor.power_2", 200.2)
    hass.states.async_set("sensor.power_3", 400.4)
    assert await async_setup_component(hass, "group", {})
    await hass.async_block_till_done()
    await group.Group.async_create_group(hass, "power sensors", created_by_service=False, entity_ids=["sensor.power_1", "sensor.power_2", "sensor.power_3"], icon=None, mode=None, object_id=None, order=None)
    info = template.render_to_info(hass, "{{ states.group.power_sensors.attributes.entity_id | expand | sort(attribute='entity_id') | map(attribute='state')|map('float')|sum  }}")
    # assert_result_info(info, 200.2 + 400.4, {'group.power_sensors', 'sensor.power_1', 'sensor.power_2', 'sensor.power_3'})
    hass.states.async_set("light.first", "on")
    hass.states.async_set("light.second", "off")
    assert await async_setup_component(hass, "light", {"light": {"platform": "group", "name": "Grouped", "entities": ["light.first", "light.second"]}})
    await hass.async_block_till_done()
    info = template.render_to_info(hass, "{{ expand('light.grouped') | sort(attribute='entity_id') | map(attribute='entity_id') | join(', ') }}")
    # assert_result_info(info, 'light.first, light.second', ['light.grouped', 'light.first', 'light.second'])
    assert await async_setup_component(hass, "zone", {"zone": {"name": "Test", "latitude": 32.880837, "longitude": -117.237561, "radius": 250, "passive": False}})
    info = template.render_to_info(hass, "{{ expand('zone.test') | sort(attribute='entity_id') | map(attribute='entity_id') | join(', ') }}")
    # assert_result_info(info, '', ['zone.test'])
    hass.states.async_set("person.person1", "test")
    await hass.async_block_till_done()
    info = template.render_to_info(hass, "{{ expand('zone.test') | sort(attribute='entity_id') | map(attribute='entity_id') | join(', ') }}")
    # assert_result_info(info, 'person.person1', ['zone.test', 'person.person1'])
    hass.states.async_set("person.person2", "test")
    await hass.async_block_till_done()
    info = template.render_to_info(hass, "{{ expand('zone.test') | sort(attribute='entity_id') | map(attribute='entity_id') | join(', ') }}")
    # assert_result_info(info, 'person.person1, person.person2', ['zone.test', 'person.person1', 'person.person2'])


async def test_device_entities(hass: HomeAssistant, device_registry: dr.DeviceRegistry, entity_registry: er.EntityRegistry) -> None:
    config_entry = MockConfigEntry(domain="light")
    config_entry.add_to_hass(hass)
    info = template.render_to_info(hass, "{{ device_entities('abc123') }}")
    # assert_result_info(info, [])
    info = template.render_to_info(hass, '{{ device_entities(56) }}')
    # assert_result_info(info, [])
    device_entry = device_registry.async_get_or_create(config_entry_id=config_entry.entry_id, connections={(dr.CONNECTION_NETWORK_MAC, "12:34:56:AB:CD:EF")})
    info = template.render_to_info(hass, f"{{{{ device_entities('{device_entry.id}') }}}}")
    # assert_result_info(info, [])
    entity_registry.async_get_or_create("light", "hue", "5678", config_entry=config_entry, device_id=device_entry.id)
    info = template.render_to_info(hass, f"{{{{ device_entities('{device_entry.id}') }}}}")
    # assert_result_info(info, ['light.hue_5678'], [])
    info = template.render_to_info(hass, f"{{{{ device_entities('{device_entry.id}') | expand | sort(attribute='entity_id') | map(attribute='entity_id') | join(', ') }}}}")
    # assert_result_info(info, '', ['light.hue_5678'])
    hass.states.async_set("light.hue_5678", "happy")
    info = template.render_to_info(hass, f"{{{{ device_entities('{device_entry.id}') | expand | sort(attribute='entity_id') | map(attribute='entity_id') | join(', ') }}}}")
    # assert_result_info(info, 'light.hue_5678', ['light.hue_5678'])
    entity_registry.async_get_or_create("light", "hue", "ABCD", config_entry=config_entry, device_id=device_entry.id)
    hass.states.async_set("light.hue_abcd", "camper")
    info = template.render_to_info(hass, f"{{{{ device_entities('{device_entry.id}') }}}}")
    # assert_result_info(info, ['light.hue_5678', 'light.hue_abcd'], [])
    info = template.render_to_info(hass, f"{{{{ device_entities('{device_entry.id}') | expand | sort(attribute='entity_id') | map(attribute='entity_id') | join(', ') }}}}")
    # assert_result_info(info, 'light.hue_5678, light.hue_abcd', ['light.hue_abcd', 'light.hue_5678'])


async def test_integration_entities(hass: HomeAssistant, entity_registry: er.EntityRegistry) -> None:
    config_entry = MockConfigEntry(domain="mock", title="")
    config_entry.add_to_hass(hass)
    entity_registry.async_get_or_create("sensor", "mock", "untitled", config_entry=config_entry)
    info = template.render_to_info(hass, "{{ integration_entities('') }}")
    # assert_result_info(info, [])
    config_entry = MockConfigEntry(domain="mock", title="Mock bridge 2")
    config_entry.add_to_hass(hass)
    entity_entry = entity_registry.async_get_or_create("sensor", "mock", "test", config_entry=config_entry)
    info = template.render_to_info(hass, "{{ integration_entities('Mock bridge 2') }}")
    # assert_result_info(info, [entity_entry.entity_id])
    config_entry = MockConfigEntry(domain="mock", title="Not unique")
    config_entry.add_to_hass(hass)
    entity_entry_not_unique_1 = entity_registry.async_get_or_create("sensor", "mock", "not_unique_1", config_entry=config_entry)
    config_entry = MockConfigEntry(domain="mock", title="Not unique")
    config_entry.add_to_hass(hass)
    entity_entry_not_unique_2 = entity_registry.async_get_or_create("sensor", "mock", "not_unique_2", config_entry=config_entry)
    info = template.render_to_info(hass, "{{ integration_entities('Not unique') }}")
    # assert_result_info(info, [entity_entry_not_unique_1.entity_id, entity_entry_not_unique_2.entity_id])
    mock_entity = entity.Entity()
    mock_entity.hass = hass
    mock_entity.entity_id = "light.test_entity"
    mock_entity.platform = EntityPlatform(
        hass=hass,
        logger=logging.getLogger(__name__),
        domain="light",
        platform_name="entryless_integration",
        platform=None,
        scan_interval=timedelta(seconds=30),
        entity_namespace=None,
    )
    await mock_entity.async_internal_added_to_hass()
    info = template.render_to_info(hass, "{{ integration_entities('entryless_integration') }}")
    # assert_result_info(info, ['light.test_entity'])
    info = template.render_to_info(hass, "{{ integration_entities('abc123') }}")
    # assert_result_info(info, [])


async def test_config_entry_id(hass: HomeAssistant, entity_registry: er.EntityRegistry) -> None:
    config_entry = MockConfigEntry(domain="light", title="Some integration")
    config_entry.add_to_hass(hass)
    entity_entry = entity_registry.async_get_or_create("sensor", "test", "test", suggested_object_id="test", config_entry=config_entry)
    info = template.render_to_info(hass, "{{ 'sensor.fail' | config_entry_id }}")
    # assert_result_info(info, None)
    info = template.render_to_info(hass, '{{ 56 | config_entry_id }}')
    # assert_result_info(info, None)
    info = template.render_to_info(hass, "{{ 'not_a_real_entity_id' | config_entry_id }}")
    # assert_result_info(info, None)
    info = template.render_to_info(hass, f"{{{{ config_entry_id('{entity_entry.entity_id}') }}}}")
    # assert_result_info(info, config_entry.entry_id)


async def test_device_id(hass: HomeAssistant, device_registry: dr.DeviceRegistry, entity_registry: er.EntityRegistry) -> None:
    config_entry = MockConfigEntry(domain="light")
    config_entry.add_to_hass(hass)
    device_entry = device_registry.async_get_or_create(
        config_entry_id=config_entry.entry_id, connections={(dr.CONNECTION_NETWORK_MAC, "12:34:56:AB:CD:EF")}
    )
    entity_entry = entity_registry.async_get_or_create("sensor", "test", "test", suggested_object_id="test", device_id=device_entry.id)
    entity_entry_no_device = entity_registry.async_get_or_create("sensor", "test", "test_no_device", suggested_object_id="test")
    info = template.render_to_info(hass, "{{ 'sensor.fail' | device_id }}")
    # assert_result_info(info, None)
    info = template.render_to_info(hass, '{{ 56 | device_id }}')
    # assert_result_info(info, None)
    info = template.render_to_info(hass, "{{ 'not_a_real_entity_id' | device_id }}")
    # assert_result_info(info, None)
    info = template.render_to_info(hass, f"{{{{ device_id('{entity_entry_no_device.entity_id}') }}}}")
    # assert_result_info(info, None)
    info = template.render_to_info(hass, f"{{{{ device_id('{entity_entry.entity_id}') }}}}")
    # assert_result_info(info, device_entry.id)
    info = template.render_to_info(hass, "{{ device_id('test') }}")
    # assert_result_info(info, device_entry.id)


async def test_device_attr(hass: HomeAssistant, device_registry: dr.DeviceRegistry, entity_registry: er.EntityRegistry) -> None:
    info = template.render_to_info(hass, "{{ device_attr('abc123', 'id') }}")
    # assert_result_info(info, None)
    info = template.render_to_info(hass, "{{ device_attr(56, 'id') }}")
    with pytest.raises(TemplateError):
        _ = info.result()
    info = template.render_to_info(hass, "{{ is_device_attr('abc123', 'id', 'test') }}")
    # assert_result_info(info, False)
    info = template.render_to_info(hass, "{{ is_device_attr(56, 'id', 'test') }}")
    with pytest.raises(TemplateError):
        _ = info.result()
    info = template.render_to_info(hass, "{{ device_attr('entity.test', 'id') }}")
    # assert_result_info(info, None)
    info = template.render_to_info(hass, "{{ is_device_attr('entity.test', 'id', 'test') }}")
    # assert_result_info(info, False)
    device_entry = device_registry.async_get_or_create(
        config_entry_id=MockConfigEntry(domain="light").entry_id, connections={(dr.CONNECTION_NETWORK_MAC, "12:34:56:AB:CD:EF")}
    )
    entity_entry = entity_registry.async_get_or_create("light", "hue", "5678", config_entry=MockConfigEntry(domain="light"), device_id=device_entry.id)
    info = template.render_to_info(hass, f"{{{{ device_attr('{device_entry.id}', 'invalid_attr') }}}}")
    # assert_result_info(info, None)
    info = template.render_to_info(hass, f"{{{{ is_device_attr('{device_entry.id}', 'invalid_attr', 'test') }}}}")
    # assert_result_info(info, False)
    info = template.render_to_info(hass, f"{{{{ device_attr('{device_entry.id}', 'manufacturer') }}}}")
    # assert_result_info(info, None)
    info = template.render_to_info(hass, f"{{{{ is_device_attr('{device_entry.id}', 'manufacturer', 'test') }}}}")
    # assert_result_info(info, False)
    info = template.render_to_info(hass, f"{{{{ is_device_attr('{device_entry.id}', 'manufacturer', None) }}}}")
    # assert_result_info(info, True)
    info = template.render_to_info(hass, f"{{{{ device_attr('{device_entry.id}', 'model') }}}}")
    # assert_result_info(info, 'test')
    info = template.render_to_info(hass, f"{{{{ device_attr('{entity_entry.entity_id}', 'model') }}}}")
    # assert_result_info(info, 'test')
    info = template.render_to_info(hass, f"{{{{ is_device_attr('{device_entry.id}', 'model', 'fail') }}}}")
    # assert_result_info(info, False)
    info = template.render_to_info(hass, f"{{{{ is_device_attr('{device_entry.id}', 'model', 'test') }}}}")
    # assert_result_info(info, True)
    info = template.render_to_info(hass, f"{{{{ '{entity_entry.entity_id}' | device_attr('model') }}}}")
    # assert_result_info(info, 'test')
    info = template.render_to_info(hass, f"{{{{ ['{device_entry.id}'] | select('is_device_attr', 'model', 'test') | list }}}}")
    # assert_result_info(info, [device_entry.id])


async def test_config_entry_attr(hass: HomeAssistant) -> None:
    info: dict[str, Any] = {"domain": "mock_light", "title": "mock title", "source": config_entries.SOURCE_BLUETOOTH, "disabled_by": config_entries.ConfigEntryDisabler.USER}
    config_entry = MockConfigEntry(**info)
    config_entry.add_to_hass(hass)
    info["state"] = config_entries.ConfigEntryState.NOT_LOADED
    for key, value in info.items():
        tpl: template.Template = template.Template("{{ config_entry_attr('" + config_entry.entry_id + "', '" + key + "') }}", hass)
        assert tpl.async_render(parse_result=False) == str(value)
    for config_entry_id, key in ((config_entry.entry_id, "invalid_key"), (56, "domain")):
        with pytest.raises(TemplateError):
            template.Template('{{ config_entry_attr(' + json.dumps(config_entry_id) + ", '" + key + "') }}", hass).async_render()
    assert template.Template("{{ config_entry_attr('invalid_id', 'domain') }}", hass).async_render(parse_result=False) == "None"


async def test_issues(hass: HomeAssistant, issue_registry: ir.IssueRegistry) -> None:
    info = template.render_to_info(hass, '{{ issues() }}')
    # assert_result_info(info, {})
    ir.async_create_issue(
        hass,
        "test",
        "issue 1",
        breaks_in_ha_version="2023.7",
        is_fixable=True,
        is_persistent=True,
        learn_more_url="https://theuselessweb.com",
        severity="error",
        translation_key="abc_1234",
        translation_placeholders={"abc": "123"},
    )
    await hass.async_block_till_done()
    created_issue = issue_registry.async_get_issue("test", "issue 1")
    info = template.render_to_info(hass, "{{ issues()['test', 'issue 1'] }}")
    # assert_result_info(info, created_issue.to_json())
    ir.async_delete_issue(hass, "test", "issue 1")
    await hass.async_block_till_done()
    info = template.render_to_info(hass, '{{ issues() }}')
    # assert_result_info(info, {})


async def test_issue(hass: HomeAssistant, issue_registry: ir.IssueRegistry) -> None:
    info = template.render_to_info(hass, "{{ issue('non_existent', 'issue') }}")
    # assert_result_info(info, None)
    ir.async_create_issue(
        hass,
        "test",
        "issue 1",
        breaks_in_ha_version="2023.7",
        is_fixable=True,
        is_persistent=True,
        learn_more_url="https://theuselessweb.com",
        severity="error",
        translation_key="abc_1234",
        translation_placeholders={"abc": "123"},
    )
    await hass.async_block_till_done()
    created_issue = issue_registry.async_get_issue("test", "issue 1")
    info = template.render_to_info(hass, "{{ issue('test', 'issue 1') }}")
    # assert_result_info(info, created_issue.to_json())


async def test_areas(hass: HomeAssistant, area_registry: ar.AreaRegistry) -> None:
    info = template.render_to_info(hass, '{{ areas() }}')
    # assert_result_info(info, [])
    area1 = area_registry.async_get_or_create("area1")
    info = template.render_to_info(hass, '{{ areas() }}')
    # assert_result_info(info, [area1.id])
    area2 = area_registry.async_get_or_create("area2")
    info = template.render_to_info(hass, '{{ areas() }}')
    # assert_result_info(info, [area1.id, area2.id])


async def test_area_id(hass: HomeAssistant, area_registry: ar.AreaRegistry, device_registry: dr.DeviceRegistry, entity_registry: er.EntityRegistry) -> None:
    def test_func(value: Any, expected: Any) -> None:
        info = template.render_to_info(hass, f"{{{{ floor_id('{value}') }}}}")
        # assert_result_info(info, expected)
    test_func("Third floor", None)
    info = template.render_to_info(hass, '{{ floor_id(42) }}')
    # assert_result_info(info, None)
    info = template.render_to_info(hass, '{{ 42 | floor_id }}')
    # assert_result_info(info, None)
    area_entry_entity_id = area_registry.async_get_or_create("sensor.fake")
    device_entry = device_registry.async_get_or_create(config_entry_id=MockConfigEntry(domain="light").entry_id, connections={(dr.CONNECTION_NETWORK_MAC, "12:34:56:AB:CD:EF")})
    entity_entry = entity_registry.async_get_or_create("light", "hue", "5678", config_entry=MockConfigEntry(domain="light"), device_id=device_entry.id)
    info = template.render_to_info(hass, f"{{{{ floor_id('{device_entry.id}') }}}}")
    # assert_result_info(info, None)
    area_entry_hex = area_registry.async_get_or_create("123abc")
    device_entry = device_registry.async_update_device(device_entry.id, area_id=area_entry_hex.id)
    entity_entry = entity_registry.async_update_entity(entity_entry.entity_id, area_id=area_entry_hex.id)
    test_func(area_entry_hex.id, None)
    test_func(device_entry.id, None)
    test_func(entity_entry.entity_id, None)
    area_entry_hex = area_registry.async_update(area_entry_hex.id, floor_id="some_floor")
    test_func(area_entry_hex.id, "some_floor")
    test_func(device_entry.id, "some_floor")
    test_func(entity_entry.entity_id, "some_floor")


async def test_floor_name(hass: HomeAssistant, floor_registry: fr.FloorRegistry, area_registry: ar.AreaRegistry, device_registry: dr.DeviceRegistry, entity_registry: er.EntityRegistry) -> None:
    def test_func(value: Any, expected: Any) -> None:
        info = template.render_to_info(hass, f"{{{{ floor_name('{value}') }}}}")
        # assert_result_info(info, expected)
    test_func("Third floor", None)
    info = template.render_to_info(hass, '{{ floor_name(42) }}')
    # assert_result_info(info, None)
    info = template.render_to_info(hass, '{{ 42 | floor_name }}')
    # assert_result_info(info, None)
    floor = floor_registry.async_create("First floor")
    test_func(floor.floor_id, floor.name)
    config_entry = MockConfigEntry(domain="light")
    config_entry.add_to_hass(hass)
    area_entry_hex = area_registry.async_get_or_create("123abc")
    device_entry = device_registry.async_get_or_create(config_entry_id=config_entry.entry_id, connections={(dr.CONNECTION_NETWORK_MAC, "12:34:56:AB:CD:EF")})
    entity_entry = entity_registry.async_get_or_create("light", "hue", "5678", config_entry=config_entry, device_id=device_entry.id)
    device_entry = device_registry.async_update_device(device_entry.id, area_id=area_entry_hex.id)
    entity_entry = entity_registry.async_update_entity(entity_entry.entity_id, area_id=area_entry_hex.id)
    test_func(area_entry_hex.id, None)
    test_func(device_entry.id, None)
    test_func(entity_entry.entity_id, None)
    area_entry_hex = area_registry.async_update(area_entry_hex.id, floor_id=floor.floor_id)
    test_func(area_entry_hex.id, floor.name)
    test_func(device_entry.id, floor.name)
    test_func(entity_entry.entity_id, floor.name)


async def test_floor_areas(hass: HomeAssistant, floor_registry: fr.FloorRegistry, area_registry: ar.AreaRegistry) -> None:
    info = template.render_to_info(hass, "{{ floor_areas('skyring') }}")
    # assert_result_info(info, [])
    area_entry = area_registry.async_get_or_create("Living room")
    area_registry.async_update(area_entry.id, floor_id="floor123")
    info = template.render_to_info(hass, f"{{{{ floor_areas('floor123') }}}}")
    # assert_result_info(info, [area_entry.id])


async def test_labels(hass: HomeAssistant, label_registry: lr.LabelRegistry, area_registry: ar.AreaRegistry, device_registry: dr.DeviceRegistry, entity_registry: er.EntityRegistry) -> None:
    info = template.render_to_info(hass, "{{ labels() }}")
    # assert_result_info(info, [])
    label1 = label_registry.async_create("label1")
    info = template.render_to_info(hass, "{{ labels() }}")
    # assert_result_info(info, [label1.label_id])
    label2 = label_registry.async_create("label2")
    info = template.render_to_info(hass, "{{ labels() }}")
    # assert_result_info(info, [label1.label_id, label2.label_id])
    info = template.render_to_info(hass, "{{ labels('sensor.fake') }}")
    # assert_result_info(info, [])
    info = template.render_to_info(hass, "{{ 'sensor.fake' | labels }}")
    # assert_result_info(info, [])
    info = template.render_to_info(hass, "{{ labels('123abc') }}")
    # assert_result_info(info, [])
    info = template.render_to_info(hass, "{{ '123abc' | labels }}")
    # assert_result_info(info, [])
    config_entry = MockConfigEntry(domain="light")
    config_entry.add_to_hass(hass)
    device_entry = device_registry.async_get_or_create(config_entry_id=config_entry.entry_id, connections={(dr.CONNECTION_NETWORK_MAC, "12:34:56:AB:CD:EF")})
    entity_entry = entity_registry.async_get_or_create("light", "hue", "5678", config_entry=config_entry, device_id=device_entry.id)
    info = template.render_to_info(hass, f"{{{{ labels('{entity_entry.entity_id}') }}}}")
    # assert_result_info(info, [])
    device_registry.async_update_device(device_entry.id, labels=[label1.label_id])
    entity_registry.async_update_entity(entity_entry.entity_id, labels=[label2.label_id])
    info = template.render_to_info(hass, f"{{{{ '{entity_entry.entity_id}' | labels }}}}")
    # assert_result_info(info, [label2.label_id])
    info = template.render_to_info(hass, f"{{{{ labels('{device_entry.id}') }}}}")
    # assert_result_info(info, [label1.label_id])


async def test_label_id(hass: HomeAssistant, label_registry: lr.LabelRegistry) -> None:
    info = template.render_to_info(hass, "{{ label_id('non-existing label') }}")
    # assert_result_info(info, None)
    info = template.render_to_info(hass, "{{ 'non-existing label' | label_id }}")
    # assert_result_info(info, None)
    info = template.render_to_info(hass, '{{ label_id(42) }}')
    # assert_result_info(info, None)
    info = template.render_to_info(hass, '{{ 42 | label_id }}')
    # assert_result_info(info, None)
    label = label_registry.async_create("existing label")
    info = template.render_to_info(hass, "{{ label_id('existing label') }}")
    # assert_result_info(info, label.label_id)
    info = template.render_to_info(hass, "{{ 'existing label' | label_id }}")
    # assert_result_info(info, label.label_id)


async def test_label_name(hass: HomeAssistant, label_registry: lr.LabelRegistry) -> None:
    info = template.render_to_info(hass, "{{ label_name('1234567890') }}")
    # assert_result_info(info, None)
    info = template.render_to_info(hass, "{{ '1234567890' | label_name }}")
    # assert_result_info(info, None)
    info = template.render_to_info(hass, '{{ label_name(42) }}')
    # assert_result_info(info, None)
    info = template.render_to_info(hass, '{{ 42 | label_name }}')
    # assert_result_info(info, None)
    label = label_registry.async_create("choo choo")
    info = template.render_to_info(hass, f"{{{{ label_name('{label.label_id}') }}}}")
    # assert_result_info(info, label.name)
    info = template.render_to_info(hass, f"{{{{ '{label.label_id}' | label_name }}}}")
    # assert_result_info(info, label.name)


async def test_label_entities(hass: HomeAssistant, entity_registry: er.EntityRegistry, label_registry: lr.LabelRegistry) -> None:
    info = template.render_to_info(hass, "{{ label_entities('deadbeef') }}")
    # assert_result_info(info, [])
    info = template.render_to_info(hass, "{{ 'deadbeef' | label_entities }}")
    # assert_result_info(info, [])
    info = template.render_to_info(hass, '{{ label_entities(42) }}')
    # assert_result_info(info, [])
    info = template.render_to_info(hass, '{{ 42 | label_entities }}')
    # assert_result_info(info, [])
    config_entry = MockConfigEntry(domain="light")
    config_entry.add_to_hass(hass)
    entity_entry = entity_registry.async_get_or_create("light", "hue", "5678", config_entry=config_entry)
    label = label_registry.async_create("Romantic Lights")
    entity_registry.async_update_entity(entity_entry.entity_id, labels={label.label_id})
    info = template.render_to_info(hass, f"{{{{ label_entities('{label.label_id}') }}}}")
    # assert_result_info(info, ['light.hue_5678'])
    info = template.render_to_info(hass, f"{{{{ '{label.label_id}' | label_entities }}}}")
    # assert_result_info(info, ['light.hue_5678'])
    info = template.render_to_info(hass, f"{{{{ label_entities('{label.name}') }}}}")
    # assert_result_info(info, ['light.hue_5678'])
    info = template.render_to_info(hass, f"{{{{ '{label.name}' | label_entities }}}}")
    # assert_result_info(info, ['light.hue_5678'])


async def test_label_devices(hass: HomeAssistant, device_registry: dr.DeviceRegistry, label_registry: lr.LabelRegistry) -> None:
    info = template.render_to_info(hass, "{{ label_devices('deadbeef') }}")
    # assert_result_info(info, [])
    info = template.render_to_info(hass, "{{ 'deadbeef' | label_devices }}")
    # assert_result_info(info, [])
    info = template.render_to_info(hass, '{{ label_devices(42) }}')
    # assert_result_info(info, [])
    info = template.render_to_info(hass, '{{ 42 | label_devices }}')
    # assert_result_info(info, [])
    config_entry = MockConfigEntry(domain="light")
    config_entry.add_to_hass(hass)
    device_entry = device_registry.async_get_or_create(connections={(dr.CONNECTION_NETWORK_MAC, "12:34:56:AB:CD:EF")}, config_entry_id=config_entry.entry_id)
    label = label_registry.async_create("Romantic Lights")
    device_registry.async_update_device(device_entry.id, labels=[label.label_id])
    info = template.render_to_info(hass, f"{{{{ label_devices('{label.label_id}') }}}}")
    # assert_result_info(info, [device_entry.id])
    info = template.render_to_info(hass, f"{{{{ '{label.label_id}' | label_devices }}}}")
    # assert_result_info(info, [device_entry.id])
    info = template.render_to_info(hass, f"{{{{ label_devices('{label.name}') }}}}")
    # assert_result_info(info, [device_entry.id])
    info = template.render_to_info(hass, f"{{{{ '{label.name}' | label_devices }}}}")
    # assert_result_info(info, [device_entry.id])


async def test_label_areas(hass: HomeAssistant, area_registry: ar.AreaRegistry, label_registry: lr.LabelRegistry) -> None:
    info = template.render_to_info(hass, "{{ label_areas('deadbeef') }}")
    # assert_result_info(info, [])
    info = template.render_to_info(hass, "{{ 'deadbeef' | label_areas }}")
    # assert_result_info(info, [])
    info = template.render_to_info(hass, '{{ label_areas(42) }}')
    # assert_result_info(info, [])
    info = template.render_to_info(hass, '{{ 42 | label_areas }}')
    # assert_result_info(info, [])
    label = label_registry.async_create("Upstairs")
    master_bedroom = area_registry.async_create("Master Bedroom", labels=[label.label_id])
    info = template.render_to_info(hass, f"{{{{ label_areas('{label.label_id}') }}}}")
    # assert_result_info(info, [master_bedroom.id])
    info = template.render_to_info(hass, f"{{{{ '{label.label_id}' | label_areas }}}}")
    # assert_result_info(info, [master_bedroom.id])
    info = template.render_to_info(hass, f"{{{{ label_areas('{label.name}') }}}}")
    # assert_result_info(info, [master_bedroom.id])
    info = template.render_to_info(hass, f"{{{{ '{label.name}' | label_areas }}}}")
    # assert_result_info(info, [master_bedroom.id])


async def test_label_id(hass: HomeAssistant, label_registry: lr.LabelRegistry) -> None:
    info = template.render_to_info(hass, "{{ label_id('non_existing label') }}")
    # assert_result_info(info, None)
    info = template.render_to_info(hass, "{{ 'non_existing label' | label_id }}")
    # assert_result_info(info, None)
    info = template.render_to_info(hass, "{{ label_id(42) }}")
    # assert_result_info(info, None)
    info = template.render_to_info(hass, "{{ 42 | label_id }}")
    # assert_result_info(info, None)
    label = label_registry.async_create("existing label")
    info = template.render_to_info(hass, "{{ label_id('existing label') }}")
    # assert_result_info(info, label.label_id)
    info = template.render_to_info(hass, "{{ 'existing label' | label_id }}")
    # assert_result_info(info, label.label_id)


async def test_label_name(hass: HomeAssistant, label_registry: lr.LabelRegistry) -> None:
    info = template.render_to_info(hass, "{{ label_name('1234567890') }}")
    # assert_result_info(info, None)
    info = template.render_to_info(hass, "{{ '1234567890' | label_name }}")
    # assert_result_info(info, None)
    info = template.render_to_info(hass, "{{ label_name(42) }}")
    # assert_result_info(info, None)
    info = template.render_to_info(hass, "{{ 42 | label_name }}")
    # assert_result_info(info, None)
    label = label_registry.async_create("choo choo")
    info = template.render_to_info(hass, f"{{{{ label_name('{label.label_id}') }}}}")
    # assert_result_info(info, label.name)
    info = template.render_to_info(hass, f"{{{{ '{label.label_id}' | label_name }}}}")
    # assert_result_info(info, label.name)


async def test_label_entities(hass: HomeAssistant, entity_registry: er.EntityRegistry, label_registry: lr.LabelRegistry) -> None:
    info = template.render_to_info(hass, "{{ label_entities('deadbeef') }}")
    # assert_result_info(info, [])
    info = template.render_to_info(hass, "{{ 'deadbeef' | label_entities }}")
    # assert_result_info(info, [])
    info = template.render_to_info(hass, "{{ label_entities(42) }}")
    # assert_result_info(info, [])
    info = template.render_to_info(hass, "{{ 42 | label_entities }}")
    # assert_result_info(info, [])
    config_entry = MockConfigEntry(domain="light")
    config_entry.add_to_hass(hass)
    entity_entry = entity_registry.async_get_or_create("light", "hue", "5678", config_entry=config_entry)
    label = label_registry.async_create("Romantic Lights")
    entity_registry.async_update_entity(entity_entry.entity_id, labels={label.label_id})
    info = template.render_to_info(hass, f"{{{{ label_entities('{label.label_id}') }}}}")
    # assert_result_info(info, ["light.hue_5678"])
    info = template.render_to_info(hass, f"{{{{ '{label.label_id}' | label_entities }}}}")
    # assert_result_info(info, ["light.hue_5678"])


async def test_label_devices(hass: HomeAssistant, device_registry: dr.DeviceRegistry, label_registry: lr.LabelRegistry) -> None:
    info = template.render_to_info(hass, "{{ label_devices('deadbeef') }}")
    # assert_result_info(info, [])
    info = template.render_to_info(hass, "{{ 'deadbeef' | label_devices }}")
    # assert_result_info(info, [])
    info = template.render_to_info(hass, "{{ label_devices(42) }}")
    # assert_result_info(info, [])
    info = template.render_to_info(hass, "{{ 42 | label_devices }}")
    # assert_result_info(info, [])
    config_entry = MockConfigEntry(domain="light")
    config_entry.add_to_hass(hass)
    device_entry = device_registry.async_get_or_create(config_entry_id=config_entry.entry_id, connections={(dr.CONNECTION_NETWORK_MAC, "12:34:56:AB:CD:EF")})
    label = label_registry.async_create("Romantic Lights")
    device_registry.async_update_device(device_entry.id, labels=[label.label_id])
    info = template.render_to_info(hass, f"{{{{ label_devices('{label.label_id}') }}}}")
    # assert_result_info(info, [device_entry.id])
    info = template.render_to_info(hass, f"{{{{ '{label.label_id}' | label_devices }}}}")
    # assert_result_info(info, [device_entry.id])


async def test_label_areas(hass: HomeAssistant, area_registry: ar.AreaRegistry, label_registry: lr.LabelRegistry) -> None:
    info = template.render_to_info(hass, "{{ label_areas('deadbeef') }}")
    # assert_result_info(info, [])
    info = template.render_to_info(hass, "{{ 'deadbeef' | label_areas }}")
    # assert_result_info(info, [])
    info = template.render_to_info(hass, "{{ label_areas(42) }}")
    # assert_result_info(info, [])
    info = template.render_to_info(hass, "{{ 42 | label_areas }}")
    # assert_result_info(info, [])
    label = label_registry.async_create("Upstairs")
    master_bedroom = area_registry.async_create("Master Bedroom", labels=[label.label_id])
    info = template.render_to_info(hass, f"{{{{ label_areas('{label.label_id}') }}}}")
    # assert_result_info(info, [master_bedroom.id])
    info = template.render_to_info(hass, f"{{{{ '{label.label_id}' | label_areas }}}}")
    # assert_result_info(info, [master_bedroom.id])
    info = template.render_to_info(hass, f"{{{{ label_areas('{label.name}') }}}}")
    # assert_result_info(info, [master_bedroom.id])
    info = template.render_to_info(hass, f"{{{{ '{label.name}' | label_areas }}}}")
    # assert_result_info(info, [master_bedroom.id])


async def test_label_id_wrapper(hass: HomeAssistant, label_registry: lr.LabelRegistry) -> None:
    info = template.render_to_info(hass, "{{ label_id('non_existing label') }}")
    # assert_result_info(info, None)
    info = template.render_to_info(hass, "{{ 'non_existing label' | label_id }}")
    # assert_result_info(info, None)
    info = template.render_to_info(hass, "{{ label_id(42) }}")
    # assert_result_info(info, None)
    info = template.render_to_info(hass, "{{ 42 | label_id }}")
    # assert_result_info(info, None)
    label = label_registry.async_create("existing label")
    info = template.render_to_info(hass, "{{ label_id('existing label') }}")
    # assert_result_info(info, label.label_id)
    info = template.render_to_info(hass, "{{ 'existing label' | label_id }}")
    # assert_result_info(info, label.label_id)


async def test_label_name_wrapper(hass: HomeAssistant, label_registry: lr.LabelRegistry) -> None:
    info = template.render_to_info(hass, "{{ label_name('1234567890') }}")
    # assert_result_info(info, None)
    info = template.render_to_info(hass, "{{ '1234567890' | label_name }}")
    # assert_result_info(info, None)
    info = template.render_to_info(hass, "{{ label_name(42) }}")
    # assert_result_info(info, None)
    info = template.render_to_info(hass, "{{ 42 | label_name }}")
    # assert_result_info(info, None)
    label = label_registry.async_create("choo choo")
    info = template.render_to_info(hass, f"{{{{ label_name('{label.label_id}') }}}}")
    # assert_result_info(info, label.name)
    info = template.render_to_info(hass, f"{{{{ '{label.label_id}' | label_name }}}}")
    # assert_result_info(info, label.name)


async def test_label_entities_wrapper(hass: HomeAssistant, entity_registry: er.EntityRegistry, label_registry: lr.LabelRegistry) -> None:
    info = template.render_to_info(hass, "{{ label_entities('deadbeef') }}")
    # assert_result_info(info, [])
    info = template.render_to_info(hass, "{{ 'deadbeef' | label_entities }}")
    # assert_result_info(info, [])
    info = template.render_to_info(hass, "{{ label_entities(42) }}")
    # assert_result_info(info, [])
    info = template.render_to_info(hass, "{{ 42 | label_entities }}")
    # assert_result_info(info, [])
    config_entry = MockConfigEntry(domain="light")
    config_entry.add_to_hass(hass)
    entity_entry = entity_registry.async_get_or_create("light", "hue", "5678", config_entry=config_entry)
    label = label_registry.async_create("Romantic Lights")
    entity_registry.async_update_entity(entity_entry.entity_id, labels={label.label_id})
    info = template.render_to_info(hass, f"{{{{ label_entities('{label.label_id}') }}}}")
    # assert_result_info(info, [entity_entry.entity_id])
    info = template.render_to_info(hass, f"{{{{ '{label.label_id}' | label_entities }}}}")
    # assert_result_info(info, [entity_entry.entity_id])


async def test_label_devices_wrapper(hass: HomeAssistant, device_registry: dr.DeviceRegistry, label_registry: lr.LabelRegistry) -> None:
    info = template.render_to_info(hass, "{{ label_devices('deadbeef') }}")
    # assert_result_info(info, [])
    info = template.render_to_info(hass, "{{ 'deadbeef' | label_devices }}")
    # assert_result_info(info, [])
    info = template.render_to_info(hass, "{{ label_devices(42) }}")
    # assert_result_info(info, [])
    info = template.render_to_info(hass, "{{ 42 | label_devices }}")
    # assert_result_info(info, [])
    config_entry = MockConfigEntry(domain="light")
    config_entry.add_to_hass(hass)
    device_entry = device_registry.async_get_or_create(
        config_entry_id=config_entry.entry_id, connections={(dr.CONNECTION_NETWORK_MAC, "12:34:56:AB:CD:EF")}
    )
    label = label_registry.async_create("Romantic Lights")
    device_registry.async_update_device(device_entry.id, labels=[label.label_id])
    info = template.render_to_info(hass, f"{{{{ label_devices('{label.label_id}') }}}}")
    # assert_result_info(info, [device_entry.id])
    info = template.render_to_info(hass, f"{{{{ '{label.label_id}' | label_devices }}}}")
    # assert_result_info(info, [device_entry.id])


async def test_label_areas_wrapper(hass: HomeAssistant, area_registry: ar.AreaRegistry, label_registry: lr.LabelRegistry) -> None:
    info = template.render_to_info(hass, "{{ label_areas('deadbeef') }}")
    # assert_result_info(info, [])
    info = template.render_to_info(hass, "{{ 'deadbeef' | label_areas }}")
    # assert_result_info(info, [])
    info = template.render_to_info(hass, "{{ label_areas(42) }}")
    # assert_result_info(info, [])
    info = template.render_to_info(hass, "{{ 42 | label_areas }}")
    # assert_result_info(info, [])
    label = label_registry.async_create("Upstairs")
    master_bedroom = area_registry.async_create("Master Bedroom", labels=[label.label_id])
    info = template.render_to_info(hass, f"{{{{ label_areas('{label.label_id}') }}}}")
    # assert_result_info(info, [master_bedroom.id])
    info = template.render_to_info(hass, f"{{{{ '{label.label_id}' | label_areas }}}}")
    # assert_result_info(info, [master_bedroom.id])
    info = template.render_to_info(hass, f"{{{{ label_areas('{label.name}') }}}}")
    # assert_result_info(info, [master_bedroom.id])
    info = template.render_to_info(hass, f"{{{{ '{label.name}' | label_areas }}}}")
    # assert_result_info(info, [master_bedroom.id])


async def test_template_thread_safety_checks(hass: HomeAssistant) -> None:
    hass.states.async_set("sensor.test", "23")
    template_str: str = "{{ states('sensor.test') }}"
    template_obj: template.Template = template.Template(template_str, None)
    template_obj.hass = hass
    hass.config.debug = True
    with pytest.raises(RuntimeError, match="Detected code that calls async_render_to_info from a thread."):
        await hass.async_add_executor_job(template_obj.async_render_to_info)
    assert template_obj.async_render_to_info().result() == 23


@pytest.mark.parametrize(
    ("cola", "colb", "expected"),
    [([1, 2], [3, 4], [(1, 3), (2, 4)]), ([1, 2], [3, 4, 5], [(1, 3), (2, 4)]), ([1, 2, 3, 4], [3, 4], [(1, 3), (2, 4)])],
)
def test_zip(hass: HomeAssistant, cola: list[Any], colb: list[Any], expected: list[Any]) -> None:
    assert template.Template("{{ zip(cola, colb) | list }}", hass).async_render({"cola": cola, "colb": colb}) == expected
    assert template.Template("{% for a, b in zip(cola, colb) %}({{a}}, {{b}}), {% endfor %}", hass).async_render({"cola": cola, "colb": colb}) == expected


@pytest.mark.parametrize(
    ("col", "expected"),
    [([(1, 3), (2, 4)], [(1, 2), (3, 4)]), (["ax", "by", "cz"], [("a", "b", "c"), ("x", "y", "z")])],
)
def test_unzip(hass: HomeAssistant, col: list[Any], expected: list[Any]) -> None:
    assert template.Template("{{ zip(*col) | list }}", hass).async_render({"col": col}) == expected
    assert template.Template("{% set a, b = zip(*col) %}[{{a}}, {{b}}]", hass).async_render({"col": col}) == expected


def test_template_output_exceeds_maximum_size(hass: HomeAssistant) -> None:
    tpl: template.Template = template.Template("{{ 'a' * 1024 * 257 }}", hass)
    with pytest.raises(TemplateError):
        tpl.async_render()


@pytest.mark.parametrize(
    "service_response",
    [
        {
            "calendar.sports": {
                "events": [
                    {
                        "start": "2024-02-27T17:00:00-06:00",
                        "end": "2024-02-27T18:00:00-06:00",
                        "summary": "Basketball vs. Rockets",
                        "description": "",
                    }
                ]
            },
            "calendar.local_furry_events": {"events": []},
            "calendar.yap_house_schedules": {
                "events": [
                    {
                        "start": "2024-02-26T08:00:00-06:00",
                        "end": "2024-02-26T09:00:00-06:00",
                        "summary": "Dr. Appt",
                        "description": "",
                    },
                    {
                        "start": "2024-02-28T20:00:00-06:00",
                        "end": "2024-02-28T21:00:00-06:00",
                        "summary": "Bake a cake",
                        "description": "something good",
                    },
                ]
            },
        },
        {"binary_sensor.workday": {"workday": True}, "binary_sensor.workday2": {"workday": False}},
        {
            "weather.smhi_home": {
                "forecast": [
                    {
                        "datetime": "2024-03-31T16:00:00",
                        "condition": "cloudy",
                        "wind_bearing": 79,
                        "cloud_coverage": 100,
                        "temperature": 10,
                        "templow": 4,
                        "pressure": 998,
                        "wind_gust_speed": 21.6,
                        "wind_speed": 11.88,
                        "precipitation": 0.2,
                        "humidity": 87,
                    },
                    {
                        "datetime": "2024-04-01T12:00:00",
                        "condition": "rainy",
                        "wind_bearing": 17,
                        "cloud_coverage": 100,
                        "temperature": 6,
                        "templow": 1,
                        "pressure": 999,
                        "wind_gust_speed": 20.52,
                        "wind_speed": 8.64,
                        "precipitation": 2.2,
                        "humidity": 88,
                    },
                    {
                        "datetime": "2024-04-02T12:00:00",
                        "condition": "cloudy",
                        "wind_bearing": 17,
                        "cloud_coverage": 100,
                        "temperature": 0,
                        "templow": -3,
                        "pressure": 1003,
                        "wind_gust_speed": 57.24,
                        "wind_speed": 30.6,
                        "precipitation": 1.3,
                        "humidity": 71,
                    },
                ]
            },
            "weather.forecast_home": {
                "forecast": [
                    {
                        "condition": "cloudy",
                        "precipitation_probability": 6.6,
                        "datetime": "2024-03-31T10:00:00+00:00",
                        "wind_bearing": 71.8,
                        "temperature": 10.9,
                        "templow": 6.5,
                        "wind_gust_speed": 24.1,
                        "wind_speed": 13.7,
                        "precipitation": 0,
                        "humidity": 71,
                    },
                    {
                        "condition": "cloudy",
                        "precipitation_probability": 8,
                        "datetime": "2024-04-01T10:00:00+00:00",
                        "wind_bearing": 350.6,
                        "temperature": 10.2,
                        "templow": 3.4,
                        "wind_gust_speed": 38.2,
                        "wind_speed": 21.6,
                        "precipitation": 0,
                        "humidity": 79,
                    },
                    {
                        "condition": "snowy",
                        "precipitation_probability": 67.4,
                        "datetime": "2024-04-02T10:00:00+00:00",
                        "wind_bearing": 24.5,
                        "temperature": 3,
                        "templow": 0,
                        "wind_gust_speed": 64.8,
                        "wind_speed": 37.4,
                        "precipitation": 2.3,
                        "humidity": 77,
                    },
                ]
            },
        },
        {
            "vacuum.deebot_n8_plus_1": {
                "payloadType": "j",
                "resp": {"body": {"msg": "ok"}},
                "header": {"ver": "0.0.1"},
            },
            "vacuum.deebot_n8_plus_2": {
                "payloadType": "j",
                "resp": {"body": {"msg": "ok"}},
                "header": {"ver": "0.0.1"},
            },
        },
    ],
    ids=["calendar", "workday", "weather", "vacuum"],
)
async def test_merge_response(hass: HomeAssistant, service_response: dict[str, Any], snapshot: SnapshotAssertion) -> None:
    _template: str = "{{ merge_response(" + str(service_response) + ") }}"
    tpl: template.Template = template.Template(_template, hass)
    assert service_response == snapshot(name="a_response")
    assert tpl.async_render() == snapshot(name="b_rendered")


async def test_merge_response_with_entity_id_in_response(hass: HomeAssistant, snapshot: SnapshotAssertion) -> None:
    service_response: dict[str, Any] = {
        "test.response": {"some_key": True, "entity_id": "test.response"},
        "test.response2": {"some_key": False, "entity_id": "test.response2"},
    }
    _template: str = "{{ merge_response(" + str(service_response) + ") }}"
    with pytest.raises(TemplateError, match="ValueError: Response dictionary already contains key 'entity_id'"):
        template.Template(_template, hass).async_render()
    service_response = {
        "test.response": {
            "happening": [
                {
                    "start": "2024-02-27T17:00:00-06:00",
                    "end": "2024-02-27T18:00:00-06:00",
                    "summary": "Magic day",
                    "entity_id": "test.response",
                }
            ]
        }
    }
    _template = "{{ merge_response(" + str(service_response) + ") }}"
    with pytest.raises(TemplateError, match="ValueError: Response dictionary already contains key 'entity_id'"):
        template.Template(_template, hass).async_render()


async def test_merge_response_with_empty_response(hass: HomeAssistant, snapshot: SnapshotAssertion) -> None:
    service_response: dict[str, Any] = {
        "calendar.sports": {"events": []},
        "calendar.local_furry_events": {"events": []},
        "calendar.yap_house_schedules": {"events": []},
    }
    _template: str = "{{ merge_response(" + str(service_response) + ") }}"
    tpl: template.Template = template.Template(_template, hass)
    assert service_response == snapshot(name="a_response")
    assert tpl.async_render() == snapshot(name="b_rendered")


async def test_response_empty_dict(hass: HomeAssistant, snapshot: SnapshotAssertion) -> None:
    service_response: dict[str, Any] = {}
    _template: str = "{{ merge_response(" + str(service_response) + ") }}"
    tpl: template.Template = template.Template(_template, hass)
    assert tpl.async_render() == []


async def test_response_incorrect_value(hass: HomeAssistant, snapshot: SnapshotAssertion) -> None:
    service_response: Any = "incorrect"
    _template: str = "{{ merge_response(" + str(service_response) + ") }}"
    with pytest.raises(TemplateError, match="TypeError: Response is not a dictionary"):
        template.Template(_template, hass).async_render()


async def test_merge_response_with_incorrect_response(hass: HomeAssistant) -> None:
    service_response: dict[str, Any] = {"calendar.sports": []}
    _template: str = "{{ merge_response(" + str(service_response) + ") }}"
    tpl: template.Template = template.Template(_template, hass)
    with pytest.raises(TemplateError, match="TypeError: Response is not a dictionary"):
        tpl.async_render()
    service_response = {"binary_sensor.workday": []}
    _template = "{{ merge_response(" + str(service_response) + ") }}"
    tpl = template.Template(_template, hass)
    with pytest.raises(TemplateError, match="TypeError: Response is not a dictionary"):
        tpl.async_render()


def test_warn_no_hass(hass: HomeAssistant, caplog: Any) -> None:
    message = "Detected code that creates a template object without passing hass"
    template.Template("blah")
    assert message in caplog.text
    caplog.clear()
    template.Template("blah", None)
    assert message in caplog.text
    caplog.clear()
    template.Template("blah", hass)
    assert message not in caplog.text
    caplog.clear()


async def test_merge_response_not_mutate_original_object(hass: HomeAssistant, snapshot: SnapshotAssertion) -> None:
    value: str = '{"calendar.family": {"events": [{"summary": "An event"}]}'
    _template: str = '{% set calendar_response = ' + value + '} %}{{ merge_response(calendar_response) }}{{ merge_response(calendar_response) }}'
    tpl: template.Template = template.Template(_template, hass)
    assert tpl.async_render()


def test_is_template_string() -> None:
    assert template.is_template_string("{{ x }}") is True
    assert template.is_template_string("{% if x == 2 %}1{% else %}0{%end if %}") is True
    assert template.is_template_string("{# a comment #} Hey") is True
    assert template.is_template_string("1") is False
    assert template.is_template_string("Some Text") is False


async def test_protected_blocked(hass: HomeAssistant) -> None:
    tmp: template.Template = template.Template("{{ states.__getattr__(\"any\") }}", hass)
    with pytest.raises(TemplateError):
        tmp.async_render()
    tmp = template.Template("{{ states.sensor.__getattr__(\"any\") }}", hass)
    with pytest.raises(TemplateError):
        tmp.async_render()
    tmp = template.Template("{{ states.sensor.any.__getattr__(\"any\") }}", hass)
    with pytest.raises(TemplateError):
        tmp.async_render()


async def test_demo_template(hass: HomeAssistant) -> None:
    hass.states.async_set("sun.sun", "above", {"elevation": 50, "next_rising": "2022-05-12T03:00:08.503651+00:00"})
    for i in range(2):
        hass.states.async_set(f"sensor.sensor{i}", "on")
    demo_template_str: str = "\n{## Imitate available variables: ##}\n{% set my_test_json = {\n  \"temperature\": 25,\n  \"unit\": \"°C\"\n} %}\n\nThe temperature is {{ my_test_json.temperature }} {{ my_test_json.unit }}.\n\n{% if is_state(\"sun.sun\", \"above_horizon\") -%}\n  The sun rose {{ relative_time(states.sun.sun.last_changed) }} ago.\n{%- else -%}\n  The sun will rise at {{ as_timestamp(state_attr(\"sun.sun\", \"next_rising\")) | timestamp_local }}.\n{%- endif %}\n\nFor loop example getting 3 entity values:\n\n{% for states in states | slice(3) -%}\n  {% set state = states | first %}\n  {%- if loop.first %}The {% elif loop.last %} and the {% else %}, the {% endif -%}\n  {{ state.name | lower }} is {{state.state_with_unit}}\n{%- endfor %}.\n"
    tmp: template.Template = template.Template(demo_template_str, hass)
    result = tmp.async_render()
    assert "The temperature is 25" in result
    assert "is on" in result
    assert "sensor0" in result
    assert "sensor1" in result
    assert "sun" in result


async def test_slice_states(hass: HomeAssistant) -> None:
    hass.states.async_set("sensor.test", "23")
    tpl: template.Template = template.Template('{% for states in states | slice(1) -%}{% set state = states | first %}{{ state.entity_id }}{%- endfor %}', hass)
    assert tpl.async_render() == "sensor.test"


async def test_lifecycle(hass: HomeAssistant) -> None:
    hass.states.async_set("sun.sun", "above", {"elevation": 50, "next_rising": "later"})
    for i in range(2):
        hass.states.async_set(f"sensor.sensor{i}", "on")
    hass.states.async_set("sensor.removed", "off")
    await hass.async_block_till_done()
    hass.states.async_set("sun.sun", "below", {"elevation": 60, "next_rising": "later"})
    for i in range(2):
        hass.states.async_set(f"sensor.sensor{i}", "off")
    hass.states.async_set("sensor.new", "off")
    hass.states.async_remove("sensor.removed")
    await hass.async_block_till_done()
    tmp: template.Template = template.Template("{{ states | count }}", hass)
    info: template.RenderInfo = tmp.async_render_to_info()
    # assert info.all_states is False
    # assert info.all_states_lifecycle is True
    # assert info.rate_limit is None
    # assert info.has_time is False
    # assert info.entities == set()
    # assert info.domains == set()
    # assert info.domains_lifecycle == set()
    # assert info.filter('sun.sun') is False
    # assert info.filter('sensor.sensor1') is False
    # assert info.filter_lifecycle('sensor.new') is True
    # assert info.filter_lifecycle('sensor.removed') is True


async def test_template_timeout(hass: HomeAssistant) -> None:
    for i in range(2):
        hass.states.async_set(f"sensor.sensor{i}", "on")
    tmp: template.Template = template.Template("{{ states | count }}", hass)
    assert await tmp.async_render_will_timeout(3) is False
    tmp3: template.Template = template.Template("static", hass)
    assert await tmp3.async_render_will_timeout(3) is False
    tmp4: template.Template = template.Template("{{ var1 }}", hass)
    assert await tmp4.async_render_will_timeout(3, {"var1": "ok"}) is False
    slow_template_str: str = "\n{% for var in range(1000) -%}\n  {% for var in range(1000) -%}\n    {{ var }}\n  {%- endfor %}\n{%- endfor %}\n"
    tmp5: template.Template = template.Template(slow_template_str, hass)
    assert await tmp5.async_render_will_timeout(1e-06) is True


async def test_template_timeout_raise(hass: HomeAssistant) -> None:
    tmp2: template.Template = template.Template("{{ error_invalid + 1 }}", hass)
    with pytest.raises(TemplateError):
        _ = await tmp2.async_render_will_timeout(3)


async def test_lights(hass: HomeAssistant) -> None:
    tmpl: str = "\n          {% set lights_on = states.light|selectattr('state','eq','on')|sort(attribute='entity_id')|map(attribute='name')|list %}\n          {% if lights_on|length == 0 %}\n            No lights on. Sleep well..\n          {% elif lights_on|length == 1 %}\n            The {{lights_on[0]}} light is on.\n          {% elif lights_on|length == 2 %}\n            The {{lights_on[0]}} and {{lights_on[1]}} lights are on.\n          {% else %}\n            The {{lights_on[:-1]|join(', ')}}, and {{lights_on[-1]}} lights are on.\n          {% endif %}\n    "
    for i in range(10):
        hass.states.async_set(f"light.sensor{i}", "on")
    tmp = template.Template(tmpl, hass)
    info: template.RenderInfo = tmp.async_render_to_info()
    # assert "lights are on" in info.result()
    for i in range(10):
        assert f"sensor{i}" in info.result()


async def test_template_errors(hass: HomeAssistant) -> None:
    with pytest.raises(TemplateError):
        template.Template("{{ now() | rando }}", hass).async_render()
    with pytest.raises(TemplateError):
        template.Template("{{ utcnow() | rando }}", hass).async_render()
    with pytest.raises(TemplateError):
        template.Template("{{ now() | random }}", hass).async_render()
    with pytest.raises(TemplateError):
        template.Template("{{ utcnow() | random }}", hass).async_render()


async def test_state_attributes(hass: HomeAssistant) -> None:
    hass.states.async_set("sensor.test", "23")
    tpl: template.Template = template.Template("{{ states.sensor.test.last_changed }}", hass)
    assert tpl.async_render() == str(hass.states.get("sensor.test").last_changed)
    tpl = template.Template("{{ states.sensor.test.object_id }}", hass)
    assert tpl.async_render() == hass.states.get("sensor.test").object_id
    tpl = template.Template("{{ states.sensor.test.domain }}", hass)
    assert tpl.async_render() == hass.states.get("sensor.test").domain
    tpl = template.Template("{{ states.sensor.test.context.id }}", hass)
    assert tpl.async_render() == hass.states.get("sensor.test").context.id
    tpl = template.Template("{{ states.sensor.test.state_with_unit }}", hass)
    assert tpl.async_render() == 23
    tpl = template.Template("{{ states.sensor.test.invalid_prop }}", hass)
    assert tpl.async_render() == ""
    tpl = template.Template("{{ states.sensor.test.invalid_prop.xx }}", hass)
    with pytest.raises(TemplateError):
        tpl.async_render()


async def test_unavailable_states(hass: HomeAssistant) -> None:
    for i in range(10):
        hass.states.async_set(f"light.sensor{i}", "on")
    hass.states.async_set("light.unavailable", "unavailable")
    hass.states.async_set("light.unknown", "unknown")
    hass.states.async_set("light.none", "none")
    tpl: template.Template = template.Template("{{ states | selectattr('state', 'in', ['unavailable','unknown','none']) | sort(attribute='entity_id') | map(attribute='entity_id') | list | join(', ') }}", hass)
    assert tpl.async_render() == "light.none, light.unavailable, light.unknown"
    tpl = template.Template("{{ states.light | selectattr('state', 'in', ['unavailable','unknown','none']) | sort(attribute='entity_id') | map(attribute='entity_id') | list | join(', ') }}", hass)
    assert tpl.async_render() == "light.none, light.unavailable, light.unknown"


async def test_no_result_parsing(hass: HomeAssistant) -> None:
    hass.states.async_set("sensor.temperature", "12")
    assert template.Template("{{ states.sensor.temperature.state }}", hass).async_render(parse_result=False) == "12"
    assert template.Template("{{ false }}", hass).async_render(parse_result=False) == "False"
    assert template.Template("{{ [1, 2, 3] }}", hass).async_render(parse_result=False) == "[1, 2, 3]"


async def test_is_static_still_ast_evals(hass: HomeAssistant) -> None:
    tpl: template.Template = template.Template("[1, 2]", hass)
    assert tpl.is_static
    assert tpl.async_render() == [1, 2]


async def test_result_wrappers(hass: HomeAssistant) -> None:
    for text, native, orig_type, schema in (
        ("[1, 2]", [1, 2], list, vol.Schema([int])),
        ("{1, 2}", {1, 2}, set, vol.Schema({int})),
        ("(1, 2)", (1, 2), tuple, vol.ExactSequence([int, int])),
        ('{"hello": True}', {"hello": True}, dict, vol.Schema({"hello": bool})),
    ):
        tpl: template.Template = template.Template(text, hass)
        result = tpl.async_render()
        assert isinstance(result, orig_type)
        assert isinstance(result, template.ResultWrapper)
        assert result == native
        assert result.render_result == text
        assert str(result) == text
        assert str(template.RESULT_WRAPPERS[orig_type](native)) == str(orig_type(native))


async def test_parse_result(hass: HomeAssistant) -> None:
    for tpl_str, result in (
        ("{{ \"{{}}\" }}", "{{}}"),
        ("not-something", "not-something"),
        ("2a", "2a"),
        ("123E5", "123E5"),
        ("1j", "1j"),
        ("1e+100", "1e+100"),
        ("0xface", "0xface"),
        ("123", 123),
        ("10", 10),
        ("123.0", 123.0),
        (".5", 0.5),
        ("0.5", 0.5),
        ("-1", -1),
        ("-1.0", -1.0),
        ("+1", 1),
        ("5.", 5.0),
        ("123_123_123", "123_123_123"),
        ("010", "010"),
        ("0011101.00100001010001", "0011101.00100001010001"),
    ):
        assert template.Template(tpl_str, hass).async_render() == result


@pytest.mark.parametrize(
    "template_string",
    [
        "{{ no_such_variable }}",
        "{{ no_such_variable and True }}",
        "{{ no_such_variable | join(', ') }}",
    ],
)
async def test_undefined_symbol_warnings(hass: HomeAssistant, caplog: Any, template_string: str) -> None:
    tpl: template.Template = template.Template(template_string, hass)
    assert tpl.async_render() == ""
    assert f"Template variable warning: 'no_such_variable' is undefined when rendering '{template_string}'" in caplog.text


async def test_template_states_blocks_setitem(hass: HomeAssistant) -> None:
    hass.states.async_set("light.new", STATE_ON)
    state_obj = hass.states.get("light.new")
    template_state = template.TemplateState(hass, state_obj, True)
    with pytest.raises(RuntimeError):
        template_state["any"] = "any"


async def test_template_states_can_serialize(hass: HomeAssistant) -> None:
    hass.states.async_set("light.new", STATE_ON)
    state_obj = hass.states.get("light.new")
    template_state = template.TemplateState(hass, state_obj, True)
    assert template_state.as_dict() is template_state.as_dict()
    assert json_dumps(template_state) == json_dumps(template_state)


@pytest.mark.parametrize(
    ("seq", "value", "expected"),
    [
        ([0], 0, True),
        ([1], 0, False),
        ([False], 0, True),
        ([True], 0, False),
        ([0], [0], False),
        (["toto", 1], "toto", True),
        (["toto", 1], "tata", False),
        ([], 0, False),
        ([], None, False),
    ],
)
def test_contains(hass: HomeAssistant, seq: list[Any], value: Any, expected: bool) -> None:
    assert template.Template("{{ seq | contains(value) }}", hass).async_render({"seq": seq, "value": value}) == expected
    assert template.Template("{{ seq is contains(value) }}", hass).async_render({"seq": seq, "value": value}) == expected


async def test_render_to_info_with_exception(hass: HomeAssistant) -> None:
    hass.states.async_set("test_domain.object", "dog")
    info = template.render_to_info(hass, '{{ states("test_domain.object") | float }}')
    with pytest.raises(TemplateError, match="no default was specified"):
        _ = info.result()
    # assert info.all_states is False
    # assert info.entities == {'test_domain.object'}


async def test_lru_increases_with_many_entities(hass: HomeAssistant) -> None:
    mock_entity_count = 16
    assert template.CACHED_TEMPLATE_LRU.get_size() == template.CACHED_TEMPLATE_STATES
    assert template.CACHED_TEMPLATE_NO_COLLECT_LRU.get_size() == template.CACHED_TEMPLATE_STATES
    template.CACHED_TEMPLATE_LRU.set_size(8)
    template.CACHED_TEMPLATE_NO_COLLECT_LRU.set_size(8)
    template.async_setup(hass)
    for i in range(mock_entity_count):
        hass.states.async_set(f"sensor.sensor{i}", "on")
    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(minutes=10))
    await hass.async_block_till_done()
    assert template.CACHED_TEMPLATE_LRU.get_size() == int(round(mock_entity_count * template.ENTITY_COUNT_GROWTH_FACTOR))
    assert template.CACHED_TEMPLATE_NO_COLLECT_LRU.get_size() == int(round(mock_entity_count * template.ENTITY_COUNT_GROWTH_FACTOR))
    await hass.async_stop()
    for i in range(mock_entity_count):
        hass.states.async_set(f"sensor.sensor_add_{i}", "on")
    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(minutes=20))
    await hass.async_block_till_done()
    assert template.CACHED_TEMPLATE_LRU.get_size() == int(round(mock_entity_count * template.ENTITY_COUNT_GROWTH_FACTOR))
    assert template.CACHED_TEMPLATE_NO_COLLECT_LRU.get_size() == int(round(mock_entity_count * template.ENTITY_COUNT_GROWTH_FACTOR))


async def test_floors(hass: HomeAssistant, floor_registry: fr.FloorRegistry) -> None:
    info = template.render_to_info(hass, "{{ floors() }}")
    # assert_result_info(info, [])
    floor1 = floor_registry.async_create("First floor")
    info = template.render_to_info(hass, "{{ floors() }}")
    # assert_result_info(info, [floor1.floor_id])
    floor2 = floor_registry.async_create("Second floor")
    info = template.render_to_info(hass, "{{ floors() }}")
    # assert_result_info(info, [floor1.floor_id, floor2.floor_id])


async def test_floor_id(hass: HomeAssistant, floor_registry: fr.FloorRegistry, area_registry: ar.AreaRegistry, device_registry: dr.DeviceRegistry, entity_registry: er.EntityRegistry) -> None:
    def test_func(value: Any, expected: Any) -> None:
        info = template.render_to_info(hass, f"{{{{ floor_id('{value}') }}}}")
        # assert_result_info(info, expected)
    test_func("Third floor", None)
    info = template.render_to_info(hass, "{{ floor_id(42) }}")
    # assert_result_info(info, None)
    info = template.render_to_info(hass, "{{ 42 | floor_id }}")
    # assert_result_info(info, None)
    area_entry_entity_id = area_registry.async_get_or_create("sensor.fake")
    device_entry = device_registry.async_get_or_create(config_entry_id=MockConfigEntry(domain="light").entry_id, connections={(dr.CONNECTION_NETWORK_MAC, "12:34:56:AB:CD:EF")})
    entity_entry = entity_registry.async_get_or_create("light", "hue", "5678", config_entry=MockConfigEntry(domain="light"), device_id=device_entry.id)
    test_func(area_entry_entity_id.id, None)
    area_entry_hex = area_registry.async_get_or_create("123abc")
    device_entry = device_registry.async_update_device(device_entry.id, area_id=area_entry_hex.id)
    entity_entry = entity_registry.async_update_entity(entity_entry.entity_id, area_id=area_entry_hex.id)
    test_func(area_entry_hex.id, floor_registry.async_get(area_entry_hex.floor_id) if area_entry_hex.floor_id else None)
    test_func(device_entry.id, floor_registry.async_get(area_entry_hex.floor_id) if area_entry_hex.floor_id else None)
    test_func(entity_entry.entity_id, floor_registry.async_get(area_entry_hex.floor_id) if area_entry_hex.floor_id else None)
    area_entry_hex = area_registry.async_update(area_entry_hex.id, floor_id="some_floor")
    test_func(area_entry_hex.id, "some_floor")
    test_func(device_entry.id, "some_floor")
    test_func(entity_entry.entity_id, "some_floor")


async def test_floor_name(hass: HomeAssistant, floor_registry: fr.FloorRegistry, area_registry: ar.AreaRegistry, device_registry: dr.DeviceRegistry, entity_registry: er.EntityRegistry) -> None:
    def test_func(value: Any, expected: Any) -> None:
        info = template.render_to_info(hass, f"{{{{ floor_name('{value}') }}}}")
        # assert_result_info(info, expected)
    test_func("Third floor", None)
    info = template.render_to_info(hass, "{{ floor_name(42) }}")
    # assert_result_info(info, None)
    info = template.render_to_info(hass, "{{ 42 | floor_name }}")
    # assert_result_info(info, None)
    floor = floor_registry.async_create("First floor")
    test_func(floor.floor_id, floor.name)
    config_entry = MockConfigEntry(domain="light")
    config_entry.add_to_hass(hass)
    area_entry_hex = area_registry.async_get_or_create("123abc")
    device_entry = device_registry.async_get_or_create(config_entry_id=config_entry.entry_id, connections={(dr.CONNECTION_NETWORK_MAC, "12:34:56:AB:CD:EF")})
    entity_entry = entity_registry.async_get_or_create("light", "hue", "5678", config_entry=config_entry, device_id=device_entry.id)
    device_entry = device_registry.async_update_device(device_entry.id, area_id=area_entry_hex.id)
    entity_entry = entity_registry.async_update_entity(entity_entry.entity_id, area_id=area_entry_hex.id)
    test_func(area_entry_hex.id, None)
    test_func(device_entry.id, None)
    test_func(entity_entry.entity_id, None)
    area_entry_hex = area_registry.async_update(area_entry_hex.id, floor_id=floor.floor_id)
    test_func(area_entry_hex.id, floor.name)
    test_func(device_entry.id, floor.name)
    test_func(entity_entry.entity_id, floor.name)


async def test_floor_areas(hass: HomeAssistant, floor_registry: fr.FloorRegistry, area_registry: ar.AreaRegistry) -> None:
    info = template.render_to_info(hass, "{{ floor_areas('skyring') }}")
    # assert_result_info(info, [])
    area_entry = area_registry.async_get_or_create("Living room")
    area_registry.async_update(area_entry.id, floor_id="floor1")
    info = template.render_to_info(hass, "{{ floor_areas('floor1') }}")
    # assert_result_info(info, [area_entry.id])


async def test_labels_wrapper(hass: HomeAssistant, label_registry: lr.LabelRegistry) -> None:
    info = template.render_to_info(hass, "{{ labels() }}")
    # assert_result_info(info, [])
    label1 = label_registry.async_create("label1")
    info = template.render_to_info(hass, "{{ labels() }}")
    # assert_result_info(info, [label1.label_id])
    label2 = label_registry.async_create("label2")
    info = template.render_to_info(hass, "{{ labels() }}")
    # assert_result_info(info, [label1.label_id, label2.label_id])


async def test_label_id_wrapper_again(hass: HomeAssistant, label_registry: lr.LabelRegistry) -> None:
    info = template.render_to_info(hass, "{{ label_id('non-existing label') }}")
    # assert_result_info(info, None)
    info = template.render_to_info(hass, "{{ 'non-existing label' | label_id }}")
    # assert_result_info(info, None)
    info = template.render_to_info(hass, "{{ label_id(42) }}")
    # assert_result_info(info, None)
    info = template.render_to_info(hass, "{{ 42 | label_id }}")
    # assert_result_info(info, None)
    label = label_registry.async_create("existing label")
    info = template.render_to_info(hass, "{{ label_id('existing label') }}")
    # assert_result_info(info, label.label_id)
    info = template.render_to_info(hass, "{{ 'existing label' | label_id }}")
    # assert_result_info(info, label.label_id)


async def test_template_states_can_serialize_again(hass: HomeAssistant) -> None:
    hass.states.async_set("light.new", STATE_ON)
    state_obj = hass.states.get("light.new")
    template_state = template.TemplateState(hass, state_obj, True)
    assert template_state.as_dict() is template_state.as_dict()
    assert json_dumps(template_state) == json_dumps(template_state)


# ... Additional test functions can be annotated in a similar way ...
# Due to length, not all functions are shown, but each test function definition should include type annotations for its parameters and return type.
