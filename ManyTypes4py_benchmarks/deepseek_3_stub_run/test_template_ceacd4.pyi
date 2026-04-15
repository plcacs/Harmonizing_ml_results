from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import date, datetime, timedelta
from types import MappingProxyType
from typing import Any, Literal, TypeVar, overload
from unittest.mock import Mock

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
from homeassistant.setup import SetupComponentReturn
from homeassistant.util import dt as dt_util
from homeassistant.util.read_only_dict import ReadOnlyDict
from homeassistant.util.unit_system import UnitSystem
from tests.common import MockConfigEntry

_T = TypeVar("_T")
_KT = TypeVar("_KT")
_VT = TypeVar("_VT")

def _set_up_units(hass: HomeAssistant) -> None: ...

def render(
    hass: HomeAssistant,
    template_str: str,
    variables: TemplateVarsType | None = None,
) -> Any: ...

def render_to_info(
    hass: HomeAssistant,
    template_str: str,
    variables: TemplateVarsType | None = None,
) -> template.RenderInfo: ...

def extract_entities(
    hass: HomeAssistant,
    template_str: str,
    variables: TemplateVarsType | None = None,
) -> frozenset[str]: ...

def assert_result_info(
    info: template.RenderInfo,
    result: Any,
    entities: list[str] | None = None,
    domains: list[str] | None = None,
    all_states: bool = False,
) -> None: ...

async def test_template_render_missing_hass(hass: HomeAssistant) -> None: ...

async def test_template_render_info_collision(hass: HomeAssistant) -> None: ...

def test_template_equality() -> None: ...

def test_invalid_template(hass: HomeAssistant) -> None: ...

def test_referring_states_by_entity_id(hass: HomeAssistant) -> None: ...

def test_invalid_entity_id(hass: HomeAssistant) -> None: ...

def test_raise_exception_on_error(hass: HomeAssistant) -> None: ...

def test_iterating_all_states(hass: HomeAssistant) -> None: ...

def test_iterating_all_states_unavailable(hass: HomeAssistant) -> None: ...

def test_iterating_domain_states(hass: HomeAssistant) -> None: ...

async def test_import(hass: HomeAssistant) -> None: ...

async def test_import_change(hass: HomeAssistant) -> None: ...

def test_loop_controls(hass: HomeAssistant) -> None: ...

def test_float_function(hass: HomeAssistant) -> None: ...

def test_float_filter(hass: HomeAssistant) -> None: ...

def test_int_filter(hass: HomeAssistant) -> None: ...

def test_int_function(hass: HomeAssistant) -> None: ...

def test_bool_function(hass: HomeAssistant) -> None: ...

def test_bool_filter(hass: HomeAssistant) -> None: ...

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
def test_isnumber(
    hass: HomeAssistant,
    value: Any,
    expected: bool,
) -> None: ...

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
def test_is_list(hass: HomeAssistant, value: Any, expected: bool) -> None: ...

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
def test_is_set(hass: HomeAssistant, value: Any, expected: bool) -> None: ...

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
def test_is_tuple(hass: HomeAssistant, value: Any, expected: bool) -> None: ...

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
def test_set(hass: HomeAssistant, value: Any, expected: set[Any]) -> None: ...

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
def test_tuple(hass: HomeAssistant, value: Any, expected: tuple[Any, ...]) -> None: ...

def test_converting_datetime_to_iterable(hass: HomeAssistant) -> None: ...

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
def test_is_datetime(
    hass: HomeAssistant,
    value: Any,
    expected: bool,
) -> None: ...

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
def test_is_string_like(
    hass: HomeAssistant,
    value: Any,
    expected: bool,
) -> None: ...

def test_rounding_value(hass: HomeAssistant) -> None: ...

def test_rounding_value_on_error(hass: HomeAssistant) -> None: ...

def test_multiply(hass: HomeAssistant) -> None: ...

def test_add(hass: HomeAssistant) -> None: ...

def test_logarithm(hass: HomeAssistant) -> None: ...

def test_sine(hass: HomeAssistant) -> None: ...

def test_cos(hass: HomeAssistant) -> None: ...

def test_tan(hass: HomeAssistant) -> None: ...

def test_sqrt(hass: HomeAssistant) -> None: ...

def test_arc_sine(hass: HomeAssistant) -> None: ...

def test_arc_cos(hass: HomeAssistant) -> None: ...

def test_arc_tan(hass: HomeAssistant) -> None: ...

def test_arc_tan2(hass: HomeAssistant) -> None: ...

def test_strptime(hass: HomeAssistant) -> None: ...

async def test_timestamp_custom(hass: HomeAssistant) -> None: ...

async def test_timestamp_local(hass: HomeAssistant) -> None: ...

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
def test_as_datetime(hass: HomeAssistant, input: str) -> None: ...

@pytest.mark.parametrize(
    ("input", "output"),
    [
        (1469119144, "2016-07-21 16:39:04+00:00"),
        (1469119144.0, "2016-07-21 16:39:04+00:00"),
        (-1, "1969-12-31 23:59:59+00:00"),
    ],
)
def test_as_datetime_from_timestamp(
    hass: HomeAssistant,
    input: int | float | str,
    output: str,
) -> None: ...

@pytest.mark.parametrize(
    ("input", "output"),
    [
        (
            "{% set dt = as_datetime('2024-01-01 16:00:00-08:00') %}",
            "2024-01-01 16:00:00-08:00",
        ),
        (
            "{% set dt = as_datetime('2024-01-29').date() %}",
            "2024-01-29 00:00:00",
        ),
    ],
)
def test_as_datetime_from_datetime(
    hass: HomeAssistant,
    input: str,
    output: str,
) -> None: ...

@pytest.mark.parametrize(
    ("input", "default", "output"),
    [
        (1469119144, 123, "2016-07-21 16:39:04+00:00"),
        ('"invalid"', ["default output"], list[str]),
        (["a", "list"], 0, int),
        ({"a": "dict"}, None, None),
    ],
)
def test_as_datetime_default(
    hass: HomeAssistant,
    input: Any,
    default: Any,
    output: Any,
) -> None: ...

def test_as_local(hass: HomeAssistant) -> None: ...

def test_to_json(hass: HomeAssistant) -> None: ...

def test_to_json_ensure_ascii(hass: HomeAssistant) -> None: ...

def test_from_json(hass: HomeAssistant) -> None: ...

def test_average(hass: HomeAssistant) -> None: ...

def test_median(hass: HomeAssistant) -> None: ...

def test_statistical_mode(hass: HomeAssistant) -> None: ...

def test_min(hass: HomeAssistant) -> None: ...

def test_max(hass: HomeAssistant) -> None: ...

@pytest.mark.parametrize("attribute", ["a", "b", "c"])
def test_min_max_attribute(
    hass: HomeAssistant,
    attribute: str,
) -> None: ...

def test_ord(hass: HomeAssistant) -> None: ...

def test_base64_encode(hass: HomeAssistant) -> None: ...

def test_base64_decode(hass: HomeAssistant) -> None: ...

def test_slugify(hass: HomeAssistant) -> None: ...

def test_ordinal(hass: HomeAssistant) -> None: ...

def test_timestamp_utc(hass: HomeAssistant) -> None: ...

def test_as_timestamp(hass: HomeAssistant) -> None: ...

@patch.object(random, "choice")
def test_random_every_time(
    test_choice: Mock,
    hass: HomeAssistant,
) -> None: ...

def test_passing_vars_as_keywords(hass: HomeAssistant) -> None: ...

def test_passing_vars_as_vars(hass: HomeAssistant) -> None: ...

def test_passing_vars_as_list(hass: HomeAssistant) -> None: ...

def test_passing_vars_as_list_element(hass: HomeAssistant) -> None: ...

def test_passing_vars_as_dict_element(hass: HomeAssistant) -> None: ...

def test_passing_vars_as_dict(hass: HomeAssistant) -> None: ...

def test_render_with_possible_json_value_with_valid_json(
    hass: HomeAssistant,
) -> None: ...

def test_render_with_possible_json_value_with_invalid_json(
    hass: HomeAssistant,
) -> None: ...

def test_render_with_possible_json_value_with_template_error_value(
    hass: HomeAssistant,
) -> None: ...

def test_render_with_possible_json_value_with_missing_json_value(
    hass: HomeAssistant,
) -> None: ...

def test_render_with_possible_json_value_valid_with_is_defined(
    hass: HomeAssistant,
) -> None: ...

def test_render_with_possible_json_value_undefined_json(
    hass: HomeAssistant,
) -> None: ...

def test_render_with_possible_json_value_undefined_json_error_value(
    hass: HomeAssistant,
) -> None: ...

def test_render_with_possible_json_value_non_string_value(
    hass: HomeAssistant,
) -> None: ...

def test_render_with_possible_json_value_and_parse_result(
    hass: HomeAssistant,
) -> None: ...

def test_render_with_possible_json_value_and_dont_parse_result(
    hass: HomeAssistant,
) -> None: ...

def test_if_state_exists(hass: HomeAssistant) -> None: ...

def test_is_hidden_entity(
    hass: HomeAssistant,
    entity_registry: er.EntityRegistry,
) -> None: ...

def test_is_state(hass: HomeAssistant) -> None: ...

def test_is_state_attr(hass: HomeAssistant) -> None: ...

def test_state_attr(hass: HomeAssistant) -> None: ...

def test_states_function(hass: HomeAssistant) -> None: ...

async def test_state_translated(
    hass: HomeAssistant,
    entity_registry: er.EntityRegistry,
) -> None: ...

def test_has_value(hass: HomeAssistant) -> None: ...

@patch(
    "homeassistant.helpers.template.TemplateEnvironment.is_safe_callable",
    return_value=True,
)
def test_now(mock_is_safe: Mock, hass: HomeAssistant) -> None: ...

@patch(
    "homeassistant.helpers.template.TemplateEnvironment.is_safe_callable",
    return_value=True,
)
def test_utcnow(mock_is_safe: Mock, hass: HomeAssistant) -> None: ...

@pytest.mark.parametrize(
    ("now", "expected", "expected_midnight", "timezone_str"),
    [
        (
            "2021-11-24 03:00:00+00:00",
            "2021-11-23T10:00:00-08:00",
            "2021-11-23T00:00:00-08:00",
            "America/Los_Angeles",
        ),
        (
            "2021-11-23 19:00:00-08:00",
            "2021-11-23T10:00:00-08:00",
            "2021-11-23T00:00:00-08:00",
            "America/Los_Angeles",
        ),
    ],
)
@patch(
    "homeassistant.helpers.template.TemplateEnvironment.is_safe_callable",
    return_value=True,