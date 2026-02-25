from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta
import json
import logging
import math
import random
from types import MappingProxyType
from typing import Any, cast
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


def _set_up_units(hass: HomeAssistant) -> None:
    """Set up the tests."""
    hass.config.units = UnitSystem(
        "custom",
        accumulated_precipitation=UnitOfPrecipitationDepth.MILLIMETERS,
        area=UnitOfArea.SQUARE_METERS,
        conversions={},
        length=UnitOfLength.METERS,
        mass=UnitOfMass.GRAMS,
        pressure=UnitOfPressure.PA,
        temperature=UnitOfTemperature.CELSIUS,
        volume=UnitOfVolume.LITERS,
        wind_speed=UnitOfSpeed.KILOMETERS_PER_HOUR,
    )


def render(
    hass: HomeAssistant, template_str: str, variables: TemplateVarsType | None = None
) -> Any:
    """Create render info from template."""
    tmp = template.Template(template_str, hass)
    return tmp.async_render(variables)


def render_to_info(
    hass: HomeAssistant, template_str: str, variables: TemplateVarsType | None = None
) -> template.RenderInfo:
    """Create render info from template."""
    tmp = template.Template(template_str, hass)
    return tmp.async_render_to_info(variables)


def extract_entities(
    hass: HomeAssistant, template_str: str, variables: TemplateVarsType | None = None
) -> set[str]:
    """Extract entities from a template."""
    info = render_to_info(hass, template_str, variables)
    return info.entities


def assert_result_info(
    info: template.RenderInfo,
    result: Any,
    entities: Iterable[str] | None = None,
    domains: Iterable[str] | None = None,
    all_states: bool = False,
) -> None:
    """Check result info."""
    assert info.result() == result
    assert info.all_states == all_states
    assert info.filter("invalid_entity_name.somewhere") == all_states
    if entities is not None:
        assert info.entities == frozenset(entities)
        assert all(info.filter(entity) for entity in entities)
        if not all_states:
            assert not info.filter("invalid_entity_name.somewhere")
    else:
        assert not info.entities
    if domains is not None:
        assert info.domains == frozenset(domains)
        assert all(info.filter(domain + ".entity") for domain in domains)
    else:
        assert not hasattr(info, "_domains")
