"""The tests for the Prometheus exporter."""

from dataclasses import dataclass
import datetime
from http import HTTPStatus
from typing import Any, Self, TypedDict, Union
from unittest import mock

from freezegun import freeze_time
import prometheus_client
from prometheus_client.utils import floatToGoString
import pytest

from homeassistant.components import (
    alarm_control_panel,
    binary_sensor,
    climate,
    counter,
    cover,
    device_tracker,
    fan,
    humidifier,
    input_boolean,
    input_number,
    light,
    lock,
    number,
    person,
    prometheus,
    sensor,
    switch,
    update,
)
from homeassistant.components.alarm_control_panel import AlarmControlPanelState
from homeassistant.components.climate import (
    ATTR_CURRENT_TEMPERATURE,
    ATTR_FAN_MODE,
    ATTR_FAN_MODES,
    ATTR_HUMIDITY,
    ATTR_HVAC_ACTION,
    ATTR_HVAC_MODES,
    ATTR_TARGET_TEMP_HIGH,
    ATTR_TARGET_TEMP_LOW,
)
from homeassistant.components.fan import (
    ATTR_DIRECTION,
    ATTR_OSCILLATING,
    ATTR_PERCENTAGE,
    ATTR_PRESET_MODE,
    ATTR_PRESET_MODES,
    DIRECTION_FORWARD,
    DIRECTION_REVERSE,
)
from homeassistant.components.humidifier import ATTR_AVAILABLE_MODES
from homeassistant.components.lock import LockState
from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.const import (
    ATTR_BATTERY_LEVEL,
    ATTR_DEVICE_CLASS,
    ATTR_FRIENDLY_NAME,
    ATTR_MODE,
    ATTR_TEMPERATURE,
    ATTR_UNIT_OF_MEASUREMENT,
    CONCENTRATION_MICROGRAMS_PER_CUBIC_METER,
    CONTENT_TYPE_TEXT_PLAIN,
    DEGREE,
    PERCENTAGE,
    STATE_CLOSED,
    STATE_CLOSING,
    STATE_HOME,
    STATE_NOT_HOME,
    STATE_OFF,
    STATE_ON,
    STATE_OPEN,
    STATE_OPENING,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
    UnitOfEnergy,
    UnitOfTemperature,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.setup import async_setup_component
from homeassistant.util import dt as dt_util

from tests.typing import ClientSessionGenerator

PROMETHEUS_PATH = "homeassistant.components.prometheus"


class EntityMetric:
    """Represents a Prometheus metric for a Home Assistant entity."""

    metric_name: str
    labels: dict[str, str]

    @classmethod
    def required_labels(cls) -> list[str]:
        """List of all required labels for a Prometheus metric."""
        return [
            "domain",
            "friendly_name",
            "entity",
        ]

    def __init__(self, metric_name: str, **kwargs: Any) -> None:
        """Create a new EntityMetric based on metric name and labels."""
        self.metric_name = metric_name
        self.labels = kwargs

        # Labels that are required for all entities.
        for labelname in self.required_labels():
            assert labelname in self.labels
            assert self.labels[labelname] != ""

    def withValue(self, value: float) -> Self:
        """Return a metric with value."""
        return EntityMetricWithValue(self, value)

    @property
    def _metric_name_string(self) -> str:
        """Return a full metric name as a string."""
        labels = ",".join(
            f'{key}="{value}"' for key, value in sorted(self.labels.items())
        )
        return f"{self.metric_name}{{{labels}}}"

    def _in_metrics(self, metrics: list[str]) -> bool:
        """Report whether this metric exists in the provided Prometheus output."""
        return any(line.startswith(self._metric_name_string) for line in metrics)

    def assert_in_metrics(self, metrics: list[str]) -> None:
        """Assert that this metric exists in the provided Prometheus output."""
        assert self._in_metrics(metrics)

    def assert_not_in_metrics(self, metrics: list[str]) -> None:
        """Assert that this metric does not exist in Prometheus output."""
        assert not self._in_metrics(metrics)


class EntityMetricWithValue(EntityMetric):
    """Represents a Prometheus metric with a value."""

    value: float

    def __init__(self, metric: EntityMetric, value: float) -> None:
        """Create a new metric with a value based on a metric."""
        super().__init__(metric.metric_name, **metric.labels)
        self.value = value

    @property
    def _metric_string(self) -> str:
        """Return a full metric string."""
        value = floatToGoString(self.value)
        return f"{self._metric_name_string} {value}"

    def assert_in_metrics(self, metrics: list[str]) -> None:
        """Assert that this metric exists in the provided Prometheus output."""
        assert self._metric_string in metrics


@dataclass
class FilterTest:
    """Class for capturing a filter test."""

    id: str
    should_pass: bool


def test_entity_metric_generates_metric_name_string_without_value() -> None:
    """Test using EntityMetric to format a simple metric string without any value."""
    domain = "sensor"
    object_id = "outside_temperature"
    entity_metric = EntityMetric(
        metric_name="homeassistant_sensor_temperature_celsius",
        domain=domain,
        friendly_name="Outside Temperature",
        entity=f"{domain}.{object_id}",
    )
    assert entity_metric._metric_name_string == (
        "homeassistant_sensor_temperature_celsius{"
        'domain="sensor",'
        'entity="sensor.outside_temperature",'
        'friendly_name="Outside Temperature"}'
    )


def test_entity_metric_generates_metric_string_with_value() -> None:
    """Test using EntityMetric to format a simple metric string but with a metric value included."""
    domain = "sensor"
    object_id = "outside_temperature"
    entity_metric = EntityMetric(
        metric_name="homeassistant_sensor_temperature_celsius",
        domain=domain,
        friendly_name="Outside Temperature",
        entity=f"{domain}.{object_id}",
    ).withValue(17.2)
    assert entity_metric._metric_string == (
        "homeassistant_sensor_temperature_celsius{"
        'domain="sensor",'
        'entity="sensor.outside_temperature",'
        'friendly_name="Outside Temperature"}'
        " 17.2"
    )


def test_entity_metric_raises_exception_without_required_labels() -> None:
    """Test using EntityMetric to raise exception when required labels are missing."""
    domain = "sensor"
    object_id = "outside_temperature"
    test_kwargs = {
        "metric_name": "homeassistant_sensor_temperature_celsius",
        "domain": domain,
        "friendly_name": "Outside Temperature",
        "entity": f"{domain}.{object_id}",
    }

    assert len(EntityMetric.required_labels()) > 0

    for labelname in EntityMetric.required_labels():
        label_kwargs = dict(test_kwargs)
        # Delete the required label and ensure we get an exception
        del label_kwargs[labelname]
        with pytest.raises(AssertionError):
            EntityMetric(**label_kwargs)


def test_entity_metric_raises_exception_if_required_label_is_empty_string() -> None:
    """Test using EntityMetric to raise exception when required label value is empty string."""
    domain = "sensor"
    object_id = "outside_temperature"
    test_kwargs = {
        "metric_name": "homeassistant_sensor_temperature_celsius",
        "domain": domain,
        "friendly_name": "Outside Temperature",
        "entity": f"{domain}.{object_id}",
    }

    assert len(EntityMetric.required_labels()) > 0

    for labelname in EntityMetric.required_labels():
        label_kwargs = dict(test_kwargs)
        # Replace the required label with "" and ensure we get an exception
        label_kwargs[labelname] = ""
        with pytest.raises(AssertionError):
            EntityMetric(**label_kwargs)


def test_entity_metric_generates_alphabetically_ordered_labels() -> None:
    """Test using EntityMetric to format a simple metric string with labels alphabetically ordered."""
    domain = "sensor"
    object_id = "outside_temperature"

    static_metric_string = (
        "homeassistant_sensor_temperature_celsius{"
        'domain="sensor",'
        'entity="sensor.outside_temperature",'
        'friendly_name="Outside Temperature",'
        'zed_label="foo"'
        "}"
        " 17.2"
    )

    ordered_entity_metric = EntityMetric(
        metric_name="homeassistant_sensor_temperature_celsius",
        domain=domain,
        entity=f"{domain}.{object_id}",
        friendly_name="Outside Temperature",
        zed_label="foo",
    ).withValue(17.2)
    assert ordered_entity_metric._metric_string == static_metric_string

    unordered_entity_metric = EntityMetric(
        metric_name="homeassistant_sensor_temperature_celsius",
        zed_label="foo",
        entity=f"{domain}.{object_id}",
        friendly_name="Outside Temperature",
        domain=domain,
    ).withValue(17.2)
    assert unordered_entity_metric._metric_string == static_metric_string


def test_entity_metric_generates_metric_string_with_non_required_labels() -> None:
    """Test using EntityMetric to format a simple metric string but with extra labels and values included."""
    mode_entity_metric = EntityMetric(
        metric_name="climate_preset_mode",
        domain="climate",
        friendly_name="Ecobee",
        entity="climate.ecobee",
        mode="away",
    ).withValue(1)
    assert mode_entity_metric._metric_string == (
        "climate_preset_mode{"
        'domain="climate",'
        'entity="climate.ecobee",'
        'friendly_name="Ecobee",'
        'mode="away"'
        "}"
        " 1.0"
    )

    action_entity_metric = EntityMetric(
        metric_name="climate_action",
        domain="climate",
        friendly_name="HeatPump",
        entity="climate.heatpump",
        action="heating",
    ).withValue(1)
    assert action_entity_metric._metric_string == (
        "climate_action{"
        'action="heating",'
        'domain="climate",'
        'entity="climate.heatpump",'
        'friendly_name="HeatPump"'
        "}"
        " 1.0"
    )

    state_entity_metric = EntityMetric(
        metric_name="cover_state",
        domain="cover",
        friendly_name="Curtain",
        entity="cover.curtain",
        state="open",
    ).withValue(1)
    assert state_entity_metric._metric_string == (
        "cover_state{"
        'domain="cover",'
        'entity="cover.curtain",'
        'friendly_name="Curtain",'
        'state="open"'
        "}"
        " 1.0"
    )

    foo_entity_metric = EntityMetric(
        metric_name="homeassistant_sensor_temperature_celsius",
        domain="sensor",
        friendly_name="Outside Temperature",
        entity="sensor.outside_temperature",
        foo="bar",
    ).withValue(17.2)
    assert foo_entity_metric._metric_string == (
        "homeassistant_sensor_temperature_celsius{"
        'domain="sensor",'
        'entity="sensor.outside_temperature",'
        'foo="bar",'
        'friendly_name="Outside Temperature"'
        "}"
        " 17.2"
    )


def test_entity_metric_assert_helpers() -> None:
    """Test using EntityMetric for both assert_in_metrics and assert_not_in_metrics."""
    temp_metric = (
        "homeassistant_sensor_temperature_celsius{"
        'domain="sensor",'
        'entity="sensor.outside_temperature",'
        'foo="bar",'
        'friendly_name="Outside Temperature"'
        "}"
    )
    climate_metric = (
        "climate_preset_mode{"
        'domain="climate",'
        'entity="climate.ecobee",'
        'friendly_name="Ecobee",'
        'mode="away"'
        "}"
    )
    excluded_cover_metric = (
        "cover_state{"
        'domain="cover",'
        'entity="cover.curtain",'
        'friendly_name="Curtain",'
        'state="open"'
        "}"
    )
    metrics = [
        temp_metric,
        climate_metric,
    ]
    # First make sure the excluded metric is not present
    assert excluded_cover_metric not in metrics
    # now check for actual metrics
    temp_entity_metric = EntityMetric(
        metric_name="homeassistant_sensor_temperature_celsius",
        domain="sensor",
        friendly_name="Outside Temperature",
        entity="sensor.outside_temperature",
        foo="bar",
    )
    assert temp_entity_metric._metric_name_string == temp_metric
    temp_entity_metric.assert_in_metrics(metrics)

    climate_entity_metric = EntityMetric(
        metric_name="climate_preset_mode",
        domain="climate",
        friendly_name="Ecobee",
        entity="climate.ecobee",
        mode="away",
    )
    assert climate_entity_metric._metric_name_string == climate_metric
    climate_entity_metric.assert_in_metrics(metrics)

    excluded_cover_entity_metric = EntityMetric(
        metric_name="cover_state",
        domain="cover",
        friendly_name="Curtain",
        entity="cover.curtain",
        state="open",
    )
    assert excluded_cover_entity_metric._metric_name_string == excluded_cover_metric
    excluded_cover_entity_metric.assert_not_in_metrics(metrics)


def test_entity_metric_with_value_assert_helpers() -> None:
    """Test using EntityMetricWithValue helpers, which is only assert_in_metrics."""
    temp_metric = (
        "homeassistant_sensor_temperature_celsius{"
        'domain="sensor",'
        'entity="sensor.outside_temperature",'
        'foo="bar",'
        'friendly_name="Outside Temperature"'
        "}"
        " 17.2"
    )
    climate_metric = (
        "climate_preset_mode{"
        'domain="climate",'
        'entity="climate.ecobee",'
        'friendly_name="Ecobee",'
        'mode="away"'
        "}"
        " 1.0"
    )
    metrics = [
        temp_metric,
        climate_metric,
    ]
    temp_entity_metric = EntityMetric(
        metric_name="homeassistant_sensor_temperature_celsius",
        domain="sensor",
        friendly_name="Outside Temperature",
        entity="sensor.outside_temperature",
        foo="bar",
    ).withValue(17.2)
    assert temp_entity_metric._metric_string == temp_metric
    temp_entity_metric.assert_in_metrics(metrics)

    climate_entity_metric = EntityMetric(
        metric_name="climate_preset_mode",
        domain="climate",
        friendly_name="Ecobee",
        entity="climate.ecobee",
        mode="away",
    ).withValue(1)
    assert climate_entity_metric._metric_string == climate_metric
    climate_entity_metric.assert_in_metrics(metrics)


@pytest.fixture(name="client")
async def setup_prometheus_client(
    hass: HomeAssistant,
    hass_client: ClientSessionGenerator,
    namespace: str,
) -> ClientSessionGenerator:
    """Initialize an hass_client with Prometheus component."""
    # Reset registry
    prometheus_client.REGISTRY = prometheus_client.CollectorRegistry(auto_describe=True)
    prometheus_client.ProcessCollector(registry=prometheus_client.REGISTRY)
    prometheus_client.PlatformCollector(registry=prometheus_client.REGISTRY)
    prometheus_client.GCCollector(registry=prometheus_client.REGISTRY)

    config = {}
    if namespace is not None:
        config[prometheus.CONF_PROM_NAMESPACE] = namespace
    assert await async_setup_component(
        hass, prometheus.DOMAIN, {prometheus.DOMAIN: config}
    )
    await hass.async_block_till_done()

    return await hass_client()


async def generate_latest_metrics(client: ClientSessionGenerator) -> list[str]:
    """Generate the latest metrics and transform the body."""
    resp = await client.get(prometheus.API_ENDPOINT)
    assert resp.status == HTTPStatus.OK
    assert resp.headers["content-type"] == CONTENT_TYPE_TEXT_PLAIN
    body = await resp.text()
    body = body.split("\n")

    assert len(body) > 3

    return body


@pytest.mark.parametrize("namespace", [""])
async def test_setup_enumeration(
    hass: HomeAssistant,
    hass_client: ClientSessionGenerator,
    entity_registry: er.EntityRegistry,
    namespace: str,
) -> None:
    """Test that setup enumerates existing states/entities."""

    # The order of when things are created must be carefully controlled in
    # this test, so we don't use fixtures.

    sensor_1 = entity_registry.async_get_or_create(
        domain=sensor.DOMAIN,
        platform="test",
        unique_id="sensor_1",
        unit_of_measurement=UnitOfTemperature.CELSIUS,
        original_device_class=SensorDeviceClass.TEMPERATURE,
        suggested_object_id="outside_temperature",
        original_name="Outside Temperature",
    )
    state = 12.3
    set_state_with_entry(hass, sensor_1, state, {})
    assert await async_setup_component(hass, prometheus.DOMAIN, {prometheus.DOMAIN: {}})

    client = await