"""The test for the History Statistics sensor platform."""

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import patch

from freezegun import freeze_time
import pytest
import voluptuous as vol

from homeassistant import config as hass_config, core as ha
from homeassistant.components.history_stats.const import (
    CONF_END,
    CONF_START,
    DEFAULT_NAME,
    DOMAIN,
)
from homeassistant.components.history_stats.sensor import (
    PLATFORM_SCHEMA as SENSOR_SCHEMA,
)
from homeassistant.components.recorder import Recorder
from homeassistant.const import (
    ATTR_DEVICE_CLASS,
    CONF_ENTITY_ID,
    CONF_NAME,
    CONF_STATE,
    CONF_TYPE,
    SERVICE_RELOAD,
    STATE_UNKNOWN,
)
from homeassistant.core import HomeAssistant, State
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.entity_component import async_update_entity
from homeassistant.setup import async_setup_component
from homeassistant.util import dt as dt_util

from tests.common import MockConfigEntry, async_fire_time_changed, get_fixture_path
from tests.components.recorder.common import async_wait_recording_done
from tests.typing import RecorderInstanceGenerator


async def test_setup(recorder_mock: Recorder, hass: HomeAssistant) -> None:
    """Test the history statistics sensor setup."""
    config = {
        "sensor": {
            "platform": "history_stats",
            "entity_id": "binary_sensor.test_id",
            "state": "on",
            "start": "{{ utcnow().replace(hour=0)"
            ".replace(minute=0).replace(second=0) }}",
            "duration": "02:00",
            "name": "Test",
        },
    }

    assert await async_setup_component(hass, "sensor", config)
    await hass.async_block_till_done()

    state = hass.states.get("sensor.test")
    assert state.state == "0.0"


async def test_setup_config_entry(
    recorder_mock: Recorder, hass: HomeAssistant, loaded_entry: MockConfigEntry
) -> None:
    """Test the history statistics sensor setup from a config entry."""
    state = hass.states.get("sensor.unnamed_statistics")
    assert state.state == "2"


async def test_setup_multiple_states(
    recorder_mock: Recorder, hass: HomeAssistant
) -> None:
    """Test the history statistics sensor setup for multiple states."""
    config = {
        "sensor": {
            "platform": "history_stats",
            "entity_id": "binary_sensor.test_id",
            "state": ["on", "true"],
            "start": "{{ utcnow().replace(hour=0)"
            ".replace(minute=0).replace(second=0) }}",
            "duration": "02:00",
            "name": "Test",
        },
    }

    assert await async_setup_component(hass, "sensor", config)
    await hass.async_block_till_done()

    state = hass.states.get("sensor.test")
    assert state.state == "0.0"


@pytest.mark.parametrize(
    "config",
    [
        {
            "platform": "history_stats",
            "entity_id": "binary_sensor.test_id",
            "name": "Test",
            "state": "on",
            "start": "{{ utcnow() }}",
            "duration": "TEST",
        },
        {
            "platform": "history_stats",
            "entity_id": "binary_sensor.test_id",
            "name": "Test",
            "state": "on",
            "start": "{{ utcnow() }}",
        },
        {
            "platform": "history_stats",
            "entity_id": "binary_sensor.test_id",
            "name": "Test",
            "state": "on",
            "start": "{{ as_timestamp(utcnow()) - 3600 }}",
            "end": "{{ utcnow() }}",
            "duration": "01:00",
        },
    ],
)
def test_setup_invalid_config(config: dict[str, Any]) -> None:
    """Test the history statistics sensor setup with invalid config."""
    with pytest.raises(vol.Invalid):
        SENSOR_SCHEMA(config)


async def test_invalid_date_for_start(
    recorder_mock: Recorder, hass: HomeAssistant
) -> None:
    """Verify with an invalid date for start."""
    await async_setup_component(
        hass,
        "sensor",
        {
            "sensor": {
                "platform": "history_stats",
                "entity_id": "binary_sensor.test_id",
                "name": "test",
                "state": "on",
                "start": "{{ INVALID }}",
                "duration": "01:00",
            },
        },
    )
    await hass.async_block_till_done()
    assert hass.states.get("sensor.test") is None
    next_update_time = dt_util.utcnow() + timedelta(minutes=1)
    with freeze_time(next_update_time):
        async_fire_time_changed(hass, next_update_time)
        await hass.async_block_till_done()
    assert hass.states.get("sensor.test") is None


async def test_invalid_date_for_end(
    recorder_mock: Recorder, hass: HomeAssistant
) -> None:
    """Verify with an invalid date for end."""
    await async_setup_component(
        hass,
        "sensor",
        {
            "sensor": {
                "platform": "history_stats",
                "entity_id": "binary_sensor.test_id",
                "name": "test",
                "state": "on",
                "end": "{{ INVALID }}",
                "duration": "01:00",
            },
        },
    )
    await hass.async_block_till_done()
    assert hass.states.get("sensor.test") is None
    next_update_time = dt_util.utcnow() + timedelta(minutes=1)
    with freeze_time(next_update_time):
        async_fire_time_changed(hass, next_update_time)
        await hass.async_block_till_done()
    assert hass.states.get("sensor.test") is None


async def test_invalid_entity_in_template(
    recorder_mock: Recorder, hass: HomeAssistant
) -> None:
    """Verify with an invalid entity in the template."""
    await async_setup_component(
        hass,
        "sensor",
        {
            "sensor": {
                "platform": "history_stats",
                "entity_id": "binary_sensor.test_id",
                "name": "test",
                "state": "on",
                "end": "{{ states('binary_sensor.invalid').attributes.time }}",
                "duration": "01:00",
            },
        },
    )
    await hass.async_block_till_done()
    assert hass.states.get("sensor.test") is None
    next_update_time = dt_util.utcnow() + timedelta(minutes=1)
    with freeze_time(next_update_time):
        async_fire_time_changed(hass, next_update_time)
        await hass.async_block_till_done()
    assert hass.states.get("sensor.test") is None


async def test_invalid_entity_returning_none_in_template(
    recorder_mock: Recorder, hass: HomeAssistant
) -> None:
    """Verify with an invalid entity returning none in the template."""
    await async_setup_component(
        hass,
        "sensor",
        {
            "sensor": {
                "platform": "history_stats",
                "entity_id": "binary_sensor.test_id",
                "name": "test",
                "state": "on",
                "end": "{{ states.binary_sensor.invalid.attributes.time }}",
                "duration": "01:00",
            },
        },
    )
    await hass.async_block_till_done()
    assert hass.states.get("sensor.test") is None
    next_update_time = dt_util.utcnow() + timedelta(minutes=1)
    with freeze_time(next_update_time):
        async_fire_time_changed(hass, next_update_time)
        await hass.async_block_till_done()
    assert hass.states.get("sensor.test") is None


async def test_reload(recorder_mock: Recorder, hass: HomeAssistant) -> None:
    """Verify we can reload history_stats sensors."""
    hass.state = ha.CoreState.not_running
    hass.states.async_set("binary_sensor.test_id", "on")

    await async_setup_component(
        hass,
        "sensor",
        {
            "sensor": {
                "platform": "history_stats",
                "entity_id": "binary_sensor.test_id",
                "name": "test",
                "state": "on",
                "start": "{{ as_timestamp(utcnow()) - 3600 }}",
                "duration": "01:00",
            },
        },
    )
    await hass.async_block_till_done()
    await hass.async_start()
    await hass.async_block_till_done()

    assert len(hass.states.async_all()) == 2

    assert hass.states.get("sensor.test")

    yaml_path = get_fixture_path("configuration.yaml", "history_stats")
    with patch.object(hass_config, "YAML_CONFIG_FILE", yaml_path):
        await hass.services.async_call(
            DOMAIN,
            SERVICE_RELOAD,
            {},
            blocking=True,
        )
        await hass.async_block_till_done()

    assert len(hass.states.async_all()) == 2

    assert hass.states.get("sensor.test") is None
    assert hass.states.get("sensor.second_test")


async def test_measure_multiple(recorder_mock: Recorder, hass: HomeAssistant) -> None:
    """Test the history statistics sensor measure for multiple ."""
    start_time = dt_util.utcnow() - timedelta(minutes=60)
    t0 = start_time + timedelta(minutes=20)
    t1 = t0 + timedelta(minutes=10)
    t2 = t1 + timedelta(minutes=10)

    def _fake_states(*args: Any, **kwargs: Any) -> dict[str, list[State]]:
        return {
            "input_select.test_id": [
                ha.State("input_select.test_id", "", last_changed=start_time),
                ha.State("input_select.test_id", "orange", last_changed=t0),
                ha.State("input_select.test_id", "default", last_changed=t1),
                ha.State("input_select.test_id", "blue", last_changed=t2),
            ]
        }

    with patch(
        "homeassistant.components.recorder.history.state_changes_during_period",
        _fake_states,
    ):
        await async_setup_component(
            hass,
            "sensor",
            {
                "sensor": [
                    {
                        "platform": "history_stats",
                        "entity_id": "input_select.test_id",
                        "name": "sensor1",
                        "state": ["orange", "blue"],
                        "start": "{{ as_timestamp(utcnow()) - 3600 }}",
                        "end": "{{ utcnow() }}",
                        "type": "time",
                    },
                    {
                        "platform": "history_stats",
                        "entity_id": "unknown.test_id",
                        "name": "sensor2",
                        "state": ["orange", "blue"],
                        "start": "{{ as_timestamp(utcnow()) - 3600 }}",
                        "end": "{{ utcnow() }}",
                        "type": "time",
                    },
                    {
                        "platform": "history_stats",
                        "entity_id": "input_select.test_id",
                        "name": "sensor3",
                        "state": ["orange", "blue"],
                        "start": "{{ as_timestamp(utcnow()) - 3600 }}",
                        "end": "{{ utcnow() }}",
                        "type": "count",
                    },
                    {
                        "platform": "history_stats",
                        "entity_id": "input_select.test_id",
                        "name": "sensor4",
                        "state": ["orange", "blue"],
                        "start": "{{ as_timestamp(utcnow()) - 3600 }}",
                        "end": "{{ utcnow() }}",
                        "type": "ratio",
                    },
                ]
            },
        )
        await hass.async_block_till_done()
        for i in range(1, 5):
            await async_update_entity(hass, f"sensor.sensor{i}")
        await hass.async_block_till_done()

    assert round(float(hass.states.get("sensor.sensor1").state), 3) == 0.5
    assert hass.states.get("sensor.sensor2").state == "0.0"
    assert hass.states.get("sensor.sensor3").state == "2"
    assert hass.states.get("sensor.sensor4").state == "50.0"


async def test_measure(recorder_mock: Recorder, hass: HomeAssistant) -> None:
    """Test the history statistics sensor measure."""
    start_time = dt_util.utcnow() - timedelta(minutes=60)
    t0 = start_time + timedelta(minutes=20)
    t1 = t0 + timedelta(minutes=10)
    t2 = t1 + timedelta(minutes=10)

    def _fake_states(*args: Any, **kwargs: Any) -> dict[str, list[State]]:
        return {
            "binary_sensor.test_id": [
                ha.State("binary_sensor.test_id", "on", last_changed=t0),
                ha.State("binary_sensor.test_id", "off", last_changed=t1),
                ha.State("binary_sensor.test_id", "on", last_changed=t2),
            ]
        }

    with patch(
        "homeassistant.components.recorder.history.state_changes_during_period",
        _fake_states,
    ):
        await async_setup_component(
            hass,
            "sensor",
            {
                "sensor": [
                    {
                        "platform": "history_stats",
                        "entity_id": "binary_sensor.test_id",
                        "name": "sensor1",
                        "state": "on",
                        "start": "{{ as_timestamp(utcnow()) - 3600 }}",
                        "end": "{{ utcnow() }}",
                        "type": "time",
                    },
                    {
                        "platform": "history_stats",
                        "entity_id": "binary_sensor.test_id",
                        "name": "sensor2",
                        "state": "on",
                        "start": "{{ as_timestamp(utcnow()) - 3600 }}",
                        "end": "{{ utcnow() }}",
                        "type": "time",
                        "unique_id": "6b1f54e3-4065-43ca-8492-d0d4506a573a",
                    },
                    {
                        "platform": "history_stats",
                        "entity_id": "binary_sensor.test_id",
                        "name": "sensor3",
                        "state": "on",
                        "start": "{{ as_timestamp(now()) - 3600 }}",
                        "end": "{{ utcnow() }}",
                        "type": "count",
                    },
                    {
                        "platform": "history_stats",
                        "entity_id": "binary_sensor.test_id",
                        "name": "sensor4",
                        "state": "on",
                        "start": "{{ as_timestamp(utcnow()) - 3600 }}",
                        "end": "{{ utcnow() }}",
                        "type": "ratio",
                    },
                ]
            },
        )
        await hass.async_block_till_done()
        for i in range(1, 5):
            await async_update_entity(hass, f"sensor.sensor{i}")
        await hass.async_block_till_done()

    assert hass.states.get("sensor.sensor1").state == "0.5"
    assert 0.499 < float(hass.states.get("sensor.sensor2").state) < 0.501
    assert hass.states.get("sensor.sensor3").state == "2"
    assert hass.states.get("sensor.sensor4").state == "50.0"


async def test_async_on_entire_period(
    recorder_mock: Recorder, hass: HomeAssistant
) -> None:
    """Test the history statistics sensor measuring as on the entire period."""
    start_time = dt_util.utcnow() - timedelta(minutes=60)
    t0 = start_time + timedelta(minutes=20)
    t1 = t0 + timedelta(minutes=10)
    t2 = t1 + timedelta(minutes=10)

    def _fake_states(*args: Any, **kwargs: Any) -> dict[str, list[State]]:
        return {
            "binary_sensor.test_on_id": [
                ha.State(
                    "binary_sensor.test_on_id",
                    "on",
                    last_changed=(start_time - timedelta(seconds=10)),
                ha.State("binary_sensor.test_on_id", "on", last_changed=t0),
                ha.State("binary_sensor.test_on_id", "on", last_changed=t1),
                ha.State("binary_sensor.test_on_id", "on", last_changed=t2),
            ]
        }

    with patch(
        "homeassistant.components.recorder.history.state_changes_during_period",
        _fake_states,
    ):
        await async_setup_component(
            hass,
            "sensor",
            {
                "sensor": [
                    {
                        "platform": "history_stats",
                        "entity_id": "binary_sensor.test_on_id",
                        "name": "on_sensor1",
                        "state": "on",
                        "start": "{{ as_timestamp(utcnow()) - 3600 }}",
                        "end": "{{ utcnow() }}",
                        "type": "time",
                    },
                    {
                        "platform": "history_stats",
                        "entity_id": "binary_sensor.test_on_id",
                        "name": "on_sensor2",
                        "state": "on",
                        "start": "{{ as_timestamp(utcnow()) - 3600 }}",
                        "end": "{{ utcnow() }}",
                        "type": "time