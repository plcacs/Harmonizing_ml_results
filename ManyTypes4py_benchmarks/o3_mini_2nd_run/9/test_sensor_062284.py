from datetime import datetime, timedelta
from unittest.mock import patch
from freezegun import freeze_time
import pytest
import voluptuous as vol
from homeassistant import config as hass_config, core as ha
from homeassistant.components.history_stats.const import CONF_END, CONF_START, DEFAULT_NAME, DOMAIN
from homeassistant.components.history_stats.sensor import PLATFORM_SCHEMA as SENSOR_SCHEMA
from homeassistant.components.recorder import Recorder
from homeassistant.const import ATTR_DEVICE_CLASS, CONF_ENTITY_ID, CONF_NAME, CONF_STATE, CONF_TYPE, SERVICE_RELOAD, STATE_UNKNOWN
from homeassistant.core import HomeAssistant, State
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.entity_component import async_update_entity
from homeassistant.setup import async_setup_component
from homeassistant.util import dt as dt_util
from tests.common import MockConfigEntry, async_fire_time_changed, get_fixture_path
from tests.components.recorder.common import async_wait_recording_done
from tests.typing import RecorderInstanceGenerator
from typing import Any, Dict, List

async def test_setup(recorder_mock: Any, hass: HomeAssistant) -> None:
    config = {
        'sensor': {
            'platform': 'history_stats',
            'entity_id': 'binary_sensor.test_id',
            'state': 'on',
            'start': '{{ utcnow().replace(hour=0).replace(minute=0).replace(second=0) }}',
            'duration': '02:00',
            'name': 'Test'
        }
    }
    assert await async_setup_component(hass, 'sensor', config)
    await hass.async_block_till_done()
    state = hass.states.get('sensor.test')
    assert state.state == '0.0'


async def test_setup_config_entry(recorder_mock: Any, hass: HomeAssistant, loaded_entry: Any) -> None:
    state = hass.states.get('sensor.unnamed_statistics')
    assert state.state == '2'


async def test_setup_multiple_states(recorder_mock: Any, hass: HomeAssistant) -> None:
    config = {
        'sensor': {
            'platform': 'history_stats',
            'entity_id': 'binary_sensor.test_id',
            'state': ['on', 'true'],
            'start': '{{ utcnow().replace(hour=0).replace(minute=0).replace(second=0) }}',
            'duration': '02:00',
            'name': 'Test'
        }
    }
    assert await async_setup_component(hass, 'sensor', config)
    await hass.async_block_till_done()
    state = hass.states.get('sensor.test')
    assert state.state == '0.0'


@pytest.mark.parametrize('config', [
    {'platform': 'history_stats', 'entity_id': 'binary_sensor.test_id', 'name': 'Test', 'state': 'on',
     'start': '{{ utcnow() }}', 'duration': 'TEST'},
    {'platform': 'history_stats', 'entity_id': 'binary_sensor.test_id', 'name': 'Test', 'state': 'on',
     'start': '{{ utcnow() }}'},
    {'platform': 'history_stats', 'entity_id': 'binary_sensor.test_id', 'name': 'Test', 'state': 'on',
     'start': '{{ as_timestamp(utcnow()) - 3600 }}', 'end': '{{ utcnow() }}', 'duration': '01:00'}
])
def test_setup_invalid_config(config: Dict[str, Any]) -> None:
    with pytest.raises(vol.Invalid):
        SENSOR_SCHEMA(config)


async def test_invalid_date_for_start(recorder_mock: Any, hass: HomeAssistant) -> None:
    await async_setup_component(hass, 'sensor', {
        'sensor': {
            'platform': 'history_stats',
            'entity_id': 'binary_sensor.test_id',
            'name': 'test',
            'state': 'on',
            'start': '{{ INVALID }}',
            'duration': '01:00'
        }
    })
    await hass.async_block_till_done()
    assert hass.states.get('sensor.test') is None
    next_update_time = dt_util.utcnow() + timedelta(minutes=1)
    with freeze_time(next_update_time):
        async_fire_time_changed(hass, next_update_time)
        await hass.async_block_till_done()
    assert hass.states.get('sensor.test') is None


async def test_invalid_date_for_end(recorder_mock: Any, hass: HomeAssistant) -> None:
    await async_setup_component(hass, 'sensor', {
        'sensor': {
            'platform': 'history_stats',
            'entity_id': 'binary_sensor.test_id',
            'name': 'test',
            'state': 'on',
            'end': '{{ INVALID }}',
            'duration': '01:00'
        }
    })
    await hass.async_block_till_done()
    assert hass.states.get('sensor.test') is None
    next_update_time = dt_util.utcnow() + timedelta(minutes=1)
    with freeze_time(next_update_time):
        async_fire_time_changed(hass, next_update_time)
        await hass.async_block_till_done()
    assert hass.states.get('sensor.test') is None


async def test_invalid_entity_in_template(recorder_mock: Any, hass: HomeAssistant) -> None:
    await async_setup_component(hass, 'sensor', {
        'sensor': {
            'platform': 'history_stats',
            'entity_id': 'binary_sensor.test_id',
            'name': 'test',
            'state': 'on',
            'end': "{{ states('binary_sensor.invalid').attributes.time }}",
            'duration': '01:00'
        }
    })
    await hass.async_block_till_done()
    assert hass.states.get('sensor.test') is None
    next_update_time = dt_util.utcnow() + timedelta(minutes=1)
    with freeze_time(next_update_time):
        async_fire_time_changed(hass, next_update_time)
        await hass.async_block_till_done()
    assert hass.states.get('sensor.test') is None


async def test_invalid_entity_returning_none_in_template(recorder_mock: Any, hass: HomeAssistant) -> None:
    await async_setup_component(hass, 'sensor', {
        'sensor': {
            'platform': 'history_stats',
            'entity_id': 'binary_sensor.test_id',
            'name': 'test',
            'state': 'on',
            'end': '{{ states.binary_sensor.invalid.attributes.time }}',
            'duration': '01:00'
        }
    })
    await hass.async_block_till_done()
    assert hass.states.get('sensor.test') is None
    next_update_time = dt_util.utcnow() + timedelta(minutes=1)
    with freeze_time(next_update_time):
        async_fire_time_changed(hass, next_update_time)
        await hass.async_block_till_done()
    assert hass.states.get('sensor.test') is None


async def test_reload(recorder_mock: Any, hass: HomeAssistant) -> None:
    hass.state = ha.CoreState.not_running
    hass.states.async_set('binary_sensor.test_id', 'on')
    await async_setup_component(hass, 'sensor', {
        'sensor': {
            'platform': 'history_stats',
            'entity_id': 'binary_sensor.test_id',
            'name': 'test',
            'state': 'on',
            'start': '{{ as_timestamp(utcnow()) - 3600 }}',
            'duration': '01:00'
        }
    })
    await hass.async_block_till_done()
    await hass.async_start()
    await hass.async_block_till_done()
    assert len(hass.states.async_all()) == 2
    assert hass.states.get('sensor.test')
    yaml_path = get_fixture_path('configuration.yaml', 'history_stats')
    with patch.object(hass_config, 'YAML_CONFIG_FILE', yaml_path):
        await hass.services.async_call(DOMAIN, SERVICE_RELOAD, {}, blocking=True)
        await hass.async_block_till_done()
    assert len(hass.states.async_all()) == 2
    assert hass.states.get('sensor.test') is None
    assert hass.states.get('sensor.second_test')


async def test_measure_multiple(recorder_mock: Any, hass: HomeAssistant) -> None:
    start_time = dt_util.utcnow() - timedelta(minutes=60)
    t0 = start_time + timedelta(minutes=20)
    t1 = t0 + timedelta(minutes=10)
    t2 = t1 + timedelta(minutes=10)

    def _fake_states(*args: Any, **kwargs: Any) -> Dict[str, List[State]]:
        return {
            'input_select.test_id': [
                State('input_select.test_id', '', last_changed=start_time),
                State('input_select.test_id', 'orange', last_changed=t0),
                State('input_select.test_id', 'default', last_changed=t1),
                State('input_select.test_id', 'blue', last_changed=t2)
            ]
        }
    with patch('homeassistant.components.recorder.history.state_changes_during_period', _fake_states):
        await async_setup_component(hass, 'sensor', {
            'sensor': [
                {
                    'platform': 'history_stats',
                    'entity_id': 'input_select.test_id',
                    'name': 'sensor1',
                    'state': ['orange', 'blue'],
                    'start': '{{ as_timestamp(utcnow()) - 3600 }}',
                    'end': '{{ utcnow() }}',
                    'type': 'time'
                },
                {
                    'platform': 'history_stats',
                    'entity_id': 'unknown.test_id',
                    'name': 'sensor2',
                    'state': ['orange', 'blue'],
                    'start': '{{ as_timestamp(utcnow()) - 3600 }}',
                    'end': '{{ utcnow() }}',
                    'type': 'time'
                },
                {
                    'platform': 'history_stats',
                    'entity_id': 'input_select.test_id',
                    'name': 'sensor3',
                    'state': ['orange', 'blue'],
                    'start': '{{ as_timestamp(utcnow()) - 3600 }}',
                    'end': '{{ utcnow() }}',
                    'type': 'count'
                },
                {
                    'platform': 'history_stats',
                    'entity_id': 'input_select.test_id',
                    'name': 'sensor4',
                    'state': ['orange', 'blue'],
                    'start': '{{ as_timestamp(utcnow()) - 3600 }}',
                    'end': '{{ utcnow() }}',
                    'type': 'ratio'
                }
            ]
        })
        await hass.async_block_till_done()
        for i in range(1, 5):
            await async_update_entity(hass, f'sensor.sensor{i}')
        await hass.async_block_till_done()
    assert round(float(hass.states.get('sensor.sensor1').state), 3) == 0.5
    assert hass.states.get('sensor.sensor2').state == '0.0'
    assert hass.states.get('sensor.sensor3').state == '2'
    assert hass.states.get('sensor.sensor4').state == '50.0'


async def test_measure(recorder_mock: Any, hass: HomeAssistant) -> None:
    start_time = dt_util.utcnow() - timedelta(minutes=60)
    t0 = start_time + timedelta(minutes=20)
    t1 = t0 + timedelta(minutes=10)
    t2 = t1 + timedelta(minutes=10)

    def _fake_states(*args: Any, **kwargs: Any) -> Dict[str, List[State]]:
        return {
            'binary_sensor.test_id': [
                State('binary_sensor.test_id', 'on', last_changed=t0),
                State('binary_sensor.test_id', 'off', last_changed=t1),
                State('binary_sensor.test_id', 'on', last_changed=t2)
            ]
        }
    with patch('homeassistant.components.recorder.history.state_changes_during_period', _fake_states):
        await async_setup_component(hass, 'sensor', {
            'sensor': [
                {
                    'platform': 'history_stats',
                    'entity_id': 'binary_sensor.test_id',
                    'name': 'sensor1',
                    'state': 'on',
                    'start': '{{ as_timestamp(utcnow()) - 3600 }}',
                    'end': '{{ utcnow() }}',
                    'type': 'time'
                },
                {
                    'platform': 'history_stats',
                    'entity_id': 'binary_sensor.test_id',
                    'name': 'sensor2',
                    'state': 'on',
                    'start': '{{ as_timestamp(utcnow()) - 3600 }}',
                    'end': '{{ utcnow() }}',
                    'type': 'time',
                    'unique_id': '6b1f54e3-4065-43ca-8492-d0d4506a573a'
                },
                {
                    'platform': 'history_stats',
                    'entity_id': 'binary_sensor.test_id',
                    'name': 'sensor3',
                    'state': 'on',
                    'start': '{{ as_timestamp(now()) - 3600 }}',
                    'end': '{{ utcnow() }}',
                    'type': 'count'
                },
                {
                    'platform': 'history_stats',
                    'entity_id': 'binary_sensor.test_id',
                    'name': 'sensor4',
                    'state': 'on',
                    'start': '{{ as_timestamp(utcnow()) - 3600 }}',
                    'end': '{{ utcnow() }}',
                    'type': 'ratio'
                }
            ]
        })
        await hass.async_block_till_done()
        for i in range(1, 5):
            await async_update_entity(hass, f'sensor.sensor{i}')
        await hass.async_block_till_done()
    assert hass.states.get('sensor.sensor1').state == '0.5'
    sensor2_state = float(hass.states.get('sensor.sensor2').state)
    assert 0.499 < sensor2_state < 0.501
    assert hass.states.get('sensor.sensor3').state == '2'
    assert hass.states.get('sensor.sensor4').state == '50.0'


async def test_async_on_entire_period(recorder_mock: Any, hass: HomeAssistant) -> None:
    await hass.config.async_set_time_zone('UTC')
    utcnow = dt_util.utcnow()
    start_time = utcnow.replace(hour=0, minute=0, second=0, microsecond=0)

    def _fake_states(*args: Any, **kwargs: Any) -> Dict[str, List[State]]:
        return {
            'binary_sensor.test_on_id': [
                State('binary_sensor.test_on_id', 'on', last_changed=start_time - timedelta(seconds=10)),
                State('binary_sensor.test_on_id', 'on', last_changed=start_time + timedelta(minutes=20)),
                State('binary_sensor.test_on_id', 'on', last_changed=start_time + timedelta(minutes=30)),
                State('binary_sensor.test_on_id', 'on', last_changed=start_time + timedelta(minutes=40))
            ]
        }
    with patch('homeassistant.components.recorder.history.state_changes_during_period', _fake_states):
        await async_setup_component(hass, 'sensor', {
            'sensor': [
                {
                    'platform': 'history_stats',
                    'entity_id': 'binary_sensor.test_on_id',
                    'name': 'on_sensor1',
                    'state': 'on',
                    'start': '{{ as_timestamp(utcnow()) - 3600 }}',
                    'end': '{{ utcnow() }}',
                    'type': 'time'
                },
                {
                    'platform': 'history_stats',
                    'entity_id': 'binary_sensor.test_on_id',
                    'name': 'on_sensor2',
                    'state': 'on',
                    'start': '{{ as_timestamp(utcnow()) - 3600 }}',
                    'end': '{{ utcnow() }}',
                    'type': 'time'
                },
                {
                    'platform': 'history_stats',
                    'entity_id': 'binary_sensor.test_on_id',
                    'name': 'on_sensor3',
                    'state': 'on',
                    'start': '{{ as_timestamp(utcnow()) - 3600 }}',
                    'end': '{{ utcnow() }}',
                    'type': 'count'
                },
                {
                    'platform': 'history_stats',
                    'entity_id': 'binary_sensor.test_on_id',
                    'name': 'on_sensor4',
                    'state': 'on',
                    'start': '{{ as_timestamp(utcnow()) - 3600 }}',
                    'end': '{{ utcnow() }}',
                    'type': 'ratio'
                }
            ]
        })
        await hass.async_block_till_done()
        for i in range(1, 5):
            await async_update_entity(hass, f'sensor.on_sensor{i}')
        await hass.async_block_till_done()
    assert hass.states.get('sensor.on_sensor1').state == '1.0'
    assert hass.states.get('sensor.on_sensor2').state == '1.0'
    assert hass.states.get('sensor.on_sensor3').state == '1'
    assert hass.states.get('sensor.on_sensor4').state == '100.0'


async def test_async_off_entire_period(recorder_mock: Any, hass: HomeAssistant) -> None:
    await hass.config.async_set_time_zone('UTC')
    utcnow = dt_util.utcnow()
    start_time = utcnow - timedelta(minutes=60)
    t0 = start_time + timedelta(minutes=20)
    t1 = t0 + timedelta(minutes=10)
    t2 = t1 + timedelta(minutes=10)

    def _fake_states(*args: Any, **kwargs: Any) -> Dict[str, List[State]]:
        return {
            'binary_sensor.test_on_id': [
                State('binary_sensor.test_on_id', 'off', last_changed=start_time),
                State('binary_sensor.test_on_id', 'off', last_changed=t0),
                State('binary_sensor.test_on_id', 'off', last_changed=t1),
                State('binary_sensor.test_on_id', 'off', last_changed=t2)
            ]
        }
    await async_setup_component(hass, 'sensor', {
        'sensor': [
            {
                'platform': 'history_stats',
                'entity_id': 'binary_sensor.test_on_id',
                'name': 'on_sensor1',
                'state': 'on',
                'start': '{{ as_timestamp(utcnow()) - 3600 }}',
                'end': '{{ utcnow() }}',
                'type': 'time'
            },
            {
                'platform': 'history_stats',
                'entity_id': 'binary_sensor.test_on_id',
                'name': 'on_sensor2',
                'state': 'on',
                'start': '{{ as_timestamp(utcnow()) - 3600 }}',
                'end': '{{ utcnow() }}',
                'type': 'time'
            },
            {
                'platform': 'history_stats',
                'entity_id': 'binary_sensor.test_on_id',
                'name': 'on_sensor3',
                'state': 'on',
                'start': '{{ as_timestamp(utcnow()) - 3600 }}',
                'end': '{{ utcnow() }}',
                'type': 'count'
            },
            {
                'platform': 'history_stats',
                'entity_id': 'binary_sensor.test_on_id',
                'name': 'on_sensor4',
                'state': 'on',
                'start': '{{ as_timestamp(utcnow()) - 3600 }}',
                'end': '{{ utcnow() }}',
                'type': 'ratio'
            }
        ]
    })
    await hass.async_block_till_done()
    with patch('homeassistant.components.recorder.history.state_changes_during_period', _fake_states):
        for i in range(1, 5):
            await async_update_entity(hass, f'sensor.on_sensor{i}')
        await hass.async_block_till_done()
    assert hass.states.get('sensor.on_sensor1').state == '0.0'
    assert hass.states.get('sensor.on_sensor2').state == '0.0'
    assert hass.states.get('sensor.on_sensor3').state == '0'
    assert hass.states.get('sensor.on_sensor4').state == '0.0'


async def test_async_start_from_history_and_switch_to_watching_state_changes_single(recorder_mock: Any, hass: HomeAssistant) -> None:
    await hass.config.async_set_time_zone('UTC')
    utcnow = dt_util.utcnow()
    start_time = utcnow.replace(hour=0, minute=0, second=0, microsecond=0)

    def _fake_states(*args: Any, **kwargs: Any) -> Dict[str, List[State]]:
        return {
            'binary_sensor.state': [
                State('binary_sensor.state', 'on', last_changed=start_time, last_updated=start_time)
            ]
        }
    with patch('homeassistant.components.recorder.history.state_changes_during_period', _fake_states), freeze_time(start_time):
        await async_setup_component(hass, 'sensor', {
            'sensor': [
                {
                    'platform': 'history_stats',
                    'entity_id': 'binary_sensor.state',
                    'name': 'sensor1',
                    'state': 'on',
                    'start': '{{ utcnow().replace(hour=0, minute=0, second=0) }}',
                    'duration': {'hours': 2},
                    'type': 'time'
                }
            ]
        })
        await hass.async_block_till_done()
        await async_update_entity(hass, 'sensor.sensor1')
        await hass.async_block_till_done()
    assert hass.states.get('sensor.sensor1').state == '0.0'
    one_hour_in = start_time + timedelta(minutes=60)
    with freeze_time(one_hour_in):
        async_fire_time_changed(hass, one_hour_in)
        await hass.async_block_till_done()
    assert hass.states.get('sensor.sensor1').state == '1.0'
    turn_off_time = start_time + timedelta(minutes=90)
    with freeze_time(turn_off_time):
        hass.states.async_set('binary_sensor.state', 'off')
        await hass.async_block_till_done()
        async_fire_time_changed(hass, turn_off_time)
        await hass.async_block_till_done()
    assert hass.states.get('sensor.sensor1').state == '1.5'
    turn_back_on_time = start_time + timedelta(minutes=105)
    with freeze_time(turn_back_on_time):
        async_fire_time_changed(hass, turn_back_on_time)
        await hass.async_block_till_done()
    assert hass.states.get('sensor.sensor1').state == '1.5'
    with freeze_time(turn_back_on_time):
        hass.states.async_set('binary_sensor.state', 'on')
        await hass.async_block_till_done()
    assert hass.states.get('sensor.sensor1').state == '1.5'
    end_time = start_time + timedelta(minutes=120)
    with freeze_time(end_time):
        async_fire_time_changed(hass, end_time)
        await hass.async_block_till_done()
    assert hass.states.get('sensor.sensor1').state == '1.75'
    after_end_time = start_time + timedelta(minutes=125)
    with freeze_time(after_end_time):
        async_fire_time_changed(hass, after_end_time)
        await hass.async_block_till_done()
    assert hass.states.get('sensor.sensor1').state == '1.75'


async def test_async_start_from_history_and_switch_to_watching_state_changes_single_expanding_window(recorder_mock: Any, hass: HomeAssistant) -> None:
    await hass.config.async_set_time_zone('UTC')
    utcnow = dt_util.utcnow()
    start_time = utcnow.replace(hour=0, minute=0, second=0, microsecond=0)

    def _fake_states(*args: Any, **kwargs: Any) -> Dict[str, List[State]]:
        return {
            'binary_sensor.state': [
                State('binary_sensor.state', 'on', last_changed=start_time, last_updated=start_time)
            ]
        }
    with patch('homeassistant.components.recorder.history.state_changes_during_period', _fake_states), freeze_time(start_time):
        await async_setup_component(hass, 'sensor', {
            'sensor': [
                {
                    'platform': 'history_stats',
                    'entity_id': 'binary_sensor.state',
                    'name': 'sensor1',
                    'state': 'on',
                    'start': '{{ utcnow().replace(hour=0, minute=0, second=0) }}',
                    'end': '{{ utcnow() }}',
                    'type': 'time'
                },
                {
                    'platform': 'history_stats',
                    'entity_id': 'binary_sensor.state',
                    'name': 'sensor2',
                    'state': 'on',
                    'start': '{{ utcnow().replace(hour=0, minute=0, second=0) }}',
                    'end': '{{ utcnow() }}',
                    'type': 'time',
                    'unique_id': '6b1f54e3-4065-43ca-8492-d0d4506a573a'
                }
            ]
        })
        await hass.async_block_till_done()
        await async_update_entity(hass, 'sensor.sensor1')
        await hass.async_block_till_done()
    assert hass.states.get('sensor.sensor1').state == '0.0'
    assert hass.states.get('sensor.sensor2').state == '0.0'
    one_hour_in = start_time + timedelta(minutes=60)
    with freeze_time(one_hour_in):
        async_fire_time_changed(hass, one_hour_in)
        await hass.async_block_till_done()
    assert hass.states.get('sensor.sensor1').state == '1.0'
    assert hass.states.get('sensor.sensor2').state == '1.0'
    turn_off_time = start_time + timedelta(minutes=90)
    with freeze_time(turn_off_time):
        hass.states.async_set('binary_sensor.state', 'off')
        await hass.async_block_till_done()
        async_fire_time_changed(hass, turn_off_time)
        await hass.async_block_till_done()
    assert hass.states.get('sensor.sensor1').state == '1.5'
    assert hass.states.get('sensor.sensor2').state == '1.5'
    turn_back_on_time = start_time + timedelta(minutes=105)
    with freeze_time(turn_back_on_time):
        async_fire_time_changed(hass, turn_back_on_time)
        await hass.async_block_till_done()
    assert hass.states.get('sensor.sensor1').state == '1.5'
    assert hass.states.get('sensor.sensor2').state == '1.5'
    with freeze_time(turn_back_on_time):
        hass.states.async_set('binary_sensor.state', 'on')
        await hass.async_block_till_done()
    assert hass.states.get('sensor.sensor1').state == '1.5'
    assert hass.states.get('sensor.sensor2').state == '1.5'
    next_update_time = start_time + timedelta(minutes=107)
    with freeze_time(next_update_time):
        async_fire_time_changed(hass, next_update_time)
        await hass.async_block_till_done()
    assert hass.states.get('sensor.sensor1').state == '1.53'
    assert hass.states.get('sensor.sensor2').state == '1.53333333333333'
    end_time = start_time + timedelta(minutes=120)
    with freeze_time(end_time):
        async_fire_time_changed(hass, end_time)
        await hass.async_block_till_done()
    assert hass.states.get('sensor.sensor1').state == '1.75'
    assert hass.states.get('sensor.sensor2').state == '1.75'


async def test_async_start_from_history_and_switch_to_watching_state_changes_multiple(recorder_mock: Any, hass: HomeAssistant) -> None:
    await hass.config.async_set_time_zone('UTC')
    utcnow = dt_util.utcnow()
    start_time = utcnow.replace(hour=0, minute=0, second=0, microsecond=0)

    def _fake_states(*args: Any, **kwargs: Any) -> Dict[str, List[State]]:
        return {
            'binary_sensor.state': [
                State('binary_sensor.state', 'on', last_changed=start_time, last_updated=start_time)
            ]
        }
    with patch('homeassistant.components.recorder.history.state_changes_during_period', _fake_states), freeze_time(start_time):
        await async_setup_component(hass, 'sensor', {
            'sensor': [
                {
                    'platform': 'history_stats',
                    'entity_id': 'binary_sensor.state',
                    'name': 'sensor1',
                    'state': 'on',
                    'start': '{{ utcnow().replace(hour=0, minute=0, second=0) }}',
                    'duration': {'hours': 2},
                    'type': 'time'
                },
                {
                    'platform': 'history_stats',
                    'entity_id': 'binary_sensor.state',
                    'name': 'sensor2',
                    'state': 'on',
                    'start': '{{ utcnow().replace(hour=0, minute=0, second=0) }}',
                    'duration': {'hours': 2},
                    'type': 'time'
                },
                {
                    'platform': 'history_stats',
                    'entity_id': 'binary_sensor.state',
                    'name': 'sensor3',
                    'state': 'on',
                    'start': '{{ utcnow().replace(hour=0, minute=0, second=0) }}',
                    'duration': {'hours': 2},
                    'type': 'count'
                },
                {
                    'platform': 'history_stats',
                    'entity_id': 'binary_sensor.state',
                    'name': 'sensor4',
                    'state': 'on',
                    'start': '{{ utcnow().replace(hour=0, minute=0, second=0) }}',
                    'duration': {'hours': 2},
                    'type': 'ratio'
                }
            ]
        })
        await hass.async_block_till_done()
        for i in range(1, 5):
            await async_update_entity(hass, f'sensor.sensor{i}')
        await hass.async_block_till_done()
    assert hass.states.get('sensor.sensor1').state == '0.0'
    assert hass.states.get('sensor.sensor2').state == '0.0'
    assert hass.states.get('sensor.sensor3').state == '1'
    assert hass.states.get('sensor.sensor4').state == '0.0'
    one_hour_in = start_time + timedelta(minutes=60)
    with freeze_time(one_hour_in):
        async_fire_time_changed(hass, one_hour_in)
        await hass.async_block_till_done()
    assert hass.states.get('sensor.sensor1').state == '1.0'
    assert hass.states.get('sensor.sensor2').state == '1.0'
    assert hass.states.get('sensor.sensor3').state == '1'
    assert hass.states.get('sensor.sensor4').state == '50.0'
    turn_off_time = start_time + timedelta(minutes=90)
    with freeze_time(turn_off_time):
        hass.states.async_set('binary_sensor.state', 'off')
        await hass.async_block_till_done()
        async_fire_time_changed(hass, turn_off_time)
        await hass.async_block_till_done()
    assert hass.states.get('sensor.sensor1').state == '1.5'
    assert hass.states.get('sensor.sensor2').state == '1.5'
    assert hass.states.get('sensor.sensor3').state == '1'
    assert hass.states.get('sensor.sensor4').state == '75.0'
    turn_back_on_time = start_time + timedelta(minutes=105)
    with freeze_time(turn_back_on_time):
        async_fire_time_changed(hass, turn_back_on_time)
        await hass.async_block_till_done()
    assert hass.states.get('sensor.sensor1').state == '1.5'
    assert hass.states.get('sensor.sensor2').state == '1.5'
    assert hass.states.get('sensor.sensor3').state == '1'
    assert hass.states.get('sensor.sensor4').state == '75.0'
    with freeze_time(turn_back_on_time):
        hass.states.async_set('binary_sensor.state', 'on')
        await hass.async_block_till_done()
    assert hass.states.get('sensor.sensor1').state == '1.5'
    assert hass.states.get('sensor.sensor2').state == '1.5'
    assert hass.states.get('sensor.sensor3').state == '2'
    assert hass.states.get('sensor.sensor4').state == '75.0'
    end_time = start_time + timedelta(minutes=120)
    with freeze_time(end_time):
        async_fire_time_changed(hass, end_time)
        await hass.async_block_till_done()
    assert hass.states.get('sensor.sensor1').state == '1.75'
    assert hass.states.get('sensor.sensor2').state == '1.75'
    assert hass.states.get('sensor.sensor3').state == '2'
    assert hass.states.get('sensor.sensor4').state == '87.5'


async def test_does_not_work_into_the_future(recorder_mock: Any, hass: HomeAssistant) -> None:
    await hass.config.async_set_time_zone('UTC')
    utcnow = dt_util.utcnow()
    start_time = utcnow.replace(hour=0, minute=0, second=0, microsecond=0)

    def _fake_states(*args: Any, **kwargs: Any) -> Dict[str, List[State]]:
        return {
            'binary_sensor.state': [
                State('binary_sensor.state', 'on', last_changed=start_time, last_updated=start_time)
            ]
        }
    with patch('homeassistant.components.recorder.history.state_changes_during_period', _fake_states):
        with freeze_time(start_time):
            await async_setup_component(hass, 'sensor', {
                'sensor': [
                    {
                        'platform': 'history_stats',
                        'entity_id': 'binary_sensor.state',
                        'name': 'sensor1',
                        'state': 'on',
                        'start': '{{ utcnow().replace(hour=23, minute=0, second=0) }}',
                        'duration': {'hours': 1},
                        'type': 'time'
                    },
                    {
                        'platform': 'history_stats',
                        'entity_id': 'binary_sensor.state',
                        'name': 'sensor2',
                        'state': 'on',
                        'start': '{{ utcnow().replace(hour=23, minute=0, second=0) }}',
                        'duration': {'hours': 1},
                        'type': 'time',
                        'unique_id': '6b1f54e3-4065-43ca-8492-d0d4506a573a'
                    }
                ]
            })
            await async_update_entity(hass, 'sensor.sensor1')
            await hass.async_block_till_done()
        assert hass.states.get('sensor.sensor1').state == STATE_UNKNOWN
        assert hass.states.get('sensor.sensor2').state == STATE_UNKNOWN
        one_hour_in = start_time + timedelta(minutes=60)
        with freeze_time(one_hour_in):
            async_fire_time_changed(hass, one_hour_in)
            await hass.async_block_till_done(wait_background_tasks=True)
        assert hass.states.get('sensor.sensor1').state == STATE_UNKNOWN
        assert hass.states.get('sensor.sensor2').state == STATE_UNKNOWN
        turn_off_time = start_time + timedelta(minutes=90)
        with freeze_time(turn_off_time):
            hass.states.async_set('binary_sensor.state', 'off')
            await hass.async_block_till_done()
            async_fire_time_changed(hass, turn_off_time)
            await hass.async_block_till_done(wait_background_tasks=True)
        assert hass.states.get('sensor.sensor1').state == STATE_UNKNOWN
        assert hass.states.get('sensor.sensor2').state == STATE_UNKNOWN
        turn_back_on_time = start_time + timedelta(minutes=105)
        with freeze_time(turn_back_on_time):
            async_fire_time_changed(hass, turn_back_on_time)
            await hass.async_block_till_done(wait_background_tasks=True)
        assert hass.states.get('sensor.sensor1').state == STATE_UNKNOWN
        assert hass.states.get('sensor.sensor2').state == STATE_UNKNOWN
        with freeze_time(turn_back_on_time):
            hass.states.async_set('binary_sensor.state', 'on')
            await hass.async_block_till_done()
        assert hass.states.get('sensor.sensor1').state == STATE_UNKNOWN
        assert hass.states.get('sensor.sensor2').state == STATE_UNKNOWN
        end_time = start_time + timedelta(minutes=120)
        with freeze_time(end_time):
            async_fire_time_changed(hass, end_time)
            await hass.async_block_till_done(wait_background_tasks=True)
        assert hass.states.get('sensor.sensor1').state == STATE_UNKNOWN
        assert hass.states.get('sensor.sensor2').state == STATE_UNKNOWN
        in_the_window = start_time + timedelta(hours=23, minutes=5)
        with freeze_time(in_the_window):
            async_fire_time_changed(hass, in_the_window)
            await hass.async_block_till_done(wait_background_tasks=True)
        assert hass.states.get('sensor.sensor1').state == '0.08'
        assert hass.states.get('sensor.sensor2').state == '0.0833333333333333'
    past_the_window = start_time + timedelta(hours=25)
    with patch('homeassistant.components.recorder.history.state_changes_during_period', return_value=[]), freeze_time(past_the_window):
        async_fire_time_changed(hass, past_the_window)
        await hass.async_block_till_done(wait_background_tasks=True)
    assert hass.states.get('sensor.sensor1').state == STATE_UNKNOWN

    def _fake_off_states(*args: Any, **kwargs: Any) -> Dict[str, List[State]]:
        return {
            'binary_sensor.state': [
                State('binary_sensor.state', 'off', last_changed=start_time, last_updated=start_time)
            ]
        }
    past_the_window_with_data = start_time + timedelta(hours=26)
    with patch('homeassistant.components.recorder.history.state_changes_during_period', _fake_off_states), freeze_time(past_the_window_with_data):
        async_fire_time_changed(hass, past_the_window_with_data)
        await hass.async_block_till_done(wait_background_tasks=True)
    assert hass.states.get('sensor.sensor1').state == STATE_UNKNOWN
    at_the_next_window_with_data = start_time + timedelta(days=1, hours=23)
    with patch('homeassistant.components.recorder.history.state_changes_during_period', _fake_off_states), freeze_time(at_the_next_window_with_data):
        async_fire_time_changed(hass, at_the_next_window_with_data)
        await hass.async_block_till_done(wait_background_tasks=True)
    assert hass.states.get('sensor.sensor1').state == '0.0'


async def test_reload_before_start_event(recorder_mock: Any, hass: HomeAssistant) -> None:
    hass.state = ha.CoreState.not_running
    hass.states.async_set('binary_sensor.test_id', 'on')
    await async_setup_component(hass, 'sensor', {
        'sensor': {
            'platform': 'history_stats',
            'entity_id': 'binary_sensor.test_id',
            'name': 'test',
            'state': 'on',
            'start': '{{ as_timestamp(now()) - 3600 }}',
            'duration': '01:00'
        }
    })
    await hass.async_block_till_done()
    assert len(hass.states.async_all()) == 2
    assert hass.states.get('sensor.test')
    yaml_path = get_fixture_path('configuration.yaml', 'history_stats')
    with patch.object(hass_config, 'YAML_CONFIG_FILE', yaml_path):
        await hass.services.async_call(DOMAIN, SERVICE_RELOAD, {}, blocking=True)
        await hass.async_block_till_done()
    assert len(hass.states.async_all()) == 2
    assert hass.states.get('sensor.test') is None
    assert hass.states.get('sensor.second_test')


async def test_measure_sliding_window(recorder_mock: Any, hass: HomeAssistant) -> None:
    start_time = dt_util.utcnow() - timedelta(minutes=60)
    t0 = start_time + timedelta(minutes=20)
    t1 = t0 + timedelta(minutes=10)
    t2 = t1 + timedelta(minutes=10)

    def _fake_states(*args: Any, **kwargs: Any) -> Dict[str, List[State]]:
        return {
            'binary_sensor.test_id': [
                State('binary_sensor.test_id', 'on', last_changed=t0),
                State('binary_sensor.test_id', 'off', last_changed=t1),
                State('binary_sensor.test_id', 'on', last_changed=t2)
            ]
        }
    with patch('homeassistant.components.recorder.history.state_changes_during_period', _fake_states), freeze_time(start_time):
        await async_setup_component(hass, 'sensor', {
            'sensor': [
                {
                    'platform': 'history_stats',
                    'entity_id': 'binary_sensor.test_id',
                    'name': 'sensor1',
                    'state': 'on',
                    'start': '{{ as_timestamp(now()) - 3600 }}',
                    'end': '{{ as_timestamp(now()) + 3600 }}',
                    'type': 'time'
                },
                {
                    'platform': 'history_stats',
                    'entity_id': 'binary_sensor.test_id',
                    'name': 'sensor2',
                    'state': 'on',
                    'start': '{{ as_timestamp(now()) - 3600 }}',
                    'end': '{{ as_timestamp(now()) + 3600 }}',
                    'type': 'time',
                    'unique_id': '6b1f54e3-4065-43ca-8492-d0d4506a573a'
                },
                {
                    'platform': 'history_stats',
                    'entity_id': 'binary_sensor.test_id',
                    'name': 'sensor3',
                    'state': 'on',
                    'start': '{{ as_timestamp(now()) - 3600 }}',
                    'end': '{{ as_timestamp(now()) + 3600 }}',
                    'type': 'count'
                },
                {
                    'platform': 'history_stats',
                    'entity_id': 'binary_sensor.test_id',
                    'name': 'sensor4',
                    'state': 'on',
                    'start': '{{ as_timestamp(now()) - 3600 }}',
                    'end': '{{ as_timestamp(now()) + 3600 }}',
                    'type': 'ratio'
                }
            ]
        })
        await hass.async_block_till_done()
        for i in range(1, 5):
            await async_update_entity(hass, f'sensor.sensor{i}')
        await hass.async_block_till_done()
    assert hass.states.get('sensor.sensor1').state == '0.0'
    assert float(hass.states.get('sensor.sensor2').state) == 0
    assert hass.states.get('sensor.sensor3').state == '0'
    assert hass.states.get('sensor.sensor4').state == '0.0'
    past_next_update = start_time + timedelta(minutes=30)
    with patch('homeassistant.components.recorder.history.state_changes_during_period', _fake_states), freeze_time(past_next_update):
        async_fire_time_changed(hass, past_next_update)
        await hass.async_block_till_done(wait_background_tasks=True)
    assert hass.states.get('sensor.sensor1').state == '0.17'
    sensor2_val = float(hass.states.get('sensor.sensor2').state)
    assert 0.166 < sensor2_val < 0.167
    assert hass.states.get('sensor.sensor3').state == '1'
    assert hass.states.get('sensor.sensor4').state == '8.3'


async def test_measure_from_end_going_backwards(recorder_mock: Any, hass: HomeAssistant) -> None:
    start_time = dt_util.utcnow() - timedelta(minutes=60)
    t0 = start_time + timedelta(minutes=20)
    t1 = t0 + timedelta(minutes=10)
    t2 = t1 + timedelta(minutes=10)

    def _fake_states(*args: Any, **kwargs: Any) -> Dict[str, List[State]]:
        return {
            'binary_sensor.test_id': [
                State('binary_sensor.test_id', 'on', last_changed=t0),
                State('binary_sensor.test_id', 'off', last_changed=t1),
                State('binary_sensor.test_id', 'on', last_changed=t2)
            ]
        }
    with patch('homeassistant.components.recorder.history.state_changes_during_period', _fake_states), freeze_time(start_time):
        await async_setup_component(hass, 'sensor', {
            'sensor': [
                {
                    'platform': 'history_stats',
                    'entity_id': 'binary_sensor.test_id',
                    'name': 'sensor1',
                    'state': 'on',
                    'duration': {'hours': 1},
                    'end': '{{ utcnow() }}',
                    'type': 'time'
                },
                {
                    'platform': 'history_stats',
                    'entity_id': 'binary_sensor.test_id',
                    'name': 'sensor2',
                    'state': 'on',
                    'duration': {'hours': 1},
                    'end': '{{ utcnow() }}',
                    'type': 'time',
                    'unique_id': '6b1f54e3-4065-43ca-8492-d0d4506a573a'
                },
                {
                    'platform': 'history_stats',
                    'entity_id': 'binary_sensor.test_id',
                    'name': 'sensor3',
                    'state': 'on',
                    'duration': {'hours': 1},
                    'end': '{{ utcnow() }}',
                    'type': 'count'
                },
                {
                    'platform': 'history_stats',
                    'entity_id': 'binary_sensor.test_id',
                    'name': 'sensor4',
                    'state': 'on',
                    'duration': {'hours': 1},
                    'end': '{{ utcnow() }}',
                    'type': 'ratio'
                }
            ]
        })
        await hass.async_block_till_done()
        for i in range(1, 5):
            await async_update_entity(hass, f'sensor.sensor{i}')
        await hass.async_block_till_done()
    assert hass.states.get('sensor.sensor1').state == '0.0'
    assert float(hass.states.get('sensor.sensor2').state) == 0
    assert hass.states.get('sensor.sensor3').state == '0'
    assert hass.states.get('sensor.sensor4').state == '0.0'
    past_next_update = start_time + timedelta(minutes=30)
    with patch('homeassistant.components.recorder.history.state_changes_during_period', _fake_states), freeze_time(past_next_update):
        async_fire_time_changed(hass, past_next_update)
        await hass.async_block_till_done(wait_background_tasks=True)
    assert hass.states.get('sensor.sensor1').state == '0.17'
    sensor2_val = float(hass.states.get('sensor.sensor2').state)
    assert 0.166 < sensor2_val < 0.167
    assert hass.states.get('sensor.sensor3').state == '1'
    sensor4_val = float(hass.states.get('sensor.sensor4').state)
    assert 16.6 <= sensor4_val <= 16.7


async def test_measure_cet(recorder_mock: Any, hass: HomeAssistant) -> None:
    await hass.config.async_set_time_zone('Europe/Berlin')
    start_time = dt_util.utcnow() - timedelta(minutes=60)
    t0 = start_time + timedelta(minutes=20)
    t1 = t0 + timedelta(minutes=10)
    t2 = t1 + timedelta(minutes=10)

    def _fake_states(*args: Any, **kwargs: Any) -> Dict[str, List[State]]:
        return {
            'binary_sensor.test_id': [
                State('binary_sensor.test_id', 'on', last_changed=t0),
                State('binary_sensor.test_id', 'off', last_changed=t1),
                State('binary_sensor.test_id', 'on', last_changed=t2)
            ]
        }
    with patch('homeassistant.components.recorder.history.state_changes_during_period', _fake_states), freeze_time(start_time + timedelta(minutes=60)):
        await async_setup_component(hass, 'sensor', {
            'sensor': [
                {
                    'platform': 'history_stats',
                    'entity_id': 'binary_sensor.test_id',
                    'name': 'sensor1',
                    'state': 'on',
                    'start': '{{ as_timestamp(utcnow()) - 3600 }}',
                    'end': '{{ utcnow() }}',
                    'type': 'time'
                },
                {
                    'platform': 'history_stats',
                    'entity_id': 'binary_sensor.test_id',
                    'name': 'sensor2',
                    'state': 'on',
                    'start': '{{ as_timestamp(utcnow()) - 3600 }}',
                    'end': '{{ utcnow() }}',
                    'type': 'time',
                    'unique_id': '6b1f54e3-4065-43ca-8492-d0d4506a573a'
                },
                {
                    'platform': 'history_stats',
                    'entity_id': 'binary_sensor.test_id',
                    'name': 'sensor3',
                    'state': 'on',
                    'start': '{{ as_timestamp(utcnow()) - 3600 }}',
                    'end': '{{ utcnow() }}',
                    'type': 'count'
                },
                {
                    'platform': 'history_stats',
                    'entity_id': 'binary_sensor.test_id',
                    'name': 'sensor4',
                    'state': 'on',
                    'start': '{{ as_timestamp(utcnow()) - 3600 }}',
                    'end': '{{ utcnow() }}',
                    'type': 'ratio'
                }
            ]
        })
        await hass.async_block_till_done()
        for i in range(1, 5):
            await async_update_entity(hass, f'sensor.sensor{i}')
        await hass.async_block_till_done()
    assert hass.states.get('sensor.sensor1').state == '0.5'
    sensor2_val = float(hass.states.get('sensor.sensor2').state)
    assert 0.499 < sensor2_val < 0.501
    assert hass.states.get('sensor.sensor3').state == '2'
    assert hass.states.get('sensor.sensor4').state == '50.0'


async def test_state_change_during_window_rollover(recorder_mock: Any, hass: HomeAssistant) -> None:
    await hass.config.async_set_time_zone('UTC')
    utcnow = dt_util.utcnow()
    start_time = utcnow.replace(hour=23, minute=0, second=0, microsecond=0)

    def _fake_states(*args: Any, **kwargs: Any) -> Dict[str, List[State]]:
        return {
            'binary_sensor.state': [
                State('binary_sensor.state', 'on', last_changed=start_time - timedelta(hours=11), last_updated=start_time - timedelta(hours=11))
            ]
        }
    with patch('homeassistant.components.recorder.history.state_changes_during_period', _fake_states), freeze_time(start_time):
        await async_setup_component(hass, 'sensor', {
            'sensor': [
                {
                    'platform': 'history_stats',
                    'entity_id': 'binary_sensor.state',
                    'name': 'sensor1',
                    'state': 'on',
                    'start': '{{ today_at() }}',
                    'end': '{{ now() }}',
                    'type': 'time'
                }
            ]
        })
        await hass.async_block_till_done()
        await async_update_entity(hass, 'sensor.sensor1')
        await hass.async_block_till_done()
    assert hass.states.get('sensor.sensor1').state == '11.0'
    t2 = start_time + timedelta(minutes=59, microseconds=300)
    with freeze_time(t2):
        async_fire_time_changed(hass, t2)
        await hass.async_block_till_done()
    assert hass.states.get('sensor.sensor1').state == '11.98'
    t3 = t2 + timedelta(minutes=1)

    def _fake_states_t3(*args: Any, **kwargs: Any) -> Dict[str, List[State]]:
        return {
            'binary_sensor.state': [
                State('binary_sensor.state', 'on', last_changed=t3.replace(hour=0, minute=0, second=0, microsecond=0), last_updated=t3.replace(hour=0, minute=0, second=0, microsecond=0))
            ]
        }
    with patch('homeassistant.components.recorder.history.state_changes_during_period', _fake_states_t3), freeze_time(t3):
        hass.states.async_set('binary_sensor.state', 'off')
        await hass.async_block_till_done(wait_background_tasks=True)
    assert hass.states.get('sensor.sensor1').state == '0.0'
    t4 = t3 + timedelta(minutes=10)
    with freeze_time(t4):
        async_fire_time_changed(hass, t4)
        await hass.async_block_till_done()
    assert hass.states.get('sensor.sensor1').state == '0.0'


@pytest.mark.parametrize('time_zone', ['Europe/Berlin', 'America/Chicago', 'US/Hawaii'])
async def test_end_time_with_microseconds_zeroed(time_zone: str, async_setup_recorder_instance: Any, hass: HomeAssistant) -> None:
    await hass.config.async_set_time_zone(time_zone)
    start_of_today = dt_util.now().replace(day=9, month=7, year=1986, hour=0, minute=0, second=0, microsecond=0)
    with freeze_time(start_of_today):
        await async_setup_recorder_instance(hass)
        await hass.async_block_till_done()
        await async_wait_recording_done(hass)
    start_time = start_of_today + timedelta(minutes=60)
    t0 = start_time + timedelta(minutes=20)
    t1 = t0 + timedelta(minutes=10)
    t2 = t1 + timedelta(minutes=10)
    time_200 = start_of_today + timedelta(hours=2)

    def _fake_states(*args: Any, **kwargs: Any) -> Dict[str, List[State]]:
        return {
            'binary_sensor.heatpump_compressor_state': [
                State('binary_sensor.heatpump_compressor_state', 'on', last_changed=t0),
                State('binary_sensor.heatpump_compressor_state', 'off', last_changed=t1),
                State('binary_sensor.heatpump_compressor_state', 'on', last_changed=t2)
            ]
        }
    with freeze_time(time_200), patch('homeassistant.components.recorder.history.state_changes_during_period', _fake_states):
        await async_setup_component(hass, 'sensor', {
            'sensor': [
                {
                    'platform': 'history_stats',
                    'entity_id': 'binary_sensor.heatpump_compressor_state',
                    'name': 'heatpump_compressor_today',
                    'state': 'on',
                    'start': '{{ now().replace(hour=0, minute=0, second=0, microsecond=0) }}',
                    'end': '{{ now().replace(microsecond=0) }}',
                    'type': 'time'
                },
                {
                    'platform': 'history_stats',
                    'entity_id': 'binary_sensor.heatpump_compressor_state',
                    'name': 'heatpump_compressor_today2',
                    'state': 'on',
                    'start': '{{ now().replace(hour=0, minute=0, second=0, microsecond=0) }}',
                    'end': '{{ now().replace(microsecond=0) }}',
                    'type': 'time',
                    'unique_id': '6b1f54e3-4065-43ca-8492-d0d4506a573a'
                }
            ]
        })
        await hass.async_block_till_done()
        await async_update_entity(hass, 'sensor.heatpump_compressor_today')
        await hass.async_block_till_done()
        assert hass.states.get('sensor.heatpump_compressor_today').state == '0.5'
        sensor_today2_val = float(hass.states.get('sensor.heatpump_compressor_today2').state)
        assert 0.499 < sensor_today2_val < 0.501
        async_fire_time_changed(hass, time_200)
        assert hass.states.get('sensor.heatpump_compressor_today').state == '0.5'
        sensor_today2_val = float(hass.states.get('sensor.heatpump_compressor_today2').state)
        assert 0.499 < sensor_today2_val < 0.501
        hass.states.async_set('binary_sensor.heatpump_compressor_state', 'off')
        await hass.async_block_till_done()
    time_400 = start_of_today + timedelta(hours=4)
    with freeze_time(time_400):
        async_fire_time_changed(hass, time_400)
        await hass.async_block_till_done(wait_background_tasks=True)
        assert hass.states.get('sensor.heatpump_compressor_today').state == '0.5'
        sensor_today2_val = float(hass.states.get('sensor.heatpump_compressor_today2').state)
        assert 0.499 < sensor_today2_val < 0.501
        hass.states.async_set('binary_sensor.heatpump_compressor_state', 'on')
        await async_wait_recording_done(hass)
    time_600 = start_of_today + timedelta(hours=6)
    with freeze_time(time_600):
        async_fire_time_changed(hass, time_600)
        await hass.async_block_till_done(wait_background_tasks=True)
        assert hass.states.get('sensor.heatpump_compressor_today').state == '2.5'
        sensor_today2_val = float(hass.states.get('sensor.heatpump_compressor_today2').state)
        assert 2.499 < sensor_today2_val < 2.501
    rolled_to_next_day = start_of_today + timedelta(days=1)
    assert rolled_to_next_day.hour == 0
    assert rolled_to_next_day.minute == 0
    assert rolled_to_next_day.second == 0
    assert rolled_to_next_day.microsecond == 0
    with freeze_time(rolled_to_next_day):
        async_fire_time_changed(hass, rolled_to_next_day)
        await hass.async_block_till_done(wait_background_tasks=True)
        assert hass.states.get('sensor.heatpump_compressor_today').state == '0.0'
        assert hass.states.get('sensor.heatpump_compressor_today2').state == '0.0'
    rolled_to_next_day_plus_12 = start_of_today + timedelta(days=1, hours=12, microseconds=0)
    with freeze_time(rolled_to_next_day_plus_12):
        async_fire_time_changed(hass, rolled_to_next_day_plus_12)
        await hass.async_block_till_done(wait_background_tasks=True)
        assert hass.states.get('sensor.heatpump_compressor_today').state == '12.0'
        assert hass.states.get('sensor.heatpump_compressor_today2').state == '12.0'
    rolled_to_next_day_plus_14 = start_of_today + timedelta(days=1, hours=14, microseconds=0)
    with freeze_time(rolled_to_next_day_plus_14):
        async_fire_time_changed(hass, rolled_to_next_day_plus_14)
        await hass.async_block_till_done(wait_background_tasks=True)
        assert hass.states.get('sensor.heatpump_compressor_today').state == '14.0'
        assert hass.states.get('sensor.heatpump_compressor_today2').state == '14.0'
    rolled_to_next_day_plus_16_860000 = start_of_today + timedelta(days=1, hours=16, microseconds=860000)
    with freeze_time(rolled_to_next_day_plus_16_860000):
        hass.states.async_set('binary_sensor.heatpump_compressor_state', 'off')
        await async_wait_recording_done(hass)
        async_fire_time_changed(hass, rolled_to_next_day_plus_16_860000)
        await hass.async_block_till_done(wait_background_tasks=True)
    rolled_to_next_day_plus_18 = start_of_today + timedelta(days=1, hours=18)
    with freeze_time(rolled_to_next_day_plus_18):
        async_fire_time_changed(hass, rolled_to_next_day_plus_18)
        await hass.async_block_till_done(wait_background_tasks=True)
        assert hass.states.get('sensor.heatpump_compressor_today').state == '16.0'
        assert hass.states.get('sensor.heatpump_compressor_today2').state == '16.0002388888929'


async def test_device_classes(recorder_mock: Any, hass: HomeAssistant) -> None:
    await async_setup_component(hass, 'sensor', {
        'sensor': [
            {
                'platform': 'history_stats',
                'entity_id': 'binary_sensor.test_id',
                'name': 'time',
                'state': 'on',
                'start': '{{ as_timestamp(utcnow()) - 3600 }}',
                'end': '{{ as_timestamp(utcnow()) + 3600 }}',
                'type': 'time'
            },
            {
                'platform': 'history_stats',
                'entity_id': 'binary_sensor.test_id',
                'name': 'count',
                'state': 'on',
                'start': '{{ as_timestamp(utcnow()) - 3600 }}',
                'end': '{{ as_timestamp(utcnow()) + 3600 }}',
                'type': 'count'
            },
            {
                'platform': 'history_stats',
                'entity_id': 'binary_sensor.test_id',
                'name': 'ratio',
                'state': 'on',
                'start': '{{ as_timestamp(utcnow()) - 3600 }}',
                'end': '{{ as_timestamp(utcnow()) + 3600 }}',
                'type': 'ratio'
            }
        ]
    })
    await hass.async_block_till_done()
    assert hass.states.get('sensor.time').attributes[ATTR_DEVICE_CLASS] == 'duration'
    assert ATTR_DEVICE_CLASS not in hass.states.get('sensor.ratio').attributes
    assert ATTR_DEVICE_CLASS not in hass.states.get('sensor.count').attributes


async def test_history_stats_handles_floored_timestamps(recorder_mock: Any, hass: HomeAssistant) -> None:
    await hass.config.async_set_time_zone('UTC')
    utcnow = dt_util.utcnow()
    start_time = utcnow.replace(hour=0, minute=0, second=0, microsecond=0)
    last_times: Any = None

    def _fake_states(hass: HomeAssistant, start: datetime, end: datetime, *args: Any, **kwargs: Any) -> Dict[str, List[State]]:
        nonlocal last_times
        last_times = (start, end)
        return {
            'binary_sensor.state': [
                State('binary_sensor.state', 'on', last_changed=start_time, last_updated=start_time)
            ]
        }
    with patch('homeassistant.components.recorder.history.state_changes_during_period', _fake_states), freeze_time(start_time):
        await async_setup_component(hass, 'sensor', {
            'sensor': [
                {
                    'platform': 'history_stats',
                    'entity_id': 'binary_sensor.state',
                    'name': 'sensor1',
                    'state': 'on',
                    'start': '{{ utcnow().replace(hour=0, minute=0, second=0, microsecond=100) }}',
                    'duration': {'hours': 2},
                    'type': 'time'
                }
            ]
        })
        await hass.async_block_till_done()
        await async_update_entity(hass, 'sensor.sensor1')
        await hass.async_block_till_done()
    assert last_times == (start_time, start_time + timedelta(hours=2))


async def test_unique_id(recorder_mock: Any, hass: HomeAssistant, entity_registry: er.EntityRegistry) -> None:
    config = {
        'sensor': {
            'platform': 'history_stats',
            'entity_id': 'binary_sensor.test_id',
            'state': 'on',
            'start': '{{ utcnow() }}',
            'duration': '01:00',
            'name': 'Test',
            'unique_id': 'some_history_stats_unique_id'
        }
    }
    assert await async_setup_component(hass, 'sensor', config)
    await hass.async_block_till_done()
    assert entity_registry.async_get('sensor.test').unique_id == 'some_history_stats_unique_id'


async def test_device_id(recorder_mock: Any, hass: HomeAssistant, entity_registry: er.EntityRegistry, device_registry: dr.DeviceRegistry) -> None:
    source_config_entry = MockConfigEntry()
    source_config_entry.add_to_hass(hass)
    source_device_entry = device_registry.async_get_or_create(
        config_entry_id=source_config_entry.entry_id,
        identifiers={('sensor', 'identifier_test')},
        connections={('mac', '30:31:32:33:34:35')}
    )
    source_entity = entity_registry.async_get_or_create(
        'binary_sensor', 'test', 'source',
        config_entry=source_config_entry,
        device_id=source_device_entry.id
    )
    await hass.async_block_till_done()
    assert entity_registry.async_get('binary_sensor.test_source') is not None
    history_stats_config_entry = MockConfigEntry(
        data={},
        domain=DOMAIN,
        options={
            CONF_NAME: DEFAULT_NAME,
            CONF_ENTITY_ID: 'binary_sensor.test_source',
            CONF_STATE: ['on'],
            CONF_TYPE: 'count',
            CONF_START: '{{ as_timestamp(utcnow()) - 3600 }}',
            CONF_END: '{{ utcnow() }}'
        },
        title='History stats'
    )
    history_stats_config_entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(history_stats_config_entry.entry_id)
    await hass.async_block_till_done()
    history_stats_entity = entity_registry.async_get('sensor.history_stats')
    assert history_stats_entity is not None
    assert history_stats_entity.device_id == source_entity.device_id