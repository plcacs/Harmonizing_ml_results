from __future__ import annotations
from collections import defaultdict
from collections.abc import Callable, Iterable
from contextlib import suppress
import datetime
import itertools
import logging
import math
from typing import Any
from sqlalchemy.orm.session import Session
from homeassistant.components.recorder import DOMAIN as RECORDER_DOMAIN, get_instance, history, statistics
from homeassistant.components.recorder.models import StatisticData, StatisticMetaData, StatisticResult
from homeassistant.const import ATTR_UNIT_OF_MEASUREMENT, REVOLUTIONS_PER_MINUTE, UnitOfIrradiance, UnitOfSoundPressure, UnitOfVolume
from homeassistant.core import HomeAssistant, State, callback, split_entity_id
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers.entity import entity_sources
from homeassistant.helpers.typing import UNDEFINED, UndefinedType
from homeassistant.loader import async_suggest_report_issue
from homeassistant.util import dt as dt_util
from homeassistant.util.async_ import run_callback_threadsafe
from homeassistant.util.enum import try_parse_enum
from homeassistant.util.hass_dict import HassKey
from .const import ATTR_LAST_RESET, ATTR_STATE_CLASS, DOMAIN, SensorStateClass, UnitOfVolumeFlowRate

_LOGGER: logging.Logger = logging.getLogger(__name__)
DEFAULT_STATISTICS: dict[SensorStateClass, set[str]] = {SensorStateClass.MEASUREMENT: {'mean', 'min', 'max'}, SensorStateClass.TOTAL: {'sum'}, SensorStateClass.TOTAL_INCREASING: {'sum'}}
EQUIVALENT_UNITS: dict[str, Any] = {'BTU/(h×ft²)': UnitOfIrradiance.BTUS_PER_HOUR_SQUARE_FOOT, 'dBa': UnitOfSoundPressure.WEIGHTED_DECIBEL_A, 'RPM': REVOLUTIONS_PER_MINUTE, 'ft3': UnitOfVolume.CUBIC_FEET, 'm3': UnitOfVolume.CUBIC_METERS, 'ft³/m': UnitOfVolumeFlowRate.CUBIC_FEET_PER_MINUTE}
SEEN_DIP: HassKey = HassKey(f'{DOMAIN}_seen_total_increasing_dip')
WARN_DIP: HassKey = HassKey(f'{DOMAIN}_warn_total_increasing_dip')
WARN_NEGATIVE: HassKey = HassKey(f'{DOMAIN}_warn_total_increasing_negative')
WARN_UNSUPPORTED_UNIT: HassKey = HassKey(f'{DOMAIN}_warn_unsupported_unit')
WARN_UNSTABLE_UNIT: HassKey = HassKey(f'{DOMAIN}_warn_unstable_unit')
LINK_DEV_STATISTICS: str = 'https://my.home-assistant.io/redirect/developer_statistics'

def _get_sensor_states(hass: HomeAssistant) -> list[State]:
    """Get the current state of all sensors for which to compile statistics."""
    instance = get_instance(hass)
    entity_filter = instance.entity_filter
    return [state for state in hass.states.all(DOMAIN) if (state_class := state.attributes.get(ATTR_STATE_CLASS)) and (type(state_class) is SensorStateClass or try_parse_enum(SensorStateClass, state_class)) and (not entity_filter or entity_filter(state.entity_id))]

def _time_weighted_average(fstates: list[tuple[float, State]], start: datetime.datetime, end: datetime.datetime) -> float:
    """Calculate a time weighted average."""
    old_fstate: float = None
    old_start_time: datetime.datetime = None
    accumulated: float = 0.0
    for fstate, state in fstates:
        start_time = max(state.last_updated, start)
        if old_start_time is None:
            start = start_time
        else:
            duration = start_time - old_start_time
            assert old_fstate is not None
            accumulated += old_fstate * duration.total_seconds()
        old_fstate = fstate
        old_start_time = start_time
    if old_fstate is not None:
        assert old_start_time is not None
        duration = end - old_start_time
        accumulated += old_fstate * duration.total_seconds()
    period_seconds = (end - start).total_seconds()
    if period_seconds == 0:
        return 0.0
    return accumulated / period_seconds

def _get_units(fstates: list[tuple[float, State]]) -> set[str]:
    """Return a set of all units."""
    return {item[1].attributes.get(ATTR_UNIT_OF_MEASUREMENT) for item in fstates}

def _equivalent_units(units: set[str]) -> bool:
    """Return True if the units are equivalent."""
    if len(units) == 1:
        return True
    units = {EQUIVALENT_UNITS[unit] if unit in EQUIVALENT_UNITS else unit for unit in units}
    return len(units) == 1

def _entity_history_to_float_and_state(entity_history: list[State]) -> list[tuple[float, State]]:
    """Return a list of (float, state) tuples for the given entity."""
    float_states = []
    append = float_states.append
    isfinite = math.isfinite
    for state in entity_history:
        try:
            if (float_state := float(state.state)) is not None and isfinite(float_state):
                append((float_state, state))
        except (ValueError, TypeError):
            pass
    return float_states

def _is_numeric(state: State) -> bool:
    """Return if the state is numeric."""
    with suppress(ValueError, TypeError):
        if (num_state := float(state.state)) is not None and math.isfinite(num_state):
            return True
    return False

def _normalize_states(hass: HomeAssistant, old_metadatas: dict[str, tuple[StatisticData, StatisticMetaData]], fstates: list[tuple[float, State]], entity_id: str) -> tuple[str, list[tuple[float, State]]]:
    """Normalize units."""
    state_unit: str = None
    state_unit = fstates[0][1].attributes.get(ATTR_UNIT_OF_MEASUREMENT)
    old_metadata = old_metadatas[entity_id][1] if entity_id in old_metadatas else None
    if not old_metadata:
        statistics_unit = state_unit
    else:
        statistics_unit = old_metadata['unit_of_measurement']
    if statistics_unit not in statistics.STATISTIC_UNIT_TO_UNIT_CONVERTER:
        all_units = _get_units(fstates)
        if not _equivalent_units(all_units):
            if WARN_UNSTABLE_UNIT not in hass.data:
                hass.data[WARN_UNSTABLE_UNIT] = set()
            if entity_id not in hass.data[WARN_UNSTABLE_UNIT]:
                hass.data[WARN_UNSTABLE_UNIT].add(entity_id)
                extra = ''
                if old_metadata:
                    extra = f' and matches the unit of already compiled statistics ({old_metadata['unit_of_measurement']})'
                _LOGGER.warning('The unit of %s is changing, got multiple %s, generation of long term statistics will be suppressed unless the unit is stable%s. Go to %s to fix this', entity_id, all_units, extra, LINK_DEV_STATISTICS)
            return (None, [])
        return (state_unit, fstates)
    converter = statistics.STATISTIC_UNIT_TO_UNIT_CONVERTER[statistics_unit]
    valid_fstates = []
    convert = None
    last_unit = UNDEFINED
    valid_units = converter.VALID_UNITS
    for fstate, state in fstates:
        state_unit = state.attributes.get(ATTR_UNIT_OF_MEASUREMENT)
        if state_unit not in valid_units:
            if WARN_UNSUPPORTED_UNIT not in hass.data:
                hass.data[WARN_UNSUPPORTED_UNIT] = set()
            if entity_id not in hass.data[WARN_UNSUPPORTED_UNIT]:
                hass.data[WARN_UNSUPPORTED_UNIT].add(entity_id)
                _LOGGER.warning('The unit of %s (%s) cannot be converted to the unit of previously compiled statistics (%s). Generation of long term statistics will be suppressed unless the unit changes back to %s or a compatible unit. Go to %s to fix this', entity_id, state_unit, statistics_unit, statistics_unit, LINK_DEV_STATISTICS)
            continue
        if state_unit != last_unit:
            if state_unit == statistics_unit:
                convert = None
            else:
                convert = converter.converter_factory(state_unit, statistics_unit)
            last_unit = state_unit
        if convert is not None:
            fstate = convert(fstate)
        valid_fstates.append((fstate, state))
    return (statistics_unit, valid_fstates)

def _suggest_report_issue(hass: HomeAssistant, entity_id: str) -> Callable:
    """Suggest to report an issue."""
    entity_info = entity_sources(hass).get(entity_id)
    return async_suggest_report_issue(hass, integration_domain=entity_info['domain'] if entity_info else None)

def warn_dip(hass: HomeAssistant, entity_id: str, state: State, previous_fstate: float) -> None:
    """Log a warning once if a sensor with state_class_total has a decreasing value."""
    if SEEN_DIP not in hass.data:
        hass.data[SEEN_DIP] = set()
    if entity_id not in hass.data[SEEN_DIP]:
        hass.data[SEEN_DIP].add(entity_id)
        return
    if WARN_DIP not in hass.data:
        hass.data[WARN_DIP] = set()
    if entity_id not in hass.data[WARN_DIP]:
        hass.data[WARN_DIP].add(entity_id)
        entity_info = entity_sources(hass).get(entity_id)
        domain = entity_info['domain'] if entity_info else None
        if domain in ['energy', 'growatt_server', 'solaredge']:
            return
        _LOGGER.warning('Entity %s %shas state class total_increasing, but its state is not strictly increasing. Triggered by state %s (%s) with last_updated set to %s. Please %s', entity_id, f'from integration {domain} ' if domain else '', state.state, previous_fstate, state.last_updated.isoformat(), _suggest_report_issue(hass, entity_id))

def warn_negative(hass: HomeAssistant, entity_id: str, state: State) -> None:
    """Log a warning once if a sensor with state_class_total has a negative value."""
    if WARN_NEGATIVE not in hass.data:
        hass.data[WARN_NEGATIVE] = set()
    if entity_id not in hass.data[WARN_NEGATIVE]:
        hass.data[WARN_NEGATIVE].add(entity_id)
        entity_info = entity_sources(hass).get(entity_id)
        domain = entity_info['domain'] if entity_info else None
        _LOGGER.warning('Entity %s %shas state class total_increasing, but its state is negative. Triggered by state %s with last_updated set to %s. Please %s', entity_id, f'from integration {domain} ' if domain else '', state.state, state.last_updated.isoformat(), _suggest_report_issue(hass, entity_id))

def reset_detected(hass: HomeAssistant, entity_id: str, fstate: float, previous_fstate: float, state: State) -> bool:
    """Test if a total_increasing sensor has been reset."""
    if previous_fstate is None:
        return False
    if 0.9 * previous_fstate <= fstate < previous_fstate:
        warn_dip(hass, entity_id, state, previous_fstate)
    if fstate < 0:
        warn_negative(hass, entity_id, state)
        raise HomeAssistantError
    return fstate < 0.9 * previous_fstate

def _wanted_statistics(sensor_states: list[State]) -> dict[str, set[str]]:
    """Prepare a dict with wanted statistics for entities."""
    return {state.entity_id: DEFAULT_STATISTICS[state.attributes[ATTR_STATE_CLASS]] for state in sensor_states}

def _last_reset_as_utc_isoformat(last_reset_s: Any, entity_id: str) -> str:
    """Parse last_reset and convert it to UTC."""
    if last_reset_s is None:
        return None
    if isinstance(last_reset_s, str):
        last_reset = dt_util.parse_datetime(last_reset_s)
    else:
        last_reset = None
    if last_reset is None:
        _LOGGER.warning("Ignoring invalid last reset '%s' for %s", last_reset_s, entity_id)
        return None
    return dt_util.as_utc(last_reset).isoformat()

def _timestamp_to_isoformat_or_none(timestamp: Any) -> str:
    """Convert a timestamp to ISO format or return None."""
    if timestamp is None:
        return None
    return dt_util.utc_from_timestamp(timestamp).isoformat()

def compile_statistics(hass: HomeAssistant, session: Session, start: datetime.datetime, end: datetime.datetime) -> statistics.PlatformCompiledStatistics:
    """Compile statistics for all entities during start-end."""
    result = []
    sensor_states = _get_sensor_states(hass)
    wanted_statistics = _wanted_statistics(sensor_states)
    entities_full_history = [i.entity_id for i in sensor_states if 'sum' in wanted_statistics[i.entity_id]]
    history_list = {}
    if entities_full_history:
        history_list = history.get_full_significant_states_with_session(hass, session, start - datetime.timedelta.resolution, end, entity_ids=entities_full_history, significant_changes_only=False)
    entities_significant_history = [i.entity_id for i in sensor_states if 'sum' not in wanted_statistics[i.entity_id]]
    if entities_significant_history:
        _history_list = history.get_full_significant_states_with_session(hass, session, start - datetime.timedelta.resolution, end, entity_ids=entities_significant_history)
        history_list = {**history_list, **_history_list}
    entities_with_float_states = {}
    for _state in sensor_states:
        entity_id = _state.entity_id
        if not (entity_history := history_list.get(entity_id, [_state])):
            continue
        if not (float_states := _entity_history_to_float_and_state(entity_history)):
            continue
        entities_with_float_states[entity_id] = float_states
    old_metadatas = statistics.get_metadata_with_session(get_instance(hass), session, statistic_ids=set(entities_with_float_states))
    to_process = []
    to_query = set()
    for _state in sensor_states:
        entity_id = _state.entity_id
        if not (maybe_float_states := entities_with_float_states.get(entity_id)):
            continue
        statistics_unit, valid_float_states = _normalize_states(hass, old_metadatas, maybe_float_states, entity_id)
        if not valid_float_states:
            continue
        state_class = _state.attributes[ATTR_STATE_CLASS]
        to_process.append((entity_id, statistics_unit, state_class, valid_float_states))
        if 'sum' in wanted_statistics[entity_id]:
            to_query.add(entity_id)
    last_stats = statistics.get_latest_short_term_statistics_with_session(hass, session, to_query, {'last_reset', 'state', 'sum'}, metadata=old_metadatas)
    for entity_id, statistics_unit, state_class, valid_float_states in to_process:
        if (old_metadata := old_metadatas.get(entity_id)):
            if not _equivalent_units({old_metadata[1]['unit_of_measurement'], statistics_unit}):
                if WARN_UNSTABLE_UNIT not in hass.data:
                    hass.data[WARN_UNSTABLE_UNIT] = set()
                if entity_id not in hass.data[WARN_UNSTABLE_UNIT]:
                    hass.data[WARN_UNSTABLE_UNIT].add(entity_id)
                    _LOGGER.warning('The unit of %s (%s) cannot be converted to the unit of previously compiled statistics (%s). Generation of long term statistics will be suppressed unless the unit changes back to %s or a compatible unit. Go to %s to fix this', entity_id, statistics_unit, old_metadata[1]['unit_of_measurement'], old_metadata[1]['unit_of_measurement'], LINK_DEV_STATISTICS)
                continue
        meta = {'has_mean': 'mean' in wanted_statistics[entity_id], 'has_sum': 'sum' in wanted_statistics[entity_id], 'name': None, 'source': RECORDER_DOMAIN, 'statistic_id': entity_id, 'unit_of_measurement': statistics_unit}
        stat = {'start': start}
        if 'max' in wanted_statistics[entity_id]:
            stat['max'] = max(*itertools.islice(zip(*valid_float_states, strict=False), 1))
        if 'min' in wanted_statistics[entity_id]:
            stat['min'] = min(*itertools.islice(zip(*valid_float_states, strict=False), 1))
        if 'mean' in wanted_statistics[entity_id]:
            stat['mean'] = _time_weighted_average(valid_float_states, start, end)
        if 'sum' in wanted_statistics[entity_id]:
            last_reset = old_last_reset = None
            new_state = old_state = None
            _sum = 0.0
            if entity_id in last_stats:
                last_stat = last_stats[entity_id][0]
                last_reset = _timestamp_to_isoformat_or_none(last_stat['last_reset'])
                old_last_reset = last_reset
                new_state = old_state = last_stat.get('state')
                _sum = last_stat.get('sum') or 0.0
            for fstate, state in valid_float_states:
                reset = False
                if state_class != SensorStateClass.TOTAL_INCREASING and (last_reset := _last_reset_as_utc_isoformat(state.attributes.get('last_reset'), entity_id)) != old_last_reset and (last_reset is not None):
                    if old_state is None:
                        _LOGGER.info('Compiling initial sum statistics for %s, zero point set to %s', entity_id, fstate)
                    else:
                        _LOGGER.info('Detected new cycle for %s, last_reset set to %s (old last_reset %s)', entity_id, last_reset, old_last_reset)
                    reset = True
                elif old_state is None and last_reset is None:
                    reset = True
                    _LOGGER.info('Compiling initial sum statistics for %s, zero point set to %s', entity_id, fstate)
                elif state_class == SensorStateClass.TOTAL_INCREASING:
                    try:
                        if old_state is None or reset_detected(hass, entity_id, fstate, new_state, state):
                            reset = True
                            _LOGGER.info('Detected new cycle for %s, value dropped from %s to %s, triggered by state with last_updated set to %s', entity_id, new_state, fstate, state.last_updated.isoformat())
                    except HomeAssistantError:
                        continue
                if reset:
                    if old_state is not None and new_state is not None:
                        _sum += new_state - old_state
                    new_state = fstate
                    old_last_reset = last_reset
                    if old_state is not None:
                        old_state = 0.0
                    else:
                        old_state = new_state
                else:
                    new_state = fstate
            if new_state is None or old_state is None:
                continue
            _sum += new_state - old_state
            if last_reset is not None:
                stat['last_reset'] = dt_util.parse_datetime(last_reset)
            stat['sum'] = _sum
            stat['state'] = new_state
        result.append({'meta': meta, 'stat': stat})
    return statistics.PlatformCompiledStatistics(result, old_metadatas)

def list_statistic_ids(hass: HomeAssistant