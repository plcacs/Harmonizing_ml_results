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
    ...

def _time_weighted_average(fstates: Iterable[tuple[float, State]], start: datetime.datetime, end: datetime.datetime) -> float:
    ...

def _get_units(fstates: Iterable[tuple[float, State]]) -> set[str]:
    ...

def _equivalent_units(units: set[str]) -> bool:
    ...

def _entity_history_to_float_and_state(entity_history: Iterable[State]) -> list[tuple[float, State]]:
    ...

def _is_numeric(state: State) -> bool:
    ...

def _normalize_states(hass: HomeAssistant, old_metadatas: dict[str, tuple[StatisticData, StatisticMetaData]], fstates: list[tuple[float, State]], entity_id: str) -> tuple[str, list[tuple[float, State]]]:
    ...

def _suggest_report_issue(hass: HomeAssistant, entity_id: str) -> Any:
    ...

def warn_dip(hass: HomeAssistant, entity_id: str, state: State, previous_fstate: float) -> None:
    ...

def warn_negative(hass: HomeAssistant, entity_id: str, state: State) -> None:
    ...

def reset_detected(hass: HomeAssistant, entity_id: str, fstate: float, previous_fstate: float, state: State) -> bool:
    ...

def _wanted_statistics(sensor_states: list[State]) -> dict[str, set[str]]:
    ...

def _last_reset_as_utc_isoformat(last_reset_s: Any, entity_id: str) -> str:
    ...

def _timestamp_to_isoformat_or_none(timestamp: Any) -> str:
    ...

def compile_statistics(hass: HomeAssistant, session: Session, start: datetime.datetime, end: datetime.datetime) -> statistics.PlatformCompiledStatistics:
    ...

def list_statistic_ids(hass: HomeAssistant, statistic_ids: set[str] = None, statistic_type: str = None) -> dict[str, dict[str, Any]]:
    ...

@callback
def _update_issues(report_issue: Callable, sensor_states: list[State], metadatas: dict[str, tuple[StatisticData, StatisticMetaData]]) -> None:
    ...

def update_statistics_issues(hass: HomeAssistant, session: Session) -> None:
    ...

def validate_statistics(hass: HomeAssistant) -> dict[str, list[statistics.ValidationIssue]]:
    ...
