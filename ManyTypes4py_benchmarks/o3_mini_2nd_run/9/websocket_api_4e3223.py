"""The Recorder websocket API."""
from __future__ import annotations
import asyncio
from datetime import datetime as dt
from typing import Any, Dict, Optional, Set
import voluptuous as vol
from homeassistant.components import websocket_api
from homeassistant.components.websocket_api import messages
from homeassistant.core import HomeAssistant, callback, valid_entity_id
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.json import json_bytes
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import (
    AreaConverter,
    BloodGlucoseConcentrationConverter,
    ConductivityConverter,
    DataRateConverter,
    DistanceConverter,
    DurationConverter,
    ElectricCurrentConverter,
    ElectricPotentialConverter,
    EnergyConverter,
    EnergyDistanceConverter,
    InformationConverter,
    MassConverter,
    PowerConverter,
    PressureConverter,
    SpeedConverter,
    TemperatureConverter,
    UnitlessRatioConverter,
    VolumeConverter,
    VolumeFlowRateConverter,
)
from .models import StatisticPeriod
from .statistics import (
    STATISTIC_UNIT_TO_UNIT_CONVERTER,
    async_add_external_statistics,
    async_change_statistics_unit,
    async_import_statistics,
    async_list_statistic_ids,
    list_statistic_ids,
    statistic_during_period,
    statistics_during_period,
    update_statistics_issues,
    validate_statistics,
)
from .util import PERIOD_SCHEMA, get_instance, resolve_period

CLEAR_STATISTICS_TIME_OUT: int = 10
UPDATE_STATISTICS_METADATA_TIME_OUT: int = 10

UNIT_SCHEMA: vol.Schema = vol.Schema({
    vol.Optional("area"): vol.In(AreaConverter.VALID_UNITS),
    vol.Optional("blood_glucose_concentration"): vol.In(BloodGlucoseConcentrationConverter.VALID_UNITS),
    vol.Optional("conductivity"): vol.In(ConductivityConverter.VALID_UNITS),
    vol.Optional("data_rate"): vol.In(DataRateConverter.VALID_UNITS),
    vol.Optional("distance"): vol.In(DistanceConverter.VALID_UNITS),
    vol.Optional("duration"): vol.In(DurationConverter.VALID_UNITS),
    vol.Optional("electric_current"): vol.In(ElectricCurrentConverter.VALID_UNITS),
    vol.Optional("voltage"): vol.In(ElectricPotentialConverter.VALID_UNITS),
    vol.Optional("energy"): vol.In(EnergyConverter.VALID_UNITS),
    vol.Optional("energy_distance"): vol.In(EnergyDistanceConverter.VALID_UNITS),
    vol.Optional("information"): vol.In(InformationConverter.VALID_UNITS),
    vol.Optional("mass"): vol.In(MassConverter.VALID_UNITS),
    vol.Optional("power"): vol.In(PowerConverter.VALID_UNITS),
    vol.Optional("pressure"): vol.In(PressureConverter.VALID_UNITS),
    vol.Optional("speed"): vol.In(SpeedConverter.VALID_UNITS),
    vol.Optional("temperature"): vol.In(TemperatureConverter.VALID_UNITS),
    vol.Optional("unitless"): vol.In(UnitlessRatioConverter.VALID_UNITS),
    vol.Optional("volume"): vol.In(VolumeConverter.VALID_UNITS),
    vol.Optional("volume_flow_rate"): vol.In(VolumeFlowRateConverter.VALID_UNITS),
})

@callback
def async_setup(hass: HomeAssistant) -> None:
    """Set up the recorder websocket API."""
    websocket_api.async_register_command(hass, ws_adjust_sum_statistics)
    websocket_api.async_register_command(hass, ws_change_statistics_unit)
    websocket_api.async_register_command(hass, ws_clear_statistics)
    websocket_api.async_register_command(hass, ws_get_statistic_during_period)
    websocket_api.async_register_command(hass, ws_get_statistics_during_period)
    websocket_api.async_register_command(hass, ws_get_statistics_metadata)
    websocket_api.async_register_command(hass, ws_list_statistic_ids)
    websocket_api.async_register_command(hass, ws_import_statistics)
    websocket_api.async_register_command(hass, ws_update_statistics_issues)
    websocket_api.async_register_command(hass, ws_update_statistics_metadata)
    websocket_api.async_register_command(hass, ws_validate_statistics)

def _ws_get_statistic_during_period(
    hass: HomeAssistant,
    msg_id: int,
    start_time: dt,
    end_time: Optional[dt],
    statistic_id: str,
    types: Optional[Set[str]],
    units: Optional[Dict[str, Any]],
) -> bytes:
    """Fetch statistics and convert them to json in the executor."""
    result = statistic_during_period(hass, start_time, end_time, statistic_id, types, units=units)
    return json_bytes(messages.result_message(msg_id, result))

@websocket_api.websocket_command({
    vol.Required("type"): "recorder/statistic_during_period",
    vol.Required("statistic_id"): str,
    vol.Optional("types"): vol.All([vol.Any("max", "mean", "min", "change")], vol.Coerce(set)),
    vol.Optional("units"): UNIT_SCHEMA,
    **PERIOD_SCHEMA.schema
})
@websocket_api.async_response
async def ws_get_statistic_during_period(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: Dict[str, Any]
) -> None:
    """Handle statistics websocket command."""
    if ("start_time" in msg or "end_time" in msg) and "duration" in msg:
        raise HomeAssistantError
    if "offset" in msg and "duration" not in msg:
        raise HomeAssistantError
    start_time, end_time = resolve_period(cast(StatisticPeriod, msg))
    result_bytes = await get_instance(hass).async_add_executor_job(
        _ws_get_statistic_during_period,
        hass,
        msg["id"],
        start_time,
        end_time,
        msg["statistic_id"],
        msg.get("types"),
        msg.get("units")
    )
    connection.send_message(result_bytes)

def _ws_get_statistics_during_period(
    hass: HomeAssistant,
    msg_id: int,
    start_time: dt,
    end_time: Optional[dt],
    statistic_ids: Set[str],
    period: str,
    units: Optional[Dict[str, Any]],
    types: Set[str],
) -> bytes:
    """Fetch statistics and convert them to json in the executor."""
    result: Dict[str, Any] = statistics_during_period(hass, start_time, end_time, statistic_ids, period, units, types)
    include_last_reset: bool = "last_reset" in types
    for statistic_rows in result.values():
        for row in statistic_rows:
            row["start"] = int(row["start"] * 1000)
            row["end"] = int(row["end"] * 1000)
            if include_last_reset and (last_reset := row.get("last_reset")) is not None:
                row["last_reset"] = int(last_reset * 1000)
    return json_bytes(messages.result_message(msg_id, result))

async def ws_handle_get_statistics_during_period(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: Dict[str, Any]
) -> None:
    """Handle statistics websocket command."""
    start_time_str: str = msg["start_time"]
    end_time_str: Optional[str] = msg.get("end_time")
    start_time = dt_util.parse_datetime(start_time_str)
    if start_time:
        start_time = dt_util.as_utc(start_time)
    else:
        connection.send_error(msg["id"], "invalid_start_time", "Invalid start_time")
        return
    if end_time_str:
        end_time = dt_util.parse_datetime(end_time_str)
        if end_time:
            end_time = dt_util.as_utc(end_time)
        else:
            connection.send_error(msg["id"], "invalid_end_time", "Invalid end_time")
            return
    else:
        end_time = None
    types: Optional[Set[str]] = msg.get("types")
    if types is None:
        types = {"change", "last_reset", "max", "mean", "min", "state", "sum"}
    result_bytes = await get_instance(hass).async_add_executor_job(
        _ws_get_statistics_during_period,
        hass,
        msg["id"],
        start_time,
        end_time,
        set(msg["statistic_ids"]),
        msg.get("period"),
        msg.get("units"),
        types,
    )
    connection.send_message(result_bytes)

@websocket_api.websocket_command({
    vol.Required("type"): "recorder/statistics_during_period",
    vol.Required("start_time"): str,
    vol.Optional("end_time"): str,
    vol.Required("statistic_ids"): vol.All([str], vol.Length(min=1)),
    vol.Required("period"): vol.Any("5minute", "hour", "day", "week", "month"),
    vol.Optional("units"): UNIT_SCHEMA,
    vol.Optional("types"): vol.All([vol.Any("change", "last_reset", "max", "mean", "min", "state", "sum")], vol.Coerce(set))
})
@websocket_api.async_response
async def ws_get_statistics_during_period(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: Dict[str, Any]
) -> None:
    """Handle statistics websocket command."""
    await ws_handle_get_statistics_during_period(hass, connection, msg)

def _ws_get_list_statistic_ids(
    hass: HomeAssistant,
    msg_id: int,
    statistic_type: Optional[str] = None
) -> bytes:
    """Fetch a list of available statistic_id and convert them to JSON.
    
    Runs in the executor.
    """
    result = list_statistic_ids(hass, None, statistic_type)
    return json_bytes(messages.result_message(msg_id, result))

async def ws_handle_list_statistic_ids(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: Dict[str, Any]
) -> None:
    """Fetch a list of available statistic_id."""
    result_bytes = await get_instance(hass).async_add_executor_job(
        _ws_get_list_statistic_ids,
        hass,
        msg["id"],
        msg.get("statistic_type")
    )
    connection.send_message(result_bytes)

@websocket_api.websocket_command({
    vol.Required("type"): "recorder/list_statistic_ids",
    vol.Optional("statistic_type"): vol.Any("sum", "mean")
})
@websocket_api.async_response
async def ws_list_statistic_ids(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: Dict[str, Any]
) -> None:
    """Fetch a list of available statistic_id."""
    await ws_handle_list_statistic_ids(hass, connection, msg)

@websocket_api.websocket_command({vol.Required("type"): "recorder/validate_statistics"})
@websocket_api.async_response
async def ws_validate_statistics(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: Dict[str, Any]
) -> None:
    """Fetch a list of available statistic_id."""
    instance = get_instance(hass)
    statistic_ids = await instance.async_add_executor_job(validate_statistics, hass)
    connection.send_result(msg["id"], statistic_ids)

@websocket_api.websocket_command({vol.Required("type"): "recorder/update_statistics_issues"})
@websocket_api.async_response
async def ws_update_statistics_issues(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: Dict[str, Any]
) -> None:
    """Update statistics issues."""
    instance = get_instance(hass)
    await instance.async_add_executor_job(update_statistics_issues, hass)
    connection.send_result(msg["id"])

@websocket_api.require_admin
@websocket_api.websocket_command({
    vol.Required("type"): "recorder/clear_statistics",
    vol.Required("statistic_ids"): [str]
})
@websocket_api.async_response
async def ws_clear_statistics(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: Dict[str, Any]
) -> None:
    """Clear statistics for a list of statistic_ids.

    Note: The WS call posts a job to the recorder's queue and then returns, it doesn't
    wait until the job is completed.
    """
    done_event: asyncio.Event = asyncio.Event()

    def clear_statistics_done() -> None:
        hass.loop.call_soon_threadsafe(done_event.set)

    get_instance(hass).async_clear_statistics(msg["statistic_ids"], on_done=clear_statistics_done)
    try:
        async with asyncio.timeout(CLEAR_STATISTICS_TIME_OUT):
            await done_event.wait()
    except TimeoutError:
        connection.send_error(msg["id"], websocket_api.ERR_TIMEOUT, "clear_statistics timed out")
        return
    connection.send_result(msg["id"])

@websocket_api.websocket_command({
    vol.Required("type"): "recorder/get_statistics_metadata",
    vol.Optional("statistic_ids"): [str]
})
@websocket_api.async_response
async def ws_get_statistics_metadata(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: Dict[str, Any]
) -> None:
    """Get metadata for a list of statistic_ids."""
    statistic_ids: Optional[list[str]] = msg.get("statistic_ids")
    statistic_ids_set_or_none: Optional[Set[str]] = set(statistic_ids) if statistic_ids else None
    metadata = await async_list_statistic_ids(hass, statistic_ids_set_or_none)
    connection.send_result(msg["id"], metadata)

@websocket_api.require_admin
@websocket_api.websocket_command({
    vol.Required("type"): "recorder/update_statistics_metadata",
    vol.Required("statistic_id"): str,
    vol.Required("unit_of_measurement"): vol.Any(str, None)
})
@websocket_api.async_response
async def ws_update_statistics_metadata(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: Dict[str, Any]
) -> None:
    """Update statistics metadata for a statistic_id.

    Only the normalized unit of measurement can be updated.
    """
    done_event: asyncio.Event = asyncio.Event()

    def update_statistics_metadata_done() -> None:
        hass.loop.call_soon_threadsafe(done_event.set)

    get_instance(hass).async_update_statistics_metadata(
        msg["statistic_id"],
        new_unit_of_measurement=msg["unit_of_measurement"],
        on_done=update_statistics_metadata_done
    )
    try:
        async with asyncio.timeout(UPDATE_STATISTICS_METADATA_TIME_OUT):
            await done_event.wait()
    except TimeoutError:
        connection.send_error(msg["id"], websocket_api.ERR_TIMEOUT, "update_statistics_metadata timed out")
        return
    connection.send_result(msg["id"])

@websocket_api.require_admin
@websocket_api.websocket_command({
    vol.Required("type"): "recorder/change_statistics_unit",
    vol.Required("statistic_id"): str,
    vol.Required("new_unit_of_measurement"): vol.Any(str, None),
    vol.Required("old_unit_of_measurement"): vol.Any(str, None)
})
@callback
def ws_change_statistics_unit(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: Dict[str, Any]
) -> None:
    """Change the unit_of_measurement for a statistic_id.

    All existing statistics will be converted to the new unit.
    """
    async_change_statistics_unit(
        hass,
        msg["statistic_id"],
        new_unit_of_measurement=msg["new_unit_of_measurement"],
        old_unit_of_measurement=msg["old_unit_of_measurement"]
    )
    connection.send_result(msg["id"])

@websocket_api.require_admin
@websocket_api.websocket_command({
    vol.Required("type"): "recorder/adjust_sum_statistics",
    vol.Required("statistic_id"): str,
    vol.Required("start_time"): str,
    vol.Required("adjustment"): vol.Any(float, int),
    vol.Required("adjustment_unit_of_measurement"): vol.Any(str, None)
})
@websocket_api.async_response
async def ws_adjust_sum_statistics(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: Dict[str, Any]
) -> None:
    """Adjust sum statistics.

    If the statistics is stored as NORMALIZED_UNIT,
    it's allowed to make an adjustment in VALID_UNIT
    """
    start_time_str: str = msg["start_time"]
    start_time: Optional[dt] = dt_util.parse_datetime(start_time_str)
    if start_time:
        start_time = dt_util.as_utc(start_time)
    else:
        connection.send_error(msg["id"], "invalid_start_time", "Invalid start time")
        return
    instance = get_instance(hass)
    metadatas = await instance.async_add_executor_job(list_statistic_ids, hass, {msg["statistic_id"]})
    if not metadatas:
        connection.send_error(msg["id"], "unknown_statistic_id", "Unknown statistic ID")
        return
    metadata: Dict[str, Any] = metadatas[0]

    def valid_units(statistics_unit: Optional[str], adjustment_unit: Optional[str]) -> bool:
        if statistics_unit == adjustment_unit:
            return True
        converter = STATISTIC_UNIT_TO_UNIT_CONVERTER.get(statistics_unit)
        if converter is not None and adjustment_unit in converter.VALID_UNITS:
            return True
        return False

    stat_unit: Optional[str] = metadata.get("statistics_unit_of_measurement")
    adjustment_unit: Optional[str] = msg["adjustment_unit_of_measurement"]
    if not valid_units(stat_unit, adjustment_unit):
        connection.send_error(msg["id"], "invalid_units", f"Can't convert {stat_unit} to {adjustment_unit}")
        return
    get_instance(hass).async_adjust_statistics(
        msg["statistic_id"],
        start_time,
        msg["adjustment"],
        adjustment_unit
    )
    connection.send_result(msg["id"])

@websocket_api.require_admin
@websocket_api.websocket_command({
    vol.Required("type"): "recorder/import_statistics",
    vol.Required("metadata"): {
        vol.Required("has_mean"): bool,
        vol.Required("has_sum"): bool,
        vol.Required("name"): vol.Any(str, None),
        vol.Required("source"): str,
        vol.Required("statistic_id"): str,
        vol.Required("unit_of_measurement"): vol.Any(str, None)
    },
    vol.Required("stats"): [{
        vol.Required("start"): cv.datetime,
        vol.Optional("mean"): vol.Any(float, int),
        vol.Optional("min"): vol.Any(float, int),
        vol.Optional("max"): vol.Any(float, int),
        vol.Optional("last_reset"): vol.Any(cv.datetime, None),
        vol.Optional("state"): vol.Any(float, int),
        vol.Optional("sum"): vol.Any(float, int)
    }]
})
@callback
def ws_import_statistics(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: Dict[str, Any]
) -> None:
    """Import statistics."""
    metadata: Dict[str, Any] = msg["metadata"]
    stats: list[Dict[str, Any]] = msg["stats"]
    if valid_entity_id(metadata["statistic_id"]):
        async_import_statistics(hass, metadata, stats)
    else:
        async_add_external_statistics(hass, metadata, stats)
    connection.send_result(msg["id"])