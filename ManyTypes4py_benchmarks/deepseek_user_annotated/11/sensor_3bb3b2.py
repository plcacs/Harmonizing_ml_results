"""Support for Dutch Smart Meter (also known as Smartmeter or P1 port)."""

from __future__ import annotations

import asyncio
from asyncio import CancelledError
from collections.abc import Callable, Generator
from contextlib import suppress
from dataclasses import dataclass
from datetime import timedelta
from enum import IntEnum
from functools import partial
from typing import Any, Optional, Union

from dsmr_parser.clients.protocol import create_dsmr_reader, create_tcp_dsmr_reader
from dsmr_parser.clients.rfxtrx_protocol import (
    create_rfxtrx_dsmr_reader,
    create_rfxtrx_tcp_dsmr_reader,
)
from dsmr_parser.objects import DSMRObject, MbusDevice, Telegram
import serial

from homeassistant.components.sensor import (
    DOMAIN as SENSOR_DOMAIN,
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    CONF_HOST,
    CONF_PORT,
    CONF_PROTOCOL,
    EVENT_HOMEASSISTANT_STOP,
    EntityCategory,
    UnitOfEnergy,
    UnitOfVolume,
)
from homeassistant.core import CoreState, Event, HomeAssistant, callback
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.dispatcher import (
    async_dispatcher_connect,
    async_dispatcher_send,
)
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import StateType
from homeassistant.util import Throttle

from . import DsmrConfigEntry
from .const import (
    CONF_DSMR_VERSION,
    CONF_SERIAL_ID,
    CONF_SERIAL_ID_GAS,
    CONF_TIME_BETWEEN_UPDATE,
    DEFAULT_PRECISION,
    DEFAULT_RECONNECT_INTERVAL,
    DEFAULT_TIME_BETWEEN_UPDATE,
    DEVICE_NAME_ELECTRICITY,
    DEVICE_NAME_GAS,
    DEVICE_NAME_HEAT,
    DEVICE_NAME_WATER,
    DOMAIN,
    DSMR_PROTOCOL,
    LOGGER,
)

EVENT_FIRST_TELEGRAM = "dsmr_first_telegram_{}"

UNIT_CONVERSION = {"m3": UnitOfVolume.CUBIC_METERS}


@dataclass(frozen=True, kw_only=True)
class DSMRSensorEntityDescription(SensorEntityDescription):
    """Represents an DSMR Sensor."""

    dsmr_versions: set[str] | None = None
    is_gas: bool = False
    is_water: bool = False
    is_heat: bool = False
    obis_reference: str


class MbusDeviceType(IntEnum):
    """Types of mbus devices (13757-3:2013)."""

    GAS = 3
    HEAT = 4
    WATER = 7


SENSORS: tuple[DSMRSensorEntityDescription, ...] = (
    DSMRSensorEntityDescription(
        key="timestamp",
        obis_reference="P1_MESSAGE_TIMESTAMP",
        device_class=SensorDeviceClass.TIMESTAMP,
        entity_category=EntityCategory.DIAGNOSTIC,
        entity_registry_enabled_default=False,
    ),
    DSMRSensorEntityDescription(
        key="current_electricity_usage",
        translation_key="current_electricity_usage",
        obis_reference="CURRENT_ELECTRICITY_USAGE",
        device_class=SensorDeviceClass.POWER,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    DSMRSensorEntityDescription(
        key="current_electricity_delivery",
        translation_key="current_electricity_delivery",
        obis_reference="CURRENT_ELECTRICITY_DELIVERY",
        device_class=SensorDeviceClass.POWER,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    DSMRSensorEntityDescription(
        key="electricity_active_tariff",
        translation_key="electricity_active_tariff",
        obis_reference="ELECTRICITY_ACTIVE_TARIFF",
        dsmr_versions={"2.2", "4", "5", "5B", "5L", "5EONHU"},
        device_class=SensorDeviceClass.ENUM,
        options=["low", "normal"],
    ),
    DSMRSensorEntityDescription(
        key="electricity_used_tariff_1",
        translation_key="electricity_used_tariff_1",
        obis_reference="ELECTRICITY_USED_TARIFF_1",
        dsmr_versions={"2.2", "4", "5", "5B", "5L", "5EONHU"},
        device_class=SensorDeviceClass.ENERGY,
        state_class=SensorStateClass.TOTAL_INCREASING,
    ),
    DSMRSensorEntityDescription(
        key="electricity_used_tariff_2",
        translation_key="electricity_used_tariff_2",
        obis_reference="ELECTRICITY_USED_TARIFF_2",
        dsmr_versions={"2.2", "4", "5", "5B", "5L", "5EONHU"},
        device_class=SensorDeviceClass.ENERGY,
        state_class=SensorStateClass.TOTAL_INCREASING,
    ),
    DSMRSensorEntityDescription(
        key="electricity_used_tariff_3",
        translation_key="electricity_used_tariff_3",
        obis_reference="ELECTRICITY_USED_TARIFF_3",
        dsmr_versions={"5EONHU"},
        force_update=True,
        device_class=SensorDeviceClass.ENERGY,
        state_class=SensorStateClass.TOTAL_INCREASING,
    ),
    DSMRSensorEntityDescription(
        key="electricity_used_tariff_4",
        translation_key="electricity_used_tariff_4",
        obis_reference="ELECTRICITY_USED_TARIFF_4",
        dsmr_versions={"5EONHU"},
        force_update=True,
        device_class=SensorDeviceClass.ENERGY,
        state_class=SensorStateClass.TOTAL_INCREASING,
    ),
    DSMRSensorEntityDescription(
        key="electricity_delivered_tariff_1",
        translation_key="electricity_delivered_tariff_1",
        obis_reference="ELECTRICITY_DELIVERED_TARIFF_1",
        dsmr_versions={"2.2", "4", "5", "5B", "5L", "5EONHU"},
        device_class=SensorDeviceClass.ENERGY,
        state_class=SensorStateClass.TOTAL_INCREASING,
    ),
    DSMRSensorEntityDescription(
        key="electricity_delivered_tariff_2",
        translation_key="electricity_delivered_tariff_2",
        obis_reference="ELECTRICITY_DELIVERED_TARIFF_2",
        dsmr_versions={"2.2", "4", "5", "5B", "5L", "5EONHU"},
        device_class=SensorDeviceClass.ENERGY,
        state_class=SensorStateClass.TOTAL_INCREASING,
    ),
    DSMRSensorEntityDescription(
        key="electricity_delivered_tariff_3",
        translation_key="electricity_delivered_tariff_3",
        obis_reference="ELECTRICITY_DELIVERED_TARIFF_3",
        dsmr_versions={"5EONHU"},
        force_update=True,
        device_class=SensorDeviceClass.ENERGY,
        state_class=SensorStateClass.TOTAL_INCREASING,
    ),
    DSMRSensorEntityDescription(
        key="electricity_delivered_tariff_4",
        translation_key="electricity_delivered_tariff_4",
        obis_reference="ELECTRICITY_DELIVERED_TARIFF_4",
        dsmr_versions={"5EONHU"},
        force_update=True,
        device_class=SensorDeviceClass.ENERGY,
        state_class=SensorStateClass.TOTAL_INCREASING,
    ),
    DSMRSensorEntityDescription(
        key="instantaneous_active_power_l1_positive",
        translation_key="instantaneous_active_power_l1_positive",
        obis_reference="INSTANTANEOUS_ACTIVE_POWER_L1_POSITIVE",
        device_class=SensorDeviceClass.POWER,
        entity_registry_enabled_default=False,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    DSMRSensorEntityDescription(
        key="instantaneous_active_power_l2_positive",
        translation_key="instantaneous_active_power_l2_positive",
        obis_reference="INSTANTANEOUS_ACTIVE_POWER_L2_POSITIVE",
        device_class=SensorDeviceClass.POWER,
        entity_registry_enabled_default=False,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    DSMRSensorEntityDescription(
        key="instantaneous_active_power_l3_positive",
        translation_key="instantaneous_active_power_l3_positive",
        obis_reference="INSTANTANEOUS_ACTIVE_POWER_L3_POSITIVE",
        device_class=SensorDeviceClass.POWER,
        entity_registry_enabled_default=False,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    DSMRSensorEntityDescription(
        key="instantaneous_active_power_l1_negative",
        translation_key="instantaneous_active_power_l1_negative",
        obis_reference="INSTANTANEOUS_ACTIVE_POWER_L1_NEGATIVE",
        device_class=SensorDeviceClass.POWER,
        entity_registry_enabled_default=False,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    DSMRSensorEntityDescription(
        key="instantaneous_active_power_l2_negative",
        translation_key="instantaneous_active_power_l2_negative",
        obis_reference="INSTANTANEOUS_ACTIVE_POWER_L2_NEGATIVE",
        device_class=SensorDeviceClass.POWER,
        entity_registry_enabled_default=False,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    DSMRSensorEntityDescription(
        key="instantaneous_active_power_l3_negative",
        translation_key="instantaneous_active_power_l3_negative",
        obis_reference="INSTANTANEOUS_ACTIVE_POWER_L3_NEGATIVE",
        device_class=SensorDeviceClass.POWER,
        entity_registry_enabled_default=False,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    DSMRSensorEntityDescription(
        key="short_power_failure_count",
        translation_key="short_power_failure_count",
        obis_reference="SHORT_POWER_FAILURE_COUNT",
        dsmr_versions={"2.2", "4", "5", "5L"},
        entity_registry_enabled_default=False,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    DSMRSensorEntityDescription(
        key="long_power_failure_count",
        translation_key="long_power_failure_count",
        obis_reference="LONG_POWER_FAILURE_COUNT",
        dsmr_versions={"2.2", "4", "5", "5L"},
        entity_registry_enabled_default=False,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    DSMRSensorEntityDescription(
        key="voltage_sag_l1_count",
        translation_key="voltage_sag_l1_count",
        obis_reference="VOLTAGE_SAG_L1_COUNT",
        dsmr_versions={"2.2", "4", "5", "5L"},
        entity_registry_enabled_default=False,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    DSMRSensorEntityDescription(
        key="voltage_sag_l2_count",
        translation_key="voltage_sag_l2_count",
        obis_reference="VOLTAGE_SAG_L2_COUNT",
        dsmr_versions={"2.2", "4", "5", "5L"},
        entity_registry_enabled_default=False,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    DSMRSensorEntityDescription(
        key="voltage_sag_l3_count",
        translation_key="voltage_sag_l3_count",
        obis_reference="VOLTAGE_SAG_L3_COUNT",
        dsmr_versions={"2.2", "4", "5", "5L"},
        entity_registry_enabled_default=False,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    DSMRSensorEntityDescription(
        key="voltage_swell_l1_count",
        translation_key="voltage_swell_l1_count",
        obis_reference="VOLTAGE_SWELL_L1_COUNT",
        dsmr_versions={"2.2", "4", "5", "5L"},
        entity_registry_enabled_default=False,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    DSMRSensorEntityDescription(
        key="voltage_swell_l2_count",
        translation_key="voltage_swell_l2_count",
        obis_reference="VOLTAGE_SWELL_L2_COUNT",
        dsmr_versions={"2.2", "4", "5", "5L"},
        entity_registry_enabled_default=False,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    DSMRSensorEntityDescription(
        key="voltage_swell_l3_count",
        translation_key="voltage_swell_l3_count",
        obis_reference="VOLTAGE_SWELL_L3_COUNT",
        dsmr_versions={"2.2", "4", "5", "5L"},
        entity_registry_enabled_default=False,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    DSMRSensorEntityDescription(
        key="instantaneous_voltage_l1",
        translation_key="instantaneous_voltage_l1",
        obis_reference="INSTANTANEOUS_VOLTAGE_L1",
        device_class=SensorDeviceClass.VOLTAGE,
        entity_registry_enabled_default=False,
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    DSMRSensorEntityDescription(
        key="instantaneous_voltage_l2",
        translation_key="instantaneous_voltage_l2",
        obis_reference="INSTANTANEOUS_VOLTAGE_L2",
        device_class=SensorDeviceClass.VOLTAGE,
        entity_registry_enabled_default=False,
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    DSMRSensorEntityDescription(
        key="instantaneous_voltage_l3",
        translation_key="instantaneous_voltage_l3",
        obis_reference="INSTANTANEOUS_VOLTAGE_L3",
        device_class=SensorDeviceClass.VOLTAGE,
        entity_registry_enabled_default=False,
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    DSMRSensorEntityDescription(
        key="instantaneous_current_l1",
        translation_key="instantaneous_current_l1",
        obis_reference="INSTANTANEOUS_CURRENT_L1",
        device_class=SensorDeviceClass.CURRENT,
        entity_registry_enabled_default=False,
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    DSMRSensorEntityDescription(
        key="instantaneous_current_l2",
        translation_key="instantaneous_current_l2",
        obis_reference="INSTANTANEOUS_CURRENT_L2",
        device_class=SensorDeviceClass.CURRENT,
        entity_registry_enabled_default=False,
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    DSMRSensorEntityDescription(
        key="instantaneous_current_l3",
        translation_key="instantaneous_current_l3",
        obis_reference="INSTANTANEOUS_CURRENT_L3",
        device_class=SensorDeviceClass.CURRENT,
        entity_registry_enabled_default=False,
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    DSMRSensorEntityDescription(
        key="belgium_max_power_per_phase",
        translation_key="max_power_per_phase",
        obis_reference="ACTUAL_TRESHOLD_ELECTRICITY",
        dsmr_versions={"5B"},
        device_class=SensorDeviceClass.POWER,
        entity_registry_enabled_default=False,
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    DSMRSensorEntityDescription(
        key="belgium_max_current_per_phase",
        translation_key="max_current_per_phase",
        obis_reference="FUSE_THRESHOLD_L1",
        dsmr_versions={"5B"},
        device_class=SensorDeviceClass.CURRENT,
        entity_registry_enabled_default=False,
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    DSMRSensorEntityDescription(
        key="electricity_imported_total",
        translation_key="electricity_imported_total",
        obis_reference="ELECTRICITY_IMPORTED_TOTAL",
        dsmr_versions={"5L", "5S", "Q3D", "5EONHU"},
        device_class=SensorDeviceClass.ENERGY,
        state_class=SensorStateClass.TOTAL_INCREASING,
    ),
    DSMRSensorEntityDescription(
        key="electricity_exported_total",
        translation_key="electricity_exported_total",
        obis_reference="ELECTRICITY_EXPORTED_TOTAL",
        dsmr_versions={"5L", "5S", "Q3D", "5EONHU"},
        device_class=SensorDeviceClass.ENERGY,
        state_class=SensorStateClass.TOTAL_INCREASING,
    ),
    DSMRSensorEntityDescription(
        key="belgium_current_average_demand",
        translation_key="current_average_demand",
        obis_reference="BELGIUM