"""Support for Huawei LTE sensors."""

from __future__ import annotations

from bisect import bisect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import re
from typing import Any, Dict, List, Optional, Pattern, Tuple, Union

from homeassistant.components.sensor import (
    DOMAIN as SENSOR_DOMAIN,
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    PERCENTAGE,
    EntityCategory,
    UnitOfDataRate,
    UnitOfFrequency,
    UnitOfInformation,
    UnitOfTime,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import StateType

from . import Router
from .const import (
    DOMAIN,
    KEY_DEVICE_INFORMATION,
    KEY_DEVICE_SIGNAL,
    KEY_MONITORING_CHECK_NOTIFICATIONS,
    KEY_MONITORING_MONTH_STATISTICS,
    KEY_MONITORING_STATUS,
    KEY_MONITORING_TRAFFIC_STATISTICS,
    KEY_NET_CURRENT_PLMN,
    KEY_NET_NET_MODE,
    KEY_SMS_SMS_COUNT,
    SENSOR_KEYS,
)
from .entity import HuaweiLteBaseEntityWithDevice

_LOGGER: logging.Logger = logging.getLogger(__name__)


def format_default(value: StateType) -> Tuple[StateType, Optional[str]]:
    """Format value."""
    unit = None
    if value is not None:
        # Clean up value and infer unit, e.g. -71dBm, 15 dB
        if match := re.match(
            r"((&[gl]t;|[><])=?)?(?P<value>.+?)\s*(?P<unit>[a-zA-Z]+)\s*$", str(value)
        ):
            try:
                value = float(match.group("value"))
                unit = match.group("unit")
            except ValueError:
                pass
    return value, unit


def format_freq_mhz(value: StateType) -> Tuple[StateType, UnitOfFrequency]]:
    """Format a frequency value for which source is in tenths of MHz."""
    return (
        float(value) / 10 if value is not None else None,
        UnitOfFrequency.MEGAHERTZ,
    )


def format_last_reset_elapsed_seconds(value: Optional[str]) -> Optional[datetime]:
    """Convert elapsed seconds to last reset datetime."""
    if value is None:
        return None
    try:
        last_reset = datetime.now() - timedelta(seconds=int(value))
        last_reset.replace(microsecond=0)
    except ValueError:
        return None
    return last_reset


def signal_icon(limits: Sequence[int], value: StateType) -> str:
    """Get signal icon."""
    return (
        "mdi:signal-cellular-outline",
        "mdi:signal-cellular-1",
        "mdi:signal-cellular-2",
        "mdi:signal-cellular-3",
    )[bisect(limits, value if value is not None else -1000)]


def bandwidth_icon(limits: Sequence[int], value: StateType) -> str:
    """Get bandwidth icon."""
    return (
        "mdi:speedometer-slow",
        "mdi:speedometer-medium",
        "mdi:speedometer",
    )[bisect(limits, value if value is not None else -1000)]


@dataclass
class HuaweiSensorGroup:
    """Class describing Huawei LTE sensor groups."""

    descriptions: Dict[str, "HuaweiSensorEntityDescription"]
    include: Optional[Pattern[str]] = None
    exclude: Optional[Pattern[str]] = None


@dataclass(frozen=True)
class HuaweiSensorEntityDescription(SensorEntityDescription):
    """Class describing Huawei LTE sensor entities."""

    name: str = ""

    format_fn: Callable[[str], Tuple[StateType, Optional[str]]] = format_default
    icon_fn: Optional[Callable[[StateType], str]] = None
    device_class_fn: Optional[Callable[[StateType], Optional[SensorDeviceClass]]] = None
    last_reset_item: Optional[str] = None
    last_reset_format_fn: Optional[Callable[[Optional[str]], Optional[datetime]]] = None


SENSOR_META: Dict[str, HuaweiSensorGroup] = {
    #
    # Device information
    #
    KEY_DEVICE_INFORMATION: HuaweiSensorGroup(
        include=re.compile(r"^(WanIP.*Address|uptime)$", re.IGNORECASE),
        descriptions={
            "uptime": HuaweiSensorEntityDescription(
                key="uptime",
                translation_key="uptime",
                icon="mdi:timer-outline",
                native_unit_of_measurement=UnitOfTime.SECONDS,
                device_class=SensorDeviceClass.DURATION,
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "WanIPAddress": HuaweiSensorEntityDescription(
                key="WanIPAddress",
                translation_key="wan_ip_address",
                icon="mdi:ip",
                entity_category=EntityCategory.DIAGNOSTIC,
                entity_registry_enabled_default=True,
            ),
            "WanIPv6Address": HuaweiSensorEntityDescription(
                key="WanIPv6Address",
                translation_key="wan_ipv6_address",
                icon="mdi:ip",
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
        },
    ),
    #
    # Signal
    #
    KEY_DEVICE_SIGNAL: HuaweiSensorGroup(
        descriptions={
            "arfcn": HuaweiSensorEntityDescription(
                key="arfcn",
                translation_key="arfcn",
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "band": HuaweiSensorEntityDescription(
                key="band",
                translation_key="band",
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "bsic": HuaweiSensorEntityDescription(
                key="bsic",
                translation_key="base_station_identity_code",
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "cell_id": HuaweiSensorEntityDescription(
                key="cell_id",
                translation_key="cell_id",
                icon="mdi:transmission-tower",
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "cqi0": HuaweiSensorEntityDescription(
                key="cqi0",
                translation_key="cqi0",
                icon="mdi:speedometer",
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "cqi1": HuaweiSensorEntityDescription(
                key="cqi1",
                translation_key="cqi1",
                icon="mdi:speedometer",
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "dl_mcs": HuaweiSensorEntityDescription(
                key="dl_mcs",
                translation_key="downlink_mcs",
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "dlbandwidth": HuaweiSensorEntityDescription(
                key="dlbandwidth",
                translation_key="downlink_bandwidth",
                icon_fn=lambda x: bandwidth_icon((8, 15), x),
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "dlfrequency": HuaweiSensorEntityDescription(
                key="dlfrequency",
                translation_key="downlink_frequency",
                device_class=SensorDeviceClass.FREQUENCY,
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "earfcn": HuaweiSensorEntityDescription(
                key="earfcn",
                translation_key="earfcn",
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "ecio": HuaweiSensorEntityDescription(
                key="ecio",
                translation_key="ecio",
                device_class=SensorDeviceClass.SIGNAL_STRENGTH,
                icon_fn=lambda x: signal_icon((-20, -10, -6), x),
                state_class=SensorStateClass.MEASUREMENT,
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "enodeb_id": HuaweiSensorEntityDescription(
                key="enodeb_id",
                translation_key="enodeb_id",
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "lac": HuaweiSensorEntityDescription(
                key="lac",
                translation_key="lac",
                icon="mdi:map-marker",
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "ltedlfreq": HuaweiSensorEntityDescription(
                key="ltedlfreq",
                translation_key="lte_downlink_frequency",
                format_fn=format_freq_mhz,
                suggested_display_precision=0,
                device_class=SensorDeviceClass.FREQUENCY,
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "lteulfreq": HuaweiSensorEntityDescription(
                key="lteulfreq",
                translation_key="lte_uplink_frequency",
                format_fn=format_freq_mhz,
                suggested_display_precision=0,
                device_class=SensorDeviceClass.FREQUENCY,
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "mode": HuaweiSensorEntityDescription(
                key="mode",
                translation_key="mode",
                format_fn=lambda x: (
                    {"0": "2G", "2": "3G", "7": "4G"}.get(x),
                    None,
                ),
                icon_fn=lambda x: (
                    {
                        "2G": "mdi:signal-2g",
                        "3G": "mdi:signal-3g",
                        "4G": "mdi:signal-4g",
                    }.get(str(x), "mdi:signal")
                ),
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "nrbler": HuaweiSensorEntityDescription(
                key="nrbler",
                translation_key="nrbler",
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "nrcqi0": HuaweiSensorEntityDescription(
                key="nrcqi0",
                translation_key="nrcqi0",
                icon="mdi:speedometer",
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "nrcqi1": HuaweiSensorEntityDescription(
                key="nrcqi1",
                translation_key="nrcqi1",
                icon="mdi:speedometer",
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "nrdlbandwidth": HuaweiSensorEntityDescription(
                key="nrdlbandwidth",
                translation_key="nrdlbandwidth",
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "nrdlmcs": HuaweiSensorEntityDescription(
                key="nrdlmcs",
                translation_key="nrdlmcs",
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "nrearfcn": HuaweiSensorEntityDescription(
                key="nrearfcn",
                translation_key="nrearfcn",
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "nrrank": HuaweiSensorEntityDescription(
                key="nrrank",
                translation_key="nrrank",
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "nrrsrp": HuaweiSensorEntityDescription(
                key="nrrsrp",
                translation_key="nrrsrp",
                device_class=SensorDeviceClass.SIGNAL_STRENGTH,
                state_class=SensorStateClass.MEASUREMENT,
                entity_category=EntityCategory.DIAGNOSTIC,
                entity_registry_enabled_default=True,
            ),
            "nrrsrq": HuaweiSensorEntityDescription(
                key="nrrsrq",
                translation_key="nrrsrq",
                device_class=SensorDeviceClass.SIGNAL_STRENGTH,
                state_class=SensorStateClass.MEASUREMENT,
                entity_category=EntityCategory.DIAGNOSTIC,
                entity_registry_enabled_default=True,
            ),
            "nrsinr": HuaweiSensorEntityDescription(
                key="nrsinr",
                translation_key="nrsinr",
                device_class=SensorDeviceClass.SIGNAL_STRENGTH,
                state_class=SensorStateClass.MEASUREMENT,
                entity_category=EntityCategory.DIAGNOSTIC,
                entity_registry_enabled_default=True,
            ),
            "nrtxpower": HuaweiSensorEntityDescription(
                key="nrtxpower",
                translation_key="nrtxpower",
                device_class_fn=lambda x: (
                    SensorDeviceClass.SIGNAL_STRENGTH
                    if isinstance(x, (float, int))
                    else None
                ),
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "nrulbandwidth": HuaweiSensorEntityDescription(
                key="nrulbandwidth",
                translation_key="nrulbandwidth",
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "nrulmcs": HuaweiSensorEntityDescription(
                key="nrulmcs",
                translation_key="nrulmcs",
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "pci": HuaweiSensorEntityDescription(
                key="pci",
                translation_key="pci",
                icon="mdi:transmission-tower",
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "plmn": HuaweiSensorEntityDescription(
                key="plmn",
                translation_key="plmn",
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "rac": HuaweiSensorEntityDescription(
                key="rac",
                translation_key="rac",
                icon="mdi:map-marker",
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "rrc_status": HuaweiSensorEntityDescription(
                key="rrc_status",
                translation_key="rrc_status",
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "rscp": HuaweiSensorEntityDescription(
                key="rscp",
                translation_key="rscp",
                device_class=SensorDeviceClass.SIGNAL_STRENGTH,
                icon_fn=lambda x: signal_icon((-95, -85, -75), x),
                state_class=SensorStateClass.MEASUREMENT,
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "rsrp": HuaweiSensorEntityDescription(
                key="rsrp",
                translation_key="rsrp",
                device_class=SensorDeviceClass.SIGNAL_STRENGTH,
                icon_fn=lambda x: signal_icon((-110, -95, -80), x),
                state_class=SensorStateClass.MEASUREMENT,
                entity_category=EntityCategory.DIAGNOSTIC,
                entity_registry_enabled_default=True,
            ),
            "rsrq": HuaweiSensorEntityDescription(
                key="rsrq",
                translation_key="rsrq",
                device_class=SensorDeviceClass.SIGNAL_STRENGTH,
                icon_fn=lambda x: signal_icon((-11, -8, -5), x),
                state_class=SensorStateClass.MEASUREMENT,
                entity_category=EntityCategory.DIAGNOSTIC,
                entity_registry_enabled_default=True,
            ),
            "rssi": HuaweiSensorEntityDescription(
                key="rssi",
                translation_key="rssi",
                device_class=SensorDeviceClass.SIGNAL_STRENGTH,
                icon_fn=lambda x: signal_icon((-80, -70, -60), x),
                state_class=SensorStateClass.MEASUREMENT,
                entity_category=EntityCategory.DIAGNOSTIC,
                entity_registry_enabled_default=True,
            ),
            "sinr": HuaweiSensorEntityDescription(
                key="sinr",
                translation_key="sinr",
                device_class=SensorDeviceClass.SIGNAL_STRENGTH,
                icon_fn=lambda x: signal_icon((0, 5, 10), x),
                state_class=SensorStateClass.MEASUREMENT,
                entity_category=EntityCategory.DIAGNOSTIC,
                entity_registry_enabled_default=True,
            ),
            "tac": HuaweiSensorEntityDescription(
                key="tac",
                translation_key="tac",
                icon="mdi:map-marker",
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "tdd": HuaweiSensorEntityDescription(
                key="tdd",
                translation_key="tdd",
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "transmode": HuaweiSensorEntityDescription(
                key="transmode",
                translation_key="transmission_mode",
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "txpower": HuaweiSensorEntityDescription(
                key="txpower",
                translation_key="transmit_power",
                device_class_fn=lambda x: (
                    SensorDeviceClass.SIGNAL_STRENGTH
                    if isinstance(x, (float, int))
                    else None
                ),
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "ul_mcs": HuaweiSensorEntityDescription(
                key="ul_mcs",
                translation_key="uplink_mcs",
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "ulbandwidth": HuaweiSensorEntityDescription(
                key="ulbandwidth",
                translation_key="uplink_bandwidth",
                icon_fn=lambda x: bandwidth_icon((8, 15), x),
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
            "ulfrequency": HuaweiSensorEntityDescription(
                key="ulfrequency",
                translation_key="uplink_frequency",
                device_class=SensorDeviceClass.FREQUENCY,
                entity_category=EntityCategory.DIAGNOSTIC,
            ),
        }
    ),
    #
    # Monitoring
    #
    KEY_MONITORING_CHECK_NOTIFICATIONS: HuaweiSensorGroup(
        exclude=re.compile(
            r"^(onlineupdatestatus|smsstoragefull)$",
            re.IGNORECASE,
        ),
        descriptions={
            "UnreadMessage": HuaweiSensorEntityDescription(
                key="UnreadMessage",
                translation_key="sms_unread",
                icon="mdi:email-arrow-left",
            ),
        },
    ),
    KEY_MONITORING_MONTH_STATISTICS: HuaweiSensorGroup(
        exclude=re.compile(
            r"^(currentday|month)(duration|lastcleartime)$", re.IGNORECASE
        ),
        descriptions={
            "CurrentDayUsed": Huawei