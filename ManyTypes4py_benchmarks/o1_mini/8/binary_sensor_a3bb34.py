"""Support for Freebox devices (Freebox v6 and Freebox mini 4K)."""
from __future__ import annotations
import logging
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple
from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
    BinarySensorEntityDescription,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import ConfigType
from .const import DOMAIN, FreeboxHomeCategory
from .entity import FreeboxHomeEntity
from .router import FreeboxRouter

_LOGGER: logging.Logger = logging.getLogger(__name__)

RAID_SENSORS: Tuple[BinarySensorEntityDescription, ...] = (
    BinarySensorEntityDescription(
        key="raid_degraded",
        name="degraded",
        device_class=BinarySensorDeviceClass.PROBLEM,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up binary sensors."""
    router: FreeboxRouter = hass.data[DOMAIN][entry.unique_id]
    _LOGGER.debug("%s - %s - %s raid(s)", router.name, router.mac, len(router.raids))
    binary_entities: List[BinarySensorEntity] = [
        FreeboxRaidDegradedSensor(router, raid, description)
        for raid in router.raids.values()
        for description in RAID_SENSORS
    ]
    for node in router.home_devices.values():
        category: FreeboxHomeCategory = node["category"]
        if category == FreeboxHomeCategory.PIR:
            binary_entities.append(FreeboxPirSensor(hass, router, node))
        elif category == FreeboxHomeCategory.DWS:
            binary_entities.append(FreeboxDwsSensor(hass, router, node))
        binary_entities.extend(
            FreeboxCoverSensor(hass, router, node)
            for endpoint in node.get("show_endpoints", [])
            if endpoint.get("name") == "cover"
            and endpoint.get("ep_type") == "signal"
            and endpoint.get("value") is not None
        )
    async_add_entities(binary_entities, True)


class FreeboxHomeBinarySensor(FreeboxHomeEntity, BinarySensorEntity):
    """Representation of a Freebox binary sensor."""

    _sensor_name: str = "trigger"

    def __init__(
        self,
        hass: HomeAssistant,
        router: FreeboxRouter,
        node: Dict[str, Any],
        sub_node: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize a Freebox binary sensor."""
        super().__init__(hass, router, node, sub_node)
        self._command_id: Optional[str] = self.get_command_id(
            node["type"]["endpoints"], "signal", self._sensor_name
        )
        self._attr_is_on: Optional[bool] = self._edit_state(
            self.get_value("signal", self._sensor_name)
        )

    async def async_update_signal(self) -> None:
        """Update name & state."""
        value: Optional[bool] = await self.get_home_endpoint_value(self._command_id)
        self._attr_is_on = self._edit_state(value)
        await FreeboxHomeEntity.async_update_signal(self)

    def _edit_state(self, state: Optional[Any]) -> Optional[bool]:
        """Edit state depending on sensor name."""
        if state is None:
            return None
        if self._sensor_name == "trigger":
            return not state
        return bool(state)


class FreeboxPirSensor(FreeboxHomeBinarySensor):
    """Representation of a Freebox motion binary sensor."""

    _attr_device_class: BinarySensorDeviceClass = BinarySensorDeviceClass.MOTION


class FreeboxDwsSensor(FreeboxHomeBinarySensor):
    """Representation of a Freebox door opener binary sensor."""

    _attr_device_class: BinarySensorDeviceClass = BinarySensorDeviceClass.DOOR


class FreeboxCoverSensor(FreeboxHomeBinarySensor):
    """Representation of a cover Freebox plastic removal cover binary sensor (for some sensors: motion detector, door opener detector...)."""

    _attr_device_class: BinarySensorDeviceClass = BinarySensorDeviceClass.SAFETY
    _attr_entity_category: EntityCategory = EntityCategory.DIAGNOSTIC
    _attr_entity_registry_enabled_default: bool = False
    _sensor_name: str = "cover"

    def __init__(
        self, hass: HomeAssistant, router: FreeboxRouter, node: Dict[str, Any]
    ) -> None:
        """Initialize a cover for another device."""
        cover_node: Optional[Dict[str, Any]] = next(
            (
                endpoint
                for endpoint in node["type"]["endpoints"]
                if endpoint.get("name") == self._sensor_name
                and endpoint.get("ep_type") == "signal"
            ),
            None,
        )
        super().__init__(hass, router, node, cover_node)


class FreeboxRaidDegradedSensor(BinarySensorEntity):
    """Representation of a Freebox raid sensor."""

    _attr_should_poll: bool = False
    _attr_has_entity_name: bool = True

    def __init__(
        self,
        router: FreeboxRouter,
        raid: Dict[str, Any],
        description: BinarySensorEntityDescription,
    ) -> None:
        """Initialize a Freebox raid degraded sensor."""
        self.entity_description: BinarySensorEntityDescription = description
        self._router: FreeboxRouter = router
        self._attr_device_info = router.device_info
        self._raid: Dict[str, Any] = raid
        self._attr_name: str = f"Raid array {raid['id']} {description.name}"
        self._attr_unique_id: str = f"{router.mac} {description.key} {raid['name']} {raid['id']}"

    @callback
    def async_update_state(self) -> None:
        """Update the Freebox Raid sensor."""
        self._raid = self._router.raids[self._raid["id"]]

    @property
    def is_on(self) -> bool:
        """Return true if degraded."""
        return bool(self._raid.get("degraded"))

    @callback
    def async_on_demand_update(self) -> None:
        """Update state."""
        self.async_update_state()
        self.async_write_ha_state()

    async def async_added_to_hass(self) -> None:
        """Register state update callback."""
        self.async_update_state()
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                self._router.signal_sensor_update,
                self.async_on_demand_update,
            )
        )
