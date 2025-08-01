"""Support the ElkM1 Gold and ElkM1 EZ8 alarm/integration panels."""
from __future__ import annotations
from collections.abc import Iterable
from enum import Enum
import logging
from typing import Any, Optional, List, Type, Dict
from elkm1_lib.elements import Element
from elkm1_lib.elk import Elk
from homeassistant.const import ATTR_CONNECTIONS
from homeassistant.core import callback
from homeassistant.helpers.device_registry import CONNECTION_NETWORK_MAC, DeviceInfo
from homeassistant.helpers.entity import Entity
from .const import DOMAIN
from .models import ELKM1Data

_LOGGER: logging.Logger = logging.getLogger(__name__)


def create_elk_entities(
    elk_data: ELKM1Data,
    elk_elements: Iterable[Element],
    element_type: str,
    class_: Type[ElkEntity],
    entities: List[ElkEntity],
) -> Optional[List[ElkEntity]]:
    """Create the ElkM1 devices of a particular class."""
    auto_configure: bool = elk_data.auto_configure
    if not auto_configure and (not elk_data.config[element_type]["enabled"]):
        return None
    elk: Elk = elk_data.elk
    _LOGGER.debug("Creating elk entities for %s", elk)
    for element in elk_elements:
        if auto_configure:
            if not element.configured:
                continue
        elif not elk_data.config[element_type]["included"][element.index]:
            continue
        entities.append(class_(element, elk, elk_data))
    return entities


class ElkEntity(Entity):
    """Base class for all Elk entities."""
    _attr_has_entity_name: bool = True
    _attr_should_poll: bool = False

    def __init__(self, element: Element, elk: Elk, elk_data: ELKM1Data) -> None:
        """Initialize the base of all Elk devices."""
        self._elk: Elk = elk
        self._element: Element = element
        self._mac: str = elk_data.mac
        self._prefix: str = elk_data.prefix
        self._temperature_unit: Any = elk_data.config["temperature_unit"]
        if self._prefix != "":
            uid_start: str = f"elkm1m_{self._prefix}"
        else:
            uid_start = "elkm1"
        self._unique_id: str = f"{uid_start}_{self._element.default_name('_')}".lower()
        self._attr_name: str = element.name

    @property
    def unique_id(self) -> str:
        """Return unique id of the element."""
        return self._unique_id

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the default attributes of the element."""
        dict_as_str: Dict[str, Any] = {}
        for key, val in self._element.as_dict().items():
            dict_as_str[key] = val.value if isinstance(val, Enum) else val
        return {**dict_as_str, **self.initial_attrs()}

    @property
    def available(self) -> bool:
        """Is the entity available to be updated."""
        return self._elk.is_connected()

    def initial_attrs(self) -> Dict[str, Any]:
        """Return the underlying element's attributes as a dict."""
        return {"index": self._element.index + 1}

    def _element_changed(self, element: Element, changeset: Any) -> None:
        pass

    @callback
    def _element_callback(self, element: Element, changeset: Any) -> None:
        """Handle callback from an Elk element that has changed."""
        self._element_changed(element, changeset)
        self.async_write_ha_state()

    async def async_added_to_hass(self) -> None:
        """Register callback for ElkM1 changes and update entity state."""
        self._element.add_callback(self._element_callback)
        self._element_callback(self._element, {})

    @property
    def device_info(self) -> DeviceInfo:
        """Device info connecting via the ElkM1 system."""
        return DeviceInfo(
            name=self._element.name,
            identifiers={(DOMAIN, self._unique_id)},
            via_device=(DOMAIN, f"{self._prefix}_system"),
        )


class ElkAttachedEntity(ElkEntity):
    """An elk entity that is attached to the elk system."""

    @property
    def device_info(self) -> DeviceInfo:
        """Device info for the underlying ElkM1 system."""
        device_name: str = "ElkM1"
        if self._prefix:
            device_name += f" {self._prefix}"
        device_info: DeviceInfo = DeviceInfo(
            identifiers={(DOMAIN, f"{self._prefix}_system")},
            manufacturer="ELK Products, Inc.",
            model="M1",
            name=device_name,
            sw_version=self._elk.panel.elkm1_version,
        )
        if self._mac:
            device_info[ATTR_CONNECTIONS] = {(CONNECTION_NETWORK_MAC, self._mac)}
        return device_info