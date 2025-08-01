"""Support for interacting with Smappee Comport Plugs, Switches and Output Modules."""
from typing import Any, List, Generator, Optional
from homeassistant.components.switch import SwitchEntity
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from . import SmappeeConfigEntry
from .const import DOMAIN

SWITCH_PREFIX = 'Switch'


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: SmappeeConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    """Set up the Smappee Comfort Plugs."""
    smappee_base = config_entry.runtime_data
    entities: List[SmappeeActuator] = []
    for service_location in smappee_base.smappee.service_locations.values():
        for actuator_id, actuator in service_location.actuators.items():
            if actuator.type in ('SWITCH', 'COMFORT_PLUG'):
                entities.append(SmappeeActuator(
                    smappee_base, 
                    service_location, 
                    actuator.name, 
                    actuator_id, 
                    actuator.type, 
                    actuator.serialnumber
                ))
            elif actuator.type == 'INFINITY_OUTPUT_MODULE':
                entities.extend(
                    SmappeeActuator(
                        smappee_base, 
                        service_location, 
                        actuator.name, 
                        actuator_id, 
                        actuator.type, 
                        actuator.serialnumber, 
                        actuator_state_option=option
                    )
                    for option in actuator.state_options
                )
    async_add_entities(entities, True)


class SmappeeActuator(SwitchEntity):
    """Representation of a Smappee Comport Plug."""

    _attr_icon: str = 'mdi:toggle-switch'

    def __init__(
        self, 
        smappee_base: Any, 
        service_location: Any, 
        name: str, 
        actuator_id: str, 
        actuator_type: str, 
        actuator_serialnumber: str, 
        actuator_state_option: Optional[str] = None
    ) -> None:
        """Initialize a new Smappee Comfort Plug."""
        self._smappee_base = smappee_base
        self._service_location = service_location
        self._actuator_name = name
        self._actuator_id = actuator_id
        self._actuator_type = actuator_type
        self._actuator_serialnumber = actuator_serialnumber
        self._actuator_state_option = actuator_state_option
        actuator = service_location.actuators.get(actuator_id)
        self._state: str = actuator.state
        self._connection_state: str = actuator.connection_state
        self._attr_device_info: DeviceInfo = DeviceInfo(
            identifiers={(DOMAIN, service_location.device_serial_number)},
            manufacturer='Smappee',
            model=service_location.device_model,
            name=service_location.service_location_name,
            sw_version=service_location.firmware_version
        )

    @property
    def name(self) -> str:
        """Return the name of the switch."""
        if self._actuator_type == 'INFINITY_OUTPUT_MODULE':
            return f'{self._service_location.service_location_name} - Output module - {self._actuator_name} - {self._actuator_state_option}'
        return f'{self._service_location.service_location_name} - {self._actuator_type.title()} - {self._actuator_name}'

    @property
    def is_on(self) -> bool:
        """Return true if switch is on."""
        if self._actuator_type == 'INFINITY_OUTPUT_MODULE':
            return self._service_location.actuators.get(self._actuator_id).state == self._actuator_state_option
        return self._state == 'ON_ON'

    def turn_on(self, **kwargs: Any) -> None:
        """Turn on Comport Plug."""
        if self._actuator_type in ('SWITCH', 'COMFORT_PLUG'):
            self._service_location.set_actuator_state(self._actuator_id, state='ON_ON')
        elif self._actuator_type == 'INFINITY_OUTPUT_MODULE':
            self._service_location.set_actuator_state(self._actuator_id, state=self._actuator_state_option)

    def turn_off(self, **kwargs: Any) -> None:
        """Turn off Comport Plug."""
        if self._actuator_type in ('SWITCH', 'COMFORT_PLUG'):
            self._service_location.set_actuator_state(self._actuator_id, state='OFF_OFF')
        elif self._actuator_type == 'INFINITY_OUTPUT_MODULE':
            self._service_location.set_actuator_state(self._actuator_id, state='PLACEHOLDER', api=False)

    @property
    def available(self) -> bool:
        """Return True if entity is available. Unavailable for COMFORT_PLUGS."""
        return self._connection_state == 'CONNECTED' or self._actuator_type == 'COMFORT_PLUG'

    @property
    def unique_id(self) -> str:
        """Return the unique ID for this switch."""
        if self._actuator_type == 'INFINITY_OUTPUT_MODULE':
            return f'{self._service_location.device_serial_number}-{self._service_location.service_location_id}-actuator-{self._actuator_id}-{self._actuator_state_option}'
        return f'{self._service_location.device_serial_number}-{self._service_location.service_location_id}-actuator-{self._actuator_id}'

    async def async_update(self) -> None:
        """Get the latest data from Smappee and update the state."""
        await self._smappee_base.async_update()
        actuator = self._service_location.actuators.get(self._actuator_id)
        new_state: str = actuator.state
        if new_state != self._state:
            self._state = new_state
            self.async_write_ha_state()
        self._connection_state = actuator.connection_state
