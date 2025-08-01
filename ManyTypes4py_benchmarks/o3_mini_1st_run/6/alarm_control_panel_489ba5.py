from __future__ import annotations
from collections.abc import Callable, Generator
import logging
from typing import Any, Dict, Optional
from pyrisco.common import Partition
from pyrisco.local.partition import Partition as LocalPartition
from homeassistant.components.alarm_control_panel import (
    AlarmControlPanelEntity,
    AlarmControlPanelEntityFeature,
    AlarmControlPanelState,
    CodeFormat,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_PIN
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from . import LocalData, is_local
from .const import (
    CONF_CODE_ARM_REQUIRED,
    CONF_CODE_DISARM_REQUIRED,
    CONF_HA_STATES_TO_RISCO,
    CONF_RISCO_STATES_TO_HA,
    DATA_COORDINATOR,
    DEFAULT_OPTIONS,
    DOMAIN,
    RISCO_ARM,
    RISCO_GROUPS,
    RISCO_PARTIAL_ARM,
)
from .coordinator import RiscoDataUpdateCoordinator
from .entity import RiscoCloudEntity

_LOGGER = logging.getLogger(__name__)

STATES_TO_SUPPORTED_FEATURES: Dict[AlarmControlPanelState, int] = {
    AlarmControlPanelState.ARMED_AWAY: AlarmControlPanelEntityFeature.ARM_AWAY,
    AlarmControlPanelState.ARMED_CUSTOM_BYPASS: AlarmControlPanelEntityFeature.ARM_CUSTOM_BYPASS,
    AlarmControlPanelState.ARMED_HOME: AlarmControlPanelEntityFeature.ARM_HOME,
    AlarmControlPanelState.ARMED_NIGHT: AlarmControlPanelEntityFeature.ARM_NIGHT,
}


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the Risco alarm control panel."""
    options: Dict[str, Any] = {**DEFAULT_OPTIONS, **config_entry.options}
    if is_local(config_entry):
        local_data: LocalData = hass.data[DOMAIN][config_entry.entry_id]
        async_add_entities(
            (
                RiscoLocalAlarm(
                    local_data.system.id,
                    partition_id,
                    partition,
                    local_data.partition_updates,
                    config_entry.data[CONF_PIN],
                    options,
                )
                for partition_id, partition in local_data.system.partitions.items()
            )
        )
    else:
        coordinator: RiscoDataUpdateCoordinator = hass.data[DOMAIN][config_entry.entry_id][DATA_COORDINATOR]
        async_add_entities(
            (
                RiscoCloudAlarm(coordinator, partition_id, config_entry.data[CONF_PIN], options)
                for partition_id in coordinator.data.partitions
            )
        )


class RiscoAlarm(AlarmControlPanelEntity):
    """Representation of a Risco cloud partition."""

    _attr_code_format: CodeFormat = CodeFormat.NUMBER
    _attr_has_entity_name: bool = True
    _attr_name: Optional[str] = None

    def __init__(
        self,
        *,
        partition_id: str,
        partition: Partition,
        code: str,
        options: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Init the partition."""
        super().__init__(**kwargs)
        self._partition_id: str = partition_id
        self._partition: Partition = partition
        self._code: str = code
        self._attr_code_arm_required: bool = options[CONF_CODE_ARM_REQUIRED]
        self._code_disarm_required: bool = options[CONF_CODE_DISARM_REQUIRED]
        self._risco_to_ha: Dict[Any, AlarmControlPanelState] = options[CONF_RISCO_STATES_TO_HA]
        self._ha_to_risco: Dict[AlarmControlPanelState, Any] = options[CONF_HA_STATES_TO_RISCO]
        for state in self._ha_to_risco:
            self._attr_supported_features |= STATES_TO_SUPPORTED_FEATURES[state]

    @property
    def alarm_state(self) -> Optional[AlarmControlPanelState]:
        """Return the state of the device."""
        if self._partition.triggered:
            return AlarmControlPanelState.TRIGGERED
        if self._partition.arming:
            return AlarmControlPanelState.ARMING
        if self._partition.disarmed:
            return AlarmControlPanelState.DISARMED
        if self._partition.armed:
            return self._risco_to_ha[RISCO_ARM]
        if self._partition.partially_armed:
            for group, armed in self._partition.groups.items():
                if armed:
                    return self._risco_to_ha[group]
            return self._risco_to_ha[RISCO_PARTIAL_ARM]
        return None

    def _validate_code(self, code: Optional[str]) -> bool:
        """Validate given code."""
        return code == self._code

    async def async_alarm_disarm(self, code: Optional[str] = None) -> None:
        """Send disarm command."""
        if self._code_disarm_required and (not self._validate_code(code)):
            _LOGGER.warning("Wrong code entered for disarming")
            return
        await self._call_alarm_method("disarm")

    async def async_alarm_arm_home(self, code: Optional[str] = None) -> None:
        """Send arm home command."""
        await self._arm(AlarmControlPanelState.ARMED_HOME, code)

    async def async_alarm_arm_away(self, code: Optional[str] = None) -> None:
        """Send arm away command."""
        await self._arm(AlarmControlPanelState.ARMED_AWAY, code)

    async def async_alarm_arm_night(self, code: Optional[str] = None) -> None:
        """Send arm night command."""
        await self._arm(AlarmControlPanelState.ARMED_NIGHT, code)

    async def async_alarm_arm_custom_bypass(self, code: Optional[str] = None) -> None:
        """Send arm custom bypass command."""
        await self._arm(AlarmControlPanelState.ARMED_CUSTOM_BYPASS, code)

    async def _arm(self, mode: AlarmControlPanelState, code: Optional[str]) -> None:
        if self.code_arm_required and (not self._validate_code(code)):
            _LOGGER.warning("Wrong code entered for %s", mode)
            return
        risco_state = self._ha_to_risco.get(mode)
        if not risco_state:
            _LOGGER.warning("No mapping for mode %s", mode)
            return
        if risco_state in RISCO_GROUPS:
            await self._call_alarm_method("group_arm", risco_state)
        else:
            await self._call_alarm_method(risco_state)

    async def _call_alarm_method(self, method: str, *args: Any) -> None:
        raise NotImplementedError


class RiscoCloudAlarm(RiscoAlarm, RiscoCloudEntity):
    """Representation of a Risco partition."""

    def __init__(
        self,
        coordinator: RiscoDataUpdateCoordinator,
        partition_id: str,
        code: str,
        options: Dict[str, Any],
    ) -> None:
        """Init the partition."""
        super().__init__(
            partition_id=partition_id,
            partition=coordinator.data.partitions[partition_id],
            code=code,
            options=options,
        )
        self._attr_unique_id = f"{self._risco.site_uuid}_{partition_id}"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, self._attr_unique_id)},
            name=f"Risco {self._risco.site_name} Partition {partition_id}",
            manufacturer="Risco",
        )
        self.coordinator: RiscoDataUpdateCoordinator = coordinator

    def _get_data_from_coordinator(self) -> None:
        self._partition = self.coordinator.data.partitions[self._partition_id]

    async def _call_alarm_method(self, method: str, *args: Any) -> None:
        alarm = await getattr(self._risco, method)(self._partition_id, *args)
        self._partition = alarm.partitions[self._partition_id]
        self.async_write_ha_state()


class RiscoLocalAlarm(RiscoAlarm):
    """Representation of a Risco local, partition."""
    _attr_should_poll: bool = False

    def __init__(
        self,
        system_id: str,
        partition_id: str,
        partition: LocalPartition,
        partition_updates: Dict[str, Callable[[], None]],
        code: str,
        options: Dict[str, Any],
    ) -> None:
        """Init the partition."""
        super().__init__(partition_id=partition_id, partition=partition, code=code, options=options)
        self._system_id: str = system_id
        self._partition_updates: Dict[str, Callable[[], None]] = partition_updates
        self._attr_unique_id = f"{system_id}_{partition_id}_local"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, self._attr_unique_id)},
            name=partition.name,
            manufacturer="Risco",
        )

    async def async_added_to_hass(self) -> None:
        """Subscribe to updates."""
        self._partition_updates[self._partition_id] = self.async_write_ha_state

    async def _call_alarm_method(self, method: str, *args: Any) -> None:
        await getattr(self._partition, method)(*args)