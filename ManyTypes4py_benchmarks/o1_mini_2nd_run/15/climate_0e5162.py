"""Climate support for Shelly."""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, cast, Optional, List

from aioshelly.block_device import Block
from aioshelly.const import BLU_TRV_IDENTIFIER, BLU_TRV_MODEL_NAME, RPC_GENERATIONS
from aioshelly.exceptions import DeviceConnectionError, InvalidAuthError
from homeassistant.components.climate import (
    DOMAIN as CLIMATE_DOMAIN,
    PRESET_NONE,
    ClimateEntity,
    ClimateEntityFeature,
    HVACAction,
    HVACMode,
)
from homeassistant.const import ATTR_TEMPERATURE, UnitOfTemperature
from homeassistant.core import HomeAssistant, State, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er, issue_registry as ir
from homeassistant.helpers.device_registry import (
    CONNECTION_BLUETOOTH,
    CONNECTION_NETWORK_MAC,
    DeviceInfo,
)
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.entity_registry import RegistryEntry
from homeassistant.helpers.restore_state import ExtraStoredData, RestoreEntity
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.util.unit_conversion import TemperatureConverter
from homeassistant.util.unit_system import UnitSystem, US_CUSTOMARY_SYSTEM

from .const import (
    BLU_TRV_TEMPERATURE_SETTINGS,
    BLU_TRV_TIMEOUT,
    DOMAIN,
    LOGGER,
    NOT_CALIBRATED_ISSUE_ID,
    RPC_THERMOSTAT_SETTINGS,
    SHTRV_01_TEMPERATURE_SETTINGS,
)
from .coordinator import ShellyBlockCoordinator, ShellyConfigEntry, ShellyRpcCoordinator
from .entity import ShellyRpcEntity
from .utils import (
    async_remove_shelly_entity,
    get_device_entry_gen,
    get_rpc_key_ids,
    is_rpc_thermostat_internal_actuator,
)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ShellyConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up climate device."""
    if get_device_entry_gen(config_entry) in RPC_GENERATIONS:
        async_setup_rpc_entry(hass, config_entry, async_add_entities)
        return
    coordinator: Optional[ShellyBlockCoordinator] = config_entry.runtime_data.block
    assert coordinator
    if coordinator.device.initialized:
        async_setup_climate_entities(async_add_entities, coordinator)
    else:
        async_restore_climate_entities(
            hass, config_entry, async_add_entities, coordinator
        )


@callback
def async_setup_climate_entities(
    async_add_entities: AddConfigEntryEntitiesCallback,
    coordinator: ShellyBlockCoordinator,
) -> None:
    """Set up online climate devices."""
    device_block: Optional[Block] = None
    sensor_block: Optional[Block] = None
    assert coordinator.device.blocks
    for block in coordinator.device.blocks:
        if block.type == "device":
            device_block = block
        if hasattr(block, "targetTemp"):
            sensor_block = block
    if sensor_block and device_block:
        LOGGER.debug("Setup online climate device %s", coordinator.name)
        async_add_entities(
            [BlockSleepingClimate(coordinator, sensor_block, device_block)]
        )


@callback
def async_restore_climate_entities(
    hass: HomeAssistant,
    config_entry: ShellyConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
    coordinator: ShellyBlockCoordinator,
) -> None:
    """Restore sleeping climate devices."""
    ent_reg: er.EntityRegistry = er.async_get(hass)
    entries: List[RegistryEntry] = er.async_entries_for_config_entry(
        ent_reg, config_entry.entry_id
    )
    for entry in entries:
        if entry.domain != CLIMATE_DOMAIN:
            continue
        LOGGER.debug("Setup sleeping climate device %s", coordinator.name)
        LOGGER.debug("Found entry %s [%s]", entry.original_name, entry.domain)
        async_add_entities([BlockSleepingClimate(coordinator, None, None, entry)])
        break


@callback
def async_setup_rpc_entry(
    hass: HomeAssistant,
    config_entry: ShellyConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up entities for RPC device."""
    coordinator: ShellyRpcCoordinator = config_entry.runtime_data.rpc
    assert coordinator
    climate_key_ids: List[str] = get_rpc_key_ids(coordinator.device.status, "thermostat")
    blutrv_key_ids: List[str] = get_rpc_key_ids(
        coordinator.device.status, BLU_TRV_IDENTIFIER
    )
    climate_ids: List[str] = []
    for id_ in climate_key_ids:
        climate_ids.append(id_)
        if is_rpc_thermostat_internal_actuator(coordinator.device.status):
            unique_id = f"{coordinator.mac}-switch:{id_}"
            async_remove_shelly_entity(hass, "switch", unique_id)
    if climate_ids:
        async_add_entities(
            (RpcClimate(coordinator, id_) for id_ in climate_ids)  # type: ignore
        )
    if blutrv_key_ids:
        async_add_entities(
            (RpcBluTrvClimate(coordinator, id_) for id_ in blutrv_key_ids)  # type: ignore
        )


@dataclass
class ShellyClimateExtraStoredData(ExtraStoredData):
    """Object to hold extra stored data."""

    last_target_temp: Optional[float] = None

    def as_dict(self) -> dict[str, Any]:
        """Return a dict representation of the text data."""
        return asdict(self)


class BlockSleepingClimate(
    CoordinatorEntity[ShellyBlockCoordinator], RestoreEntity, ClimateEntity
):
    """Representation of a Shelly climate device."""

    _attr_hvac_modes: List[HVACMode] = [HVACMode.OFF, HVACMode.HEAT]
    _attr_max_temp: float = SHTRV_01_TEMPERATURE_SETTINGS["max"]
    _attr_min_temp: float = SHTRV_01_TEMPERATURE_SETTINGS["min"]
    _attr_supported_features: ClimateEntityFeature = (
        ClimateEntityFeature.TARGET_TEMPERATURE
        | ClimateEntityFeature.PRESET_MODE
        | ClimateEntityFeature.TURN_OFF
        | ClimateEntityFeature.TURN_ON
    )
    _attr_target_temperature_step: float = SHTRV_01_TEMPERATURE_SETTINGS["step"]
    _attr_temperature_unit: UnitOfTemperature = UnitOfTemperature.CELSIUS

    block: Optional[Block]
    device_block: Optional[Block]
    last_state: Optional[State]
    last_state_attributes: Mapping[str, Any] = {}
    _preset_modes: List[str]
    _last_target_temp: float
    _unique_id: str
    _channel: int

    def __init__(
        self,
        coordinator: ShellyBlockCoordinator,
        sensor_block: Optional[Block],
        device_block: Optional[Block],
        entry: Optional[RegistryEntry] = None,
    ) -> None:
        """Initialize climate."""
        super().__init__(coordinator)
        self.block = sensor_block
        self.control_result: Optional[Any] = None
        self.device_block = device_block
        self.last_state = None
        self._preset_modes = []
        self._last_target_temp = SHTRV_01_TEMPERATURE_SETTINGS["default"]
        self._attr_name = coordinator.name
        if self.block is not None and self.device_block is not None:
            self._unique_id = f"{self.coordinator.mac}-{self.block.description}"
            assert self.block.channel is not None
            self._preset_modes = [
                PRESET_NONE,
                *coordinator.device.settings["thermostats"][int(self.block.channel)][
                    "schedule_profile_names"
                ],
            ]
        elif entry is not None:
            self._unique_id = entry.unique_id
        self._attr_device_info = DeviceInfo(
            connections={(CONNECTION_NETWORK_MAC, coordinator.mac)}
        )
        self._channel = cast(int, int(self._unique_id.split("_")[1]))

    @property
    def extra_restore_state_data(self) -> ShellyClimateExtraStoredData:
        """Return text specific state data to be restored."""
        return ShellyClimateExtraStoredData(self._last_target_temp)

    @property
    def unique_id(self) -> str:
        """Return unique id of entity."""
        return self._unique_id

    @property
    def target_temperature(self) -> Optional[float]:
        """Return target temperature."""
        if self.block is not None:
            return cast(float, self.block.targetTemp)
        target_temp = self.last_state_attributes.get("temperature")
        if (
            isinstance(self.hass.config.units, UnitSystem)
            and self.hass.config.units is US_CUSTOMARY_SYSTEM
            and target_temp is not None
        ):
            return TemperatureConverter.convert(
                cast(float, target_temp),
                UnitOfTemperature.FAHRENHEIT,
                UnitOfTemperature.CELSIUS,
            )
        return cast(Optional[float], target_temp)

    @property
    def current_temperature(self) -> Optional[float]:
        """Return current temperature."""
        if self.block is not None:
            return cast(float, self.block.temp)
        current_temp = self.last_state_attributes.get("current_temperature")
        if (
            isinstance(self.hass.config.units, UnitSystem)
            and self.hass.config.units is US_CUSTOMARY_SYSTEM
            and current_temp is not None
        ):
            return TemperatureConverter.convert(
                cast(float, current_temp),
                UnitOfTemperature.FAHRENHEIT,
                UnitOfTemperature.CELSIUS,
            )
        return cast(Optional[float], current_temp)

    @property
    def available(self) -> bool:
        """Device availability."""
        if self.device_block is not None:
            return not cast(bool, self.device_block.valveError)
        return super().available

    @property
    def hvac_mode(self) -> HVACMode:
        """HVAC current mode."""
        if self.device_block is None:
            if self.last_state and self.last_state.state in list(HVACMode):
                return HVACMode(self.last_state.state)
            return HVACMode.OFF
        if self.device_block.mode is None or self._check_is_off():
            return HVACMode.OFF
        return HVACMode.HEAT

    @property
    def preset_mode(self) -> Optional[str]:
        """Preset current mode."""
        if self.device_block is None:
            return cast(Optional[str], self.last_state_attributes.get("preset_mode"))
        if self.device_block.mode is None:
            return PRESET_NONE
        return self._preset_modes[cast(int, self.device_block.mode)]

    @property
    def hvac_action(self) -> HVACAction:
        """HVAC current action."""
        if self.device_block is None or self.device_block.status is None or self._check_is_off():
            return HVACAction.OFF
        return HVACAction.HEATING if bool(self.device_block.status) else HVACAction.IDLE

    @property
    def preset_modes(self) -> List[str]:
        """Preset available modes."""
        return self._preset_modes

    def _check_is_off(self) -> bool:
        """Return if valve is off or on."""
        return bool(
            self.target_temperature is None
            or (self.target_temperature <= self._attr_min_temp)
        )

    async def set_state_full_path(self, **kwargs: Any) -> Any:
        """Set block state (HTTP request)."""
        LOGGER.debug("Setting state for entity %s, state: %s", self.name, kwargs)
        try:
            return await self.coordinator.device.http_request(
                "get", f"thermostat/{self._channel}", kwargs
            )
        except DeviceConnectionError as err:
            self.coordinator.last_update_success = False
            raise HomeAssistantError(
                f"Setting state for entity {self.name} failed, state: {kwargs}, error: {err!r}"
            ) from err
        except InvalidAuthError:
            await self.coordinator.async_shutdown_device_and_start_reauth()

    async def async_set_temperature(self, **kwargs: Any) -> None:
        """Set new target temperature."""
        current_temp: Optional[float] = kwargs.get(ATTR_TEMPERATURE)
        if current_temp is None:
            return
        if self.block is not None and self.block.channel is not None:
            therm: Mapping[str, Any] = self.coordinator.device.settings["thermostats"][
                int(self.block.channel)
            ]
            LOGGER.debug("Themostat settings: %s", therm)
            if therm.get("target_t", {}).get("units", "C") == "F":
                current_temp = TemperatureConverter.convert(
                    cast(float, current_temp),
                    UnitOfTemperature.CELSIUS,
                    UnitOfTemperature.FAHRENHEIT,
                )
        await self.set_state_full_path(target_t_enabled=1, target_t=f"{current_temp}")

    async def async_set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        """Set hvac mode."""
        if hvac_mode == HVACMode.OFF:
            if isinstance(self.target_temperature, float):
                self._last_target_temp = self.target_temperature
            await self.set_state_full_path(
                target_t_enabled=1, target_t=f"{self._attr_min_temp}"
            )
        if hvac_mode == HVACMode.HEAT:
            await self.set_state_full_path(
                target_t_enabled=1, target_t=self._last_target_temp
            )

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        """Set preset mode."""
        if not self._preset_modes:
            return
        try:
            preset_index: int = self._preset_modes.index(preset_mode)
        except ValueError:
            return
        if preset_index == 0:
            await self.set_state_full_path(schedule=0)
        else:
            await self.set_state_full_path(
                schedule=1, schedule_profile=f"{preset_index}"
            )

    async def async_added_to_hass(self) -> None:
        """Handle entity which will be added."""
        LOGGER.info("Restoring entity %s", self.name)
        last_state: Optional[State] = await self.async_get_last_state()
        if last_state is not None:
            self.last_state = last_state
            self.last_state_attributes = self.last_state.attributes
            self._preset_modes = cast(
                List[str], self.last_state.attributes.get("preset_modes", [])
            )
        last_extra_data: Optional[ShellyClimateExtraStoredData] = await self.async_get_last_extra_data()
        if last_extra_data is not None:
            self._last_target_temp = last_extra_data.as_dict().get("last_target_temp", self._last_target_temp)
        await super().async_added_to_hass()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle device update."""
        if not self.coordinator.device.initialized:
            self.async_write_ha_state()
            return
        if self.coordinator.device.status.get("calibrated") is False:
            ir.async_create_issue(
                self.hass,
                DOMAIN,
                NOT_CALIBRATED_ISSUE_ID.format(unique=self.coordinator.mac),
                is_fixable=False,
                is_persistent=False,
                severity=ir.IssueSeverity.ERROR,
                translation_key="device_not_calibrated",
                translation_placeholders={
                    "device_name": self.coordinator.name,
                    "ip_address": self.coordinator.device.ip_address,
                },
            )
        else:
            ir.async_delete_issue(
                self.hass, DOMAIN, NOT_CALIBRATED_ISSUE_ID.format(unique=self.coordinator.mac)
            )
        assert self.coordinator.device.blocks
        for block in self.coordinator.device.blocks:
            if block.type == "device":
                self.device_block = block
            if hasattr(block, "targetTemp"):
                self.block = block
        if self.device_block and self.block:
            LOGGER.debug("Entity %s attached to blocks", self.name)
            assert self.block.channel is not None
            try:
                self._preset_modes = [
                    PRESET_NONE,
                    *self.coordinator.device.settings["thermostats"][
                        int(self.block.channel)
                    ]["schedule_profile_names"],
                ]
            except InvalidAuthError:
                self.hass.async_create_task(
                    self.coordinator.async_shutdown_device_and_start_reauth()
                )
            else:
                self.async_write_ha_state()


class RpcClimate(ShellyRpcEntity, ClimateEntity):
    """Entity that controls a thermostat on RPC based Shelly devices."""

    _attr_max_temp: float = RPC_THERMOSTAT_SETTINGS["max"]
    _attr_min_temp: float = RPC_THERMOSTAT_SETTINGS["min"]
    _attr_supported_features: ClimateEntityFeature = (
        ClimateEntityFeature.TARGET_TEMPERATURE
        | ClimateEntityFeature.TURN_OFF
        | ClimateEntityFeature.TURN_ON
    )
    _attr_target_temperature_step: float = RPC_THERMOSTAT_SETTINGS["step"]
    _attr_temperature_unit: UnitOfTemperature = UnitOfTemperature.CELSIUS

    def __init__(self, coordinator: ShellyRpcCoordinator, id_: str) -> None:
        """Initialize."""
        super().__init__(coordinator, f"thermostat:{id_}")
        self._id: str = id_
        self._thermostat_type: str = cast(
            str, coordinator.device.config[f"thermostat:{id_}"].get("type", "heating")
        )
        if self._thermostat_type == "cooling":
            self._attr_hvac_modes = [HVACMode.OFF, HVACMode.COOL]
        else:
            self._attr_hvac_modes = [HVACMode.OFF, HVACMode.HEAT]
        self._humidity_key: Optional[str] = None
        humidity_key_candidate: str = f"humidity:{id_}"
        if humidity_key_candidate in self.coordinator.device.status:
            self._humidity_key = humidity_key_candidate

    @property
    def target_temperature(self) -> float:
        """Return target temperature."""
        return cast(float, self.status["target_C"])

    @property
    def current_temperature(self) -> float:
        """Return current temperature."""
        return cast(float, self.status["current_C"])

    @property
    def current_humidity(self) -> Optional[float]:
        """Return current humidity."""
        if self._humidity_key is None:
            return None
        return cast(float, self.coordinator.device.status[self._humidity_key]["rh"])

    @property
    def hvac_mode(self) -> HVACMode:
        """HVAC current mode."""
        if not self.status["enable"]:
            return HVACMode.OFF
        return HVACMode.COOL if self._thermostat_type == "cooling" else HVACMode.HEAT

    @property
    def hvac_action(self) -> HVACAction:
        """HVAC current action."""
        if not self.status["output"]:
            return HVACAction.IDLE
        return HVACAction.COOLING if self._thermostat_type == "cooling" else HVACAction.HEATING

    async def async_set_temperature(self, **kwargs: Any) -> None:
        """Set new target temperature."""
        target_temp: Optional[float] = kwargs.get(ATTR_TEMPERATURE)
        if target_temp is None:
            return
        await self.call_rpc(
            "Thermostat.SetConfig",
            {"config": {"id": self._id, "target_C": target_temp}},
        )

    async def async_set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        """Set hvac mode."""
        mode: bool = hvac_mode in (HVACMode.COOL, HVACMode.HEAT)
        await self.call_rpc(
            "Thermostat.SetConfig",
            {"config": {"id": self._id, "enable": mode}},
        )


class RpcBluTrvClimate(ShellyRpcEntity, ClimateEntity):
    """Entity that controls a thermostat on RPC based Shelly devices."""

    _attr_max_temp: float = BLU_TRV_TEMPERATURE_SETTINGS["max"]
    _attr_min_temp: float = BLU_TRV_TEMPERATURE_SETTINGS["min"]
    _attr_supported_features: ClimateEntityFeature = (
        ClimateEntityFeature.TARGET_TEMPERATURE | ClimateEntityFeature.TURN_ON
    )
    _attr_hvac_modes: List[HVACMode] = [HVACMode.HEAT]
    _attr_hvac_mode: HVACMode = HVACMode.HEAT
    _attr_target_temperature_step: float = BLU_TRV_TEMPERATURE_SETTINGS["step"]
    _attr_temperature_unit: UnitOfTemperature = UnitOfTemperature.CELSIUS
    _attr_has_entity_name: bool = True

    _id: str
    _config: Mapping[str, Any]

    def __init__(self, coordinator: ShellyRpcCoordinator, id_: str) -> None:
        """Initialize."""
        super().__init__(coordinator, f"{BLU_TRV_IDENTIFIER}:{id_}")
        self._id = id_
        self._config = coordinator.device.config[f"{BLU_TRV_IDENTIFIER}:{id_}"]
        ble_addr: str = self._config["addr"]
        self._attr_unique_id: str = f"{ble_addr}-{self.key}"
        name: str = self._config["name"] or f"shellyblutrv-{ble_addr.replace(':', '')}"
        model_id: str = self._config.get("local_name", "")
        self._attr_device_info = DeviceInfo(
            connections={(CONNECTION_BLUETOOTH, ble_addr)},
            identifiers={(DOMAIN, ble_addr)},
            via_device=(DOMAIN, self.coordinator.mac),
            manufacturer="Shelly",
            model=BLU_TRV_MODEL_NAME.get(model_id, ""),
            model_id=model_id,
            name=name,
        )
        self._attr_name = None

    @property
    def target_temperature(self) -> Optional[float]:
        """Return target temperature."""
        if not self._config["enable"]:
            return None
        return cast(float, self.status["target_C"])

    @property
    def current_temperature(self) -> float:
        """Return current temperature."""
        return cast(float, self.status["current_C"])

    @property
    def hvac_action(self) -> HVACAction:
        """HVAC current action."""
        if not self.status["pos"]:
            return HVACAction.IDLE
        return HVACAction.HEATING

    async def async_set_temperature(self, **kwargs: Any) -> None:
        """Set new target temperature."""
        target_temp: Optional[float] = kwargs.get(ATTR_TEMPERATURE)
        if target_temp is None:
            return
        await self.call_rpc(
            "BluTRV.Call",
            {
                "id": self._id,
                "method": "Trv.SetTarget",
                "params": {"id": 0, "target_C": target_temp},
            },
            timeout=BLU_TRV_TIMEOUT,
        )
