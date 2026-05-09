@dataclass(frozen=True, kw_only=True)
class BlockSensorDescription(BlockEntityDescription, SensorEntityDescription):
    """Class to describe a BLOCK sensor."""

@dataclass(frozen=True, kw_only=True)
class RpcSensorDescription(RpcEntityDescription, SensorEntityDescription):
    """Class to describe a RPC sensor."""
    device_class_fn: Callable[[dict], SensorDeviceClass | None] | None = None

@dataclass(frozen=True, kw_only=True)
class RestSensorDescription(RestEntityDescription, SensorEntityDescription):
    """Class to describe a REST sensor."""

class RpcSensor(ShellyRpcAttributeEntity, SensorEntity):
    """Represent a RPC sensor."""

    def __init__(self, coordinator: ShellyRpcCoordinator, key: str, attribute: str, description: RpcSensorDescription) -> None:
        """Initialize select."""
        super().__init__(coordinator, key, attribute, description)
        if self.option_map:
            self._attr_options = list(self.option_map.values())
        if description.device_class_fn is not None:
            if (device_class := description.device_class_fn(coordinator.device.config[key])):
                self._attr_device_class = device_class

    @property
    def native_value(self) -> StateType:
        """Return value of sensor."""
        attribute_value = self.attribute_value
        if not self.option_map:
            return attribute_value
        if not isinstance(attribute_value, str):
            return None
        return self.option_map[attribute_value]

class RpcBluTrvSensor(RpcSensor):
    """Represent a RPC BluTrv sensor."""

    def __init__(self, coordinator: ShellyRpcCoordinator, key: str, attribute: str, description: RpcSensorDescription) -> None:
        """Initialize."""
        super().__init__(coordinator, key, attribute, description)
        ble_addr = coordinator.device.config[key]['addr']
        self._attr_device_info = DeviceInfo(connections={(CONNECTION_BLUETOOTH, ble_addr)})

SENSORS: dict[tuple[str, str], BlockSensorDescription] = {...}

REST_SENSORS: dict[str, RestSensorDescription] = {...}

RPC_SENSORS: dict[str, RpcSensorDescription] = {...}

async def async_setup_entry(hass: HomeAssistant, config_entry: ShellyConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    """Set up sensors for device."""
    if get_device_entry_gen(config_entry) in RPC_GENERATIONS:
        if config_entry.data[CONF_SLEEP_PERIOD]:
            async_setup_entry_rpc(hass, config_entry, async_add_entities, RPC_SENSORS, RpcSleepingSensor)
        else:
            coordinator = config_entry.runtime_data.rpc
            assert coordinator
            async_setup_entry_rpc(hass, config_entry, async_add_entities, RPC_SENSORS, RpcSensor)
            async_remove_orphaned_entities(hass, config_entry.entry_id, coordinator.mac, SENSOR_PLATFORM, coordinator.device.status)
            virtual_component_ids = get_virtual_component_ids(coordinator.device.config, SENSOR_PLATFORM)
            for component in ('enum', 'number', 'text'):
                async_remove_orphaned_entities(hass, config_entry.entry_id, coordinator.mac, SENSOR_PLATFORM, virtual_component_ids, component)
        return
    if config_entry.data[CONF_SLEEP_PERIOD]:
        async_setup_entry_attribute_entities(hass, config_entry, async_add_entities, SENSORS, BlockSleepingSensor)
    else:
        async_setup_entry_attribute_entities(hass, config_entry, async_add_entities, SENSORS, BlockSensor)
        async_setup_entry_rest(hass, config_entry, async_add_entities, REST_SENSORS, RestSensor)

class BlockSensor(ShellyBlockAttributeEntity, SensorEntity):
    """Represent a block sensor."""

    def __init__(self, coordinator: ShellyBlockCoordinator, block: Block, attribute: str, description: BlockSensorDescription) -> None:
        """Initialize sensor."""
        super().__init__(coordinator, block, attribute, description)
        self._attr_native_unit_of_measurement = description.native_unit_of_measurement

    @property
    def native_value(self) -> StateType:
        """Return value of sensor."""
        return self.attribute_value

class RestSensor(ShellyRestAttributeEntity, SensorEntity):
    """Represent a REST sensor."""

    @property
    def native_value(self) -> StateType:
        """Return value of sensor."""
        return self.attribute_value

class BlockSleepingSensor(ShellySleepingBlockAttributeEntity, RestoreSensor):
    """Represent a block sleeping sensor."""

    def __init__(self, coordinator: ShellyBlockCoordinator, block: Block, attribute: str, description: BlockSensorDescription, entry: RegistryEntry | None = None) -> None:
        """Initialize the sleeping sensor."""
        super().__init__(coordinator, block, attribute, description, entry)
        self.restored_data: SensorExtraStoredData | None = None

    async def async_added_to_hass(self) -> None:
        """Handle entity which will be added."""
        await super().async_added_to_hass()
        self.restored_data = await self.async_get_last_sensor_data()

    @property
    def native_value(self) -> StateType | None:
        """Return value of sensor."""
        if self.block is not None:
            return self.attribute_value
        if self.restored_data is None:
            return None
        return cast(StateType, self.restored_data.native_value)

    @property
    def native_unit_of_measurement(self) -> str | None:
        """Return the unit of measurement of the sensor, if any."""
        if self.block is not None:
            return self.entity_description.native_unit_of_measurement
        if self.restored_data is None:
            return None
        return self.restored_data.native_unit_of_measurement

class RpcSleepingSensor(ShellySleepingRpcAttributeEntity, RestoreSensor):
    """Represent a RPC sleeping sensor."""

    def __init__(self, coordinator: ShellyRpcCoordinator, key: str, attribute: str, description: RpcSensorDescription, entry: RegistryEntry | None = None) -> None:
        """Initialize the sleeping sensor."""
        super().__init__(coordinator, key, attribute, description, entry)
        self.restored_data: SensorExtraStoredData | None = None

    async def async_added_to_hass(self) -> None:
        """Handle entity which will be added."""
        await super().async_added_to_hass()
        self.restored_data = await self.async_get_last_sensor_data()

    @property
    def native_value(self) -> StateType | None:
        """Return value of sensor."""
        if self.coordinator.device.initialized:
            return self.attribute_value
        if self.restored_data is None:
            return None
        return cast(StateType, self.restored_data.native_value)

    @property
    def native_unit_of_measurement(self) -> str:
        """Return the unit of measurement of the sensor, if any."""
        return self.entity_description.native_unit_of_measurement
