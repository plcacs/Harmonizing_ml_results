async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class OctoprintPrinterButton(CoordinatorEntity[OctoprintDataUpdateCoordinator], ButtonEntity):
    """Represent an OctoPrint binary sensor."""

    def __init__(self, coordinator: OctoprintDataUpdateCoordinator, button_type: str, device_id: str, client: OctoprintClient) -> None:
        ...

class OctoprintSystemButton(CoordinatorEntity[OctoprintDataUpdateCoordinator], ButtonEntity):
    """Represent an OctoPrint binary sensor."""

    def __init__(self, coordinator: OctoprintDataUpdateCoordinator, button_type: str, device_id: str, client: OctoprintClient) -> None:
        ...

class OctoprintPauseJobButton(OctoprintPrinterButton):
    """Pause the active job."""

    def __init__(self, coordinator: OctoprintDataUpdateCoordinator, device_id: str, client: OctoprintClient) -> None:
        ...

    async def async_press(self) -> None:
        ...

class OctoprintResumeJobButton(OctoprintPrinterButton):
    """Resume the active job."""

    def __init__(self, coordinator: OctoprintDataUpdateCoordinator, device_id: str, client: OctoprintClient) -> None:
        ...

    async def async_press(self) -> None:
        ...

class OctoprintStopJobButton(OctoprintPrinterButton):
    """Resume the active job."""

    def __init__(self, coordinator: OctoprintDataUpdateCoordinator, device_id: str, client: OctoprintClient) -> None:
        ...

    async def async_press(self) -> None:
        ...

class OctoprintShutdownSystemButton(OctoprintSystemButton):
    """Shutdown the system."""

    def __init__(self, coordinator: OctoprintDataUpdateCoordinator, device_id: str, client: OctoprintClient) -> None:
        ...

    async def async_press(self) -> None:
        ...

class OctoprintRebootSystemButton(OctoprintSystemButton):
    """Reboot the system."""

    _attr_device_class: ButtonDeviceClass = ButtonDeviceClass.RESTART

    def __init__(self, coordinator: OctoprintDataUpdateCoordinator, device_id: str, client: OctoprintClient) -> None:
        ...

    async def async_press(self) -> None:
        ...

class OctoprintRestartOctoprintButton(OctoprintSystemButton):
    """Restart Octoprint."""

    _attr_device_class: ButtonDeviceClass = ButtonDeviceClass.RESTART

    def __init__(self, coordinator: OctoprintDataUpdateCoordinator, device_id: str, client: OctoprintClient) -> None:
        ...

    async def async_press(self) -> None:
        ...

class InvalidPrinterState(HomeAssistantError):
    """Service attempted in invalid state."""

    def __init__(self, message: str) -> None:
        ...
