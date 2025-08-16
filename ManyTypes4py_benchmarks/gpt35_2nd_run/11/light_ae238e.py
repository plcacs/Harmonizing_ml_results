def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    ...

class GreenwaveLight(LightEntity):
    _attr_color_mode: ColorMode = ColorMode.BRIGHTNESS
    _attr_supported_color_modes: set[ColorMode] = {ColorMode.BRIGHTNESS}

    def __init__(self, light: dict[str, Any], host: str, token: str, gatewaydata: GatewayData) -> None:
        ...

    @property
    def is_on(self) -> bool:
        ...

    def turn_on(self, **kwargs: Any) -> None:
        ...

    def turn_off(self, **kwargs: Any) -> None:
        ...

    def update(self) -> None:
        ...

class GatewayData:
    def __init__(self, host: str, token: str) -> None:
        ...

    @property
    def greenwave(self) -> dict[str, Any]:
        ...

    @Throttle(MIN_TIME_BETWEEN_UPDATES)
    def update(self) -> dict[str, Any]:
        ...
