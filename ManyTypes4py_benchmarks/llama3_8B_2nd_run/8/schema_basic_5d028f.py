class MqttLight(MqttEntity, LightEntity, RestoreEntity):
    """Representation of a MQTT light."""
    _default_name: str = DEFAULT_NAME
    _entity_id_format: str = ENTITY_ID_FORMAT
    _attributes_extra_blocked: frozenset[str] = MQTT_LIGHT_ATTRIBUTES_BLOCKED

    @staticmethod
    def config_schema() -> vol.Schema:
        """Return the config schema."""
        return DISCOVERY_SCHEMA_BASIC

    def _setup_from_config(self, config: ConfigType) -> None:
        """(Re)Setup the entity."""
        # ...

    # ...

    @callback
    def _state_received(self, msg: ReceiveMessage) -> None:
        """Handle new MQTT messages."""
        # ...

    @callback
    def _brightness_received(self, msg: ReceiveMessage) -> None:
        """Handle new MQTT messages for the brightness."""
        # ...

    @callback
    def _rgbx_received(self, msg: ReceiveMessage, template: str, color_mode: ColorMode, convert_color: Callable[[int, int, int], tuple[int, int, int]]) -> tuple[int, int, int]:
        """Process MQTT messages for RGBW and RGBWW."""
        # ...

    # ...

    async def _subscribe_topics(self) -> None:
        """(Re)Subscribe to topics."""
        # ...

    async def async_turn_on(self, **kwargs: dict[str, Any]) -> None:
        """Turn the device on.

        This method is a coroutine.
        """
        # ...

    async def async_turn_off(self, **kwargs: dict[str, Any]) -> None:
        """Turn the device off.

        This method is a coroutine.
        """
        # ...
