def validate_topic_required(config: dict) -> dict:
    ...

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class MqttImage(MqttEntity, ImageEntity):
    def __init__(self, hass: HomeAssistant, config: ConfigType, config_entry: ConfigEntry, discovery_data: DiscoveryInfoType) -> None:
        ...

    @staticmethod
    def config_schema() -> VolSchemaType:
        ...

    def _setup_from_config(self, config: dict) -> None:
        ...

    @callback
    def _image_data_received(self, msg: ReceiveMessage) -> None:
        ...

    @callback
    def _image_from_url_request_received(self, msg: ReceiveMessage) -> None:
        ...

    @callback
    def _prepare_subscribe_topics(self) -> None:
        ...

    async def _subscribe_topics(self) -> None:
        ...

    async def async_image(self) -> bytes:
        ...
