class AlexaDirective:
    """An incoming Alexa directive."""

    def __init__(self, request: MappingProxyType) -> None:
        """Initialize a directive."""
        # ...

    def load_entity(self, hass: HomeAssistant, config: AbstractConfig) -> None:
        """Set attributes related to the entity for this request."""
        # ...

    def response(self, name: str, namespace: str, payload: Any = None) -> AlexaResponse:
        """Create an API formatted response."""
        # ...

    def error(self, namespace: str, error_type: str, error_message: str, payload: Any = None) -> AlexaResponse:
        """Create a API formatted error response."""
        # ...

class AlexaResponse:
    """Class to hold a response."""

    def __init__(self, name: str, namespace: str, payload: Any = None) -> None:
        """Initialize the response."""
        # ...

    @property
    def name(self) -> str:
        """Return the name of this response."""
        # ...

    @property
    def namespace(self) -> str:
        """Return the namespace of this response."""
        # ...

    def set_correlation_token(self, token: str) -> None:
        """Set the correlationToken."""
        # ...

    def set_endpoint_full(self, bearer_token: str, endpoint_id: str) -> None:
        """Set the endpoint dictionary."""
        # ...

    def set_endpoint(self, endpoint: Any) -> None:
        """Set the endpoint."""
        # ...

    def _properties(self) -> list:
        """Return the context properties."""
        # ...

    def add_context_property(self, prop: Any) -> None:
        """Add a property to the response context."""
        # ...

    def merge_context_properties(self, endpoint: Any) -> None:
        """Add all properties from given endpoint if not already set."""
        # ...

    def serialize(self) -> dict:
        """Return response as a JSON-able data structure."""
        # ...

async def async_enable_proactive_mode(hass: HomeAssistant, smart_home_config: AbstractConfig) -> None:
    """Enable the proactive mode."""
    # ...

async def async_send_changereport_message(hass: HomeAssistant, config: AbstractConfig, alexa_entity: AlexaEntity, alexa_properties: list, *, invalidate_access_token: bool = True) -> None:
    """Send a ChangeReport message for an Alexa entity."""
    # ...

async def async_send_add_or_update_message(hass: HomeAssistant, config: AbstractConfig, entity_ids: list) -> None:
    """Send an AddOrUpdateReport message for entities."""
    # ...

async def async_send_delete_message(hass: HomeAssistant, config: AbstractConfig, entity_ids: list) -> None:
    """Send an DeleteReport message for entities."""
    # ...

async def async_send_doorbell_event_message(hass: HomeAssistant, config: AbstractConfig, alexa_entity: AlexaEntity) -> None:
    """Send a DoorbellPress event message for an Alexa entity."""
    # ...
