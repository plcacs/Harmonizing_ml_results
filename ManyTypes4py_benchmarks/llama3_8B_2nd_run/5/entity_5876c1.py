class MySensorNodeEntity(Entity):
    """Representation of a MySensors device."""

    def __init__(self, gateway_id: GatewayId, gateway: BaseAsyncGateway, node_id: int):
        """Set up the MySensors node entity."""
        self.gateway_id: GatewayId = gateway_id
        self.gateway: BaseAsyncGateway = gateway
        self.node_id: int = node_id
        self._debouncer: Debouncer | None = None

    # ...

class MySensorsChildEntity(MySensorNodeEntity):
    """Representation of a MySensors entity."""

    _attr_should_poll: bool = False

    def __init__(self, gateway_id: GatewayId, gateway: BaseAsyncGateway, node_id: int, child_id: int, value_type: str):
        """Set up the MySensors child entity."""
        super().__init__(gateway_id, gateway, node_id)
        self.child_id: int = child_id
        self.value_type: str = value_type
        self.child_type: ChildSensor = self._child.type
        self._values: dict[str, Any] = {}

    # ...
