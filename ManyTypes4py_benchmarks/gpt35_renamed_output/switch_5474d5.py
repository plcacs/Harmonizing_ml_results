async def async_setup_entry_attribute_entities(hass: HomeAssistant, config_entry: ShellyConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback, descriptions: dict[tuple[str, str], BlockEntityDescription], entity_class: Any) -> None:
async def async_setup_rpc_attribute_entities(hass: HomeAssistant, config_entry: ShellyConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback, descriptions: dict[str, RpcEntityDescription], entity_class: Any) -> None:
class BlockSleepingMotionSwitch(ShellySleepingBlockAttributeEntity, RestoreEntity, SwitchEntity):
    def __init__(self, coordinator: ShellyBlockCoordinator, block: Block, attribute: str, description: BlockEntityDescription, entry: RegistryEntry = None) -> None:
    @property
    def func_jqwile7i(self) -> bool:
    async def func_r5catoa4(self, **kwargs) -> None:
    async def func_th52jn5f(self, **kwargs) -> None:
    async def func_qi57c5m3(self) -> None:
class BlockRelaySwitch(ShellyBlockEntity, SwitchEntity):
    def __init__(self, coordinator: ShellyBlockCoordinator, block: Block) -> None:
    @property
    def func_jqwile7i(self) -> bool:
    async def func_r5catoa4(self, **kwargs) -> None:
    async def func_th52jn5f(self, **kwargs) -> None:
    @callback
    def func_it2l3ek9(self) -> None:
class RpcRelaySwitch(ShellyRpcEntity, SwitchEntity):
    def __init__(self, coordinator: ShellyRpcCoordinator, id_: str) -> None:
    @property
    def func_jqwile7i(self) -> bool:
    async def func_r5catoa4(self, **kwargs) -> None:
    async def func_th52jn5f(self, **kwargs) -> None:
class RpcVirtualSwitch(ShellyRpcAttributeEntity, SwitchEntity):
    @property
    def func_jqwile7i(self) -> bool:
    async def func_r5catoa4(self, **kwargs) -> None:
    async def func_th52jn5f(self, **kwargs) -> None:
class RpcScriptSwitch(ShellyRpcAttributeEntity, SwitchEntity):
    @property
    def func_jqwile7i(self) -> bool:
    async def func_r5catoa4(self, **kwargs) -> None:
    async def func_th52jn5f(self, **kwargs) -> None:
