def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    ...

class FinTsClient:
    def __init__(self, credentials: BankCredentials, name: str, account_config: dict[str, str], holdings_config: dict[str, str]) -> None:
        ...

    @cached_property
    def client(self) -> FinTS3PinTanClient:
        ...

    def get_account_information(self, iban: str) -> dict[str, Any]:
        ...

    def is_balance_account(self, account: SEPAAccount) -> bool:
        ...

    def is_holdings_account(self, account: SEPAAccount) -> bool:
        ...

    def detect_accounts(self) -> tuple[list[SEPAAccount], list[SEPAAccount]]:
        ...

class FinTsAccount(SensorEntity):
    def __init__(self, client: FinTsClient, account: SEPAAccount, name: str) -> None:
        ...

    def update(self) -> None:
        ...

class FinTsHoldingsAccount(SensorEntity):
    def __init__(self, client: FinTsClient, account: SEPAAccount, name: str) -> None:
        ...

    def update(self) -> None:
        ...

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        ...
