from raiden.api.python import RaidenAPI
from raiden.constants import BLOCK_ID_LATEST, UINT256_MAX
from raiden.network.proxies.token_network import TokenNetwork
from raiden.raiden_service import RaidenService
from raiden.settings import DEFAULT_RETRY_TIMEOUT
from raiden.utils.typing import Address, AddressHex, Any, BlockTimeout, Dict, NetworkTimeout, TokenAddress, TokenAmount, TokenNetworkRegistryAddress
from raiden_contracts.constants import CONTRACT_HUMAN_STANDARD_TOKEN

def print_usage() -> None:
    ...

class Console(gevent.Greenlet):
    def __init__(self, raiden_service: RaidenService) -> None:
        ...

    def _run(self) -> None:
        ...

class ConsoleTools:
    def __init__(self, raiden_service: RaidenService) -> None:
        ...

    def create_token(self, registry_address_hex: str, initial_alloc: int = 10 ** 6, name: str = 'raidentester', symbol: str = 'RDT', decimals: int = 2, timeout: int = 60, auto_register: bool = True) -> str:
        ...

    def register_token(self, registry_address_hex: str, token_address_hex: str, retry_timeout: int = DEFAULT_RETRY_TIMEOUT) -> TokenNetwork:
        ...

    def open_channel_with_funding(self, registry_address_hex: str, token_address_hex: str, peer_address_hex: str, total_deposit: int, settle_timeout: int = None) -> None:
        ...

    def wait_for_contract(self, contract_address_hex: str, timeout: int = None) -> bool:
        ...
