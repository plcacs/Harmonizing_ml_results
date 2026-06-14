from typing import Any, Dict, List

from raiden.constants import BLOCK_ID_LATEST, GENESIS_BLOCK_NUMBER
from raiden.network.proxies.proxy_manager import ProxyManager
from raiden.raiden_service import RaidenService
from raiden.utils.typing import (
    BlockIdentifier,
    ChannelID,
    TokenAddress,
    TokenNetworkAddress,
    TokenNetworkRegistryAddress,
    Balance,
)
from raiden_contracts.contract_manager import ContractManager


def get_netting_channel_closed_events(
    proxy_manager: ProxyManager,
    token_network_address: TokenNetworkAddress,
    netting_channel_identifier: ChannelID,
    contract_manager: ContractManager,
    from_block: BlockIdentifier = GENESIS_BLOCK_NUMBER,
    to_block: BlockIdentifier = BLOCK_ID_LATEST,
) -> List[Dict[str, Any]]: ...
def get_netting_channel_deposit_events(
    proxy_manager: ProxyManager,
    token_network_address: TokenNetworkAddress,
    netting_channel_identifier: ChannelID,
    contract_manager: ContractManager,
    from_block: BlockIdentifier = GENESIS_BLOCK_NUMBER,
    to_block: BlockIdentifier = BLOCK_ID_LATEST,
) -> List[Dict[str, Any]]: ...
def get_netting_channel_settled_events(
    proxy_manager: ProxyManager,
    token_network_address: TokenNetworkAddress,
    netting_channel_identifier: ChannelID,
    contract_manager: ContractManager,
    from_block: BlockIdentifier = GENESIS_BLOCK_NUMBER,
    to_block: BlockIdentifier = BLOCK_ID_LATEST,
) -> List[Dict[str, Any]]: ...
def wait_both_channel_open(
    app0: RaidenService,
    app1: RaidenService,
    registry_address: TokenNetworkRegistryAddress,
    token_address: TokenAddress,
    retry_timeout: float,
) -> None: ...
def test_channel_new(
    raiden_chain: List[RaidenService],
    retry_timeout: float,
    token_addresses: List[TokenAddress],
) -> None: ...
def test_channel_deposit(
    raiden_chain: List[RaidenService],
    deposit: Balance,
    retry_timeout: float,
    token_addresses: List[TokenAddress],
) -> None: ...
def test_query_events(
    raiden_chain: List[RaidenService],
    token_addresses: List[TokenAddress],
    deposit: Balance,
    settle_timeout: int,
    retry_timeout: float,
    contract_manager: ContractManager,
    blockchain_type: str,
) -> None: ...
def test_secret_revealed_on_chain(
    raiden_chain: List[RaidenService],
    deposit: Balance,
    settle_timeout: int,
    token_addresses: List[TokenAddress],
    retry_interval_initial: float,
) -> None: ...
def test_clear_closed_queue(
    raiden_network: List[RaidenService],
    token_addresses: List[TokenAddress],
    network_wait: float,
) -> None: ...