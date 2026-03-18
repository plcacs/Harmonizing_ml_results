from typing import Any, Dict, List
from raiden_contracts.contract_manager import ContractManager
from raiden.network.proxies.proxy_manager import ProxyManager
from raiden.utils.typing import Address, BlockIdentifier, BlockNumber, ChannelID, TokenNetworkAddress

def get_netting_channel_closed_events(
    proxy_manager: ProxyManager,
    token_network_address: TokenNetworkAddress,
    netting_channel_identifier: ChannelID,
    contract_manager: ContractManager,
    from_block: BlockNumber = ...,
    to_block: BlockIdentifier = ...,
) -> List[Dict[str, Any]]: ...

def get_netting_channel_deposit_events(
    proxy_manager: ProxyManager,
    token_network_address: TokenNetworkAddress,
    netting_channel_identifier: ChannelID,
    contract_manager: ContractManager,
    from_block: BlockNumber = ...,
    to_block: BlockIdentifier = ...,
) -> List[Dict[str, Any]]: ...

def get_netting_channel_settled_events(
    proxy_manager: ProxyManager,
    token_network_address: TokenNetworkAddress,
    netting_channel_identifier: ChannelID,
    contract_manager: ContractManager,
    from_block: BlockNumber = ...,
    to_block: BlockIdentifier = ...,
) -> List[Dict[str, Any]]: ...

def wait_both_channel_open(
    app0: Any,
    app1: Any,
    registry_address: Address,
    token_address: Address,
    retry_timeout: Any,
) -> None: ...

def test_channel_new(raiden_chain: Any, retry_timeout: Any, token_addresses: Any) -> None: ...

def test_channel_deposit(raiden_chain: Any, deposit: Any, retry_timeout: Any, token_addresses: Any) -> None: ...

def test_query_events(
    raiden_chain: Any,
    token_addresses: Any,
    deposit: Any,
    settle_timeout: Any,
    retry_timeout: Any,
    contract_manager: Any,
    blockchain_type: Any,
) -> None: ...

def test_secret_revealed_on_chain(
    raiden_chain: Any,
    deposit: Any,
    settle_timeout: Any,
    token_addresses: Any,
    retry_interval_initial: Any,
) -> None: ...

def test_clear_closed_queue(raiden_network: Any, token_addresses: Any, network_wait: Any) -> None: ...