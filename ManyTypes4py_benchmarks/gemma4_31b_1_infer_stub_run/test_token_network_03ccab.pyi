import pytest
from typing import Any, Optional, Union, Set
from raiden.utils.typing import T_ChannelID
from raiden.network.proxies.token_network import TokenNetwork
from raiden.network.rpc.client import JSONRPCClient
from raiden.network.proxies.proxy_manager import ProxyManager

SIGNATURE_SIZE_IN_BITS: int = ...

def test_token_network_deposit_race(
    token_network_proxy: TokenNetwork,
    private_keys: list[str],
    token_proxy: Any,
    web3: Any,
    contract_manager: Any
) -> None: ...

def test_token_network_proxy(
    token_network_proxy: TokenNetwork,
    private_keys: list[str],
    token_proxy: Any,
    chain_id: int,
    web3: Any,
    contract_manager: Any
) -> None: ...

def test_token_network_proxy_update_transfer(
    token_network_proxy: TokenNetwork,
    private_keys: list[str],
    token_proxy: Any,
    chain_id: int,
    web3: Any,
    contract_manager: Any
) -> None: ...

def test_query_pruned_state(
    token_network_proxy: TokenNetwork,
    private_keys: list[str],
    web3: Any,
    contract_manager: Any
) -> None: ...

def test_token_network_actions_at_pruned_blocks(
    token_network_proxy: TokenNetwork,
    private_keys: list[str],
    token_proxy: Any,
    web3: Any,
    chain_id: int,
    contract_manager: Any
) -> None: ...

def test_concurrent_set_total_deposit(token_network_proxy: TokenNetwork) -> None: ...