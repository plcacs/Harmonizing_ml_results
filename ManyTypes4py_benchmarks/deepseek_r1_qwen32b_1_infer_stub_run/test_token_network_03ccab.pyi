from eth_utils import decode_hex, encode_hex, to_canonical_address
from gevent.greenlet import Greenlet
from raiden.network.proxies.proxy_manager import ProxyManager, ProxyManagerMetadata
from raiden.network.proxies.token_network import TokenNetwork
from raiden.network.rpc.client import JSONRPCClient
from raiden.tests.integration.network.proxies import BalanceProof
from raiden.tests.utils.factories import make_address
from raiden.utils.typing import Set, T_ChannelID
from web3 import Web3
from raiden_contracts.contract_manager import ContractManager

def test_token_network_deposit_race(
    token_network_proxy: TokenNetwork,
    private_keys: list[bytes],
    token_proxy: JSONRPCClient,
    web3: Web3,
    contract_manager: ContractManager,
) -> None:
    ...

def test_token_network_proxy(
    token_network_proxy: TokenNetwork,
    private_keys: list[bytes],
    token_proxy: JSONRPCClient,
    chain_id: int,
    web3: Web3,
    contract_manager: ContractManager,
) -> None:
    ...

def test_token_network_proxy_update_transfer(
    token_network_proxy: TokenNetwork,
    private_keys: list[bytes],
    token_proxy: JSONRPCClient,
    chain_id: int,
    web3: Web3,
    contract_manager: ContractManager,
) -> None:
    ...

def test_query_pruned_state(
    token_network_proxy: TokenNetwork,
    private_keys: list[bytes],
    web3: Web3,
    contract_manager: ContractManager,
) -> None:
    ...

def test_token_network_actions_at_pruned_blocks(
    token_network_proxy: TokenNetwork,
    private_keys: list[bytes],
    token_proxy: JSONRPCClient,
    web3: Web3,
    chain_id: int,
    contract_manager: ContractManager,
) -> None:
    ...

def test_concurrent_set_total_deposit(token_network_proxy: TokenNetwork) -> None:
    ...