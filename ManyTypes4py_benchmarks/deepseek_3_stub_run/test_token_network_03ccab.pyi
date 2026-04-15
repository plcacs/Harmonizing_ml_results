import pytest
from _pytest.python import Function
from eth_utils.typing import HexStr
from gevent.greenlet import Greenlet
from gevent.queue import Queue
from raiden.constants import BlockIdentifier
from raiden.exceptions import (
    BrokenPreconditionError,
    InvalidChannelID,
    InvalidSettleTimeout,
    RaidenRecoverableError,
    RaidenUnrecoverableError,
    SamePeerAddress,
)
from raiden.network.proxies.proxy_manager import ProxyManager, ProxyManagerMetadata
from raiden.network.proxies.token_network import TokenNetwork
from raiden.network.rpc.client import JSONRPCClient
from raiden.tests.integration.network.proxies import BalanceProof
from raiden.tests.utils.factories import make_address
from raiden.utils.formatting import to_hex_address
from raiden.utils.signer import LocalSigner
from raiden.utils.typing import (
    Address,
    BalanceHash,
    BlockHash,
    BlockNumber,
    ChainID,
    ChannelID,
    Locksroot,
    Nonce,
    Signature,
    TokenAmount,
    TransactionHash,
    T_ChannelID,
)
from raiden_contracts.contract_manager import ContractManager
from web3 import Web3

SIGNATURE_SIZE_IN_BITS: int = ...

def test_token_network_deposit_race(
    token_network_proxy: TokenNetwork,
    private_keys: list[bytes],
    token_proxy: object,
    web3: Web3,
    contract_manager: ContractManager,
) -> None: ...

def test_token_network_proxy(
    token_network_proxy: TokenNetwork,
    private_keys: list[bytes],
    token_proxy: object,
    chain_id: ChainID,
    web3: Web3,
    contract_manager: ContractManager,
) -> None: ...

def test_token_network_proxy_update_transfer(
    token_network_proxy: TokenNetwork,
    private_keys: list[bytes],
    token_proxy: object,
    chain_id: ChainID,
    web3: Web3,
    contract_manager: ContractManager,
) -> None: ...

def test_query_pruned_state(
    token_network_proxy: TokenNetwork,
    private_keys: list[bytes],
    web3: Web3,
    contract_manager: ContractManager,
) -> None: ...

def test_token_network_actions_at_pruned_blocks(
    token_network_proxy: TokenNetwork,
    private_keys: list[bytes],
    token_proxy: object,
    web3: Web3,
    chain_id: ChainID,
    contract_manager: ContractManager,
) -> None: ...

def test_concurrent_set_total_deposit(
    token_network_proxy: TokenNetwork,
) -> None: ...