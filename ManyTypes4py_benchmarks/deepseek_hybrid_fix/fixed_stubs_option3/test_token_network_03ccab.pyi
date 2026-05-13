from __future__ import annotations
import random
import gevent
import pytest
from eth_utils import decode_hex, encode_hex, to_canonical_address
from gevent.greenlet import Greenlet
from gevent.queue import Queue
from raiden.constants import BLOCK_ID_LATEST, EMPTY_BALANCE_HASH, EMPTY_HASH, EMPTY_SIGNATURE, GENESIS_BLOCK_NUMBER, LOCKSROOT_OF_NO_LOCKS, STATE_PRUNING_AFTER_BLOCKS, BlockIdentifier
from raiden.exceptions import BrokenPreconditionError, InvalidChannelID, InvalidSettleTimeout, RaidenRecoverableError, RaidenUnrecoverableError, SamePeerAddress
from raiden.network.proxies.proxy_manager import ProxyManager, ProxyManagerMetadata
from raiden.network.proxies.token_network import TokenNetwork
from raiden.network.rpc.client import JSONRPCClient
from raiden.tests.integration.network.proxies import BalanceProof
from raiden.tests.utils import factories
from raiden.tests.utils.factories import make_address
from raiden.tests.utils.smartcontracts import is_tx_hash_bytes
from raiden.utils.formatting import to_hex_address
from raiden.utils.signer import LocalSigner
from raiden.utils.typing import Set, T_ChannelID, Address, BalanceHash, BlockHash, BlockNumber, ChainID, ChannelID, Locksroot, Nonce, Signature, TokenAmount, TransactionHash
from raiden_contracts.constants import TEST_SETTLE_TIMEOUT_MAX, TEST_SETTLE_TIMEOUT_MIN, MessageTypeId
from _pytest.python import Function
from eth_utils.typing import HexStr
from raiden_contracts.contract_manager import ContractManager
from typing import Any
from web3 import Web3

class _TokenProxy:
    def transfer(self, address: Address, amount: int) -> Any: ...
    def balance_of(self, address: Address) -> int: ...

SIGNATURE_SIZE_IN_BITS: int = 520

def test_token_network_deposit_race(token_network_proxy: TokenNetwork, private_keys: list[bytes], token_proxy: _TokenProxy, web3: Web3, contract_manager: ContractManager) -> None: ...
def test_token_network_proxy(token_network_proxy: TokenNetwork, private_keys: list[bytes], token_proxy: _TokenProxy, chain_id: ChainID, web3: Web3, contract_manager: ContractManager) -> None: ...
def test_token_network_proxy_update_transfer(token_network_proxy: TokenNetwork, private_keys: list[bytes], token_proxy: _TokenProxy, chain_id: ChainID, web3: Web3, contract_manager: ContractManager) -> None: ...
def test_query_pruned_state(token_network_proxy: TokenNetwork, private_keys: list[bytes], web3: Web3, contract_manager: ContractManager) -> None: ...
def test_token_network_actions_at_pruned_blocks(token_network_proxy: TokenNetwork, private_keys: list[bytes], token_proxy: _TokenProxy, web3: Web3, chain_id: ChainID, contract_manager: ContractManager) -> None: ...
def test_concurrent_set_total_deposit(token_network_proxy: TokenNetwork) -> None: ...