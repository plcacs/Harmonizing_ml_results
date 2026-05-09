import random
import gevent
import pytest
from eth_utils import decode_hex, encode_hex, to_canonical_address
from gevent.greenlet import Greenlet
from gevent.queue import Queue
from raiden.constants import BLOCK_ID_LATEST, EMPTY_BALANCE_HASH, EMPTY_HASH, EMPTY_SIGNATURE, GENESIS_BLOCK_NUMBER, LOCKSROOT_OF_NO_LOCKS, STATE_PRUNING_AFTER_BLOCKS
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
from raiden.utils.typing import Set, T_ChannelID
from raiden_contracts.constants import TEST_SETTLE_TIMEOUT_MAX, TEST_SETTLE_TIMEOUT_MIN, MessageTypeId

def test_token_network_deposit_race(token_network_proxy: TokenNetwork, private_keys: List[bytes], token_proxy: TokenNetwork, web3: JSONRPCClient, contract_manager: ProxyManager) -> None:
    ...

def test_token_network_proxy(token_network_proxy: TokenNetwork, private_keys: List[bytes], token_proxy: TokenNetwork, chain_id: int, web3: JSONRPCClient, contract_manager: ProxyManager) -> None:
    ...

def test_token_network_proxy_update_transfer(token_network_proxy: TokenNetwork, private_keys: List[bytes], token_proxy: TokenNetwork, chain_id: int, web3: JSONRPCClient, contract_manager: ProxyManager) -> None:
    ...

def test_query_pruned_state(token_network_proxy: TokenNetwork, private_keys: List[bytes], web3: JSONRPCClient, contract_manager: ProxyManager) -> None:
    ...

def test_token_network_actions_at_pruned_blocks(token_network_proxy: TokenNetwork, private_keys: List[bytes], token_proxy: TokenNetwork, web3: JSONRPCClient, chain_id: int, contract_manager: ProxyManager) -> None:
    ...

def test_concurrent_set_total_deposit(token_network_proxy: TokenNetwork) -> None:
    ...