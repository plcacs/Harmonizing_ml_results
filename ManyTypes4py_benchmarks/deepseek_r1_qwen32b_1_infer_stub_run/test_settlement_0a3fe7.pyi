import gevent
from eth_utils import keccak
from raiden.settings import DEFAULT_RETRY_TIMEOUT
from raiden.tests.utils.protocol import HoldRaidenEventHandler, WaitForMessage
from raiden.transfer.state import NettingChannelState
from raiden.utils.typing import (
    Address,
    Balance,
    BlockNumber,
    BlockTimeout as BlockOffset,
    List,
    MessageID,
    PaymentAmount,
    PaymentID,
    Secret,
    SecretRegistryAddress,
    TargetAddress,
    TokenAddress,
    TokenAmount,
    TokenNetworkAddress,
    WithdrawAmount,
)
from raiden.raiden_service import RaidenService
from raiden.transfer.identifiers import CanonicalIdentifier
from raiden.tests.utils.client import burn_eth
from raiden.tests.utils.detect_failure import expect_failure, raise_on_failure
from raiden.tests.utils.events import raiden_state_changes_search_for_item, search_for_item
from raiden.tests.utils.network import CHAIN
from raiden.tests.utils.transfer import assert_synced_channel_state, block_offset_timeout, create_route_state_for_route, get_channelstate, transfer
from typing import Any, Callable, Optional

def wait_for_batch_unlock(app: RaidenService, token_network_address: TokenNetworkAddress, receiver: Address, sender: Address) -> None:
    ...

def is_channel_registered(node_app: RaidenService, partner_app: RaidenService, canonical_identifier: CanonicalIdentifier) -> bool:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
def test_settle_is_automatically_called(raiden_network: List[RaidenService], token_addresses: List[TokenAddress]) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
def test_coop_settle_is_automatically_called(raiden_network: List[RaidenService], token_addresses: List[TokenAddress]) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
def test_coop_settle_fails_with_pending_lock(raiden_network: List[RaidenService], token_addresses: List[TokenAddress]) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
def test_lock_expiry(raiden_network: List[RaidenService], token_addresses: List[TokenAddress], deposit: int) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
def test_batch_unlock(raiden_network: List[RaidenService], secret_registry_address: SecretRegistryAddress, token_addresses: List[TokenAddress], deposit: int) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
def test_register_secret(raiden_network: List[RaidenService], token_addresses: List[TokenAddress], secret_registry_address: SecretRegistryAddress) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
def test_channel_withdraw(raiden_network: List[RaidenService], token_addresses: List[TokenAddress], deposit: int, retry_timeout: BlockOffset, pfs_mock: Any) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
def test_channel_withdraw_expired(raiden_network: List[RaidenService], network_wait: BlockOffset, number_of_nodes: int, token_addresses: List[TokenAddress], deposit: int, retry_timeout: BlockOffset, pfs_mock: Any) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('channels_per_node', [CHAIN])
def test_settled_lock(token_addresses: List[TokenAddress], raiden_network: List[RaidenService], deposit: int, retry_timeout: BlockOffset) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('channels_per_node', [1])
def test_automatic_secret_registration(raiden_chain: List[RaidenService], token_addresses: List[TokenAddress]) -> None:
    ...

@raise_on_failure
@pytest.mark.xfail(reason='test incomplete')
@pytest.mark.parametrize('number_of_nodes', [3])
def test_start_end_attack(token_addresses: List[TokenAddress], raiden_chain: List[RaidenService], deposit: int) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
def test_automatic_dispute(raiden_network: List[RaidenService], deposit: int, token_addresses: List[TokenAddress]) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
def test_batch_unlock_after_restart(raiden_network: List[RaidenService], restart_node: Callable[[RaidenService], None], token_addresses: List[TokenAddress], deposit: int) -> None:
    ...

@expect_failure
@pytest.mark.parametrize('number_of_nodes', (2,))
@pytest.mark.parametrize('channels_per_node', (1,))
def test_handle_insufficient_eth(raiden_network: List[RaidenService], restart_node: Callable[[RaidenService], None], token_addresses: List[TokenAddress], caplog: Any) -> None:
    ...