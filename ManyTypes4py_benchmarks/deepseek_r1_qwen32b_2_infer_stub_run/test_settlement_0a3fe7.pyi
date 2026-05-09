import gevent
import pytest
from eth_utils import Secret
from gevent import Timeout
from raiden.api.python import RaidenAPI
from raiden.raiden_service import RaidenService
from raiden.settings import DEFAULT_RETRY_TIMEOUT
from raiden.tests.utils.protocol import HoldRaidenEventHandler, WaitForMessage
from raiden.tests.utils.transfer import TransferDescription, TransferResult
from raiden.transfer.state import CanonicalIdentifier, NettingChannelState
from raiden.utils.typing import (
    Address,
    BlockNumber,
    BlockTimeout,
    List,
    MessageID,
    PaymentAmount,
    PaymentID,
    Secret,
    SecretRegistryAddress,
    TargetAddress,
    TokenAddress,
    TokenAmount,
    WithdrawAmount,
)

def wait_for_batch_unlock(app: RaidenService, token_network_address: TokenAddress, receiver: Address, sender: Address) -> None: ...

def is_channel_registered(node_app: RaidenService, partner_app: RaidenService, canonical_identifier: CanonicalIdentifier) -> bool: ...

@pytest.mark.parametrize('number_of_nodes', [2])
def test_settle_is_automatically_called(raiden_network: List[RaidenService], token_addresses: List[TokenAddress]) -> None: ...

@pytest.mark.flaky
@pytest.mark.parametrize('number_of_nodes', [2])
def test_coop_settle_is_automatically_called(raiden_network: List[RaidenService], token_addresses: List[TokenAddress]) -> None: ...

@pytest.mark.parametrize('number_of_nodes', [2])
def test_coop_settle_fails_with_pending_lock(raiden_network: List[RaidenService], token_addresses: List[TokenAddress]) -> None: ...

@pytest.mark.parametrize('number_of_nodes', [2])
def test_lock_expiry(raiden_network: List[RaidenService], token_addresses: List[TokenAddress], deposit: TokenAmount) -> None: ...

@pytest.mark.parametrize('number_of_nodes', [2])
def test_batch_unlock(raiden_network: List[RaidenService], token_addresses: List[TokenAddress], secret_registry_address: SecretRegistryAddress, deposit: TokenAmount) -> None: ...

@pytest.mark.parametrize('number_of_nodes', [2])
def test_register_secret(raiden_network: List[RaidenService], token_addresses: List[TokenAddress], secret_registry_address: SecretRegistryAddress) -> None: ...

@pytest.mark.parametrize('number_of_nodes', [2])
def test_channel_withdraw(raiden_network: List[RaidenService], token_addresses: List[TokenAddress], deposit: TokenAmount, retry_timeout: BlockTimeout, pfs_mock: pytest.fixture) -> None: ...

@pytest.mark.parametrize('number_of_nodes', [2])
def test_channel_withdraw_expired(raiden_network: List[RaidenService], network_wait: BlockTimeout, number_of_nodes: int, token_addresses: List[TokenAddress], deposit: TokenAmount, retry_timeout: BlockTimeout, pfs_mock: pytest.fixture) -> None: ...

@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('channels_per_node', [CHAIN])
def test_settled_lock(token_addresses: List[TokenAddress], raiden_network: List[RaidenService], deposit: TokenAmount, retry_timeout: BlockTimeout) -> None: ...

@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('channels_per_node', [1])
def test_automatic_secret_registration(raiden_chain: List[RaidenService], token_addresses: List[TokenAddress]) -> None: ...

@pytest.mark.xfail(reason='test incomplete')
@pytest.mark.parametrize('number_of_nodes', [3])
def test_start_end_attack(token_addresses: List[TokenAddress], raiden_chain: List[RaidenService], deposit: TokenAmount) -> None: ...

@pytest.mark.parametrize('number_of_nodes', [2])
def test_automatic_dispute(raiden_network: List[RaidenService], deposit: TokenAmount, token_addresses: List[TokenAddress]) -> None: ...

@pytest.mark.parametrize('number_of_nodes', [2])
def test_batch_unlock_after_restart(raiden_network: List[RaidenService], restart_node: pytest.fixture, token_addresses: List[TokenAddress], deposit: TokenAmount) -> None: ...

@expect_failure
@pytest.mark.parametrize('number_of_nodes', (2,))
@pytest.mark.parametrize('channels_per_node', (1,))
def test_handle_insufficient_eth(raiden_network: List[RaidenService], restart_node: pytest.fixture, token_addresses: List[TokenAddress], caplog: pytest.fixture) -> None: ...