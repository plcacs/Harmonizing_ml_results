import pytest
from eth_utils import keccak
from raiden.api.python import RaidenAPI
from raiden.constants import DeviceIDs, RoutingMode
from raiden.messages.abstract import Message
from raiden.messages.decode import balanceproof_from_envelope
from raiden.messages.path_finding_service import PFSCapacityUpdate, PFSFeeUpdate
from raiden.messages.transfers import Unlock
from raiden.raiden_service import RaidenService
from raiden.settings import MediationFeeConfig
from raiden.tests.utils.detect_failure import raise_on_failure
from raiden.tests.utils.factories import make_transaction_hash
from raiden.tests.utils.network import CHAIN
from raiden.tests.utils.transfer import assert_succeeding_transfer_invariants, block_timeout_for_transfer_by_secrethash, transfer, wait_assert
from raiden.tests.utils.transport import TestMatrixTransport
from raiden.transfer import views
from raiden.transfer.state import TransactionChannelDeposit
from raiden.transfer.state_change import ContractReceiveChannelDeposit, ReceiveUnlock
from raiden.utils.typing import List, LockedAmount, Locksroot, MessageID, Nonce, PaymentAmount, PaymentID, Secret, Signature, TokenAddress, TokenAmount, WithdrawAmount
from raiden.waiting import wait_for_block

def get_messages(app: RaidenAPI) -> List[Message]:
    assert isinstance(app.transport, TestMatrixTransport), 'Transport is not a `TestMatrixTransport`'
    return app.transport.broadcast_messages[DeviceIDs.PFS.value]

def reset_messages(app: RaidenAPI) -> None:
    assert isinstance(app.transport, TestMatrixTransport), 'Transport is not a `TestMatrixTransport`'
    app.transport.broadcast_messages[DeviceIDs.PFS.value] = []

def wait_all_apps(raiden_network: List[RaidenAPI]) -> None:
    last_known_block = max((app.rpc_client.block_number() for app in raiden_network))
    for app in raiden_network:
        wait_for_block(app, last_known_block, 0.5)

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [3])
@pytest.mark.parametrize('channels_per_node', [0])
@pytest.mark.parametrize('routing_mode', [RoutingMode.PFS])
def test_pfs_send_capacity_updates_on_deposit_and_withdraw(raiden_network: List[RaidenAPI], token_addresses: List[TokenAddress], pfs_mock: object) -> None:
    # ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [3])
@pytest.mark.parametrize('channels_per_node', [CHAIN])
@pytest.mark.parametrize('routing_mode', [RoutingMode.PFS])
def test_pfs_send_capacity_updates_during_mediated_transfer(raiden_network: List[RaidenAPI], number_of_nodes: int, deposit: TokenAmount, token_addresses: List[TokenAddress], network_wait: int) -> None:
    # ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('channels_per_node', [CHAIN])
@pytest.mark.parametrize('routing_mode', [RoutingMode.PFS])
def test_pfs_send_unique_capacity_and_fee_updates_during_mediated_transfer(raiden_network: List[RaidenAPI]) -> None:
    # ...
