from raiden.api.python import RaidenAPI
from raiden.constants import DeviceIDs, RoutingMode
from raiden.messages.path_finding_service import PFSCapacityUpdate, PFSFeeUpdate
from raiden.tests.utils.transport import TestMatrixTransport
from raiden.transfer.state import TransactionChannelDeposit
from raiden.transfer.state_change import ContractReceiveChannelDeposit, ReceiveUnlock
from raiden.utils.typing import List, LockedAmount, Locksroot, MessageID, Nonce, PaymentAmount, PaymentID, Secret, Signature, TokenAddress, TokenAmount, WithdrawAmount
from raiden.waiting import wait_for_block

def get_messages(app: RaidenService) -> List[Message]:
    assert isinstance(app.transport, TestMatrixTransport), 'Transport is not a `TestMatrixTransport`'
    return app.transport.broadcast_messages[DeviceIDs.PFS.value]

def reset_messages(app: RaidenService) -> None:
    assert isinstance(app.transport, TestMatrixTransport), 'Transport is not a `TestMatrixTransport`'
    app.transport.broadcast_messages[DeviceIDs.PFS.value] = []

def wait_all_apps(raiden_network: List[RaidenService]) -> None:
    last_known_block = max((app.rpc_client.block_number() for app in raiden_network))
    for app in raiden_network:
        wait_for_block(app, last_known_block, 0.5)

def test_pfs_send_capacity_updates_on_deposit_and_withdraw(raiden_network: List[RaidenService], token_addresses: List[TokenAddress], pfs_mock: Any) -> None:
    ...

def test_pfs_send_capacity_updates_during_mediated_transfer(raiden_network: List[RaidenService], number_of_nodes: int, deposit: TokenAmount, token_addresses: List[TokenAddress], network_wait: float) -> None:
    ...

def test_pfs_send_unique_capacity_and_fee_updates_during_mediated_transfer(raiden_network: List[RaidenService]) -> None:
    ...
