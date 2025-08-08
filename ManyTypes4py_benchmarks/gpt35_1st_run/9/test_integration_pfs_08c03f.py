from raiden.api.python import RaidenAPI
from raiden.constants import DeviceIDs, RoutingMode
from raiden.messages.path_finding_service import PFSCapacityUpdate, PFSFeeUpdate
from raiden.tests.utils.transport import TestMatrixTransport
from raiden.transfer.state import TransactionChannelDeposit
from raiden.transfer.state_change import ContractReceiveChannelDeposit, ReceiveUnlock
from raiden.utils.typing import PaymentAmount, PaymentID, TokenAmount, WithdrawAmount
from raiden.waiting import wait_for_block
from raiden.tests.utils.detect_failure import raise_on_failure
from raiden.tests.utils.factories import make_transaction_hash
from raiden.tests.utils.transfer import assert_succeeding_transfer_invariants, block_timeout_for_transfer_by_secrethash, transfer, wait_assert

def get_messages(app: RaidenAPI) -> List[Message]:
    ...

def reset_messages(app: RaidenAPI) -> None:
    ...

def wait_all_apps(raiden_network: List[RaidenAPI]) -> None:
    ...

def test_pfs_send_capacity_updates_on_deposit_and_withdraw(raiden_network: List[RaidenAPI], token_addresses: List[TokenAddress], pfs_mock: Any) -> None:
    ...

def test_pfs_send_capacity_updates_during_mediated_transfer(raiden_network: List[RaidenAPI], number_of_nodes: int, deposit: TokenAmount, token_addresses: List[TokenAddress], network_wait: float) -> None:
    ...

def test_pfs_send_unique_capacity_and_fee_updates_during_mediated_transfer(raiden_network: List[RaidenAPI]) -> None:
    ...
