from typing import Any, List, Optional
from raiden.raiden_service import RaidenService
from raiden.settings import BlockNumber
from raiden.tests.utils.transfer import TransferState
from raiden.tests.utils.protocol import WaitForMessage
from raiden.utils.typing import (
    Address,
    BlockExpiration,
    BlockTimeout,
    FeeAmount,
    PaymentAmount,
    PaymentID,
    ProportionalFeeAmount,
    Secret,
    TargetAddress,
    TokenAddress,
    TokenAmount,
)
from raiden.transfer.mediated_transfer.state_change import ActionInitMediator, ActionInitTarget
from raiden.transfer.mediated_transfer.tasks import InitiatorTask

def test_transfer_with_secret(raiden_network: List[RaidenService], number_of_nodes: int, deposit: TokenAmount, token_addresses: List[TokenAddress], network_wait: float) -> Optional[Any]:
    ...

def test_mediated_transfer(raiden_network: List[RaidenService], number_of_nodes: int, deposit: TokenAmount, token_addresses: List[TokenAddress], network_wait: float) -> Optional[Any]:
    ...

def test_locked_transfer_secret_registered_onchain(raiden_network: List[RaidenService], token_addresses: List[TokenAddress], secret_registry_address: Address, retry_timeout: float) -> Optional[Any]:
    ...

def test_mediated_transfer_with_entire_deposit(raiden_network: List[RaidenService], number_of_nodes: int, token_addresses: List[TokenAddress], deposit: TokenAmount, network_wait: float) -> Optional[Any]:
    ...

def test_mediated_transfer_messages_out_of_order(raiden_network: List[RaidenService], deposit: TokenAmount, token_addresses: List[TokenAddress], network_wait: float) -> Optional[Any]:
    ...

def test_mediated_transfer_calls_pfs(raiden_chain: List[RaidenService], token_addresses: List[TokenAddress]) -> Optional[Any]:
    ...

def test_mediated_transfer_with_node_consuming_more_than_allocated_fee(raiden_network: List[RaidenService], number_of_nodes: int, deposit: TokenAmount, token_addresses: List[TokenAddress], network_wait: float) -> Optional[Any]:
    ...

def test_mediated_transfer_with_fees(raiden_network: List[RaidenService], number_of_nodes: int, deposit: TokenAmount, token_addresses: List[TokenAddress], network_wait: float, case_no: int) -> Optional[Any]:
    ...

def test_max_locks_reached(raiden_network: List[RaidenService], token_addresses: List[TokenAddress], settle_timeout: int) -> Optional[Any]:
    ...