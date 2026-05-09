from typing import List, Any, Sequence, Optional, Union
from raiden.utils.typing import (
    Address, BlockExpiration, BlockNumber, BlockTimeout, FeeAmount, 
    PaymentAmount, PaymentID, ProportionalFeeAmount, Secret, 
    TargetAddress, TokenAddress, TokenAmount
)
from raiden.transfer.events import EventPaymentSentFailed, EventPaymentSentSuccess

def test_transfer_with_secret(
    raiden_network: Sequence[Any], 
    number_of_nodes: int, 
    deposit: int, 
    token_addresses: List[TokenAddress], 
    network_wait: int
) -> None: ...

def test_mediated_transfer(
    raiden_network: Sequence[Any], 
    number_of_nodes: int, 
    deposit: int, 
    token_addresses: List[TokenAddress], 
    network_wait: int
) -> None: ...

def test_locked_transfer_secret_registered_onchain(
    raiden_network: Sequence[Any], 
    token_addresses: List[TokenAddress], 
    secret_registry_address: Address, 
    retry_timeout: int
) -> None: ...

def test_mediated_transfer_with_entire_deposit(
    raiden_network: Sequence[Any], 
    number_of_nodes: int, 
    token_addresses: List[TokenAddress], 
    deposit: int, 
    network_wait: int
) -> None: ...

def test_mediated_transfer_messages_out_of_order(
    raiden_network: Sequence[Any], 
    deposit: int, 
    token_addresses: List[TokenAddress], 
    network_wait: int
) -> None: ...

def test_mediated_transfer_calls_pfs(
    raiden_chain: Sequence[Any], 
    token_addresses: List[TokenAddress]
) -> None: ...

def test_mediated_transfer_with_node_consuming_more_than_allocated_fee(
    decrypt_patch: Any, 
    raiden_network: Sequence[Any], 
    number_of_nodes: int, 
    deposit: int, 
    token_addresses: List[TokenAddress], 
    network_wait: int
) -> None: ...

def test_mediated_transfer_with_fees(
    raiden_network: Sequence[Any], 
    number_of_nodes: int, 
    deposit: int, 
    token_addresses: List[TokenAddress], 
    network_wait: int, 
    case_no: int
) -> None: ...

def test_max_locks_reached(
    raiden_network: Sequence[Any], 
    token_addresses: List[TokenAddress]
) -> None: ...