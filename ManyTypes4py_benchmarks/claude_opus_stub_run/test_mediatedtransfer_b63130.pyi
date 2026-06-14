from typing import List
from unittest.mock import Mock

import pytest

from raiden.raiden_service import RaidenService
from raiden.utils.typing import (
    Address,
    FeeAmount,
    PaymentAmount,
    PaymentID,
    TokenAddress,
)

def test_transfer_with_secret(
    raiden_network: list[RaidenService],
    number_of_nodes: int,
    deposit: int,
    token_addresses: list[TokenAddress],
    network_wait: float,
) -> None: ...

def test_mediated_transfer(
    raiden_network: list[RaidenService],
    number_of_nodes: int,
    deposit: int,
    token_addresses: list[TokenAddress],
    network_wait: float,
) -> None: ...

def test_locked_transfer_secret_registered_onchain(
    raiden_network: list[RaidenService],
    token_addresses: list[TokenAddress],
    secret_registry_address: Address,
    retry_timeout: float,
) -> None: ...

def test_mediated_transfer_with_entire_deposit(
    raiden_network: list[RaidenService],
    number_of_nodes: int,
    token_addresses: list[TokenAddress],
    deposit: int,
    network_wait: float,
) -> None: ...

def test_mediated_transfer_messages_out_of_order(
    raiden_network: list[RaidenService],
    deposit: int,
    token_addresses: list[TokenAddress],
    network_wait: float,
) -> None: ...

def test_mediated_transfer_calls_pfs(
    raiden_chain: list[RaidenService],
    token_addresses: list[TokenAddress],
) -> None: ...

def test_mediated_transfer_with_node_consuming_more_than_allocated_fee(
    decrypt_patch: Mock,
    raiden_network: list[RaidenService],
    number_of_nodes: int,
    deposit: int,
    token_addresses: list[TokenAddress],
    network_wait: float,
) -> None: ...

def test_mediated_transfer_with_fees(
    raiden_network: list[RaidenService],
    number_of_nodes: int,
    deposit: int,
    token_addresses: list[TokenAddress],
    network_wait: float,
    case_no: int,
) -> None: ...

def test_max_locks_reached(
    raiden_network: list[RaidenService],
    token_addresses: list[TokenAddress],
) -> None: ...