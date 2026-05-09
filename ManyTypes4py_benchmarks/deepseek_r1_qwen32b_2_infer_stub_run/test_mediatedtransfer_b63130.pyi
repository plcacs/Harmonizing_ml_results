from typing import List, Optional, Tuple, Union
from unittest.mock import Mock, WaitForMessage
from eth_utils.crypto import keccak
from raiden.settings import DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS
from raiden.tests.utils.transfer import TransferState
from raiden.utils.typing import (
    Address,
    BlockExpiration,
    BlockNumber,
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
from raiden.raiden_service import RaidenService
from raiden.transfer.mediated_transfer.state_change import ActionInitMediator, ActionInitTarget
from raiden.exceptions import RaidenUnrecoverableError, InvalidSecret

@raise_on_failure
@pytest.mark.parametrize('channels_per_node', [CHAIN])
@pytest.mark.parametrize('number_of_nodes', [2])
def test_transfer_with_secret(raiden_network: List[RaidenService], number_of_nodes: int, deposit: int, token_addresses: List[TokenAddress], network_wait: int) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('channels_per_node', [CHAIN])
@pytest.mark.parametrize('number_of_nodes', [3])
def test_mediated_transfer(raiden_network: List[RaidenService], number_of_nodes: int, deposit: int, token_addresses: List[TokenAddress], network_wait: int) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('channels_per_node', [CHAIN])
@pytest.mark.parametrize('number_of_nodes', [1])
def test_locked_transfer_secret_registered_onchain(raiden_network: List[RaidenService], token_addresses: List[TokenAddress], secret_registry_address: Address, retry_timeout: int) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('channels_per_node', [CHAIN])
@pytest.mark.parametrize('number_of_nodes', [3])
def test_mediated_transfer_with_entire_deposit(raiden_network: List[RaidenService], number_of_nodes: int, token_addresses: List[TokenAddress], deposit: int, network_wait: int) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('channels_per_node', [CHAIN])
@pytest.mark.parametrize('number_of_nodes', [3])
def test_mediated_transfer_messages_out_of_order(raiden_network: List[RaidenService], deposit: int, token_addresses: List[TokenAddress], network_wait: int) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', (3,))
@pytest.mark.parametrize('channels_per_node', (CHAIN,))
def test_mediated_transfer_calls_pfs(raiden_chain: List[RaidenService], token_addresses: List[TokenAddress]) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('channels_per_node', [CHAIN])
@pytest.mark.parametrize('number_of_nodes', [3])
@patch('raiden.message_handler.decrypt_secret', side_effect=InvalidSecret)
def test_mediated_transfer_with_node_consuming_more_than_allocated_fee(decrypt_patch: Mock, raiden_network: List[RaidenService], number_of_nodes: int, deposit: int, token_addresses: List[TokenAddress], network_wait: int) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('case_no', range(8))
@pytest.mark.parametrize('channels_per_node', [CHAIN])
@pytest.mark.parametrize('number_of_nodes', [4])
def test_mediated_transfer_with_fees(raiden_network: List[RaidenService], number_of_nodes: int, deposit: int, token_addresses: List[TokenAddress], network_wait: int, case_no: int) -> None:
    ...