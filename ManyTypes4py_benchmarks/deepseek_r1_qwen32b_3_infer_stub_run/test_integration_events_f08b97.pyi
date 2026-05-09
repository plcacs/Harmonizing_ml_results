from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Generator, TypeVar, cast
from eth_utils import keccak
from gevent import Timeout
from raiden.settings import DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS, INTERNAL_ROUTING_DEFAULT_FEE_PERC
from raiden.tests.utils.protocol import HoldRaidenEventHandler
from raiden.tests.utils.transfer import assert_synced_channel_state, block_offset_timeout, create_route_state_for_route, get_channelstate, watch_for_unlock_failures
from raiden.utils.typing import Address, Balance, BlockNumber, ChannelID, FeeAmount, PaymentAmount, PaymentID, Secret, TargetAddress, TokenNetworkAddress
from raiden_contracts.contract_manager import ContractManager
from raiden.raiden_service import RaidenService
from raiden.transfer.events import ContractSendChannelClose
from raiden.transfer.mediated_transfer.events import SendLockedTransfer
from raiden.transfer.mediated_transfer.state_change import ReceiveSecretReveal
from raiden.transfer.state import BalanceProofSignedState
from raiden.transfer.state_change import ContractReceiveChannelBatchUnlock

def get_netting_channel_closed_events(proxy_manager: Any, token_network_address: Address, netting_channel_identifier: ChannelID, contract_manager: ContractManager, from_block: BlockNumber = GENESIS_BLOCK_NUMBER, to_block: BlockIdentifier = BLOCK_ID_LATEST) -> List[Dict[str, Any]]:
    ...

def get_netting_channel_deposit_events(proxy_manager: Any, token_network_address: Address, netting_channel_identifier: ChannelID, contract_manager: ContractManager, from_block: BlockNumber = GENESIS_BLOCK_NUMBER, to_block: BlockIdentifier = BLOCK_ID_LATEST) -> List[Dict[str, Any]]:
    ...

def get_netting_channel_settled_events(proxy_manager: Any, token_network_address: Address, netting_channel_identifier: ChannelID, contract_manager: ContractManager, from_block: BlockNumber = GENESIS_BLOCK_NUMBER, to_block: BlockIdentifier = BLOCK_ID_LATEST) -> List[Dict[str, Any]]:
    ...

def wait_both_channel_open(app0: RaidenService, app1: RaidenService, registry_address: Address, token_address: Address, retry_timeout: float) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('channels_per_node', [0])
def test_channel_new(raiden_chain: Tuple[RaidenService, ...], retry_timeout: float, token_addresses: List[Address]) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('privatekey_seed', ['event_new_channel:{}'])
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('channels_per_node', [0])
def test_channel_deposit(raiden_chain: Tuple[RaidenService, ...], deposit: Balance, retry_timeout: float, token_addresses: List[Address]) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('channels_per_node', [0])
def test_query_events(raiden_chain: Tuple[RaidenService, ...], token_addresses: List[Address], deposit: Balance, settle_timeout: int, retry_timeout: float, contract_manager: ContractManager, blockchain_type: str) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [3])
@pytest.mark.parametrize('channels_per_node', [CHAIN])
def test_secret_revealed_on_chain(raiden_chain: Tuple[RaidenService, ...], deposit: Balance, settle_timeout: int, token_addresses: List[Address], retry_interval_initial: float) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
def test_clear_closed_queue(raiden_network: Tuple[RaidenService, ...], token_addresses: List[Address], network_wait: float) -> None:
    ...