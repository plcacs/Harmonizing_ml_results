from typing import Any, Dict, List, Optional, Union
from unittest.mock import Mock, patch
import gevent
import pytest
from eth_utils import keccak
from web3._utils.events import construct_event_topic_set
from raiden import waiting
from raiden.api.python import RaidenAPI
from raiden.blockchain.events import get_all_netting_channel_events, get_contract_events
from raiden.constants import BLOCK_ID_LATEST, GENESIS_BLOCK_NUMBER
from raiden.network.proxies.proxy_manager import ProxyManager
from raiden.network.proxies.token_network import TokenNetwork
from raiden.raiden_service import RaidenService
from raiden.settings import DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS, INTERNAL_ROUTING_DEFAULT_FEE_PERC
from raiden.tests.utils import factories
from raiden.tests.utils.detect_failure import raise_on_failure
from raiden.tests.utils.events import must_have_event, search_for_item, wait_for_state_change
from raiden.tests.utils.network import CHAIN
from raiden.tests.utils.protocol import HoldRaidenEventHandler
from raiden.tests.utils.transfer import assert_synced_channel_state, block_offset_timeout, create_route_state_for_route, get_channelstate, watch_for_unlock_failures
from raiden.transfer import views
from raiden.transfer.events import ContractSendChannelClose
from raiden.transfer.mediated_transfer.events import SendLockedTransfer
from raiden.transfer.mediated_transfer.state_change import ReceiveSecretReveal
from raiden.transfer.state import BalanceProofSignedState
from raiden.transfer.state_change import ContractReceiveChannelBatchUnlock
from raiden.utils.formatting import to_checksum_address
from raiden.utils.secrethash import sha256_secrethash
from raiden.utils.typing import Address, Balance, BlockIdentifier, BlockNumber, ChannelID, Dict, FeeAmount, List, PaymentAmount, PaymentID, Secret, TargetAddress, TokenNetworkAddress
from raiden.waiting import wait_until
from raiden_contracts.constants import CONTRACT_TOKEN_NETWORK, CONTRACT_TOKEN_NETWORK_REGISTRY, EVENT_TOKEN_NETWORK_CREATED, ChannelEvent
from raiden_contracts.contract_manager import ContractManager

def get_netting_channel_closed_events(
    proxy_manager: ProxyManager,
    token_network_address: TokenNetworkAddress,
    netting_channel_identifier: ChannelID,
    contract_manager: ContractManager,
    from_block: BlockIdentifier = ...,
    to_block: BlockIdentifier = ...
) -> List[Dict[str, Any]]: ...

def get_netting_channel_deposit_events(
    proxy_manager: ProxyManager,
    token_network_address: TokenNetworkAddress,
    netting_channel_identifier: ChannelID,
    contract_manager: ContractManager,
    from_block: BlockIdentifier = ...,
    to_block: BlockIdentifier = ...
) -> List[Dict[str, Any]]: ...

def get_netting_channel_settled_events(
    proxy_manager: ProxyManager,
    token_network_address: TokenNetworkAddress,
    netting_channel_identifier: ChannelID,
    contract_manager: ContractManager,
    from_block: BlockIdentifier = ...,
    to_block: BlockIdentifier = ...
) -> List[Dict[str, Any]]: ...

def wait_both_channel_open(
    app0: RaidenService,
    app1: RaidenService,
    registry_address: Address,
    token_address: Address,
    retry_timeout: float
) -> None: ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('channels_per_node', [0])
def test_channel_new(
    raiden_chain: List[RaidenService],
    retry_timeout: float,
    token_addresses: List[Address]
) -> None: ...

@raise_on_failure
@pytest.mark.parametrize('privatekey_seed', ['event_new_channel:{}'])
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('channels_per_node', [0])
def test_channel_deposit(
    raiden_chain: List[RaidenService],
    deposit: Balance,
    retry_timeout: float,
    token_addresses: List[Address]
) -> None: ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('channels_per_node', [0])
def test_query_events(
    raiden_chain: List[RaidenService],
    token_addresses: List[Address],
    deposit: Balance,
    settle_timeout: int,
    retry_timeout: float,
    contract_manager: ContractManager,
    blockchain_type: str
) -> None: ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [3])
@pytest.mark.parametrize('channels_per_node', [CHAIN])
def test_secret_revealed_on_chain(
    raiden_chain: List[RaidenService],
    deposit: Balance,
    settle_timeout: int,
    token_addresses: List[Address],
    retry_interval_initial: float
) -> None: ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
def test_clear_closed_queue(
    raiden_network: List[RaidenService],
    token_addresses: List[Address],
    network_wait: float
) -> None: ...