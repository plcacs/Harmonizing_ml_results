from typing import List, Address, BlockNumber, BlockTimeout, PaymentAmount, TokenAmount, TokenAddress
import gevent
import pytest
from eth_utils import to_canonical_address
from raiden import waiting
from raiden.api.python import RaidenAPI
from raiden.constants import UINT256_MAX, Environment
from raiden.exceptions import AlreadyRegisteredTokenAddress, DepositMismatch, DepositOverLimit, InsufficientEth, InsufficientGasReserve, InvalidBinaryAddress, InvalidSecret, InvalidSettleTimeout, RaidenRecoverableError, SamePeerAddress, TokenNotRegistered, UnknownTokenAddress
from raiden.raiden_service import RaidenService
from raiden.settings import DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS
from raiden.tests.utils.client import burn_eth
from raiden.tests.utils.detect_failure import raise_on_failure
from raiden.tests.utils.events import wait_for_state_change
from raiden.tests.utils.factories import make_address
from raiden.tests.utils.protocol import HoldRaidenEventHandler
from raiden.tests.utils.transfer import get_channelstate
from raiden.transfer import channel, views
from raiden.transfer.events import EventPaymentSentFailed
from raiden.transfer.mediated_transfer.events import SendSecretRequest
from raiden.transfer.state import ChannelState, NetworkState
from raiden.transfer.state_change import ContractReceiveChannelSettled, ContractReceiveNewTokenNetwork
from raiden.utils.gas_reserve import GAS_RESERVE_ESTIMATE_SECURITY_FACTOR, get_required_gas_estimate
from raiden_contracts.contract_manager import ContractManager

@raise_on_failure
@pytest.mark.parametrize('privatekey_seed', ['test_token_registration:{}'])
@pytest.mark.parametrize('number_of_nodes', [1])
@pytest.mark.parametrize('channels_per_node', [0])
@pytest.mark.parametrize('number_of_tokens', [1])
@pytest.mark.parametrize('environment_type', [Environment.DEVELOPMENT])
def test_register_token(raiden_network: List[RaidenService], retry_timeout: int, unregistered_token: TokenAddress) -> None:
    # ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('channels_per_node', [0])
def test_funds_check_for_openchannel(raiden_network: List[RaidenService], token_addresses: List[TokenAddress]) -> None:
    # ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('number_of_tokens', [1])
def test_token_registered_race(raiden_chain: List[RaidenService], retry_timeout: int, unregistered_token: TokenAddress) -> None:
    # ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('channels_per_node', [1])
def test_transfer_with_invalid_address_type(raiden_network: List[RaidenService], token_addresses: List[TokenAddress]) -> None:
    # ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('channels_per_node', [1])
def test_insufficient_funds(raiden_network: List[RaidenService], token_addresses: List[TokenAddress], deposit: TokenAmount) -> None:
    # ...

@raise_on_failure
@pytest.mark.parametrize('privatekey_seed', ['test_set_deposit_limit_crash:{}'])
@pytest.mark.parametrize('number_of_nodes', [1])
@pytest.mark.parametrize('channels_per_node', [0])
@pytest.mark.parametrize('number_of_tokens', [0])
@pytest.mark.parametrize('environment_type', [Environment.DEVELOPMENT])
def test_participant_deposit_amount_must_be_smaller_than_the_limit(raiden_network: List[RaidenService], contract_manager: ContractManager, retry_timeout: int) -> None:
    # ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('number_of_tokens', [1])
def test_deposit_amount_must_be_smaller_than_the_token_network_limit(raiden_network: List[RaidenService], token_addresses: List[TokenAddress]) -> None:
    # ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('channels_per_node', [0])
def test_raidenapi_channel_lifecycle(raiden_network: List[RaidenService], token_addresses: List[TokenAddress], deposit: TokenAmount, retry_timeout: int, settle_timeout_max: BlockTimeout) -> None:
    # ...
