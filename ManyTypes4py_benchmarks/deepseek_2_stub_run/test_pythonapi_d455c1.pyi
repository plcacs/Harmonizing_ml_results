```python
from typing import Any, List, Set
from unittest.mock import Mock
import gevent
import pytest
from eth_utils import Address as EthAddress
from raiden import waiting
from raiden.api.python import RaidenAPI
from raiden.constants import Environment
from raiden.exceptions import (
    AlreadyRegisteredTokenAddress,
    DepositMismatch,
    DepositOverLimit,
    InsufficientEth,
    InsufficientGasReserve,
    InvalidBinaryAddress,
    InvalidSecret,
    InvalidSettleTimeout,
    RaidenRecoverableError,
    SamePeerAddress,
    TokenNotRegistered,
    UnknownTokenAddress,
)
from raiden.raiden_service import RaidenService
from raiden.tests.utils.client import RPCClient
from raiden.tests.utils.detect_failure import raise_on_failure
from raiden.tests.utils.events import wait_for_state_change
from raiden.tests.utils.factories import make_address
from raiden.tests.utils.protocol import HoldRaidenEventHandler
from raiden.tests.utils.transfer import get_channelstate
from raiden.transfer import channel, views
from raiden.transfer.events import EventPaymentSentFailed
from raiden.transfer.mediated_transfer.events import SendSecretRequest
from raiden.transfer.state import ChannelState, NetworkState
from raiden.transfer.state_change import (
    ContractReceiveChannelSettled,
    ContractReceiveNewTokenNetwork,
)
from raiden.utils.gas_reserve import get_required_gas_estimate
from raiden.utils.typing import (
    Address,
    BlockNumber,
    BlockTimeout,
    PaymentAmount,
    TargetAddress,
    TokenAddress,
    TokenAmount,
)
from raiden_contracts.contract_manager import ContractManager

def test_register_token(
    raiden_network: List[RaidenService],
    retry_timeout: float,
    unregistered_token: TokenAddress,
) -> None: ...

def test_register_token_insufficient_eth(
    raiden_network: List[RaidenService],
    retry_timeout: float,
    unregistered_token: TokenAddress,
) -> None: ...

def test_token_registered_race(
    raiden_chain: List[RaidenService],
    retry_timeout: float,
    unregistered_token: TokenAddress,
) -> None: ...

def test_deposit_updates_balance_immediately(
    raiden_chain: List[RaidenService],
    token_addresses: List[TokenAddress],
) -> None: ...

def test_transfer_with_invalid_address_type(
    raiden_network: List[RaidenService],
    token_addresses: List[TokenAddress],
) -> None: ...

def test_insufficient_funds(
    raiden_network: List[RaidenService],
    token_addresses: List[TokenAddress],
    deposit: TokenAmount,
) -> None: ...

def test_funds_check_for_openchannel(
    raiden_network: List[RaidenService],
    token_addresses: List[TokenAddress],
) -> None: ...

def test_payment_timing_out_if_partner_does_not_respond(
    raiden_network: List[RaidenService],
    token_addresses: List[TokenAddress],
    reveal_timeout: int,
    retry_timeout: float,
) -> None: ...

def test_participant_deposit_amount_must_be_smaller_than_the_limit(
    raiden_network: List[RaidenService],
    contract_manager: ContractManager,
    retry_timeout: float,
) -> None: ...

def test_deposit_amount_must_be_smaller_than_the_token_network_limit(
    raiden_network: List[RaidenService],
    contract_manager: ContractManager,
    retry_timeout: float,
) -> None: ...

def test_token_addresses(
    raiden_network: List[RaidenService],
    token_addresses: List[TokenAddress],
) -> None: ...

def run_test_token_addresses(
    raiden_network: List[RaidenService],
    token_addresses: List[TokenAddress],
) -> None: ...

def test_raidenapi_channel_lifecycle(
    raiden_network: List[RaidenService],
    token_addresses: List[TokenAddress],
    deposit: TokenAmount,
    retry_timeout: float,
    settle_timeout_max: int,
) -> None: ...

def test_same_addresses_for_payment(
    raiden_network: List[RaidenService],
    token_addresses: List[TokenAddress],
) -> None: ...
```