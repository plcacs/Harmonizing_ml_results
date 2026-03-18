```python
from typing import Any
from typing import List
from typing import Optional
from typing import Union
from unittest.mock import Mock
import gevent
import pytest
from raiden.constants import MAXIMUM_PENDING_TRANSFERS
from raiden.exceptions import InvalidSecret
from raiden.exceptions import RaidenUnrecoverableError
from raiden.message_handler import MessageHandler
from raiden.messages.transfers import LockedTransfer
from raiden.messages.transfers import RevealSecret
from raiden.messages.transfers import SecretRequest
from raiden.network.pathfinding import PFSConfig
from raiden.network.pathfinding import PFSInfo
from raiden.network.pathfinding import PFSProxy
from raiden.raiden_service import RaidenService
from raiden.settings import DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS
from raiden.storage.sqlite import RANGE_ALL_STATE_CHANGES
from raiden.tests.utils import factories
from raiden.tests.utils.detect_failure import raise_on_failure
from raiden.tests.utils.events import search_for_item
from raiden.tests.utils.factories import make_secret
from raiden.tests.utils.mediation_fees import get_amount_for_sending_before_and_after_fees
from raiden.tests.utils.network import CHAIN
from raiden.tests.utils.protocol import HoldRaidenEventHandler
from raiden.tests.utils.protocol import WaitForMessage
from raiden.tests.utils.transfer import TransferState
from raiden.tests.utils.transfer import assert_succeeding_transfer_invariants
from raiden.tests.utils.transfer import assert_synced_channel_state
from raiden.tests.utils.transfer import block_timeout_for_transfer_by_secrethash
from raiden.tests.utils.transfer import create_route_state_for_route
from raiden.tests.utils.transfer import transfer
from raiden.tests.utils.transfer import transfer_and_assert_path
from raiden.tests.utils.transfer import wait_assert
from raiden.transfer import views
from raiden.transfer.events import EventPaymentSentFailed
from raiden.transfer.events import EventPaymentSentSuccess
from raiden.transfer.mediated_transfer.events import SendSecretRequest
from raiden.transfer.mediated_transfer.initiator import calculate_fee_margin
from raiden.transfer.mediated_transfer.mediation_fee import FeeScheduleState
from raiden.transfer.mediated_transfer.state_change import ActionInitMediator
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget
from raiden.transfer.mediated_transfer.tasks import InitiatorTask
from raiden.utils.secrethash import sha256_secrethash
from raiden.utils.typing import Address
from raiden.utils.typing import BlockExpiration
from raiden.utils.typing import BlockNumber
from raiden.utils.typing import BlockTimeout
from raiden.utils.typing import FeeAmount
from raiden.utils.typing import PaymentAmount
from raiden.utils.typing import PaymentID
from raiden.utils.typing import ProportionalFeeAmount
from raiden.utils.typing import Secret
from raiden.utils.typing import TargetAddress
from raiden.utils.typing import TokenAddress
from raiden.utils.typing import TokenAmount
from raiden.waiting import wait_for_block

def test_transfer_with_secret(
    raiden_network: Any,
    number_of_nodes: int,
    deposit: Any,
    token_addresses: Any,
    network_wait: Any
) -> None: ...

def test_mediated_transfer(
    raiden_network: Any,
    number_of_nodes: int,
    deposit: Any,
    token_addresses: Any,
    network_wait: Any
) -> None: ...

def test_locked_transfer_secret_registered_onchain(
    raiden_network: Any,
    token_addresses: Any,
    secret_registry_address: Any,
    retry_timeout: Any
) -> None: ...

def test_mediated_transfer_with_entire_deposit(
    raiden_network: Any,
    number_of_nodes: int,
    token_addresses: Any,
    deposit: Any,
    network_wait: Any
) -> None: ...

def test_mediated_transfer_messages_out_of_order(
    raiden_network: Any,
    deposit: Any,
    token_addresses: Any,
    network_wait: Any
) -> None: ...

def test_mediated_transfer_calls_pfs(
    raiden_chain: Any,
    token_addresses: Any
) -> None: ...

def test_mediated_transfer_with_node_consuming_more_than_allocated_fee(
    decrypt_patch: Any,
    raiden_network: Any,
    number_of_nodes: int,
    deposit: Any,
    token_addresses: Any,
    network_wait: Any
) -> None: ...

def test_mediated_transfer_with_fees(
    raiden_network: Any,
    number_of_nodes: int,
    deposit: Any,
    token_addresses: Any,
    network_wait: Any,
    case_no: int
) -> None: ...

def test_max_locks_reached(
    raiden_network: Any,
    token_addresses: Any
) -> None: ...
```