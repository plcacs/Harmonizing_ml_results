from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    TYPE_CHECKING,
)
from unittest.mock import Mock, patch
import gevent
import pytest
from eth_utils.crypto import keccak
from raiden.constants import MAXIMUM_PENDING_TRANSFERS
from raiden.exceptions import InvalidSecret, RaidenUnrecoverableError
from raiden.message_handler import MessageHandler
from raiden.messages.transfers import LockedTransfer, RevealSecret, SecretRequest
from raiden.network.pathfinding import PFSConfig, PFSInfo, PFSProxy
from raiden.raiden_service import RaidenService
from raiden.settings import DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS
from raiden.storage.sqlite import RANGE_ALL_STATE_CHANGES
from raiden.tests.utils import factories
from raiden.tests.utils.detect_failure import raise_on_failure
from raiden.tests.utils.events import search_for_item
from raiden.tests.utils.factories import make_secret
from raiden.tests.utils.mediation_fees import get_amount_for_sending_before_and_after_fees
from raiden.tests.utils.network import CHAIN
from raiden.tests.utils.protocol import HoldRaidenEventHandler, WaitForMessage
from raiden.tests.utils.transfer import (
    TransferState,
    assert_succeeding_transfer_invariants,
    assert_synced_channel_state,
    block_timeout_for_transfer_by_secrethash,
    create_route_state_for_route,
    transfer,
    transfer_and_assert_path,
    wait_assert,
)
from raiden.transfer import views
from raiden.transfer.events import EventPaymentSentFailed, EventPaymentSentSuccess
from raiden.transfer.mediated_transfer.events import SendSecretRequest
from raiden.transfer.mediated_transfer.initiator import calculate_fee_margin
from raiden.transfer.mediated_transfer.mediation_fee import FeeScheduleState
from raiden.transfer.mediated_transfer.state_change import ActionInitMediator, ActionInitTarget
from raiden.transfer.mediated_transfer.tasks import InitiatorTask
from raiden.utils.secrethash import sha256_secrethash
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
from raiden.waiting import wait_for_block

if TYPE_CHECKING:
    from raiden.tests.utils.factories import (
        LockedTransferProperties,
        UNIT_TRANSFER_INITIATOR,
        HOP1,
        HOP1_KEY,
        make_canonical_identifier,
        make_address,
    )
    from raiden.tests.utils.mediation_fees import FeeCalculation
    from raiden.transfer.state import (
        ChainState,
        TokenNetworkAddress,
        TokenNetworkRegistryAddress,
        ChannelState,
    )
    from raiden.raiden_service import RaidenService
    from raiden.storage.wal import WriteAheadLog

@raise_on_failure
@pytest.mark.parametrize("channels_per_node", [CHAIN])
@pytest.mark.parametrize("number_of_nodes", [2])
def test_transfer_with_secret(
    raiden_network: Tuple["RaidenService", "RaidenService"],
    number_of_nodes: int,
    deposit: int,
    token_addresses: List["TokenAddress"],
    network_wait: int,
) -> None: ...

@raise_on_failure
@pytest.mark.parametrize("channels_per_node", [CHAIN])
@pytest.mark.parametrize("number_of_nodes", [3])
def test_mediated_transfer(
    raiden_network: Tuple["RaidenService", "RaidenService", "RaidenService"],
    number_of_nodes: int,
    deposit: int,
    token_addresses: List["TokenAddress"],
    network_wait: int,
) -> None: ...

@raise_on_failure
@pytest.mark.parametrize("channels_per_node", [CHAIN])
@pytest.mark.parametrize("number_of_nodes", [1])
def test_locked_transfer_secret_registered_onchain(
    raiden_network: Tuple["RaidenService"],
    token_addresses: List["TokenAddress"],
    secret_registry_address: "Address",
    retry_timeout: int,
) -> None: ...

@raise_on_failure
@pytest.mark.parametrize("channels_per_node", [CHAIN])
@pytest.mark.parametrize("number_of_nodes", [3])
def test_mediated_transfer_with_entire_deposit(
    raiden_network: Tuple["RaidenService", "RaidenService", "RaidenService"],
    number_of_nodes: int,
    token_addresses: List["TokenAddress"],
    deposit: int,
    network_wait: int,
) -> None: ...

@pytest.mark.skip(reason="flaky, see https://github.com/raiden-network/raiden/issues/4694")
@raise_on_failure
@pytest.mark.parametrize("channels_per_node", [CHAIN])
@pytest.mark.parametrize("number_of_nodes", [3])
def test_mediated_transfer_messages_out_of_order(
    raiden_network: Tuple["RaidenService", "RaidenService", "RaidenService"],
    deposit: int,
    token_addresses: List["TokenAddress"],
    network_wait: int,
) -> None: ...

@raise_on_failure
@pytest.mark.parametrize("number_of_nodes", (3,))
@pytest.mark.parametrize("channels_per_node", (CHAIN,))
def test_mediated_transfer_calls_pfs(
    raiden_chain: Tuple["RaidenService", "RaidenService", "RaidenService"],
    token_addresses: List["TokenAddress"],
) -> None: ...

@raise_on_failure
@pytest.mark.parametrize("channels_per_node", [CHAIN])
@pytest.mark.parametrize("number_of_nodes", [3])
@patch("raiden.message_handler.decrypt_secret", side_effect=InvalidSecret)
def test_mediated_transfer_with_node_consuming_more_than_allocated_fee(
    decrypt_patch: Mock,
    raiden_network: Tuple["RaidenService", "RaidenService", "RaidenService"],
    number_of_nodes: int,
    deposit: int,
    token_addresses: List["TokenAddress"],
    network_wait: int,
) -> None: ...

@raise_on_failure
@pytest.mark.parametrize("case_no", range(8))
@pytest.mark.parametrize("channels_per_node", [CHAIN])
@pytest.mark.parametrize("number_of_nodes", [4])
def test_mediated_transfer_with_fees(
    raiden_network: Tuple["RaidenService", "RaidenService", "RaidenService", "RaidenService"],
    number_of_nodes: int,
    deposit: int,
    token_addresses: List["TokenAddress"],
    network_wait: int,
    case_no: int,
) -> None: ...

@pytest.mark.flaky
@raise_on_failure
@pytest.mark.parametrize("channels_per_node", [CHAIN])
@pytest.mark.parametrize("number_of_nodes", [2])
@pytest.mark.parametrize("settle_timeout", [1000])
def test_max_locks_reached(
    raiden_network: Tuple["RaidenService", "RaidenService"],
    token_addresses: List["TokenAddress"],
) -> None: ...