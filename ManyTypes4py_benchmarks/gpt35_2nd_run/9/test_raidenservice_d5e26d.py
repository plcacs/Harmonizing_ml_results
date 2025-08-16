from raiden.api.python import RaidenAPI
from raiden.constants import BLOCK_ID_LATEST, DeviceIDs, RoutingMode
from raiden.exceptions import RaidenUnrecoverableError
from raiden.message_handler import MessageHandler
from raiden.messages.monitoring_service import RequestMonitoring
from raiden.messages.path_finding_service import PFSCapacityUpdate, PFSFeeUpdate
from raiden.network.transport import MatrixTransport
from raiden.raiden_event_handler import RaidenEventHandler
from raiden.raiden_service import RaidenService
from raiden.settings import DEFAULT_MEDIATION_FLAT_FEE, DEFAULT_MEDIATION_PROPORTIONAL_FEE, DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS
from raiden.storage.sqlite import RANGE_ALL_STATE_CHANGES
from raiden.tests.integration.fixtures.raiden_network import RestartNode
from raiden.tests.integration.test_integration_pfs import wait_all_apps
from raiden.tests.utils.detect_failure import expect_failure, raise_on_failure
from raiden.tests.utils.events import search_for_item
from raiden.tests.utils.network import CHAIN
from raiden.tests.utils.transfer import transfer
from raiden.transfer import views
from raiden.transfer.state import NettingChannelState
from raiden.transfer.state_change import Block, ContractReceiveChannelClosed, ContractReceiveChannelNew
from raiden.ui.startup import RaidenBundle
from raiden.utils.copy import deepcopy
from raiden.utils.typing import BlockNumber, FeeAmount, List, PaymentAmount, PaymentID, ProportionalFeeAmount, TokenAddress, Type

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes: int', [1])
@pytest.mark.parametrize('channels_per_node: int', [0])
@pytest.mark.parametrize('number_of_tokens: int', [1])
def test_regression_filters_must_be_installed_from_confirmed_block(raiden_network: List[RaidenService]) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes: int', [2])
@pytest.mark.parametrize('channels_per_node: int', [CHAIN])
def test_broadcast_messages_must_be_sent_before_protocol_messages_on_restarts(raiden_network: List[RaidenService], restart_node: RestartNode, number_of_nodes: int, token_addresses: List[TokenAddress], network_wait: int) -> None:
    ...

@expect_failure
@pytest.mark.parametrize('start_raiden_apps: bool', [False])
@pytest.mark.parametrize('deposit: int', [0])
@pytest.mark.parametrize('channels_per_node: int', [CHAIN])
@pytest.mark.parametrize('number_of_nodes: int', [2])
def test_alarm_task_first_run_syncs_blockchain_events(raiden_network: List[RaidenService], blockchain_services: Type) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('deposit: int', [0])
@pytest.mark.parametrize('channels_per_node: int', [CHAIN])
@pytest.mark.parametrize('number_of_nodes: int', [2])
def test_initialize_wal_throws_when_lock_is_taken(raiden_network: List[RaidenService]) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes: int', [2])
def test_fees_are_updated_during_startup(raiden_network: List[RaidenService], restart_node: RestartNode, token_addresses: List[TokenAddress], deposit: int, retry_timeout: int) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes: int', [2])
@pytest.mark.parametrize('channels_per_node: int', [0])
@pytest.mark.parametrize('number_of_tokens: int', [1])
def test_blockchain_event_processed_interleaved(raiden_network: List[RaidenService], token_addresses: List[TokenAddress], restart_node: RestartNode) -> None:
    ...
