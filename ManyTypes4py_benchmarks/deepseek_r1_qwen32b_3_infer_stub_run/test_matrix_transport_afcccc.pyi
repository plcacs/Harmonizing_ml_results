from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import UUID

import gevent
import pytest
from eth_utils import encode_hex
from gevent.event import Event
from matrix_client.user import User
from raiden.constants import Environment, MatrixMessageType
from raiden.messages.transfers import SecretRequest
from raiden.network.transport import MatrixTransport
from raiden.settings import MatrixTransportConfig
from raiden.tests.utils.factories import (
    make_message_identifier,
    make_signer,
)
from raiden.tests.utils.mocks import MockRaidenService
from raiden.transfer.identifiers import CANONICAL_IDENTIFIER_UNORDERED_QUEUE, QueueIdentifier
from raiden.utils.typing import (
    Address,
    AddressMetadata,
    BlockExpiration,
    PaymentAmount,
    PaymentID,
    UserID,
)

USERID0: UserID = ...
USERID1: UserID = ...

@pytest.fixture
def skip_userid_validation() -> pytest.FixtureFunction[None]:
    ...

@pytest.fixture
def mock_raiden_service() -> pytest.FixtureFunction[MockRaidenService]:
    ...

@pytest.fixture
def mock_matrix(
    monkeypatch: Any,
    mock_raiden_service: MockRaidenService,
    retry_interval_initial: float,
    retry_interval_max: float,
    retries_before_backoff: int,
) -> pytest.FixtureFunction[MatrixTransport]:
    ...

def create_new_users_for_address(
    signer: Optional[LocalSigner] = ...,
    number_of_users: int = ...,
) -> List[User]:
    ...

@pytest.fixture(scope='session')
def sync_filter_dict() -> pytest.FixtureFunction[Dict[int, Any]]:
    ...

@pytest.fixture
def create_sync_filter_patch(
    monkeypatch: Any,
    sync_filter_dict: Dict[int, Any],
) -> pytest.FixtureFunction[None]:
    ...

@pytest.fixture
def record_sent_messages(mock_matrix: MatrixTransport) -> pytest.FixtureFunction[None]:
    ...

def make_message_event(
    recipient: Address,
    address_metadata: Optional[AddressMetadata] = ...,
    canonical_identifier: CANONICAL_IDENTIFIER_UNORDERED_QUEUE = ...,
) -> SendSecretRequest:
    ...

def make_message(
    sign: bool = ...,
    message_event: Optional[SendSecretRequest] = ...,
) -> SecretRequest:
    ...

def make_message_text(
    sign: bool = ...,
    overwrite_data: Optional[Union[str, bytes]] = ...,
) -> Dict[str, Any]:
    ...

def test_normal_processing_json(
    mock_matrix: MatrixTransport,
    skip_userid_validation: None,
) -> None:
    ...

def test_processing_invalid_json(
    mock_matrix: MatrixTransport,
    skip_userid_validation: None,
) -> None:
    ...

def test_non_signed_message_is_rejected(
    mock_matrix: MatrixTransport,
    skip_userid_validation: None,
) -> None:
    ...

def test_sending_nonstring_body(
    mock_matrix: MatrixTransport,
    skip_userid_validation: None,
) -> None:
    ...

@pytest.mark.parametrize('message_input', [pytest.param('{"this": 1, "message": 5, "is": 3, "not_valid": 5}', id='json-1'), pytest.param('[', id='json-2')])
def test_processing_invalid_message_json(
    mock_matrix: MatrixTransport,
    skip_userid_validation: None,
    message_input: str,
) -> None:
    ...

def test_processing_invalid_message_type_json(
    mock_matrix: MatrixTransport,
    skip_userid_validation: None,
) -> None:
    ...

def test_retry_queue_batch_by_user_id(
    mock_matrix: MatrixTransport,
) -> None:
    ...

@pytest.mark.parametrize('retry_interval_initial', [0.01])
@pytest.mark.usefixtures('record_sent_messages')
def test_retry_queue_does_not_resend_removed_messages(
    mock_matrix: MatrixTransport,
    retry_interval_initial: float,
) -> None:
    ...

@pytest.mark.parametrize('retry_interval_initial', [0.05])
def test_retryqueue_idle_terminate(
    mock_matrix: MatrixTransport,
    retry_interval_initial: float,
) -> None:
    ...

@pytest.mark.parametrize('retry_interval_initial', [0.05])
def test_retryqueue_not_idle_with_messages(
    mock_matrix: MatrixTransport,
    retry_interval_initial: float,
) -> None:
    ...

def test_retryqueue_enqueue_not_blocking(
    mock_matrix: MatrixTransport,
    monkeypatch: Any,
) -> None:
    ...