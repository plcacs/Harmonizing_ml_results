from collections import defaultdict
from typing import Any, Optional

import pytest
from _pytest.monkeypatch import MonkeyPatch
from gevent.event import Event
from matrix_client.user import User

from raiden.constants import MatrixMessageType
from raiden.messages.transfers import SecretRequest
from raiden.network.transport import MatrixTransport
from raiden.network.transport.matrix.transport import _RetryQueue
from raiden.settings import MatrixTransportConfig
from raiden.tests.utils.mocks import MockRaidenService
from raiden.transfer.identifiers import CANONICAL_IDENTIFIER_UNORDERED_QUEUE, QueueIdentifier
from raiden.transfer.mediated_transfer.events import SendSecretRequest
from raiden.utils.signer import LocalSigner
from raiden.utils.typing import Address, AddressMetadata, UserID

USERID0: UserID
USERID1: UserID

@pytest.fixture()
def skip_userid_validation(monkeypatch: MonkeyPatch) -> None: ...

@pytest.fixture()
def mock_raiden_service() -> MockRaidenService: ...

@pytest.fixture()
def mock_matrix(
    monkeypatch: MonkeyPatch,
    mock_raiden_service: MockRaidenService,
    retry_interval_initial: float,
    retry_interval_max: float,
    retries_before_backoff: int,
) -> MatrixTransport: ...

def create_new_users_for_address(
    signer: Optional[LocalSigner] = ..., number_of_users: int = ...
) -> list[User]: ...

@pytest.fixture(scope="session")
def sync_filter_dict() -> dict[Any, Any]: ...

@pytest.fixture
def create_sync_filter_patch(
    monkeypatch: MonkeyPatch, sync_filter_dict: dict[Any, Any]
) -> None: ...

@pytest.fixture
def record_sent_messages(mock_matrix: MatrixTransport) -> Any: ...

def make_message_event(
    recipient: Address,
    address_metadata: Optional[AddressMetadata] = ...,
    canonical_identifier: Any = ...,
) -> SendSecretRequest: ...

def make_message(
    sign: bool = ..., message_event: Optional[SendSecretRequest] = ...
) -> SecretRequest: ...

def make_message_text(
    sign: bool = ..., overwrite_data: Optional[Any] = ...
) -> dict[str, Any]: ...

def test_normal_processing_json(
    mock_matrix: MatrixTransport, skip_userid_validation: None
) -> None: ...

def test_processing_invalid_json(
    mock_matrix: MatrixTransport, skip_userid_validation: None
) -> None: ...

def test_non_signed_message_is_rejected(
    mock_matrix: MatrixTransport, skip_userid_validation: None
) -> None: ...

def test_sending_nonstring_body(
    mock_matrix: MatrixTransport, skip_userid_validation: None
) -> None: ...

def test_processing_invalid_message_json(
    mock_matrix: MatrixTransport, skip_userid_validation: None, message_input: str
) -> None: ...

def test_processing_invalid_message_type_json(
    mock_matrix: MatrixTransport, skip_userid_validation: None
) -> None: ...

def test_retry_queue_batch_by_user_id(mock_matrix: MatrixTransport) -> None: ...

def test_retry_queue_does_not_resend_removed_messages(
    mock_matrix: MatrixTransport, retry_interval_initial: float
) -> None: ...

def test_retryqueue_idle_terminate(
    mock_matrix: MatrixTransport, retry_interval_initial: float
) -> None: ...

def test_retryqueue_not_idle_with_messages(
    mock_matrix: MatrixTransport, retry_interval_initial: float
) -> None: ...

def test_retryqueue_enqueue_not_blocking(
    mock_matrix: MatrixTransport, monkeypatch: MonkeyPatch
) -> None: ...