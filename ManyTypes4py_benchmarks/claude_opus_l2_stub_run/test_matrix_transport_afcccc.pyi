from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import pytest
from matrix_client.user import User

from raiden.network.transport import MatrixTransport
from raiden.network.transport.matrix.transport import _RetryQueue
from raiden.transfer.identifiers import CanonicalIdentifier, QueueIdentifier
from raiden.transfer.mediated_transfer.events import SendSecretRequest
from raiden.messages.transfers import SecretRequest
from raiden.utils.typing import Address, AddressMetadata, UserID

USERID0: UserID
USERID1: UserID

@pytest.fixture()
def skip_userid_validation(monkeypatch: pytest.MonkeyPatch) -> None: ...

@pytest.fixture()
def mock_raiden_service() -> Any: ...

@pytest.fixture()
def mock_matrix(
    monkeypatch: pytest.MonkeyPatch,
    mock_raiden_service: Any,
    retry_interval_initial: float,
    retry_interval_max: float,
    retries_before_backoff: int,
) -> MatrixTransport: ...

def create_new_users_for_address(
    signer: Optional[Any] = ..., number_of_users: int = ...
) -> List[User]: ...

@pytest.fixture(scope="session")
def sync_filter_dict() -> Dict[int, Any]: ...

@pytest.fixture
def create_sync_filter_patch(
    monkeypatch: pytest.MonkeyPatch, sync_filter_dict: Dict[int, Any]
) -> None: ...

@pytest.fixture
def record_sent_messages(mock_matrix: MatrixTransport) -> Any: ...

def make_message_event(
    recipient: Address,
    address_metadata: Optional[AddressMetadata] = ...,
    canonical_identifier: CanonicalIdentifier = ...,
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
    mock_matrix: MatrixTransport, monkeypatch: pytest.MonkeyPatch
) -> None: ...