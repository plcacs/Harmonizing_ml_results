from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
from gevent.event import Event
from raiden.constants import Environment, MatrixMessageType
from raiden.messages.transfers import SecretRequest
from raiden.network.transport import MatrixTransport
from raiden.network.transport.matrix.client import GMatrixHttpApi
from raiden.network.transport.matrix.rtc.web_rtc import WebRTCManager
from raiden.network.transport.matrix.transport import RETRY_QUEUE_IDLE_AFTER, _RetryQueue
from raiden.settings import MatrixTransportConfig
from raiden.storage.serialization.serializer import MessageSerializer
from raiden.tests.utils import factories
from raiden.tests.utils.factories import make_message_identifier, make_signer
from raiden.tests.utils.mocks import MockRaidenService
from raiden.transfer.identifiers import CANONICAL_IDENTIFIER_UNORDERED_QUEUE, QueueIdentifier
from raiden.transfer.mediated_transfer.events import SendSecretRequest
from raiden.utils.formatting import to_hex_address
from raiden.utils.signer import LocalSigner
from raiden.utils.typing import Address, AddressMetadata, PaymentAmount, PaymentID, UserID

USERID0: UserID = UserID('@0x1234567890123456789012345678901234567890:RestaurantAtTheEndOfTheUniverse')
USERID1: UserID = UserID(f'@{to_hex_address(factories.HOP1)}:Wonderland')

def create_new_users_for_address(signer: Optional[LocalSigner] = None, number_of_users: int = 1) -> List[User]:
    ...

def make_message_event(recipient: Address, address_metadata: Optional[AddressMetadata] = None, canonical_identifier: QueueIdentifier = CANONICAL_IDENTIFIER_UNORDERED_QUEUE) -> SendSecretRequest:
    ...

def make_message(sign: bool = True, message_event: Optional[SendSecretRequest] = None) -> SecretRequest:
    ...

def make_message_text(sign: bool = True, overwrite_data: Optional[bytes] = None) -> Dict[str, Any]:
    ...

def test_normal_processing_json(mock_matrix: MatrixTransport, skip_userid_validation: None) -> None:
    ...

def test_processing_invalid_json(mock_matrix: MatrixTransport, skip_userid_validation: None) -> None:
    ...

def test_non_signed_message_is_rejected(mock_matrix: MatrixTransport, skip_userid_validation: None) -> None:
    ...

def test_sending_nonstring_body(mock_matrix: MatrixTransport, skip_userid_validation: None) -> None:
    ...

def test_processing_invalid_message_json(mock_matrix: MatrixTransport, skip_userid_validation: None, message_input: str) -> None:
    ...

def test_processing_invalid_message_type_json(mock_matrix: MatrixTransport, skip_userid_validation: None) -> None:
    ...

def test_retry_queue_batch_by_user_id(mock_matrix: MatrixTransport) -> None:
    ...

def test_retry_queue_does_not_resend_removed_messages(mock_matrix: MatrixTransport, retry_interval_initial: float) -> None:
    ...

def test_retryqueue_idle_terminate(mock_matrix: MatrixTransport, retry_interval_initial: float) -> None:
    ...

def test_retryqueue_not_idle_with_messages(mock_matrix: MatrixTransport, retry_interval_initial: float) -> None:
    ...

def test_retryqueue_enqueue_not_blocking(mock_matrix: MatrixTransport, monkeypatch: Any) -> None:
    ...
