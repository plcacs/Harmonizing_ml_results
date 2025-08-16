from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
import gevent
import pytest
from eth_utils import encode_hex
from gevent import Timeout
from gevent.event import Event
from matrix_client.errors import MatrixRequestError
from matrix_client.user import User
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
    users: List[User] = []
    if signer is None:
        signer = make_signer()
    for i in range(number_of_users):
        user_id = f'@{signer.address_hex.lower()}:server{i}'
        signature_bytes = signer.sign(user_id.encode())
        signature_hex = encode_hex(signature_bytes)
        user = User(api=None, user_id=user_id, displayname=signature_hex)
        users.append(user)
    return users

def make_message_event(recipient: Address, address_metadata: Optional[AddressMetadata] = None, canonical_identifier: QueueIdentifier = CANONICAL_IDENTIFIER_UNORDERED_QUEUE) -> SendSecretRequest:
    return SendSecretRequest(recipient=recipient, recipient_metadata=address_metadata, canonical_identifier=canonical_identifier, message_identifier=make_message_identifier(), payment_identifier=PaymentID(1), amount=PaymentAmount(1), expiration=BlockExpiration(10), secrethash=factories.UNIT_SECRETHASH)

def make_message(sign: bool = True, message_event: Optional[SendSecretRequest] = None) -> SecretRequest:
    if message_event is None:
        message_event = make_message_event(Address(factories.HOP1))
    message = SecretRequest.from_event(message_event)
    if sign:
        message.sign(LocalSigner(factories.HOP1_KEY))
    return message

def make_message_text(sign: bool = True, overwrite_data: Optional[bytes] = None) -> Dict[str, Any]:
    if not overwrite_data:
        data = MessageSerializer.serialize(make_message(sign=sign))
    else:
        data = overwrite_data
    event = dict(type='m.room.message', sender=USERID1, content={'msgtype': 'm.text', 'body': data})
    return event
